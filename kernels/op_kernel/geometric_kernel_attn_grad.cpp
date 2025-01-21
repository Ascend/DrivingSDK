/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

class GeometricKernelAttnGrad {
public:
    __aicore__ inline GeometricKernelAttnGrad() = default;

    __aicore__ inline void Init(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm, GM_ADDR level_start_index_gm,
                                GM_ADDR sampling_locations_gm, GM_ADDR attn_weights_gm, GM_ADDR grad_output_gm,
                                GM_ADDR grad_value_gm, GM_ADDR grad_attn_weights_gm,
                                const GeometricKernelAttnGradTilingData *tiling_data, TPipe *tmpPipe)
    {
        pipe = tmpPipe;
        GetTilingData(tiling_data);
        InitProperties();
        
        SetGlobalBuffer(value_gm, spatial_shapes_gm, level_start_index_gm, sampling_locations_gm, attn_weights_gm,
                        grad_output_gm, grad_value_gm, grad_attn_weights_gm);
        InitBuffer();
        GetLocalTensor();
    }

    __aicore__ inline void Process();

private:
    __aicore__ inline void GetTilingData(const GeometricKernelAttnGradTilingData *tiling_data)
    {
        batchSize = tiling_data->batchSize;
        numHeads = tiling_data->numHeads;
        embedDims = tiling_data->embedDims;
        numKeys = tiling_data->numKeys;
        numLevels = tiling_data->numLevels;
        numQueries = tiling_data->numQueries;
        numPoints = tiling_data->numPoints;
        coreNum = tiling_data->coreNum;
        taskPerCore = tiling_data->taskPerCore;
        taskCoreTail = tiling_data->taskCoreTail;
        taskCompNum = tiling_data->taskCompNum;
    }

    __aicore__ inline void InitProperties()
    {
        blockBytes = 32;
        numItemsPerBlock = blockBytes / sizeof(DTYPE_VALUE);
        numLevelsAligned = AlignUp(numLevels, numItemsPerBlock);
        numPointsAligned = AlignUp(numPoints, numItemsPerBlock);

        coreId = GetBlockIdx();
        if (coreId < taskCoreTail) {
            taskPerCore += 1;
            taskStart = coreId * taskPerCore;
            taskEnd = taskStart + taskPerCore;
        } else {
            taskStart = coreId * taskPerCore + taskCoreTail;
            taskEnd = taskStart + taskPerCore;
        }
        taskCompTail = taskPerCore % taskCompNum;

        taskCompNum = min(taskCompNum, taskPerCore);

        dstShapeCGO[0] = numLevels * numPointsAligned;
        dstShapeCGO[1] = embedDims;
        srcShapeCGO[0] = 1;
        srcShapeCGO[1] = embedDims;
        dstShapeCAW[0] = numHeads * numLevels * numPointsAligned;
        dstShapeCAW[1] = embedDims;
        srcShapeCAW[0] = numHeads * numLevels * numPointsAligned;
        srcShapeCAW[1] = 1;

        levelTmpBroadSrcShape[0] = numLevels;
        levelTmpBroadSrcShape[1] = 1;
        levelTmpBroadDstShape[0] = numLevels;
        levelTmpBroadDstShape[1] = numPointsAligned;
        levelBroadSrcShape[0] = 1;
        levelBroadSrcShape[1] = numLevels * numPointsAligned;
        levelBroadDstShape[0] = taskCompNum * numHeads;
        levelBroadDstShape[1] = numLevels * numPointsAligned;

        spatailTmpBroadSrcShape[0] = 2 * numLevels;
        spatailTmpBroadSrcShape[1] = 1;
        spatailTmpBroadDstShape[0] = 2 * numLevels;
        spatailTmpBroadDstShape[1] = numPointsAligned;
        spatailBroadSrcShape[0] = 1;
        spatailBroadSrcShape[1] = numLevels * numPointsAligned;
        spatailBroadDstShape[0] = taskCompNum * numHeads;
        spatailBroadDstShape[1] = numLevels * numPointsAligned;

        headBroadSrcShape[0] = 1;
        headBroadSrcShape[1] = numHeads * numLevels * numPointsAligned;
        headBroadDstShape[0] = taskCompNum;
        headBroadDstShape[1] = numHeads * numLevels * numPointsAligned;

        copyParams.blockLen = numPoints * sizeof(DTYPE_VALUE);
        samplingCopyParams.blockLen = numPoints * 2 * sizeof(DTYPE_VALUE);

        samplingGatherParams.src0RepeatStride = (numPoints * 2 + numItemsPerBlock - 1) / numItemsPerBlock;
        samplingGatherParams.src0BlockStride = 1;

        samplingGMOffset = batchSize * numQueries * numHeads * numLevels * numPoints;
        samplingUBOffset = taskCompNum * numHeads * numLevels * numPointsAligned;
    }

    __aicore__ inline void SetGlobalBuffer(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm, GM_ADDR level_start_index_gm,
                                           GM_ADDR sampling_locations_gm, GM_ADDR attn_weights_gm, GM_ADDR grad_output_gm,
                                           GM_ADDR grad_value_gm, GM_ADDR grad_attn_weights_gm)
    {
        valueGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(value_gm), batchSize * numHeads * numKeys * embedDims);
        spatialShapesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_SPATIAL_SHAPES *>(spatial_shapes_gm), numLevels * 2);
        levelStartIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_SPATIAL_SHAPES *>(level_start_index_gm), numLevels);
        samplingLocationsGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_SAMPLING_LOCATIONS *>(sampling_locations_gm), batchSize * numQueries * numHeads * numLevels * numPoints * 2);
        attnWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(attn_weights_gm), batchSize * numQueries * numHeads * numLevels * numPoints);
        gradOutputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(grad_output_gm), batchSize * numQueries * numHeads * embedDims);

        gradValueGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(grad_value_gm), batchSize * numHeads * numKeys * embedDims);
        gradAttnWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(grad_attn_weights_gm), batchSize * numQueries * numHeads * numLevels * numPoints);
    }

    __aicore__ inline void InitBuffer()
    {
        uint32_t compLenTHLP = taskCompNum * numHeads * numLevels * numPointsAligned;
        uint32_t compLenHLPD = numHeads * numLevels * numPointsAligned * embedDims;

        pipe->InitBuffer(spatialShapesOriginUb, 2 * numLevelsAligned * sizeof(DTYPE_SPATIAL_SHAPES));
        pipe->InitBuffer(spatialShapesUb, 2 * numLevelsAligned * sizeof(DTYPE_SPATIAL_SHAPES));
        pipe->InitBuffer(levelStartIdxUb, numLevelsAligned * sizeof(DTYPE_SPATIAL_SHAPES));

        pipe->InitBuffer(spatialShapesBroadUb, compLenTHLP * sizeof(DTYPE_SPATIAL_SHAPES) * 2);
        pipe->InitBuffer(levelStartIdxBroadUb, compLenTHLP * sizeof(DTYPE_SPATIAL_SHAPES));
        pipe->InitBuffer(headStartIdxBroadUb, compLenTHLP * sizeof(DTYPE_SPATIAL_SHAPES));

        pipe->InitBuffer(samplingLocationUb, compLenTHLP * sizeof(DTYPE_SPATIAL_SHAPES) * 2);
        pipe->InitBuffer(keyIdxsUb, compLenTHLP * sizeof(DTYPE_SPATIAL_SHAPES) * 2);
        
        pipe->InitBuffer(gradOutputUb, taskCompNum * numHeads * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(attnWeightUb, compLenTHLP * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(valueUb, compLenHLPD * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(gradOutputBoradUb, compLenHLPD * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(gradValueUb, compLenHLPD * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(gradAttnWeightsUb, compLenTHLP * sizeof(DTYPE_VALUE));
    }

    __aicore__ inline void GetLocalTensor()
    {
        spatialShapesLocal = spatialShapesUb.Get<DTYPE_SPATIAL_SHAPES>();
        spatialShapesOriginLocal = spatialShapesOriginUb.Get<DTYPE_SPATIAL_SHAPES>();
        levelStartIdxLocal = levelStartIdxUb.Get<DTYPE_SPATIAL_SHAPES>();
        spatialShapesBroadLocal = spatialShapesBroadUb.Get<DTYPE_SPATIAL_SHAPES>();
        levelStartIdxBroadLocal = levelStartIdxBroadUb.Get<DTYPE_SPATIAL_SHAPES>();
        headStartIdxBroadLocal = headStartIdxBroadUb.Get<DTYPE_SPATIAL_SHAPES>();

        samplingLocationLocal = samplingLocationUb.Get<DTYPE_SPATIAL_SHAPES>();
        keyIdxsLocal = keyIdxsUb.Get<DTYPE_SPATIAL_SHAPES>();
        
        gradOutputLocal = gradOutputUb.Get<DTYPE_VALUE>();
        attnWeightLocal = attnWeightUb.Get<DTYPE_VALUE>();
        valueLocal = valueUb.Get<DTYPE_VALUE>();
        gradOutputBoradLocal = gradOutputBoradUb.Get<DTYPE_VALUE>();

        gradValueLocal = gradValueUb.Get<DTYPE_VALUE>();
        gradAttnWeightsLocal = gradAttnWeightsUb.Get<DTYPE_VALUE>();
    }

    template<typename DType>
    __aicore__ inline void BroadCastForDtype32B(const LocalTensor<DType> dstLocal, const LocalTensor<DType> srcLocal,
                                           int32_t broadDim, uint32_t dstOffset, uint32_t srcOffset, uint32_t dstShape[2], uint32_t srcShape[2])
    {
        LocalTensor<float> broadSrcLocal = srcLocal[srcOffset].template ReinterpretCast<float>();
        LocalTensor<float> broadDstLocal = dstLocal[dstOffset].template ReinterpretCast<float>();
        if (broadDim == 0) {
            BroadCast<float, 2, 0>(broadDstLocal, broadSrcLocal, dstShape, srcShape);
        } else {
            BroadCast<float, 2, 1>(broadDstLocal, broadSrcLocal, dstShape, srcShape);
        }
    }

private:
    TPipe *pipe;

    GlobalTensor<DTYPE_VALUE> valueGm, attnWeightsGm, gradOutputGm;
    GlobalTensor<DTYPE_VALUE> gradValueGm, gradAttnWeightsGm;
    GlobalTensor<DTYPE_SPATIAL_SHAPES> spatialShapesGm, levelStartIndexGm;
    GlobalTensor<DTYPE_SAMPLING_LOCATIONS> samplingLocationsGm;

    TBuf<TPosition::VECCALC> spatialShapesUb, spatialShapesOriginUb, levelStartIdxUb;
    TBuf<TPosition::VECCALC> spatialShapesBroadUb, levelStartIdxBroadUb, headStartIdxBroadUb;
    TBuf<TPosition::VECCALC> samplingLocationUb, keyIdxsUb;
    TBuf<TPosition::VECCALC> gradOutputUb, attnWeightUb, valueUb, gradOutputBoradUb;
    TBuf<TPosition::VECCALC> gradValueUb, gradAttnWeightsUb;

    LocalTensor<DTYPE_SPATIAL_SHAPES> spatialShapesLocal, spatialShapesOriginLocal, levelStartIdxLocal;
    LocalTensor<DTYPE_SPATIAL_SHAPES> spatialShapesBroadLocal, levelStartIdxBroadLocal, headStartIdxBroadLocal;
    LocalTensor<DTYPE_SPATIAL_SHAPES> samplingLocationLocal, keyIdxsLocal;
    LocalTensor<DTYPE_VALUE> gradOutputLocal, attnWeightLocal, valueLocal, gradOutputBoradLocal;
    LocalTensor<DTYPE_VALUE> gradValueLocal, gradAttnWeightsLocal;

    DTYPE_SPATIAL_SHAPES H, W;

    uint32_t numItemsPerBlock, blockBytes, coreNum, coreId;
    uint32_t batchSize, embedDims, numKeys, numQueries, numPoints, numLevels, numHeads;
    uint32_t numLevelsAligned, numPointsAligned;
    uint32_t batchIdx, keyIdx, queryIdx, headIdx, levelIdx, pointIdx;
    uint32_t dstShapeCGO[2], dstShapeCAW[2], srcShapeCGO[2], srcShapeCAW[2];
    uint32_t levelTmpBroadSrcShape[2], levelTmpBroadDstShape[2], levelBroadSrcShape[2], levelBroadDstShape[2];
    uint32_t spatailTmpBroadSrcShape[2], spatailTmpBroadDstShape[2], spatailBroadSrcShape[2], spatailBroadDstShape[2];
    uint32_t headBroadSrcShape[2], headBroadDstShape[2];
    uint64_t samplingLocationOffset, rsvdCnt;
    uint64_t gradOutputOffset, attnWeightsOffset, samplingGMOffset, samplingUBOffset;
    uint32_t taskIdxStart, pointLoopIdx, pointIdxActual, compLenActual, gradOutputIdx;
    uint64_t valueMapIdx, valueLocalIdx;
    uint32_t taskPerCore, taskCoreTail, taskCompNum;
    uint32_t taskStart, taskEnd, taskCompTail, taskIdx;
    uint32_t taskCompIdx, taskCompActual, groupCompIdx, taskOffset;
    uint32_t broadDstOffset, groupOffsetStart, valueMapStart;

    DataCopyParams copyParams {1, 0, 0, 0};
    DataCopyParams samplingCopyParams {1, 0, 0, 0};
    GatherMaskParams samplingGatherParams {1, 0, 0, 0};
};

__aicore__ inline void GeometricKernelAttnGrad::Process()
{
    for (headIdx = 0; headIdx < numHeads; headIdx++) {
        Duplicate(keyIdxsLocal[headIdx * numLevels * numPointsAligned], static_cast<DTYPE_SPATIAL_SHAPES>(headIdx * embedDims), numLevels * numPointsAligned);
    }
    BroadCastForDtype32B<DTYPE_SPATIAL_SHAPES>(headStartIdxBroadLocal, keyIdxsLocal, 0, 0, 0, headBroadDstShape, headBroadSrcShape);

    DataCopy(spatialShapesOriginLocal, spatialShapesGm, 2 * numLevelsAligned);
    DataCopy(levelStartIdxLocal, levelStartIndexGm, numLevelsAligned);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    for (levelIdx = 0; levelIdx < numLevels; levelIdx++) {
        H = spatialShapesOriginLocal.GetValue(levelIdx * 2);
        W = spatialShapesOriginLocal.GetValue(levelIdx * 2 + 1);
        spatialShapesLocal.SetValue(levelIdx, H);
        spatialShapesLocal.SetValue(levelIdx + numLevels, W);
    }

    BroadCastForDtype32B<DTYPE_SPATIAL_SHAPES>(keyIdxsLocal, levelStartIdxLocal, 1, 0, 0, levelTmpBroadDstShape, levelTmpBroadSrcShape);
    BroadCastForDtype32B<DTYPE_SPATIAL_SHAPES>(levelStartIdxBroadLocal, keyIdxsLocal, 0, 0, 0, levelBroadDstShape, levelBroadSrcShape);

    BroadCastForDtype32B<DTYPE_SPATIAL_SHAPES>(keyIdxsLocal, spatialShapesLocal, 1, 0, 0, spatailTmpBroadDstShape, spatailTmpBroadSrcShape);
    BroadCastForDtype32B<DTYPE_SPATIAL_SHAPES>(spatialShapesBroadLocal, keyIdxsLocal, 0, 0, 0, spatailBroadDstShape, spatailBroadSrcShape);
    BroadCastForDtype32B<DTYPE_SPATIAL_SHAPES>(spatialShapesBroadLocal, keyIdxsLocal, 0, samplingUBOffset, numLevels * numPointsAligned, spatailBroadDstShape, spatailBroadSrcShape);

    for (taskIdx = taskStart; taskIdx < taskEnd; taskIdx += taskCompNum) {
        taskCompActual = (taskIdx + taskCompNum <= taskEnd) ? taskCompNum : taskCompTail;
        copyParams.blockCount = taskCompActual * numHeads * numLevels;
        samplingCopyParams.blockCount = taskCompActual * numHeads * numLevels;
        samplingGatherParams.repeatTimes = taskCompActual * numHeads * numLevels;
        compLenActual = taskCompActual * numHeads * numLevels * numPointsAligned;

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        samplingLocationOffset = taskIdx * numHeads * numLevels * numPoints * 2;
        DataCopyPad(keyIdxsLocal.ReinterpretCast<DTYPE_SAMPLING_LOCATIONS>(), samplingLocationsGm[samplingLocationOffset], samplingCopyParams, {});

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        Cast(keyIdxsLocal.ReinterpretCast<DTYPE_SPATIAL_SHAPES>(), keyIdxsLocal.ReinterpretCast<DTYPE_SAMPLING_LOCATIONS>(),
             AscendC::RoundMode::CAST_RINT, taskCompActual * numHeads * numLevels * numPointsAligned * 2);
        GatherMask(samplingLocationLocal, keyIdxsLocal, 2, true, numPointsAligned * 2, samplingGatherParams, rsvdCnt);
        GatherMask(samplingLocationLocal[samplingUBOffset], keyIdxsLocal, 1, true, numPointsAligned * 2, samplingGatherParams, rsvdCnt);

        Maxs(samplingLocationLocal, samplingLocationLocal, static_cast<DTYPE_SPATIAL_SHAPES>(0), samplingUBOffset * 2);
        Adds(keyIdxsLocal, spatialShapesBroadLocal, static_cast<DTYPE_SPATIAL_SHAPES>(-1), samplingUBOffset * 2);
        Min(samplingLocationLocal, samplingLocationLocal, keyIdxsLocal, samplingUBOffset * 2);

        Mul(keyIdxsLocal, samplingLocationLocal, spatialShapesBroadLocal[samplingUBOffset], compLenActual);
        Add(keyIdxsLocal, keyIdxsLocal, samplingLocationLocal[samplingUBOffset], compLenActual);
        Add(keyIdxsLocal, keyIdxsLocal, levelStartIdxBroadLocal, compLenActual);
        Muls(keyIdxsLocal, keyIdxsLocal, static_cast<DTYPE_SPATIAL_SHAPES>(numHeads * embedDims), compLenActual);
        Add(keyIdxsLocal, keyIdxsLocal, headStartIdxBroadLocal, compLenActual);

        gradOutputOffset = taskIdx * numHeads * embedDims;
        attnWeightsOffset = taskIdx * numHeads * numLevels * numPoints;

        DataCopy(gradOutputLocal, gradOutputGm[gradOutputOffset], taskCompActual * numHeads * embedDims);
        DataCopyPad(attnWeightLocal, attnWeightsGm[attnWeightsOffset], copyParams, {});

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        for (taskCompIdx = 0; taskCompIdx < taskCompActual; taskCompIdx++) {
            batchIdx = (taskCompIdx + taskIdx) / numQueries;
            valueMapStart = batchIdx * numKeys * numHeads * embedDims;
            taskIdxStart = taskCompIdx * numHeads * numLevels * numPointsAligned;

            // (heads, embed) -> (heads, levels * points, embed)
            for (headIdx = 0; headIdx < numHeads; headIdx++) {
                gradOutputIdx = (taskCompIdx * numHeads + headIdx) * embedDims;
                broadDstOffset = headIdx * numLevels * numPointsAligned * embedDims;
                BroadCastForDtype32B<DTYPE_VALUE>(gradOutputBoradLocal, gradOutputLocal, 0, broadDstOffset, gradOutputIdx, dstShapeCGO, srcShapeCGO);
            }
            
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            BroadCast<DTYPE_VALUE, 2, 1>(gradValueLocal, attnWeightLocal[taskIdxStart], dstShapeCAW, srcShapeCAW);
            Mul(gradValueLocal, gradValueLocal, gradOutputBoradLocal, numHeads * numLevels * numPointsAligned * embedDims);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            for (headIdx = 0; headIdx < numHeads; headIdx++) {
                for (levelIdx = 0; levelIdx < numLevels; levelIdx++) {
                    for (pointIdx = 0; pointIdx < numPoints; pointIdx++) {
                        pointIdxActual = (headIdx * numLevels + levelIdx) * numPointsAligned + pointIdx;
                        valueLocalIdx = pointIdxActual * embedDims;
                        valueMapIdx = valueMapStart + keyIdxsLocal.GetValue(taskIdxStart + pointIdxActual);

                        SetAtomicAdd<DTYPE_VALUE>();
                        DataCopy(gradValueGm[valueMapIdx], gradValueLocal[valueLocalIdx], embedDims);
                        SetAtomicNone();
                        DataCopy(valueLocal[valueLocalIdx], valueGm[valueMapIdx], embedDims);
                    }
                }
            }

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            Mul(valueLocal, valueLocal, gradOutputBoradLocal, numHeads * numLevels * numPointsAligned * embedDims);
            RepeatReduceSum(gradAttnWeightsLocal[taskIdxStart], valueLocal, numHeads * numLevels * numPointsAligned,
                            embedDims, 0, 1, 1, embedDims / (32 / sizeof(DTYPE_VALUE)));
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        DataCopyPad(gradAttnWeightsGm[attnWeightsOffset], gradAttnWeightsLocal, copyParams);
    }
}

extern "C" __global__ __aicore__ void geometric_kernel_attn_grad(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm,
    GM_ADDR level_start_index_gm, GM_ADDR sampling_locations_gm, GM_ADDR attn_weights_gm, GM_ADDR grad_output_gm,
    GM_ADDR grad_value_gm, GM_ADDR grad_attn_weights_gm, GM_ADDR workspace, GM_ADDR tiling_data)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_datas, tiling_data);

    TPipe pipe;
    GeometricKernelAttnGrad op;

    op.Init(value_gm, spatial_shapes_gm, level_start_index_gm, sampling_locations_gm, attn_weights_gm, grad_output_gm,
        grad_value_gm, grad_attn_weights_gm, &tiling_datas, &pipe);
    op.Process();
}
