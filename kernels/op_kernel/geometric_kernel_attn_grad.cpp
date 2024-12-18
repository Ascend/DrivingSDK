/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

constexpr MatmulConfig MATMUL_CFG = GetMDLConfig(false, false, 0, false, false, false, true);

template<bool useOpt>
class GeometricKernelAttnGrad {
public:
    using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND_ALIGN, float>;

    matmul::Matmul<AType, BType, CType, CType, MATMUL_CFG> mmObj;

    __aicore__ inline GeometricKernelAttnGrad() = default;

    __aicore__ inline void Init(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm, GM_ADDR level_start_index_gm,
                                GM_ADDR sampling_locations_gm, GM_ADDR attn_weights_gm, GM_ADDR grad_output_gm,
                                GM_ADDR grad_value_gm, GM_ADDR grad_attn_weights_gm, GM_ADDR usrWorkspace,
                                const GeometricKernelAttnGradTilingData *tiling_data, TPipe *tmpPipe)
    {
        pipe = tmpPipe;
        curBlockIdx = GetBlockIdx();
        blockBytes = 32;
        numItemsPerBlock = blockBytes / sizeof(DTYPE_VALUE);

        GetTilingData(tiling_data);
        InitProperties();
        AllocEventID();
        
        SetGlobalBuffer(value_gm, spatial_shapes_gm, level_start_index_gm, sampling_locations_gm, attn_weights_gm,
                        grad_output_gm, grad_value_gm, grad_attn_weights_gm, usrWorkspace);
        InitBuffer();
        GetLocalTensor();
    }

    __aicore__ inline void Process();

private:
    __aicore__ inline void GetTilingData(const GeometricKernelAttnGradTilingData *tiling_data)
    {
        batchSize = tiling_data->batchSize;
        embedDims = tiling_data->embedDims;
        numKeys = tiling_data->numKeys;
        numLevels = tiling_data->numLevels;
        numQueries = tiling_data->numQueries;
        numPoints = tiling_data->numPoints;
        coreNum = tiling_data->coreNum;
        numLargeCores = tiling_data->numLargeCores;
        numQueriesPerBundle = tiling_data->numQueriesPerBundle;
        numQueriesPerLargeCore = tiling_data->numQueriesPerLargeCore;
    }

    __aicore__ inline void InitProperties()
    {
        numLevelsAligned = AlignUp(numLevels, numItemsPerBlock);
        numKeysAligned = AlignUp(numKeys, numItemsPerBlock);
        numPointsAligned = AlignUp(numPoints, numItemsPerBlock);

        if (curBlockIdx < numLargeCores) {
            numQueriesCurCore = numQueriesPerLargeCore;
            startOffset = curBlockIdx * numQueriesPerLargeCore;
        } else {
            numQueriesCurCore = numQueriesPerLargeCore - 1;
            startOffset = numLargeCores * numQueriesPerLargeCore + (curBlockIdx - numLargeCores) * numQueriesCurCore;
            mmObj.SetTail(numQueriesCurCore, numKeys, embedDims);
        }

        numQueryLoops = (numQueriesCurCore + numQueriesPerBundle - 1) / numQueriesPerBundle;
        numQueriesLastBundle = numQueriesCurCore - (numQueryLoops - 1) * numQueriesPerBundle;

        mmTmp4GradAttnWeightsIdx = static_cast<uint64_t>(startOffset) * numKeysAligned;

        dstShape[0] = numPointsAligned;
        dstShape[1] = embedDims;
        srcShapeCGO[0] = 1;
        srcShapeCGO[1] = embedDims;
        srcShapeCAW[0] = numPointsAligned;
        srcShapeCAW[1] = 1;

        copyParams.blockLen = numPoints * sizeof(DTYPE_VALUE);
    }

    __aicore__ inline void AllocEventID()
    {
        eventIdMte2ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdMte3ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());
        eventIdVToMte2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMte3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdMte3ToMte2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_MTE2>());
    }

    __aicore__ inline void SetGlobalBuffer(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm, GM_ADDR level_start_index_gm,
                                           GM_ADDR sampling_locations_gm, GM_ADDR attn_weights_gm, GM_ADDR grad_output_gm,
                                           GM_ADDR grad_value_gm, GM_ADDR grad_attn_weights_gm, GM_ADDR usrWorkspace)
    {
        valueGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(value_gm),
                                batchSize * numKeys * embedDims);
        spatialShapesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_SPATIAL_SHAPES *>(spatial_shapes_gm),
                                        numLevels * 2);
        levelStartIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_SPATIAL_SHAPES *>(level_start_index_gm),
                                          numLevels);
        samplingLocationsGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(sampling_locations_gm),
                                            batchSize * numQueries * numLevels * numPoints * 2);
        attnWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(attn_weights_gm),
                                      batchSize * numQueries * numLevels * numPoints);
        gradOutputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(grad_output_gm),
                                     batchSize * numQueries * embedDims);

        gradValueGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(grad_value_gm),
                                    batchSize * numKeys * embedDims);
        gradAttnWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(grad_attn_weights_gm),
                                          batchSize * numQueries * numLevels * numPoints);
        tmp4GradAttnWeightsGM.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(usrWorkspace),
                                              numQueries * numKeysAligned);
    }

    __aicore__ inline void InitBuffer()
    {
        pipe->InitBuffer(spatialShapesUb, static_cast<uint64_t>(2) * numLevelsAligned * sizeof(DTYPE_SPATIAL_SHAPES));
        pipe->InitBuffer(levelStartIdxUb, static_cast<uint64_t>(numLevelsAligned) * sizeof(DTYPE_SPATIAL_SHAPES));

        pipe->InitBuffer(curGradAttnWeightsUb, static_cast<uint64_t>(numQueriesPerBundle) * numPointsAligned * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(keyIdxsUb, static_cast<uint64_t>(numQueriesPerBundle) * numPointsAligned * sizeof(DTYPE_VALUE));
        
        pipe->InitBuffer(curGradOutputUb, static_cast<uint64_t>(numQueriesPerBundle) * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(curAttnWeightUb, static_cast<uint64_t>(numQueriesPerBundle) * numPointsAligned * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(curTmp4GradAttnWeightsUb, static_cast<uint64_t>(numQueriesPerBundle) * numKeysAligned * sizeof(DTYPE_VALUE));
        
        pipe->InitBuffer(curAttnWeightBrcbUb, static_cast<uint64_t>(numPointsAligned) * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmp4GradValueUb, static_cast<uint64_t>(numPointsAligned) * embedDims * sizeof(DTYPE_VALUE));

        if (useOpt) {
            pipe->InitBuffer(curGradValueUb, static_cast<uint64_t>(numKeys) * embedDims * sizeof(DTYPE_VALUE));
        }
    }

    __aicore__ inline void GetLocalTensor()
    {
        spatialShapesLocal = spatialShapesUb.Get<DTYPE_SPATIAL_SHAPES>();
        levelStartIdxLocal = levelStartIdxUb.Get<DTYPE_SPATIAL_SHAPES>();

        curGradAttnWeightsLocal = curGradAttnWeightsUb.Get<DTYPE_VALUE>();
        keyIdxsLocal = keyIdxsUb.Get<DTYPE_VALUE>();
        
        curGradOutputLocal = curGradOutputUb.Get<DTYPE_VALUE>();
        curAttnWeightLocal = curAttnWeightUb.Get<DTYPE_VALUE>();
        curTmp4GradAttnWeightsLocal = curTmp4GradAttnWeightsUb.Get<DTYPE_VALUE>();

        curAttnWeightBrcbLocal = curAttnWeightBrcbUb.Get<DTYPE_VALUE>();
        tmp4GradValueLocal = tmp4GradValueUb.Get<DTYPE_VALUE>();

        if (useOpt) {
            curGradValueLocal = curGradValueUb.Get<DTYPE_VALUE>();
        }
    }

    __aicore__ inline void ComputeCurLevel();

    __aicore__ inline void ComputeCurBatch();

    __aicore__ inline void ComputeCurLoopFront();

    __aicore__ inline void ComputeCurLoopMiddle();

    __aicore__ inline void ComputeCurLoopBack();

    __aicore__ inline void ComputeCurLoopBackOpt();

    __aicore__ inline void ComputeCurQueryPoints();

    __aicore__ inline void ComputeCurQueryPointsOpt();

private:
    TPipe *pipe;

    GlobalTensor<DTYPE_VALUE> valueGm, samplingLocationsGm, attnWeightsGm, gradOutputGm;
    GlobalTensor<DTYPE_VALUE> gradValueGm, gradAttnWeightsGm, tmp4GradAttnWeightsGM;
    GlobalTensor<DTYPE_SPATIAL_SHAPES> spatialShapesGm, levelStartIndexGm;

    TBuf<TPosition::VECCALC> spatialShapesUb, levelStartIdxUb;
    TBuf<TPosition::VECCALC> curGradAttnWeightsUb, keyIdxsUb;
    TBuf<TPosition::VECCALC> curGradOutputUb, curAttnWeightUb, curTmp4GradAttnWeightsUb;
    TBuf<TPosition::VECCALC> curAttnWeightBrcbUb, tmp4GradValueUb, curGradValueUb;

    LocalTensor<DTYPE_SPATIAL_SHAPES> spatialShapesLocal, levelStartIdxLocal;
    LocalTensor<DTYPE_VALUE> curGradAttnWeightsLocal, keyIdxsLocal;
    LocalTensor<DTYPE_VALUE> curGradOutputLocal, curAttnWeightLocal, curTmp4GradAttnWeightsLocal;
    LocalTensor<DTYPE_VALUE> curAttnWeightBrcbLocal, tmp4GradValueLocal, curGradValueLocal;

    DTYPE_SPATIAL_SHAPES h, w, levelStartIdx;
    DTYPE_VALUE curTmp4GradAttnWeight;

    uint32_t numItemsPerBlock, blockBytes;
    uint32_t coreNum, curBlockIdx, numLargeCores;
    uint32_t batchSize, embedDims, numKeys, numQueries, numPoints, numLevels;
    uint32_t numLevelsAligned, numKeysAligned, numPointsAligned;
    uint32_t batchIdx, keyIdx, queryIdx, headIdx, levelIdx, pointIdx;
    uint32_t dstShape[2], srcShapeCGO[2], srcShapeCAW[2];
    uint32_t numQueriesPerLargeCore, numQueriesCurCore, numQueriesPerBundle, numQueriesLastBundle, numQueriesCurBundle;
    uint32_t loopIdx, numQueryLoops;
    uint32_t startOffset, queryIdxStart, queryIdxEnd, queryOffsetBundle;
    uint64_t samplingLocationXIdx, samplingLocationYIdx;
    uint64_t gradOutputIdx, curGradOutputIdx, gradValueIdx, tmp4GradValueIdx, curGradValueIdx;
    uint64_t attnWeightsIdx, curAttnWeightsIdx, gradAttnWeightsIdx, tmp4GradAttnWeightsIdx, curTmp4GradAttnWeightIdx;
    uint64_t mmTmp4GradAttnWeightsIdx, mmGradOutputIdx, mmValueIdx;

    DataCopyParams copyParams {1, 0, 0, 0};

    event_t eventIdVToMte2, eventIdVToMte3;
    event_t eventIdMte2ToV, eventIdMte3ToV;
    event_t eventIdMte3ToMte2;
};

template<bool useOpt>
__aicore__ inline void GeometricKernelAttnGrad<useOpt>::ComputeCurLevel()
{
    levelStartIdx = levelStartIdxLocal.GetValue(levelIdx);
    h = spatialShapesLocal.GetValue(levelIdx * 2);
    w = spatialShapesLocal.GetValue(levelIdx * 2 + 1);
}

template<bool useOpt>
__aicore__ inline void GeometricKernelAttnGrad<useOpt>::ComputeCurBatch()
{
    mmGradOutputIdx = static_cast<uint64_t>(batchIdx) * numQueries * embedDims + startOffset * embedDims;
    mmObj.SetTensorA(gradOutputGm[mmGradOutputIdx]);

    mmValueIdx = static_cast<uint64_t>(batchIdx) * numKeys * embedDims;
    mmObj.SetTensorB(valueGm[mmValueIdx], true);

    mmObj.IterateAll(tmp4GradAttnWeightsGM[mmTmp4GradAttnWeightsIdx]);
    mmObj.End();
}

template<bool useOpt>
__aicore__ inline void GeometricKernelAttnGrad<useOpt>::ComputeCurLoopFront()
{
    numQueriesCurBundle = (loopIdx == numQueryLoops - 1) ? numQueriesLastBundle : numQueriesPerBundle;
    copyParams.blockCount = numQueriesCurBundle;

    queryIdxStart = startOffset + loopIdx * numQueriesPerBundle;

    samplingLocationXIdx = static_cast<uint64_t>(levelIdx) * batchSize * 2 * numQueries * numPoints + \
                           batchIdx * 2 * numQueries * numPoints + \
                           queryIdxStart * numPoints;
    DataCopyPad(curGradAttnWeightsLocal, samplingLocationsGm[samplingLocationXIdx], copyParams, {});

    samplingLocationYIdx = samplingLocationXIdx + numQueries * numPoints;
    DataCopyPad(keyIdxsLocal, samplingLocationsGm[samplingLocationYIdx], copyParams, {});

    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    gradOutputIdx = static_cast<uint64_t>(batchIdx) * numQueries * embedDims + queryIdxStart * embedDims;
    attnWeightsIdx = static_cast<uint64_t>(levelIdx) * batchSize * numQueries * numPoints + \
                     batchIdx * numQueries * numPoints + \
                     queryIdxStart * numPoints;
    tmp4GradAttnWeightsIdx = static_cast<uint64_t>(queryIdxStart) * numKeysAligned;

    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    ClampMin(curGradAttnWeightsLocal, curGradAttnWeightsLocal, static_cast<DTYPE_VALUE>(0), numQueriesCurBundle * numPointsAligned);
    ClampMin(keyIdxsLocal, keyIdxsLocal, static_cast<DTYPE_VALUE>(0), numQueriesCurBundle * numPointsAligned);
    ClampMax(curGradAttnWeightsLocal, curGradAttnWeightsLocal, static_cast<DTYPE_VALUE>(w - 1), numQueriesCurBundle * numPointsAligned);
    ClampMax(keyIdxsLocal, keyIdxsLocal, static_cast<DTYPE_VALUE>(h - 1), numQueriesCurBundle * numPointsAligned);
    Muls(keyIdxsLocal, keyIdxsLocal, static_cast<DTYPE_VALUE>(w), numQueriesCurBundle * numPointsAligned);
    Add(keyIdxsLocal, keyIdxsLocal, curGradAttnWeightsLocal, numQueriesCurBundle * numPointsAligned);
    Adds(keyIdxsLocal, keyIdxsLocal, static_cast<DTYPE_VALUE>(levelStartIdx), numQueriesCurBundle * numPointsAligned);
    Duplicate(curGradAttnWeightsLocal, static_cast<DTYPE_VALUE>(0), numQueriesCurBundle * numPointsAligned);

    DataCopy(curGradOutputLocal, gradOutputGm[gradOutputIdx], numQueriesCurBundle * embedDims);
    DataCopyPad(curAttnWeightLocal, attnWeightsGm[attnWeightsIdx], copyParams, {});
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    DataCopy(curTmp4GradAttnWeightsLocal, tmp4GradAttnWeightsGM[tmp4GradAttnWeightsIdx], numQueriesCurBundle * numKeysAligned);

    if (useOpt) {
        Duplicate(curGradValueLocal, static_cast<DTYPE_VALUE>(0), numKeys * embedDims);
        gradValueIdx = static_cast<uint64_t>(batchIdx) * numKeys * embedDims;
    }

    queryIdxEnd = queryIdxStart + numQueriesCurBundle;

    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
}

template<bool useOpt>
__aicore__ inline void GeometricKernelAttnGrad<useOpt>::ComputeCurQueryPoints()
{
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    SetAtomicAdd<DTYPE_VALUE>();
    for (pointIdx = 0; pointIdx < numPoints; pointIdx++) {
        gradAttnWeightsIdx = static_cast<uint64_t>(queryOffsetBundle) * numPointsAligned + pointIdx;
        keyIdx = static_cast<DTYPE_SPATIAL_SHAPES>(keyIdxsLocal.GetValue(gradAttnWeightsIdx));

        gradValueIdx = static_cast<uint64_t>(batchIdx) * numKeys * embedDims + keyIdx * embedDims;
        tmp4GradValueIdx = static_cast<uint64_t>(pointIdx) * embedDims;
        DataCopy(gradValueGm[gradValueIdx], tmp4GradValueLocal[tmp4GradValueIdx], embedDims);

        curTmp4GradAttnWeightIdx = static_cast<uint64_t>(queryOffsetBundle) * numKeysAligned + keyIdx;
        curTmp4GradAttnWeight = curTmp4GradAttnWeightsLocal.GetValue(curTmp4GradAttnWeightIdx);
        curGradAttnWeightsLocal.SetValue(gradAttnWeightsIdx, curTmp4GradAttnWeight);
    }
    SetAtomicNone();

    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
}

template<bool useOpt>
__aicore__ inline void GeometricKernelAttnGrad<useOpt>::ComputeCurQueryPointsOpt()
{
    for (pointIdx = 0; pointIdx < numPoints; pointIdx++) {
        gradAttnWeightsIdx = static_cast<uint64_t>(queryOffsetBundle) * numPointsAligned + pointIdx;
        keyIdx = static_cast<DTYPE_SPATIAL_SHAPES>(keyIdxsLocal.GetValue(gradAttnWeightsIdx));

        curGradValueIdx = static_cast<uint64_t>(embedDims) * keyIdx;
        tmp4GradValueIdx = static_cast<uint64_t>(embedDims) * pointIdx;
        PipeBarrier<PIPE_V>();
        Add(curGradValueLocal[curGradValueIdx], curGradValueLocal[curGradValueIdx], tmp4GradValueLocal[tmp4GradValueIdx], embedDims);

        curTmp4GradAttnWeightIdx = static_cast<uint64_t>(queryOffsetBundle) * numKeysAligned + keyIdx;
        curTmp4GradAttnWeight = curTmp4GradAttnWeightsLocal.GetValue(curTmp4GradAttnWeightIdx);
        curGradAttnWeightsLocal.SetValue(gradAttnWeightsIdx, curTmp4GradAttnWeight);
    }
}

template<bool useOpt>
__aicore__ inline void GeometricKernelAttnGrad<useOpt>::ComputeCurLoopMiddle()
{
#pragma bisheng auto_sync parallel
    for (queryIdx = queryIdxStart; queryIdx < queryIdxEnd; queryIdx++) {
        queryOffsetBundle = queryIdx - queryIdxStart;

        curGradOutputIdx = static_cast<uint64_t>(queryOffsetBundle) * embedDims;
        PipeBarrier<PIPE_V>();
        BroadCast<DTYPE_VALUE, 2, 0>(tmp4GradValueLocal, curGradOutputLocal[curGradOutputIdx], dstShape, srcShapeCGO);

        curAttnWeightsIdx = static_cast<uint64_t>(queryOffsetBundle) * numPointsAligned;
        BroadCast<DTYPE_VALUE, 2, 1>(curAttnWeightBrcbLocal, curAttnWeightLocal[curAttnWeightsIdx], dstShape, srcShapeCAW);
        
        PipeBarrier<PIPE_V>();
        Mul(tmp4GradValueLocal, tmp4GradValueLocal, curAttnWeightBrcbLocal, numPointsAligned * embedDims);

        if (useOpt) {
            ComputeCurQueryPointsOpt();
        } else {
            ComputeCurQueryPoints();
        }
    }
}

template<bool useOpt>
__aicore__ inline void GeometricKernelAttnGrad<useOpt>::ComputeCurLoopBack()
{
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);

    DataCopyPad(gradAttnWeightsGm[attnWeightsIdx], curGradAttnWeightsLocal, copyParams);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

template<bool useOpt>
__aicore__ inline void GeometricKernelAttnGrad<useOpt>::ComputeCurLoopBackOpt()
{
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    SetAtomicAdd<DTYPE_VALUE>();
    DataCopy(gradValueGm[gradValueIdx], curGradValueLocal, numKeys * embedDims);
    SetAtomicNone();
    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);

    DataCopyPad(gradAttnWeightsGm[attnWeightsIdx], curGradAttnWeightsLocal, copyParams);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

template<bool useOpt>
__aicore__ inline void GeometricKernelAttnGrad<useOpt>::Process()
{
    DataCopy(spatialShapesLocal, spatialShapesGm, 2 * numLevelsAligned);
    DataCopy(levelStartIdxLocal, levelStartIndexGm, numLevelsAligned);

    for (levelIdx = 0; levelIdx < numLevels; levelIdx++) {
        ComputeCurLevel();

        for (batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            ComputeCurBatch();

            for (loopIdx = 0; loopIdx < numQueryLoops; loopIdx++) {
                ComputeCurLoopFront();
                ComputeCurLoopMiddle();
                if (useOpt) {
                    ComputeCurLoopBackOpt();
                } else {
                    ComputeCurLoopBack();
                }
            }
        }
    }
}

extern "C" __global__ __aicore__ void geometric_kernel_attn_grad(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm,
    GM_ADDR level_start_index_gm, GM_ADDR sampling_locations_gm, GM_ADDR attn_weights_gm, GM_ADDR grad_output_gm,
    GM_ADDR grad_value_gm, GM_ADDR grad_attn_weights_gm, GM_ADDR workspace, GM_ADDR tiling_data)
{
    GET_TILING_DATA(tiling_datas, tiling_data);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }

    TPipe pipe;

    if (TILING_KEY_IS(0)) {
        GeometricKernelAttnGrad<false> op;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mmObj, &(tiling_datas.mmTilingData));

        op.Init(value_gm, spatial_shapes_gm, level_start_index_gm, sampling_locations_gm, attn_weights_gm, grad_output_gm,
            grad_value_gm, grad_attn_weights_gm, usrWorkspace, &tiling_datas, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        GeometricKernelAttnGrad<true> op;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mmObj, &(tiling_datas.mmTilingData));

        op.Init(value_gm, spatial_shapes_gm, level_start_index_gm, sampling_locations_gm, attn_weights_gm, grad_output_gm,
            grad_value_gm, grad_attn_weights_gm, usrWorkspace, &tiling_datas, &pipe);
        op.Process();
    }
}
