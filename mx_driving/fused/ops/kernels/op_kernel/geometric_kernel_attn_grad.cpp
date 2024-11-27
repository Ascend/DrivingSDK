/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

constexpr MatmulConfig MATMUL_CFG = GetMDLConfig(false, false, 2, false, false, false, true);

class GeometricKernelAttnGrad {
public:
    using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND_ALIGN, float>;

    matmul::Matmul<AType, BType, CType, CType, MATMUL_CFG> mm_;

    __aicore__ inline GeometricKernelAttnGrad(){};
    __aicore__ inline void Init(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm, GM_ADDR level_start_index_gm,
                                GM_ADDR sampling_locations_gm, GM_ADDR attn_weights_gm, GM_ADDR grad_output_gm,
                                GM_ADDR grad_value_gm, GM_ADDR grad_attn_weights_gm, GM_ADDR usrWorkspace,
                                const GeometricKernelAttnGradTilingData *tiling_data, TPipe *tmpPipe)
    {
        pipe = tmpPipe;
        curBlockIdx = GetBlockIdx();
        blockBytes = 32;
        dataAlign = blockBytes / sizeof(DTYPE_VALUE);

        GetTilingData(tiling_data);
        InitProperties();
        AllocEventID();
        
        SetGlobalBuffer(value_gm, spatial_shapes_gm, level_start_index_gm, sampling_locations_gm, attn_weights_gm,
                        grad_output_gm, grad_value_gm, grad_attn_weights_gm, usrWorkspace);
        InitBuffer();
        GetLocalTensor();
    }

    __aicore__ inline void Process();

    __aicore__ inline void ReleaseEventID()
    {
        pipe->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
        pipe->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV);
        pipe->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
        pipe->ReleaseEventID<HardEvent::S_V>(eventIdSToV);
        pipe->ReleaseEventID<HardEvent::V_S>(eventIdVToS);
        pipe->ReleaseEventID<HardEvent::MTE2_S>(eventIdMte2ToS);
        pipe->ReleaseEventID<HardEvent::S_MTE2>(eventIdSToMte2);
        pipe->ReleaseEventID<HardEvent::S_MTE3>(eventIdSToMte3);
        pipe->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }

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
        numQueriesPerBundle = tiling_data->numQueriesPerBundle;
        numQueriesPerCore = tiling_data->numQueriesPerCore;
    }

    __aicore__ inline void InitProperties()
    {
        numLevelsAlign = AlignUp(numLevels, dataAlign);
        numKeysAlign = AlignUp(numKeys, dataAlign);
        numPointsAlign = AlignUp(numPoints, dataAlign);

        startOffset = curBlockIdx * numQueriesPerCore;
        endOffset = (curBlockIdx + 1) * numQueriesPerCore;
        if (endOffset > numQueries) {
            endOffset = numQueries;
        }
        numQueriesCurCore = endOffset - startOffset;
        if (numQueriesCurCore < numQueriesPerCore) {
            mm_.SetTail(numQueriesCurCore, numKeys, embedDims);
        }

        numQueryLoops = (numQueriesCurCore + numQueriesPerBundle - 1) / numQueriesPerBundle;
        numQueriesLastBundle = numQueriesCurCore - (numQueryLoops - 1) * numQueriesPerBundle;

        dstShape[0] = numPointsAlign;
        dstShape[1] = embedDims;
        srcShapeCGO[0] = 1;
        srcShapeCGO[1] = embedDims;
        srcShapeCAW[0] = numPointsAlign;
        srcShapeCAW[1] = 1;

        copyParams.blockLen = numPoints * sizeof(DTYPE_VALUE);
    }

    __aicore__ inline void AllocEventID()
    {
        eventIdMte2ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdMte3ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());
        eventIdVToMte2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMte3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdSToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::S_V>());
        eventIdVToS = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_S>());
        eventIdMte2ToS = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_S>());
        eventIdSToMte2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::S_MTE2>());
        eventIdSToMte3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::S_MTE3>());
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
                                          numLevels * batchSize * numQueries * numKeysAlign);
    }

    __aicore__ inline void InitBuffer()
    {
        pipe->InitBuffer(spatialShapesUb, 2 * numLevelsAlign * sizeof(DTYPE_SPATIAL_SHAPES));
        pipe->InitBuffer(levelStartIdxUb, numLevelsAlign * sizeof(DTYPE_SPATIAL_SHAPES));

        pipe->InitBuffer(curGradAttnWeightsUb, numQueriesPerBundle * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(keyIdxsUb, numQueriesPerBundle * numPointsAlign * sizeof(DTYPE_VALUE));
        
        pipe->InitBuffer(curGradOutputUb, numQueriesPerBundle * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(curAttnWeightUb, numQueriesPerBundle * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(curTmp4GradAttnWeightsUb, numQueriesPerBundle * numKeysAlign * sizeof(DTYPE_VALUE));
        
        pipe->InitBuffer(curAttnWeightBrcbUb, numPointsAlign * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmp4GradValueUb, numPointsAlign * embedDims * sizeof(DTYPE_VALUE));
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
    }

    __aicore__ inline void Compute();

    __aicore__ inline void ComputeCurLevel();

    __aicore__ inline void ComputeCurBatch();

    __aicore__ inline void ComputeCurLoopFront();

    __aicore__ inline void ComputeCurLoopMiddle();

    __aicore__ inline void ComputeCurLoopBack();

private:
    TPipe *pipe;

    GlobalTensor<DTYPE_VALUE> valueGm, samplingLocationsGm, attnWeightsGm, gradOutputGm;
    GlobalTensor<DTYPE_VALUE> gradValueGm, gradAttnWeightsGm, tmp4GradAttnWeightsGM;
    GlobalTensor<DTYPE_SPATIAL_SHAPES> spatialShapesGm, levelStartIndexGm;

    TBuf<TPosition::VECCALC> spatialShapesUb, levelStartIdxUb;
    TBuf<TPosition::VECCALC> curGradAttnWeightsUb, keyIdxsUb;
    TBuf<TPosition::VECCALC> curGradOutputUb, curAttnWeightUb, curTmp4GradAttnWeightsUb;
    TBuf<TPosition::VECCALC> curAttnWeightBrcbUb, tmp4GradValueUb;

    LocalTensor<DTYPE_SPATIAL_SHAPES> spatialShapesLocal, levelStartIdxLocal;
    LocalTensor<DTYPE_VALUE> curGradAttnWeightsLocal, keyIdxsLocal;
    LocalTensor<DTYPE_VALUE> curGradOutputLocal, curAttnWeightLocal, curTmp4GradAttnWeightsLocal;
    LocalTensor<DTYPE_VALUE> curAttnWeightBrcbLocal, tmp4GradValueLocal;

    DTYPE_SPATIAL_SHAPES h, w, levelStartIdx;
    DTYPE_VALUE curTmp4GradAttnWeight;

    uint32_t coreNum, curBlockIdx;
    uint32_t dataAlign, blockBytes;
    uint32_t batchSize, embedDims, numKeys, numQueries, numPoints, numLevels;
    uint32_t numLevelsAlign, numKeysAlign, numPointsAlign;
    uint32_t batchIdx, keyIdx, queryIdx, headIdx, levelIdx, pointIdx;
    uint32_t dstShape[2], srcShapeCGO[2], srcShapeCAW[2];
    uint32_t numQueriesPerCore, numQueriesCurCore, numQueriesPerBundle, numQueriesLastBundle, numQueriesCurBundle;
    uint32_t loopIdx, numQueryLoops;
    uint32_t startOffset, endOffset, queryIdxStart, queryIdxEnd, queryOffsetBundle;
    uint32_t samplingLocationXIdx, samplingLocationYIdx;
    uint32_t gradOutputIdx, curGradOutputIdx, valueIdx, gradValueIdx, tmp4GradValueIdx;
    uint32_t attnWeightsIdx, curAttnWeightsIdx, gradAttnWeightsIdx, tmp4GradAttnWeightsIdx;

    DataCopyParams copyParams {1, 0, 0, 0};

    event_t eventIdVToMte2, eventIdVToMte3, eventIdMte2ToV, eventIdMte3ToV;
    event_t eventIdSToV, eventIdVToS, eventIdMte2ToS, eventIdSToMte2;
    event_t eventIdSToMte3, eventIdMte3ToMte2;
};

__aicore__ inline void GeometricKernelAttnGrad::ComputeCurLevel()
{
    levelStartIdx = levelStartIdxLocal.GetValue(levelIdx);
    h = spatialShapesLocal.GetValue(levelIdx * 2);
    w = spatialShapesLocal.GetValue(levelIdx * 2 + 1);
}

__aicore__ inline void GeometricKernelAttnGrad::ComputeCurBatch()
{
    gradOutputIdx = batchIdx * numQueries * embedDims + startOffset * embedDims;
    mm_.SetTensorA(gradOutputGm[gradOutputIdx]);

    valueIdx = batchIdx * numKeys * embedDims;
    mm_.SetTensorB(valueGm[valueIdx], true);

    tmp4GradAttnWeightsIdx = levelIdx * batchSize * numQueries * numKeysAlign + \
                                batchIdx * numQueries * numKeysAlign + \
                                startOffset * numKeysAlign;
    mm_.template IterateAll<true>(tmp4GradAttnWeightsGM[tmp4GradAttnWeightsIdx]);
    mm_.End();
}

__aicore__ inline void GeometricKernelAttnGrad::ComputeCurLoopFront()
{
    numQueriesCurBundle = (loopIdx == numQueryLoops - 1) ? numQueriesLastBundle : numQueriesPerBundle;
    copyParams.blockCount = numQueriesCurBundle;

    queryIdxStart = startOffset + loopIdx * numQueriesPerBundle;

    samplingLocationXIdx = levelIdx * batchSize * 2 * numQueries * numPoints + \
                        batchIdx * 2 * numQueries * numPoints + \
                        queryIdxStart * numPoints;
    DataCopyPad(curGradAttnWeightsLocal, samplingLocationsGm[samplingLocationXIdx], copyParams, {});

    samplingLocationYIdx = samplingLocationXIdx + numQueries * numPoints;
    DataCopyPad(keyIdxsLocal, samplingLocationsGm[samplingLocationYIdx], copyParams, {});

    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    gradOutputIdx = batchIdx * numQueries * embedDims + queryIdxStart * embedDims;
    attnWeightsIdx = levelIdx * batchSize * numQueries * numPoints + \
                    batchIdx * numQueries * numPoints + \
                    queryIdxStart * numPoints;
    tmp4GradAttnWeightsIdx = levelIdx * batchSize * numQueries * numKeysAlign + \
                                batchIdx * numQueries * numKeysAlign + \
                                queryIdxStart * numKeysAlign;

    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    ClampMin(curGradAttnWeightsLocal, curGradAttnWeightsLocal, static_cast<DTYPE_VALUE>(0), numQueriesCurBundle * numPointsAlign);
    ClampMin(keyIdxsLocal, keyIdxsLocal, static_cast<DTYPE_VALUE>(0), numQueriesCurBundle * numPointsAlign);
    ClampMax(curGradAttnWeightsLocal, curGradAttnWeightsLocal, static_cast<DTYPE_VALUE>(w - 1), numQueriesCurBundle * numPointsAlign);
    ClampMax(keyIdxsLocal, keyIdxsLocal, static_cast<DTYPE_VALUE>(h - 1), numQueriesCurBundle * numPointsAlign);
    Muls(keyIdxsLocal, keyIdxsLocal, static_cast<DTYPE_VALUE>(w), numQueriesCurBundle * numPointsAlign);
    Add(keyIdxsLocal, keyIdxsLocal, curGradAttnWeightsLocal, numQueriesCurBundle * numPointsAlign);
    Adds(keyIdxsLocal, keyIdxsLocal, static_cast<DTYPE_VALUE>(levelStartIdx), numQueriesCurBundle * numPointsAlign);
    Duplicate(curGradAttnWeightsLocal, (DTYPE_VALUE)0, numQueriesCurBundle * numPointsAlign);
    SetFlag<HardEvent::V_S>(eventIdVToS);

    DataCopy(curGradOutputLocal, gradOutputGm[gradOutputIdx], numQueriesCurBundle * embedDims);
    DataCopyPad(curAttnWeightLocal, attnWeightsGm[attnWeightsIdx], copyParams, {});
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    DataCopy(curTmp4GradAttnWeightsLocal, tmp4GradAttnWeightsGM[tmp4GradAttnWeightsIdx], numQueriesCurBundle * numKeysAlign);
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

    queryIdxEnd = queryIdxStart + numQueriesCurBundle;

    WaitFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
}

__aicore__ inline void GeometricKernelAttnGrad::ComputeCurLoopMiddle()
{
#pragma bisheng auto_sync parallel
    for (queryIdx = queryIdxStart; queryIdx < queryIdxEnd; queryIdx++) {
        queryOffsetBundle = queryIdx - queryIdxStart;

        curGradOutputIdx = queryOffsetBundle * embedDims;
        BroadCast<DTYPE_VALUE, 2, 0>(tmp4GradValueLocal, curGradOutputLocal[curGradOutputIdx], dstShape, srcShapeCGO);

        curAttnWeightsIdx = queryOffsetBundle * numPointsAlign;
        BroadCast<DTYPE_VALUE, 2, 1>(curAttnWeightBrcbLocal, curAttnWeightLocal[curAttnWeightsIdx], dstShape, srcShapeCAW);
        
        PipeBarrier<PIPE_V>();
        Mul(tmp4GradValueLocal, tmp4GradValueLocal, curAttnWeightBrcbLocal, numPointsAlign * embedDims);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        for (pointIdx = 0; pointIdx < numPoints; pointIdx++) {
            gradAttnWeightsIdx = queryOffsetBundle * numPointsAlign + pointIdx;
            keyIdx = static_cast<DTYPE_SPATIAL_SHAPES>(keyIdxsLocal.GetValue(gradAttnWeightsIdx));

            gradValueIdx = batchIdx * numKeys * embedDims + keyIdx * embedDims;
            tmp4GradValueIdx = pointIdx * embedDims;
            DataCopy(gradValueGm[gradValueIdx], tmp4GradValueLocal[tmp4GradValueIdx], embedDims);

            tmp4GradAttnWeightsIdx = queryOffsetBundle * numKeysAlign + keyIdx;
            curTmp4GradAttnWeight = curTmp4GradAttnWeightsLocal.GetValue(tmp4GradAttnWeightsIdx);
            curGradAttnWeightsLocal.SetValue(gradAttnWeightsIdx, curTmp4GradAttnWeight);
        }
        
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }
}

__aicore__ inline void GeometricKernelAttnGrad::ComputeCurLoopBack()
{
    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    DataCopyPad(gradAttnWeightsGm[attnWeightsIdx], curGradAttnWeightsLocal, copyParams);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
}

__aicore__ inline void GeometricKernelAttnGrad::Compute()
{
    for (levelIdx = 0; levelIdx < numLevels; levelIdx++) {
        ComputeCurLevel();

        for (batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            ComputeCurBatch();

            for (loopIdx = 0; loopIdx < numQueryLoops; loopIdx++) {
                ComputeCurLoopFront();

                SetAtomicAdd<DTYPE_VALUE>();
                ComputeCurLoopMiddle();
                SetAtomicNone();

                ComputeCurLoopBack();
            }
        }
    }
}

__aicore__ inline void GeometricKernelAttnGrad::Process()
{
    if (startOffset < endOffset) {
        DataCopy(spatialShapesLocal, spatialShapesGm, 2 * numLevelsAlign);
        DataCopy(levelStartIdxLocal, levelStartIndexGm, numLevelsAlign);

        Compute();
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
    GeometricKernelAttnGrad op;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm_, &(tiling_datas.mmTilingData));

    op.Init(value_gm, spatial_shapes_gm, level_start_index_gm, sampling_locations_gm, attn_weights_gm, grad_output_gm,
        grad_value_gm, grad_attn_weights_gm, usrWorkspace, &tiling_datas, &pipe);
    op.Process();
    op.ReleaseEventID();
}
