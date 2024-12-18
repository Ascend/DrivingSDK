/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef _DYNAMIC_SCATTER_GRAD_BASE_H_
#define _DYNAMIC_SCATTER_GRAD_BASE_H_

#include <cmath>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace DynamicScatterGrad {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t RESERVED_NUM = 1000;

template<typename T>
class DynamicScatterGradBase {
public:
    __aicore__ inline DynamicScatterGradBase() {}
    __aicore__ inline void BaseInit(GM_ADDR grad_voxel_feats, GM_ADDR prefix_sum_point_per_voxel, GM_ADDR argsort_coor,
        GM_ADDR grad_point_feats, DynamicScatterGradTilingData* tilingData, TPipe* in_pipe)
    {
        pipe = in_pipe;

        TilingDataInit(tilingData);
        MemberDataInit();
        CopyParamasInit();
        GlobalBufInit(grad_voxel_feats, prefix_sum_point_per_voxel, argsort_coor, grad_point_feats);
        BufInit();

        eventIdSToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::S_V>());
        eventIdMte2ToS = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_S>());
        eventIdSToMTE2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::S_MTE2>());
        eventIdSToMTE3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::S_MTE3>());
        eventIdMTE3ToMTE2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_MTE2>());
    }

    __aicore__ inline void TilingDataInit(DynamicScatterGradTilingData* tilingData)
    {
        alignedNum = tilingData->alignedNum;
        totalPointNum = tilingData->totalPointNum;
        totalVoxelNum = tilingData->totalVoxelNum;
        pointGradNum = tilingData->pointGradNum;
        voxelNumPerCore = tilingData->voxelNumPerCore;
        voxelNumLastCore = tilingData->voxelNumLastCore;
        eleNumPerCore = tilingData->eleNumPerCore;
        eleNumLastCore = tilingData->eleNumLastCore;
        featDim = tilingData->featDim;
        featDimAligned = tilingData->featDimAligned;
        blockLen = tilingData->blockLen;
        blockLenPad = tilingData->blockLenPad;
        isFeatsAligned = tilingData->isFeatsAligned;
        usedCoreNum = tilingData->usedCoreNum;
    }

    __aicore__ inline void MemberDataInit()
    {
        if (GetBlockIdx() < usedCoreNum - 1) {
            voxelFeatNum = eleNumPerCore;
            voxelNum = voxelNumPerCore;
            voxelOffset = voxelNum * GetBlockIdx();
        } else {
            voxelFeatNum = eleNumLastCore;
            voxelNum = voxelNumLastCore;
            voxelOffset = voxelNumPerCore * (usedCoreNum - 1);
        }
        voxelfeatsOffset = voxelOffset * featDim;
    }

    __aicore__ inline void CopyParamasInit()
    {
        copyFeatParams.blockCount = 1;
        copyFeatParams.blockLen = blockLen;
        copyFeatParams.srcStride = 0;
        copyFeatParams.dstStride = 0;
        if (!isFeatsAligned) {
            copyOutPadParams.blockCount = 1;
            copyOutPadParams.blockLen = blockLenPad;
            copyOutPadParams.srcStride = 0;
            copyOutPadParams.dstStride = 0;
            copyOutPadParams.rsv = 0;
        }
        copyprefixSumParams.blockCount = 1;
        copyprefixSumParams.blockLen = 1;
        copyprefixSumParams.srcStride = 0;
        copyprefixSumParams.dstStride = 0;
        copyArgsortCoorParams.blockCount = 1;
        copyArgsortCoorParams.srcStride = 0;
        copyArgsortCoorParams.dstStride = 0;
    }

    __aicore__ inline void GlobalBufInit(
        GM_ADDR grad_voxel_feats, GM_ADDR prefix_sum_point_per_voxel, GM_ADDR argsort_coor, GM_ADDR grad_point_feats)
    {
        voxelGradGm.SetGlobalBuffer((__gm__ T*)grad_voxel_feats + voxelfeatsOffset, voxelFeatNum);
        prefixSumGm.SetGlobalBuffer((__gm__ int32_t*)prefix_sum_point_per_voxel + voxelOffset, totalVoxelNum);
        argsortCoorGm.SetGlobalBuffer((__gm__ int32_t*)argsort_coor, totalVoxelNum - 1);
        pointGradGm.SetGlobalBuffer((__gm__ T*)grad_point_feats, pointGradNum);
    }

    __aicore__ inline void BufInit()
    {
        pipe->InitBuffer(voxelGradBuf, featDimAligned * sizeof(T));
        pipe->InitBuffer(prefixSumBuf, alignedNum * sizeof(int32_t));
        pipe->InitBuffer(argsortCoorBuf, RESERVED_NUM * sizeof(int32_t));
    }

    __aicore__ inline void GetPointNum(uint32_t voxel_idx, LocalTensor<int32_t> prefixSumLocal)
    {
        if (GetBlockIdx() == usedCoreNum - 1 && voxel_idx == voxelNum - 1) {
            DataCopy(prefixSumLocal, prefixSumGm[voxel_idx], copyprefixSumParams);
            SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
            startPoint = prefixSumLocal.GetValue(0);
            pointNum = totalPointNum - startPoint;
        } else {
            DataCopy(prefixSumLocal, prefixSumGm[voxel_idx], copyprefixSumParams);
            SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
            startPoint = prefixSumLocal.GetValue(0);
            pointNum = prefixSumLocal.GetValue(1) - startPoint;
        }
    }

    __aicore__ inline void CopyPointGradOut(LocalTensor<T> voxelGradLocal)
    {
        SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
        if (isFeatsAligned) {
            DataCopy(pointGradGm[point_idx * featDim], voxelGradLocal, copyFeatParams);
        } else {
            DataCopyPad(pointGradGm[point_idx * featDim], voxelGradLocal, copyOutPadParams);
        }
    }

    __aicore__ inline void ReleaseEvent()
    {
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(eventIdMte2ToS);
        GetTPipePtr()->ReleaseEventID<HardEvent::S_MTE2>(eventIdSToMTE2);
        GetTPipePtr()->ReleaseEventID<HardEvent::S_MTE3>(eventIdSToMTE3);
        GetTPipePtr()->ReleaseEventID<HardEvent::S_V>(eventIdSToV);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
    }

protected:
    TPipe* pipe;

    GlobalTensor<T> voxelGradGm, pointGradGm;
    GlobalTensor<int32_t> prefixSumGm, argsortCoorGm;

    TBuf<TPosition::VECCALC> voxelGradBuf, prefixSumBuf, argsortCoorBuf;

    uint32_t voxelNumPerCore, voxelNumLastCore, eleNumPerCore, eleNumLastCore, usedCoreNum;
    uint64_t totalPointNum, totalVoxelNum, alignedNum, blockLen, blockLenPad;
    uint64_t pointGradNum, featDim, featDimAligned, voxelFeatNum, voxelNum, voxelOffset, voxelfeatsOffset;
    uint32_t point_idx, pointNum, startPoint;
    bool isFeatsAligned;

    DataCopyParams copyFeatParams, copyprefixSumParams, copyArgsortCoorParams;
    DataCopyExtParams copyOutPadParams;

    event_t eventIdMte2ToS, eventIdSToMTE2, eventIdSToV, eventIdMTE3ToMTE2, eventIdSToMTE3;
};
} // namespace DynamicScatterGrad
#endif // _DYNAMIC_SCATTER_GRAD_BASE_H_