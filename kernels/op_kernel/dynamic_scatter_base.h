/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 */
#ifndef _DYNAMIC_SCATTER_BASE_H_
#define _DYNAMIC_SCATTER_BASE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace DynamicScatter {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t RESERVED_NUM = 1000;

template<typename T>
class DynamicScatterBase {
public:
    __aicore__ inline DynamicScatterBase() {}
    __aicore__ inline void BaseInit(GM_ADDR point_feats, GM_ADDR prefix_sum_point_per_voxel, GM_ADDR argsort_coor,
        GM_ADDR voxel_feats, DynamicScatterTilingData* tilingData, TPipe* in_pipe)
    {
        pipe = in_pipe;

        TilingDataInit(tilingData);
        MemberDataInit();
        CopyParamasInit();
        GlobalBufInit(point_feats, prefix_sum_point_per_voxel, argsort_coor, voxel_feats);
        BufInit();

        eventIdSToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::S_V>());
        eventIdSToMTE2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::S_MTE2>());
        eventIdSToMTE3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::S_MTE3>());
        eventIdVToMTE3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdMTE2ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdMTE2ToS = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_S>());
        eventIdMTE3ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());
        eventIdMTE3ToS = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_S>());
        eventIdMTE2ToMTE3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_MTE3>());
        eventIdMTE3ToMTE2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_MTE2>());
    }

    __aicore__ inline void TilingDataInit(DynamicScatterTilingData* tilingData)
    {
        usedCoreNum = tilingData->usedCoreNum;
        totalPointNum = tilingData->totalPointNum;
        totalVoxelNum = tilingData->totalVoxelNum;
        featsDim = tilingData->featsDim;
        pointFeatsNum = tilingData->pointFeatsNum;
        voxelNumPerCore = tilingData->voxelNumPerCore;
        voxelNumLastCore = tilingData->voxelNumLastCore;
        voxelFeatsNumPerCore = tilingData->voxelFeatsNumPerCore;
        voxelFeatsNumLastCore = tilingData->voxelFeatsNumLastCore;
        alignedNum = tilingData->alignedNum;
        featsDimAligned = tilingData->featsDimAligned;
        availablePointNum = tilingData->availablePointNum;
        blockLen = tilingData->blockLen;
        blockLenPad = tilingData->blockLenPad;
        isFeatsAligned = tilingData->isFeatsAligned;
    }

    __aicore__ inline void MemberDataInit()
    {
        if (GetBlockIdx() < usedCoreNum - 1) {
            voxelFeatNum = voxelFeatsNumPerCore;
            voxelNum = voxelNumPerCore;
            voxelOffset = voxelNum * GetBlockIdx();
        } else {
            voxelFeatNum = voxelFeatsNumLastCore;
            voxelNum = voxelNumLastCore;
            voxelOffset = voxelNumPerCore * (usedCoreNum - 1);
        }
        voxelfeatsOffset = voxelOffset * featsDim;
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
        GM_ADDR point_feats, GM_ADDR prefix_sum_point_per_voxel, GM_ADDR argsort_coor, GM_ADDR voxel_feats)
    {
        pointFeatsGm.SetGlobalBuffer((__gm__ T*)point_feats, pointFeatsNum);
        prefixSumGm.SetGlobalBuffer((__gm__ int32_t*)prefix_sum_point_per_voxel + voxelOffset, voxelNum);
        argsortCoorGm.SetGlobalBuffer((__gm__ int32_t*)argsort_coor, totalPointNum);
        voxelFeatsGm.SetGlobalBuffer((__gm__ T*)voxel_feats + voxelfeatsOffset, voxelFeatNum);
    }

    __aicore__ inline void BufInit()
    {
        pipe->InitBuffer(prefixSumBuf, alignedNum * sizeof(int32_t));
        this->pipe->InitBuffer(this->argsortCoorBuf, RESERVED_NUM * sizeof(int32_t));
        this->pipe->InitBuffer(this->pointFeatsBuf, availablePointNum * this->featsDimAligned * sizeof(T));
    }

    __aicore__ inline void GetPointNum(uint32_t voxelIdx, const LocalTensor<int32_t>& prefixSumLocal)
    {
        if (GetBlockIdx() == usedCoreNum - 1 && voxelIdx == voxelNum - 1) {
            DataCopy(prefixSumLocal, prefixSumGm[voxelIdx], copyprefixSumParams);
            SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
            startPoint = prefixSumLocal.GetValue(0);
            pointNum = totalPointNum - startPoint;
        } else {
            DataCopy(prefixSumLocal, prefixSumGm[voxelIdx], copyprefixSumParams);
            SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
            startPoint = prefixSumLocal.GetValue(0);
            pointNum = prefixSumLocal.GetValue(1) - startPoint;
        }
    }

    __aicore__ inline void CopyFeatsOut(
        uint32_t voxelIdx, const LocalTensor<T>& pointFeatsLocal, bool atomicMax, uint32_t offset = 0)
    {
        if (atomicMax) {
            SetAtomicMax<T>();
        } else {
            SetAtomicAdd<T>();
        }
        if (isFeatsAligned) {
            DataCopy(voxelFeatsGm[voxelIdx * featsDim], pointFeatsLocal[offset], copyFeatParams);
        } else {
            DataCopyPad(voxelFeatsGm[voxelIdx * featsDim], pointFeatsLocal[offset], copyOutPadParams);
        }
        SetAtomicNone();
    }

    __aicore__ inline void ReleaseEvent()
    {
        GetTPipePtr()->ReleaseEventID<HardEvent::S_V>(eventIdSToV);
        GetTPipePtr()->ReleaseEventID<HardEvent::S_MTE2>(eventIdSToMTE2);
        GetTPipePtr()->ReleaseEventID<HardEvent::S_MTE3>(eventIdSToMTE3);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMTE3);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMTE2ToV);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_S>(eventIdMTE2ToS);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMTE3ToV);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_S>(eventIdMTE3ToS);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
    }

protected:
    TPipe* pipe;

    GlobalTensor<T> pointFeatsGm, voxelFeatsGm;
    GlobalTensor<int32_t> prefixSumGm, argsortCoorGm;

    TBuf<TPosition::VECCALC> pointFeatsBuf, prefixSumBuf, argsortCoorBuf;

    uint64_t totalPointNum, totalVoxelNum;
    uint64_t featsDim, pointFeatsNum, alignedNum, featsDimAligned, availablePointNum;
    uint32_t usedCoreNum, voxelNumPerCore, voxelNumLastCore, voxelFeatsNumPerCore, voxelFeatsNumLastCore;
    uint32_t blockLen, blockLenPad;
    uint64_t voxelFeatNum, voxelNum, voxelOffset, voxelfeatsOffset;
    uint32_t startPoint, pointNum, pointIdx, alignedPointNum;
    bool isFeatsAligned;

    DataCopyParams copyFeatParams, copyprefixSumParams, copyArgsortCoorParams;
    DataCopyExtParams copyOutPadParams;

    event_t eventIdSToV, eventIdSToMTE2, eventIdSToMTE3, eventIdVToMTE3, eventIdMTE2ToV, eventIdMTE2ToS, eventIdMTE3ToV;
    event_t eventIdMTE3ToS, eventIdMTE2ToMTE3, eventIdMTE3ToMTE2;
};
} // namespace DynamicScatter
#endif // _DYNAMIC_SCATTER_BASE_H_