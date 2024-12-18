/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef _DYNAMIC_SCATTER_GRAD_MAX_H_
#define _DYNAMIC_SCATTER_GRAD_MAX_H_

#include "dynamic_scatter_grad_base.h"

namespace DynamicScatterGrad {
using namespace AscendC;

template<typename T>
class DynamicScatterGradMax : public DynamicScatterGradBase<T> {
public:
    __aicore__ inline DynamicScatterGradMax() {}
    __aicore__ inline void Init(GM_ADDR grad_voxel_feats, GM_ADDR prefix_sum_point_per_voxel, GM_ADDR argsort_coor,
        GM_ADDR compare_mask, GM_ADDR grad_point_feats, DynamicScatterGradTilingData* tilingData, TPipe* in_pipe)
    {
        this->BaseInit(
            grad_voxel_feats, prefix_sum_point_per_voxel, argsort_coor, grad_point_feats, tilingData, in_pipe);

        maskNum = tilingData->maskNum;
        maskDim = tilingData->maskDim;
        maskDimAligned = tilingData->maskDimAligned;
        blockLenMask = tilingData->blockLenMask;

        copyMaskParams.blockCount = 1;
        copyMaskParams.blockLen = blockLenMask;
        copyMaskParams.srcStride = 0;
        copyMaskParams.dstStride = 0;

        compareMaskGm.SetGlobalBuffer((__gm__ uint8_t*)compare_mask, maskNum);

        this->pipe->InitBuffer(compareMaskBuf, maskDimAligned * sizeof(uint8_t));
        this->pipe->InitBuffer(zeroBuf, this->featDimAligned * sizeof(T));
        this->pipe->InitBuffer(pointGradBuf, this->featDimAligned * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        Compute();
        this->ReleaseEvent();
    }

private:
    __aicore__ inline void Compute()
    {
        LocalTensor<T> zeroLocal = zeroBuf.template Get<T>();
        LocalTensor<T> pointGradLocal = pointGradBuf.template Get<T>();
        LocalTensor<T> voxelGradLocal = this->voxelGradBuf.template Get<T>();
        LocalTensor<uint8_t> compareMaskLocal = compareMaskBuf.template Get<uint8_t>();
        LocalTensor<int32_t> prefixSumLocal = this->prefixSumBuf.template Get<int32_t>();
        LocalTensor<int32_t> argsortCoorLocal = this->argsortCoorBuf.template Get<int32_t>();
        Duplicate(zeroLocal, static_cast<T>(0), this->featDimAligned);

        for (uint32_t voxel_idx = 0; voxel_idx < this->voxelNum; voxel_idx++) {
            DataCopy(voxelGradLocal, this->voxelGradGm[voxel_idx * this->featDim], this->copyFeatParams);
            this->GetPointNum(voxel_idx, prefixSumLocal);
            uint32_t aligned_point_num = AlignUp(this->pointNum, this->alignedNum);
            this->copyArgsortCoorParams.blockLen = aligned_point_num / this->alignedNum;

            SetFlag<HardEvent::S_MTE2>(this->eventIdSToMTE2);
            WaitFlag<HardEvent::S_MTE2>(this->eventIdSToMTE2);
            DataCopy(argsortCoorLocal, this->argsortCoorGm[this->startPoint], this->copyArgsortCoorParams);

            SetFlag<HardEvent::MTE2_S>(this->eventIdMte2ToS);
            WaitFlag<HardEvent::MTE2_S>(this->eventIdMte2ToS);
            for (uint32_t idx = 0; idx < this->pointNum; idx++) {
                this->point_idx = argsortCoorLocal.GetValue(idx);
                SetFlag<HardEvent::S_MTE2>(this->eventIdSToMTE2);
                WaitFlag<HardEvent::S_MTE2>(this->eventIdSToMTE2);
                DataCopy(compareMaskLocal, compareMaskGm[this->point_idx * maskDim], copyMaskParams);
                PipeBarrier<PIPE_ALL>();
                Select<T, uint8_t>(pointGradLocal, compareMaskLocal, voxelGradLocal, zeroLocal,
                    SELMODE::VSEL_TENSOR_TENSOR_MODE, this->featDim);
                PipeBarrier<PIPE_ALL>();
                this->CopyPointGradOut(pointGradLocal);
            }
            SetFlag<HardEvent::MTE3_MTE2>(this->eventIdMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(this->eventIdMTE3ToMTE2);
        }
    }

private:
    DataCopyParams copyMaskParams;
    GlobalTensor<uint8_t> compareMaskGm;
    uint64_t maskNum, maskDim, maskDimAligned, blockLenMask;
    TBuf<TPosition::VECCALC> compareMaskBuf, zeroBuf, pointGradBuf;
};
} // namespace DynamicScatterGrad
#endif // _DYNAMIC_SCATTER_GRAD_MAX_H_