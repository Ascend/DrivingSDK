/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 */
#ifndef _DYNAMIC_SCATTER_MAX_H_
#define _DYNAMIC_SCATTER_MAX_H_

#include <cmath>

#include "dynamic_scatter_base.h"

namespace DynamicScatter {
using namespace AscendC;

template<typename T>
class DynamicScatterMax : public DynamicScatterBase<T> {
public:
    __aicore__ inline DynamicScatterMax() {}
    __aicore__ inline void Init(GM_ADDR point_feats, GM_ADDR prefix_sum_point_per_voxel, GM_ADDR argsort_coor,
        GM_ADDR voxel_feats, GM_ADDR compare_mask, DynamicScatterTilingData* tilingData, TPipe* in_pipe)
    {
        this->BaseInit(point_feats, prefix_sum_point_per_voxel, argsort_coor, voxel_feats, tilingData, in_pipe);

        maskNum = tilingData->maskNum;
        maskDim = tilingData->maskDim;
        maskDimAligned = tilingData->maskDimAligned;
        maskDimAlignedB16 = tilingData->maskDimAlignedB16;
        blockLenMask = tilingData->blockLenMask;
        repeatTimes = tilingData->repeatTimes;
        curBlockIdx = GetBlockIdx();

        compareMaskGm.SetGlobalBuffer((__gm__ uint8_t*)compare_mask, maskNum);

        this->pipe->InitBuffer(voxelFeatsBuf, this->featsDimAligned * sizeof(T));
        this->pipe->InitBuffer(recordMaskBuf, maskDimAlignedB16 * sizeof(uint16_t));
        this->pipe->InitBuffer(bitMaskBuf, maskDimAligned * sizeof(uint8_t));
        this->pipe->InitBuffer(bitMaskTmpBuf, maskDimAlignedB16 * sizeof(uint16_t));

        compareParams.dstBlkStride = 1;
        compareParams.src0BlkStride = 1;
        compareParams.src1BlkStride = 1;
        compareParams.dstRepStride = 8;
        compareParams.src0RepStride = 8;
        compareParams.src1RepStride = 8;

        copyMaskOutParams.blockCount = 1;
        copyMaskOutParams.blockLen = blockLenMask;
        copyMaskOutParams.srcStride = 1;
        copyMaskOutParams.dstStride = 1;

        if (curBlockIdx == 0) {
            InitOutput<T>(this->voxelFeatsGm, this->totalVoxelNum * this->featsDim, static_cast<T>(-3.4e38));
        }
        SyncAll();
    }

    __aicore__ inline void Process()
    {
        Compute();
        this->ReleaseEvent();
    }

private:
    __aicore__ inline void Compute()
    {
        LocalTensor<int32_t> prefixSumLocal = this->prefixSumBuf.template Get<int32_t>();
        LocalTensor<T> voxelFeatsLocal = this->voxelFeatsBuf.template Get<T>();
        LocalTensor<uint16_t> recordMaskLocal = recordMaskBuf.template Get<uint16_t>();
        LocalTensor<uint8_t> bitMaskLocal = bitMaskBuf.template Get<uint8_t>();
        LocalTensor<uint16_t> bitMaskLocalB16 = bitMaskLocal.ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> bitMaskTmpLocal = bitMaskTmpBuf.template Get<uint16_t>();
        LocalTensor<int32_t> argsortCoorLocal = this->argsortCoorBuf.template Get<int32_t>();
        LocalTensor<T> pointFeatsLocal = this->pointFeatsBuf.template Get<T>();

        for (uint32_t voxelIdx = 0; voxelIdx < this->voxelNum; voxelIdx++) {
            this->GetPointNum(voxelIdx, prefixSumLocal);
            this->alignedPointNum = AlignUp(this->pointNum, this->alignedNum);
            this->copyArgsortCoorParams.blockLen = this->alignedPointNum / this->alignedNum;

            SetFlag<HardEvent::S_MTE2>(this->eventIdSToMTE2);
            WaitFlag<HardEvent::S_MTE2>(this->eventIdSToMTE2);
            DataCopy(argsortCoorLocal, this->argsortCoorGm[this->startPoint], this->copyArgsortCoorParams);

            SetFlag<HardEvent::MTE2_S>(this->eventIdMTE2ToS);
            WaitFlag<HardEvent::MTE2_S>(this->eventIdMTE2ToS);
            for (uint32_t idx = 0; idx < this->pointNum; idx++) {
                this->pointIdx = argsortCoorLocal.GetValue(idx);
                SetFlag<HardEvent::S_MTE2>(this->eventIdSToMTE2);
                WaitFlag<HardEvent::S_MTE2>(this->eventIdSToMTE2);
                DataCopy(pointFeatsLocal[idx * this->featsDimAligned],
                    this->pointFeatsGm[this->pointIdx * this->featsDim], this->copyFeatParams);
                SetFlag<HardEvent::MTE2_MTE3>(this->eventIdMTE2ToMTE3);
                WaitFlag<HardEvent::MTE2_MTE3>(this->eventIdMTE2ToMTE3);
                this->CopyFeatsOut(voxelIdx, pointFeatsLocal, true, idx * this->featsDimAligned);
                SetFlag<HardEvent::MTE3_MTE2>(this->eventIdMTE3ToMTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(this->eventIdMTE3ToMTE2);
            }
            Duplicate(recordMaskLocal, static_cast<uint16_t>(0), this->maskDimAlignedB16);
            DataCopy(voxelFeatsLocal, this->voxelFeatsGm[voxelIdx * this->featsDim], this->copyFeatParams);

            SetFlag<HardEvent::MTE2_V>(this->eventIdMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(this->eventIdMTE2ToV);
            for (uint32_t idx = 0; idx < this->pointNum; idx++) {
                this->pointIdx = argsortCoorLocal.GetValue(idx);
                Compare(bitMaskLocal, voxelFeatsLocal, pointFeatsLocal[idx * this->featsDimAligned], CMPMODE::EQ, mask,
                    repeatTimes, compareParams);
                pipe_barrier(PIPE_ALL);

                Not(bitMaskTmpLocal, recordMaskLocal, maskDimAlignedB16);
                And(bitMaskLocalB16, bitMaskLocalB16, bitMaskTmpLocal, maskDimAlignedB16);
                Or(recordMaskLocal, bitMaskLocalB16, recordMaskLocal, maskDimAlignedB16);
                SetFlag<HardEvent::V_MTE3>(this->eventIdVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(this->eventIdVToMTE3);

                SetFlag<HardEvent::S_MTE3>(this->eventIdSToMTE3);
                WaitFlag<HardEvent::S_MTE3>(this->eventIdSToMTE3);
                DataCopyPad(compareMaskGm[this->pointIdx * maskDim], bitMaskLocal, copyMaskOutParams);
                SetFlag<HardEvent::MTE3_S>(this->eventIdMTE3ToS);
                WaitFlag<HardEvent::MTE3_S>(this->eventIdMTE3ToS);
            }
            SetFlag<HardEvent::MTE3_MTE2>(this->eventIdMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(this->eventIdMTE3ToMTE2);
        }
    }

private:
    uint64_t maskNum, maskDim, maskDimAligned, maskDimAlignedB16, blockLenMask, repeatTimes, curBlockIdx, maskOffset;
    uint64_t mask = 64;
    GlobalTensor<uint8_t> compareMaskGm;
    TBuf<TPosition::VECCALC> voxelFeatsBuf, bitMaskBuf, bitMaskTmpBuf, recordMaskBuf;
    BinaryRepeatParams compareParams;
    DataCopyExtParams copyMaskOutParams;
};
} // namespace DynamicScatter
#endif // _DYNAMIC_SCATTER_MAX_H_
