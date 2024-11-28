/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 */
#ifndef _DYNAMIC_SCATTER_SUM_H_
#define _DYNAMIC_SCATTER_SUM_H_

#include "dynamic_scatter_base.h"

namespace DynamicScatter {
using namespace AscendC;

template<typename T>
class DynamicScatterSum : public DynamicScatterBase<T> {
public:
    __aicore__ inline DynamicScatterSum() {}
    __aicore__ inline void Init(GM_ADDR point_feats, GM_ADDR prefix_sum_point_per_voxel, GM_ADDR argsort_coor,
        GM_ADDR voxel_feats, DynamicScatterTilingData* tilingData, TPipe* in_pipe)
    {
        this->BaseInit(point_feats, prefix_sum_point_per_voxel, argsort_coor, voxel_feats, tilingData, in_pipe);
    }

    __aicore__ inline void Process()
    {
        Compute();
        this->ReleaseEvent();
    }

private:
    __aicore__ inline void Compute()
    {
        LocalTensor<T> pointFeatsLocal = this->pointFeatsBuf.template Get<T>();
        LocalTensor<int32_t> prefixSumLocal = this->prefixSumBuf.template Get<int32_t>();
        LocalTensor<int32_t> argsortCoorLocal = this->argsortCoorBuf.template Get<int32_t>();

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
                DataCopy(pointFeatsLocal, this->pointFeatsGm[this->pointIdx * this->featsDim], this->copyFeatParams);
                SetFlag<HardEvent::MTE2_MTE3>(this->eventIdMTE2ToMTE3);
                WaitFlag<HardEvent::MTE2_MTE3>(this->eventIdMTE2ToMTE3);
                this->CopyFeatsOut(voxelIdx, pointFeatsLocal, false);
                SetFlag<HardEvent::MTE3_MTE2>(this->eventIdMTE3ToMTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(this->eventIdMTE3ToMTE2);
            }
        }
    }
};
} // namespace DynamicScatter
#endif // _DYNAMIC_SCATTER_SUM_H_