/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef _DYNAMIC_SCATTER_GRAD_SUM_H_
#define _DYNAMIC_SCATTER_GRAD_SUM_H_

#include "dynamic_scatter_grad_base.h"

namespace DynamicScatterGrad {
using namespace AscendC;

template<typename T>
class DynamicScatterGradSum : public DynamicScatterGradBase<T> {
public:
    __aicore__ inline DynamicScatterGradSum() {}
    __aicore__ inline void Init(GM_ADDR grad_voxel_feats, GM_ADDR prefix_sum_point_per_voxel, GM_ADDR argsort_coor,
        GM_ADDR grad_point_feats, DynamicScatterGradTilingData* tilingData, TPipe* in_pipe)
    {
        this->BaseInit(
            grad_voxel_feats, prefix_sum_point_per_voxel, argsort_coor, grad_point_feats, tilingData, in_pipe);
    }

    __aicore__ inline void Process()
    {
        Compute();
        this->ReleaseEvent();
    }

private:
    __aicore__ inline void Compute()
    {
        LocalTensor<T> voxelGradLocal = this->voxelGradBuf.template Get<T>();
        LocalTensor<int32_t> prefixSumLocal = this->prefixSumBuf.template Get<int32_t>();
        LocalTensor<int32_t> argsortCoorLocal = this->argsortCoorBuf.template Get<int32_t>();

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
                this->CopyPointGradOut(voxelGradLocal);
            }
            SetFlag<HardEvent::MTE3_MTE2>(this->eventIdMTE3ToMTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(this->eventIdMTE3ToMTE2);
        }
    }
};
} // namespace DynamicScatterGrad
#endif // _DYNAMIC_SCATTER_GRAD_SUM_H_