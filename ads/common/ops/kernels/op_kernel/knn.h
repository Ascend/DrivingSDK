/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef _KNN_H_
#define _KNN_H_
#include <cmath>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace AscendC {
class knnTilingArgs {
public:
    __aicore__ inline knnTilingArgs() = default;
public:
    uint32_t batch;
    uint32_t npoint;
    uint32_t nsample;
    uint32_t nsample_aligned;
    uint32_t nsource;
    uint32_t nsource_aligned;
    uint32_t nsource_aligned2;
    uint32_t nsource_aligned_size;
    uint32_t nsource_aligned_size2;
    bool is_from_knn;
    uint32_t inner;
    uint32_t inner2;
    uint32_t topkmax;
    uint32_t topkmax2;
    uint32_t loop_times;
    uint32_t b_times_m;
    uint32_t big_core_num;
    uint32_t small_core_num;
    uint32_t big_core_len;
    uint32_t small_core_len;
    uint32_t aligned_big_len;
    uint32_t aligned_big_size;
    uint32_t aligned_small_len;
    uint32_t aligned_small_size;
    TopkTiling topkTilingData;
    TopkTiling topkTilingData2;
    TopKInfo topkInfo;
    TopKInfo topkInfo2;
};
// T is the dtype of input and output dist2(float32 or float16) while U is for the output idx(only int32_t)
template<typename T, typename U>
class KnnKernel {
public:
    __aicore__ inline KnnKernel()
    {
        this->core_id = GetBlockIdx();
    };
    __aicore__ inline ~KnnKernel()
    {
        if (this->tilingKernel->is_from_knn) {
            this->compareUb.template FreeTensor<uint8_t>(this->compareLocal);
        }
        this->targetUb.template FreeTensor(this->targetLocal);
        this->idxUb.template FreeTensor(this->idxLocal);
        this->dist2Ub.template FreeTensor(this->dist2Local);
    };
    __aicore__ inline void InitGm(GM_ADDR xyz, GM_ADDR center_xyz, GM_ADDR idx, GM_ADDR dist2, knnTilingArgs* tmpTiling,
        TPipe *tmpPipe)
    {
        uint32_t start_offset;
        uint32_t end_offset;
        this->pipe         = tmpPipe;
        this->tilingKernel = tmpTiling;
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        // calc data offsets where each core starts to deal with
        if (this->core_id < this->tilingKernel->big_core_num) {
            start_offset              = this->core_id * this->tilingKernel->big_core_len;
            end_offset                = start_offset + this->tilingKernel->big_core_len;
            this->actual_len          = this->tilingKernel->big_core_len;
            this->aligned_actual_size = this->tilingKernel->aligned_big_size;
            this->aligned_actual_len  = this->tilingKernel->aligned_big_len;
        } else {
            start_offset              = this->tilingKernel->big_core_num * this->tilingKernel->big_core_len +
                (this->core_id - this->tilingKernel->big_core_num) * this->tilingKernel->small_core_len;
            end_offset                = start_offset + this->tilingKernel->small_core_len;
            this->actual_len          = this->tilingKernel->small_core_len;
            this->aligned_actual_size = this->tilingKernel->aligned_small_size;
            this->aligned_actual_len  = this->tilingKernel->aligned_small_len;
        }
        this->task_b     = start_offset / this->tilingKernel->npoint;
        this->task_m     = start_offset - this->task_b * this->tilingKernel->npoint; // 0~(np-1)
        this->offset_sourceGm  = this->task_b * this->tilingKernel->nsource * 3;
        this->offset_targetGm  = this->task_b * this->tilingKernel->npoint * 3 + this->task_m * 3;
        this->offset_outputGm  = this->task_b * this->tilingKernel->npoint * this->tilingKernel->nsample +
            this->task_m * this->tilingKernel->nsample;
        // calc the final position
        this->last_task_b  = end_offset / this->tilingKernel->npoint;
        this->num_batch    = this->last_task_b - this->task_b + 1;

        this->sourceGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(xyz) + this->offset_sourceGm,
            this->num_batch * this->tilingKernel->nsource * 3);
        this->targetGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(center_xyz) + this->offset_targetGm,
            this->aligned_actual_len * 3);
        this->idxGm.SetGlobalBuffer(reinterpret_cast<__gm__ U *>(idx) + this->offset_outputGm,
            this->actual_len * this->tilingKernel->nsample);
        this->dist2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dist2) + this->offset_outputGm,
            this->actual_len * this->tilingKernel->nsample);
    }
    __aicore__ inline void copyInTarget()
    {
        this->current_point = 0;
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        DataCopy(this->targetLocal, this->targetGm, this->aligned_actual_len * 3);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    }
    __aicore__ inline void calcNpStartAndEnd(uint32_t &start, uint32_t &end)
    {
        this->source_offset_x = (this->current_b - this->task_b) * this->tilingKernel->nsource * 3;
        this->source_offset_y = this->source_offset_x + this->tilingKernel->nsource;
        this->source_offset_z = this->source_offset_y + this->tilingKernel->nsource;
        if (this->num_batch > 1) {
            if (this->current_b == this->task_b) {
                start = this->task_m;
                end   = this->tilingKernel->npoint;
            } else if (this->current_b == this->last_task_b) {
                start = 0;
                end   = this->actual_len - (this->num_batch - 1) * this->tilingKernel->npoint + this->task_m;
            } else {
                start = 0;
                end   = this->tilingKernel->npoint;
            }
        } else {
            start = this->task_m;
            end   = this->task_m + this->actual_len;
        }
    }
    __aicore__ inline void spetialDeal()
    {
        uint32_t dist2Local_start = 0;
        uint32_t dist2Local_end   = this->tilingKernel->nsample - 1;
        uint32_t first_inf;

        Mins<T>(this->dist2Local, this->dist2Local, static_cast<T>(1e10f), this->tilingKernel->nsample);
        set_flag(PIPE_V,  PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        while (dist2Local_start <= dist2Local_end) {
            first_inf = (dist2Local_start + dist2Local_end) / 2;
            if (static_cast<float>(this->dist2Local.GetValue(first_inf)) < static_cast<float>(1e10f)) {
                dist2Local_start = first_inf + 1;
            } else {
                dist2Local_end   = first_inf - 1;
            }
        }
        first_inf = dist2Local_start;
        if (static_cast<float>(this->dist2Local.GetValue(first_inf)) < static_cast<float>(1e10f)) {
            first_inf++;
        }
        for (uint32_t i = first_inf; i < this->tilingKernel->nsample; i++) {
            this->idxLocal.SetValue(i, 0);
        }
    }
    __aicore__ inline void copyOut()
    {
        if (this->tilingKernel->is_from_knn) {
            this->spetialDeal();
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        DataCopyPad(this->idxGm[this->current_point * this->tilingKernel->nsample], this->idxLocal,
            {1, static_cast<uint32_t>(this->tilingKernel->nsample * sizeof(U)), 0, 0, 0});
        DataCopyPad(this->dist2Gm[this->current_point * this->tilingKernel->nsample], this->dist2Local,
            {1, static_cast<uint32_t>(this->tilingKernel->nsample * sizeof(T)), 0, 0, 0});
    }
public:
    TPipe *pipe;
    knnTilingArgs *tilingKernel;
    GlobalTensor<T> sourceGm, targetGm, dist2Gm;
    GlobalTensor<U> idxGm;
    TQue<QuePosition::VECOUT, 1> targetUb, idxUb, dist2Ub, tmpBuf;
    TQue<QuePosition::VECIN,  1> sourceUb, compareUb, distUb;
    LocalTensor<T> sourceLocal, targetLocal, distLocal, dist2Local;
    LocalTensor<U> idxLocal;
    LocalTensor<uint8_t> tmpLocal;
    LocalTensor<uint8_t> compareLocal;
    uint32_t core_id;
    uint32_t offset_sourceGm, offset_targetGm, offset_outputGm;
    uint32_t source_offset_x, source_offset_y, source_offset_z;
    uint32_t actual_len, aligned_actual_len, aligned_actual_size;
    uint32_t task_b, task_m, last_task_b, num_batch;
    uint32_t current_b, current_m, current_point;
};
} // namespace AscendC

#endif  // _KNN_H_