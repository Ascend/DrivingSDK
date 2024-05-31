/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef _KNN_BIG_N_H_
#define _KNN_BIG_N_H_
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "knn.h"

namespace AscendC {
// T is the dtype of input and output dist2(float32 or float16) while U is for the output idx(only int32_t)
template<typename T, typename U>
class KnnCase2 : public KnnKernel<T, U> {
public:
    __aicore__ inline KnnCase2() : KnnKernel<T, U>() {}
    __aicore__ inline ~KnnCase2()
    {
        this->idxUb.template FreeTensor(this->idxLocal);
        this->dist2Ub.template FreeTensor(this->dist2Local);
    }
public:
    __aicore__ inline void Init(GM_ADDR xyz, GM_ADDR center_xyz, GM_ADDR idx, GM_ADDR dist2, knnTilingArgs* tmpTiling,
        TPipe *tmpPipe)
    {
        this->InitGm(xyz, center_xyz, idx, dist2, tmpTiling, tmpPipe);
        this->InitPipe();

        this->pipe->InitBuffer(this->tmpBuf2, 1, this->tilingKernel->topkmax2);
        this->pipe->InitBuffer(this->sourceBackupUb, 1, this->tilingKernel->nsource_aligned_size2 * 3);
        this->pipe->InitBuffer(this->sourceUb, 1, this->tilingKernel->nsource_aligned_size2 * 3);
        this->pipe->InitBuffer(this->distUb, 1, this->tilingKernel->inner * sizeof(T));
        this->pipe->InitBuffer(this->idxUb, 1, this->tilingKernel->nsample_aligned * sizeof(U) * 2);
        this->pipe->InitBuffer(this->dist2Ub, 1, this->tilingKernel->inner2 * sizeof(T));
        this->sourceBackupLocal = this->sourceBackupUb.template AllocTensor<T>();
        this->sourceLocal = this->sourceUb.template AllocTensor<T>();
        this->targetLocal = this->targetUb.template AllocTensor<T>();
        this->idxLocal = this->idxUb.template AllocTensor<U>();
        this->dist2Local = this->dist2Ub.template AllocTensor<T>();
    }
    __aicore__ inline void Process()
    {
        this->current_point = 0;
        for (this->current_b = this->task_b; this->current_b < (this->last_task_b + 1); this->current_b++) {
            uint32_t start = 0;
            uint32_t end   = 0;
            // calc the offset of sourceLocal and the start/end of target at current batch.
            this->calcNpStartAndEnd(start, end);
            // every loop deals with one block of source/xyz
            for (this->current_m = start; this->current_m < end; this->current_m++) {
                for (this->current_loop = 0; this->current_loop < this->tilingKernel->loop_times; this->current_loop++) {
                    // move source/xyz from GM to UB
                    this->source_offset = this->current_loop * this->tilingKernel->nsource_aligned2;
                    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID2);
                    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID2);
                    DataCopy(this->sourceLocal, this->sourceGm[this->source_offset_x + this->source_offset],
                        this->tilingKernel->nsource_aligned2);
                    DataCopy(this->sourceLocal[this->tilingKernel->nsource_aligned2],
                        this->sourceGm[this->source_offset_y + this->source_offset], this->tilingKernel->nsource_aligned2);
                    DataCopy(this->sourceLocal[this->tilingKernel->nsource_aligned2 * 2],
                        this->sourceGm[this->source_offset_z + this->source_offset], this->tilingKernel->nsource_aligned2);
                    this->target_pos_in_targe_num = this->current_point & (this->target_x_num - 1);
                    if (this->target_pos_in_targe_num == 0) {
                        this->copyInTarget();
                    }
                    Compute();
                }
                Muls<T>(this->dist2Local, this->dist2Local, -1, (int32_t)(this->tilingKernel->nsample));
                this->copyOut();
                this->current_point++;
            }
        }
    }
private:
    __aicore__ inline void Compute()
    {
        LocalTensor<int32_t> sourceLocalIndex;
        LocalTensor<bool> sourceLocalFinish;
        uint32_t dst_offset;
        uint32_t actual_dist;
        uint32_t start;
        uint32_t end;
        T target_x = (-1) * static_cast<float>(this->targetLocal.GetValue(this->target_pos_in_targe_num * 3));
        T target_y = (-1) * static_cast<float>(this->targetLocal.GetValue(this->target_pos_in_targe_num * 3 + 1));
        T target_z = (-1) * static_cast<float>(this->targetLocal.GetValue(this->target_pos_in_targe_num * 3 + 2));

        this->distLocal = this->distUb.template AllocTensor<T>();
        Duplicate<T>(this->distLocal, INFINITY, this->tilingKernel->inner);
        if (this->current_loop == (this->tilingKernel->loop_times - 1)) {
            actual_dist = this->tilingKernel->nsource_aligned2 -
                (this->tilingKernel->loop_times * this->tilingKernel->nsource_aligned2 - this->tilingKernel->nsource);
        } else {
            actual_dist = this->tilingKernel->nsource_aligned2;
        }
        this->tilingKernel->topkInfo.n = actual_dist;

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        // calc the distance of each target/center_xyz to the split block of source/xyz
        Adds<T>(this->sourceBackupLocal, this->sourceLocal, (T)target_x, (int32_t)actual_dist);
        Adds<T>(this->sourceBackupLocal[this->tilingKernel->nsource_aligned2],
            this->sourceLocal[this->tilingKernel->nsource_aligned2], (T)target_y, (int32_t)actual_dist);
        Adds<T>(this->sourceBackupLocal[this->tilingKernel->nsource_aligned2 * 2],
            this->sourceLocal[this->tilingKernel->nsource_aligned2 * 2], (T)target_z, (int32_t)actual_dist);
        Mul<T>(this->sourceBackupLocal, this->sourceBackupLocal, this->sourceBackupLocal,
            (int32_t)this->tilingKernel->nsource_aligned2 * 3);
        Add<T>(this->distLocal, this->sourceBackupLocal, this->sourceBackupLocal[this->tilingKernel->nsource_aligned2],
            (int32_t)actual_dist);
        Add<T>(this->distLocal, this->distLocal, this->sourceBackupLocal[this->tilingKernel->nsource_aligned2 * 2],
            (int32_t)actual_dist);
        // alloc topkmax(used for TopK)
        this->tmpLocal = this->tmpBuf.template AllocTensor<uint8_t>();
        dst_offset = (this->current_loop == 0) ? 0 : this->tilingKernel->nsample_aligned;

        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);

        // calc the topk nearest points in source/xyz
        Muls<T>(this->distLocal, this->distLocal, -1, (int32_t)(this->tilingKernel->inner));
        TopK<T>(this->dist2Local[dst_offset], this->idxLocal[dst_offset], this->distLocal, sourceLocalIndex,
            sourceLocalFinish, this->tmpLocal, this->tilingKernel->nsample, this->tilingKernel->topkTilingData,
            this->tilingKernel->topkInfo, true);
        // clear the tail of dist2Local to -INFINITY
        start = (this->current_loop == 0) ?
            (this->tilingKernel->nsample) : (this->tilingKernel->nsample + this->tilingKernel->nsample_aligned);
        end   = (this->current_loop == 0) ? (this->tilingKernel->nsample_aligned) : (this->tilingKernel->inner2);
        for (uint32_t j = start; j < end; j++) {
            this->dist2Local.SetValue(j, -INFINITY);
        }
        this->distUb.template FreeTensor<T>(this->distLocal);
        this->tmpBuf.template FreeTensor<uint8_t>(this->tmpLocal);
        if (this->current_loop == 0) {
            return ;
        }
        // compare the results with the one before and sort using TopK
        this->tmpLocal2 = this->tmpBuf2.template AllocTensor<uint8_t>();
        Adds<U>(this->idxLocal[dst_offset], this->idxLocal[dst_offset], this->source_offset, this->tilingKernel->nsample);
        TopK<T, true>(this->dist2Local, this->idxLocal, this->dist2Local, this->idxLocal, sourceLocalFinish,
            this->tmpLocal2, this->tilingKernel->nsample, this->tilingKernel->topkTilingData2,
            this->tilingKernel->topkInfo2, true);
        this->tmpBuf2.template FreeTensor<uint8_t>(this->tmpLocal2);
    }
private:
    TQue<QuePosition::VECOUT, 1> tmpBuf2;
    LocalTensor<uint8_t> tmpLocal2;
    uint32_t current_loop;
    uint32_t source_offset;
};
} // namespace AscendC

#endif  // _KNN_BIG_N_H_