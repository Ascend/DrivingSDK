/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef _KNN_SMALL_N_H_
#define _KNN_SMALL_N_H_
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "knn.h"

namespace AscendC {
// T is the dtype of input and output dist2(float32 or float16) while U is for the output idx(only int32_t)
template<typename T, typename U>
class KnnCase1 : public KnnKernel<T, U> {
public:
    __aicore__ inline KnnCase1() : KnnKernel<T, U>() {}
    __aicore__ inline ~KnnCase1()
    {
        this->tmpBuf.template FreeTensor(this->tmpLocal);
    }
public:
    __aicore__ inline void Init(GM_ADDR xyz, GM_ADDR center_xyz, GM_ADDR idx, GM_ADDR dist2, knnTilingArgs* tmpTiling,
        TPipe *tmpPipe)
    {
        this->InitGm(xyz, center_xyz, idx, dist2, tmpTiling, tmpPipe);

        this->pipe->InitBuffer(this->tmpBuf,   1, this->tilingKernel->topkmax);
        this->pipe->InitBuffer(this->targetUb, 1, this->aligned_actual_size);
        if (this->tilingKernel->is_from_knn) {
            this->pipe->InitBuffer(this->compareUb, 1, this->tilingKernel->nsample_aligned);
            this->compareLocal = this->compareUb.template AllocTensor<uint8_t>();
        }
        this->pipe->InitBuffer(this->sourceUb, 1, this->tilingKernel->nsource_aligned_size * 3);
        this->pipe->InitBuffer(this->distUb,   1, this->tilingKernel->inner * sizeof(T));
        this->pipe->InitBuffer(this->idxUb,    1, this->tilingKernel->nsample_aligned * sizeof(U));
        this->pipe->InitBuffer(this->dist2Ub,  1, this->tilingKernel->nsample_aligned * sizeof(T));
        this->targetLocal = this->targetUb.template AllocTensor<T>();
    }
    __aicore__ inline void Process()
    {
        this->copyInTarget();
        for (this->current_b = this->task_b; this->current_b < (this->last_task_b + 1); this->current_b++) {
            uint32_t start = 0;
            uint32_t end   = 0;
            this->calcNpStartAndEnd(start, end);
            for (this->current_m = start; this->current_m < end; this->current_m++) {
                Compute();
                this->current_point++;
            }
        }
    }
private:
    __aicore__ inline void Compute()
    {
        LocalTensor<int32_t> sourceLocalIndex;
        LocalTensor<bool> sourceLocalFinish;
        T target_x = (-1) * static_cast<float>(this->targetLocal.GetValue(this->current_point * 3));
        T target_y = (-1) * static_cast<float>(this->targetLocal.GetValue(this->current_point * 3 + 1));
        T target_z = (-1) * static_cast<float>(this->targetLocal.GetValue(this->current_point * 3 + 2));
        this->tilingKernel->topkInfo.n = this->tilingKernel->nsource;

        this->sourceLocal = this->sourceUb.template AllocTensor<T>();
        this->distLocal   = this->distUb.template AllocTensor<T>();

        DataCopy(this->sourceLocal, this->sourceGm[this->source_offset_x], this->tilingKernel->nsource_aligned);
        DataCopy(this->sourceLocal[this->tilingKernel->nsource_aligned], this->sourceGm[this->source_offset_y],
            this->tilingKernel->nsource_aligned);
        DataCopy(this->sourceLocal[this->tilingKernel->nsource_aligned * 2], this->sourceGm[this->source_offset_z],
            this->tilingKernel->nsource_aligned);
        set_flag(PIPE_MTE2,  PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        set_flag(PIPE_S,  PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

        Duplicate<T>(this->distLocal, INFINITY, this->tilingKernel->inner);
        Adds<T>(this->sourceLocal, this->sourceLocal, (T)target_x, (int32_t)this->tilingKernel->nsource);
        Adds<T>(this->sourceLocal[this->tilingKernel->nsource_aligned],
            this->sourceLocal[this->tilingKernel->nsource_aligned], (T)target_y, (int32_t)this->tilingKernel->nsource);
        Adds<T>(this->sourceLocal[this->tilingKernel->nsource_aligned * 2],
            this->sourceLocal[this->tilingKernel->nsource_aligned * 2], (T)target_z, (int32_t)this->tilingKernel->nsource);
        Mul<T>(this->sourceLocal, this->sourceLocal, this->sourceLocal, (int32_t)this->tilingKernel->nsource_aligned * 3);
        Add<T>(this->distLocal, this->sourceLocal, this->sourceLocal[this->tilingKernel->nsource_aligned],
            (int32_t)this->tilingKernel->nsource);
        Add<T>(this->distLocal, this->distLocal, this->sourceLocal[this->tilingKernel->nsource_aligned * 2],
            (int32_t)this->tilingKernel->nsource);

        this->tmpLocal   = this->tmpBuf.template AllocTensor<uint8_t>();
        this->idxLocal   = this->idxUb.template AllocTensor<U>();
        this->dist2Local = this->dist2Ub.template AllocTensor<T>();

        Muls<T>(this->distLocal, this->distLocal, -1, (int32_t)(this->tilingKernel->inner));
        TopK<T>(this->dist2Local, this->idxLocal, this->distLocal, sourceLocalIndex, sourceLocalFinish, this->tmpLocal,
            this->tilingKernel->nsample, this->tilingKernel->topkTilingData, this->tilingKernel->topkInfo, true);
        Muls<T>(this->dist2Local, this->dist2Local, -1, (int32_t)(this->tilingKernel->nsample));
        this->sourceUb.template FreeTensor<T>(this->sourceLocal);
        this->distUb.template FreeTensor<T>(this->distLocal);
        this->copyOut();
    }
};
} // namespace AscendC

#endif  // _KNN_SMALL_N_H_