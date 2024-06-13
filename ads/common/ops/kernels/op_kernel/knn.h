/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef _KNN_H_
#define _KNN_H_
#include <cmath>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace AscendC {
// T is the dtype of input and output dist2(float32 or float16) while U is for the output idx(only int32_t)
template<typename T, typename U>
class KnnKernel {
public:
    __aicore__ inline KnnKernel(GM_ADDR xyz, GM_ADDR center_xyz, GM_ADDR dist, const KnnTilingData* tiling_data, TPipe *tmpPipe)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        batch = (uint64_t)tiling_data->batch;
        npoint = (uint64_t)tiling_data->npoint;
        nsource = (uint64_t)tiling_data->nsource;
        core_num = (uint64_t)tiling_data->core_num;
        is_from_knn = tiling_data->is_from_knn;
        former_task_num = Ceil(batch * npoint, core_num);

        comp_num = 256; // 256 : In UB, we will calc comp_num once

        core_id = GetBlockIdx();
        InitGm(xyz, center_xyz, dist, tmpPipe);

        pipe->InitBuffer(targetUb, 32);
        pipe->InitBuffer(sourceBackupUb, comp_num * sizeof(T) * 3);
        pipe->InitBuffer(sourceUb, comp_num * sizeof(T) * 3);
        pipe->InitBuffer(distUb, comp_num * sizeof(T));
    }
    __aicore__ inline void InitGm(GM_ADDR xyz, GM_ADDR center_xyz, GM_ADDR dist, TPipe *tmpPipe)
    {
        pipe = tmpPipe;
        start_task = core_id * former_task_num;
        end_task = start_task + former_task_num;
        if (end_task > (batch * npoint)) {
            end_task = batch * npoint;
        }

        sourceGm.SetGlobalBuffer((__gm__ T *)xyz, batch * nsource * 3);
        targetGm.SetGlobalBuffer((__gm__ T *)center_xyz, batch * npoint * 3);
        distGm.SetGlobalBuffer((__gm__ T *)dist, batch * npoint * nsource);
    }
    __aicore__ inline void Process()
    {
        // 计算loop time
        uint64_t loop_times = nsource / (uint64_t)comp_num;
        uint64_t tail_num = nsource % (uint64_t)comp_num;
        uint64_t tail_num_align = AlignUp(tail_num, 8);
        sourceBackupLocal = sourceBackupUb.Get<T>();
        sourceLocal = sourceUb.Get<T>();
        targetLocal = targetUb.Get<T>();
        distLocal = distUb.Get<T>();

        for (uint64_t current_task = start_task; current_task < end_task; current_task++) {
            uint64_t current_batch = current_task / npoint;
            uint64_t source_offset = current_batch * nsource * 3; // B 3 N
            uint64_t target_offset = current_task * 3; // B M 3
            uint64_t dist_offset = current_task * nsource; // B M N
            DataCopy(targetLocal, targetGm[target_offset], 8);
            pipe_barrier(PIPE_ALL);
            Duplicate<T>(sourceBackupLocal, targetLocal.GetValue(0), (int32_t)comp_num);
            Duplicate<T>(sourceBackupLocal[comp_num], targetLocal.GetValue(1), (int32_t)comp_num);
            Duplicate<T>(sourceBackupLocal[comp_num * 2], targetLocal.GetValue(2), (int32_t)comp_num);
            pipe_barrier(PIPE_ALL);
            for (uint64_t current_loop = 0; current_loop < loop_times; current_loop++) {
                DataCopy(sourceLocal, sourceGm[source_offset + current_loop * comp_num], comp_num);
                DataCopy(sourceLocal[comp_num], sourceGm[source_offset + current_loop * comp_num + nsource], comp_num);
                DataCopy(sourceLocal[comp_num * 2], sourceGm[source_offset + current_loop * comp_num + nsource * 2], comp_num);
                pipe_barrier(PIPE_ALL);
                Sub<T>(sourceLocal, sourceLocal, sourceBackupLocal, comp_num * 3);
                Mul<T>(sourceLocal, sourceLocal, sourceLocal, comp_num * 3);
                Add<T>(distLocal, sourceLocal, sourceLocal[comp_num], comp_num);
                Add<T>(distLocal, distLocal, sourceLocal[comp_num * 2], comp_num);
                if (is_from_knn) {
                    Mins<T>(distLocal, distLocal, static_cast<T>(1e10f), comp_num);
                }

                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                DataCopyPad(distGm[dist_offset + current_loop * comp_num], distLocal,
                    {1, static_cast<uint32_t>(comp_num * sizeof(T)), 0, 0, 0});
            }
            if (tail_num > 0) {
                DataCopy(sourceLocal, sourceGm[source_offset + loop_times * comp_num], tail_num_align);
                DataCopy(sourceLocal[comp_num], sourceGm[source_offset + loop_times * comp_num + nsource], tail_num_align);
                DataCopy(sourceLocal[comp_num * 2], sourceGm[source_offset + loop_times * comp_num + nsource * 2], tail_num_align);
                pipe_barrier(PIPE_ALL);
                Sub<T>(sourceLocal, sourceLocal, sourceBackupLocal, comp_num * 3);
                Mul<T>(sourceLocal, sourceLocal, sourceLocal, comp_num * 3);
                Add<T>(distLocal, sourceLocal, sourceLocal[comp_num], comp_num);
                Add<T>(distLocal, distLocal, sourceLocal[comp_num * 2], comp_num);
                if (is_from_knn) {
                    Mins<T>(distLocal, distLocal, static_cast<T>(1e10f), comp_num);
                }

                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                DataCopyPad(distGm[dist_offset + loop_times * comp_num], distLocal,
                    {1, static_cast<uint32_t>(tail_num * sizeof(T)), 0, 0, 0});
            }
        }
    }
public:
    TPipe *pipe;
    GlobalTensor<T> sourceGm, targetGm, distGm;
    TBuf<TPosition::VECCALC> sourceUb, sourceBackupUb, targetUb, distUb;
    LocalTensor<T> sourceLocal, sourceBackupLocal, targetLocal, distLocal;
    uint32_t core_id;
    uint32_t start_task, end_task;
    uint32_t comp_num;
    uint64_t former_task_num;
public:
    // tiling
    uint64_t batch;
    uint64_t npoint;
    uint64_t nsource;
    uint64_t core_num;
    bool is_from_knn;
};
} // namespace AscendC

#endif  // _KNN_H_