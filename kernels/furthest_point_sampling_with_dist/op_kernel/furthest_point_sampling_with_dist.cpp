/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1u;
constexpr uint32_t BLOCK_BYTE_SIZE = 32u;


template <typename dataType, typename idxType>
class KernelFurthestPointSamplingWithDist {
public:
    __aicore__ inline KernelFurthestPointSamplingWithDist() = default;
    __aicore__ inline void Init(GM_ADDR points_dist, GM_ADDR nearest_temp, GM_ADDR index, GM_ADDR workspace,
                            const FurthestPointSamplingWithDistTilingData* __restrict tiling);
    __aicore__ inline void Process();
private:
    __aicore__ inline void init_tiling_value();
    __aicore__ inline void ProcessEachBatch(uint64_t batch_id);
    __aicore__ inline void PointSampling(uint32_t id_times, uint32_t id_len);
    __aicore__ inline void CalculatePartUb(uint32_t n_times, uint32_t n_len);
    __aicore__ inline void CopyIn(uint32_t n_times, uint32_t n_len);
    __aicore__ inline void Compute(uint32_t n_times, uint32_t n_len);
    __aicore__ inline void CopyOutDist(uint32_t n_times, uint32_t n_len);
    __aicore__ inline void CopyOut(uint32_t id_times, uint32_t id_len);
    __aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y);
    
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> points_dist_queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> temp_queue;

    TQue<QuePosition::VECOUT, BUFFER_NUM> temp_out_queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> idx_queue;

    TBuf<TPosition::VECCALC> work_Buf;
    TBuf<TPosition::VECCALC> dist_Buf;

    GlobalTensor<dataType> points_dist_gm;
    GlobalTensor<idxType> idx_gm;
    GlobalTensor<dataType> temp_gm;

    uint32_t used_core_num {0};
    uint32_t points_num {0};
    uint32_t task_num {0};
    uint32_t task_num_tail {0};
    uint32_t n {0};
    uint32_t batch_dist_offset {0};
    uint32_t batch_idx_offset {0};
    uint32_t part_ub {0};
    uint32_t move_n_times {0};
    uint32_t n_tail {0};
    uint32_t repeat_id_times {0};
    uint32_t id_move_len {0};
    uint32_t id_tail {0};
    uint32_t work_size {0};

    int32_t last_idx {-1};
    uint32_t batch_size {0};
    uint32_t data_type_size {0};
    uint32_t idx_type_size {0};
    uint32_t mask_type_size {0};
    uint64_t dist_begin_offset {0};
    uint64_t temp_begin_offset {0};
    uint64_t idx_begin_offset {0};
    uint32_t now_max_dim {0};
    uint32_t block_size {32};
    uint32_t block_per_size {0};
    float now_max_dist {0};

    const FurthestPointSamplingWithDistTilingData* __restrict tiling_device {nullptr};
};

template<typename dataType, typename idxType>
__aicore__ inline void KernelFurthestPointSamplingWithDist<dataType, idxType>::Init(GM_ADDR points_dist, GM_ADDR nearest_temp, GM_ADDR idx, GM_ADDR workspace, const FurthestPointSamplingWithDistTilingData* __restrict tiling) {
    // init tiling
    this->tiling_device = tiling;
    init_tiling_value();

    uint32_t core_id = GetBlockIdx();

    uint32_t batch_begin_offset = core_id * task_num;

    // 判断是否为尾核
    bool is_last_core = (core_id == (used_core_num - 1));
    if (!is_last_core) {
        batch_size = task_num;
    }
    else {
        batch_size = task_num_tail;
    }

    // calculate begin offset
    uint64_t gm_dist_begin_offset = batch_begin_offset * batch_dist_offset;
    uint64_t gm_temp_begin_offset = batch_begin_offset * n;
    uint64_t gm_idx_begin_offset = batch_begin_offset * batch_idx_offset;
    
    // set LocalTensor base addr
    this->points_dist_gm.SetGlobalBuffer((__gm__ dataType *)points_dist + gm_dist_begin_offset, static_cast<uint64_t>(batch_size) * batch_dist_offset);
    this->temp_gm.SetGlobalBuffer((__gm__ dataType *)nearest_temp + gm_temp_begin_offset, static_cast<uint64_t>(batch_size) * n);
    this->idx_gm.SetGlobalBuffer((__gm__ idxType *)idx + gm_idx_begin_offset, static_cast<uint64_t>(batch_size) * batch_idx_offset);

    data_type_size = sizeof(dataType);
    idx_type_size = sizeof(idxType);
    mask_type_size = sizeof(uint8_t);
    block_per_size = block_size / data_type_size;

    this->pipe.InitBuffer(this->points_dist_queue, BUFFER_NUM, part_ub * data_type_size);
    this->pipe.InitBuffer(this->temp_queue, BUFFER_NUM, part_ub * data_type_size);
    
    this->pipe.InitBuffer(this->temp_out_queue, BUFFER_NUM, part_ub * data_type_size);
    this->pipe.InitBuffer(this->idx_queue, BUFFER_NUM, id_move_len * idx_type_size);

    this->pipe.InitBuffer(this->dist_Buf, block_size);
    this->pipe.InitBuffer(this->work_Buf, work_size * data_type_size);
}

template<typename dataType, typename idxType>
__aicore__ inline void KernelFurthestPointSamplingWithDist<dataType, idxType>::Process() {
    for (uint64_t batch_id = 0; batch_id < batch_size; ++batch_id) {
        last_idx = -1;
        ProcessEachBatch(batch_id);
    }
}

template<typename dataType, typename idxType>
__aicore__ inline void KernelFurthestPointSamplingWithDist<dataType, idxType>::init_tiling_value() {
    used_core_num = tiling_device->used_core_num;
    points_num = tiling_device->points_num;
    task_num = tiling_device->task_num;
    task_num_tail = tiling_device->task_num_tail;
    n = tiling_device->n;
    batch_dist_offset = tiling_device->batch_dist_offset;
    batch_idx_offset = tiling_device->batch_idx_offset;
    part_ub = tiling_device->part_ub;
    move_n_times = tiling_device->move_n_times;
    n_tail = tiling_device->n_tail;
    id_move_len = tiling_device->id_move_len;
    repeat_id_times = tiling_device->repeat_id_times;
    id_tail = tiling_device->id_tail;
    work_size = tiling_device->work_size;
}

template<typename dataType, typename idxType>
__aicore__ inline void KernelFurthestPointSamplingWithDist<dataType, idxType>::ProcessEachBatch(uint64_t batch_id) {
    dist_begin_offset = batch_id * batch_dist_offset;
    temp_begin_offset = batch_id * n;
    idx_begin_offset = batch_id * batch_idx_offset;
    
    for (uint32_t id_times = 0; id_times < repeat_id_times - 1; ++id_times) {
        PointSampling(id_times, id_move_len);
    }
    PointSampling(repeat_id_times - 1, id_tail);
}

template<typename dataType, typename idxType>
__aicore__ inline void KernelFurthestPointSamplingWithDist<dataType, idxType>::PointSampling(uint32_t id_times, uint32_t id_len) {
    LocalTensor<idxType> idx_local = idx_queue.AllocTensor<idxType>();

    for (uint32_t i = 0; i < id_len; i++) {
        now_max_dim = last_idx;
        now_max_dist = 0;
        for (uint32_t j = 0; j < move_n_times - 1; j++) {
            CalculatePartUb(j, part_ub);
        }
        CalculatePartUb(move_n_times - 1, n_tail);
        last_idx = now_max_dim;
        idx_local.SetValue(i, now_max_dim);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }
    idx_queue.EnQue<idxType>(idx_local);
    CopyOut(id_times, id_len);
}

template<typename dataType, typename idxType>
__aicore__ inline void KernelFurthestPointSamplingWithDist<dataType, idxType>::CalculatePartUb(uint32_t n_times, uint32_t n_len) {
    CopyIn(n_times, n_len);
    pipe_barrier(PIPE_ALL);
    Compute(n_times, n_len);
    CopyOutDist(n_times, n_len);
}

template<typename dataType, typename idxType>
__aicore__ inline void KernelFurthestPointSamplingWithDist<dataType, idxType>::CopyIn(uint32_t n_times, uint32_t n_len) {
    LocalTensor<dataType> dist1_local = temp_queue.AllocTensor<dataType>();
    LocalTensor<dataType> dist2_local = points_dist_queue.AllocTensor<dataType>();
    // calculate offset
    uint64_t dist1_offset = temp_begin_offset + static_cast<uint64_t>(n_times) * part_ub;
    uint64_t dist2_offset = dist_begin_offset + static_cast<uint64_t>(last_idx) * n + static_cast<uint64_t>(n_times) * part_ub;
    uint32_t move_len = CeilDiv(n_len, block_per_size) * block_per_size;
    // data copy
    DataCopy(dist1_local, temp_gm[dist1_offset], move_len);
    if (last_idx == -1) {
        DataCopy(dist2_local, temp_gm[dist1_offset], move_len);
    }
    else {
        DataCopy(dist2_local, points_dist_gm[dist2_offset], move_len);
    }
    temp_queue.EnQue<dataType>(dist1_local);
    points_dist_queue.EnQue<dataType>(dist2_local);
}

template<typename dataType, typename idxType>
__aicore__ inline void KernelFurthestPointSamplingWithDist<dataType, idxType>::Compute(uint32_t n_times, uint32_t n_len) {
    LocalTensor<dataType> dist1_local = temp_queue.DeQue<dataType>();
    LocalTensor<dataType> dist2_local = points_dist_queue.DeQue<dataType>();
    LocalTensor<dataType> dist3_local = temp_out_queue.AllocTensor<dataType>();

    LocalTensor<dataType> workLocal = work_Buf.Get<dataType>();
    LocalTensor<dataType> dstLocal = dist_Buf.Get<dataType>();

    Min(dist3_local, dist1_local, dist2_local, n_len);
    // calculate reduce_max
    ReduceMax<dataType>(dstLocal, dist3_local, workLocal, n_len, true);
    float dist = dstLocal.GetValue(0);
    LocalTensor<uint32_t> idx_int32 = dstLocal.template ReinterpretCast<uint32_t>();
    int32_t idx = idx_int32.GetValue(1);
    
    if (dist > now_max_dist) {
        now_max_dist = dist;
        now_max_dim = idx + n_times * part_ub;
    }
    
    // enque
    temp_out_queue.EnQue<dataType>(dist3_local);

    temp_queue.FreeTensor(dist1_local);
    points_dist_queue.FreeTensor(dist2_local);
}

template<typename dataType, typename idxType>
__aicore__ inline void KernelFurthestPointSamplingWithDist<dataType, idxType>::CopyOutDist(uint32_t n_times, uint32_t n_len) {
    LocalTensor<dataType> temp_local = temp_out_queue.DeQue<dataType>();

    uint64_t dist1_offset = temp_begin_offset + static_cast<uint64_t>(n_times) * part_ub;

    DataCopyParams copyParams{1, (uint16_t)(n_len * sizeof(dataType)), 0, 0};
    DataCopyPad(temp_gm[dist1_offset], temp_local, copyParams);
    
    temp_out_queue.FreeTensor(temp_local);
}

template<typename dataType, typename idxType>
__aicore__ inline void KernelFurthestPointSamplingWithDist<dataType, idxType>::CopyOut(uint32_t id_times, uint32_t id_len) {
    LocalTensor<idxType> idx_local = idx_queue.DeQue<idxType>();

    uint64_t idx_offset = idx_begin_offset + static_cast<uint64_t>(id_times) * id_move_len;

    DataCopyParams copyParams{1, (uint16_t)(id_len * sizeof(idxType)), 0, 0};
    DataCopyPad(idx_gm[idx_offset], idx_local, copyParams);
    idx_queue.FreeTensor(idx_local);
}

template<typename dataType, typename idxType>
__aicore__ inline uint32_t KernelFurthestPointSamplingWithDist<dataType, idxType>::CeilDiv(uint32_t x, uint32_t y) {
    return y == 0 ? x : (x + y - 1) / y;
}

extern "C" __global__ __aicore__ void furthest_point_sampling_with_dist(GM_ADDR points_dist, GM_ADDR nearest_temp, GM_ADDR index, GM_ADDR workspace, GM_ADDR tiling) {
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR user_ws = GetUserWorkspace(workspace);
    if (user_ws == nullptr) {
        return;
    }
    
    GET_TILING_DATA(tiling_data, tiling);
    const FurthestPointSamplingWithDistTilingData* __restrict tiling_device = &tiling_data;
    KernelFurthestPointSamplingWithDist<float, int32_t> op;
    op.Init(points_dist, nearest_temp, index, user_ws, tiling_device);
    op.Process();
}
