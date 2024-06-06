/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

class KernelSubmSparseConv3d {
public:
    __aicore__ inline KernelSubmSparseConv3d() {}

    TQue<QuePosition::VECIN, 1> inQueueIndices, inQueueWeight, inQueueFeature;
    TBuf<TPosition::VECCALC> tempbuf, tempbuf4, uint8buf, zerobuf, onebuf;
    TBuf<TPosition::VECCALC> tempbuf2, tempbuf3, dstbuf, indicesoffsetbuf, indicespairbuf;
    TQue<QuePosition::VECOUT, 1> outQueueOUTPUT;
    GlobalTensor<DTYPE_FEATURE> featureGm;
    GlobalTensor<DTYPE_INDICES> indicesGm;
    GlobalTensor<DTYPE_FEATURE> weightGm;
    GlobalTensor<DTYPE_FEATURE> outputGm;
    GlobalTensor<DTYPE_INDICES> indices_offsetGm;
    GlobalTensor<DTYPE_INDICES> indices_pairGm;
    uint32_t core_used;
    uint32_t core_data;
    uint32_t copy_loop;
    uint32_t copy_tail;
    uint32_t last_copy_loop;
    uint32_t last_copy_tail;
    uint32_t inchannel;
    uint32_t indices_number;
    uint32_t feature_map_size;
    uint32_t available_ub_size;
    uint32_t K0;
    uint32_t K1;
    uint32_t K2;
    uint32_t out_channel;
    uint32_t batch_size;
    int32_t outSpatialShape[3];
    int32_t total_feature;
    LocalTensor<DTYPE_INDICES> indices_ub;
    LocalTensor<DTYPE_FEATURE> weight_ub;
    LocalTensor<DTYPE_FEATURE> feature_ub;
    LocalTensor<DTYPE_INDICES> indices_ub_temp;
    LocalTensor<DTYPE_INDICES> indices_ub_temp2;
    LocalTensor<uint8_t> temp_ub;
    LocalTensor<DTYPE_FEATURE> one_ub;
    LocalTensor<DTYPE_FEATURE> zero_ub;
    LocalTensor<DTYPE_FEATURE> dst_ub;
    LocalTensor<int32_t> indices_offset_ub;
    LocalTensor<int32_t> indices_pair_ub;
    LocalTensor<DTYPE_FEATURE> result_temp;
    LocalTensor<DTYPE_FEATURE> compute_temp;
    DataCopyPadParams padParams{false, 0, 0, 0};
    int32_t total_kernel_size = 27;

    __aicore__ inline void Init(GM_ADDR feature, GM_ADDR indices,
                                GM_ADDR weight,
                                GM_ADDR feature_out,
                                GM_ADDR indices_offset,
                                GM_ADDR indices_pair,
                                GM_ADDR workspace,
                                SubmSparseConv3dTilingData *tiling_data, TPipe* pipe)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zeronumber!");
        this->core_used = tiling_data->core_used;
        this->core_data = tiling_data->core_data;
        this->copy_loop = tiling_data->copy_loop;
        this->copy_tail = tiling_data->copy_tail;
        this->last_copy_loop = tiling_data->last_copy_loop;
        this->last_copy_tail = tiling_data->last_copy_tail;
        this->inchannel = tiling_data->inchannel;
        this->indices_number = tiling_data->indices_number;
        this->feature_map_size = tiling_data->feature_map_size;
        this->available_ub_size = tiling_data->available_ub_size;
        this->total_feature = tiling_data->total_feature;
        this->K0 = tiling_data->K0;
        this->K1 = tiling_data->K1;
        this->K2 = tiling_data->K2;
        this->batch_size = tiling_data->batch_size;
        this->out_channel = tiling_data->outchannel;
        this->outSpatialShape[0] = tiling_data->D;
        this->outSpatialShape[1] = tiling_data->H;
        this->outSpatialShape[2] = tiling_data->W;

        indicesGm.SetGlobalBuffer((__gm__ DTYPE_INDICES*)indices, this->indices_number * 4);
        weightGm.SetGlobalBuffer((__gm__ DTYPE_FEATURE*)weight, 27*this->out_channel * this->inchannel);
        featureGm.SetGlobalBuffer((__gm__ DTYPE_FEATURE*)feature, this->core_data * this->inchannel);
        outputGm.SetGlobalBuffer(
            (__gm__ DTYPE_FEATURE*)feature_out, this->core_data * 27 * this->out_channel);
        indices_offsetGm.SetGlobalBuffer(
            (__gm__ int32_t*)indices_offset, this->core_data * 27);
        indices_pairGm.SetGlobalBuffer(
            (__gm__ int32_t*)indices_pair, this->core_data * 27 * 4);
        pipe->InitBuffer(inQueueIndices, 1, this->available_ub_size * 4 * sizeof(DTYPE_FEATURE));
        pipe->InitBuffer(inQueueWeight, 1, this->out_channel * this->inchannel * sizeof(DTYPE_FEATURE));
        pipe->InitBuffer(inQueueFeature, 1, this->inchannel * sizeof(DTYPE_FEATURE));
        pipe->InitBuffer(indicespairbuf, total_kernel_size * 4 * sizeof(int32_t));
        pipe->InitBuffer(tempbuf, this->available_ub_size * sizeof(DTYPE_FEATURE));
        pipe->InitBuffer(tempbuf4, this->available_ub_size * sizeof(DTYPE_FEATURE));
        pipe->InitBuffer(uint8buf, this->available_ub_size * sizeof(uint8_t));
        pipe->InitBuffer(zerobuf, this->available_ub_size * sizeof(DTYPE_FEATURE));
        pipe->InitBuffer(onebuf, this->available_ub_size * sizeof(DTYPE_FEATURE));
        pipe->InitBuffer(tempbuf2, this->available_ub_size * sizeof(DTYPE_FEATURE));
        pipe->InitBuffer(tempbuf3, this->available_ub_size * sizeof(DTYPE_FEATURE));
        pipe->InitBuffer(dstbuf, this->available_ub_size * sizeof(DTYPE_FEATURE));
        pipe->InitBuffer(indicesoffsetbuf, total_kernel_size * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        uint32_t core_id = GetBlockIdx();
        uint64_t start_address = core_id * this->core_data;
        if (core_id > this->core_used) {
            return;
        }
        if (core_id != (this->core_used -1)) {
            for (uint32_t i = 0; i < this->copy_loop; i++) {
                uint64_t address = start_address + i * this->available_ub_size;
                IndicesCompute(i, this->available_ub_size, address);
            }
            if (this->copy_tail != 0) {
                uint64_t address = start_address + this->copy_loop * this->available_ub_size;
                IndicesCompute(this->copy_loop, this->copy_tail, address);
            }
        } else {
            for (uint32_t i = 0; i < this->last_copy_loop; i++) {
                uint64_t address = start_address + i * this->available_ub_size;
                IndicesCompute(i, this->available_ub_size, address);
            }
            if (this->last_copy_tail != 0) {
                uint64_t address = start_address + this->last_copy_loop * this->available_ub_size;
                IndicesCompute(this->last_copy_loop, this->last_copy_tail, address);
            }
        }
    }

private:
    __aicore__ inline void convcompute(int32_t tensor_size, int32_t offset, int32_t il, int32_t point_offset,
                                       int32_t point_idx, uint64_t address, float batch_id,
                                       float point0, float point1, float point2, int32_t kernel_size_offset)
    {
        uint64_t mask = 64;
        if (tensor_size < 64) {
            mask = tensor_size;
        }
        int repeat = (tensor_size + mask - 1) / mask;
        DataCopyParams copyParams_weight{1,
                                        (uint16_t)(this->out_channel * this->inchannel * sizeof(DTYPE_FEATURE)), 0, 0};
        DataCopyParams copyParams_feature{1, (uint16_t)(this->inchannel * sizeof(DTYPE_FEATURE)), 0, 0};
        DataCopyParams copyParams_output{1, (uint16_t)(this->out_channel * sizeof(DTYPE_FEATURE)), 0, 0};
        Duplicate(weight_ub, (float)(point_offset), tensor_size);
        temp_ub = feature_ub == weight_ub;
        BinaryRepeatParams repeatParams = { 1, 1, 1, 8, 8, 8 };
        Select(feature_ub, temp_ub, one_ub, zero_ub,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, mask, repeat, repeatParams);
        ReduceMax<DTYPE_FEATURE>(dst_ub, feature_ub,
                                 compute_temp, tensor_size, true);
        // 判断point是否在输入索引中
        if (dst_ub.GetValue(0) == 1) {
            auto indices_idx = dst_ub.GetValue(1);
            DataCopyPad(weight_ub,
                        weightGm[offset * this->inchannel * this->out_channel],
                        copyParams_weight, padParams);
            DataCopyPad(feature_ub, featureGm[(int32_t)(indices_idx) + il * tensor_size * this->inchannel],
                        copyParams_feature, padParams);
            for (int32_t mmi = 0; mmi < this->out_channel; mmi++) {
                Mul(result_temp, feature_ub,
                    weight_ub[mmi*this->inchannel], this->inchannel);
                ReduceSum<DTYPE_FEATURE>(result_temp, result_temp,
                                         compute_temp, this->inchannel);
                dst_ub.SetValue(mmi, result_temp.GetValue(0));
            }
            DataCopyPad(outputGm[(int32_t)((address + point_idx) * total_kernel_size + kernel_size_offset)* this->out_channel],
                        dst_ub, copyParams_output);
            indices_pair_ub.SetValue(kernel_size_offset*4, batch_id);
            indices_pair_ub.SetValue(kernel_size_offset*4 + 1, point0);
            indices_pair_ub.SetValue(kernel_size_offset*4 + 2, point1);
            indices_pair_ub.SetValue(kernel_size_offset*4 + 3, point2);
            indices_offset_ub.SetValue(kernel_size_offset, point_offset);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void indicesreshape(int32_t tensor_size, int32_t il)
    {
        DataCopyParams copyParams_indices_stride{1, (uint16_t)(tensor_size * sizeof(DTYPE_FEATURE)), 0, 0};
        DataCopyPad(indices_ub_temp, indicesGm[il * tensor_size],
                    copyParams_indices_stride, padParams);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        Muls(indices_ub_temp, indices_ub_temp, this->total_feature, tensor_size);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        DataCopyPad(indices_ub_temp2,
                    indicesGm[this->indices_number * 3 + il * tensor_size],
                    copyParams_indices_stride, padParams);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        Add(indices_ub_temp, indices_ub_temp2, indices_ub_temp, tensor_size);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        DataCopyPad(indices_ub_temp2,
                    indicesGm[this->indices_number *2 + il * tensor_size],
                    copyParams_indices_stride, padParams);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        Muls(indices_ub_temp2, indices_ub_temp2, outSpatialShape[0], tensor_size);
        Add(indices_ub_temp, indices_ub_temp2, indices_ub_temp, tensor_size);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        DataCopyPad(indices_ub_temp2,
                    indicesGm[this->indices_number * 1 + il * tensor_size],
                    copyParams_indices_stride, padParams);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        Muls(indices_ub_temp2, indices_ub_temp2,
                outSpatialShape[1]*outSpatialShape[0], tensor_size);
        Add(indices_ub_temp, indices_ub_temp2, indices_ub_temp, tensor_size);
        Cast(feature_ub, indices_ub_temp, RoundMode::CAST_CEIL, tensor_size);
    }

    __aicore__ inline void IndicesCompute(int32_t progress, int32_t tensor_size, uint64_t address)
    {
        indices_ub = inQueueIndices.AllocTensor<DTYPE_INDICES>();
        weight_ub = inQueueWeight.AllocTensor<DTYPE_FEATURE>();
        feature_ub = inQueueFeature.AllocTensor<DTYPE_FEATURE>();
        result_temp = tempbuf.Get<DTYPE_FEATURE>();
        indices_ub_temp = tempbuf2.Get<DTYPE_INDICES>();
        indices_ub_temp2 = tempbuf3.Get<DTYPE_INDICES>();
        temp_ub = uint8buf.Get<uint8_t>();
        one_ub = onebuf.Get<DTYPE_FEATURE>();
        zero_ub = zerobuf.Get<DTYPE_FEATURE>();
        dst_ub = dstbuf.Get<DTYPE_FEATURE>();
        indices_pair_ub = indicespairbuf.Get<DTYPE_INDICES>();
        indices_offset_ub = indicesoffsetbuf.Get<DTYPE_INDICES>();
        compute_temp = tempbuf4.Get<DTYPE_FEATURE>();
        if (this->indices_number - tensor_size > 0) {
            PipeBarrier<PIPE_ALL>();
            DataCopyParams copyParams_indices_large{1, (uint16_t)(tensor_size * sizeof(DTYPE_INDICES)), 0, 0};
            DataCopyPad(indices_ub[0], indicesGm[address], copyParams_indices_large, padParams);
            DataCopyPad(indices_ub[this->available_ub_size],
                        indicesGm[address + this->indices_number], copyParams_indices_large, padParams);
            DataCopyPad(indices_ub[this->available_ub_size*2],
                        indicesGm[address + this->indices_number*2], copyParams_indices_large, padParams);
            DataCopyPad(indices_ub[this->available_ub_size*3],
                        indicesGm[address + this->indices_number*3], copyParams_indices_large, padParams);
        } else {
            DataCopyParams copyParams_indices{1, (uint16_t)(tensor_size * 4 * sizeof(DTYPE_INDICES)), 0, 0};
            DataCopyPad(indices_ub, indicesGm[address * 4], copyParams_indices, padParams);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyParams copyParams_feature{1, (uint16_t)(this->inchannel * sizeof(DTYPE_FEATURE)), 0, 0};
        DataCopyParams copyParams_weight{1,
                                        (uint16_t)(this->out_channel * this->inchannel * sizeof(DTYPE_FEATURE)), 0, 0};
        DataCopyParams copyParams_output{1, (uint16_t)(this->out_channel * sizeof(DTYPE_FEATURE)), 0, 0};
        DataCopyParams copyParams_count{1, (uint16_t)(total_kernel_size * 4 * sizeof(DTYPE_FEATURE)), 0, 0};
        DataCopyParams copyParams_count_offset{1, (uint16_t)(total_kernel_size * sizeof(DTYPE_FEATURE)), 0, 0};
        int32_t point[5];
        auto center = (this->K1 * this->K2 * this->K0 - 1) / 2;
        for (int32_t i = 0; i < tensor_size; i++) {
            int32_t batch_id = indices_ub.GetValue(i);
            int32_t indice_z = indices_ub.GetValue(i + tensor_size);
            int32_t indice_y = indices_ub.GetValue(i + tensor_size * 2);
            int32_t indice_x = indices_ub.GetValue(i + tensor_size * 3);
            uint64_t mask = 64;
            if (tensor_size < 64) {
                mask = tensor_size;
            }
            int repeat = (tensor_size + mask - 1) / mask;
            // dup full onenumber tensor
            Duplicate<DTYPE_FEATURE>(one_ub, 1, mask, repeat, 1, 8);
            // dup full zeronumber tensor
            Duplicate<DTYPE_FEATURE>(zero_ub, 0, mask, repeat, 1, 8);
            Duplicate<int32_t>(indices_offset_ub, -1, 27, 1, 1, 8);
            PipeBarrier<PIPE_ALL>();
            // 这里可以把每一列单独读出来，进行vector级别的offset计算
            auto indices_loop = this->indices_number / tensor_size;
            auto indices_tail = this->indices_number - indices_loop * tensor_size;
            for (int32_t il = 0; il < indices_loop; il++) {
                indicesreshape(tensor_size, il);
                for (int32_t iz = 0; iz < this->K0; iz++) {
                    for (int32_t iy = 0; iy < this->K1; iy++) {
                        for (int32_t ix = 0; ix < this->K2; ix++) {
                            auto offset = iz * this->K1 * this->K0 + iy * this->K0 + ix;
                            point[0] = indice_z - iz + K2 / 2;
                            point[1] = indice_y - iy + K1 / 2;
                            point[2] = indice_x - ix + K0 / 2;
                            if (offset != center) {
                                if (point[1] >= 0 && point[1] < outSpatialShape[1] &&
                                    point[2] >= 0 && point[2] < outSpatialShape[2] &&
                                    point[0] >= 0 && point[0] < outSpatialShape[0]) {
                                        int32_t point_offset = point[0] * outSpatialShape[1] * this->outSpatialShape[0] +
                                                               point[1] * this->outSpatialShape[0] + point[2] +
                                                               this->feature_map_size * batch_id;
                                        // 这段for循环可以放在最外层，省去多次的搬运(优化点)
                                        convcompute(tensor_size, offset, il, point_offset, i, address, batch_id,
                                                    point[0], point[1],  point[2], offset);
                                    }
                            }
                        }
                    }
                }
            }
            if (indices_tail > 0) {
                indicesreshape(indices_tail, indices_loop);
                for (int32_t iz = 0; iz < this->K0; iz++) {
                    for (int32_t iy = 0; iy < this->K1; iy++) {
                        for (int32_t ix = 0; ix < this->K2; ix++) {
                            auto offset = iz * this->K1 * this->K0 + iy * this->K0 + ix;
                            point[0] = indice_z - iz + K2 / 2;
                            point[1] = indice_y - iy + K1 / 2;
                            point[2] = indice_x - ix + K0 / 2;
                            if (offset != center) {
                                if (point[1] >= 0 && point[1] < outSpatialShape[1] &&
                                    point[2] >= 0 && point[2] < outSpatialShape[2] &&
                                    point[0] >= 0 && point[0] < outSpatialShape[0]) {
                                        int32_t point_offset = point[0] * outSpatialShape[1] * this->outSpatialShape[0] +
                                                            point[1] * this->outSpatialShape[0] + point[2] +
                                                            this->feature_map_size * batch_id;
                                        // 这段for循环可以放在最外层，省去多次的搬运(优化点)
                                        convcompute(indices_tail, offset, indices_loop, point_offset, i, address, batch_id,
                                                    point[0], point[1], point[2], offset);
                                    }
                            }
                        }
                    }
                }
            }
            int32_t point_offset = indice_z * outSpatialShape[1] * this->outSpatialShape[0] +
                                indice_y * this->outSpatialShape[0] + indice_x +
                                this->feature_map_size * batch_id;
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(weight_ub,
                        weightGm[center * this->inchannel * this->out_channel],
                        copyParams_weight, padParams);
            DataCopyPad(feature_ub, featureGm[(address + i) * this->inchannel],
                        copyParams_feature, padParams);
            for (int32_t mmi = 0; mmi < this->out_channel; mmi++) {
                Mul(result_temp, feature_ub,
                    weight_ub[mmi*this->inchannel], this->inchannel);
                ReduceSum<DTYPE_FEATURE>(result_temp, result_temp,
                                            compute_temp, this->inchannel);
                dst_ub.SetValue(mmi, result_temp.GetValue(0));
            }
            DataCopyPad(outputGm[(int32_t)((address + i) * total_kernel_size + center)* this->out_channel],
                        dst_ub, copyParams_output);
            PipeBarrier<PIPE_ALL>();
            indices_pair_ub.SetValue(center*4, batch_id);
            indices_pair_ub.SetValue(center*4 + 1, point[0]);
            indices_pair_ub.SetValue(center*4 + 2, point[1]);
            indices_pair_ub.SetValue(center*4 + 3, point[2]);
            indices_offset_ub.SetValue(center,  point_offset);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(indices_pairGm[(int32_t)(address + i)* total_kernel_size * 4],
                        indices_pair_ub, copyParams_count);
            DataCopyPad(indices_offsetGm[(int32_t)(address + i)* total_kernel_size],
                        indices_offset_ub, copyParams_output);
            PipeBarrier<PIPE_ALL>();
        }
        inQueueIndices.FreeTensor(indices_ub);
        inQueueWeight.FreeTensor(weight_ub);
        inQueueFeature.FreeTensor(feature_ub);
    }
};

extern "C" __global__ __aicore__ void subm_sparse_conv3d(GM_ADDR feature, GM_ADDR indices,
                                                        GM_ADDR weight,
                                                        GM_ADDR feature_out,
                                                        GM_ADDR indices_offset,
                                                        GM_ADDR indices_pair,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSubmSparseConv3d op;
    TPipe pipe;
    op.Init(feature, indices, weight, feature_out, indices_offset, indices_pair, workspace, &tiling_data, &pipe);
    op.Process();
}