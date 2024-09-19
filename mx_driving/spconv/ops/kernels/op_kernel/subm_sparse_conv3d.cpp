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
    TBuf<TPosition::VECCALC> tempbuf2, tempbuf3, dstbuf, indicesoffsetbuf, indicespairbuf, tempgmbuf;
    TQue<QuePosition::VECOUT, 1> outQueueOUTPUT;
    GlobalTensor<DTYPE_FEATURE> featureGm;
    GlobalTensor<DTYPE_INDICES> indicesGm;
    GlobalTensor<DTYPE_FEATURE> weightGm;
    GlobalTensor<DTYPE_TEMP> tempGm;
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
    LocalTensor<DTYPE_TEMP> temp_gm_ub;
    LocalTensor<DTYPE_FEATURE> dst_ub;
    LocalTensor<int32_t> indices_offset_ub;
    LocalTensor<DTYPE_FEATURE> result_temp;
    DataCopyPadParams padParams{false, 0, 0, 0};
    int32_t total_kernel_size = 27;
    int32_t data_each_block = 8;
    DataCopyParams copyParams_feature;
    DataCopyParams copyParams_weight;
    DataCopyParams copyParams_output;
    DataCopyParams copyParams_count;
    DataCopyParams copyParams_count_offset;
    DataCopyPadParams featurepadParams;
    DataCopyParams copyParams_weight_ub;

    __aicore__ inline void Init(GM_ADDR feature, GM_ADDR indices,
                                GM_ADDR weight,
                                GM_ADDR temp,
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
        this->K0 = (int32_t)(tiling_data->K0);
        this->K1 = (int32_t)(tiling_data->K1);
        this->K2 = (int32_t)(tiling_data->K2);
        this->batch_size = tiling_data->batch_size;
        this->out_channel = tiling_data->outchannel;
        this->outSpatialShape[0] = tiling_data->D;
        this->outSpatialShape[1] = tiling_data->H;
        this->outSpatialShape[2] = tiling_data->W;

        indicesGm.SetGlobalBuffer((__gm__ DTYPE_INDICES*)indices, this->indices_number * 4);
        weightGm.SetGlobalBuffer((__gm__ DTYPE_FEATURE*)weight,
                                 total_kernel_size * this->out_channel * this->inchannel);
        featureGm.SetGlobalBuffer((__gm__ DTYPE_FEATURE*)feature, this->core_data * this->inchannel);
        tempGm.SetGlobalBuffer((__gm__ DTYPE_TEMP*)temp, this->batch_size * this->feature_map_size);
        outputGm.SetGlobalBuffer(
            (__gm__ DTYPE_FEATURE*)feature_out, this->indices_number * this->out_channel);
        indices_offsetGm.SetGlobalBuffer(
            (__gm__ int32_t*)indices_offset, this->indices_number * total_kernel_size);
        indices_pairGm.SetGlobalBuffer(
            (__gm__ int32_t*)indices_pair, this->indices_number);
        int weightnumber = (this->inchannel + data_each_block - 1) / data_each_block * data_each_block;
        int inchannelalign = AlignUp(this->inchannel, data_each_block);
        pipe->InitBuffer(inQueueIndices, 1, this->available_ub_size * 4 * sizeof(DTYPE_FEATURE));
        pipe->InitBuffer(inQueueWeight, 1, this->out_channel * weightnumber * sizeof(DTYPE_FEATURE));
        pipe->InitBuffer(inQueueFeature, 1, inchannelalign * sizeof(DTYPE_FEATURE));
        pipe->InitBuffer(indicespairbuf, total_kernel_size * 4 * sizeof(int32_t));
        pipe->InitBuffer(dstbuf, this->available_ub_size * sizeof(DTYPE_FEATURE));
        pipe->InitBuffer(indicesoffsetbuf, total_kernel_size * sizeof(int32_t));
        pipe->InitBuffer(tempgmbuf, total_kernel_size * sizeof(int32_t));
        copyParams_feature = {1, (uint16_t)(this->inchannel * sizeof(DTYPE_FEATURE)), 0, 0};
        copyParams_weight = {(uint16_t)(this->out_channel),
                                        (uint16_t)(this->inchannel * sizeof(DTYPE_FEATURE)), 0, 0};
        copyParams_output = {1, (uint16_t)(this->out_channel * sizeof(DTYPE_FEATURE)), 0, 0};
        copyParams_count = {1, (uint16_t)(1 * sizeof(DTYPE_FEATURE)), 0, 0};
        copyParams_count_offset = {1, (uint16_t)(total_kernel_size * sizeof(DTYPE_FEATURE)), 0, 0};
    }

    __aicore__ inline void Process()
    {
        uint32_t core_id = GetBlockIdx();
        uint64_t start_address = core_id * this->core_data;
        if (core_id >= this->core_used) {
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
    __aicore__ inline void IndicesCompute(int32_t progress, int32_t tensor_size, uint64_t address)
    {
        indices_ub = inQueueIndices.AllocTensor<DTYPE_INDICES>();
        weight_ub = inQueueWeight.AllocTensor<DTYPE_FEATURE>();
        feature_ub = inQueueFeature.AllocTensor<DTYPE_FEATURE>();
        dst_ub = dstbuf.Get<DTYPE_FEATURE>();
        indices_offset_ub = indicesoffsetbuf.Get<DTYPE_INDICES>();
        temp_gm_ub = tempgmbuf.Get<DTYPE_TEMP>();
        int32_t point[5];
        int inchannelalign = AlignUp(this->inchannel, data_each_block);
        uint8_t inchannelalign_block = static_cast<uint8_t>((inchannelalign + data_each_block - 1) / data_each_block);
        int padnumber = inchannelalign - this->inchannel;
        uint64_t mask = 64;
        if (this->available_ub_size < 64) {
            mask = this->available_ub_size;
        }
        int repeat = (this->available_ub_size + mask - 1) / mask;
        uint64_t mask_inchannel = 64;
        if (inchannelalign < 64) {
            mask_inchannel = inchannelalign;
        }
        int repeat_inchannel = (inchannelalign + mask_inchannel - 1) / mask_inchannel;
        uint8_t inchannelskip_block = static_cast<uint8_t>((inchannelalign + data_each_block - 1) / data_each_block);
        // 计算indices的loop参数
        auto inchannel_ailgn_32b = AlignUp(this->inchannel, 8);
        featurepadParams = {true, 0, (uint8_t)(inchannel_ailgn_32b-this->inchannel), 0};
        DataCopyParams copyParams_indices_large{1, (uint16_t)(tensor_size * sizeof(DTYPE_INDICES)), 0, 0};
        DataCopyParams copyParams_temp_gm_ub{1, (uint16_t)(16 * sizeof(DTYPE_TEMP)), 0, 0};
        if (this->out_channel == 128) {
            copyParams_weight_ub = {(uint16_t)(this->out_channel),
                                    (uint16_t)(inchannelalign * sizeof(DTYPE_TEMP)), 0, 0};
        } else {
            copyParams_weight_ub = {1, (uint16_t)(this->out_channel * inchannelalign * sizeof(DTYPE_TEMP)), 0, 0};
        }
        DataCopyPad(indices_ub[0], indicesGm[address], copyParams_indices_large, padParams);
        DataCopyPad(indices_ub[this->available_ub_size],
                    indicesGm[address + this->indices_number], copyParams_indices_large, padParams);
        DataCopyPad(indices_ub[this->available_ub_size*2],
                    indicesGm[address + this->indices_number*2], copyParams_indices_large, padParams);
        DataCopyPad(indices_ub[this->available_ub_size*3],
                    indicesGm[address + this->indices_number*3], copyParams_indices_large, padParams);
        PipeBarrier<PIPE_ALL>();
        for (int32_t i = 0; i < tensor_size; i++) {
            Duplicate<int32_t>(indices_offset_ub, -1, total_kernel_size, 1, 1, 8);
            int32_t batch_id = indices_ub.GetValue(i);
            int32_t indice_z = indices_ub.GetValue(i + this->available_ub_size);
            int32_t indice_y = indices_ub.GetValue(i + this->available_ub_size * 2);
            int32_t indice_x = indices_ub.GetValue(i + this->available_ub_size * 3);
            DataCopyPad(feature_ub, featureGm[(address +i) * this->inchannel],
                        copyParams_feature, featurepadParams);
            for (int32_t iz = 0; iz < this->K0; iz++) {
                for (int32_t iy = 0; iy < this->K1; iy++) {
                    for (int32_t ix = 0; ix < this->K2; ix++) {
                        auto offset = iz * this->K1 * this->K0 + iy * this->K0 + ix;
                        point[0] = indice_z - iz + K2 / 2;
                        point[1] = indice_y - iy + K1 / 2;
                        point[2] = indice_x - ix + K0 / 2;
                        if (point[1] >= 0 && point[1] < outSpatialShape[1] &&
                            point[2] >= 0 && point[2] < outSpatialShape[2] &&
                            point[0] >= 0 && point[0] < outSpatialShape[0]) {
                                int32_t point_offset = point[0] * outSpatialShape[1] *
                                                        this->outSpatialShape[2] +
                                                        point[1] * this->outSpatialShape[2] + point[2] +
                                                        this->feature_map_size * batch_id;
                                DataCopyPad(temp_gm_ub, tempGm[point_offset],
                                            copyParams_temp_gm_ub, padParams);
                                auto feature_address = temp_gm_ub.GetValue(0);
                                if (feature_address != -1) {
                                    DataCopyPad(weight_ub,
                                                weightGm[offset * inchannelalign * this->out_channel],
                                                copyParams_weight_ub, padParams);
                                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                                    if (repeat_inchannel == 1) {
                                        Mul(weight_ub, feature_ub, weight_ub, mask_inchannel,
                                            this->out_channel,
                                            { 1, 1, 1, inchannelalign_block, 0, inchannelalign_block});
                                        WholeReduceSum(dst_ub, weight_ub, mask_inchannel,
                                                       this->out_channel, 1, 1, inchannelalign_block);
                                        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                                        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                                        SetAtomicAdd<DTYPE_FEATURE>();
                                        DataCopyPad(outputGm[(int32_t)(feature_address) * this->out_channel],
                                                    dst_ub, copyParams_output);
                                        SetAtomicNone();
                                        PipeBarrier<PIPE_ALL>();
                                    } else {
                                        for (int re = 0; re < repeat_inchannel; re++) {
                                            Mul(weight_ub[re * mask_inchannel], feature_ub[re * mask_inchannel],
                                                weight_ub[re * mask_inchannel], mask_inchannel,
                                                this->out_channel, { 1, 1, 1, inchannelskip_block,
                                                0, inchannelskip_block});
                                            WholeReduceSum(dst_ub, weight_ub[re * mask_inchannel],
                                                           mask_inchannel, this->out_channel,
                                                           1, 1, inchannelskip_block);
                                            PipeBarrier<PIPE_ALL>();
                                            SetAtomicAdd<DTYPE_FEATURE>();
                                            DataCopyPad(outputGm[(int32_t)(feature_address) * this->out_channel],
                                                        dst_ub, copyParams_output);
                                            SetAtomicNone();
                                            PipeBarrier<PIPE_ALL>();
                                        }
                                        PipeBarrier<PIPE_ALL>();
                                    }
                                    indices_offset_ub.SetValue(offset,  point_offset);
                                }
                        }
                    }
                }
            }
            DataCopyPad(indices_offsetGm[(int32_t)(address + i)* total_kernel_size],
                        indices_offset_ub, copyParams_count_offset);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
        inQueueIndices.FreeTensor(indices_ub);
        inQueueWeight.FreeTensor(weight_ub);
        inQueueFeature.FreeTensor(feature_ub);
    }
};

extern "C" __global__ __aicore__ void subm_sparse_conv3d(GM_ADDR feature, GM_ADDR indices,
                                                        GM_ADDR weight,
                                                        GM_ADDR temp,
                                                        GM_ADDR feature_out,
                                                        GM_ADDR indices_offset,
                                                        GM_ADDR indices_pair,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSubmSparseConv3d op;
    TPipe pipe;
    op.Init(feature, indices, weight, temp, feature_out, indices_offset, indices_pair, workspace, &tiling_data, &pipe);
    op.Process();
}