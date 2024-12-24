/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;  
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

class KernelSubmSparseConv3dGrad {
public:
    __aicore__ inline KernelSubmSparseConv3dGrad() {}

    TQue<QuePosition::VECIN, 1> inQueueOffset, inQueueValid, inQueueGrad;
    GlobalTensor<DTYPE_OUTIDX_OFFSET> outidxOffsetGm;
    GlobalTensor<DTYPE_OUTIDX_OFFSET> validIndicesGm;
    GlobalTensor<DTYPE_GRAD_OUT_FEATURES> GradOutGm;
    GlobalTensor<DTYPE_GRAD_OUT_FEATURES> outputGm;
    uint64_t core_used;
    uint64_t core_data;
    uint64_t copy_loop;
    uint64_t copy_tail;
    uint64_t last_copy_loop;
    uint64_t last_copy_tail;
    uint64_t inchannel;
    uint64_t outchannel;
    uint64_t indices_number;
    uint64_t available_ub_size;
    uint64_t K0;
    uint64_t K1;
    uint64_t K2;
    uint64_t valid_number;
    LocalTensor<DTYPE_OUTIDX_OFFSET> indices_ub;
    LocalTensor<DTYPE_OUTIDX_OFFSET> offset_ub;
    LocalTensor<DTYPE_GRAD_OUT_FEATURES> grad_ub;
    DataCopyPadParams padParams = {false, 0 , 0, 0};
    int32_t total_kernel_size;
    int32_t data_each_block = 8;
    DataCopyParams copyParams_offset;
    DataCopyParams copyParams_valid;
    DataCopyParams copyParams_grad;

    __aicore__ inline void Init(GM_ADDR ouidx_offset, GM_ADDR valid_indices,
                                GM_ADDR grad_out_features,
                                GM_ADDR grad_out_features_iml2col,
                                GM_ADDR workspace,
                                SubmSparseConv3dGradTilingData *tiling_data, TPipe* pipe)
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
        this->valid_number = tiling_data->valid_number;
        this->available_ub_size = tiling_data->available_ub_size;
        this->K0 = (int32_t)(tiling_data->K0);
        this->K1 = (int32_t)(tiling_data->K1);
        this->K2 = (int32_t)(tiling_data->K2);
        this->outchannel = tiling_data->outchannel;
        total_kernel_size = this->K0 * this->K1 * this->K2;
        outidxOffsetGm.SetGlobalBuffer((__gm__ DTYPE_OUTIDX_OFFSET*)ouidx_offset, this->valid_number);
        validIndicesGm.SetGlobalBuffer((__gm__ DTYPE_OUTIDX_OFFSET*)valid_indices, this->valid_number);
        GradOutGm.SetGlobalBuffer((__gm__ DTYPE_GRAD_OUT_FEATURES*)grad_out_features, this->indices_number * this->outchannel);
        outputGm.SetGlobalBuffer(
            (__gm__ DTYPE_GRAD_OUT_FEATURES*)grad_out_features_iml2col, this->indices_number * total_kernel_size * this->inchannel);
        pipe->InitBuffer(inQueueOffset, 1, this->available_ub_size * sizeof(DTYPE_OUTIDX_OFFSET));
        pipe->InitBuffer(inQueueValid, 1, this->available_ub_size * sizeof(DTYPE_OUTIDX_OFFSET));
        pipe->InitBuffer(inQueueGrad, 1, this->outchannel * total_kernel_size * sizeof(DTYPE_GRAD_OUT_FEATURES));
        copyParams_offset = {1, (uint16_t)(this->available_ub_size * sizeof(DTYPE_OUTIDX_OFFSET)), 0, 0};
        copyParams_grad = {1, (uint16_t)(this->outchannel * sizeof(DTYPE_GRAD_OUT_FEATURES)), 0, 0};
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
                IndicesComputeAlgin(core_id, this->available_ub_size, address);
            }
            if (this->copy_tail != 0) {
                uint64_t address = start_address + this->copy_loop * this->available_ub_size;
                IndicesCompute(core_id, this->copy_tail, address);
            }
        } else {
            for (uint32_t i = 0; i < this->last_copy_loop; i++) {
                uint64_t address = start_address + i * this->available_ub_size;
                IndicesComputeAlgin(core_id, this->available_ub_size, address);
            }
            if (this->last_copy_tail != 0) {
                uint64_t address = start_address + this->last_copy_loop * this->available_ub_size;
                IndicesCompute(core_id, this->last_copy_tail, address);
            }
        }
    }

private:
    __aicore__ inline void IndicesComputeAlgin(int32_t progress, int32_t tensor_size, uint64_t address)
    {
        offset_ub = inQueueOffset.AllocTensor<DTYPE_OUTIDX_OFFSET>();
        indices_ub = inQueueValid.AllocTensor<DTYPE_OUTIDX_OFFSET>();
        grad_ub = inQueueGrad.AllocTensor<DTYPE_GRAD_OUT_FEATURES>();
        // 计算indices的loop参数
        copyParams_valid = {1, (uint16_t)(tensor_size * sizeof(DTYPE_OUTIDX_OFFSET)), 0, 0};
        auto outchannel_ailgn_32b = AlignUp(this->outchannel, 8);
        DataCopyPadParams gradpadParams = {true, 0, (uint8_t)(outchannel_ailgn_32b-this->outchannel), 0};
        DataCopyParams copyParams_out = {(uint16_t)(total_kernel_size),
                                         (uint16_t)(this->outchannel * sizeof(DTYPE_OUTIDX_OFFSET)), 0, 0};
        DataCopyPad(indices_ub, validIndicesGm[address], copyParams_valid, padParams);
        DataCopyPad(offset_ub, outidxOffsetGm[address], copyParams_valid, padParams);
        PipeBarrier<PIPE_ALL>();
        for (int32_t i = 0; i < tensor_size/64; i++) {
            for (int32_t j = 0; j < 64; j++) {
                int64_t feature_idx = indices_ub.GetValue(i * 64 + j) / total_kernel_size;
                DataCopyPad(grad_ub[j * outchannel_ailgn_32b], GradOutGm[feature_idx * this->outchannel],
                            copyParams_grad, gradpadParams);
            }
            PipeBarrier<PIPE_ALL>();
            for (int32_t j = 0; j< 64; j++) {
                int64_t valid_idx = indices_ub.GetValue(i * 64 + j) % total_kernel_size;
                int64_t offset_idx = offset_ub.GetValue(i * 64 + j);
                DataCopyPad(outputGm[(int32_t)(offset_idx * total_kernel_size * this->outchannel + valid_idx* this->outchannel)],
                            grad_ub[j * outchannel_ailgn_32b], copyParams_grad);
            }  
            PipeBarrier<PIPE_ALL>();
        }
        inQueueOffset.FreeTensor(offset_ub);
        inQueueValid.FreeTensor(indices_ub);
        inQueueGrad.FreeTensor(grad_ub);
    }
    __aicore__ inline void IndicesCompute(int32_t progress, int32_t tensor_size, uint64_t address)
    {
        offset_ub = inQueueOffset.AllocTensor<DTYPE_OUTIDX_OFFSET>();
        indices_ub = inQueueValid.AllocTensor<DTYPE_OUTIDX_OFFSET>();
        grad_ub = inQueueGrad.AllocTensor<DTYPE_GRAD_OUT_FEATURES>();
        // 计算indices的loop参数
        copyParams_valid = {1,(uint16_t)(tensor_size * sizeof(DTYPE_OUTIDX_OFFSET)), 0, 0};
        DataCopyPad(indices_ub, validIndicesGm[address], copyParams_valid, padParams);
        DataCopyPad(offset_ub, outidxOffsetGm[address], copyParams_valid, padParams);
        PipeBarrier<PIPE_ALL>();
        for (int32_t i = 0; i < tensor_size; i++) {  
            int64_t feature_idx = indices_ub.GetValue(i) / total_kernel_size;
            int64_t valid_idx = indices_ub.GetValue(i) % total_kernel_size;
            int64_t offset_idx = offset_ub.GetValue(i);
            DataCopyPad(grad_ub, GradOutGm[feature_idx * this->outchannel], copyParams_grad, padParams);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            DataCopyPad(outputGm[(int32_t)(offset_idx * total_kernel_size * this->outchannel + valid_idx* this->outchannel)],
                        grad_ub, copyParams_grad);
            PipeBarrier<PIPE_ALL>();
        }       
        inQueueOffset.FreeTensor(offset_ub);
        inQueueValid.FreeTensor(indices_ub);
        inQueueGrad.FreeTensor(grad_ub);
    }
};

extern "C" __global__ __aicore__ void subm_sparse_conv3d_grad(GM_ADDR ouidx_offset, GM_ADDR valid_indices,
                                                        GM_ADDR grad_out_features,
                                                        GM_ADDR grad_out_features_iml2col,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSubmSparseConv3dGrad op;
    TPipe pipe;
    op.Init(ouidx_offset, valid_indices, grad_out_features, grad_out_features_iml2col, workspace, &tiling_data, &pipe);
    op.Process();
}