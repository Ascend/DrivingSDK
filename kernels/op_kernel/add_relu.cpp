/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "kernel_operator.h"
using namespace AscendC;

class KernelAddRelu {
public:
    __aicore__ inline KernelAddRelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                AddReluTilingData *tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zeronumber!");
        this->core_used = tiling_data->core_used;
        this->core_data = tiling_data->core_data;
        this->copy_loop = tiling_data->copy_loop;
        this->copy_tail = tiling_data->copy_tail;
        this->last_copy_loop = tiling_data->last_copy_loop;
        this->last_copy_tail = tiling_data->last_copy_tail;
        this->box_number = tiling_data->box_number;
        this->available_ub_size = tiling_data->available_ub_size;

        ptsGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + GetBlockIdx() * this->core_data, this->core_data);
        boxesGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + GetBlockIdx() * this->core_data, this->core_data);
        pipe.InitBuffer(inQueuePTS, this->available_ub_size * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueBOXES, this->available_ub_size * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueOUTPUT, this->available_ub_size * sizeof(DTYPE_X));
    }

    __aicore__ inline void Process()
    {
        uint32_t core_id = GetBlockIdx();
        if (core_id > this->core_used) {
            return;
        }
        if (core_id != (this->core_used -1)) {
            for (int32_t i = 0; i < this->copy_loop; i++) {
                uint64_t address = i * this->available_ub_size;
                Compute(i, this->available_ub_size, address);
            }
            if (this->copy_tail != 0) {
                uint64_t address = this->copy_loop * this->available_ub_size;
                Compute(this->copy_loop, this->copy_tail, address);
            }
        } else {
            for (int32_t i = 0; i < this->last_copy_loop; i++) {
                uint64_t address = i * this->available_ub_size;
                Compute(i, this->available_ub_size, address);
            }
            if (this->last_copy_tail != 0) {
                uint64_t address = this->last_copy_loop * this->available_ub_size;
                Compute(this->last_copy_loop, this->last_copy_tail, address);
            }
        }
    }

private:
    __aicore__ inline void Compute(int32_t progress, int32_t tensor_size, uint64_t address)
    {
        input_x = inQueueBOXES.Get<DTYPE_Y>();
        input_y = inQueuePTS.Get<DTYPE_Y>();
        zLocal = outQueueOUTPUT.Get<DTYPE_Y>();
        DataCopyParams copyParams_out{1, (uint16_t)(tensor_size * sizeof(DTYPE_X)), 0, 0};
        DataCopyParams copyParams_in{1, (uint16_t)(tensor_size* sizeof(DTYPE_X)), 0, 0};
        DataCopyParams copyParams_box{1, (uint16_t)(tensor_size * sizeof(DTYPE_X)), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};
        DataCopyPad(input_x, ptsGm[address], copyParams_in, padParams);
        DataCopyPad(input_y, boxesGm[address], copyParams_box, padParams);
        pipe_barrier(PIPE_ALL);
        AddRelu(zLocal, input_x, input_y, tensor_size);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        DataCopyPad(ptsGm[address], zLocal, copyParams_out);
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> inQueuePTS, inQueueBOXES, outQueueOUTPUT;
    GlobalTensor<DTYPE_X> boxesGm;
    GlobalTensor<DTYPE_X> ptsGm;
    GlobalTensor<DTYPE_X> outputGm;
    uint32_t core_used;
    uint32_t core_data;
    uint32_t copy_loop;
    uint32_t copy_tail;
    uint32_t last_copy_loop;
    uint32_t last_copy_tail;
    uint32_t box_number;
    uint32_t available_ub_size;
    LocalTensor<DTYPE_X> zLocal;
    LocalTensor<DTYPE_X> input_x;
    LocalTensor<DTYPE_X> input_y;
};

extern "C" __global__ __aicore__ void add_relu(GM_ADDR x, GM_ADDR y,
                                               GM_ADDR x_ref,
                                               GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelAddRelu op;
    op.Init(x, y, &tiling_data);
    op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void add_relu_do(uint32_t blockDim, void* l2ctrl,
                 void* stream, uint8_t* boxes, uint8_t* pts, uint8_t* boxes_idx_of_points,
                 uint8_t* workspace, uint8_t* tiling)
{
    add_relu<<<blockDim, l2ctrl, stream>>>(boxes, pts, boxes_idx_of_points, workspace, tiling);
}
#endif