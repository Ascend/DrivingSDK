/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "kernel_operator.h"
using namespace AscendC;

class KernelAddRelu {
public:
    __aicore__ inline KernelAddRelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                AddReluTilingData *tiling_data, TPipe *pipe)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zeronumber!");

        pipe_ = pipe;

        this->core_used = tiling_data->core_used;
        this->core_data = tiling_data->core_data;
        this->copy_loop = tiling_data->copy_loop;
        this->copy_tail = tiling_data->copy_tail;
        this->last_copy_loop = tiling_data->last_copy_loop;
        this->last_copy_tail = tiling_data->last_copy_tail;
        this->box_number = tiling_data->box_number;
        this->available_ub_size = tiling_data->available_ub_size;

        ptsGm.SetGlobalBuffer((__gm__ DTYPE_X*) x + GetBlockIdx() * this->core_data, this->core_data);
        boxesGm.SetGlobalBuffer((__gm__ DTYPE_Y*) y + GetBlockIdx() * this->core_data, this->core_data);
        pipe_->InitBuffer(inQueuePTS, this->available_ub_size * sizeof(DTYPE_X));
        pipe_->InitBuffer(inQueueBOXES, this->available_ub_size * sizeof(DTYPE_X));
        pipe_->InitBuffer(outQueueOUTPUT, this->available_ub_size * sizeof(DTYPE_X));
    }

    __aicore__ inline void Process()
    {
        SetFlag<HardEvent::V_MTE2>(0);
        SetFlag<HardEvent::MTE3_V>(0);
        
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

        WaitFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }

private:
    template<typename T>
    static __aicore__ inline void AddReluVf(__local_mem__ T* dstPtr, __local_mem__ T* src0Ptr, __local_mem__ T* src1Ptr, uint32_t count, uint16_t oneRepeatSize, uint16_t repeatTimes)
    {
        __VEC_SCOPE__ {
            MicroAPI::RegTensor<T> vSrcReg0;
            MicroAPI::RegTensor<T> vSrcReg1;
            MicroAPI::RegTensor<T> vDstReg0;

            MicroAPI::MaskReg maskReg;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                maskReg = MicroAPI::UpdateMask<T>(count);

                MicroAPI::DataCopy(vSrcReg0, src0Ptr + i * oneRepeatSize);
                MicroAPI::DataCopy(vSrcReg1, src1Ptr + i * oneRepeatSize);
                MicroAPI::Add(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
                MicroAPI::Relu(vDstReg0, vDstReg0, maskReg);
                MicroAPI::DataCopy(dstPtr + i * oneRepeatSize, vDstReg0, maskReg);
            }
        }
    }

    __aicore__ inline void Compute(int32_t progress, int32_t tensor_size, uint64_t address)
    {
        
        uint16_t oneRepeatSize = GetVecLen() / sizeof(DTYPE_X);
        int32_t tensor_size_aligned = AlignUp(tensor_size, oneRepeatSize);
        uint16_t repeatTimes = CeilDivision(tensor_size_aligned, oneRepeatSize);

        input_x = inQueueBOXES.Get<DTYPE_Y>();
        input_y = inQueuePTS.Get<DTYPE_Y>();
        zLocal = outQueueOUTPUT.Get<DTYPE_Y>();
        DataCopyParams copyParams_out{1, (uint16_t)(tensor_size * sizeof(DTYPE_X)), 0, 0};
        DataCopyParams copyParams_in{1, (uint16_t)(tensor_size* sizeof(DTYPE_X)), 0, 0};
        DataCopyParams copyParams_box{1, (uint16_t)(tensor_size * sizeof(DTYPE_X)), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};

        WaitFlag<HardEvent::V_MTE2>(0);
        DataCopyPad(input_x, ptsGm[address], copyParams_in, padParams);
        DataCopyPad(input_y, boxesGm[address], copyParams_box, padParams);
        
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        __local_mem__ DTYPE_X* zPtr = (__local_mem__ DTYPE_X*) zLocal.GetPhyAddr();
        __local_mem__ DTYPE_X* xPtr = (__local_mem__ DTYPE_X*) input_x.GetPhyAddr();
        __local_mem__ DTYPE_X* yPtr = (__local_mem__ DTYPE_X*) input_y.GetPhyAddr();
        WaitFlag<HardEvent::MTE3_V>(0);
        AddReluVf<DTYPE_X>(zPtr, xPtr, yPtr, tensor_size_aligned, oneRepeatSize, repeatTimes);

        SetFlag<HardEvent::V_MTE2>(0);
        
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        DataCopyPad(ptsGm[address], zLocal, copyParams_out);
        SetFlag<HardEvent::MTE3_V>(0);
    }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> inQueuePTS, inQueueBOXES, outQueueOUTPUT;
    GlobalTensor<DTYPE_X> boxesGm;
    GlobalTensor<DTYPE_X> ptsGm;
    GlobalTensor<DTYPE_X> outputGm;
    uint64_t core_used;
    uint64_t core_data;
    uint64_t copy_loop;
    uint64_t copy_tail;
    uint64_t last_copy_loop;
    uint64_t last_copy_tail;
    uint64_t box_number;
    uint64_t available_ub_size;
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
    TPipe pipe;
    op.Init(x, y, &tiling_data, &pipe);
    op.Process();
}
