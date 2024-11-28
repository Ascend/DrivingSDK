/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 * This sample is a very basic sample that implements vector add on Ascend plaform.
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BUF_SIZE_UNIT = 32;
constexpr int32_t NUM_SIZE = 1;

class KernelGatherNms3dMask {
public:
    __aicore__ inline KernelGatherNms3dMask() {}
    __aicore__ inline void Init(GM_ADDR mask, GM_ADDR keep, GM_ADDR num_out, GatherNms3dMaskTilingData *tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        box_num = tiling_data->box_num;
        mask_num = tiling_data->mask_num;

        int32_t assign_num = (box_num * sizeof(int16_t) + BUF_SIZE_UNIT - 1) / BUF_SIZE_UNIT;
        mask_size = assign_num * BUF_SIZE_UNIT / sizeof(int16_t);

        maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t * > (mask), box_num * mask_num);
        keepGm.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t * > (keep), box_num);
        numOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t * > (num_out), NUM_SIZE);

        pipe.InitBuffer(inQueueMask, BUFFER_NUM, mask_size * sizeof(int16_t));
        pipe.InitBuffer(maskBuf, mask_size * sizeof(int16_t));
        pipe.InitBuffer(keepBuf, mask_size * sizeof(int16_t));
        pipe.InitBuffer(numOutBuf, BUF_SIZE_UNIT);
    }
    __aicore__ inline void Process()
    {
        InitCmp();
        for (int32_t i = 0; i < box_num; ++i) {
            if (maskTemp.GetValue(i) == 1) {
                SaveKeep(i);
                CopyIn(i);
                Compute(i);
            }
        }
        EndCmp();
    }

private:
    __aicore__ inline void InitCmp()
    {
        maskTemp = maskBuf.Get<int16_t>();
        keepTemp = keepBuf.Get<int16_t>();
        Duplicate(maskTemp, static_cast<int16_t>(1), mask_size);
        Duplicate(keepTemp, static_cast<int16_t>(0), mask_size);
        DataCopyParams copyParams{1, static_cast<uint16_t>(box_num * sizeof(int16_t)), 0, 0};
        DataCopyPadParams padParams{false, 0, 2, 0};
        DataCopyPad(maskTemp, maskGm, copyParams, padParams);
    }
    __aicore__ inline void CopyIn(int32_t idx)
    {
        LocalTensor<int16_t> maskLocal = inQueueMask.AllocTensor<int16_t>();
        Duplicate(maskLocal, static_cast<int16_t>(1), mask_size);
        DataCopyParams copyParams{1, static_cast<uint16_t>(box_num * sizeof(int16_t)), 0, 0};
        DataCopyPadParams padParams{false, 0, 0, 2};
        DataCopyPad(maskLocal, maskGm[idx * mask_num], copyParams, padParams);
        inQueueMask.EnQue(maskLocal);
    }
    __aicore__ inline void Compute(int32_t idx)
    {
        LocalTensor<int16_t> maskLocal = inQueueMask.DeQue<int16_t>();
        maskTemp = maskLocal & maskTemp;
        pipe_barrier(PIPE_ALL);
        inQueueMask.FreeTensor(maskLocal);
    }
    __aicore__ inline void SaveKeep(int32_t idx)
    {
        keepTemp.SetValue(keep_num, idx);
        keep_num = keep_num + 1;
    }
    __aicore__ inline void EndCmp()
    {
        DataCopyParams copyMaskParams{1, static_cast<uint16_t>(box_num * sizeof(int16_t)), 0, 0};
        DataCopyPad(keepGm, keepTemp, copyMaskParams);
        LocalTensor<int16_t> numOutLocal = numOutBuf.Get<int16_t>();
        numOutLocal.SetValue(0, keep_num);
        DataCopyParams copyNumParams{1, static_cast<uint16_t>(NUM_SIZE * sizeof(int16_t)), 0, 0};
        DataCopyPad(numOutGm, numOutLocal, copyNumParams);
    }

private:
    TPipe pipe;
    TQue <QuePosition::VECIN, BUFFER_NUM> inQueueMask;

    GlobalTensor <int16_t> maskGm;
    GlobalTensor <int16_t> keepGm;
    GlobalTensor <int16_t> numOutGm;

    LocalTensor <int16_t> maskTemp;
    LocalTensor <int16_t> keepTemp;

    TBuf <TPosition::VECCALC> maskBuf, keepBuf, numOutBuf;

    uint32_t box_num;
    uint32_t mask_num;
    uint32_t mask_size;
    uint32_t keep_num = 0;
};

extern "C" __global__ __aicore__
void gather_nms3d_mask(GM_ADDR mask, GM_ADDR keep, GM_ADDR num_out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGatherNms3dMask op;
    op.Init(mask, keep, num_out, &tiling_data);
    op.Process();
}
