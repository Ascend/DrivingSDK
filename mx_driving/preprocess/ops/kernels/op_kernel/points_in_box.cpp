/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 *
 * Function : z = x + y
 * This sample is a very basic sample that implements vector Add on Ascend plaform.
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue

class KernelPointsInBox {
public:
    __aicore__ inline KernelPointsInBox() {}

    __aicore__ inline void Init(GM_ADDR boxes, GM_ADDR pts, GM_ADDR boxes_idx_of_points,
                                PointsInBoxTilingData *tiling_data)
    {
        this->core_used = tiling_data->core_used;
        this->core_data = tiling_data->core_data;
        this->copy_loop = tiling_data->copy_loop;
        this->copy_tail = tiling_data->copy_tail;
        this->last_copy_loop = tiling_data->last_copy_loop;
        this->last_copy_tail = tiling_data->last_copy_tail;
        this->batch = tiling_data->batch;
        this->npoints = tiling_data->npoints;
        this->box_number = tiling_data->box_number;
        this->available_ub_size = tiling_data->available_ub_size;

        ptsGm.SetGlobalBuffer((__gm__ DTYPE_PTS*)pts + GetBlockIdx() * this->core_data * 3, this->core_data * 3);
        boxesGm.SetGlobalBuffer((__gm__ DTYPE_PTS*)boxes, this->box_number * 7);
        outputGm.SetGlobalBuffer(
            (__gm__ DTYPE_BOXES_IDX_OF_POINTS*)boxes_idx_of_points + GetBlockIdx() * this->core_data, this->core_data);
        pipe.InitBuffer(inQueuePTS, BUFFER_NUM, this->available_ub_size * 3 * sizeof(DTYPE_PTS));
        pipe.InitBuffer(inQueueBOXES, BUFFER_NUM, this->available_ub_size * 7 * sizeof(DTYPE_PTS));
        pipe.InitBuffer(outQueueOUTPUT, BUFFER_NUM, this->available_ub_size * sizeof(DTYPE_BOXES_IDX_OF_POINTS));
        pipe.InitBuffer(shiftxque, this->available_ub_size * sizeof(DTYPE_PTS));
        pipe.InitBuffer(shiftyque, this->available_ub_size * sizeof(DTYPE_PTS));
        pipe.InitBuffer(cosaque, this->available_ub_size * sizeof(DTYPE_PTS));
        pipe.InitBuffer(sinaque, this->available_ub_size * sizeof(DTYPE_PTS));
        pipe.InitBuffer(xlocalque, this->available_ub_size * sizeof(DTYPE_PTS));
        pipe.InitBuffer(ylocalque, this->available_ub_size * sizeof(DTYPE_PTS));
        pipe.InitBuffer(tempque, this->available_ub_size * sizeof(DTYPE_PTS));
        pipe.InitBuffer(uint8que, this->available_ub_size * sizeof(DTYPE_PTS));
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
    __aicore__ inline void ComputeBoxs(int32_t i)
    {
        float oneminsnumber = -1;
        float halfnumber =  0.5;
        float zeronumber =  0;
        float onenumber =  1;
        uint64_t mask = 64;
        auto x = pointLocal.GetValue(i * 3);
        auto y = pointLocal.GetValue(i * 3 + 1);
        auto z = pointLocal.GetValue(i * 3 + 2);
        int repeat = (this->box_number + mask - 1) / mask;
        BinaryRepeatParams repeatParams = { 1, 1, 1, 8, 8, 8 };
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

        // shift_x = x - boxes_ub[ :, 0]
        Muls(shiftx, boxesLocal_cx, oneminsnumber, mask, repeat, { 1, 1, 8, 8 });
        Adds(shiftx, shiftx, x, mask, repeat, { 1, 1, 8, 8 });

        // shift_y = y - boxes_ub[ :, 1]
        Muls(shifty, boxesLocal_cy, oneminsnumber, mask, repeat, { 1, 1, 8, 8 });
        Adds(shifty, shifty, y, mask, repeat, { 1, 1, 8, 8 });

        // cosa = Cos(-boxes_ub[ :, 6])
        Muls(temp, boxesLocal_rz, oneminsnumber, mask, repeat, { 1, 1, 8, 8 });
        Cos<DTYPE_BOXES, false>(cosa, temp, uint8temp, this->box_number);

        // sina = Sin(-boxes_ub[ :, 6])
        Muls(temp, boxesLocal_rz, oneminsnumber, mask, repeat, { 1, 1, 8, 8 });
        Sin<DTYPE_BOXES, false>(sina, temp, uint8temp, this->box_number);

        // local_x = shift_x * cosa + shift_y * (-sina)
        Mul(temp, shiftx, cosa, mask, repeat, {1, 1, 1, 8, 8, 8 });
        Duplicate<DTYPE_BOXES>(xlocal, zeronumber, this->box_number);
        Add(xlocal, xlocal, temp, mask, repeat, {1, 1, 1, 8, 8, 8 });
        Muls(temp, sina, oneminsnumber, mask, repeat, { 1, 1, 8, 8 });
        Mul(temp, shifty, temp, mask, repeat, {1, 1, 1, 8, 8, 8 });
        Add(xlocal, xlocal, temp, mask, repeat, {1, 1, 1, 8, 8, 8 });

        // local_y = shift_x * sina + shift_y * cosa
        Mul(temp, shiftx, sina, mask, repeat, {1, 1, 1, 8, 8, 8 });
        Mul(sina, shifty, cosa, mask, repeat, {1, 1, 1, 8, 8, 8 });
        Add(ylocal, sina, temp,  mask, repeat, {1, 1, 1, 8, 8, 8 });

        Abs(xlocal, xlocal, mask, repeat, { 1, 1, 8, 8 });
        pipe_barrier(PIPE_V);
        Abs(ylocal, ylocal, mask, repeat, { 1, 1, 8, 8 });

        // dup full zeronumber tensor
        Duplicate<DTYPE_BOXES>(sina, zeronumber, mask, repeat, 1, 8);
        // dup full onenumber tensor
        Duplicate<DTYPE_BOXES>(temp, onenumber, mask, repeat, 1, 8);

        // x_size + 1e-5 shiftx
        Muls(shiftx, boxesLocal_dx, halfnumber, mask, repeat, { 1, 1, 8, 8 });

        // y_size + 1e-5 shifty
        Muls(shifty, boxesLocal_dy, halfnumber, mask, repeat, { 1, 1, 8, 8 });

        // cmp_1 = Abs(local_x) < x_size + 1e-5
        pipe_barrier(PIPE_ALL);
        uint8temp = xlocal <= shiftx;
        Duplicate<DTYPE_BOXES>(xlocal, zeronumber, mask, repeat, 1, 8);
        Select(xlocal, uint8temp, temp, sina,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, mask, repeat, repeatParams);

        // cmp_2 = Abs(local_y) < y_size+ 1e-5
        pipe_barrier(PIPE_ALL);
        uint8temp = ylocal <= shifty;
        Duplicate<DTYPE_BOXES>(ylocal, zeronumber, mask, repeat, 1, 8);
        Select(ylocal, uint8temp, temp, sina,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, mask, repeat, repeatParams);

        // zlocal = z-cz  sina
        Muls(sina, boxesLocal_cz, oneminsnumber, mask, repeat, { 1, 1, 8, 8 });
        Adds(sina, sina, z, mask, repeat, { 1, 1, 8, 8 });
        Abs(sina, sina, mask, repeat, { 1, 1, 8, 8 });
        pipe_barrier(PIPE_ALL);

        // z_size + 1e-5 cosa
        Muls(cosa, boxesLocal_dz, halfnumber, mask, repeat, { 1, 1, 8, 8 });
        pipe_barrier(PIPE_ALL);

        // dup full zeronumber tensor
        Duplicate<DTYPE_BOXES>(shifty, zeronumber, mask, repeat, 1, 8);
        // dup full onenumber tensor
        Duplicate<DTYPE_BOXES>(temp, onenumber, mask, repeat, 1, 8);

        // cmp_3 = Abs(zlocal) < z_size
        pipe_barrier(PIPE_ALL);
        uint8temp = sina <= cosa;
        Duplicate<DTYPE_BOXES>(sina, zeronumber, mask, repeat, 1, 8);
        pipe_barrier(PIPE_ALL);
        Select(sina, uint8temp, temp, shifty,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, mask, repeat, repeatParams);
        pipe_barrier(PIPE_ALL);
        
        // select which point is in box
        Add(ylocal, ylocal, sina, this->box_number);
        Add(ylocal, ylocal, xlocal, this->box_number);
        ReduceMax<float>(xlocal, ylocal, sina, this->box_number, true);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        // if sumnumber is equal 3 means the point is in this box
        if (xlocal.GetValue(0) == 3) {
            DTYPE_BOXES a = xlocal.GetValue(1);
            zLocal.SetValue(i, static_cast<DTYPE_BOXES_IDX_OF_POINTS>(*reinterpret_cast<int32_t *>(&a)));
            pipe_barrier(PIPE_ALL);
        }
    }
    __aicore__ inline void Compute(int32_t progress, int32_t tensor_size, uint64_t address)
    {
        float oneminsnumber = -1;
        boxesLocal_cx = inQueueBOXES.AllocTensor<DTYPE_BOXES>();
        boxesLocal_cy = boxesLocal_cx[this->available_ub_size];
        boxesLocal_cz = boxesLocal_cx[this->available_ub_size * 2];
        boxesLocal_dx = boxesLocal_cx[this->available_ub_size * 3];
        boxesLocal_dy = boxesLocal_cx[this->available_ub_size * 4];
        boxesLocal_dz = boxesLocal_cx[this->available_ub_size * 5];
        boxesLocal_rz = boxesLocal_cx[this->available_ub_size * 6];
        pointLocal = inQueuePTS.AllocTensor<DTYPE_PTS>();
        zLocal = outQueueOUTPUT.AllocTensor<DTYPE_BOXES_IDX_OF_POINTS>();
        shiftx = shiftxque.Get<DTYPE_BOXES>();
        shifty = shiftyque.Get<DTYPE_BOXES>();
        cosa = cosaque.Get<DTYPE_BOXES>();
        sina = sinaque.Get<DTYPE_BOXES>();
        xlocal = xlocalque.Get<DTYPE_BOXES>();
        ylocal = ylocalque.Get<DTYPE_BOXES>();
        temp = tempque.Get<DTYPE_BOXES>();
        uint8temp = uint8que.Get<uint8_t>();
        DataCopyParams copyParams_out{1, (uint16_t)(tensor_size * sizeof(DTYPE_BOXES_IDX_OF_POINTS)), 0, 0};
        DataCopyParams copyParams_in{1, (uint16_t)(tensor_size * 3 * sizeof(DTYPE_BOXES)), 0, 0};
        DataCopyParams copyParams_box{1, (uint16_t)(this->box_number * sizeof(DTYPE_BOXES)), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};
        // move points to localtensor
        DataCopyPad(pointLocal, ptsGm[address * 3], copyParams_in, padParams);
        Duplicate<DTYPE_BOXES_IDX_OF_POINTS>(zLocal, oneminsnumber, tensor_size);
        // move boxes number to each localtensor
        DataCopyPad(boxesLocal_cx, boxesGm, copyParams_box, padParams);
        DataCopyPad(boxesLocal_cy, boxesGm[this->box_number], copyParams_box, padParams);
        DataCopyPad(boxesLocal_cz, boxesGm[this->box_number*2], copyParams_box, padParams);
        DataCopyPad(boxesLocal_dx, boxesGm[this->box_number*3], copyParams_box, padParams);
        DataCopyPad(boxesLocal_dy, boxesGm[this->box_number*4], copyParams_box, padParams);
        DataCopyPad(boxesLocal_dz, boxesGm[this->box_number*5], copyParams_box, padParams);
        DataCopyPad(boxesLocal_rz, boxesGm[this->box_number*6], copyParams_box, padParams);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        for (int32_t i = 0; i < tensor_size; i++) {
            if (zLocal.GetValue(i) == -1) {
                ComputeBoxs(i);
            }
        }
        pipe_barrier(PIPE_ALL);
        DataCopyPad(outputGm[address], zLocal, copyParams_out);
        inQueuePTS.FreeTensor(boxesLocal_cx);
        inQueueBOXES.FreeTensor(pointLocal);
        outQueueOUTPUT.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueuePTS, inQueueBOXES;
    TBuf<TPosition::VECCALC> shiftxque, shiftyque, cosaque, sinaque, xlocalque, ylocalque, tempque, uint8que;
    TQue<QuePosition::VECOUT, 1> outQueueOUTPUT;
    GlobalTensor<DTYPE_BOXES> boxesGm;
    GlobalTensor<DTYPE_PTS> ptsGm;
    GlobalTensor<DTYPE_BOXES_IDX_OF_POINTS> outputGm;
    uint32_t core_used;
    uint32_t core_data;
    uint32_t copy_loop;
    uint32_t copy_tail;
    uint32_t last_copy_loop;
    uint32_t last_copy_tail;
    uint32_t batch;
    uint32_t npoints;
    uint32_t box_number;
    uint32_t available_ub_size;
    LocalTensor<DTYPE_BOXES> boxesLocal_cx;
    LocalTensor<DTYPE_BOXES> boxesLocal_cy;
    LocalTensor<DTYPE_BOXES> boxesLocal_cz;
    LocalTensor<DTYPE_BOXES> boxesLocal_dx;
    LocalTensor<DTYPE_BOXES> boxesLocal_dy;
    LocalTensor<DTYPE_BOXES> boxesLocal_dz;
    LocalTensor<DTYPE_BOXES> boxesLocal_rz;
    LocalTensor<DTYPE_PTS> pointLocal;
    LocalTensor<DTYPE_BOXES_IDX_OF_POINTS> zLocal;
    LocalTensor<DTYPE_BOXES> shiftx;
    LocalTensor<DTYPE_BOXES> shifty;
    LocalTensor<DTYPE_BOXES> cosa;
    LocalTensor<DTYPE_BOXES> sina;
    LocalTensor<DTYPE_BOXES> xlocal;
    LocalTensor<DTYPE_BOXES> ylocal;
    LocalTensor<DTYPE_BOXES> temp;
    LocalTensor<uint8_t> uint8temp;
};

extern "C" __global__ __aicore__ void points_in_box(GM_ADDR boxes, GM_ADDR pts,
                                                    GM_ADDR boxes_idx_of_points,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelPointsInBox op;
    op.Init(boxes, pts, boxes_idx_of_points, &tiling_data);
    op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void points_in_box_do(uint32_t blockDim, void* l2ctrl,
                      void* stream, uint8_t* boxes, uint8_t* pts, uint8_t* boxes_idx_of_points,
                      uint8_t* workspace, uint8_t* tiling)
{
    points_in_box<<<blockDim, l2ctrl, stream>>>(boxes, pts, boxes_idx_of_points, workspace, tiling);
}
#endif