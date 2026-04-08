/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"
#include "scatter_add_normal.h"
#include "scatter_add_notail.h"
#include "scatter_add_notail_bighead.h"

extern "C" __global__ __aicore__ void scatter_add_v2(GM_ADDR src, GM_ADDR indices, GM_ADDR var, GM_ADDR out,
                                                              GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(2)) {
        ScatterAddNoTail op;
        op.Init(src, indices, var, out, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KernelScatterAddLine op;
        op.Init(src, indices, var, out, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        ScatterAddNoTailBigHead op;
        op.Init(src, indices, var, out, &tiling_data, &pipe);
        op.Process();
    }
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void scatter_add_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* src, uint8_t* indices, uint8_t* var,
    uint8_t* out, uint8_t* workspace, uint8_t* tiling)
{
    scatter_add<<<blockDim, l2ctrl, stream>>>(src, indices, var, out, workspace, tiling);
}
#endif