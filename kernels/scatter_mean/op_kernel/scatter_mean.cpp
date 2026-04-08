/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Function : z = x + y
 * This sample is a very basic sample that implements vector add on Ascend plaform.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"
#include "scatter_mean_normal.h"
#include "scatter_mean_notail.h"
#include "scatter_mean_notail_bighead.h"

extern "C" __global__ __aicore__ void scatter_mean(GM_ADDR src, GM_ADDR indices, GM_ADDR var, GM_ADDR out, GM_ADDR count,
                                                              GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(2)) {
        ScatterMeanNoTail op;
        op.Init(src, indices, var, out, count, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KernelScatterMeanFix op;
        op.Init(src, indices, var, out, count, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        ScatterMeanNoTailBigHead op;
        op.Init(src, indices, var, out, count, &tiling_data, &pipe);
        op.Process();
    }
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void scatter_mean_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* src, uint8_t* indices, uint8_t* var,
    uint8_t* out, uint8_t* count, uint8_t* workspace, uint8_t* tiling)
{
    scatter_mean<<<blockDim, l2ctrl, stream>>>(src, indices, var, out, count, workspace, tiling);
}
#endif