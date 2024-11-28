/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scatter_mean_grad.h"
#include "scatter_mean_grad_line.h"
#include "scatter_mean_grad_large.h"

using namespace ScatterMeanGradNS;

extern "C" __global__ __aicore__ void scatter_mean_grad(GM_ADDR grad_out, GM_ADDR index, GM_ADDR count, GM_ADDR grad_in, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        ScatterMeanGradNS::ScatterMeanGrad<float> op;
        op.Init(grad_out, index, count, grad_in, &tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(2)) {
        ScatterMeanGradNS::ScatterMeanGradLine<float> op;
        op.InitLine(grad_out, index, count, grad_in, &tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(3)) {
        ScatterMeanGradNS::ScatterMeanGradLarge<float> op;
        op.Init(grad_out, index, count, grad_in, &tilingData);
        op.Process();
    }
}