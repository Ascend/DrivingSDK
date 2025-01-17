/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scatter_add_grad_base.h"
#include "scatter_add_grad.h"
#include "scatter_add_grad_line.h"
#include "scatter_add_grad_large.h"

using namespace ScatterAddGradNS;

extern "C" __global__ __aicore__ void scatter_add_grad_v2(GM_ADDR grad_out, GM_ADDR index, GM_ADDR grad_in, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(2)) {
        ScatterAddGradNS::ScatterAddGradLine<float> op;
        op.InitLine(grad_out, index, grad_in, &tilingData);
        op.Process();
    }
    if (TILING_KEY_IS(1)) {
        ScatterAddGradNS::ScatterAddGradV2<float> op;
        op.Init(grad_out, index, grad_in, &tilingData);
        op.Process();
    }
    else if (TILING_KEY_IS(3)) {
        ScatterAddGradNS::ScatterAddGradLarge<float> op;
        op.Init(grad_out, index, grad_in, &tilingData);
        op.Process();
    }
}