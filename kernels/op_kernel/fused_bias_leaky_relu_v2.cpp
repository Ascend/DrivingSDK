/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#include "fused_bias_leaky_relu_v2.h"


using namespace AscendC;

extern "C" __global__ __aicore__ void fused_bias_leaky_relu_v2(GM_ADDR x, GM_ADDR bias, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelFusedBiasLeakyReluV2 op(x, bias, output, &tiling_data);
    op.Process();
}