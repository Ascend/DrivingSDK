/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 */
#include "scatter_max_v1.h"

#include "kernel_operator.h"

using namespace AscendC;


extern "C" __global__ __aicore__ void scatter_max_v1(
    GM_ADDR src, GM_ADDR idx, GM_ADDR res, GM_ADDR argmax, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);

    if (TILING_KEY_IS(0)) { // TILING_KEY_SMALL_TAIL
        KernelScatterMaxV1<true> op(src, idx, res, argmax, &tiling_data, &pipe);
        op.Process();
    } else { // TILING_KEY_LARGE_TAIL
        KernelScatterMaxV1<false> op(src, idx, res, argmax, &tiling_data, &pipe);
        op.Process();
    }
}
