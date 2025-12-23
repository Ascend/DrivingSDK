/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 */
#include "kernel_operator.h"
#include "scatter_max_v3.h"

using namespace AscendC;

#define TILING_KEY_SMALL_TAIL 0
#define TILING_KEY_LARGE_TAIL 1


extern "C" __global__ __aicore__ void scatter_max_argmax_v3(
    GM_ADDR src, GM_ADDR idx, GM_ADDR res, GM_ADDR argmax, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);

    if (TILING_KEY_IS(TILING_KEY_SMALL_TAIL)) {
        KernelScatterMaxArgmaxV3<true> op(src, idx, res, argmax, &tiling_data, &pipe);
        op.Process();
    } else {
        KernelScatterMaxArgmaxV3<false> op(src, idx, res, argmax, &tiling_data, &pipe);
        op.Process();
    }
}
