/*
* Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
*/
#include "kernel_operator.h"
#include "unique.h"


extern "C" __global__ __aicore__ void unique(
    GM_ADDR input, GM_ADDR output, GM_ADDR uniqueCnt, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    KernelUnique<DTYPE_INPUT> op(pipe);
    op.Init(input,
            output,
            uniqueCnt,
            workspace,
            tiling_data.totalLength,
            tiling_data.shortBlockTileNum,
            tiling_data.tileLength,
            tiling_data.tailLength,
            tiling_data.aivNum,
            tiling_data.blockNum,
            tiling_data.shortBlockNum);
    op.Process();
}