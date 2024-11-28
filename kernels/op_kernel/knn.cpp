/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#include "knn.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void knn(
    GM_ADDR xyz,
    GM_ADDR center_xyz,
    GM_ADDR dist,
    GM_ADDR idx,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    TPipe tmpPipe;
    GET_TILING_DATA(tiling_data, tiling);

    KnnKernel<float, int32_t> op(xyz, center_xyz, dist, idx, &tiling_data, &tmpPipe);
    op.Process();
}