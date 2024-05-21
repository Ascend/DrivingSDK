/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */

#include "voxel_pooling_train.h"

extern "C" __global__ __aicore__ void voxel_pooling_train(
    GM_ADDR geom,
    GM_ADDR featuresIn,
    GM_ADDR featuresOut,
    GM_ADDR posMemo,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);

    GET_TILING_DATA(tilingData, tiling);
    const VoxelPoolingTilingData* __restrict tilingDevice = &tilingData;
    VoxelPoolingTrainKernel<float> op(geom, featuresIn, featuresOut, posMemo, workspace, tilingDevice);
    op.Process();
}
