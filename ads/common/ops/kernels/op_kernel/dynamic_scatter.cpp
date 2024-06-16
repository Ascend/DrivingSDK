/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#include "dynamic_scatter_max.h"
#include "dynamic_scatter_mean.h"
#include "dynamic_scatter_sum.h"

using namespace DynamicScatter;

extern "C" __global__ __aicore__ void dynamic_scatter(GM_ADDR point_feats, GM_ADDR prefix_sum_point_per_voxel,
    GM_ADDR argsort_coor, GM_ADDR voxel_feats, GM_ADDR compare_mask, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    SetSysWorkspace(workspace);
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(102)) {
        DynamicScatter::DynamicScatterMax<float> op;
        op.Init(point_feats, prefix_sum_point_per_voxel, argsort_coor, voxel_feats, compare_mask, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(101)) {
        DynamicScatter::DynamicScatterMean<float> op;
        op.Init(point_feats, prefix_sum_point_per_voxel, argsort_coor, voxel_feats, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(100)) {
        DynamicScatter::DynamicScatterSum<float> op;
        op.Init(point_feats, prefix_sum_point_per_voxel, argsort_coor, voxel_feats, &tilingData, &pipe);
        op.Process();
    }
}