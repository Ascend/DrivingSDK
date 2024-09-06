/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "dynamic_scatter_grad_max.h"
#include "dynamic_scatter_grad_mean.h"
#include "dynamic_scatter_grad_sum.h"

using namespace DynamicScatterGrad;

extern "C" __global__ __aicore__ void dynamic_scatter_grad(GM_ADDR grad_voxel_feats, GM_ADDR prefix_sum_point_per_voxel,
    GM_ADDR argsort_coor, GM_ADDR compare_mask, GM_ADDR grad_point_feats, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(102)) {
        DynamicScatterGrad::DynamicScatterGradMax<float> op;
        op.Init(grad_voxel_feats, prefix_sum_point_per_voxel, argsort_coor, compare_mask, grad_point_feats, &tilingData,
            &pipe);
        op.Process();
    } else if (TILING_KEY_IS(101)) {
        DynamicScatterGrad::DynamicScatterGradMean<float> op;
        op.Init(grad_voxel_feats, prefix_sum_point_per_voxel, argsort_coor, grad_point_feats, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(100)) {
        DynamicScatterGrad::DynamicScatterGradSum<float> op;
        op.Init(grad_voxel_feats, prefix_sum_point_per_voxel, argsort_coor, grad_point_feats, &tilingData, &pipe);
        op.Process();
    }
}