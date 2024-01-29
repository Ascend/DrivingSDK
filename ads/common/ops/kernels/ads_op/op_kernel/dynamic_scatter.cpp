/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#include "dynamic_scatter_max.h"
#include "dynamic_scatter_sum.h"

using namespace DynamicScatterN;

extern "C" __global__ __aicore__ void dynamic_scatter(GM_ADDR feats, GM_ADDR coors_map, GM_ADDR reduced_feats,
                                                      GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(100)) {
        DynamicScatterN::DynamicScatterMax<float> op;
        op.Init(feats, coors_map, reduced_feats, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(101)) {
        DynamicScatterN::DynamicScatterSum<float> op;
        op.Init(feats, coors_map, reduced_feats, &tilingData);
        op.Process();
    }
}