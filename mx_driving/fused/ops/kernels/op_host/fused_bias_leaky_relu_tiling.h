/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef FUSED_BIAS_LEAK_RELU_TILING_H
#define FUSED_BIAS_LEAK_RELU_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FusedBiasLeakyReluTilingData)
    TILING_DATA_FIELD_DEF(float, negative_slope);
    TILING_DATA_FIELD_DEF(float, scale);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, average);
    TILING_DATA_FIELD_DEF(uint32_t, remainder);
    TILING_DATA_FIELD_DEF(uint32_t, totalDataLength);
    TILING_DATA_FIELD_DEF(uint32_t, singleBlock);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedBiasLeakyRelu, FusedBiasLeakyReluTilingData)
}
#endif
// FUSED_BIAS_LEAK_RELU_TILING_H