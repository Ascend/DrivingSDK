/*
* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef GRID_SAMPLER2D_V2_GRAD_TILING_H
#define GRID_SAMPLER2D_V2_GRAD_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GridSampler2dV2GradTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, pNumPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, tailPNum);
    TILING_DATA_FIELD_DEF(uint32_t, channel);
    TILING_DATA_FIELD_DEF(uint32_t, alignedChannel);
    TILING_DATA_FIELD_DEF(uint32_t, height);
    TILING_DATA_FIELD_DEF(uint32_t, width);
    TILING_DATA_FIELD_DEF(uint32_t, gridH);
    TILING_DATA_FIELD_DEF(uint32_t, gridW);
    TILING_DATA_FIELD_DEF(uint32_t, blockNum);
    TILING_DATA_FIELD_DEF(uint32_t, calcCountPerLoop);
    TILING_DATA_FIELD_DEF(uint32_t, interpolation);
    TILING_DATA_FIELD_DEF(uint32_t, padding);
    TILING_DATA_FIELD_DEF(bool, alignCorners)
    TILING_DATA_FIELD_DEF(int32_t, groupSize)
    TILING_DATA_FIELD_DEF(int32_t, coordPosition)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(GridSampler2dV2Grad, GridSampler2dV2GradTilingData)
} // namespace optiling
#endif // GRID_SAMPLER2D_V2_GRAD_TILING_H