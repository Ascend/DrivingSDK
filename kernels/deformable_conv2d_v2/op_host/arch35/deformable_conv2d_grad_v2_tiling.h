/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef DEFORMABLE_CONV2D_GRAD_V2_TILING_H
#define DEFORMABLE_CONV2D_GRAD_V2_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DeformableConv2dGradV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, n)
TILING_DATA_FIELD_DEF(int64_t, cIn)
TILING_DATA_FIELD_DEF(int64_t, hIn)
TILING_DATA_FIELD_DEF(int64_t, wIn)
TILING_DATA_FIELD_DEF(int64_t, cOut)
TILING_DATA_FIELD_DEF(int64_t, hOut)
TILING_DATA_FIELD_DEF(int64_t, wOut)
TILING_DATA_FIELD_DEF(int64_t, kH)
TILING_DATA_FIELD_DEF(int64_t, kW)
TILING_DATA_FIELD_DEF(int64_t, padH)
TILING_DATA_FIELD_DEF(int64_t, padW)
TILING_DATA_FIELD_DEF(int64_t, strideH)
TILING_DATA_FIELD_DEF(int64_t, strideW)
TILING_DATA_FIELD_DEF(int64_t, dilationH)
TILING_DATA_FIELD_DEF(int64_t, dilationW)
TILING_DATA_FIELD_DEF(int64_t, groups)

TILING_DATA_FIELD_DEF(int64_t, coreCount)
TILING_DATA_FIELD_DEF(int64_t, singleLoopTask)
TILING_DATA_FIELD_DEF(int64_t, bigCoreCount)
TILING_DATA_FIELD_DEF(int64_t, coreTaskCount)
TILING_DATA_FIELD_DEF(int64_t, cube0TileTaskCount)
TILING_DATA_FIELD_DEF(int64_t, cube1TileTaskCount)
TILING_DATA_FIELD_DEF(int64_t, doubleBuffer)

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm0TilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(DeformableConv2dGradV2, DeformableConv2dGradV2TilingData)
} // namespace optiling
#endif // DEFORMABLE_CONV2D_GRAD_V2_TILING_H