/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#ifndef GRID_SAMPLE_2D_TILING_H
#define GRID_SAMPLE_2D_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GridSampler2dV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, interpolationMode)
TILING_DATA_FIELD_DEF(int64_t, paddingMode)
TILING_DATA_FIELD_DEF(bool, alignCorners)
TILING_DATA_FIELD_DEF(int32_t, batchSize)
TILING_DATA_FIELD_DEF(int32_t, channel)
TILING_DATA_FIELD_DEF(int32_t, inHeight)
TILING_DATA_FIELD_DEF(int32_t, inWidth)
TILING_DATA_FIELD_DEF(int32_t, outHeight)
TILING_DATA_FIELD_DEF(int32_t, outWidth)
TILING_DATA_FIELD_DEF(int32_t, taskNumPerCore)
TILING_DATA_FIELD_DEF(int32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(int32_t, alignedChannel)
TILING_DATA_FIELD_DEF(int32_t, alignedTaskNumPerLoop)
TILING_DATA_FIELD_DEF(int32_t, copyLoop)
TILING_DATA_FIELD_DEF(int32_t, copyTail)
TILING_DATA_FIELD_DEF(int32_t, lastCopyLoop)
TILING_DATA_FIELD_DEF(int32_t, lastCopyTail)
TILING_DATA_FIELD_DEF(int32_t, coordPosition)
TILING_DATA_FIELD_DEF(int32_t, groupSize)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(GridSampler2dV2, GridSampler2dV2TilingData)
} // namespace optiling
#endif // GRID_SAMPLE_2D_TILING_H