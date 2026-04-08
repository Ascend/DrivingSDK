/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#ifndef HARD_VOXELIZE_TILING_H
#define HARD_VOXELIZE_TILING_H
#include "register/tilingdata_base.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(HardVoxelizeTilingData)
TILING_DATA_FIELD_DEF(int32_t, usedDiffBlkNum)
TILING_DATA_FIELD_DEF(int32_t, avgDiffTasks)
TILING_DATA_FIELD_DEF(int32_t, tailDiffTasks)
TILING_DATA_FIELD_DEF(int32_t, totalDiffTasks)
TILING_DATA_FIELD_DEF(int32_t, avgPts)
TILING_DATA_FIELD_DEF(int32_t, tailPts)
TILING_DATA_FIELD_DEF(int32_t, totalPts)
TILING_DATA_FIELD_DEF(int32_t, numPts)
TILING_DATA_FIELD_DEF(int32_t, usedCopyBlkNum)
TILING_DATA_FIELD_DEF(int32_t, avgCopyTasks)
TILING_DATA_FIELD_DEF(int32_t, tailCopyTasks)
TILING_DATA_FIELD_DEF(int32_t, totalCopyTasks)
TILING_DATA_FIELD_DEF(int32_t, avgVoxs)
TILING_DATA_FIELD_DEF(int32_t, tailVoxs)
TILING_DATA_FIELD_DEF(int32_t, totalVoxs)
TILING_DATA_FIELD_DEF(int32_t, featNum)
TILING_DATA_FIELD_DEF(int32_t, freeNum)
TILING_DATA_FIELD_DEF(int32_t, maxPoints)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(HardVoxelize, HardVoxelizeTilingData)
} // namespace optiling
#endif // HARD_VOXELIZE_TILING_H
