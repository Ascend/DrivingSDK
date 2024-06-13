/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#ifndef UNIQUE_VOXEL_TILING_H
#define UNIQUE_VOXEL_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(UniqueVoxelTilingData)
TILING_DATA_FIELD_DEF(int32_t, avgTasks)
TILING_DATA_FIELD_DEF(int32_t, tailTasks)
TILING_DATA_FIELD_DEF(int32_t, totalTasks)
TILING_DATA_FIELD_DEF(int32_t, avgPts)
TILING_DATA_FIELD_DEF(int32_t, tailPts)
TILING_DATA_FIELD_DEF(int32_t, totalPts)
TILING_DATA_FIELD_DEF(int32_t, usedBlkNum)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(UniqueVoxel, UniqueVoxelTilingData)
} // namespace optiling
#endif // UNIQUE_VOXEL_TILING_H
