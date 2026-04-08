/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#ifndef POINT_TO_VOXEL_TILING_H
#define POINT_TO_VOXEL_TILING_H
#include "register/tilingdata_base.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(PointToVoxelTilingData)
TILING_DATA_FIELD_DEF(int32_t, gridX)
TILING_DATA_FIELD_DEF(int32_t, gridY)
TILING_DATA_FIELD_DEF(int32_t, gridZ)
TILING_DATA_FIELD_DEF(float, voxelSizeX)
TILING_DATA_FIELD_DEF(float, voxelSizeY)
TILING_DATA_FIELD_DEF(float, voxelSizeZ)
TILING_DATA_FIELD_DEF(float, coorXMin)
TILING_DATA_FIELD_DEF(float, coorYMin)
TILING_DATA_FIELD_DEF(float, coorZMin)
TILING_DATA_FIELD_DEF(int32_t, avgTasks)
TILING_DATA_FIELD_DEF(int32_t, tailTasks)
TILING_DATA_FIELD_DEF(int32_t, totalTasks)
TILING_DATA_FIELD_DEF(int32_t, avgPts)
TILING_DATA_FIELD_DEF(int32_t, tailPts)
TILING_DATA_FIELD_DEF(int32_t, totalPts)
TILING_DATA_FIELD_DEF(int32_t, usedBlkNum)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(PointToVoxel, PointToVoxelTilingData)
REGISTER_TILING_DATA_CLASS(VoxelToPoint, PointToVoxelTilingData)
} // namespace optiling
#endif // POINT_TO_VOXEL_TILING_H
