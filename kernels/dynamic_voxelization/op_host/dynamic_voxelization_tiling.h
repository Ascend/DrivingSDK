/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef DYNAMIC_VOXELIZATION_TILING_H
#define DYNAMIC_VOXELIZATION_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DynamicVoxTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, core_num)
    TILING_DATA_FIELD_DEF(uint32_t, points_num_in_core)
    TILING_DATA_FIELD_DEF(uint32_t, points_num_in_last_core)
    TILING_DATA_FIELD_DEF(uint32_t, pts_num)
    TILING_DATA_FIELD_DEF(uint32_t, pts_feature)
    TILING_DATA_FIELD_DEF(float, coors_min_x)
    TILING_DATA_FIELD_DEF(float, coors_min_y)
    TILING_DATA_FIELD_DEF(float, coors_min_z)
    TILING_DATA_FIELD_DEF(float, voxel_x)
    TILING_DATA_FIELD_DEF(float, voxel_y)
    TILING_DATA_FIELD_DEF(float, voxel_z)
    TILING_DATA_FIELD_DEF(int32_t, grid_x)
    TILING_DATA_FIELD_DEF(int32_t, grid_y)
    TILING_DATA_FIELD_DEF(int32_t, grid_z)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(DynamicVoxelization, DynamicVoxTilingData)
} // namespace optiling

#endif // DYNAMIC_VOXELIZATION_TILING_H

