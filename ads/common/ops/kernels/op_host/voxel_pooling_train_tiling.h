/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
 */
#ifndef VOXEL_POOLING_TRAIN_TILING_H
#define VOXEL_POOLING_TRAIN_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(VoxelPoolingTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, core_num)
    TILING_DATA_FIELD_DEF(uint32_t, features_num_in_core)
    TILING_DATA_FIELD_DEF(uint32_t, features_num_in_last_core)
    TILING_DATA_FIELD_DEF(uint32_t, batch_size)
    TILING_DATA_FIELD_DEF(uint32_t, num_points)
    TILING_DATA_FIELD_DEF(uint32_t, num_channels)
    TILING_DATA_FIELD_DEF(uint32_t, num_voxel_x)
    TILING_DATA_FIELD_DEF(uint32_t, num_voxel_y)
    TILING_DATA_FIELD_DEF(uint32_t, num_voxel_z)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(VoxelPoolingTrain, VoxelPoolingTilingData)
} // namespace optiling

#endif // VOXEL_POOLING_TRAIN_TILING_H