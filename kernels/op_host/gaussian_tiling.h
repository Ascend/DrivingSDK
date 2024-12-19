/*
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
*/
#ifndef GAUSSIAN_TILING_H
#define GAUSSIAN_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GaussianTilingData)
    TILING_DATA_FIELD_DEF(float, out_size_factor);
    TILING_DATA_FIELD_DEF(float, gaussian_overlap);
    TILING_DATA_FIELD_DEF(int32_t, min_radius);
    TILING_DATA_FIELD_DEF(float, voxel_size_x);
    TILING_DATA_FIELD_DEF(float, voxel_size_y);
    TILING_DATA_FIELD_DEF(float, pc_range_x);
    TILING_DATA_FIELD_DEF(float, pc_range_y);
    TILING_DATA_FIELD_DEF(int32_t, feature_map_size_x);
    TILING_DATA_FIELD_DEF(int32_t, feature_map_size_y);
    TILING_DATA_FIELD_DEF(int32_t, num_objs);
    TILING_DATA_FIELD_DEF(bool, norm_bbox);
    TILING_DATA_FIELD_DEF(uint32_t, core_data);
    TILING_DATA_FIELD_DEF(uint32_t, average);
    TILING_DATA_FIELD_DEF(uint32_t, former_num);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Gaussian, GaussianTilingData)
} // namespace optiling

#endif // GAUSSIAN_TILING_H