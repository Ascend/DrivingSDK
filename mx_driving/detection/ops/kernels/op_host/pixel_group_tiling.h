/*
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
*/
#ifndef PIXEL_GROUP_TILING_H
#define PIXEL_GROUP_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PixelGroupTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, core_used);
    TILING_DATA_FIELD_DEF(uint32_t, total_pixels);
    TILING_DATA_FIELD_DEF(uint32_t, average_pixels);
    TILING_DATA_FIELD_DEF(uint32_t, pixel_last);

    TILING_DATA_FIELD_DEF(uint32_t, embedding_dim);
    TILING_DATA_FIELD_DEF(uint32_t, dim_align);
    TILING_DATA_FIELD_DEF(int32_t, kernel_region_num);
    TILING_DATA_FIELD_DEF(float, distance_threshold);
    TILING_DATA_FIELD_DEF(uint64_t, available_ub_size);
    TILING_DATA_FIELD_DEF(uint32_t, loop_time_front);
    TILING_DATA_FIELD_DEF(uint32_t, last_loop_front);
    TILING_DATA_FIELD_DEF(uint32_t, loop_time_rear);
    TILING_DATA_FIELD_DEF(uint32_t, last_loop_rear);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PixelGroup, PixelGroupTilingData)
} // namespace optiling

#endif // PIXEL_GROUP_TILING_H