/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef roi_align_rotated_v2_tiling_h
#define roi_align_rotated_v2_tiling_h
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RoiAlignRotatedV2TilingData)
    TILING_DATA_FIELD_DEF(bool, aligned);
    TILING_DATA_FIELD_DEF(bool, clockwise);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    TILING_DATA_FIELD_DEF(uint32_t, rois_num_per_Lcore);
    TILING_DATA_FIELD_DEF(uint32_t, rois_num_per_Score);
    TILING_DATA_FIELD_DEF(uint32_t, Lcore_num);
    TILING_DATA_FIELD_DEF(uint32_t, Score_num);
    TILING_DATA_FIELD_DEF(uint32_t, input_buffer_size);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint32_t, batch_size);
    TILING_DATA_FIELD_DEF(uint32_t, channels);
    TILING_DATA_FIELD_DEF(uint32_t, channels_aligned);
    TILING_DATA_FIELD_DEF(uint32_t, input_h);
    TILING_DATA_FIELD_DEF(uint32_t, input_w);
    TILING_DATA_FIELD_DEF(uint32_t, rois_num_aligned);
    TILING_DATA_FIELD_DEF(uint32_t, tail_num);
    TILING_DATA_FIELD_DEF(float, spatial_scale);
    TILING_DATA_FIELD_DEF(int32_t, sampling_ratio);
    TILING_DATA_FIELD_DEF(int32_t, pooled_height);
    TILING_DATA_FIELD_DEF(int32_t, pooled_width);
    TILING_DATA_FIELD_DEF(uint64_t, ub_total_size);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RoiAlignRotatedV2, RoiAlignRotatedV2TilingData)
}
#endif