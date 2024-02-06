/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef FURTHEST_POINT_SAMPLING_WITH_DIST_TILING_H
#define FURTHEST_POINT_SAMPLING_WITH_DIST_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FurthestPointSamplingWithDistTilingData)

    TILING_DATA_FIELD_DEF(uint32_t, used_core_num)
    TILING_DATA_FIELD_DEF(uint32_t, points_num)
    TILING_DATA_FIELD_DEF(uint32_t, task_num)
    TILING_DATA_FIELD_DEF(uint32_t, task_num_tail)
    TILING_DATA_FIELD_DEF(uint32_t, n)
    TILING_DATA_FIELD_DEF(uint32_t, batch_dist_offset)
    TILING_DATA_FIELD_DEF(uint32_t, batch_idx_offset)
    TILING_DATA_FIELD_DEF(uint32_t, part_ub)
    TILING_DATA_FIELD_DEF(uint32_t, move_n_times)
    TILING_DATA_FIELD_DEF(uint32_t, n_tail)
    TILING_DATA_FIELD_DEF(uint32_t, id_move_len)
    TILING_DATA_FIELD_DEF(uint32_t, repeat_id_times)
    TILING_DATA_FIELD_DEF(uint32_t, id_tail)
    TILING_DATA_FIELD_DEF(uint32_t, work_size)

END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(FurthestPointSamplingWithDist, FurthestPointSamplingWithDistTilingData)
} // namespace optiling

#endif // FURTHEST_POINT_SAMPLING_WITH_DIST_TILING_H
