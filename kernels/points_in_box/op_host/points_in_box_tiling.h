/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(PointsInBoxTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, core_used);
    TILING_DATA_FIELD_DEF(uint64_t, core_data);
    TILING_DATA_FIELD_DEF(uint32_t, copy_loop);
    TILING_DATA_FIELD_DEF(uint32_t, copy_tail);
    TILING_DATA_FIELD_DEF(uint32_t, last_copy_loop);
    TILING_DATA_FIELD_DEF(uint32_t, last_copy_tail);
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, npoints);
    TILING_DATA_FIELD_DEF(uint64_t, box_number);
    TILING_DATA_FIELD_DEF(uint32_t, available_ub_size);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(PointsInBox, PointsInBoxTilingData)
}
#endif // ADD_CUSTOM_TILING_H