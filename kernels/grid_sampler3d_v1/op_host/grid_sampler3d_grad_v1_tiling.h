/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#include "register/tilingdata_base.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(GridSampler3dGradV1TilingData)
        TILING_DATA_FIELD_DEF(uint32_t, batch);
        TILING_DATA_FIELD_DEF(uint32_t, pNumPerCore);
        TILING_DATA_FIELD_DEF(uint32_t, tailPNum);
        TILING_DATA_FIELD_DEF(uint32_t, channel);
        TILING_DATA_FIELD_DEF(uint32_t, depth);
        TILING_DATA_FIELD_DEF(uint32_t, height);
        TILING_DATA_FIELD_DEF(uint32_t, width);
        TILING_DATA_FIELD_DEF(uint32_t, gridD);
        TILING_DATA_FIELD_DEF(uint32_t, gridH);
        TILING_DATA_FIELD_DEF(uint32_t, gridW);
        TILING_DATA_FIELD_DEF(uint32_t, blockNum);
        TILING_DATA_FIELD_DEF(uint32_t, ubFactorElement);
        TILING_DATA_FIELD_DEF(uint32_t, interpolation);
        TILING_DATA_FIELD_DEF(uint32_t, padding);
        TILING_DATA_FIELD_DEF(bool, alignCorners);
        TILING_DATA_FIELD_DEF(uint32_t, group);
    END_TILING_DATA_DEF;
    REGISTER_TILING_DATA_CLASS(GridSampler3dGradV1, GridSampler3dGradV1TilingData)

    struct InputParamsInfo {
        uint32_t batch = 0;
        uint32_t channel = 0;
        uint32_t depth = 0;
        uint32_t height = 0;
        uint32_t width = 0;
        uint32_t gridD = 0;
        uint32_t gridH = 0;
        uint32_t gridW = 0;
        uint32_t tilingKey = 1;
        uint32_t interpolation = 0;
        uint32_t padding = 0;
        bool alignCorners = false;
    };
} // namespace optiling