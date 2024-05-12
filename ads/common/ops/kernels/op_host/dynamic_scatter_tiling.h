/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef DYNAMIC_SCATTER_TILING_H
#define DYNAMIC_SCATTER_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DynamicScatterTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, featsNum);
    TILING_DATA_FIELD_DEF(uint32_t, pointNum);
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
    TILING_DATA_FIELD_DEF(uint32_t, outNum);
    TILING_DATA_FIELD_DEF(uint32_t, reduceMode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DynamicScatter, DynamicScatterTilingData)
}
#endif // DYNAMIC_SCATTER_TILING_H
