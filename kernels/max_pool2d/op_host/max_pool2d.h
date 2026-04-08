/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef MAX_POOL2D_TILING_H
#define MAX_POOL2D_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MaxPool2dTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, channel)
TILING_DATA_FIELD_DEF(uint32_t, inHeight)
TILING_DATA_FIELD_DEF(uint32_t, inWidth)
TILING_DATA_FIELD_DEF(uint32_t, outHeight)
TILING_DATA_FIELD_DEF(uint32_t, outWidth)
TILING_DATA_FIELD_DEF(uint32_t, coreNum)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPool2d, MaxPool2dTilingData)
} // namespace optiling
#endif // MAX_POOL2D_TILING_H
