/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 *
 */
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(UniqueTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint32_t, shortBlockTileNum);
    TILING_DATA_FIELD_DEF(uint16_t, tailLength);
    TILING_DATA_FIELD_DEF(uint8_t, blockNum);
    TILING_DATA_FIELD_DEF(uint8_t, shortBlockNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Unique, UniqueTilingData)
}
