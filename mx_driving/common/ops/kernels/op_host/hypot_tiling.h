/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef HYPOT_TILING_H
#define HYPOT_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HypotTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, formerNum);
    TILING_DATA_FIELD_DEF(uint32_t, tailNum);
    TILING_DATA_FIELD_DEF(uint64_t, formerLength);
    TILING_DATA_FIELD_DEF(uint32_t, tailLength);
    TILING_DATA_FIELD_DEF(uint32_t, formerTileLength);
    TILING_DATA_FIELD_DEF(uint32_t, tailTileLength);
    TILING_DATA_FIELD_DEF(uint32_t, formerTileNum);
    TILING_DATA_FIELD_DEF(uint32_t, tailTileNum);
    TILING_DATA_FIELD_DEF(uint32_t, formerRemainTileLength);
    TILING_DATA_FIELD_DEF(uint32_t, tailRemainTileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Hypot, HypotTilingData)
REGISTER_TILING_DATA_CLASS(HypotGrad, HypotTilingData)
}

#endif // HYPOT_TILING_H
