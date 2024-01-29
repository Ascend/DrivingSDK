/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef DYNAMIC_SCATTER_TILING_H
#define DYNAMIC_SCATTER_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DynamicScatterTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, formerNum);
    TILING_DATA_FIELD_DEF(uint32_t, tailNum);
    TILING_DATA_FIELD_DEF(uint32_t, formerLength);
    TILING_DATA_FIELD_DEF(uint32_t, tailLength);
    TILING_DATA_FIELD_DEF(uint32_t, alignNum);
    TILING_DATA_FIELD_DEF(uint32_t, totalLengthAligned);
    TILING_DATA_FIELD_DEF(uint32_t, formerInputNum);
    TILING_DATA_FIELD_DEF(uint32_t, tailInputNum);
    TILING_DATA_FIELD_DEF(uint32_t, featsNum);
    TILING_DATA_FIELD_DEF(uint32_t, outPointNum);
    TILING_DATA_FIELD_DEF(uint32_t, outPointNumAligned);
    TILING_DATA_FIELD_DEF(uint32_t, featsAligned);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DynamicScatter, DynamicScatterTilingData)
}
#endif // DYNAMIC_SCATTER_TILING_H