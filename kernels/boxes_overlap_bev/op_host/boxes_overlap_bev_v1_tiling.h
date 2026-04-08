/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef BOXES_OVERLAP_BEV_TILING_H
#define BOXES_OVERLAP_BEV_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BoxesOverlapBevV1TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, M);
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, totalCoreCount);
    TILING_DATA_FIELD_DEF(uint32_t, tileCountN);
    TILING_DATA_FIELD_DEF(uint32_t, tileCountM);
    TILING_DATA_FIELD_DEF(uint32_t, tileN);
    TILING_DATA_FIELD_DEF(uint32_t, modeFlag);
    TILING_DATA_FIELD_DEF(float, margin);
    
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BoxesOverlapBevV1, BoxesOverlapBevV1TilingData)
}
#endif // TIK_TOOLS_TILING_H