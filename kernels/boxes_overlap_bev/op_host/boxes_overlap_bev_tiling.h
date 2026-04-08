/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef BOXES_OVERLAP_BEV_TILING_H
#define BOXES_OVERLAP_BEV_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BoxesOverlapBevTilingData)
TILING_DATA_FIELD_DEF(uint32_t, boxesANum)
TILING_DATA_FIELD_DEF(uint32_t, boxesBNum)
TILING_DATA_FIELD_DEF(uint32_t, boxesFormatSize)
TILING_DATA_FIELD_DEF(uint32_t, numLargeCores)
TILING_DATA_FIELD_DEF(uint64_t, numTasksPerLargeCore)
TILING_DATA_FIELD_DEF(int32_t, formatFlag)
TILING_DATA_FIELD_DEF(int32_t, modeFlag)
TILING_DATA_FIELD_DEF(float, margin)

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BoxesOverlapBev, BoxesOverlapBevTilingData)
} // namespace optiling

#endif // BOXES_OVERLAP_BEV_TILING_H
