/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef BOX_IOU_TILING_H
#define BOX_IOU_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BoxIouTilingData)
TILING_DATA_FIELD_DEF(uint32_t, boxesANum)
TILING_DATA_FIELD_DEF(uint32_t, boxesBNum)
TILING_DATA_FIELD_DEF(uint32_t, taskNum)
TILING_DATA_FIELD_DEF(uint32_t, taskNumPerCore)
TILING_DATA_FIELD_DEF(uint32_t, outerLoopCnt)
TILING_DATA_FIELD_DEF(uint32_t, innerLoopCnt)
TILING_DATA_FIELD_DEF(uint32_t, boxesDescDimNum)

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BoxIou, BoxIouTilingData)
} // namespace optiling

#endif // BOX_IOU_TILING_H
