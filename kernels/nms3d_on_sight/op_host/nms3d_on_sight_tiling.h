/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef NMS3D_ON_SIGHT_TILING_H
#define NMS3D_ON_SIGHT_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Nms3dOnSightTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)  // used cores
    TILING_DATA_FIELD_DEF(uint32_t, boxNum)  // count of boxes
    TILING_DATA_FIELD_DEF(uint32_t, loopTime)  // loop times
    TILING_DATA_FIELD_DEF(uint32_t, assignBox) // boxesNum align 256B
    TILING_DATA_FIELD_DEF(uint32_t, alignedN) // boxesNum align 32B
    TILING_DATA_FIELD_DEF(float, threshold)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Nms3dOnSight, Nms3dOnSightTilingData)
} // namespace optiling

#endif // NMS3D_ON_SIGHT_TILING_H
