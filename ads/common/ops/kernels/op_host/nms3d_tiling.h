/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef NMS3D_TILING_H
#define NMS3D_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Nms3dTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)  // used cores
    TILING_DATA_FIELD_DEF(uint32_t, boxNum)  // count of boxes
    TILING_DATA_FIELD_DEF(uint32_t, loopTime)  // loop times
    TILING_DATA_FIELD_DEF(uint32_t, eachSum) // count of each core, = loop_time * 8
    TILING_DATA_FIELD_DEF(uint32_t, tailSum) // count of tail core
    TILING_DATA_FIELD_DEF(uint32_t, tailNum) // last time count of tail core
    TILING_DATA_FIELD_DEF(uint32_t, maskNum) // mask align 32bit
    TILING_DATA_FIELD_DEF(float, overlapThresh)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Nms3d, Nms3dTilingData)
} // namespace optiling

#endif // NMS3D_TILING_H