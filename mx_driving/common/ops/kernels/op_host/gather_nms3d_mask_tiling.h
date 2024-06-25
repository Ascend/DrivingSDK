/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef GATHER_NMS3D_MASK_TILING_H
#define GATHER_NMS3D_MASK_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GatherNms3dMaskTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, box_num);
    TILING_DATA_FIELD_DEF(uint32_t, mask_num);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GatherNms3dMask, GatherNms3dMaskTilingData)
}

#endif // GATHER_NMS3D_MASK_TILING_H
