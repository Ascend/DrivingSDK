/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef ROIPOINT_POOL3D_FORWARD_TILING_H
#define ROIPOINT_POOL3D_FORWARD_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RoipointPool3dForwardTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numSampledPoints);
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, pointNum);
    TILING_DATA_FIELD_DEF(uint32_t, featureLen);
    TILING_DATA_FIELD_DEF(uint32_t, boxesNum);
    TILING_DATA_FIELD_DEF(uint32_t, eachCoreBoxes);
    TILING_DATA_FIELD_DEF(uint32_t, ubSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RoipointPool3dForward, RoipointPool3dForwardTilingData)
}
#endif // ROIPOINT_POOL3D_FORWARD_TILING_H