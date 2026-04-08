/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef ROIAWARE_POOL3D_TILING_H
#define ROIAWARE_POOL3D_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(RoiawarePool3dTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, coreBoxNums);
    TILING_DATA_FIELD_DEF(uint32_t, coreBoxTail);
    TILING_DATA_FIELD_DEF(uint32_t, boxNum);
    TILING_DATA_FIELD_DEF(uint32_t, ptsNum);
    TILING_DATA_FIELD_DEF(uint32_t, channelNum);
    TILING_DATA_FIELD_DEF(uint32_t, maxPtsPerVoxel);
    TILING_DATA_FIELD_DEF(uint32_t, outx);
    TILING_DATA_FIELD_DEF(uint32_t, outy);
    TILING_DATA_FIELD_DEF(uint32_t, outz);
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
    TILING_DATA_FIELD_DEF(uint32_t, mode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RoiawarePool3d, RoiawarePool3dTilingData)
}
#endif