/*
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
*/
#ifndef GAUSSIAN_TILING_H
#define GAUSSIAN_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GaussianTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
    TILING_DATA_FIELD_DEF(uint32_t, numObjs)
    TILING_DATA_FIELD_DEF(uint32_t, totalCoreTaskNum)
    TILING_DATA_FIELD_DEF(uint32_t, coreProcessTaskNum)
    TILING_DATA_FIELD_DEF(uint32_t, lastCoreProcessTaskNum)
    TILING_DATA_FIELD_DEF(uint32_t, singleProcessTaskNum)
    TILING_DATA_FIELD_DEF(uint32_t, featureMapSizeX)
    TILING_DATA_FIELD_DEF(uint32_t, featureMapSizeY)
    TILING_DATA_FIELD_DEF(float, voxelXSize)
    TILING_DATA_FIELD_DEF(float, voxelYSize)
    TILING_DATA_FIELD_DEF(float, prcX)
    TILING_DATA_FIELD_DEF(float, prcY)
    TILING_DATA_FIELD_DEF(uint32_t, featureMapStride)
    TILING_DATA_FIELD_DEF(uint32_t, numMaxObjs)
    TILING_DATA_FIELD_DEF(uint32_t, minRadius)
    TILING_DATA_FIELD_DEF(float, minOverLap)
    TILING_DATA_FIELD_DEF(uint32_t, dimSize)
    TILING_DATA_FIELD_DEF(bool, normBbox)
    TILING_DATA_FIELD_DEF(bool, flipAngle)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Gaussian, GaussianTilingData)
} // namespace optiling

#endif // GAUSSIAN_TILING_H