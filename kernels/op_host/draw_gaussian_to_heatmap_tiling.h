/*
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#ifndef DRAW_GAUSSIAN_TO_HEATMAP_TILING_H
#define DRAW_GAUSSIAN_TO_HEATMAP_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DrawGaussianToHeatmapTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
    TILING_DATA_FIELD_DEF(uint32_t, numClasses)
    TILING_DATA_FIELD_DEF(uint32_t, coreTaskLen)
    TILING_DATA_FIELD_DEF(uint32_t, taskObj)
    TILING_DATA_FIELD_DEF(uint32_t, taskRepeatTimes)
    TILING_DATA_FIELD_DEF(uint32_t, singlePorcessCopyLen)
    TILING_DATA_FIELD_DEF(uint32_t, featureMapSizeX)
    TILING_DATA_FIELD_DEF(uint32_t, featureMapSizeY)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DrawGaussianToHeatmap, DrawGaussianToHeatmapTilingData)
} // namespace optiling

#endif // DRAW_GAUSSIAN_TO_HEATMAP_TILING_H