/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef ROI_ALIGN_ROTATED_GRAD_TILING_H
#define ROI_ALIGN_ROTATED_GRAD_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(RoiAlignRotatedGradV2TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, coreRoisNums);
    TILING_DATA_FIELD_DEF(uint32_t, coreRoisTail);
    TILING_DATA_FIELD_DEF(uint32_t, boxSize);
    TILING_DATA_FIELD_DEF(int32_t, pooledHeight);
    TILING_DATA_FIELD_DEF(int32_t, pooledWidth);
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, channelNum);
    TILING_DATA_FIELD_DEF(uint32_t, width);
    TILING_DATA_FIELD_DEF(uint32_t, height);
    TILING_DATA_FIELD_DEF(bool, aligned);
    TILING_DATA_FIELD_DEF(bool, clockwise);
    TILING_DATA_FIELD_DEF(int32_t, samplingRatio);
    TILING_DATA_FIELD_DEF(float, spatialScale);
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RoiAlignRotatedGradV2, RoiAlignRotatedGradV2TilingData)
} // namespace optiling
#endif // ROI_ALIGN_ROTATED_GRAD_TILING_H