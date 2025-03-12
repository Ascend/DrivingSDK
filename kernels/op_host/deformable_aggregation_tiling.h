/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef DEFORMABLE_AGGREGATION_TILING_H
#define DEFORMABLE_AGGREGATION_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DeformableAggregationTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bs);
TILING_DATA_FIELD_DEF(uint32_t, numFeats);
TILING_DATA_FIELD_DEF(uint32_t, numEmbeds);
TILING_DATA_FIELD_DEF(uint32_t, numAnchors);
TILING_DATA_FIELD_DEF(uint32_t, numPoints);
TILING_DATA_FIELD_DEF(uint32_t, numCams);
TILING_DATA_FIELD_DEF(uint32_t, numScales);
TILING_DATA_FIELD_DEF(uint32_t, numGroups);
TILING_DATA_FIELD_DEF(uint32_t, cAligned);
TILING_DATA_FIELD_DEF(uint32_t, memoryFlag);
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DeformableAggregation, DeformableAggregationTilingData)
} // namespace optiling
#endif // DEFORMABLE_AGGREGATION_TILING_H