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
TILING_DATA_FIELD_DEF(uint32_t, numAnchor);
TILING_DATA_FIELD_DEF(uint32_t, numPoints);
TILING_DATA_FIELD_DEF(uint32_t, numCams);
TILING_DATA_FIELD_DEF(uint32_t, numScale);
TILING_DATA_FIELD_DEF(uint32_t, numGroups);
TILING_DATA_FIELD_DEF(uint32_t, cAligned);
TILING_DATA_FIELD_DEF(uint32_t, singleAligned);
TILING_DATA_FIELD_DEF(uint32_t, average);
TILING_DATA_FIELD_DEF(uint32_t, taskLast);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, groupAligned);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DeformableAggregation, DeformableAggregationTilingData)
} // namespace optiling
#endif // DEFORMABLE_AGGREGATION_TILING_H