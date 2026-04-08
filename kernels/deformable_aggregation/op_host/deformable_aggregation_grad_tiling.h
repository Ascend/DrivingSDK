/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef DEFORMABLE_AGGREGATION_GRAD_TILING_H
#define DEFORMABLE_AGGREGATION_GRAD_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DeformableAggregationGradTilingData)

TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, avgWeightNum);
TILING_DATA_FIELD_DEF(uint32_t, tailWeightNum);
TILING_DATA_FIELD_DEF(uint32_t, singleProcessTaskLen);
TILING_DATA_FIELD_DEF(uint32_t, numPoints);
TILING_DATA_FIELD_DEF(uint32_t, numCams);
TILING_DATA_FIELD_DEF(uint32_t, numScale);
TILING_DATA_FIELD_DEF(uint32_t, numGroups);
TILING_DATA_FIELD_DEF(uint32_t, numEmbeds);
TILING_DATA_FIELD_DEF(uint32_t, numFeat);
TILING_DATA_FIELD_DEF(uint32_t, numAnchors);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DeformableAggregationGrad, DeformableAggregationGradTilingData)
} // namespace optiling
#endif // DEFORMABLE_AGGREGATION_GRAD_TILING_H
