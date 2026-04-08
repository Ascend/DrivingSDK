
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef ASSIGN_SCORE_WITHK_TILING_H
#define ASSIGN_SCORE_WITHK_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
/****************TilingData definition*****************/
BEGIN_TILING_DATA_DEF(AssignScoreWithkTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numCore);
    TILING_DATA_FIELD_DEF(uint64_t, npointPerCore);
    TILING_DATA_FIELD_DEF(uint64_t, npointRemained);
    TILING_DATA_FIELD_DEF(uint32_t, aggregate);
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, nsource);
    TILING_DATA_FIELD_DEF(uint32_t, npoint);
    TILING_DATA_FIELD_DEF(uint32_t, numWeights);
    TILING_DATA_FIELD_DEF(uint32_t, numNeighbors);
    TILING_DATA_FIELD_DEF(uint32_t, numFeatures);
    TILING_DATA_FIELD_DEF_STRUCT(UnPadTiling, unpadTilingData)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AssignScoreWithk, AssignScoreWithkTilingData)
REGISTER_TILING_DATA_CLASS(AssignScoreWithkGrad, AssignScoreWithkTilingData)
}

#endif // ASSIGN_SCORE_WITHK_TILING_H