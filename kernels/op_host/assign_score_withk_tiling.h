
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef ASSIGN_SCORE_WITHK_TILING_H
#define ASSIGN_SCORE_WITHK_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
/****************TilingData definition*****************/
BEGIN_TILING_DATA_DEF(AssignScoreWithkTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, num_core);
    TILING_DATA_FIELD_DEF(uint32_t, npoint_per_core);
    TILING_DATA_FIELD_DEF(uint32_t, npoint_remained);
    TILING_DATA_FIELD_DEF(uint32_t, aggregate);
    TILING_DATA_FIELD_DEF(uint32_t, batch_size);
    TILING_DATA_FIELD_DEF(uint32_t, nsource);
    TILING_DATA_FIELD_DEF(uint32_t, npoint);
    TILING_DATA_FIELD_DEF(uint32_t, num_weights);
    TILING_DATA_FIELD_DEF(uint32_t, num_neighbors);
    TILING_DATA_FIELD_DEF(uint32_t, num_features);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AssignScoreWithk, AssignScoreWithkTilingData)
}

#endif // ASSIGN_SCORE_WITHK_TILING_H