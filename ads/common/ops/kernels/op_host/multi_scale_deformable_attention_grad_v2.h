/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_MUTISCALEDEFORMABLEATTENTIONGRADV2_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_MUTISCALEDEFORMABLEATTENTIONGRADV2_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MultiScaleDeformableAttentionGradV2TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize)
    TILING_DATA_FIELD_DEF(uint32_t, numKeys)
    TILING_DATA_FIELD_DEF(uint32_t, numHeads)
    TILING_DATA_FIELD_DEF(uint32_t, embedDims)
    TILING_DATA_FIELD_DEF(uint32_t, numLevels)
    TILING_DATA_FIELD_DEF(uint32_t, numQueries)
    TILING_DATA_FIELD_DEF(uint32_t, numPoints)
    TILING_DATA_FIELD_DEF(uint32_t, coreNum)
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(MultiScaleDeformableAttentionGradV2, MultiScaleDeformableAttentionGradV2TilingData)
}
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_MUTISCALEDEFORMABLEATTENTIONGRADV2_H
