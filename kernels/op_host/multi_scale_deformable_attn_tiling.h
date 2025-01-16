/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 */
#ifndef MULIT_SCALE_DEFOEMABLE_ATTN_TILING_H
#define MULIT_SCALE_DEFOEMABLE_ATTN_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MultiScaleDeformableAttnTilingData)
TILING_DATA_FIELD_DEF(uint64_t, batchSize)
TILING_DATA_FIELD_DEF(uint64_t, numKeys)
TILING_DATA_FIELD_DEF(uint64_t, numHeads)
TILING_DATA_FIELD_DEF(uint64_t, embedDims)
TILING_DATA_FIELD_DEF(uint64_t, numLevels)
TILING_DATA_FIELD_DEF(uint64_t, numQueries)
TILING_DATA_FIELD_DEF(uint64_t, numPoints)
TILING_DATA_FIELD_DEF(uint32_t, coreNum)
TILING_DATA_FIELD_DEF(uint64_t, realLevels)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(MultiScaleDeformableAttn, MultiScaleDeformableAttnTilingData)
REGISTER_TILING_DATA_CLASS(MultiScaleDeformableAttnGrad, MultiScaleDeformableAttnTilingData)
} // namespace optiling
#endif // MULIT_SCALE_DEFOEMABLE_ATTN_TILING_H
