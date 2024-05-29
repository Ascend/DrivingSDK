/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef MULIT_SCALE_DEFOEMABLE_ATTN_GRAD_TILING_V2_H
#define MULIT_SCALE_DEFOEMABLE_ATTN_GRAD_TILING_V2_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MultiScaleDeformableAttnGradTilingDataV2)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, numKeys)
TILING_DATA_FIELD_DEF(uint32_t, numHeads)
TILING_DATA_FIELD_DEF(uint32_t, embedDims)
TILING_DATA_FIELD_DEF(uint32_t, numLevels)
TILING_DATA_FIELD_DEF(uint32_t, numQueries)
TILING_DATA_FIELD_DEF(uint32_t, numPoints)
TILING_DATA_FIELD_DEF(uint32_t, maxUbNum)
TILING_DATA_FIELD_DEF(uint32_t, coreNum)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MultiScaleDeformableAttnGradV2, MultiScaleDeformableAttnGradTilingDataV2)
} // namespace optiling
#endif // MULIT_SCALE_DEFOEMABLE_ATTN_GRAD_TILING_V2_H
