/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef MULIT_SCALE_DEFOEMABLE_ATTN_FUNCTION_TILING_H
#define MULIT_SCALE_DEFOEMABLE_ATTN_FUNCTION_TILING_H
#include "register/tilingdata_base.h"

namespace optiling
{
    BEGIN_TILING_DATA_DEF(MultiScaleDeformableAttnFunctionTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize)
    TILING_DATA_FIELD_DEF(uint32_t, numKeys)
    TILING_DATA_FIELD_DEF(uint32_t, numHeads)
    TILING_DATA_FIELD_DEF(uint32_t, embedDims)
    TILING_DATA_FIELD_DEF(uint32_t, numLevels)
    TILING_DATA_FIELD_DEF(uint32_t, numQueries)
    TILING_DATA_FIELD_DEF(uint32_t, numPoints)
    TILING_DATA_FIELD_DEF(uint32_t, coreNum)

    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(MultiScaleDeformableAttnFunction, MultiScaleDeformableAttnFunctionTilingData)
} // namespace optiling
#endif // MULIT_SCALE_DEFOEMABLE_ATTN_FUNCTION_TILING_H
