/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef GEOMETRIC_KERNEL_ATTN_GRAD_TILING_H
#define GEOMETRIC_KERNEL_ATTN_GRAD_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GeometricKernelAttnGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, numHeads)
TILING_DATA_FIELD_DEF(uint32_t, embedDims)
TILING_DATA_FIELD_DEF(uint32_t, numKeys)
TILING_DATA_FIELD_DEF(uint32_t, numQueries)
TILING_DATA_FIELD_DEF(uint32_t, numLevels)
TILING_DATA_FIELD_DEF(uint32_t, numPoints)
TILING_DATA_FIELD_DEF(uint32_t, coreNum)
TILING_DATA_FIELD_DEF(uint32_t, taskPerCore)
TILING_DATA_FIELD_DEF(uint32_t, taskCoreTail)
TILING_DATA_FIELD_DEF(uint32_t, taskCompNum)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GeometricKernelAttnGrad, GeometricKernelAttnGradTilingData)
} // namespace optiling
#endif // GEOMETRIC_KERNEL_ATTN_GRAD_TILING_H
