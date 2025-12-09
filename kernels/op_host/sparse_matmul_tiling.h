 /*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef SPARSE_MATMUL_TILING_H
#define SPARSE_MATMUL_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
 
namespace optiling {
BEGIN_TILING_DATA_DEF(SparseMatmulTilingData)
TILING_DATA_FIELD_DEF(uint32_t, k0)
TILING_DATA_FIELD_DEF(uint32_t, k1)
TILING_DATA_FIELD_DEF(uint32_t, k2)
TILING_DATA_FIELD_DEF(uint32_t, inChannels)
TILING_DATA_FIELD_DEF(uint32_t, outChannels)

TILING_DATA_FIELD_DEF(uint32_t, outputCoreTaskCount)
TILING_DATA_FIELD_DEF(uint32_t, outputBigCoreCount)
TILING_DATA_FIELD_DEF(uint32_t, outputSingleLoopTask)
TILING_DATA_FIELD_DEF(uint32_t, outputTaskCount)
TILING_DATA_FIELD_DEF(uint32_t, matmulTaskPerIter)

TILING_DATA_FIELD_DEF(uint32_t, availableUBSize)
TILING_DATA_FIELD_DEF(uint32_t, aivNum)
TILING_DATA_FIELD_DEF(uint32_t, featureBufLen)

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm0TilingData)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SparseMatmul, SparseMatmulTilingData)
}
#endif // ADD_CUSTOM_TILING_H