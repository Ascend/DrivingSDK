 /*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef SUBM_SPARSE_CONV3D_V3_TILING_H
#define SUBM_SPARSE_CONV3D_V3_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
 
namespace optiling {
BEGIN_TILING_DATA_DEF(SubmSparseConv3dV3TilingData)
TILING_DATA_FIELD_DEF(uint32_t, k0)
TILING_DATA_FIELD_DEF(uint32_t, k1)
TILING_DATA_FIELD_DEF(uint32_t, k2)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, withKey)
TILING_DATA_FIELD_DEF(uint32_t, inChannels)
TILING_DATA_FIELD_DEF(uint32_t, outChannels)
TILING_DATA_FIELD_DEF(uint32_t, spatialShape0)
TILING_DATA_FIELD_DEF(uint32_t, spatialShape1)
TILING_DATA_FIELD_DEF(uint32_t, spatialShape2)
TILING_DATA_FIELD_DEF(uint32_t, coreTaskCount)
TILING_DATA_FIELD_DEF(uint32_t, bigCoreCount)
TILING_DATA_FIELD_DEF(uint32_t, singleLoopTask)
TILING_DATA_FIELD_DEF(uint32_t, totalTaskCount)
TILING_DATA_FIELD_DEF(uint32_t, availableUBSize)
TILING_DATA_FIELD_DEF(uint32_t, featureBufLen)
TILING_DATA_FIELD_DEF(uint32_t, stage2SingleLoopTask)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm0TilingData)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SubmSparseConv3dV3, SubmSparseConv3dV3TilingData)
}
#endif // ADD_CUSTOM_TILING_H