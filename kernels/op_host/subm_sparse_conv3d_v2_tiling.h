 /*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef SUBM_SPARSE_CONV3D_V2_TILING_H
#define SUBM_SPARSE_CONV3D_V2_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
 
namespace optiling {
BEGIN_TILING_DATA_DEF(SubmSparseConv3dV2TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, k0);
    TILING_DATA_FIELD_DEF(uint32_t, k1);
    TILING_DATA_FIELD_DEF(uint32_t, k2);
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inChannels);
    TILING_DATA_FIELD_DEF(uint32_t, spatialShape0);
    TILING_DATA_FIELD_DEF(uint32_t, spatialShape1);
    TILING_DATA_FIELD_DEF(uint32_t, spatialShape2);
    TILING_DATA_FIELD_DEF(uint32_t, coreTaskCount);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreCount);
    TILING_DATA_FIELD_DEF(uint32_t, singleLoopTask);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SubmSparseConv3dV2, SubmSparseConv3dV2TilingData)
}
#endif // ADD_CUSTOM_TILING_H