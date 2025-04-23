/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef RADIUS_TILING_H
#define RADIUS_TILING_H

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"

namespace optiling {
/****************TilingData definition*****************/
BEGIN_TILING_DATA_DEF(RadiusTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, numPointsX);
    TILING_DATA_FIELD_DEF(uint32_t, numPointsY);
    TILING_DATA_FIELD_DEF(uint32_t, maxNumNeighbors);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, headCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, batchPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, batchPerCoreTail);
    TILING_DATA_FIELD_DEF(uint32_t, bufferSizePtr);
    TILING_DATA_FIELD_DEF(uint32_t, bufferSizePoints);
    TILING_DATA_FIELD_DEF(uint32_t, numLocalPtr);
    TILING_DATA_FIELD_DEF(uint32_t, numLocalPoints);
    
    TILING_DATA_FIELD_DEF(float, r);
    
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Radius, RadiusTilingData)
}

#endif // RADIUS_TILING_H
