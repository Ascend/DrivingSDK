/*
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
*/
#ifndef MIN_AREA_POLYGONS_TILING_H
#define MIN_AREA_POLYGONS_TILING_H
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
namespace optiling {
/********TilingData definition********/
BEGIN_TILING_DATA_DEF(MinAreaPolygonsTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, pointsetNum)
    TILING_DATA_FIELD_DEF(uint32_t, pointNum)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
    TILING_DATA_FIELD_DEF(uint32_t, coreTask)
    TILING_DATA_FIELD_DEF(uint32_t, lastCoreTask)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(MinAreaPolygons, MinAreaPolygonsTilingData)
}
#endif // MIN_AREA_POLYGONS_TILING_H