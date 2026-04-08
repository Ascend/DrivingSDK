/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PointsInBoxAllTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, coreData);
    TILING_DATA_FIELD_DEF(uint32_t, copyLoop);
    TILING_DATA_FIELD_DEF(uint32_t, copyTail);
    TILING_DATA_FIELD_DEF(uint32_t, lastCopyLoop);
    TILING_DATA_FIELD_DEF(uint32_t, lastCopyTail);
    TILING_DATA_FIELD_DEF(uint32_t, npoints);
    TILING_DATA_FIELD_DEF(uint32_t, boxNumber);
    TILING_DATA_FIELD_DEF(uint32_t, availableUbSize);
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PointsInBoxAll, PointsInBoxAllTilingData)
}
#endif // ADD_CUSTOM_TILING_H