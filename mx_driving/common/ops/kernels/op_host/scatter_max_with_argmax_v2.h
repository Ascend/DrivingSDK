/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef SCATTER_MAX_WITH_ARGMAX_TILING_H
#define SCATTER_MAX_WITH_ARGMAX_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterMaxTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint64_t, tilingMode);
    TILING_DATA_FIELD_DEF(uint64_t, outTailNum);
    TILING_DATA_FIELD_DEF(uint64_t, outEachCore);
    TILING_DATA_FIELD_DEF(uint64_t, outLastCore);
    TILING_DATA_FIELD_DEF(uint64_t, outNum);
    TILING_DATA_FIELD_DEF(uint64_t, indicesNum);
    TILING_DATA_FIELD_DEF(uint64_t, updatesNum);
    TILING_DATA_FIELD_DEF(uint64_t, taskNumPerCore);
    TILING_DATA_FIELD_DEF(uint64_t, taskNumLastCore);
    TILING_DATA_FIELD_DEF(uint64_t, outLineEachTask);
    TILING_DATA_FIELD_DEF(uint64_t, outeachCoreLastNum);
    TILING_DATA_FIELD_DEF(uint64_t, outLastCoreLastNum);
    // indices / updates ub
    TILING_DATA_FIELD_DEF(uint64_t, ubIndicesNum);
    TILING_DATA_FIELD_DEF(uint64_t, ubUpdatesNum);
    TILING_DATA_FIELD_DEF(uint64_t, indicesLoop);
    TILING_DATA_FIELD_DEF(uint64_t, indicesLastNum);
    TILING_DATA_FIELD_DEF(uint64_t, updatesTail);
    TILING_DATA_FIELD_DEF(uint64_t, unpdatesLastNum);
    TILING_DATA_FIELD_DEF(uint64_t, argmaxGap);
    TILING_DATA_FIELD_DEF(uint64_t, initArgmax);
    TILING_DATA_FIELD_DEF(bool, isAligned);
    TILING_DATA_FIELD_DEF(bool, isOneDeal);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterMaxWithArgmaxV2, ScatterMaxTilingData)
}
#endif // SCATTER_MAX_WITH_ARGMAX_TILING_H