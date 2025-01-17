/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterAddGradTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, dimRange);
    TILING_DATA_FIELD_DEF(uint64_t, dimRangeOut);
    TILING_DATA_FIELD_DEF(uint64_t, paramsPro);
    TILING_DATA_FIELD_DEF(uint64_t, gradInUbSize);
    TILING_DATA_FIELD_DEF(uint64_t, indexUbSize);
    TILING_DATA_FIELD_DEF(uint64_t, gradOutUbSize);
    TILING_DATA_FIELD_DEF(uint64_t, indexSumUbSize);
    TILING_DATA_FIELD_DEF(uint64_t, gradInNum);
    TILING_DATA_FIELD_DEF(uint64_t, indexNum);
    TILING_DATA_FIELD_DEF(uint64_t, gradOutNum);
    TILING_DATA_FIELD_DEF(uint64_t, tail);
    TILING_DATA_FIELD_DEF(uint32_t, taskNum);
    TILING_DATA_FIELD_DEF(uint64_t, taskEachLine);
    TILING_DATA_FIELD_DEF(uint64_t, taskLastLine);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);
    TILING_DATA_FIELD_DEF(uint64_t, ubTailNum);
    TILING_DATA_FIELD_DEF(uint64_t, bacthSmallCore);
    TILING_DATA_FIELD_DEF(uint32_t, tilingMode);
    TILING_DATA_FIELD_DEF(uint64_t, headTaskSmall);
    TILING_DATA_FIELD_DEF(uint32_t, taskNumSmall);
    TILING_DATA_FIELD_DEF(uint64_t, headLastTaskSmall);
    TILING_DATA_FIELD_DEF(uint64_t, headTaskBig);
    TILING_DATA_FIELD_DEF(uint32_t, taskNumBig);
    TILING_DATA_FIELD_DEF(uint64_t, headLastTaskBig);
    TILING_DATA_FIELD_DEF(uint32_t, taskEachHead);
    TILING_DATA_FIELD_DEF(uint32_t, dbTimes);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterAddGradV2, ScatterAddGradTilingData)
}