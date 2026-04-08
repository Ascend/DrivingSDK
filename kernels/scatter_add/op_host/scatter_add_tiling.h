/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#ifndef SCATTER_MEAN_H
#define SCATTER_MEAN_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterAddTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, outNum);
    TILING_DATA_FIELD_DEF(uint64_t, srcNum);
    TILING_DATA_FIELD_DEF(uint64_t, indicesNum);
    TILING_DATA_FIELD_DEF(uint64_t, bigCoreNum);
    TILING_DATA_FIELD_DEF(uint64_t, tail);
    TILING_DATA_FIELD_DEF(uint64_t, head);
    TILING_DATA_FIELD_DEF(uint64_t, lineSmallCore);
    TILING_DATA_FIELD_DEF(uint64_t, lineBigCore);
    TILING_DATA_FIELD_DEF(uint32_t, taskNum);
    TILING_DATA_FIELD_DEF(uint32_t, taskEachLine);
    TILING_DATA_FIELD_DEF(uint32_t, taskLastLine);
    TILING_DATA_FIELD_DEF(uint64_t, outLineEachCore);
    TILING_DATA_FIELD_DEF(uint64_t, coreEachHead);
    TILING_DATA_FIELD_DEF(uint32_t, taskNumLast);
    TILING_DATA_FIELD_DEF(uint32_t, taskEachLineLast);
    TILING_DATA_FIELD_DEF(uint32_t, taskLastLineLast);
    TILING_DATA_FIELD_DEF(uint32_t, indicesDealNum);
    TILING_DATA_FIELD_DEF(uint64_t, outDimSize);
    TILING_DATA_FIELD_DEF(uint64_t, dimSize);
    TILING_DATA_FIELD_DEF(uint32_t, ubTailNum);
    TILING_DATA_FIELD_DEF(uint64_t, headNumSmallLast);
    TILING_DATA_FIELD_DEF(uint64_t, headNumBigLast);
    TILING_DATA_FIELD_DEF(uint64_t, headNumEachTask);
    TILING_DATA_FIELD_DEF(uint32_t, tilingMode);
    TILING_DATA_FIELD_DEF(uint32_t, dbTimes);
    TILING_DATA_FIELD_DEF(uint32_t, ubSrcNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterAddV2, ScatterAddTilingData)
}

#endif // SCATTER_MEAN_H