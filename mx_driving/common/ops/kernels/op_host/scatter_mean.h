/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef SCATTER_MEAN_H
#define SCATTER_MEAN_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterMeanTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint64_t, outNum);
    TILING_DATA_FIELD_DEF(uint64_t, srcNum);
    TILING_DATA_FIELD_DEF(uint64_t, indicesNum);
    TILING_DATA_FIELD_DEF(uint64_t, bigCoreNum);
    TILING_DATA_FIELD_DEF(uint64_t, tail);
    TILING_DATA_FIELD_DEF(uint64_t, head);
    TILING_DATA_FIELD_DEF(uint64_t, bacthSmallCore);
    TILING_DATA_FIELD_DEF(uint64_t, bacthBigCore);
    TILING_DATA_FIELD_DEF(uint64_t, taskNum);
    TILING_DATA_FIELD_DEF(uint64_t, taskEachLine);
    TILING_DATA_FIELD_DEF(uint64_t, taskLastLine);
    TILING_DATA_FIELD_DEF(uint64_t, outLineEachBacth);
    TILING_DATA_FIELD_DEF(uint64_t, coreEachHead);
    TILING_DATA_FIELD_DEF(uint64_t, taskNumLast);
    TILING_DATA_FIELD_DEF(uint64_t, taskEachLineLast);
    TILING_DATA_FIELD_DEF(uint64_t, taskLastLineLast);
    TILING_DATA_FIELD_DEF(uint64_t, indicesLoop);
    TILING_DATA_FIELD_DEF(uint64_t, indicesLastNum);
    TILING_DATA_FIELD_DEF(uint64_t, ubIndicesNum);
    TILING_DATA_FIELD_DEF(uint64_t, body);
    TILING_DATA_FIELD_DEF(uint64_t, outDimSize);
    TILING_DATA_FIELD_DEF(uint64_t, dimSize);
    TILING_DATA_FIELD_DEF(uint64_t, ubTailNum);
    TILING_DATA_FIELD_DEF(bool, isOneDeal);
    TILING_DATA_FIELD_DEF(uint64_t, headNumSmallLast);
    TILING_DATA_FIELD_DEF(uint64_t, headNumBigLast);
    TILING_DATA_FIELD_DEF(uint64_t, headNumEachTask);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterMean, ScatterMeanTilingData)
}


namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterMeanDivTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint64_t, outNum);
    TILING_DATA_FIELD_DEF(uint64_t, countNum);
    TILING_DATA_FIELD_DEF(uint64_t, srcNum);
    TILING_DATA_FIELD_DEF(uint64_t, bigCoreNum);
    TILING_DATA_FIELD_DEF(uint64_t, tail);
    TILING_DATA_FIELD_DEF(uint64_t, head);
    TILING_DATA_FIELD_DEF(uint64_t, coreSmallLine);
    TILING_DATA_FIELD_DEF(uint64_t, coreBigLine);
    TILING_DATA_FIELD_DEF(uint64_t, taskNum);
    TILING_DATA_FIELD_DEF(uint64_t, taskEachLine);
    TILING_DATA_FIELD_DEF(uint64_t, taskLastLine);
    TILING_DATA_FIELD_DEF(uint64_t, taskEachLineSmall);
    TILING_DATA_FIELD_DEF(uint64_t, taskLastLineSmall);
    TILING_DATA_FIELD_DEF(uint64_t, taskNumSmall);
    TILING_DATA_FIELD_DEF(uint64_t, ubCountNum);
    TILING_DATA_FIELD_DEF(uint64_t, ubTailNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterMeanDiv, ScatterMeanDivTilingData)
}
#endif // SCATTER_MEAN_H