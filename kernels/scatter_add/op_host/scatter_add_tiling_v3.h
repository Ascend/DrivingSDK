/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
 #ifndef SCATTER_ADD_TILING_V3_H 
 #define SCATTER_ADD_TILING_V3_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterAddTilingDataV3)
    // base tiling data
    TILING_DATA_FIELD_DEF(uint64_t, totalHead);
    TILING_DATA_FIELD_DEF(uint64_t, tailLen);
    TILING_DATA_FIELD_DEF(uint64_t, dimSize);
    TILING_DATA_FIELD_DEF(uint64_t, srcDimSize);
    TILING_DATA_FIELD_DEF(uint64_t, ubSize);
    TILING_DATA_FIELD_DEF(uint64_t, totalSrcNum);
    TILING_DATA_FIELD_DEF(uint64_t, totalOutNum);
    TILING_DATA_FIELD_DEF(uint64_t, totalIndicesNum);
    TILING_DATA_FIELD_DEF(uint64_t, outNumPerHead);
    TILING_DATA_FIELD_DEF(uint64_t, indicesNumPerHead);
    // TILING_KEY_NO_TAIL_FULLY_LOAD
    TILING_DATA_FIELD_DEF(uint64_t, indicesNumBigCore);
    TILING_DATA_FIELD_DEF(uint64_t, indicesNumSmallCore);
    TILING_DATA_FIELD_DEF(uint64_t, indicesMaxLoadableNumBigCore);
    TILING_DATA_FIELD_DEF(uint64_t, indicesMaxLoadableNumSmallCore);
    // TILING_KEY_NO_TAIL_MULTI_HEADS
    TILING_DATA_FIELD_DEF(uint64_t, bigCoreNum);
    TILING_DATA_FIELD_DEF(uint64_t, headNumBigCore);
    TILING_DATA_FIELD_DEF(uint64_t, headNumSmallCore);
    TILING_DATA_FIELD_DEF(uint64_t, headNumPerTask);
    // TILING_KEY_NO_TAIL_LARGE_HEAD
    TILING_DATA_FIELD_DEF(uint64_t, indicesNumPerBatch);
    TILING_DATA_FIELD_DEF(uint64_t, maxOutNumPerBatch);
    // TILING_KEY_NO_TAIL_FEW_HEADS
    TILING_DATA_FIELD_DEF(uint64_t, coreNumPerHead);
    TILING_DATA_FIELD_DEF(uint64_t, outNumPerCore);
    // tmp TILING_KEY_WITH_TAIL
    TILING_DATA_FIELD_DEF(uint64_t, srcTailBigCore);
    TILING_DATA_FIELD_DEF(uint64_t, srcTailSmallCore);
    TILING_DATA_FIELD_DEF(uint64_t, tailLenThreshold);
    
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterAddV3, ScatterAddTilingDataV3)
}

#endif