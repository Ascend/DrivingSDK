/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#ifndef SCATTER_MAX_TILING_V3_H
#define SCATTER_MAX_TILING_V3_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterMaxTilingDataV3)
    TILING_DATA_FIELD_DEF(uint64_t, srcElemNum);
    TILING_DATA_FIELD_DEF(uint64_t, idxElemNum);
    TILING_DATA_FIELD_DEF(uint64_t, resElemNum);
    TILING_DATA_FIELD_DEF(uint64_t, tailElemNum);
    TILING_DATA_FIELD_DEF(uint64_t, tailSize);
    TILING_DATA_FIELD_DEF(uint64_t, elemNumPerBlock);
    TILING_DATA_FIELD_DEF(uint64_t, idxNumPerCore);
    TILING_DATA_FIELD_DEF(uint64_t, idxBatchNum);
    TILING_DATA_FIELD_DEF(uint64_t, tailBatchNum);
    TILING_DATA_FIELD_DEF(uint64_t, srcBatchNum);
    TILING_DATA_FIELD_DEF(uint64_t, coreNumPerTail);
    TILING_DATA_FIELD_DEF(uint64_t, leftSrcNumBigCore);
    TILING_DATA_FIELD_DEF(uint64_t, leftSrcBigCoreNum);
    TILING_DATA_FIELD_DEF(uint64_t, leftSrcBatchNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterMaxV3, ScatterMaxTilingDataV3)
REGISTER_TILING_DATA_CLASS(ScatterMaxArgmaxV3, ScatterMaxTilingDataV3)
}

#endif // SCATTER_MAX_TILING_V3_H