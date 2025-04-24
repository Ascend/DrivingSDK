 /*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BatchMatmulVectorTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, coreUsed);
    TILING_DATA_FIELD_DEF(uint64_t, coreData);
    TILING_DATA_FIELD_DEF(uint64_t, copyLoop);
    TILING_DATA_FIELD_DEF(uint64_t, copyTail);
    TILING_DATA_FIELD_DEF(uint64_t, lastCopyLoop);
    TILING_DATA_FIELD_DEF(uint64_t, lastCopyTail);
    TILING_DATA_FIELD_DEF(uint64_t, availableUbSize);
    TILING_DATA_FIELD_DEF(uint64_t, totalResult);
    TILING_DATA_FIELD_DEF(uint64_t, ptsTotal);
    TILING_DATA_FIELD_DEF(uint64_t, dimSizeSecondLast);
    TILING_DATA_FIELD_DEF(uint64_t, dimSizeLast);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchMatmulVector, BatchMatmulVectorTilingData)
}
#endif // ADD_CUSTOM_TILING_H