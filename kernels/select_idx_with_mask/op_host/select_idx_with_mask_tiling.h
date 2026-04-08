/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#ifndef TIK_TOOLS_TILING_H
#define TIK_TOOLS_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SelectIdxWithMaskTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, numPoint);
    TILING_DATA_FIELD_DEF(uint32_t, numIdx);
    TILING_DATA_FIELD_DEF(uint32_t, compBatchNum);
    TILING_DATA_FIELD_DEF(uint32_t, numTaskPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, numTaskTail);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SelectIdxWithMask, SelectIdxWithMaskTilingData)
}
#endif // SELECT_IDX_WITH_MASK_TILING_H