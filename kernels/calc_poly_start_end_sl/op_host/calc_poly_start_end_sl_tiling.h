
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef TIK_TOOLS_TILING_H
#define TIK_TOOLS_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CalcPolyStartEndSlTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, npoints);
    TILING_DATA_FIELD_DEF(uint32_t, numIdx);
    TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum);
    TILING_DATA_FIELD_DEF(uint32_t, numTaskPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, numTaskRemained);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CalcPolyStartEndSl, CalcPolyStartEndSlTilingData)
}

#endif // TIK_TOOLS_TILING_H