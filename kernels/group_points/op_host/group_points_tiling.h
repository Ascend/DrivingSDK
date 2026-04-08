/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef GROUP_POINTS_TILING_H
#define GROUP_POINTS_TILING_H
#include "register/tilingdata_base.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(GroupPointsTilingData)

TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, cSize);
TILING_DATA_FIELD_DEF(uint32_t, nSize);
TILING_DATA_FIELD_DEF(uint32_t, npoints);
TILING_DATA_FIELD_DEF(uint32_t, nsample);
TILING_DATA_FIELD_DEF(uint32_t, cAligned);
TILING_DATA_FIELD_DEF(uint32_t, maxUbTaskNum);
TILING_DATA_FIELD_DEF(uint32_t, coreTaskNum);
TILING_DATA_FIELD_DEF(uint32_t, lastCoreTaskNum);
TILING_DATA_FIELD_DEF(uint32_t, mainCoreLoop);
TILING_DATA_FIELD_DEF(uint32_t, mainCoreTail);
TILING_DATA_FIELD_DEF(uint32_t, lastCoreLoop);
TILING_DATA_FIELD_DEF(uint32_t, lastCoreTail);
TILING_DATA_FIELD_DEF(uint32_t, lastCoreTailAligned);
TILING_DATA_FIELD_DEF(uint32_t, useCoreNum);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(GroupPoints, GroupPointsTilingData)
} // namespace optiling
#endif // GROUP_POINTS_GRAD_TILING_H