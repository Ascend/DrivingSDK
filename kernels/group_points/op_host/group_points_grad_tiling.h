/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef GROUP_POINTS_GRAD_TILING_H
#define GROUP_POINTS_GRAD_TILING_H
#include "register/tilingdata_base.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(GroupPointsGradTilingData)

TILING_DATA_FIELD_DEF(uint32_t, b);
TILING_DATA_FIELD_DEF(uint32_t, c);
TILING_DATA_FIELD_DEF(uint32_t, n);
TILING_DATA_FIELD_DEF(uint32_t, npoints);
TILING_DATA_FIELD_DEF(uint32_t, nsample);
TILING_DATA_FIELD_DEF(uint32_t, cAligned);
TILING_DATA_FIELD_DEF(uint32_t, indicesAligned);
TILING_DATA_FIELD_DEF(uint32_t, average);
TILING_DATA_FIELD_DEF(uint32_t, taskLast);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);


END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupPointsGrad, GroupPointsGradTilingData)
} // namespace optiling
#endif // GROUP_POINTS_GRAD_TILING_H