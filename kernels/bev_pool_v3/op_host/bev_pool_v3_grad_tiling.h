/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#ifndef BEV_POOL_V3_GRAD_TILING_H
#define BEV_POOL_V3_GRAD_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BEVPoolGradV3TilingData)
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, avgTaskNum)
TILING_DATA_FIELD_DEF(uint64_t, tailTaskNum)
TILING_DATA_FIELD_DEF(uint64_t, totalTaskNum)
TILING_DATA_FIELD_DEF(uint64_t, avgRankNum)
TILING_DATA_FIELD_DEF(uint64_t, tailRankNum)
TILING_DATA_FIELD_DEF(uint64_t, channel)
TILING_DATA_FIELD_DEF(uint64_t, rankStep)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(BEVPoolV3Grad, BEVPoolGradV3TilingData)
} // namespace optiling
#endif // BEV_POOL_V3_GRAD_TILING_H
