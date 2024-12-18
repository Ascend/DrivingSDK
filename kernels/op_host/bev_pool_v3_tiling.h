/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef BEV_POOL_V3_TILING_H
#define BEV_POOL_V3_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BEVPoolV3TilingData)
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, avgTaskNum)
TILING_DATA_FIELD_DEF(uint64_t, tailTaskNum)
TILING_DATA_FIELD_DEF(uint64_t, totalTaskNum)
TILING_DATA_FIELD_DEF(uint64_t, avgRankNum)
TILING_DATA_FIELD_DEF(uint64_t, tailRankNum)
TILING_DATA_FIELD_DEF(uint64_t, channel)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(BEVPoolV3, BEVPoolV3TilingData)
REGISTER_TILING_DATA_CLASS(BEVPoolV3Grad, BEVPoolV3TilingData)
} // namespace optiling
#endif // BEV_POOL_TILING_H
