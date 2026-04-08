/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef BEV_POOL_TILING_H
#define BEV_POOL_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BEVPoolTilingData)
TILING_DATA_FIELD_DEF(uint64_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, avgTaskNum)
TILING_DATA_FIELD_DEF(uint64_t, tailTaskNum)
TILING_DATA_FIELD_DEF(uint64_t, totalTaskNum)
TILING_DATA_FIELD_DEF(uint64_t, stride0)
TILING_DATA_FIELD_DEF(uint64_t, stride1)
TILING_DATA_FIELD_DEF(uint64_t, stride2)
TILING_DATA_FIELD_DEF(uint64_t, stride3)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(BEVPool, BEVPoolTilingData)
REGISTER_TILING_DATA_CLASS(BEVPoolGrad, BEVPoolTilingData)
REGISTER_TILING_DATA_CLASS(BEVPoolV2, BEVPoolTilingData)
REGISTER_TILING_DATA_CLASS(BEVPoolV2Grad, BEVPoolTilingData)
} // namespace optiling
#endif // BEV_POOL_TILING_H
