 /*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef TO_SPARSE_TILING_H
#define TO_SPARSE_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(ToSparseTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
    TILING_DATA_FIELD_DEF(uint32_t, coreTask)
    TILING_DATA_FIELD_DEF(uint32_t, lastCoreTask)
    TILING_DATA_FIELD_DEF(uint32_t, moveLen)
    TILING_DATA_FIELD_DEF(uint32_t, repeatTimes)
    TILING_DATA_FIELD_DEF(uint32_t, moveTail)
    TILING_DATA_FIELD_DEF(uint32_t, lastRepeatTimes)
    TILING_DATA_FIELD_DEF(uint32_t, lastMoveTail)
    TILING_DATA_FIELD_DEF(uint32_t, outChannels)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(ToSparse, ToSparseTilingData)
}
#endif // TO_SPARSE_TILING_H