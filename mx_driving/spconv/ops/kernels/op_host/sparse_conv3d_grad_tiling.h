 /*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#ifndef SPARSE_CONV3D_GRAD_TILING_H
#define SPARSE_CONV3D_GRAD_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SparseConv3dGradTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
    TILING_DATA_FIELD_DEF(uint32_t, coreTask)
    TILING_DATA_FIELD_DEF(uint32_t, lastCoreTask)
    TILING_DATA_FIELD_DEF(uint32_t, moveLen)
    TILING_DATA_FIELD_DEF(uint32_t, repeatTimes)
    TILING_DATA_FIELD_DEF(uint32_t, moveTail)
    TILING_DATA_FIELD_DEF(uint32_t, lastRepeatTimes)
    TILING_DATA_FIELD_DEF(uint32_t, lastMoveTail)
    TILING_DATA_FIELD_DEF(uint32_t, kernelSize)
    TILING_DATA_FIELD_DEF(uint32_t, kernelIC)
    TILING_DATA_FIELD_DEF(uint32_t, kernelOC)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(SparseConv3dGrad, SparseConv3dGradTilingData)
}
#endif // SPARSE_CONV3D_GRAD_TILING_H