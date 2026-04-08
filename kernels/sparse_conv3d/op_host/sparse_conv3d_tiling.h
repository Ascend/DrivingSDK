 /*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#ifndef SPARSE_CONV3D_TILING_H
#define SPARSE_CONV3D_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SparseConv3dTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
    TILING_DATA_FIELD_DEF(uint32_t, coreTask)
    TILING_DATA_FIELD_DEF(uint32_t, lastCoreTask)
    TILING_DATA_FIELD_DEF(uint32_t, moveLen)
    TILING_DATA_FIELD_DEF(uint32_t, repeatTimes)
    TILING_DATA_FIELD_DEF(uint32_t, moveTail)
    TILING_DATA_FIELD_DEF(uint32_t, lastRepeatTimes)
    TILING_DATA_FIELD_DEF(uint32_t, lastMoveTail)
    TILING_DATA_FIELD_DEF(uint32_t, kernelD)
    TILING_DATA_FIELD_DEF(uint32_t, kernelH)
    TILING_DATA_FIELD_DEF(uint32_t, kernelW)
    TILING_DATA_FIELD_DEF(uint32_t, kernelSize)
    TILING_DATA_FIELD_DEF(uint32_t, outfeatureB)
    TILING_DATA_FIELD_DEF(uint32_t, outputDepth)
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight)
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth)
    TILING_DATA_FIELD_DEF(uint32_t, strideDepth)
    TILING_DATA_FIELD_DEF(uint32_t, strideHeight)
    TILING_DATA_FIELD_DEF(uint32_t, strideWidth)
    TILING_DATA_FIELD_DEF(uint32_t, paddingDepth)
    TILING_DATA_FIELD_DEF(uint32_t, paddingHeight)
    TILING_DATA_FIELD_DEF(uint32_t, paddingWidth)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(SparseConv3d, SparseConv3dTilingData)
}
#endif // SPARSE_CONV_TILING_H

