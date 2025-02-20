/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef TIK_TOOLS_TILING_H
#define TIK_TOOLS_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(CartesianToFrenet1TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numPolyLinePoints);
    TILING_DATA_FIELD_DEF(uint32_t, pointDim);
    TILING_DATA_FIELD_DEF(uint32_t, taskSize);
    TILING_DATA_FIELD_DEF(uint32_t, taskSizeElem);
    TILING_DATA_FIELD_DEF(uint32_t, taskSizeAligned);
    TILING_DATA_FIELD_DEF(uint32_t, copyInAlignNum);

    TILING_DATA_FIELD_DEF(uint32_t, dstStride);
    TILING_DATA_FIELD_DEF(uint32_t, rightPadding);

    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileTaskNum);
    TILING_DATA_FIELD_DEF(uint32_t, formerTileNum);
    TILING_DATA_FIELD_DEF(uint32_t, formerTileRemainder);
    TILING_DATA_FIELD_DEF(uint32_t, tailTileNum);
    TILING_DATA_FIELD_DEF(uint32_t, tailTileRemainder);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreCount);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, avgTaskNum);

    TILING_DATA_FIELD_DEF(uint32_t, numTaskCurCore_b);
    TILING_DATA_FIELD_DEF(uint32_t, TaskLengthCurCore_b);
    TILING_DATA_FIELD_DEF(uint32_t, TaskLengthCurCore_s);

    TILING_DATA_FIELD_DEF(uint32_t, tileTaskNum_b);
    TILING_DATA_FIELD_DEF(uint32_t, tileTaskNum_s);
    TILING_DATA_FIELD_DEF(uint32_t, taskResultSizeAligned_b);
    TILING_DATA_FIELD_DEF(uint32_t, axisSizeAligned_b);
    TILING_DATA_FIELD_DEF(uint32_t, taskResultSizeAligned_s);
    TILING_DATA_FIELD_DEF(uint32_t, axisSizeAligned_s);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CartesianToFrenet1, CartesianToFrenet1TilingData)
}
#endif // TIK_TOOLS_TILING_H