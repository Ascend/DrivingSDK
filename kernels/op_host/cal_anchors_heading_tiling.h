/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef TIK_TOOLS_TILING_H
#define TIK_TOOLS_TILING_H
#include "register/tilingdata_base.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(CalAnchorsHeadingTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, anchorsNum);
    TILING_DATA_FIELD_DEF(uint32_t, seqLength);
    TILING_DATA_FIELD_DEF(uint32_t, coreAnchorNumTask);
    TILING_DATA_FIELD_DEF(uint32_t, taskMemAlignedByte);
    TILING_DATA_FIELD_DEF(uint32_t, taskElemCountAligned);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreCount);
    TILING_DATA_FIELD_DEF(uint32_t, singleLoopTask);
    TILING_DATA_FIELD_DEF(uint32_t, copyInDataBlockElemCountAligned);
    TILING_DATA_FIELD_DEF(uint16_t, copyInLocalStride);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CalAnchorsHeading, CalAnchorsHeadingTilingData)
}
#endif // TIK_TOOLS_TILING_H