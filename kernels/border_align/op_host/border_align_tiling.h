/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef border_align_tiling_h
#define border_align_tiling_h
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BorderAlignTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, roisNumPerLcore);
    TILING_DATA_FIELD_DEF(uint32_t, roisNumPerScore);
    TILING_DATA_FIELD_DEF(uint32_t, lcoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, scoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, inputBufferSize);
    TILING_DATA_FIELD_DEF(uint32_t, roisBufferSize);
    TILING_DATA_FIELD_DEF(uint32_t, roisNumPerLoop);
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, inputH);
    TILING_DATA_FIELD_DEF(uint32_t, inputW);
    TILING_DATA_FIELD_DEF(uint32_t, channels);
    TILING_DATA_FIELD_DEF(uint32_t, moveInLength);
    TILING_DATA_FIELD_DEF(uint32_t, moveOutLength);
    TILING_DATA_FIELD_DEF(uint32_t, roisNumAligned);
    TILING_DATA_FIELD_DEF(uint32_t, tailNum);
    TILING_DATA_FIELD_DEF(int32_t, pooledSize);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BorderAlign, BorderAlignTilingData)
}
#endif