/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef BORDER_ALIGN_GRAD_TILING_H
#define BORDER_ALIGN_GRAD_TILING_H
#include "register/tilingdata_base.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(BorderAlignGradTilingData)

TILING_DATA_FIELD_DEF(uint32_t, channels);
TILING_DATA_FIELD_DEF(uint32_t, boxSize);
TILING_DATA_FIELD_DEF(uint32_t, height);
TILING_DATA_FIELD_DEF(uint32_t, width);
TILING_DATA_FIELD_DEF(int64_t, coreCompNum);
TILING_DATA_FIELD_DEF(int64_t, taskLast);
TILING_DATA_FIELD_DEF(uint32_t, poolSize);
TILING_DATA_FIELD_DEF(int64_t, batchSize);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(BorderAlignGrad, BorderAlignGradTilingData)
} // namespace optiling
#endif // GROUP_POINTS_GRAD_TILING_H