/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#ifndef SIGMOID_FOCAL_LOSS_TILING_H
#define SIGMOID_FOCAL_LOSS_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SigmoidFocalLossTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numSamples);
    TILING_DATA_FIELD_DEF(uint32_t, numClasses);
    TILING_DATA_FIELD_DEF(uint32_t, numClassesAlign);
    
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, numHeadCores);
    TILING_DATA_FIELD_DEF(uint32_t, numTailCores);
    TILING_DATA_FIELD_DEF(uint32_t, numTaskOnHeadCore);
    TILING_DATA_FIELD_DEF(uint32_t, numTaskOnTailCore);
    
    TILING_DATA_FIELD_DEF(uint32_t, numLoopOnHeadCore);
    TILING_DATA_FIELD_DEF(uint32_t, numTaskPerLoopOnHeadCore);
    TILING_DATA_FIELD_DEF(uint32_t, numTaskTailOnHeadCore);
    TILING_DATA_FIELD_DEF(uint32_t, numLoopOnTailCore);
    TILING_DATA_FIELD_DEF(uint32_t, numTaskPerLoopOnTailCore);
    TILING_DATA_FIELD_DEF(uint32_t, numTaskTailOnTailCore);

    TILING_DATA_FIELD_DEF(float, gamma);
    TILING_DATA_FIELD_DEF(float, alpha);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SigmoidFocalLoss, SigmoidFocalLossTilingData)
REGISTER_TILING_DATA_CLASS(SigmoidFocalLossGrad, SigmoidFocalLossTilingData)
}
#endif