/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 */

#include "ms_deform_attn_generic.h"
#include "ms_deform_attn_high_perf.h"

extern "C" __global__ __aicore__ void multi_scale_deformable_attn(GM_ADDR value, GM_ADDR valueSpatialShapes,
    GM_ADDR valueLevelStartIndex, GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1002)) {
        KernelMultiScaleDeformableAttnOpt<2, 16> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1004)) {
        KernelMultiScaleDeformableAttnOpt<4, 16> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1008)) {
        KernelMultiScaleDeformableAttnOpt<8, 16> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2002)) {
        KernelMultiScaleDeformableAttnOpt<2, 32> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2004)) {
        KernelMultiScaleDeformableAttnOpt<4, 32> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2008)) {
        KernelMultiScaleDeformableAttnOpt<8, 32> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(4002)) {
        KernelMultiScaleDeformableAttnOpt<2, 64> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(4004)) {
        KernelMultiScaleDeformableAttnOpt<4, 64> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(4008)) {
        KernelMultiScaleDeformableAttnOpt<8, 64> op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations,
            attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        KernelMultiScaleDeformableAttn op;
        op.Init(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations, attentionWeights, output,
            &tilingData, &pipe);
        op.Process();
    }
}
