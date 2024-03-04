/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file multi_scale_deformable_attention_grad.h
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

namespace {
constexpr static int32_t BUFFER_NUM = 1;
};

class MultiScaleDeformableAttentionGrad {
public:
    __aicore__ inline MultiScaleDeformableAttentionGrad(){};
    __aicore__ inline void Init(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm, GM_ADDR level_start_index_gm,
                            GM_ADDR sampling_loc_gm, GM_ADDR attn_weight_gm, GM_ADDR grad_output_gm,
                            GM_ADDR grad_value_gm, GM_ADDR grad_sampling_loc_gm, GM_ADDR grad_attn_weight_gm,
                            MultiScaleDeformableAttentionGradTilingData *tiling_data, TPipe *tmpPipe)
    {
        pipe = tmpPipe;
        curBlockIdx = GetBlockIdx();
        dataAlign = blockBytes / sizeof(DTYPE_VALUE);

        numKeys = tiling_data->numKeys;
        numHeads = tiling_data->numHeads;
        embedDims = tiling_data->embedDims;
        numLevels = tiling_data->numLevels;
        numQueries = tiling_data->numQueries;
        numPoints = tiling_data->numPoints;
        batchSize = tiling_data->batchSize;
        coreNum = tiling_data->coreNum;
        
        wStride = numHeads * embedDims;

        taskNum = numQueries;
        taskNumPerCore = DivCeil(taskNum, coreNum);

        embedDimsAlign = AlignUp(embedDims, dataAlign);
        numPointsAlign = AlignUp(numPoints, dataAlign);
        numLevelsAlign = AlignUp(numLevels, dataAlign);

        batchOffset = numPoints * embedDimsAlign;

        curBlockIdx = GetBlockIdx();
        startOffset = curBlockIdx * taskNumPerCore;
        endOffset = (curBlockIdx + 1) * taskNumPerCore;
        if (endOffset > taskNum) {
            endOffset = taskNum;
        }

        valueGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(value_gm),
                                batchSize * numKeys * numHeads * embedDims);

        valueSpatialShapesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_SPATIAL_SHAPES *>(spatial_shapes_gm),
                                             numLevels * 2);
        valueLevelStartIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_SPATIAL_SHAPES *>(level_start_index_gm),
                                               numLevels);

        locationGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(sampling_loc_gm),
                                   batchSize * numQueries * numHeads * numLevels * numPoints * 2);
        attentionWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(attn_weight_gm),
                                           batchSize * numQueries * numHeads * numLevels * numPoints);

        gradOutputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(grad_output_gm),
                                     batchSize * numQueries * numHeads * embedDims);

        gradValueGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(grad_value_gm),
                                    batchSize * numKeys * numHeads * embedDims);
        gradLocationGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(grad_sampling_loc_gm),
                                       batchSize * numQueries * numHeads * numLevels * 2 * numPoints);
        gradWeightGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(grad_attn_weight_gm),
                                     batchSize * numQueries * numHeads * numLevels * numPoints);

        pipe->InitBuffer(shapeQueue, BUFFER_NUM, AlignUp(numLevels * 2, dataAlign) * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(offsetQueue, BUFFER_NUM, numLevelsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(locationQueue, BUFFER_NUM,
                         AlignUp(numHeads * numLevels * numPoints * 2, dataAlign) * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(attentionWeightsUb, BUFFER_NUM,
                         AlignUp(numHeads * numLevels * numPoints, dataAlign) * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(gradQueue, BUFFER_NUM, embedDimsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(gradValueQueue, BUFFER_NUM,
                         AlignUp(numHeads * numLevels * numPoints * 2, dataAlign) * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(gradLocationQueue, BUFFER_NUM,
                         AlignUp(numHeads * numLevels * numPoints * 2, dataAlign) * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(gradWeightQueue, BUFFER_NUM,
                         AlignUp(numHeads * numLevels * numPoints, dataAlign) * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(floatOneUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(topGradUb, BUFFER_NUM, embedDimsAlign * sizeof(DTYPE_VALUE));


        pipe->InitBuffer(tmpXUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpYUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(weightSumUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(weightQueue, BUFFER_NUM, 4 * numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(valueUb, BUFFER_NUM, batchOffset * 4 * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(locWUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(locHUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(hImUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(wImUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(hLowUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_SPATIAL_SHAPES));
        pipe->InitBuffer(wLowUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_SPATIAL_SHAPES));
        pipe->InitBuffer(hHighUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_SPATIAL_SHAPES));
        pipe->InitBuffer(wHighUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_SPATIAL_SHAPES));

        pipe->InitBuffer(hLowFloatUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(wLowFloatUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(hHighFloatUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(wHighFloatUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(hHighPtrOffsetUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(hLowPtrOffsetUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(wHighPtrOffsetUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(wLowPtrOffsetUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(w1Ub, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(w2Ub, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(w3Ub, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(w4Ub, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(v1Ub, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(v2Ub, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(v3Ub, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(v4Ub, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(lwUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(lhUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(hwUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(hhUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(gradHWeightUb, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(gradWWeightUb, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(topGradValueUb, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(gradWeightUb, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(tmpUb, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmp1Ub, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmp2Ub, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmp3Ub, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmp4Ub, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmp5Ub, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmp6Ub, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmp7Ub, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmp8Ub, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmp9Ub, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmp10Ub, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(tmpAUb, BUFFER_NUM, embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpBUb, BUFFER_NUM, embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(midUb, BUFFER_NUM, 4 * numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(gradSampleXLocUb, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(gradSampleYLocUb, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
    }
    __aicore__ inline void Process()
    {
        for (uint32_t taskIdx = startOffset; taskIdx < endOffset; taskIdx++) {
            SetAtomicAdd<DTYPE_VALUE>();
            Compute(taskIdx);
            SetAtomicNone();
        }
    }

private:
    __aicore__ inline void Compute(uint32_t query)
    {
        LocalTensor<DTYPE_VALUE> locationLocal = locationQueue.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> attentionWeightLocal = attentionWeightsUb.Get<DTYPE_VALUE>();

        LocalTensor<DTYPE_SPATIAL_SHAPES> shapesLocal = shapeQueue.Get<DTYPE_SPATIAL_SHAPES>();
        LocalTensor<DTYPE_SPATIAL_SHAPES> offsetLocal = offsetQueue.Get<DTYPE_SPATIAL_SHAPES>();

        DataCopy(shapesLocal, valueSpatialShapesGm, AlignUp(numLevels * 2, dataAlign));
        DataCopy(offsetLocal, valueLevelStartIndexGm, numLevelsAlign);

        DataCopyParams copyParamsA{1, (uint16_t)(embedDims * sizeof(DTYPE_VALUE)), 0, 0};
        DataCopyParams copyParamsB{1, (uint16_t)(numPoints * sizeof(DTYPE_VALUE)), 0, 0};

        LocalTensor<DTYPE_VALUE> valueLocal = valueUb.Get<DTYPE_VALUE>();

        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());

        for (uint32_t batch = 0; batch < batchSize; batch++) {
            LocalTensor<DTYPE_VALUE> weightLocal = weightQueue.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> xLocal = tmpXUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> yLocal = tmpYUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> weightSumLocal = weightSumUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> floatOneLocal = floatOneUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> topGradLocal = topGradUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> lwLocal = lwUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> lhLocal = lhUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> locWLocal = locWUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> locHLocal = locHUb.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE> hImLocal = hImUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> wImLocal = wImUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_SPATIAL_SHAPES> hLowLocal = hLowUb.Get<DTYPE_SPATIAL_SHAPES>();
            LocalTensor<DTYPE_SPATIAL_SHAPES> wLowLocal = wLowUb.Get<DTYPE_SPATIAL_SHAPES>();
            LocalTensor<DTYPE_SPATIAL_SHAPES> hHighLocal = hHighUb.Get<DTYPE_SPATIAL_SHAPES>();
            LocalTensor<DTYPE_SPATIAL_SHAPES> wHighLocal = wHighUb.Get<DTYPE_SPATIAL_SHAPES>();

            LocalTensor<DTYPE_VALUE> hLowFloatLocal = hLowFloatUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> wLowFloatLocal = wLowFloatUb.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_SPATIAL_SHAPES> hHighPtrOffsetLocal = hHighPtrOffsetUb.Get<DTYPE_SPATIAL_SHAPES>();
            LocalTensor<DTYPE_SPATIAL_SHAPES> hLowPtrOffsetLocal = hLowPtrOffsetUb.Get<DTYPE_SPATIAL_SHAPES>();
            LocalTensor<DTYPE_SPATIAL_SHAPES> wHighPtrOffsetLocal = wHighPtrOffsetUb.Get<DTYPE_SPATIAL_SHAPES>();
            LocalTensor<DTYPE_SPATIAL_SHAPES> wLowPtrOffsetLocal = wLowPtrOffsetUb.Get<DTYPE_SPATIAL_SHAPES>();
            LocalTensor<DTYPE_VALUE> w1Local = w1Ub.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> w2Local = w2Ub.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> w3Local = w3Ub.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> w4Local = w4Ub.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE> v1Local = v1Ub.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> v2Local = v2Ub.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> v3Local = v3Ub.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> v4Local = v4Ub.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE> hwLocal = hwUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> hhLocal = hhUb.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE> gradHWeightLocal = gradHWeightUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> gradWWeightLocal = gradWWeightUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> topGradValueLocal = topGradValueUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> gradWeightLocal = gradWeightUb.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE> tmpLocal = tmpUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> tmp1Local = tmp1Ub.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> tmp2Local = tmp2Ub.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> tmp3Local = tmp3Ub.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> tmp4Local = tmp4Ub.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> tmp5Local = tmp5Ub.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> tmp6Local = tmp6Ub.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> tmp7Local = tmp7Ub.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> tmp8Local = tmp8Ub.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> tmp9Local = tmp9Ub.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> tmp10Local = tmp10Ub.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE> tmpALocal = tmpAUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> tmpBLocal = tmpBUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> midLocal = midUb.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE> gradSampleXLocLocal = gradSampleXLocUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> gradSampleYLocLocal = gradSampleYLocUb.Get<DTYPE_VALUE>();

            Duplicate<DTYPE_VALUE>(floatOneLocal, (DTYPE_VALUE)1, numPointsAlign);
            for (uint32_t head = 0; head < numHeads; head++) {
                offsetWeight = (batch * numQueries * numHeads + query * numHeads + head) * numLevels * numPoints;
                offsetLocation = 2 * offsetWeight;
                DataCopy(topGradLocal, gradOutputGm[batch * numQueries * wStride + query * wStride + head * embedDims],
                         embedDimsAlign);
                for (uint32_t level = 0; level < numLevels; level++) {
                    levelStartId = offsetLocal.GetValue(level);
                    h = shapesLocal.GetValue(level * 2);
                    w = shapesLocal.GetValue(level * 2 + 1);
                    offsetValue = batch * numKeys * numHeads * embedDims + levelStartId * numHeads * embedDims;
                    DataCopy(locWLocal, locationGm[offsetLocation + level * numPoints * 2], numPointsAlign);
                    DataCopy(locHLocal, locationGm[offsetLocation + level * numPoints * 2 + numPoints], numPointsAlign);
                    DataCopy(attentionWeightLocal, attentionWeightsGm[offsetWeight + level * numPoints],
                             numPointsAlign);
                    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                    Muls(hImLocal, locHLocal, (DTYPE_VALUE)h, numPointsAlign);
                    Muls(wImLocal, locWLocal, (DTYPE_VALUE)w, numPointsAlign);
                    Adds(hImLocal, hImLocal, DTYPE_VALUE(-0.5), numPointsAlign);
                    Adds(wImLocal, wImLocal, DTYPE_VALUE(-0.5), numPointsAlign);
                    Cast(hLowLocal, hImLocal, RoundMode::CAST_FLOOR, numPointsAlign);
                    Cast(wLowLocal, wImLocal, RoundMode::CAST_FLOOR, numPointsAlign);
                    Adds(hHighLocal, hLowLocal, (DTYPE_SPATIAL_SHAPES)1, numPointsAlign);
                    Adds(wHighLocal, wLowLocal, (DTYPE_SPATIAL_SHAPES)1, numPointsAlign);

                    Cast(wLowFloatLocal, wLowLocal, RoundMode::CAST_NONE, numPointsAlign);
                    Cast(hLowFloatLocal, hLowLocal, RoundMode::CAST_NONE, numPointsAlign);

                    Sub(lhLocal, hImLocal, hLowFloatLocal, numPointsAlign);
                    Sub(lwLocal, wImLocal, wLowFloatLocal, numPointsAlign);

                    Sub(hhLocal, floatOneLocal, lhLocal, numPointsAlign);
                    Sub(hwLocal, floatOneLocal, lwLocal, numPointsAlign);
                    wStride = numHeads * embedDims;
                    hStride = w * wStride;
                    Muls(hLowPtrOffsetLocal, hLowLocal, hStride, numPointsAlign);
                    Adds(hHighPtrOffsetLocal, hLowPtrOffsetLocal, hStride, numPointsAlign);
                    Muls(wLowPtrOffsetLocal, wLowLocal, wStride, numPointsAlign);
                    Adds(wHighPtrOffsetLocal, wLowPtrOffsetLocal, wStride, numPointsAlign);
                    basePtr = head * embedDims;

                    Mul(w1Local, hhLocal, hwLocal, numPointsAlign);
                    Mul(w2Local, hhLocal, lwLocal, numPointsAlign);
                    Mul(w3Local, lhLocal, hwLocal, numPointsAlign);
                    Mul(w4Local, lhLocal, lwLocal, numPointsAlign);

                    Duplicate(gradHWeightLocal, (DTYPE_VALUE)0, numPoints * embedDimsAlign);
                    Duplicate(gradWWeightLocal, (DTYPE_VALUE)0, numPoints * embedDimsAlign);
                    Duplicate(topGradValueLocal, (DTYPE_VALUE)0, numPoints * embedDimsAlign);
                    Duplicate(gradWeightLocal, (DTYPE_VALUE)0, numPoints * embedDimsAlign);

                    Duplicate(v1Local, (DTYPE_VALUE)0, numPoints * embedDimsAlign);
                    Duplicate(v2Local, (DTYPE_VALUE)0, numPoints * embedDimsAlign);
                    Duplicate(v3Local, (DTYPE_VALUE)0, numPoints * embedDimsAlign);
                    Duplicate(v4Local, (DTYPE_VALUE)0, numPoints * embedDimsAlign);

                    for (uint32_t point = 0; point < numPoints; point++) {
                        if (hImLocal.GetValue(point) > -1 && wImLocal.GetValue(point) > -1 &&
                            hImLocal.GetValue(point) < h && wImLocal.GetValue(point) < w) {
                            Muls(topGradValueLocal[point * embedDimsAlign], topGradLocal,
                                 attentionWeightLocal.GetValue(point), embedDimsAlign);
                            if (hLowLocal.GetValue(point) >= 0) {
                                if (wLowLocal.GetValue(point) >= 0) {
                                    ptr = hLowPtrOffsetLocal.GetValue(point) + wLowPtrOffsetLocal.GetValue(point) +
                                          basePtr;
                                    DataCopy(v1Local[point * embedDimsAlign], valueGm[offsetValue + ptr],
                                             embedDimsAlign);
                                    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                                    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                                    Muls(tmpALocal, v1Local[point * embedDimsAlign], hwLocal.GetValue(point),
                                         embedDims);
                                    Muls(tmpBLocal, v1Local[point * embedDimsAlign], hhLocal.GetValue(point),
                                         embedDims);
                                    Sub(gradHWeightLocal[point * embedDimsAlign],
                                        gradHWeightLocal[point * embedDimsAlign], tmpALocal, embedDims);
                                    Sub(gradWWeightLocal[point * embedDimsAlign],
                                        gradWWeightLocal[point * embedDimsAlign], tmpBLocal, embedDims);
                                    Muls(midLocal[point * embedDimsAlign], topGradValueLocal[point * embedDimsAlign],
                                         w1Local.GetValue(point), embedDims);
                                    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                                    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                                    DataCopyPad(gradValueGm[offsetValue + ptr], midLocal[point * embedDimsAlign],
                                                copyParamsA);
                                }
                                if (wHighLocal.GetValue(point) < w) {
                                    ptr = hLowPtrOffsetLocal.GetValue(point) + wHighPtrOffsetLocal.GetValue(point) +
                                          basePtr;
                                    DataCopy(v2Local[point * embedDimsAlign], valueGm[offsetValue + ptr],
                                             embedDimsAlign);
                                    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                                    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                                    Muls(tmpALocal, v2Local[point * embedDimsAlign], lwLocal.GetValue(point),
                                         embedDims);
                                    Muls(tmpBLocal, v2Local[point * embedDimsAlign], hhLocal.GetValue(point),
                                         embedDims);
                                    Sub(gradHWeightLocal[point * embedDimsAlign],
                                        gradHWeightLocal[point * embedDimsAlign], tmpALocal, embedDims);
                                    Add(gradWWeightLocal[point * embedDimsAlign],
                                        gradWWeightLocal[point * embedDimsAlign], tmpBLocal, embedDims);
                                    Muls(midLocal[point * embedDimsAlign + numPoints * embedDimsAlign],
                                         topGradValueLocal[point * embedDimsAlign], w2Local.GetValue(point), embedDims);
                                    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                                    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                                    DataCopyPad(gradValueGm[offsetValue + ptr],
                                                midLocal[point * embedDimsAlign + numPoints * embedDimsAlign],
                                                copyParamsA);
                                }
                            }
                            if (hHighLocal.GetValue(point) < h) {
                                if (wLowLocal.GetValue(point) >= 0) {
                                    ptr = hHighPtrOffsetLocal.GetValue(point) + wLowPtrOffsetLocal.GetValue(point) +
                                          basePtr;
                                    DataCopy(v3Local[point * embedDimsAlign], valueGm[offsetValue + ptr],
                                             embedDimsAlign);
                                    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                                    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                                    Muls(tmpALocal, v3Local[point * embedDimsAlign], hwLocal.GetValue(point),
                                         embedDims);
                                    Muls(tmpBLocal, v3Local[point * embedDimsAlign], lhLocal.GetValue(point),
                                         embedDims);
                                    Add(gradHWeightLocal[point * embedDimsAlign],
                                        gradHWeightLocal[point * embedDimsAlign], tmpALocal, embedDims);
                                    Sub(gradWWeightLocal[point * embedDimsAlign],
                                        gradWWeightLocal[point * embedDimsAlign], tmpBLocal, embedDims);
                                    Muls(midLocal[point * embedDimsAlign + numPoints * embedDimsAlign * 2],
                                         topGradValueLocal[point * embedDimsAlign], w3Local.GetValue(point), embedDims);
                                    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                                    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                                    DataCopyPad(gradValueGm[offsetValue + ptr],
                                                midLocal[point * embedDimsAlign + numPoints * embedDimsAlign * 2],
                                                copyParamsA);
                                }
                                if (wHighLocal.GetValue(point) < w) {
                                    ptr = hHighPtrOffsetLocal.GetValue(point) + wHighPtrOffsetLocal.GetValue(point) +
                                          basePtr;
                                    DataCopy(v4Local[point * embedDimsAlign], valueGm[offsetValue + ptr],
                                             embedDimsAlign);
                                    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                                    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                                    Muls(tmpALocal, v4Local[point * embedDimsAlign], lwLocal.GetValue(point),
                                         embedDims);
                                    Muls(tmpBLocal, v4Local[point * embedDimsAlign], lhLocal.GetValue(point),
                                         embedDims);
                                    Add(gradHWeightLocal[point * embedDimsAlign],
                                        gradHWeightLocal[point * embedDimsAlign], tmpALocal, embedDims);
                                    Add(gradWWeightLocal[point * embedDimsAlign],
                                        gradWWeightLocal[point * embedDimsAlign], tmpBLocal, embedDims);
                                    Muls(midLocal[point * embedDimsAlign + numPoints * embedDimsAlign * 3],
                                         topGradValueLocal[point * embedDimsAlign], w4Local.GetValue(point), embedDims);
                                    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                                    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                                    DataCopyPad(gradValueGm[offsetValue + ptr],
                                                midLocal[point * embedDimsAlign + numPoints * embedDimsAlign * 3],
                                                copyParamsA);
                                }
                            }
                            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
                            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
                            Muls(tmp1Local[point * embedDimsAlign], v1Local[point * embedDimsAlign],
                                 w1Local.GetValue(point), embedDimsAlign);
                            Muls(tmp2Local[point * embedDimsAlign], v2Local[point * embedDimsAlign],
                                 w2Local.GetValue(point), embedDimsAlign);
                            Muls(tmp3Local[point * embedDimsAlign], v3Local[point * embedDimsAlign],
                                 w3Local.GetValue(point), embedDimsAlign);
                            Muls(tmp4Local[point * embedDimsAlign], v4Local[point * embedDimsAlign],
                                 w4Local.GetValue(point), embedDimsAlign);
                            Add(tmp5Local[point * embedDimsAlign], tmp1Local[point * embedDimsAlign],
                                tmp2Local[point * embedDimsAlign], embedDimsAlign);
                            Add(tmp6Local[point * embedDimsAlign], tmp3Local[point * embedDimsAlign],
                                tmp4Local[point * embedDimsAlign], embedDimsAlign);
                            Add(tmp7Local[point * embedDimsAlign], tmp5Local[point * embedDimsAlign],
                                tmp6Local[point * embedDimsAlign], embedDimsAlign);
                            Mul(gradWeightLocal[point * embedDimsAlign], topGradLocal,
                                tmp7Local[point * embedDimsAlign], embedDimsAlign);
                        }
                    }
                    Mul(tmp9Local, topGradValueLocal, gradWWeightLocal, numPoints * embedDimsAlign);
                    Muls(gradSampleXLocLocal, tmp9Local, (DTYPE_VALUE)w, numPoints * embedDimsAlign);
                    Mul(tmp10Local, topGradValueLocal, gradHWeightLocal, numPoints * embedDimsAlign);
                    Muls(gradSampleYLocLocal, tmp10Local, (DTYPE_VALUE)h, numPoints * embedDimsAlign);
                    SumParams sumParams{numPoints, embedDimsAlign, embedDims};
                    Sum(xLocal, gradSampleXLocLocal, sumParams);
                    Sum(yLocal, gradSampleYLocLocal, sumParams);
                    Sum(weightSumLocal, gradWeightLocal, sumParams);
                    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                    DataCopyPad(gradWeightGm[offsetWeight + level * numPoints], weightSumLocal, copyParamsB);
                    DataCopyPad(gradLocationGm[offsetLocation + level * 2 * numPoints], xLocal, copyParamsB);
                    DataCopyPad(gradLocationGm[offsetLocation + level * 2 * numPoints + numPoints], yLocal,
                                copyParamsB);
                }
            }
        }
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV);
    }

private:
    TPipe *pipe;
    GlobalTensor<DTYPE_VALUE> valueGm, locationGm, attentionWeightsGm, gradOutputGm, gradValueGm, gradLocationGm,
        gradWeightGm;
    GlobalTensor<DTYPE_SPATIAL_SHAPES> valueSpatialShapesGm, valueLevelStartIndexGm;

    TBuf<TPosition::VECCALC> locationQueue, attentionWeightsUb, shapeQueue, offsetQueue, gradQueue;
    TBuf<TPosition::VECCALC> gradValueQueue, gradLocationQueue, gradWeightQueue;

    TBuf<TPosition::VECCALC> tmpXUb, tmpYUb, weightSumUb;
    TBuf<TPosition::VECCALC> intOneUb, floatOneUb, weightQueue, emptyUb, topGradUb;
    TBuf<TPosition::VECCALC> valueUb, locWUb, locHUb, hImUb, wImUb, hLowUb, wLowUb, hHighUb, wHighUb, hLowFloatUb,
        wLowFloatUb, hHighFloatUb, wHighFloatUb, hHighPtrOffsetUb, hLowPtrOffsetUb, wHighPtrOffsetUb, wLowPtrOffsetUb;

    TBuf<TPosition::VECCALC> lwUb, lhUb, hwUb, hhUb, w1Ub, w2Ub, w3Ub, w4Ub, v1Ub, v2Ub, v3Ub, v4Ub;

    TBuf<TPosition::VECCALC> tmpUb, tmp1Ub, tmp2Ub, tmp3Ub, tmp4Ub, tmp5Ub, tmp6Ub, tmp7Ub, tmp8Ub, tmp9Ub, tmp10Ub,
        tmpAUb, tmpBUb, midUb;
    TBuf<TPosition::VECCALC> gradHWeightUb, gradWWeightUb, topGradValueUb, gradWeightUb, gradSampleXLocUb,
        gradSampleYLocUb;

    uint32_t batchSize;
    uint32_t numKeys;
    uint32_t numHeads;
    uint32_t embedDims;

    uint32_t numLevels;
    uint32_t numQueries;
    uint32_t numPoints;
    uint32_t coreNum;

    uint32_t embedDimsAlign;
    uint32_t numPointsAlign;
    uint32_t numLevelsAlign;

    uint32_t batch;
    uint32_t query;
    uint32_t head;

    uint32_t taskNum;
    uint32_t taskNumPerCore;
    uint32_t curBlockIdx;
    uint32_t startOffset;
    uint32_t endOffset;
    uint32_t dataAlign;
    uint32_t blockBytes = 32;

    DTYPE_VALUE tmp1, tmp2, leftTopWeight, rightTopWeiight, leftBottomWeight, rightBottomWeight, attnWeight;
    DTYPE_SPATIAL_SHAPES h, w, x0, y0, x1, y1, valueOffset, weightOffset, locationOffset, batchOffset, levelStartId,
        offsetValue;

    DTYPE_SPATIAL_SHAPES offsetWeight, offsetLocation, wStride, hStride, basePtr, ptr;
};

// core func
extern "C" __global__ __aicore__ void multi_scale_deformable_attention_grad(
    GM_ADDR value_gm, GM_ADDR spatial_shapes_gm, GM_ADDR level_start_index_gm, GM_ADDR sampling_loc_gm,
    GM_ADDR attn_weight_gm, GM_ADDR grad_output_gm, GM_ADDR grad_value_gm, GM_ADDR grad_sampling_loc_gm,
    GM_ADDR grad_attn_weight_gm, GM_ADDR workspace, GM_ADDR tiling_data)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_datas, tiling_data);

    MultiScaleDeformableAttentionGrad op;
    op.Init(value_gm, spatial_shapes_gm, level_start_index_gm, sampling_loc_gm, attn_weight_gm, grad_output_gm,
            grad_value_gm, grad_sampling_loc_gm, grad_attn_weight_gm, &tiling_datas, &pipe);

    op.Process();
}
