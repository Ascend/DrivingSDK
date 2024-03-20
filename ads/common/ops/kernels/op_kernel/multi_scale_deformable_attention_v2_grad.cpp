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
 * \file multi_scale_deformable_attention_v2_grad.h
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

namespace {
constexpr static int32_t BUFFER_NUM = 1;
};

class MultiScaleDeformableAttentionV2Grad {
public:
    __aicore__ inline MultiScaleDeformableAttentionV2Grad(){};
    __aicore__ inline void Init(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm, GM_ADDR level_start_index_gm,
                            GM_ADDR sampling_loc_gm, GM_ADDR attn_weight_gm, GM_ADDR grad_output_gm,
                            GM_ADDR grad_value_gm, GM_ADDR grad_sampling_loc_gm, GM_ADDR grad_attn_weight_gm,
                            const MultiScaleDeformableAttentionV2GradTilingData *tiling_data, TPipe *tmpPipe)
    {
        pipe = tmpPipe;
        curBlockIdx = GetBlockIdx();
        blockBytes = 32;
        dataAlign = blockBytes / sizeof(DTYPE_VALUE);

        numKeys = tiling_data->numKeys;
        numHeads = tiling_data->numHeads;
        embedDims = tiling_data->embedDims;
        numLevels = tiling_data->numLevels;
        numQueries = tiling_data->numQueries;
        numPoints = tiling_data->numPoints;
        batchSize = tiling_data->batchSize;
        coreNum = tiling_data->coreNum;

        taskNum = numQueries;
        taskNumPerCore = DivCeil(taskNum, coreNum);

        numPointsAlign = AlignUp(numPoints, dataAlign);
        numLevelsAlign = AlignUp(numLevels, dataAlign);

        startOffset = curBlockIdx * taskNumPerCore;
        endOffset = (curBlockIdx + 1) * taskNumPerCore;
        if (endOffset > taskNum) {
            endOffset = taskNum;
        }

        // offsets
        gradOutStride0 = embedDims;
        gradOutStride1 = numHeads * gradOutStride0;
        gradOutStride2 = numQueries * gradOutStride1;
        weightStride0 = numLevels * numPoints;
        weightStride1 = numHeads * weightStride0;
        weightStride2 = numQueries * weightStride1;
        valueStride0 = embedDims;
        valueStride1 = numHeads * valueStride0;
        valueStride2 = numKeys * valueStride1;

        hOffsetUb = numPointsAlign;
        baseOffsetUb = numPoints * embedDims;

        eventIdVToMte3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdMte2ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdMte3ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());

        copyParamsA = {1, (uint16_t)(numPoints * sizeof(DTYPE_VALUE)), 0, 0};
        sumParams = {numPoints, embedDims, embedDims};

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
    }

    __aicore__ inline void InitBuffer()
    {
        pipe->InitBuffer(shapeUb, BUFFER_NUM, 2 * numLevelsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(offsetUb, BUFFER_NUM, numLevelsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(locationUb, BUFFER_NUM, numHeads * numLevels * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(attentionWeightsUb, BUFFER_NUM, numHeads * numLevels * numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(floatOneUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(topGradUb, BUFFER_NUM, embedDims * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(tmpXUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpYUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(weightSumUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(locWUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(locHUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(imUb, BUFFER_NUM, 2 * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(lowUb, BUFFER_NUM, 2 * numPointsAlign * sizeof(DTYPE_SPATIAL_SHAPES));
        pipe->InitBuffer(highUb, BUFFER_NUM, 2 * numPointsAlign * sizeof(DTYPE_SPATIAL_SHAPES));
        pipe->InitBuffer(lowFloatUb, BUFFER_NUM, 2 * numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(distLowUb, BUFFER_NUM, 2 * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(distHighUb, BUFFER_NUM, 2 * numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(zerosUb, BUFFER_NUM, 8 * numPoints * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(emptyUb, BUFFER_NUM, 2 * numHeads * numLevels * numPoints * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(w1v1Ub, BUFFER_NUM, numPoints * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(w2v2Ub, BUFFER_NUM, numPoints * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(w3v3Ub, BUFFER_NUM, numPoints * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(w4v4Ub, BUFFER_NUM, numPoints * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpUb, BUFFER_NUM, numPoints * embedDims * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(tmpAUb, BUFFER_NUM, embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpBUb, BUFFER_NUM, embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(midUb, BUFFER_NUM, 4 * numPoints * embedDims * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(gradSampleXLocUb, BUFFER_NUM, numPoints * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(gradSampleYLocUb, BUFFER_NUM, numPoints * embedDims * sizeof(DTYPE_VALUE));
    }

    __aicore__ inline void GetLocalTensor()
    {
        locationLocal = locationUb.Get<DTYPE_VALUE>();
        attentionWeightLocal = attentionWeightsUb.Get<DTYPE_VALUE>();
        shapesLocal = shapeUb.Get<DTYPE_SPATIAL_SHAPES>();
        offsetLocal = offsetUb.Get<DTYPE_SPATIAL_SHAPES>();
        xLocal = tmpXUb.Get<DTYPE_VALUE>();
        yLocal = tmpYUb.Get<DTYPE_VALUE>();
        weightSumLocal = weightSumUb.Get<DTYPE_VALUE>();
        floatOneLocal = floatOneUb.Get<DTYPE_VALUE>();
        topGradLocal = topGradUb.Get<DTYPE_VALUE>();
        locWLocal = locWUb.Get<DTYPE_VALUE>();
        locHLocal = locHUb.Get<DTYPE_VALUE>();

        imLocal = imUb.Get<DTYPE_VALUE>();
        lowLocal = lowUb.Get<DTYPE_SPATIAL_SHAPES>();
        highLocal = highUb.Get<DTYPE_SPATIAL_SHAPES>();
        lowFloatLocal = lowFloatUb.Get<DTYPE_VALUE>();
        zerosLocal = zerosUb.Get<DTYPE_VALUE>();
        emptyLocal = emptyUb.Get<DTYPE_VALUE>();

        distLowLocal = distLowUb.Get<DTYPE_VALUE>();
        distHighLocal = distHighUb.Get<DTYPE_VALUE>();

        w1v1Local = w1v1Ub.Get<DTYPE_VALUE>();
        w2v2Local = w2v2Ub.Get<DTYPE_VALUE>();
        w3v3Local = w3v3Ub.Get<DTYPE_VALUE>();
        w4v4Local = w4v4Ub.Get<DTYPE_VALUE>();
        tmpLocal = tmpUb.Get<DTYPE_VALUE>();

        tmpALocal = tmpAUb.Get<DTYPE_VALUE>();
        tmpBLocal = tmpBUb.Get<DTYPE_VALUE>();
        midLocal = midUb.Get<DTYPE_VALUE>();

        gradSampleXLocLocal = gradSampleXLocUb.Get<DTYPE_VALUE>();
        gradSampleYLocLocal = gradSampleYLocUb.Get<DTYPE_VALUE>();
    }

    __aicore__ inline void Process()
    {
        if (curBlockIdx == 0) {
            InitOutput<DTYPE_VALUE>(gradValueGm, batchSize * numKeys * numHeads * embedDims, 0);
        }
        if ASCEND_IS_AIV {
            SyncAll();
        }
        DataCopy(shapesLocal, valueSpatialShapesGm, AlignUp(numLevels * 2, dataAlign));
        DataCopy(offsetLocal, valueLevelStartIndexGm, numLevelsAlign);
        Duplicate<DTYPE_VALUE>(floatOneLocal, (DTYPE_VALUE)1, numPointsAlign);
        Duplicate<DTYPE_VALUE>(emptyLocal, (DTYPE_VALUE)0, numHeads * numLevels * numPoints * 2);
        for (uint32_t taskIdx = startOffset; taskIdx < endOffset; taskIdx++) {
            Compute(taskIdx);
        }
    }

    __aicore__ inline void ReleaseEventID()
    {
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
        pipe->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
        pipe->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV);
    }

private:
    __aicore__ inline void PrepareScalar()
    {
        hIm = imLocal.GetValue(hOffsetUb + point);
        wIm = imLocal.GetValue(point);
        hLow = lowLocal.GetValue(hOffsetUb + point);
        wLow = lowLocal.GetValue(point);
        hHigh = highLocal.GetValue(hOffsetUb + point);
        wHigh = highLocal.GetValue(point);
        hLowPtrOffset = hLow * hStride;
        wLowPtrOffset = wLow * wStride;
        hHighPtrOffset = hHigh * hStride;
        wHighPtrOffset = wHigh * wStride;
    }

    template <bool AddH, bool AddW>
    __aicore__ inline void ComputeGrad(uint32_t midId, uint32_t vId, DTYPE_VALUE distH, DTYPE_VALUE distW,
                                       uint32_t hPtrOffset, uint32_t wPtrOffset, DTYPE_VALUE w)
    {
        uint32_t offsetMid = (point + midId * numPoints) * embedDims;
        uint32_t offsetV = vId * baseOffsetUb;
        uint32_t offsetGradHWeight = pointOffset + gradHWeightId * baseOffsetUb;
        uint32_t offsetGradWWeight = pointOffset + gradWWeightId * baseOffsetUb;
        uint32_t ptr = hPtrOffset + wPtrOffset + basePtr;
        DataCopy(zerosLocal[pointOffset + offsetV], valueGm[offsetValue + ptr], embedDims);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        Muls(midLocal[offsetMid], zerosLocal[pointOffset + topGradValueId * baseOffsetUb], w, embedDims);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Muls(tmpALocal, zerosLocal[pointOffset + offsetV], distW, embedDims);
        Muls(tmpBLocal, zerosLocal[pointOffset + offsetV], distH, embedDims);
        if (AddH) {
            Add(zerosLocal[offsetGradHWeight], zerosLocal[offsetGradHWeight], tmpALocal, embedDims);
        } else {
            Sub(zerosLocal[offsetGradHWeight], zerosLocal[offsetGradHWeight], tmpALocal, embedDims);
        }
        if (AddW) {
            Add(zerosLocal[offsetGradWWeight], zerosLocal[offsetGradWWeight], tmpBLocal, embedDims);
        } else {
            Sub(zerosLocal[offsetGradWWeight], zerosLocal[offsetGradWWeight], tmpBLocal, embedDims);
        }

        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopy(gradValueGm[offsetValue + ptr], midLocal[offsetMid], embedDims);
    }

    __aicore__ inline void Compute(uint32_t query)
    {
        for (batch = 0; batch < batchSize; batch++) {
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            DataCopy(gradWeightGm[batch * weightStride2 + query * weightStride1], emptyLocal, numHeads * numLevels * numPoints);
            DataCopy(gradLocationGm[batch * weightStride2 * 2 + query * weightStride1 * 2], emptyLocal, numHeads * numLevels * numPoints * 2);
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            for (head = 0; head < numHeads; head++) {
                offsetWeight = batch * weightStride2 + query * weightStride1 + head * weightStride0;
                offsetLocation = 2 * offsetWeight;
                basePtr = head * embedDims;
                DataCopy(topGradLocal,
                         gradOutputGm[batch * gradOutStride2 + query * gradOutStride1 + head * gradOutStride0],
                         embedDims);
                SetAtomicAdd<DTYPE_VALUE>();
                for (level = 0; level < numLevels; level++) {
                    levelStartId = offsetLocal.GetValue(level);
                    h = shapesLocal.GetValue(level * 2);
                    w = shapesLocal.GetValue(level * 2 + 1);
                    offsetValue = batch * valueStride2 + levelStartId * valueStride1;
                    wStride = numHeads * embedDims;
                    hStride = w * wStride;
                    DataCopy(locWLocal, locationGm[offsetLocation + level * numPoints * 2], numPointsAlign);
                    DataCopy(locHLocal, locationGm[offsetLocation + level * numPoints * 2 + numPoints], numPointsAlign);
                    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                    DataCopy(attentionWeightLocal, attentionWeightsGm[offsetWeight + level * numPoints],
                             numPointsAlign);
                    Muls(imLocal[hOffsetUb], locHLocal, (DTYPE_VALUE)h, numPointsAlign);
                    Muls(imLocal, locWLocal, (DTYPE_VALUE)w, numPointsAlign);
                    Adds(imLocal, imLocal, DTYPE_VALUE(-0.5), 2 * numPointsAlign);
                    Cast(lowLocal, imLocal, RoundMode::CAST_FLOOR, 2 * numPointsAlign);
                    Adds(highLocal, lowLocal, (DTYPE_SPATIAL_SHAPES)1, 2 * numPointsAlign);
                    Cast(lowFloatLocal, lowLocal, RoundMode::CAST_NONE, 2 * numPointsAlign);

                    Sub(distLowLocal, imLocal, lowFloatLocal, 2 * numPointsAlign);
                    Sub(distHighLocal, floatOneLocal, distLowLocal, 2 * numPointsAlign);

                    Duplicate(zerosLocal, (DTYPE_VALUE)0, 8 * numPoints * embedDims);

                    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                    for (point = 0; point < numPoints; point++) {
                        pointOffset = point * embedDims;
                        PrepareScalar();
                        if (hIm > -1 && wIm > -1 && hIm < h && wIm < w) {
                            Muls(zerosLocal[pointOffset + topGradValueId * baseOffsetUb], topGradLocal,
                                 attentionWeightLocal.GetValue(point), embedDims);
                            if (hLow >= 0) {
                                if (wLow >= 0) {
                                    DTYPE_VALUE distH = distHighLocal.GetValue(hOffsetUb + point);
                                    DTYPE_VALUE distW = distHighLocal.GetValue(point);
                                    w1 = distH * distW;
                                    ComputeGrad<false, false>(0, v1Id, distH, distW, hLowPtrOffset, wLowPtrOffset,
                                                              w1);
                                }
                                if (wHigh < w) {
                                    DTYPE_VALUE distH = distHighLocal.GetValue(hOffsetUb + point);
                                    DTYPE_VALUE distW = distLowLocal.GetValue(point);
                                    w2 = distH * distW;
                                    ComputeGrad<false, true>(1, v2Id, distH, distW, hLowPtrOffset, wHighPtrOffset,
                                                             w2);
                                }
                            }
                            if (hHigh < h) {
                                if (wLow >= 0) {
                                    DTYPE_VALUE distH = distLowLocal.GetValue(hOffsetUb + point);
                                    DTYPE_VALUE distW = distHighLocal.GetValue(point);
                                    w3 = distH * distW;
                                    ComputeGrad<true, false>(2, v3Id, distH, distW, hHighPtrOffset, wLowPtrOffset,
                                                             w3);
                                }
                                if (wHigh < w) {
                                    DTYPE_VALUE distH = distLowLocal.GetValue(hOffsetUb + point);
                                    DTYPE_VALUE distW = distLowLocal.GetValue(point);
                                    w4 = distH * distW;
                                    ComputeGrad<true, true>(3, v4Id, distH, distW, hHighPtrOffset, wHighPtrOffset,
                                                            w4);
                                }
                            }
                            Muls(w1v1Local[pointOffset], zerosLocal[pointOffset + v1Id * baseOffsetUb],
                                 w1, embedDims);
                            Muls(w2v2Local[pointOffset], zerosLocal[pointOffset + v2Id * baseOffsetUb],
                                 w2, embedDims);
                            Muls(w3v3Local[pointOffset], zerosLocal[pointOffset + v3Id * baseOffsetUb],
                                 w3, embedDims);
                            Muls(w4v4Local[pointOffset], zerosLocal[pointOffset + v4Id * baseOffsetUb],
                                 w4, embedDims);
                            Add(w1v1Local[pointOffset], w1v1Local[pointOffset], w2v2Local[pointOffset], embedDims);
                            Add(w1v1Local[pointOffset], w1v1Local[pointOffset], w3v3Local[pointOffset], embedDims);
                            Add(w1v1Local[pointOffset], w1v1Local[pointOffset], w4v4Local[pointOffset], embedDims);
                            Mul(zerosLocal[pointOffset + gradWeightId * baseOffsetUb], topGradLocal,
                                w1v1Local[pointOffset], embedDims);
                        }
                    }
                    Mul(tmpLocal, zerosLocal[topGradValueId * baseOffsetUb], zerosLocal[gradWWeightId * baseOffsetUb],
                        numPoints * embedDims);
                    Muls(gradSampleXLocLocal, tmpLocal, (DTYPE_VALUE)w, numPoints * embedDims);
                    Mul(tmpLocal, zerosLocal[topGradValueId * baseOffsetUb], zerosLocal[gradHWeightId * baseOffsetUb],
                        numPoints * embedDims);
                    Muls(gradSampleYLocLocal, tmpLocal, (DTYPE_VALUE)h, numPoints * embedDims);
                    Sum(xLocal, gradSampleXLocLocal, sumParams);
                    Sum(yLocal, gradSampleYLocLocal, sumParams);
                    Sum(weightSumLocal, zerosLocal[gradWeightId * baseOffsetUb], sumParams);
                    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                    DataCopyPad(gradWeightGm[offsetWeight + level * numPoints], weightSumLocal, copyParamsA);
                    DataCopyPad(gradLocationGm[offsetLocation + level * 2 * numPoints], xLocal, copyParamsA);
                    DataCopyPad(gradLocationGm[offsetLocation + level * 2 * numPoints + numPoints], yLocal, copyParamsA);
                    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
                    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
                }
                SetAtomicNone();
            }
        }
    }

private:
    TPipe *pipe;
    GlobalTensor<DTYPE_VALUE> valueGm, locationGm, attentionWeightsGm, gradOutputGm, gradValueGm, gradLocationGm,
        gradWeightGm;
    GlobalTensor<DTYPE_SPATIAL_SHAPES> valueSpatialShapesGm, valueLevelStartIndexGm;

    TBuf<TPosition::VECCALC> locationUb, attentionWeightsUb, shapeUb, offsetUb;

    TBuf<TPosition::VECCALC> tmpXUb, tmpYUb, weightSumUb;
    TBuf<TPosition::VECCALC> floatOneUb, topGradUb;
    TBuf<TPosition::VECCALC> locWUb, locHUb, imUb, lowUb, highUb, lowFloatUb, hHighPtrOffsetUb, hLowPtrOffsetUb,
        wHighPtrOffsetUb, wLowPtrOffsetUb;

    TBuf<TPosition::VECCALC> distLowUb, distHighUb, w1Ub, w2Ub, w3Ub, w4Ub, zerosUb, emptyUb;

    TBuf<TPosition::VECCALC> w1v1Ub, w2v2Ub, w3v3Ub, w4v4Ub, tmpUb, tmpAUb, tmpBUb, midUb;
    TBuf<TPosition::VECCALC> gradSampleXLocUb, gradSampleYLocUb;

    uint32_t coreNum;
    uint32_t batchSize, numKeys, numHeads, embedDims, numLevels, numQueries, numPoints;
    uint32_t numPointsAlign, numLevelsAlign;

    uint32_t batch, query, head, level, point;

    uint32_t curBlockIdx;
    uint32_t taskNum, taskNumPerCore;
    uint32_t startOffset, endOffset;

    uint32_t dataAlign, blockBytes;

    uint32_t gradOutStride0, gradOutStride1, gradOutStride2;
    uint32_t weightStride0, weightStride1, weightStride2;
    uint32_t valueStride0, valueStride1, valueStride2;
    uint32_t hOffsetUb, baseOffsetUb, pointOffset;
    uint32_t gradHWeightId = 0, gradWWeightId = 1, topGradValueId = 2, gradWeightId = 3;
    uint32_t v1Id = 4, v2Id = 5, v3Id = 6, v4Id = 7;

    DTYPE_SPATIAL_SHAPES h, w, valueOffset, weightOffset, locationOffset, levelStartId, offsetValue;
    DTYPE_SPATIAL_SHAPES offsetWeight, offsetLocation, wStride, hStride, basePtr, ptr;

    DTYPE_VALUE hIm, wIm;
    DTYPE_VALUE w1 = 0, w2 = 0, w3 = 0, w4 = 0;
    DTYPE_SPATIAL_SHAPES hLowPtrOffset, wLowPtrOffset, hHighPtrOffset, wHighPtrOffset;
    DTYPE_SPATIAL_SHAPES hLow, wLow, hHigh, wHigh;

    LocalTensor<DTYPE_SPATIAL_SHAPES> shapesLocal, offsetLocal;
    LocalTensor<DTYPE_SPATIAL_SHAPES> lowLocal, highLocal;
    LocalTensor<DTYPE_VALUE> lowFloatLocal;
    LocalTensor<DTYPE_VALUE> floatOneLocal;
    LocalTensor<DTYPE_VALUE> xLocal, yLocal;
    LocalTensor<DTYPE_VALUE> distLowLocal, distHighLocal;
    LocalTensor<DTYPE_VALUE> locWLocal, locHLocal;
    LocalTensor<DTYPE_VALUE> imLocal;
    LocalTensor<DTYPE_VALUE> zerosLocal, emptyLocal;
    LocalTensor<DTYPE_VALUE> w1v1Local, w2v2Local, w3v3Local, w4v4Local;
    LocalTensor<DTYPE_VALUE> weightSumLocal, midLocal, tmpLocal, tmpALocal, tmpBLocal;
    LocalTensor<DTYPE_VALUE> gradSampleXLocLocal, gradSampleYLocLocal;
    LocalTensor<DTYPE_VALUE> topGradLocal, locationLocal, attentionWeightLocal;

    event_t eventIdVToMte3, eventIdMte2ToV, eventIdMte3ToV;
    DataCopyParams copyParamsA;
    SumParams sumParams;
};

// core func
extern "C" __global__ __aicore__ void multi_scale_deformable_attention_v2_grad(
    GM_ADDR value_gm, GM_ADDR spatial_shapes_gm, GM_ADDR level_start_index_gm, GM_ADDR sampling_loc_gm,
    GM_ADDR attn_weight_gm, GM_ADDR grad_output_gm, GM_ADDR grad_value_gm, GM_ADDR grad_sampling_loc_gm,
    GM_ADDR grad_attn_weight_gm, GM_ADDR workspace, GM_ADDR tiling_data)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_datas, tiling_data);

    MultiScaleDeformableAttentionV2Grad op;
    op.Init(value_gm, spatial_shapes_gm, level_start_index_gm, sampling_loc_gm, attn_weight_gm, grad_output_gm,
            grad_value_gm, grad_sampling_loc_gm, grad_attn_weight_gm, &tiling_datas, &pipe);
    op.InitBuffer();
    op.GetLocalTensor();
    op.Process();
    op.ReleaseEventID();
}