/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
        valueStride1 = numKeys * valueStride0;
        valueStride2 = numHeads * valueStride1;

        baseOffsetUb = numPoints * embedDims;

        eventIdMte2ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdMte3ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());
        eventIdVToMte2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMte3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdVToMteWeight = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdVToMte3X = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdVToMte3Y = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());

        copyParams = {1, (uint16_t)(numPoints * sizeof(DTYPE_VALUE)), 0, 0};
        sumParams = {numPoints, embedDims, embedDims};

        copyParamsV2 = {1, (uint16_t)(2 * numPoints * sizeof(DTYPE_VALUE)), 0, 0};
        sumParamsV2 = {2 * numPoints, embedDims, embedDims};

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
        pipe->InitBuffer(shapeUb, 2 * numLevelsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(offsetUb, numLevelsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(locationUb, numHeads * numLevels * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(attentionWeightsUb, numHeads * numLevels * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(topGradUb, embedDims * sizeof(DTYPE_VALUE));
        
        pipe->InitBuffer(floatOneUb, 2 * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(locSumUb, 2 * numLevels * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(weightSumUb, 2 * numLevels * numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(locWUb, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(locHUb, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(imUb, 2 * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(lowUb, 2 * numPointsAlign * sizeof(DTYPE_SPATIAL_SHAPES));
        pipe->InitBuffer(lowFloatUb, 2 * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(distLowUb, 2 * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(distHighUb, 2 * numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(zerosUb, 8 * numPoints * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(w1v1Ub, numPoints * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(w2v2Ub, numPoints * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(w3v3Ub, numPoints * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(w4v4Ub, numPoints * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpUb, 2 * numPoints * embedDims * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(tmpAUb, embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpBUb, embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(midUb, 4 * numPoints * embedDims * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(gradSampleXLocUb, 2 * numPoints * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(gradSampleYLocUb, 2 * numPoints * embedDims * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(whUb, 2 * numPoints * embedDims * sizeof(DTYPE_VALUE));

        
    }

    __aicore__ inline void GetLocalTensor()
    {
        locationLocal = locationUb.Get<DTYPE_VALUE>();
        attentionWeightLocal = attentionWeightsUb.Get<DTYPE_VALUE>();
        shapesLocal = shapeUb.Get<DTYPE_SPATIAL_SHAPES>();
        offsetLocal = offsetUb.Get<DTYPE_SPATIAL_SHAPES>();
        locSumLocal = locSumUb.Get<DTYPE_VALUE>();

        weightSumLocal = weightSumUb.Get<DTYPE_VALUE>();
        floatOneLocal = floatOneUb.Get<DTYPE_VALUE>();
        topGradLocal = topGradUb.Get<DTYPE_VALUE>();
        locWLocal = locWUb.Get<DTYPE_VALUE>();
        locHLocal = locHUb.Get<DTYPE_VALUE>();

        imLocal = imUb.Get<DTYPE_VALUE>();
        lowLocal = lowUb.Get<DTYPE_SPATIAL_SHAPES>();
        lowFloatLocal = lowFloatUb.Get<DTYPE_VALUE>();
        zerosLocal = zerosUb.Get<DTYPE_VALUE>();

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

        whLocLocal = whUb.Get<DTYPE_VALUE>();
    }
    
    __aicore__ inline void ClearOutput()
    {
        switch (curBlockIdx) {
            case 0:
                InitOutput<DTYPE_VALUE>(gradValueGm, batchSize * numKeys * numHeads * embedDims, 0);
                break;
            case 1:
                InitOutput<DTYPE_VALUE>(gradLocationGm, 2 * batchSize * numQueries * numHeads * numLevels * numPoints);
                break;
            case 2:
                InitOutput<DTYPE_VALUE>(gradWeightGm, batchSize * numQueries * numHeads * numLevels * numPoints);
                break;
            default:
                break;
        }
        if ASCEND_IS_AIV {
            SyncAll();
        }
    }

    __aicore__ inline void Process()
    {
        DataCopy(shapesLocal, valueSpatialShapesGm, 2 * numLevelsAlign);
        DataCopy(offsetLocal, valueLevelStartIndexGm, numLevelsAlign);
        Duplicate<DTYPE_VALUE>(floatOneLocal, (DTYPE_VALUE)1, 2 * numPointsAlign);

        SetAtomicAdd<DTYPE_VALUE>();
        if (embedDims == 32 && numPoints == 2) {
            ComputeOpt<2>();
        } else if (embedDims == 32 && numPoints == 4) {
            ComputeOpt<4>();
        } else if (embedDims == 32 && numPoints == 8) {
            ComputeOpt<8>();
        } else {
            Compute();
        }
        SetAtomicNone();
    }

    __aicore__ inline void ReleaseEventID()
    {
        pipe->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
        pipe->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV);
        pipe->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMteWeight);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3X);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3Y);
    }

private:
    template <bool AddH, bool AddW>
    __aicore__ inline void ComputeGrad(uint32_t midId, uint32_t vId, DTYPE_VALUE distH, DTYPE_VALUE distW,
                                       uint32_t hPtrOffset, uint32_t wPtrOffset, DTYPE_VALUE w)
    {
        uint32_t offsetMid = (point + midId * numPoints) * embedDims;
        uint32_t offsetV = vId * baseOffsetUb;
        uint32_t offsetGradHWeight = pointOffset + gradHWeightId * baseOffsetUb;
        uint32_t offsetGradWWeight = pointOffset + gradWWeightId * baseOffsetUb;
        uint32_t ptr = hPtrOffset + wPtrOffset;
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

    template<uint32_t NUM_POINTS>
    __aicore__ inline void ComputeOpt()
    {
        for (query = startOffset; query < endOffset; query++) {
            for (batch = 0; batch < batchSize; batch++) {
                for (head = 0; head < numHeads; head++) {
                    offsetWeight = batch * weightStride2 + query * weightStride1 + head * weightStride0;
                    offsetLocation = 2 * offsetWeight;
                    DataCopy(topGradLocal,
                            gradOutputGm[batch * gradOutStride2 + query * gradOutStride1 + head * gradOutStride0],
                            embedDimsOpt);
                    for (level = 0; level < numLevels; level++) {
                        levelStartId = offsetLocal.GetValue(level);
                        h = shapesLocal.GetValue(level * 2);
                        w = shapesLocal.GetValue(level * 2 + 1);

                        Duplicate(whLocLocal, (DTYPE_VALUE)w, NUM_POINTS * embedDimsOpt);
                        Duplicate(whLocLocal[NUM_POINTS * embedDimsOpt], (DTYPE_VALUE)h, NUM_POINTS * embedDimsOpt);

                        offsetValue = batch * valueStride2 + head * valueStride1 + levelStartId * valueStride0;
                        wStride = embedDimsOpt;
                        hStride = w * wStride;

                        tmpOffset1 = offsetWeight + level * NUM_POINTS;

                        DataCopy(locWLocal, locationGm[tmpOffset1 * 2], numPointsAlign);
                        DataCopy(locHLocal, locationGm[tmpOffset1 * 2 + NUM_POINTS], numPointsAlign);
                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                        DataCopy(attentionWeightLocal, attentionWeightsGm[tmpOffset1], numPointsAlign);
                        Muls(imLocal[numPointsAlign], locHLocal, (DTYPE_VALUE)h, numPointsAlign);
                        Muls(imLocal, locWLocal, (DTYPE_VALUE)w, numPointsAlign);
                        Adds(imLocal, imLocal, DTYPE_VALUE(-0.5), 2 * numPointsAlign);
                        Cast(lowLocal, imLocal, RoundMode::CAST_FLOOR, 2 * numPointsAlign);
                        Cast(lowFloatLocal, lowLocal, RoundMode::CAST_NONE, 2 * numPointsAlign);

                        Sub(distLowLocal, imLocal, lowFloatLocal, 2 * numPointsAlign);
                        Sub(distHighLocal, floatOneLocal, distLowLocal, 2 * numPointsAlign);

                        Duplicate(zerosLocal, (DTYPE_VALUE)0, 8 * NUM_POINTS * embedDimsOpt);

                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                        for (point = 0; point < NUM_POINTS; point++) {
                            pointOffset = point * embedDimsOpt;
                            hIm = imLocal.GetValue(numPointsAlign + point);
                            wIm = imLocal.GetValue(point);
                            if (hIm > -1 && wIm > -1 && hIm < h && wIm < w) {
                                hLow = lowLocal.GetValue(numPointsAlign + point);
                                wLow = lowLocal.GetValue(point);
                                hLowPtrOffset = hLow * hStride;
                                wLowPtrOffset = wLow * wStride;
                                Muls(zerosLocal[pointOffset + topGradValueId * NUM_POINTS * embedDimsOpt], topGradLocal,
                                    attentionWeightLocal.GetValue(point), embedDimsOpt);
                                if (hLow >= 0) {
                                    if (wLow >= 0 && wLow < w - 1) {
                                        DTYPE_VALUE distH = distHighLocal.GetValue(numPointsAlign + point);
                                        DTYPE_VALUE distW = distHighLocal.GetValue(point);
                                        w1 = distH * distW;
                                        DTYPE_VALUE distW2 = distLowLocal.GetValue(point);
                                        w2 = distH * distW2;
                                        uint32_t offsetMid = point * 4 * embedDimsOpt;
                                        uint32_t offsetV = v1Id * NUM_POINTS * embedDimsOpt;
                                        uint32_t offsetMid_ = (point * 4 + mid2Id) * embedDimsOpt;
                                        uint32_t offsetV_ = v2Id * NUM_POINTS * embedDimsOpt;
                                        uint32_t offsetGradHWeight = pointOffset + gradHWeightId * NUM_POINTS * embedDimsOpt;
                                        uint32_t offsetGradWWeight = pointOffset + gradWWeightId * NUM_POINTS * embedDimsOpt;
                                        uint32_t ptr = hLowPtrOffset + wLowPtrOffset;
                                        DataCopy(zerosLocal[pointOffset + offsetV], valueGm[offsetValue + ptr], embedDimsOpt);
                                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                                        Muls(midLocal[offsetMid], zerosLocal[pointOffset + topGradValueId * NUM_POINTS * embedDimsOpt], w1, embedDimsOpt);

                                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                                        Muls(tmpALocal, zerosLocal[pointOffset + offsetV], distW, embedDimsOpt);
                                        Muls(tmpBLocal, zerosLocal[pointOffset + offsetV], distH, embedDimsOpt);
                                        Sub(zerosLocal[offsetGradHWeight], zerosLocal[offsetGradHWeight], tmpALocal, embedDimsOpt);
                                        Sub(zerosLocal[offsetGradWWeight], zerosLocal[offsetGradWWeight], tmpBLocal, embedDimsOpt);


                                        DataCopy(zerosLocal[pointOffset + offsetV_], valueGm[offsetValue + ptr + wStride], embedDimsOpt);
                                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                                        Muls(midLocal[offsetMid_], zerosLocal[pointOffset + topGradValueId * NUM_POINTS * embedDimsOpt], w2, embedDimsOpt);
                                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                                        Muls(tmpALocal, zerosLocal[pointOffset + offsetV_], distW2, embedDimsOpt);
                                        Muls(tmpBLocal, zerosLocal[pointOffset + offsetV_], distH, embedDimsOpt);
                                        Sub(zerosLocal[offsetGradHWeight], zerosLocal[offsetGradHWeight], tmpALocal, embedDimsOpt);
                                        Add(zerosLocal[offsetGradWWeight], zerosLocal[offsetGradWWeight], tmpBLocal, embedDimsOpt);

                                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                                        DataCopy(gradValueGm[offsetValue + ptr], midLocal[offsetMid], embedDimsOptTwice);

                                    } else if (wLow >= 0) {
                                        DTYPE_VALUE distH = distHighLocal.GetValue(numPointsAlign + point);
                                        DTYPE_VALUE distW = distHighLocal.GetValue(point);
                                        w1 = distH * distW;

                                        uint32_t offsetMid = (point * 4 + mid1Id) * embedDimsOpt;
                                        uint32_t offsetV = v1Id * NUM_POINTS * embedDimsOpt;
                                        uint32_t offsetGradHWeight = pointOffset + gradHWeightId * NUM_POINTS * embedDimsOpt;
                                        uint32_t offsetGradWWeight = pointOffset + gradWWeightId * NUM_POINTS * embedDimsOpt;
                                        uint32_t ptr = hLowPtrOffset + wLowPtrOffset;
                                        DataCopy(zerosLocal[pointOffset + offsetV], valueGm[offsetValue + ptr], embedDimsOpt);
                                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                                        Muls(midLocal[offsetMid], zerosLocal[pointOffset + topGradValueId * NUM_POINTS * embedDimsOpt], w1, embedDimsOpt);
                                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                                        Muls(tmpALocal, zerosLocal[pointOffset + offsetV], distW, embedDimsOpt);
                                        Muls(tmpBLocal, zerosLocal[pointOffset + offsetV], distH, embedDimsOpt);
                                        Sub(zerosLocal[offsetGradHWeight], zerosLocal[offsetGradHWeight], tmpALocal, embedDimsOpt);
                                        Sub(zerosLocal[offsetGradWWeight], zerosLocal[offsetGradWWeight], tmpBLocal, embedDimsOpt);

                                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                                        DataCopy(gradValueGm[offsetValue + ptr], midLocal[offsetMid], embedDimsOpt);
                                    } else if (wLow < w - 1) {
                                        DTYPE_VALUE distH = distHighLocal.GetValue(numPointsAlign + point);
                                        DTYPE_VALUE distW = distLowLocal.GetValue(point);
                                        w2 = distH * distW;

                                        uint32_t offsetMid = (point * 4 + mid2Id) * embedDimsOpt;
                                        uint32_t offsetV = v2Id * NUM_POINTS * embedDimsOpt;
                                        uint32_t offsetGradHWeight = pointOffset + gradHWeightId * NUM_POINTS * embedDimsOpt;
                                        uint32_t offsetGradWWeight = pointOffset + gradWWeightId * NUM_POINTS * embedDimsOpt;
                                        uint32_t ptr = hLowPtrOffset + wLowPtrOffset + wStride;
                                        DataCopy(zerosLocal[pointOffset + offsetV], valueGm[offsetValue + ptr], embedDimsOpt);
                                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                                        Muls(midLocal[offsetMid], zerosLocal[pointOffset + topGradValueId * NUM_POINTS * embedDimsOpt], w2, embedDimsOpt);
                                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                                        Muls(tmpALocal, zerosLocal[pointOffset + offsetV], distW, embedDimsOpt);
                                        Muls(tmpBLocal, zerosLocal[pointOffset + offsetV], distH, embedDimsOpt);
                                        Sub(zerosLocal[offsetGradHWeight], zerosLocal[offsetGradHWeight], tmpALocal, embedDimsOpt);
                                        Add(zerosLocal[offsetGradWWeight], zerosLocal[offsetGradWWeight], tmpBLocal, embedDimsOpt);

                                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                                        DataCopy(gradValueGm[offsetValue + ptr], midLocal[offsetMid], embedDimsOpt);
                                            
                                    }
                                }
                                if (hLow < h - 1) {
                                    if (wLow >= 0 && wLow < w - 1) {
                                        DTYPE_VALUE distH = distLowLocal.GetValue(numPointsAlign + point);
                                        DTYPE_VALUE distW = distHighLocal.GetValue(point);
                                        w3 = distH * distW;
                                        DTYPE_VALUE distW2 = distLowLocal.GetValue(point);
                                        w4 = distH * distW2;
                                        uint32_t offsetMid = (point * 4 + mid3Id) * embedDimsOpt;
                                        uint32_t offsetV = v3Id * NUM_POINTS * embedDimsOpt;
                                        uint32_t offsetMid_ = (point * 4 + mid4Id) * embedDimsOpt;
                                        uint32_t offsetV_ = v4Id * NUM_POINTS * embedDimsOpt;
                                        uint32_t offsetGradHWeight = pointOffset + gradHWeightId * NUM_POINTS * embedDimsOpt;
                                        uint32_t offsetGradWWeight = pointOffset + gradWWeightId * NUM_POINTS * embedDimsOpt;
                                        uint32_t ptr = hLowPtrOffset + hStride + wLowPtrOffset;
                                        DataCopy(zerosLocal[pointOffset + offsetV], valueGm[offsetValue + ptr], embedDimsOpt);
                                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                                        Muls(midLocal[offsetMid], zerosLocal[pointOffset + topGradValueId * NUM_POINTS * embedDimsOpt], w3, embedDimsOpt);

                                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                                        Muls(tmpALocal, zerosLocal[pointOffset + offsetV], distW, embedDimsOpt);
                                        Muls(tmpBLocal, zerosLocal[pointOffset + offsetV], distH, embedDimsOpt);
                                        Add(zerosLocal[offsetGradHWeight], zerosLocal[offsetGradHWeight], tmpALocal, embedDimsOpt);
                                        Sub(zerosLocal[offsetGradWWeight], zerosLocal[offsetGradWWeight], tmpBLocal, embedDimsOpt);

                                        DataCopy(zerosLocal[pointOffset + offsetV_], valueGm[offsetValue + ptr + wStride], embedDimsOpt);
                                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                                        Muls(midLocal[offsetMid_], zerosLocal[pointOffset + topGradValueId * NUM_POINTS * embedDimsOpt], w4, embedDimsOpt);
                                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                                        Muls(tmpALocal, zerosLocal[pointOffset + offsetV_], distW2, embedDimsOpt);
                                        Muls(tmpBLocal, zerosLocal[pointOffset + offsetV_], distH, embedDimsOpt);
                                        Add(zerosLocal[offsetGradHWeight], zerosLocal[offsetGradHWeight], tmpALocal, embedDimsOpt);
                                        Add(zerosLocal[offsetGradWWeight], zerosLocal[offsetGradWWeight], tmpBLocal, embedDimsOpt);

                                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                                        DataCopy(gradValueGm[offsetValue + ptr], midLocal[offsetMid], embedDimsOptTwice);
                                    } else if (wLow >= 0) {
                                        DTYPE_VALUE distH = distLowLocal.GetValue(numPointsAlign + point);
                                        DTYPE_VALUE distW = distHighLocal.GetValue(point);
                                        w3 = distH * distW;

                                        uint32_t offsetMid = (point * 4 + mid3Id) * embedDimsOpt;
                                        uint32_t offsetV = v3Id * NUM_POINTS * embedDimsOpt;
                                        uint32_t offsetGradHWeight = pointOffset + gradHWeightId * NUM_POINTS * embedDimsOpt;
                                        uint32_t offsetGradWWeight = pointOffset + gradWWeightId * NUM_POINTS * embedDimsOpt;
                                        uint32_t ptr = hLowPtrOffset + hStride + wLowPtrOffset;
                                        DataCopy(zerosLocal[pointOffset + offsetV], valueGm[offsetValue + ptr], embedDimsOpt);
                                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                                        Muls(midLocal[offsetMid], zerosLocal[pointOffset + topGradValueId * NUM_POINTS * embedDimsOpt], w3, embedDimsOpt);
                                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                                        Muls(tmpALocal, zerosLocal[pointOffset + offsetV], distW, embedDimsOpt);
                                        Muls(tmpBLocal, zerosLocal[pointOffset + offsetV], distH, embedDimsOpt);
                                        Add(zerosLocal[offsetGradHWeight], zerosLocal[offsetGradHWeight], tmpALocal, embedDimsOpt);
                                        Sub(zerosLocal[offsetGradWWeight], zerosLocal[offsetGradWWeight], tmpBLocal, embedDimsOpt);

                                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                                        DataCopy(gradValueGm[offsetValue + ptr], midLocal[offsetMid], embedDimsOpt);
                                            
                                    } else if (wLow < w - 1) {
                                        DTYPE_VALUE distH = distLowLocal.GetValue(numPointsAlign + point);
                                        DTYPE_VALUE distW = distLowLocal.GetValue(point);
                                        w4 = distH * distW;

                                        uint32_t offsetMid = (point * 4 + mid4Id) * embedDimsOpt;
                                        uint32_t offsetV = v4Id * NUM_POINTS * embedDimsOpt;
                                        uint32_t offsetGradHWeight = pointOffset + gradHWeightId * NUM_POINTS * embedDimsOpt;
                                        uint32_t offsetGradWWeight = pointOffset + gradWWeightId * NUM_POINTS * embedDimsOpt;
                                        uint32_t ptr = hLowPtrOffset + hStride + wLowPtrOffset + wStride;
                                        DataCopy(zerosLocal[pointOffset + offsetV], valueGm[offsetValue + ptr], embedDimsOpt);
                                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                                        Muls(midLocal[offsetMid], zerosLocal[pointOffset + topGradValueId * NUM_POINTS * embedDimsOpt], w4, embedDimsOpt);
                                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                                        Muls(tmpALocal, zerosLocal[pointOffset + offsetV], distW, embedDimsOpt);
                                        Muls(tmpBLocal, zerosLocal[pointOffset + offsetV], distH, embedDimsOpt);
                                        Add(zerosLocal[offsetGradHWeight], zerosLocal[offsetGradHWeight], tmpALocal, embedDimsOpt);
                                        Add(zerosLocal[offsetGradWWeight], zerosLocal[offsetGradWWeight], tmpBLocal, embedDimsOpt);

                                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                                        DataCopy(gradValueGm[offsetValue + ptr], midLocal[offsetMid], embedDimsOpt);
                                    }
                                }
                                Muls(w1v1Local[pointOffset], zerosLocal[pointOffset + v1Id * NUM_POINTS * embedDimsOpt],
                                    w1, embedDimsOpt);
                                Muls(w2v2Local[pointOffset], zerosLocal[pointOffset + v2Id * NUM_POINTS * embedDimsOpt],
                                    w2, embedDimsOpt);
                                Muls(w3v3Local[pointOffset], zerosLocal[pointOffset + v3Id * NUM_POINTS * embedDimsOpt],
                                    w3, embedDimsOpt);
                                Muls(w4v4Local[pointOffset], zerosLocal[pointOffset + v4Id * NUM_POINTS * embedDimsOpt],
                                    w4, embedDimsOpt);
                                Add(w1v1Local[pointOffset], w1v1Local[pointOffset], w2v2Local[pointOffset], embedDimsOpt);
                                Add(w1v1Local[pointOffset], w1v1Local[pointOffset], w3v3Local[pointOffset], embedDimsOpt);
                                Add(w1v1Local[pointOffset], w1v1Local[pointOffset], w4v4Local[pointOffset], embedDimsOpt);
                                Mul(zerosLocal[pointOffset + gradWeightId * NUM_POINTS * embedDimsOpt], topGradLocal,
                                    w1v1Local[pointOffset], embedDimsOpt);
                            }
                        }
                        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
                        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);

                        if (NUM_POINTS < 8) {
                            Mul(tmpLocal, zerosLocal[topGradValueId * NUM_POINTS * embedDimsOpt], zerosLocal[gradWWeightId * NUM_POINTS * embedDimsOpt],
                                NUM_POINTS * embedDimsOpt);
                            Mul(tmpLocal[NUM_POINTS * embedDimsOpt], zerosLocal[topGradValueId * NUM_POINTS * embedDimsOpt],
                                zerosLocal[gradHWeightId * NUM_POINTS * embedDimsOpt], NUM_POINTS * embedDimsOpt);

                            Mul(gradSampleXLocLocal, tmpLocal, whLocLocal, 2 * NUM_POINTS * embedDimsOpt);  

                            Sum(weightSumLocal, zerosLocal[gradWeightId * NUM_POINTS * embedDimsOpt], sumParams);
                            SetFlag<HardEvent::V_MTE3>(eventIdVToMteWeight);
                            Sum(locSumLocal, gradSampleXLocLocal, sumParamsV2);
                            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3X);

                            WaitFlag<HardEvent::V_MTE3>(eventIdVToMteWeight);
                            DataCopyPad(gradWeightGm[offsetWeight + level * NUM_POINTS], weightSumLocal, copyParams);
                        
                            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3X);
                            DataCopyPad(gradLocationGm[offsetLocation + level * 2 * NUM_POINTS], locSumLocal, copyParamsV2);
                        } else{
                            Mul(tmpLocal, zerosLocal[topGradValueId * NUM_POINTS * embedDimsOpt], zerosLocal[gradWWeightId * NUM_POINTS * embedDimsOpt],
                                NUM_POINTS * embedDimsOpt);
                            Mul(tmpLocal[NUM_POINTS * embedDimsOpt], zerosLocal[topGradValueId * NUM_POINTS * embedDimsOpt],
                                zerosLocal[gradHWeightId * NUM_POINTS * embedDimsOpt], NUM_POINTS * embedDimsOpt);

                            Mul(gradSampleXLocLocal, tmpLocal, whLocLocal, 2 * NUM_POINTS * embedDimsOpt);  

                            Sum(weightSumLocal[NUM_POINTS * level], zerosLocal[gradWeightId * NUM_POINTS * embedDimsOpt], sumParams);
                            Sum(locSumLocal[NUM_POINTS * 2 * level], gradSampleXLocLocal, sumParamsV2);
                        }

                        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
                        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                    }

                    if (NUM_POINTS == 8) {
                        SetFlag<HardEvent::V_MTE3>(eventIdVToMteWeight);
                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMteWeight);

                        DataCopy(gradWeightGm[offsetWeight], weightSumLocal, NUM_POINTS * numLevels);
                        DataCopy(gradLocationGm[offsetLocation], locSumLocal, NUM_POINTS * numLevels * 2);
                    }
                }
            }
        }
    }

    __aicore__ inline void Compute()
    {
        for (query = startOffset; query < endOffset; query++) {
            for (batch = 0; batch < batchSize; batch++) {
                for (head = 0; head < numHeads; head++) {
                    offsetWeight = batch * weightStride2 + query * weightStride1 + head * weightStride0;
                    offsetLocation = 2 * offsetWeight;
                    DataCopy(topGradLocal,
                            gradOutputGm[batch * gradOutStride2 + query * gradOutStride1 + head * gradOutStride0],
                            embedDims);
                    for (level = 0; level < numLevels; level++) {
                        levelStartId = offsetLocal.GetValue(level);
                        h = shapesLocal.GetValue(level * 2);
                        w = shapesLocal.GetValue(level * 2 + 1);

                        Duplicate(whLocLocal, (DTYPE_VALUE)w, numPoints * embedDims);
                        Duplicate(whLocLocal[numPoints * embedDims], (DTYPE_VALUE)h, numPoints * embedDims);

                        offsetValue = batch * valueStride2 + head * valueStride1 + levelStartId * valueStride0;
                        wStride = embedDims;
                        hStride = w * wStride;
                        DataCopy(locWLocal, locationGm[offsetLocation + level * numPoints * 2], numPointsAlign);
                        DataCopy(locHLocal, locationGm[offsetLocation + level * numPoints * 2 + numPoints], numPointsAlign);
                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                        DataCopy(attentionWeightLocal, attentionWeightsGm[offsetWeight + level * numPoints],
                                numPointsAlign);
                        Muls(imLocal[numPointsAlign], locHLocal, (DTYPE_VALUE)h, numPointsAlign);
                        Muls(imLocal, locWLocal, (DTYPE_VALUE)w, numPointsAlign);
                        Adds(imLocal, imLocal, DTYPE_VALUE(-0.5), 2 * numPointsAlign);
                        Cast(lowLocal, imLocal, RoundMode::CAST_FLOOR, 2 * numPointsAlign);
                        Cast(lowFloatLocal, lowLocal, RoundMode::CAST_NONE, 2 * numPointsAlign);

                        Sub(distLowLocal, imLocal, lowFloatLocal, 2 * numPointsAlign);
                        Sub(distHighLocal, floatOneLocal, distLowLocal, 2 * numPointsAlign);

                        Duplicate(zerosLocal, (DTYPE_VALUE)0, 8 * numPoints * embedDims);

                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                        for (point = 0; point < numPoints; point++) {
                            pointOffset = point * embedDims;
                            hIm = imLocal.GetValue(numPointsAlign + point);
                            wIm = imLocal.GetValue(point);
                            if (hIm > -1 && wIm > -1 && hIm < h && wIm < w) {
                                hLow = lowLocal.GetValue(numPointsAlign + point);
                                wLow = lowLocal.GetValue(point);
                                hLowPtrOffset = hLow * hStride;
                                wLowPtrOffset = wLow * wStride;
                                Muls(zerosLocal[pointOffset + topGradValueId * baseOffsetUb], topGradLocal,
                                    attentionWeightLocal.GetValue(point), embedDims);
                                if (hLow >= 0) {
                                    if (wLow >= 0) {
                                        DTYPE_VALUE distH = distHighLocal.GetValue(numPointsAlign + point);
                                        DTYPE_VALUE distW = distHighLocal.GetValue(point);
                                        w1 = distH * distW;
                                        ComputeGrad<false, false>(mid1Id, v1Id, distH, distW, hLowPtrOffset, wLowPtrOffset,
                                                                w1);
                                    }
                                    if (wLow < w - 1) {
                                        DTYPE_VALUE distH = distHighLocal.GetValue(numPointsAlign + point);
                                        DTYPE_VALUE distW = distLowLocal.GetValue(point);
                                        w2 = distH * distW;
                                        ComputeGrad<false, true>(mid2Id, v2Id, distH, distW, hLowPtrOffset, wLowPtrOffset + wStride,
                                                                w2);
                                    }
                                }
                                if (hLow < h - 1) {
                                    if (wLow >= 0) {
                                        DTYPE_VALUE distH = distLowLocal.GetValue(numPointsAlign + point);
                                        DTYPE_VALUE distW = distHighLocal.GetValue(point);
                                        w3 = distH * distW;
                                        ComputeGrad<true, false>(mid3Id, v3Id, distH, distW, hLowPtrOffset + hStride, wLowPtrOffset,
                                                                w3);
                                    }
                                    if (wLow < w - 1) {
                                        DTYPE_VALUE distH = distLowLocal.GetValue(numPointsAlign + point);
                                        DTYPE_VALUE distW = distLowLocal.GetValue(point);
                                        w4 = distH * distW;
                                        ComputeGrad<true, true>(mid4Id, v4Id, distH, distW, hLowPtrOffset + hStride, wLowPtrOffset + wStride,
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
                        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
                        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                        Mul(tmpLocal, zerosLocal[topGradValueId * baseOffsetUb], zerosLocal[gradWWeightId * baseOffsetUb],
                            numPoints * embedDims);

                        Mul(tmpLocal[numPoints * embedDims], zerosLocal[topGradValueId * baseOffsetUb],
                            zerosLocal[gradHWeightId * baseOffsetUb], numPoints * embedDims);

                        Mul(gradSampleXLocLocal, tmpLocal, whLocLocal, 2 * baseOffsetUb);  

                        Sum(weightSumLocal, zerosLocal[gradWeightId * baseOffsetUb], sumParams);
                        SetFlag<HardEvent::V_MTE3>(eventIdVToMteWeight);
                        Sum(locSumLocal, gradSampleXLocLocal, sumParamsV2);
                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3X);

                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMteWeight);
                        DataCopyPad(gradWeightGm[offsetWeight + level * numPoints], weightSumLocal, copyParams);
                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3X);
                        DataCopyPad(gradLocationGm[offsetLocation + level * 2 * numPoints], locSumLocal, copyParamsV2);
                        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
                        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                    }
                }
            }
        }
    }

private:
    TPipe *pipe;
    GlobalTensor<DTYPE_VALUE> valueGm, locationGm, attentionWeightsGm, gradOutputGm, gradValueGm, gradLocationGm,
        gradWeightGm;
    GlobalTensor<DTYPE_SPATIAL_SHAPES> valueSpatialShapesGm, valueLevelStartIndexGm;

    TBuf<TPosition::VECCALC> locationUb, attentionWeightsUb, shapeUb, offsetUb, topGradUb;
    TBuf<TPosition::VECCALC> locSumUb, weightSumUb;
    TBuf<TPosition::VECCALC> floatOneUb, zerosUb;
    TBuf<TPosition::VECCALC> locWUb, locHUb, imUb, lowUb, lowFloatUb;
    TBuf<TPosition::VECCALC> distLowUb, distHighUb, w1Ub, w2Ub, w3Ub, w4Ub;
    TBuf<TPosition::VECCALC> w1v1Ub, w2v2Ub, w3v3Ub, w4v4Ub, tmpUb, tmpAUb, tmpBUb, midUb;
    TBuf<TPosition::VECCALC> gradSampleXLocUb, gradSampleYLocUb, whUb;

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
    uint32_t baseOffsetUb, pointOffset;
    uint32_t mid1Id = 0, mid2Id = 1, mid3Id = 2, mid4Id = 3;
    uint32_t gradWWeightId = 0, gradHWeightId = 1, topGradValueId = 2, gradWeightId = 3;
    uint32_t v1Id = 4, v2Id = 5, v3Id = 6, v4Id = 7;

    uint32_t embedDimsOpt = 32;
    uint32_t embedDimsOptTwice = 64;
    uint32_t numPointsAlignOpt = 8;
    uint32_t numPointsAlignOptTwice = 16;
    uint32_t numPointsAlignOptTriple = 24;

    DTYPE_VALUE hIm, wIm;
    DTYPE_VALUE w1 = 0, w2 = 0, w3 = 0, w4 = 0;
    DTYPE_SPATIAL_SHAPES h, w, levelStartId;
    DTYPE_SPATIAL_SHAPES offsetValue, offsetWeight, offsetLocation, wStride, hStride;
    DTYPE_SPATIAL_SHAPES hLowPtrOffset, wLowPtrOffset, tmpOffset1, tmpOffset2;
    DTYPE_SPATIAL_SHAPES hLow, wLow;

    LocalTensor<DTYPE_VALUE> lowFloatLocal;
    LocalTensor<DTYPE_VALUE> floatOneLocal;
    LocalTensor<DTYPE_VALUE> locSumLocal;
    LocalTensor<DTYPE_VALUE> distLowLocal, distHighLocal;
    LocalTensor<DTYPE_VALUE> locWLocal, locHLocal;
    LocalTensor<DTYPE_VALUE> imLocal;
    LocalTensor<DTYPE_VALUE> zerosLocal;
    LocalTensor<DTYPE_VALUE> w1v1Local, w2v2Local, w3v3Local, w4v4Local;
    LocalTensor<DTYPE_VALUE> weightSumLocal, midLocal, tmpLocal, tmpALocal, tmpBLocal;
    LocalTensor<DTYPE_VALUE> gradSampleXLocLocal, gradSampleYLocLocal, whLocLocal;
    LocalTensor<DTYPE_VALUE> topGradLocal, locationLocal, attentionWeightLocal;
    LocalTensor<DTYPE_SPATIAL_SHAPES> shapesLocal, offsetLocal;
    LocalTensor<DTYPE_SPATIAL_SHAPES> lowLocal;

    SumParams sumParams, sumParamsV2;
    DataCopyParams copyParams, copyParamsV2;
    event_t eventIdVToMte2, eventIdVToMte3, eventIdMte2ToV, eventIdMte3ToV,
            eventIdVToMteWeight, eventIdVToMte3X, eventIdVToMte3Y;
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
    op.ClearOutput();
    op.Process();
    op.ReleaseEventID();
}
