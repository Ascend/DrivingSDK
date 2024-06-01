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
 * \file multi_scale_deformable_attention_grad_high_perf_v2.h
 * \brief
 */

#ifndef MS_DEFORM_ATTN_GRAD_HIGH_PERF_V2_H_
#define MS_DEFORM_ATTN_GRAD_HIGH_PERF_V2_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

class MultiScaleDeformableAttnGradHighPerfV2 {
public:
    __aicore__ inline MultiScaleDeformableAttnGradHighPerfV2() {};
    __aicore__ inline void Init(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm, GM_ADDR level_start_index_gm,
        GM_ADDR sampling_loc_gm, GM_ADDR attn_weight_gm, GM_ADDR grad_output_gm, GM_ADDR grad_value_gm,
        GM_ADDR grad_sampling_loc_gm, GM_ADDR grad_attn_weight_gm,
        const MultiScaleDeformableAttnGradTilingDataV2* tiling_data, TPipe* tmpPipe)
    {
        pipe = tmpPipe;
        curBlockIdx = GetBlockIdx();
        blockBytes = 32;
        dataAlign = blockBytes / sizeof(DTYPE_VALUE);

        numKeys = tiling_data->numKeys;
        numHeads = tiling_data->numHeads;
        numLevels = tiling_data->numLevels;
        numQueries = tiling_data->numQueries;
        numPoints = tiling_data->numPoints;
        batchSize = tiling_data->batchSize;
        maxUbNum = tiling_data->maxUbNum;
        coreNum = tiling_data->coreNum;

        numLevelsAlign = AlignUp(numLevels, dataAlign);

        numQueriesper = DivCeil(numQueries, maxUbNum);
        numQueriestail = numQueries - (numQueriesper - 1) * maxUbNum;
        numQueriestail = AlignUp(numQueriestail, dataAlign);
        numQueriesAlign = numQueries <= maxUbNum ? numQueriestail : maxUbNum;

        taskNum = batchSize * numHeads * numLevels * numPoints * numQueriesper;
        taskNumPerCore = DivCeil(taskNum, coreNum);

        startOffset = curBlockIdx * taskNumPerCore;
        endOffset = (curBlockIdx + 1) * taskNumPerCore;
        if (endOffset > taskNum) {
            endOffset = taskNum;
        }

        gradOutStride0 = embedDimsOpt;
        gradOutStride1 = numHeads * gradOutStride0;
        gradOutStride2 = numQueries * gradOutStride1;

        weightStride0 = numQueries;
        weightStride1 = numPoints * weightStride0;
        weightStride2 = numLevels * weightStride1;
        weightStride3 = numHeads * weightStride2;

        valueStride0 = embedDimsOpt;
        valueStride1 = numHeads * valueStride0;
        valueStride2 = numKeys * valueStride1;
        wStride = numHeads * embedDimsOpt;

        baseOffsetUb = numQueriesAlign * embedDimsOpt;

        eventIdMte2ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdMte2ToV_1 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdMte2ToV_2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdMte3ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());
        eventIdVToMte2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMte3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdVToMteWeight = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdVToMte3X = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdVToMte3Y = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());

        valueGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_VALUE*>(value_gm), batchSize * numKeys * numHeads * embedDimsOpt);
        valueSpatialShapesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(spatial_shapes_gm), numLevels * 2);
        valueLevelStartIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(level_start_index_gm), numLevels);
        locationGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE*>(sampling_loc_gm),
            batchSize * numQueries * numHeads * numLevels * numPoints * 2);
        attentionWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE*>(attn_weight_gm),
            batchSize * numQueries * numHeads * numLevels * numPoints);
        gradOutputGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_VALUE*>(grad_output_gm), batchSize * numQueries * numHeads * embedDimsOpt);

        gradValueGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_VALUE*>(grad_value_gm), batchSize * numKeys * numHeads * embedDimsOpt);
        gradLocationGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE*>(grad_sampling_loc_gm),
            batchSize * numQueries * numHeads * numLevels * 2 * numPoints);
        gradWeightGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE*>(grad_attn_weight_gm),
            batchSize * numQueries * numHeads * numLevels * numPoints);
    }

    __aicore__ inline void InitBuffer()
    {
        pipe->InitBuffer(shapeUb, 2 * numLevelsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(offsetUb, numLevelsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(zerosUb, 8 * numQueriesAlign * embedDimsOpt * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(topGradUb, numQueriesAlign * embedDimsOpt * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(mid1Ub, numQueriesAlign * embedDimsOpt * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(mid2Ub, numQueriesAlign * embedDimsOpt * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(mid3Ub, numQueriesAlign * embedDimsOpt * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(mid4Ub, numQueriesAlign * embedDimsOpt * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(attentionWeightsUb, numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpXUb, numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpYUb, numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(weightSumUb, numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(locWUb, numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(locHUb, numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(imUb, 2 * numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(lowUb, 2 * numQueriesAlign * sizeof(int32_t));
        pipe->InitBuffer(lowFloatUb, 2 * numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(distLowUb, 2 * numQueriesAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(w4Ub, numQueriesAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(tmpAUb, 2 * embedDimsOpt * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpBUb, 2 * embedDimsOpt * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(wv1Ub, embedDimsOpt * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(wv2Ub, embedDimsOpt * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(wv3Ub, embedDimsOpt * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(wv4Ub, embedDimsOpt * sizeof(DTYPE_VALUE));
    }

    __aicore__ inline void GetLocalTensor()
    {
        attentionWeightLocal = attentionWeightsUb.Get<DTYPE_VALUE>();
        shapesLocal = shapeUb.Get<int32_t>();
        offsetLocal = offsetUb.Get<int32_t>();
        xLocal = tmpXUb.Get<DTYPE_VALUE>();
        yLocal = tmpYUb.Get<DTYPE_VALUE>();
        weightSumLocal = weightSumUb.Get<DTYPE_VALUE>();
        topGradLocal = topGradUb.Get<DTYPE_VALUE>();
        locWLocal = locWUb.Get<DTYPE_VALUE>();
        locHLocal = locHUb.Get<DTYPE_VALUE>();

        imLocal = imUb.Get<DTYPE_VALUE>();
        lowLocal = lowUb.Get<int32_t>();
        lowFloatLocal = lowFloatUb.Get<DTYPE_VALUE>();
        zerosLocal = zerosUb.Get<DTYPE_VALUE>();
        distLowLocal = distLowUb.Get<DTYPE_VALUE>();

        tmpALocal = tmpAUb.Get<DTYPE_VALUE>();
        tmpBLocal = tmpBUb.Get<DTYPE_VALUE>();

        w4Local = w4Ub.Get<DTYPE_VALUE>();
        wv1Local = wv1Ub.Get<DTYPE_VALUE>();
        wv2Local = wv2Ub.Get<DTYPE_VALUE>();
        wv3Local = wv3Ub.Get<DTYPE_VALUE>();
        wv4Local = wv4Ub.Get<DTYPE_VALUE>();

        mid1Local = mid1Ub.Get<DTYPE_VALUE>();
        mid2Local = mid2Ub.Get<DTYPE_VALUE>();
        mid3Local = mid3Ub.Get<DTYPE_VALUE>();
        mid4Local = mid4Ub.Get<DTYPE_VALUE>();
    }

    __aicore__ inline void Process()
    {
        copyInParams = {2, uint32_t(embedDimsOpt * sizeof(DTYPE_VALUE)),
            uint32_t((wStride - embedDimsOpt) * sizeof(DTYPE_VALUE)), 0, 0};
        uint32_t startIdx = startOffset;
        nqloop = startIdx % numQueriesper;
        startIdx = startIdx / numQueriesper;
        point = startIdx % numPoints;
        startIdx = startIdx / numPoints;
        level = startIdx % numLevels;
        startIdx = startIdx / numLevels;
        head = startIdx % numHeads;
        batch = startIdx / numHeads;

        offsetWeight = batch * weightStride3 + head * weightStride2 + level * weightStride1 + point * weightStride0;
        offsetLocation = 2 * offsetWeight;
        offsetGrad = batch * gradOutStride2 + nqloop * maxUbNum * gradOutStride1 + head * gradOutStride0;

        thisCycleNumAlign = (nqloop == numQueriesper - 1) ? numQueriestail : maxUbNum;
        thisCycleNum = (nqloop == numQueriesper - 1) ? (numQueries - (numQueriesper - 1) * maxUbNum) : maxUbNum;
        copyParams = {1, (uint16_t)(thisCycleNum * sizeof(DTYPE_VALUE)), 0, 0};
        sumParams = {thisCycleNum, embedDimsOpt, embedDimsOpt};

        DataCopy(shapesLocal, valueSpatialShapesGm, 2 * numLevelsAlign);
        DataCopy(offsetLocal, valueLevelStartIndexGm, numLevelsAlign);

        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        for (uint32_t taskIdx = startOffset; taskIdx < endOffset; taskIdx++) {
            Compute(taskIdx);
        }
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }

private:
    template<bool AddH, bool AddW>
    __aicore__ inline void ComputeGrad(const LocalTensor<DTYPE_VALUE>& wvLocal, const LocalTensor<DTYPE_VALUE>& mid,
        uint32_t vId, DTYPE_VALUE distH, DTYPE_VALUE distW, uint32_t hPtrOffset, uint32_t wPtrOffset, DTYPE_VALUE w)
    {
        DataCopy(zerosLocal[vId * embedDimsOpt + queryOffsetv + 4 * baseOffsetUb],
            valueGm[offsetValue + hPtrOffset + wPtrOffset], embedDimsOpt);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        Muls(mid[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w, embedDimsOpt);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Muls(wvLocal, zerosLocal[vId * embedDimsOpt + queryOffsetv + 4 * baseOffsetUb], w, embedDimsOpt);
        Muls(tmpALocal, zerosLocal[vId * embedDimsOpt + queryOffsetv + 4 * baseOffsetUb], distW, embedDimsOpt);
        Muls(tmpBLocal, zerosLocal[vId * embedDimsOpt + queryOffsetv + 4 * baseOffsetUb], distH, embedDimsOpt);
        if (AddH) {
            Add(zerosLocal[queryOffset + gradHWeightId * baseOffsetUb],
                zerosLocal[queryOffset + gradHWeightId * baseOffsetUb], tmpALocal, embedDimsOpt);
        } else {
            Sub(zerosLocal[queryOffset + gradHWeightId * baseOffsetUb],
                zerosLocal[queryOffset + gradHWeightId * baseOffsetUb], tmpALocal, embedDimsOpt);
        }
        if (AddW) {
            Add(zerosLocal[queryOffset + gradWWeightId * baseOffsetUb],
                zerosLocal[queryOffset + gradWWeightId * baseOffsetUb], tmpBLocal, embedDimsOpt);
        } else {
            Sub(zerosLocal[queryOffset + gradWWeightId * baseOffsetUb],
                zerosLocal[queryOffset + gradWWeightId * baseOffsetUb], tmpBLocal, embedDimsOpt);
        }
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopy(gradValueGm[offsetValue + hPtrOffset + wPtrOffset], mid[queryOffset], embedDimsOpt);
    }

    __aicore__ inline void ComputeGradSeparate(DTYPE_VALUE distHH, DTYPE_VALUE distHW, DTYPE_VALUE distLH,
        DTYPE_VALUE distLW, DTYPE_VALUE w1, DTYPE_VALUE w2, DTYPE_VALUE w3, DTYPE_VALUE w4, DTYPE_VALUE attentionWeight)
    {
        Muls(zerosLocal[queryOffset + topGradValueId * baseOffsetUb], topGradLocal[query * embedDimsOpt],
            attentionWeight, embedDimsOpt);
        if (hLow >= 0 && wLow >= 0) {
            ComputeGrad<false, false>(wv1Local, mid1Local, v1Id, distHH, distHW, hLowPtrOffset, wLowPtrOffset, w1);
        }
        if (hLow >= 0 && wLow < w - 1) {
            ComputeGrad<false, true>(
                wv2Local, mid2Local, v2Id, distHH, distLW, hLowPtrOffset, wLowPtrOffset + wStride, w2);
        }
        if (hLow < h - 1 && wLow >= 0) {
            ComputeGrad<true, false>(
                wv3Local, mid3Local, v3Id, distLH, distHW, hLowPtrOffset + hStride, wLowPtrOffset, w3);
        }
        if (hLow < h - 1 && wLow < w - 1) {
            ComputeGrad<true, true>(
                wv4Local, mid4Local, v4Id, distLH, distLW, hLowPtrOffset + hStride, wLowPtrOffset + wStride, w4);
        }
        Add(wv1Local, wv1Local, wv2Local, embedDimsOpt);
        Add(wv3Local, wv3Local, wv4Local, embedDimsOpt);
        Add(wv1Local, wv1Local, wv3Local, embedDimsOpt);
        Mul(zerosLocal[queryOffset + gradWeightId * baseOffsetUb], topGradLocal[query * embedDimsOpt], wv1Local,
            embedDimsOpt);
    }

    __aicore__ inline void ComputeGradTogether(DTYPE_VALUE distLH, DTYPE_VALUE distLW, DTYPE_VALUE w1, DTYPE_VALUE w2,
        DTYPE_VALUE w3, DTYPE_VALUE w4, DTYPE_VALUE attentionWeight)
    {
        DataCopyPad(zerosLocal[v1Id * embedDimsOpt + queryOffsetv + 4 * baseOffsetUb],
            valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset], copyInParams, padParams);
        DataCopyPad(zerosLocal[v3Id * embedDimsOpt + queryOffsetv + 4 * baseOffsetUb],
            valueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + hStride], copyInParams, padParams);

        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        Muls(zerosLocal[queryOffset + topGradValueId * baseOffsetUb], topGradLocal[query * embedDimsOpt],
            attentionWeight, embedDimsOpt);

        Muls(mid1Local[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w1, embedDimsOpt);
        Muls(mid2Local[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w2, embedDimsOpt);
        Muls(mid3Local[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w3, embedDimsOpt);
        Muls(mid4Local[queryOffset], zerosLocal[queryOffset + topGradValueId * baseOffsetUb], w4, embedDimsOpt);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Muls(wv1Local, zerosLocal[v1Id * embedDimsOpt + queryOffsetv + 4 * baseOffsetUb], w1, embedDimsOpt);
        Muls(wv2Local, zerosLocal[v2Id * embedDimsOpt + queryOffsetv + 4 * baseOffsetUb], w2, embedDimsOpt);
        Muls(wv3Local, zerosLocal[v3Id * embedDimsOpt + queryOffsetv + 4 * baseOffsetUb], w3, embedDimsOpt);
        Muls(wv4Local, zerosLocal[v4Id * embedDimsOpt + queryOffsetv + 4 * baseOffsetUb], w4, embedDimsOpt);

        // V3 - V1
        Sub(zerosLocal[queryOffset + gradHWeightId * baseOffsetUb],
            zerosLocal[v3Id * embedDimsOpt + queryOffsetv + 4 * baseOffsetUb],
            zerosLocal[v1Id * embedDimsOpt + queryOffsetv + 4 * baseOffsetUb], embedDimsOpt);
        // V4 - V2
        Sub(tmpALocal, zerosLocal[v4Id * embedDimsOpt + queryOffsetv + 4 * baseOffsetUb],
            zerosLocal[v2Id * embedDimsOpt + queryOffsetv + 4 * baseOffsetUb], embedDimsOpt);
        // V2 -V1
        Sub(zerosLocal[queryOffset + gradWWeightId * baseOffsetUb],
            zerosLocal[v2Id * embedDimsOpt + queryOffsetv + 4 * baseOffsetUb],
            zerosLocal[v1Id * embedDimsOpt + queryOffsetv + 4 * baseOffsetUb], embedDimsOpt);
        // V4 + V1 - V3 - V2
        Sub(tmpALocal, tmpALocal, zerosLocal[queryOffset + gradHWeightId * baseOffsetUb], embedDimsOpt);
        // (V4 + V1 - V3 - V2) * distLH
        Muls(tmpBLocal, tmpALocal, distLH, embedDimsOpt);
        // (V4 + V1 - V3 - V2) * distLW
        Muls(tmpALocal, tmpALocal, distLW, embedDimsOpt);
        // (V2 - V1) + (V4 + V1 - V3 - V2) * distLH
        Add(zerosLocal[queryOffset + gradWWeightId * baseOffsetUb],
            zerosLocal[queryOffset + gradWWeightId * baseOffsetUb], tmpBLocal, embedDimsOpt);
        // (V3 - V1) + (V4 + V1 - V3 - V2) * distLW
        Add(zerosLocal[queryOffset + gradHWeightId * baseOffsetUb],
            zerosLocal[queryOffset + gradHWeightId * baseOffsetUb], tmpALocal, embedDimsOpt);

        Add(wv1Local, wv1Local, wv2Local, embedDimsOpt);
        Add(wv3Local, wv3Local, wv4Local, embedDimsOpt);
        Add(wv1Local, wv1Local, wv3Local, embedDimsOpt);
        Mul(zerosLocal[queryOffset + gradWeightId * baseOffsetUb], topGradLocal[query * embedDimsOpt], wv1Local,
            embedDimsOpt);

        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopy(gradValueGm[offsetValue + hLowPtrOffset + wLowPtrOffset], mid1Local[queryOffset], embedDimsOpt);
        DataCopy(
            gradValueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + wStride], mid2Local[queryOffset], embedDimsOpt);
        DataCopy(
            gradValueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + hStride], mid3Local[queryOffset], embedDimsOpt);
        DataCopy(gradValueGm[offsetValue + hLowPtrOffset + wLowPtrOffset + hStride + wStride], mid4Local[queryOffset],
            embedDimsOpt);
    }

    __aicore__ inline void Compute(uint32_t taskIdx)
    {
        levelStartId = offsetLocal.GetValue(level);
        h = shapesLocal.GetValue(level * 2);
        w = shapesLocal.GetValue(level * 2 + 1);
        offsetValue = batch * valueStride2 + levelStartId * valueStride1 + head * valueStride0;
        hStride = w * wStride;

        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        Duplicate(zerosLocal, (DTYPE_VALUE)0, 8 * numQueriesAlign * embedDimsOpt);

        DataCopy(locWLocal, locationGm[offsetLocation + nqloop * maxUbNum], thisCycleNumAlign);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);
        DataCopy(locHLocal, locationGm[offsetLocation + numQueries + nqloop * maxUbNum], thisCycleNumAlign);
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_2);

        DataCopy(attentionWeightLocal, attentionWeightsGm[offsetWeight + nqloop * maxUbNum], thisCycleNumAlign);
        for (query = 0; query < thisCycleNum; query++) {
            DataCopy(
                topGradLocal[query * embedDimsOpt], gradOutputGm[offsetGrad + query * gradOutStride1], embedDimsOpt);
        }
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);
        Muls(imLocal, locWLocal, (DTYPE_VALUE)w, thisCycleNumAlign);

        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_2);
        Muls(imLocal[thisCycleNumAlign], locHLocal, (DTYPE_VALUE)h, thisCycleNumAlign);

        Adds(imLocal, imLocal, DTYPE_VALUE(-0.5), 2 * thisCycleNumAlign);
        Cast(lowLocal, imLocal, RoundMode::CAST_FLOOR, 2 * thisCycleNumAlign);
        Cast(lowFloatLocal, lowLocal, RoundMode::CAST_NONE, 2 * thisCycleNumAlign);
        Sub(distLowLocal, imLocal, lowFloatLocal, 2 * thisCycleNumAlign);
        Mul(w4Local, distLowLocal[thisCycleNumAlign], distLowLocal, thisCycleNumAlign);

        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        SetAtomicAdd<DTYPE_VALUE>();
        for (query = 0; query < thisCycleNum; query++) {
            queryOffset = query * embedDimsOpt;
            queryOffsetv = query * 4 * embedDimsOpt;
            hIm = imLocal.GetValue(thisCycleNumAlign + query);
            wIm = imLocal.GetValue(query);
            if (hIm > -1 && wIm > -1 && hIm < h && wIm < w) {
                hLow = lowLocal.GetValue(thisCycleNumAlign + query);
                wLow = lowLocal.GetValue(query);
                hLowPtrOffset = hLow * hStride;
                wLowPtrOffset = wLow * wStride;
                DTYPE_VALUE distH = distLowLocal.GetValue(thisCycleNumAlign + query);
                DTYPE_VALUE distW = distLowLocal.GetValue(query);
                DTYPE_VALUE attenWeight = attentionWeightLocal.GetValue(query);
                w4 = w4Local.GetValue(query);
                w1 = w4 + 1 - distH - distW;
                w2 = distW - w4;
                w3 = distH - w4;
                if (hLow >= 0 && wLow >= 0 && hLow < h - 1 && wLow < w - 1) {
                    ComputeGradTogether(distH, distW, w1, w2, w3, w4, attenWeight);
                } else {
                    Duplicate(wv1Local, (DTYPE_VALUE)0, embedDimsOpt);
                    Duplicate(wv2Local, (DTYPE_VALUE)0, embedDimsOpt);
                    Duplicate(wv3Local, (DTYPE_VALUE)0, embedDimsOpt);
                    Duplicate(wv4Local, (DTYPE_VALUE)0, embedDimsOpt);
                    ComputeGradSeparate(1 - distH, 1 - distW, distH, distW, w1, w2, w3, w4, attenWeight);
                }
            }
        }
        SetAtomicNone();
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);

        Mul(zerosLocal[gradWWeightId * baseOffsetUb], zerosLocal[topGradValueId * baseOffsetUb],
            zerosLocal[gradWWeightId * baseOffsetUb], thisCycleNum * embedDimsOpt);
        Mul(zerosLocal[gradHWeightId * baseOffsetUb], zerosLocal[topGradValueId * baseOffsetUb],
            zerosLocal[gradHWeightId * baseOffsetUb], thisCycleNum * embedDimsOpt);
        Muls(zerosLocal[gradWWeightId * baseOffsetUb], zerosLocal[gradWWeightId * baseOffsetUb], (DTYPE_VALUE)w,
            thisCycleNum * embedDimsOpt);
        Muls(zerosLocal[gradHWeightId * baseOffsetUb], zerosLocal[gradHWeightId * baseOffsetUb], (DTYPE_VALUE)h,
            thisCycleNum * embedDimsOpt);

        Sum(weightSumLocal, zerosLocal[gradWeightId * baseOffsetUb], sumParams);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMteWeight);
        Sum(xLocal, zerosLocal[gradWWeightId * baseOffsetUb], sumParams);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3X);
        Sum(yLocal, zerosLocal[gradHWeightId * baseOffsetUb], sumParams);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3Y);

        WaitFlag<HardEvent::V_MTE3>(eventIdVToMteWeight);
        DataCopy(gradWeightGm[offsetWeight + nqloop * maxUbNum], weightSumLocal, maxUbNum);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3X);
        DataCopy(gradLocationGm[offsetLocation + nqloop * maxUbNum], xLocal, maxUbNum);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3Y);
        DataCopy(gradLocationGm[offsetLocation + numQueries + nqloop * maxUbNum], yLocal, maxUbNum);
        if (taskIdx < endOffset - 1) {
            nqloop += 1;
            point += (nqloop / numQueriesper);
            nqloop %= numQueriesper;
            level += (point / numPoints);
            point %= numPoints;
            head += (level / numLevels);
            level %= numLevels;
            batch += (head / numHeads);
            head %= numHeads;

            offsetGrad = batch * gradOutStride2 + nqloop * maxUbNum * gradOutStride1 + head * gradOutStride0;
            offsetWeight = batch * weightStride3 + head * weightStride2 + level * weightStride1 + point * weightStride0;
            offsetLocation = 2 * offsetWeight;

            thisCycleNumAlign = (nqloop == numQueriesper - 1) ? numQueriestail : maxUbNum;
            thisCycleNum = (nqloop == numQueriesper - 1) ? (numQueries - (numQueriesper - 1) * maxUbNum) : maxUbNum;

            copyParams = {1, (uint16_t)(thisCycleNum * sizeof(DTYPE_VALUE)), 0, 0};
            sumParams = {thisCycleNum, embedDimsOpt, embedDimsOpt};
        }
    }

private:
    TPipe* pipe;
    GlobalTensor<DTYPE_VALUE> valueGm, locationGm, attentionWeightsGm, gradOutputGm, gradValueGm, gradLocationGm,
        gradWeightGm;
    GlobalTensor<int32_t> valueSpatialShapesGm, valueLevelStartIndexGm;

    TBuf<TPosition::VECCALC> attentionWeightsUb, shapeUb, offsetUb, topGradUb;
    TBuf<TPosition::VECCALC> tmpXUb, tmpYUb, weightSumUb;
    TBuf<TPosition::VECCALC> zerosUb;
    TBuf<TPosition::VECCALC> locWUb, locHUb, imUb, lowUb, lowFloatUb;
    TBuf<TPosition::VECCALC> distLowUb, w4Ub;
    TBuf<TPosition::VECCALC> tmpAUb, tmpBUb;

    uint32_t coreNum;
    uint32_t embedDimsOpt = 32;
    uint32_t batchSize, numKeys, numHeads, numLevels, numQueries, numPoints;
    uint32_t numQueriesAlign, numQueriesper, numQueriestail, numLevelsAlign;
    uint32_t batch, query, head, level, point, nqloop;
    uint32_t curBlockIdx;
    uint32_t taskNum, taskNumPerCore;
    uint32_t startOffset, endOffset;
    uint32_t dataAlign, blockBytes;
    uint32_t gradOutStride0, gradOutStride1, gradOutStride2;
    uint32_t weightStride0, weightStride1, weightStride2, weightStride3;
    uint32_t valueStride0, valueStride1, valueStride2;
    uint32_t hOffsetUb, baseOffsetUb, queryOffset, queryOffsetv;
    uint32_t gradHWeightId = 0, gradWWeightId = 1, topGradValueId = 2, gradWeightId = 3;
    uint32_t v1Id = 4, v2Id = 5, v3Id = 6, v4Id = 7;
    uint32_t thisCycleNum, thisCycleNumAlign, maxUbNum;

    DTYPE_VALUE hIm, wIm;
    DTYPE_VALUE w1 = 0, w2 = 0, w3 = 0, w4 = 0;
    int32_t h, w, levelStartId;
    int32_t offsetValue, offsetWeight, offsetLocation, offsetGrad, wStride, hStride;
    int32_t hLowPtrOffset, wLowPtrOffset;
    int32_t hLow, wLow;

    LocalTensor<DTYPE_VALUE> lowFloatLocal;
    LocalTensor<DTYPE_VALUE> xLocal, yLocal;
    LocalTensor<DTYPE_VALUE> distLowLocal;
    LocalTensor<DTYPE_VALUE> locWLocal, locHLocal;
    LocalTensor<DTYPE_VALUE> imLocal;
    LocalTensor<DTYPE_VALUE> zerosLocal;
    LocalTensor<DTYPE_VALUE> weightSumLocal, tmpALocal, tmpBLocal;
    LocalTensor<DTYPE_VALUE> topGradLocal, attentionWeightLocal;
    LocalTensor<DTYPE_VALUE> wv1Local, wv2Local, wv3Local, wv4Local, w4Local;
    LocalTensor<DTYPE_VALUE> mid1Local, mid2Local, mid3Local, mid4Local;
    LocalTensor<int32_t> shapesLocal, offsetLocal;
    LocalTensor<int32_t> lowLocal;

    SumParams sumParams;
    DataCopyParams copyParams;
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<DTYPE_VALUE> padParams {false, 0, 0, 0};

    event_t eventIdVToMte2, eventIdVToMte3, eventIdMte2ToV, eventIdMte3ToV, eventIdVToMteWeight, eventIdVToMte3X,
        eventIdVToMte3Y, eventIdMte2ToV_1, eventIdMte2ToV_2;

    TBuf<TPosition::VECCALC> wv1Ub, wv2Ub, wv3Ub, wv4Ub, mid1Ub, mid2Ub, mid3Ub, mid4Ub;
};
#endif // MS_DEFORM_ATTN_GRAD_HIGH_PERF_V2_H_
