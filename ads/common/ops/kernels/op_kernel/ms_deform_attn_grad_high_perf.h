/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef _MS_DEFORM_ATTN_GRAD_HIGH_PERF_H_
#define _MS_DEFORM_ATTN_GRAD_HIGH_PERF_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

template <uint32_t NUM_POINTS>
class MultiScaleDeformableAttentionV2GradHighPerf {
public:
    __aicore__ inline MultiScaleDeformableAttentionV2GradHighPerf(){};
    __aicore__ inline void Init(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm, GM_ADDR level_start_index_gm,
                                GM_ADDR sampling_loc_gm, GM_ADDR attn_weight_gm, GM_ADDR grad_output_gm,
                                GM_ADDR grad_value_gm, GM_ADDR grad_sampling_loc_gm, GM_ADDR grad_attn_weight_gm,
                                const MultiScaleDeformableAttentionV2GradTilingData *tiling_data, TPipe *tmpPipe)
    {
        pipe = tmpPipe;
        curBlockIdx = GetBlockIdx();
        blockBytes = 32;

        numPointsAlign = 8;
        dataAlign = blockBytes / sizeof(DTYPE_VALUE);

        embedDims = 32;
        numKeys = tiling_data->numKeys;
        numHeads = tiling_data->numHeads;
        numLevels = tiling_data->numLevels;
        numQueries = tiling_data->numQueries;
        batchSize = tiling_data->batchSize;
        coreNum = tiling_data->coreNum;

        taskNum = numQueries;
        taskNumPerCore = DivCeil(taskNum, coreNum);

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
        weightStride0 = numLevels * NUM_POINTS;
        weightStride1 = numHeads * weightStride0;
        weightStride2 = numQueries * weightStride1;
        valueStride0 = embedDims;
        valueStride1 = numKeys * valueStride0;
        valueStride2 = numHeads * valueStride1;

        baseOffsetUb = NUM_POINTS * embedDims;

        eventIdMte2ToV_1 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdMte2ToV_2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdMte2ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdMte3ToV = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());
        eventIdVToMte2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMte3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdVToMteWeight = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());

        copyParams = {1, (uint16_t)(NUM_POINTS * sizeof(DTYPE_VALUE)), 0, 0};
        sumParams = {NUM_POINTS, embedDims, embedDims};

        copyParamsV2 = {1, (uint16_t)(2 * NUM_POINTS * sizeof(DTYPE_VALUE)), 0, 0};
        sumParamsV2 = {2 * NUM_POINTS, embedDims, embedDims};
        mulParams = {1, 1, 1, dstRepStride, src0RepStride, src1RepStride};

        valueGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(value_gm),
                                batchSize * numKeys * numHeads * embedDims);
        valueSpatialShapesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_SPATIAL_SHAPES *>(spatial_shapes_gm),
                                             numLevels * 2);
        valueLevelStartIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_SPATIAL_SHAPES *>(level_start_index_gm),
                                               numLevels);
        locationGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(sampling_loc_gm),
                                   batchSize * numQueries * numHeads * numLevels * NUM_POINTS * 2);
        attentionWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(attn_weight_gm),
                                           batchSize * numQueries * numHeads * numLevels * NUM_POINTS);
        gradOutputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(grad_output_gm),
                                     batchSize * numQueries * numHeads * embedDims);

        gradValueGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(grad_value_gm),
                                    batchSize * numKeys * numHeads * embedDims);
        gradLocationGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(grad_sampling_loc_gm),
                                       batchSize * numQueries * numHeads * numLevels * 2 * NUM_POINTS);
        gradWeightGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(grad_attn_weight_gm),
                                     batchSize * numQueries * numHeads * numLevels * NUM_POINTS);
    }

    __aicore__ inline void InitBuffer()
    {
        pipe->InitBuffer(shapeUb, 2 * numLevelsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(offsetUb, numLevelsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(locationUb, numHeads * numLevels * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(attentionWeightsUb, numHeads * numLevels * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(topGradUb, embedDims * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(locSumUb, 2 * numLevels * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(weightSumUb, 2 * numLevels * numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(locUb, 2 * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(imUb, 2 * numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(zerosUb, 4 * NUM_POINTS * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(v1Ub, NUM_POINTS * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(v2Ub, NUM_POINTS * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(v3Ub, NUM_POINTS * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(v4Ub, NUM_POINTS * embedDims * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(w1v1Ub, NUM_POINTS * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(w2v2Ub, NUM_POINTS * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(w3v3Ub, NUM_POINTS * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(w4v4Ub, NUM_POINTS * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpUb, 2 * NUM_POINTS * embedDims * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(tmpAUb, embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpBUb, embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(midUb, 4 * NUM_POINTS * embedDims * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(gradSampleLocUb, 2 * NUM_POINTS * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(whUb, 2 * NUM_POINTS * embedDims * sizeof(DTYPE_VALUE));
    }

    __aicore__ inline void GetLocalTensor()
    {
        locationLocal = locationUb.Get<DTYPE_VALUE>();
        attentionWeightLocal = attentionWeightsUb.Get<DTYPE_VALUE>();
        shapesLocal = shapeUb.Get<DTYPE_SPATIAL_SHAPES>();
        offsetLocal = offsetUb.Get<DTYPE_SPATIAL_SHAPES>();
        locSumLocal = locSumUb.Get<DTYPE_VALUE>();

        weightSumLocal = weightSumUb.Get<DTYPE_VALUE>();
        topGradLocal = topGradUb.Get<DTYPE_VALUE>();
        locLocal = locUb.Get<DTYPE_VALUE>();
        imLocal = imUb.Get<DTYPE_VALUE>();
        zerosLocal = zerosUb.Get<DTYPE_VALUE>();
        v1Local = v1Ub.Get<DTYPE_VALUE>();
        v2Local = v2Ub.Get<DTYPE_VALUE>();
        v3Local = v3Ub.Get<DTYPE_VALUE>();
        v4Local = v4Ub.Get<DTYPE_VALUE>();
        w1v1Local = w1v1Ub.Get<DTYPE_VALUE>();
        w2v2Local = w2v2Ub.Get<DTYPE_VALUE>();
        w3v3Local = w3v3Ub.Get<DTYPE_VALUE>();
        w4v4Local = w4v4Ub.Get<DTYPE_VALUE>();
        tmpLocal = tmpUb.Get<DTYPE_VALUE>();

        tmpALocal = tmpAUb.Get<DTYPE_VALUE>();
        tmpBLocal = tmpBUb.Get<DTYPE_VALUE>();
        midLocal = midUb.Get<DTYPE_VALUE>();

        gradSampleLocLocal = gradSampleLocUb.Get<DTYPE_VALUE>();
        whLocLocal = whUb.Get<DTYPE_VALUE>();
    }

    __aicore__ inline void ClearOutput()
    {
        switch (curBlockIdx) {
            case 0:
                InitOutput<DTYPE_VALUE>(gradValueGm, batchSize * numKeys * numHeads * embedDims, 0);
                break;
            case 1:
                InitOutput<DTYPE_VALUE>(gradLocationGm, 2 * batchSize * numQueries * numHeads * numLevels * NUM_POINTS);
                break;
            case 2:
                InitOutput<DTYPE_VALUE>(gradWeightGm, batchSize * numQueries * numHeads * numLevels * NUM_POINTS);
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
        if (innerClean == 1) {
            ClearOutput();
        }
        DataCopy(shapesLocal, valueSpatialShapesGm, 2 * numLevelsAlign);
        DataCopy(offsetLocal, valueLevelStartIndexGm, numLevelsAlign);
        SetAtomicAdd<DTYPE_VALUE>();
        Compute();
        SetAtomicNone();
    }

    __aicore__ inline void ReleaseEventID()
    {
        pipe->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
        pipe->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV_1);
        pipe->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV_2);
        pipe->ReleaseEventID<HardEvent::MTE3_V>(eventIdMte3ToV);
        pipe->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMteWeight);
    }

private:
    __aicore__ inline DTYPE_SPATIAL_SHAPES floor(DTYPE_VALUE x)
    {
        DTYPE_SPATIAL_SHAPES res = static_cast<DTYPE_SPATIAL_SHAPES>(x);
        if (x >= 0 || x == res) {
            return res;
        } else {
            return res - 1;
        }
    }

    __aicore__ inline void clearUB()
    {
        Duplicate(zerosLocal, (DTYPE_VALUE)0, 4 * NUM_POINTS * embedDims);
        Duplicate(v1Local, (DTYPE_VALUE)0, NUM_POINTS * embedDims);
        Duplicate(v2Local, (DTYPE_VALUE)0, NUM_POINTS * embedDims);
        Duplicate(v3Local, (DTYPE_VALUE)0, NUM_POINTS * embedDims);
        Duplicate(v4Local, (DTYPE_VALUE)0, NUM_POINTS * embedDims);
        Duplicate(w1v1Local, (DTYPE_VALUE)0, NUM_POINTS * embedDims);
        Duplicate(w2v2Local, (DTYPE_VALUE)0, NUM_POINTS * embedDims);
        Duplicate(w3v3Local, (DTYPE_VALUE)0, NUM_POINTS * embedDims);
        Duplicate(w4v4Local, (DTYPE_VALUE)0, NUM_POINTS * embedDims);
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

                        Duplicate(whLocLocal, (DTYPE_VALUE)w, NUM_POINTS * embedDims);
                        Duplicate(whLocLocal[baseOffsetUb], (DTYPE_VALUE)h, NUM_POINTS * embedDims);

                        offsetValue = batch * valueStride2 + head * valueStride1 + levelStartId * valueStride0;
                        wStride = embedDims;
                        hStride = w * wStride;
                        tmpOffset = offsetWeight + level * NUM_POINTS;

                        DataCopy(locLocal, locationGm[tmpOffset * 2], numPointsAlign);
                        DataCopy(locLocal[numPointsAlign], locationGm[tmpOffset * 2 + NUM_POINTS], numPointsAlign);

                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                        DataCopy(attentionWeightLocal, attentionWeightsGm[tmpOffset], numPointsAlign);
                        Muls(imLocal[numPointsAlign], locLocal[numPointsAlign], (DTYPE_VALUE)h, numPointsAlign);
                        Muls(imLocal, locLocal, (DTYPE_VALUE)w, numPointsAlign);
                        Adds(imLocal, imLocal, DTYPE_VALUE(-0.5), 2 * numPointsAlign);
                        clearUB();
                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                        for (point = 0; point < NUM_POINTS; point++) {
                            hIm = imLocal.GetValue(numPointsAlign + point);
                            wIm = imLocal.GetValue(point);
                            if (hIm > -1 && wIm > -1 && hIm < h && wIm < w) {
                                pointOffset = point * embedDims;
                                hLow = floor(hIm);
                                wLow = floor(wIm);
                                hLowPtrOffset = hLow * hStride;
                                wLowPtrOffset = wLow * wStride;
                                uint32_t offsetTopGradValue = pointOffset + topGradValueId * baseOffsetUb;
                                Muls(zerosLocal[offsetTopGradValue], topGradLocal, attentionWeightLocal.GetValue(point),
                                     embedDims);
                                if (hLow >= 0) {
                                    if (wLow >= 0 && wLow < w - 1) {
                                        DTYPE_VALUE distW2 = wIm - wLow;
                                        DTYPE_VALUE distH = 1 - hIm + hLow;
                                        DTYPE_VALUE distW = 1 - distW2;
                                        w1 = distH * distW;
                                        w2 = distH * distW2;
                                        uint32_t offsetMid = point * 4 * embedDims;
                                        uint32_t offsetMid_ = (point * 4 + mid2Id) * embedDims;
                                        uint32_t offsetGradHWeight = pointOffset + gradHWeightId * baseOffsetUb;
                                        uint32_t offsetGradWWeight = pointOffset + gradWWeightId * baseOffsetUb;
                                        uint32_t ptr = hLowPtrOffset + wLowPtrOffset;
                                        DataCopy(v1Local[pointOffset], valueGm[offsetValue + ptr], embedDims);
                                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);
                                        DataCopy(v2Local[pointOffset], valueGm[offsetValue + ptr + wStride], embedDims);
                                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_2);

                                        Muls(midLocal[offsetMid], zerosLocal[offsetTopGradValue], w1, embedDims);

                                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);
                                        Muls(tmpALocal, v1Local[pointOffset], distW, embedDims);
                                        Muls(tmpBLocal, v1Local[pointOffset], distH, embedDims);
                                        Sub(zerosLocal[offsetGradHWeight], zerosLocal[offsetGradHWeight], tmpALocal,
                                            embedDims);
                                        Sub(zerosLocal[offsetGradWWeight], zerosLocal[offsetGradWWeight], tmpBLocal,
                                            embedDims);

                                        Muls(midLocal[offsetMid_], zerosLocal[offsetTopGradValue], w2, embedDims);
                                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_2);
                                        Muls(tmpALocal, v2Local[pointOffset], distW2, embedDims);
                                        Muls(tmpBLocal, v2Local[pointOffset], distH, embedDims);
                                        Sub(zerosLocal[offsetGradHWeight], zerosLocal[offsetGradHWeight], tmpALocal,
                                            embedDims);
                                        Add(zerosLocal[offsetGradWWeight], zerosLocal[offsetGradWWeight], tmpBLocal,
                                            embedDims);

                                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                                        DataCopy(gradValueGm[offsetValue + ptr], midLocal[offsetMid], 2 * embedDims);
                                    } else if (wLow >= 0) {
                                        DTYPE_VALUE distH = 1 - hIm + hLow;
                                        DTYPE_VALUE distW = 1 - wIm + wLow;
                                        w1 = distH * distW;
                                        uint32_t offsetMid = (point * 4 + mid1Id) * embedDims;
                                        uint32_t offsetGradHWeight = pointOffset + gradHWeightId * baseOffsetUb;
                                        uint32_t offsetGradWWeight = pointOffset + gradWWeightId * baseOffsetUb;
                                        uint32_t ptr = hLowPtrOffset + wLowPtrOffset;
                                        DataCopy(v1Local[pointOffset], valueGm[offsetValue + ptr], embedDims);
                                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                                        Muls(midLocal[offsetMid], zerosLocal[offsetTopGradValue], w1, embedDims);
                                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                                        Muls(tmpALocal, v1Local[pointOffset], distW, embedDims);
                                        Muls(tmpBLocal, v1Local[pointOffset], distH, embedDims);
                                        Sub(zerosLocal[offsetGradHWeight], zerosLocal[offsetGradHWeight], tmpALocal,
                                            embedDims);
                                        Sub(zerosLocal[offsetGradWWeight], zerosLocal[offsetGradWWeight], tmpBLocal,
                                            embedDims);

                                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                                        DataCopy(gradValueGm[offsetValue + ptr], midLocal[offsetMid], embedDims);
                                    } else if (wLow < w - 1) {
                                        DTYPE_VALUE distH = 1 - hIm + hLow;
                                        DTYPE_VALUE distW = wIm - wLow;
                                        w2 = distH * distW;
                                        uint32_t offsetMid = (point * 4 + mid2Id) * embedDims;
                                        uint32_t offsetGradHWeight = pointOffset + gradHWeightId * baseOffsetUb;
                                        uint32_t offsetGradWWeight = pointOffset + gradWWeightId * baseOffsetUb;
                                        uint32_t ptr = hLowPtrOffset + wLowPtrOffset + wStride;
                                        DataCopy(v2Local[pointOffset], valueGm[offsetValue + ptr], embedDims);
                                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                                        Muls(midLocal[offsetMid], zerosLocal[offsetTopGradValue], w2, embedDims);
                                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                                        Muls(tmpALocal, v2Local[pointOffset], distW, embedDims);
                                        Muls(tmpBLocal, v2Local[pointOffset], distH, embedDims);
                                        Sub(zerosLocal[offsetGradHWeight], zerosLocal[offsetGradHWeight], tmpALocal,
                                            embedDims);
                                        Add(zerosLocal[offsetGradWWeight], zerosLocal[offsetGradWWeight], tmpBLocal,
                                            embedDims);

                                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                                        DataCopy(gradValueGm[offsetValue + ptr], midLocal[offsetMid], embedDims);
                                    }
                                }
                                if (hLow < h - 1) {
                                    if (wLow >= 0 && wLow < w - 1) {
                                        DTYPE_VALUE distW2 = wIm - wLow;
                                        DTYPE_VALUE distH = hIm - hLow;
                                        DTYPE_VALUE distW = 1 - distW2;
                                        w3 = distH * distW;
                                        w4 = distH * distW2;
                                        uint32_t offsetMid = (point * 4 + mid3Id) * embedDims;
                                        uint32_t offsetMid_ = (point * 4 + mid4Id) * embedDims;
                                        uint32_t offsetGradHWeight = pointOffset + gradHWeightId * baseOffsetUb;
                                        uint32_t offsetGradWWeight = pointOffset + gradWWeightId * baseOffsetUb;
                                        uint32_t ptr = hLowPtrOffset + hStride + wLowPtrOffset;
                                        DataCopy(v3Local[pointOffset], valueGm[offsetValue + ptr], embedDims);
                                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);
                                        DataCopy(v4Local[pointOffset], valueGm[offsetValue + ptr + wStride], embedDims);
                                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_2);

                                        Muls(midLocal[offsetMid], zerosLocal[offsetTopGradValue], w3, embedDims);

                                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);
                                        Muls(tmpALocal, v3Local[pointOffset], distW, embedDims);
                                        Muls(tmpBLocal, v3Local[pointOffset], distH, embedDims);
                                        Add(zerosLocal[offsetGradHWeight], zerosLocal[offsetGradHWeight], tmpALocal,
                                            embedDims);
                                        Sub(zerosLocal[offsetGradWWeight], zerosLocal[offsetGradWWeight], tmpBLocal,
                                            embedDims);

                                        Muls(midLocal[offsetMid_], zerosLocal[offsetTopGradValue], w4, embedDims);
                                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_2);
                                        Muls(tmpALocal, v4Local[pointOffset], distW2, embedDims);
                                        Muls(tmpBLocal, v4Local[pointOffset], distH, embedDims);
                                        Add(zerosLocal[offsetGradHWeight], zerosLocal[offsetGradHWeight], tmpALocal,
                                            embedDims);
                                        Add(zerosLocal[offsetGradWWeight], zerosLocal[offsetGradWWeight], tmpBLocal,
                                            embedDims);

                                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                                        DataCopy(gradValueGm[offsetValue + ptr], midLocal[offsetMid], 2 * embedDims);
                                    } else if (wLow >= 0) {
                                        DTYPE_VALUE distH = hIm - hLow;
                                        DTYPE_VALUE distW = 1 - wIm + wLow;
                                        w3 = distH * distW;
                                        uint32_t offsetMid = (point * 4 + mid3Id) * embedDims;
                                        uint32_t offsetGradHWeight = pointOffset + gradHWeightId * baseOffsetUb;
                                        uint32_t offsetGradWWeight = pointOffset + gradWWeightId * baseOffsetUb;
                                        uint32_t ptr = hLowPtrOffset + hStride + wLowPtrOffset;
                                        DataCopy(v3Local[pointOffset], valueGm[offsetValue + ptr], embedDims);
                                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                                        Muls(midLocal[offsetMid], zerosLocal[offsetTopGradValue], w3, embedDims);
                                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                                        Muls(tmpALocal, v3Local[pointOffset], distW, embedDims);
                                        Muls(tmpBLocal, v3Local[pointOffset], distH, embedDims);
                                        Add(zerosLocal[offsetGradHWeight], zerosLocal[offsetGradHWeight], tmpALocal,
                                            embedDims);
                                        Sub(zerosLocal[offsetGradWWeight], zerosLocal[offsetGradWWeight], tmpBLocal,
                                            embedDims);

                                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                                        DataCopy(gradValueGm[offsetValue + ptr], midLocal[offsetMid], embedDims);
                                    } else if (wLow < w - 1) {
                                        DTYPE_VALUE distH = hIm - hLow;
                                        DTYPE_VALUE distW = wIm - wLow;
                                        w4 = distH * distW;
                                        uint32_t offsetMid = (point * 4 + mid4Id) * embedDims;
                                        uint32_t offsetGradHWeight = pointOffset + gradHWeightId * baseOffsetUb;
                                        uint32_t offsetGradWWeight = pointOffset + gradWWeightId * baseOffsetUb;
                                        uint32_t ptr = hLowPtrOffset + hStride + wLowPtrOffset + wStride;
                                        DataCopy(v4Local[pointOffset], valueGm[offsetValue + ptr], embedDims);
                                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                                        Muls(midLocal[offsetMid], zerosLocal[offsetTopGradValue], w4, embedDims);
                                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                                        Muls(tmpALocal, v4Local[pointOffset], distW, embedDims);
                                        Muls(tmpBLocal, v4Local[pointOffset], distH, embedDims);
                                        Add(zerosLocal[offsetGradHWeight], zerosLocal[offsetGradHWeight], tmpALocal,
                                            embedDims);
                                        Add(zerosLocal[offsetGradWWeight], zerosLocal[offsetGradWWeight], tmpBLocal,
                                            embedDims);

                                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                                        DataCopy(gradValueGm[offsetValue + ptr], midLocal[offsetMid], embedDims);
                                    }
                                }
                                Muls(w1v1Local[pointOffset], v1Local[pointOffset], w1, embedDims);
                                Muls(w2v2Local[pointOffset], v2Local[pointOffset], w2, embedDims);
                                Muls(w3v3Local[pointOffset], v3Local[pointOffset], w3, embedDims);
                                Muls(w4v4Local[pointOffset], v4Local[pointOffset], w4, embedDims);
                            }
                        }
                        Add(w1v1Local, w1v1Local, w2v2Local, NUM_POINTS * embedDims);
                        Add(w1v1Local, w1v1Local, w3v3Local, NUM_POINTS * embedDims);
                        Add(w1v1Local, w1v1Local, w4v4Local, NUM_POINTS * embedDims);
                        Mul(zerosLocal[gradWeightId * baseOffsetUb], w1v1Local, topGradLocal, embedDims, NUM_POINTS,
                            mulParams);
                        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
                        if (NUM_POINTS < 8) {
                            Mul(tmpLocal, zerosLocal[topGradValueId * baseOffsetUb],
                                zerosLocal[gradWWeightId * baseOffsetUb], NUM_POINTS * embedDims);
                            Mul(tmpLocal[baseOffsetUb], zerosLocal[topGradValueId * baseOffsetUb],
                                zerosLocal[gradHWeightId * baseOffsetUb], NUM_POINTS * embedDims);

                            Mul(gradSampleLocLocal, tmpLocal, whLocLocal, 2 * NUM_POINTS * embedDims);

                            Sum(weightSumLocal, zerosLocal[gradWeightId * baseOffsetUb], sumParams);
                            SetFlag<HardEvent::V_MTE3>(eventIdVToMteWeight);
                            Sum(locSumLocal, gradSampleLocLocal, sumParamsV2);
                            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                            WaitFlag<HardEvent::V_MTE3>(eventIdVToMteWeight);
                            DataCopyPad(gradWeightGm[offsetWeight + level * NUM_POINTS], weightSumLocal, copyParams);

                            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                            DataCopyPad(gradLocationGm[offsetLocation + level * 2 * NUM_POINTS], locSumLocal,
                                        copyParamsV2);
                        } else {
                            Mul(tmpLocal, zerosLocal[topGradValueId * baseOffsetUb],
                                zerosLocal[gradWWeightId * baseOffsetUb], NUM_POINTS * embedDims);
                            Mul(tmpLocal[baseOffsetUb], zerosLocal[topGradValueId * baseOffsetUb],
                                zerosLocal[gradHWeightId * baseOffsetUb], NUM_POINTS * embedDims);
                            Mul(gradSampleLocLocal, tmpLocal, whLocLocal, 2 * NUM_POINTS * embedDims);

                            Sum(weightSumLocal[NUM_POINTS * level], zerosLocal[gradWeightId * baseOffsetUb], sumParams);
                            Sum(locSumLocal[NUM_POINTS * 2 * level], gradSampleLocLocal, sumParamsV2);
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

private:
    TPipe *pipe;
    GlobalTensor<DTYPE_VALUE> valueGm, locationGm, attentionWeightsGm, gradOutputGm, gradValueGm, gradLocationGm,
        gradWeightGm;
    GlobalTensor<DTYPE_SPATIAL_SHAPES> valueSpatialShapesGm, valueLevelStartIndexGm;

    TBuf<TPosition::VECCALC> locationUb, attentionWeightsUb, shapeUb, offsetUb, topGradUb;
    TBuf<TPosition::VECCALC> locSumUb, weightSumUb;
    TBuf<TPosition::VECCALC> zerosUb;
    TBuf<TPosition::VECCALC> locUb, imUb, lowUb;
    TBuf<TPosition::VECCALC> v1Ub, v2Ub, v3Ub, v4Ub;
    TBuf<TPosition::VECCALC> w1v1Ub, w2v2Ub, w3v3Ub, w4v4Ub, tmpUb, tmpAUb, tmpBUb, midUb;
    TBuf<TPosition::VECCALC> gradSampleLocUb, whUb;

    uint32_t coreNum;
    uint32_t batchSize, numKeys, numHeads, embedDims, numLevels, numQueries;
    uint32_t numLevelsAlign, numPointsAlign;
    uint32_t batch, query, head, level, point;
    uint32_t curBlockIdx;
    uint32_t taskNum, taskNumPerCore;
    uint32_t startOffset, endOffset;
    uint32_t dataAlign, blockBytes;
    uint32_t gradOutStride0, gradOutStride1, gradOutStride2;
    uint32_t weightStride0, weightStride1, weightStride2;
    uint32_t valueStride0, valueStride1, valueStride2;
    uint32_t baseOffsetUb, pointOffset, tmpOffset;
    uint32_t mid1Id = 0, mid2Id = 1, mid3Id = 2, mid4Id = 3;
    uint32_t gradWWeightId = 0, gradHWeightId = 1, topGradValueId = 2, gradWeightId = 3;
    uint32_t dstRepStride = 4, src0RepStride = 4, src1RepStride = 0;
    uint32_t innerClean = 0;

    DTYPE_VALUE hIm, wIm;
    DTYPE_VALUE w1 = 0, w2 = 0, w3 = 0, w4 = 0;
    DTYPE_SPATIAL_SHAPES h, w, levelStartId;
    DTYPE_SPATIAL_SHAPES offsetValue, offsetWeight, offsetLocation, wStride, hStride;
    DTYPE_SPATIAL_SHAPES hLowPtrOffset, wLowPtrOffset;
    DTYPE_SPATIAL_SHAPES hLow, wLow;

    LocalTensor<DTYPE_VALUE> locSumLocal;
    LocalTensor<DTYPE_VALUE> locLocal;
    LocalTensor<DTYPE_VALUE> imLocal;
    LocalTensor<DTYPE_VALUE> zerosLocal;
    LocalTensor<DTYPE_VALUE> v1Local, v2Local, v3Local, v4Local;
    LocalTensor<DTYPE_VALUE> w1v1Local, w2v2Local, w3v3Local, w4v4Local;
    LocalTensor<DTYPE_VALUE> weightSumLocal, midLocal, tmpLocal, tmpALocal, tmpBLocal;
    LocalTensor<DTYPE_VALUE> gradSampleLocLocal, whLocLocal;
    LocalTensor<DTYPE_VALUE> topGradLocal, locationLocal, attentionWeightLocal;
    LocalTensor<DTYPE_SPATIAL_SHAPES> shapesLocal, offsetLocal;
    LocalTensor<DTYPE_SPATIAL_SHAPES> lowLocal;

    SumParams sumParams, sumParamsV2;
    DataCopyParams copyParams, copyParamsV2;
    BinaryRepeatParams mulParams;
    event_t eventIdVToMte2, eventIdVToMte3, eventIdMte2ToV, eventIdMte3ToV, eventIdVToMteWeight, eventIdMte2ToV_1,
        eventIdMte2ToV_2;
};
#endif // _MS_DEFORM_ATTN_GRAD_HIGH_PERF_H_