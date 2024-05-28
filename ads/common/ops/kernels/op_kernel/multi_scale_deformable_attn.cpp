/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * This sample is a very basic sample that implements vector add on Ascend plaform.
 */
#include "kernel_operator.h"
using namespace AscendC;

class KernelMultiScaleDeformableAttn {
public:
    __aicore__ inline KernelMultiScaleDeformableAttn() {}
    __aicore__ inline void Init(GM_ADDR value, GM_ADDR valueSpatialShapes, GM_ADDR valuLevelStartIndex,
        GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output,
        const MultiScaleDeformableAttnTilingData* tiling_data, TPipe* tmpPipe)
    {
        pipe = tmpPipe;
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        dataAlign = blockNum / sizeof(DTYPE_VALUE);
        batchSize = tiling_data->batchSize;
        numKeys = tiling_data->numKeys;
        numHeads = tiling_data->numHeads;
        embedDims = tiling_data->embedDims;

        numLevels = tiling_data->numLevels;
        numQueries = tiling_data->numQueries;
        numPoints = tiling_data->numPoints;
        coreNum = tiling_data->coreNum;

        tailNum = numHeads * embedDims;
        taskNum = numQueries;

        numPointsAlign = AlignUp(numPoints, dataAlign);

        if (embedDims >= 256) {
            miniBatch = miniBatch / 4;
        } else if (embedDims >= 128) {
            miniBatch = miniBatch / 2;
        }
        if (numQueries % miniBatch != 0) {
            miniBatch = 8;
        }

        taskNum = DivCeil(numQueries, miniBatch);

        taskNumPerCore = DivCeil(taskNum, coreNum);

        numPointsAlign = AlignUp(numPoints, dataAlign);
        numLevelsAlign = AlignUp(numLevels, dataAlign);

        batchOffset = numPoints * embedDims;

        curBlockIdx = GetBlockIdx();
        startLoop = curBlockIdx * taskNumPerCore;
        endLoop = (curBlockIdx + 1) * taskNumPerCore;
        if (endLoop > taskNum) {
            endLoop = taskNum;
        }

        valueGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_VALUE*>(value), batchSize * numHeads * numKeys * embedDims);
        locationGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE*>(samplingLocations),
            batchSize * numHeads * numLevels * numPoints * 2 * numQueries);
        attentionWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE*>(attentionWeights),
            batchSize * numHeads * numLevels * numPoints * numQueries);
        outputGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_VALUE*>(output), batchSize * numQueries * numHeads * embedDims);

        valueSpatialShapesGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_VALUE_SPATIAL_SHAPES*>(valueSpatialShapes), numLevels * 2);
        valueLevelStartIndexGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_VALUE_SPATIAL_SHAPES*>(valuLevelStartIndex), numLevels);

        pipe->InitBuffer(shapeQueue, AlignUp(numLevels * 2, dataAlign) * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(offsetQueue, numLevelsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(locationQueue, 2 * miniBatch * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(attentionWeightQueue, miniBatch * numPoints * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(outputQueue, embedDims * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(floatOneUb, miniBatch * numPoints * 2 * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(tmpXUb, miniBatch * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpParamUb, miniBatch * numPoints * 2 * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(tmpIntUb, 4 * miniBatch * numPoints * sizeof(DTYPE_VALUE_SPATIAL_SHAPES));
        pipe->InitBuffer(tmpFloatUb, 4 * miniBatch * numPoints * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(halfUb, 2 * miniBatch * numPoints * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(locUb, 2 * miniBatch * numPoints * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(weightQueue, 4 * miniBatch * numPoints * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(valueUb, 4 * numPointsAlign * embedDims * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(cornerWeightUb, 4 * miniBatch * numPointsAlign * embedDims * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(tmpResUb, 2 * batchOffset * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpResUb2, miniBatch * numPoints * embedDims * sizeof(DTYPE_VALUE));
    }

    __aicore__ inline void Process()
    {
#if __CCE_AICORE__ == 220
        if (embedDims == 32 && numPoints == 4 && numLevels == 1) {
            ComputePointOpt<4>();
        } else if (embedDims == 32 && numPoints == 8 && numLevels == 4) {
            ComputePointEight();
        } else {
            ComputePointCommon();
        }
#else
        ComputePointCommon();
#endif
    }

private:
    __aicore__ inline bool isInRange(DTYPE_VALUE_SPATIAL_SHAPES x, DTYPE_VALUE_SPATIAL_SHAPES upper)
    {
        return -1 < x && x < upper;
    }

    __aicore__ inline void ComputePointEight()
    {
        event_t eventIdVToMte3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        event_t eventIdVToMte2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>());
        event_t eventIdMte2ToV_0 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        event_t eventIdMte2ToV_1 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        event_t eventIdMte2ToV_2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());

        LocalTensor<DTYPE_VALUE> locationLocal = locationQueue.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> attentionWeightLocal = attentionWeightQueue.Get<DTYPE_VALUE>();

        LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> shapesLocal = shapeQueue.Get<DTYPE_VALUE_SPATIAL_SHAPES>();
        LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> offsetLocal = offsetQueue.Get<DTYPE_VALUE_SPATIAL_SHAPES>();
        LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> tmpIntLocal = tmpIntUb.Get<DTYPE_VALUE_SPATIAL_SHAPES>();
        LocalTensor<DTYPE_VALUE> tmpFloatLocal = tmpFloatUb.Get<DTYPE_VALUE>();

        DataCopy(shapesLocal, valueSpatialShapesGm, AlignUp(numLevels * 2, dataAlign));
        DataCopy(offsetLocal, valueLevelStartIndexGm, numLevelsAlign);

        LocalTensor<DTYPE_VALUE> valueLocal = valueUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> cornerWeightLocal = cornerWeightUb.Get<DTYPE_VALUE>();

        LocalTensor<DTYPE_VALUE> floatOneLocal = floatOneUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> halfLocal = halfUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> locLocal = locUb.Get<DTYPE_VALUE>();

        Duplicate<DTYPE_VALUE>(floatOneLocal, (DTYPE_VALUE)1, miniBatch * numPointsAlignOpt * 2);
        Duplicate<DTYPE_VALUE>(halfLocal, (DTYPE_VALUE)0.5, miniBatch * numPointsAlignOpt * 2);

        LocalTensor<DTYPE_VALUE> weightLocal = weightQueue.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> xLocal = tmpXUb.Get<DTYPE_VALUE>();

        LocalTensor<DTYPE_VALUE> tmpResLocal = tmpResUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> tmpResLocal2 = tmpResUb2.Get<DTYPE_VALUE>();

        LocalTensor<DTYPE_VALUE> paramLocal = tmpParamUb.Get<DTYPE_VALUE>();

        uint32_t srcShape_[2] = {4 * numPointsAlignOpt, 1};
        uint32_t dstShape_[2] = {4 * numPointsAlignOpt, embedDimsOpt};

        DataCopyExtParams copyParams {2, uint32_t(embedDimsOpt * sizeof(DTYPE_VALUE)), 0,
            uint32_t((numPointsAlignOpt * embedDimsOptTwice - embedDimsOpt) * sizeof(DTYPE_VALUE) / 32), 0};
        DataCopyPadExtParams<DTYPE_VALUE> padParams {false, 0, 0, 0};

        SetAtomicAdd<DTYPE_VALUE>();
        for (uint32_t level = 0; level < numLevels; level++) {
            h = shapesLocal.GetValue(level * 2);
            w = shapesLocal.GetValue(level * 2 + 1);
            dataOffset = offsetLocal.GetValue(level);

            Duplicate<DTYPE_VALUE>(locLocal, (DTYPE_VALUE)w, miniBatch * numPointsAlignOpt);
            Duplicate<DTYPE_VALUE>(
                locLocal[miniBatch * numPointsAlignOpt], (DTYPE_VALUE)h, miniBatch * numPointsAlignOpt);
            for (uint32_t batch = 0; batch < batchSize; batch++) {
                for (uint32_t head = 0; head < numHeads; head++) {
                    baseOffset = batch * numHeads * numLevels * numQueries * numPointsAlignOpt +
                                 head * numLevels * numQueries * numPointsAlignOpt +
                                 level * numQueries * numPointsAlignOpt;

                    valueOffset = batch * numHeads * numKeys + head * numKeys + dataOffset;
                    moveOffset = (batch * numQueries * numHeads + head) * embedDimsOpt;

                    for (uint32_t queryloop = startLoop; queryloop < endLoop; queryloop++) {
                        queryBase = queryloop * miniBatch;
                        weightOffset = baseOffset + queryBase * numPointsAlignOpt;
                        locationOffset = baseOffset * 2 + queryBase * numPointsAlignOpt;

                        DataCopy(locationLocal, locationGm[locationOffset], miniBatch * numPointsAlignOpt);
                        DataCopy(locationLocal[miniBatch * numPointsAlignOpt],
                            locationGm[locationOffset + numQueries * numPointsAlignOpt], miniBatch * numPointsAlignOpt);

                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);
                        DataCopy(attentionWeightLocal, attentionWeightsGm[weightOffset], miniBatch * numPointsAlignOpt);
                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_2);

                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);
                        Mul(locationLocal, locationLocal, locLocal, 2 * miniBatch * numPointsAlignOpt);
                        Add(tmpFloatLocal, locationLocal, halfLocal, 2 * miniBatch * numPointsAlignOpt);
                        Cast(tmpIntLocal, tmpFloatLocal, RoundMode::CAST_FLOOR, 2 * miniBatch * numPointsAlignOpt);

                        Sub(tmpFloatLocal[miniBatch * numPointsAlignOpt * 2], tmpFloatLocal, floatOneLocal,
                            miniBatch * numPointsAlignOpt * 2);
                        Cast(tmpFloatLocal, tmpIntLocal, RoundMode::CAST_NONE, miniBatch * numPointsAlignOpt * 2);

                        Sub(paramLocal, tmpFloatLocal, tmpFloatLocal[miniBatch * numPointsAlignOpt * 2],
                            miniBatch * numPointsAlignOpt * 2);

                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_2);

                        for (uint32_t curQuery = 0; curQuery < miniBatch; curQuery++) {
                            Duplicate<DTYPE_VALUE>(valueLocal, DTYPE_VALUE(0), 4 * numPointsAlignOpt * embedDimsOpt);
                            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);

                            tmpOffset2 = curQuery * numPointsAlignOpt;
                            tmpOffset1 = tmpOffset2 + miniBatch * numPointsAlignOpt;
                            srcOffset = tmpOffset2 * embedDimsOpt;

                            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                            for (uint32_t point = 0; point < numPointsAlignOpt; point++) {
                                valueLocalOffset = point * embedDimsOpt;
                                y1 = tmpIntLocal.GetValue(tmpOffset1 + point);
                                x1 = tmpIntLocal.GetValue(tmpOffset2 + point);

                                x0 = x1 - 1;
                                y0 = y1 - 1;
                                if (isInRange(y0, h)) {
                                    if (0 < x1 && x1 < w) {
                                        DataCopyPad(valueLocal[valueLocalOffset],
                                            valueGm[(valueOffset + y0 * w + x0) * embedDimsOpt], copyParams, padParams);
                                    } else if (isInRange(x0, w)) {
                                        DataCopy(valueLocal[valueLocalOffset],
                                            valueGm[(valueOffset + y0 * w + x0) * embedDimsOpt], embedDimsOpt);
                                    } else if (isInRange(x1, w)) {
                                        DataCopy(valueLocal[valueLocalOffset + numPointsAlignOpt * embedDimsOptTwice],
                                            valueGm[(valueOffset + y0 * w + x1) * embedDimsOpt], embedDimsOpt);
                                    }
                                }
                                if (isInRange(y1, h)) {
                                    if (0 < x1 && x1 < w) {
                                        DataCopyPad(valueLocal[valueLocalOffset + numPointsAlignOpt * embedDimsOpt],
                                            valueGm[(valueOffset + y1 * w + x0) * embedDimsOpt], copyParams, padParams);
                                    } else if (isInRange(x0, w)) {
                                        DataCopy(valueLocal[valueLocalOffset + numPointsAlignOpt * embedDimsOpt],
                                            valueGm[(valueOffset + y1 * w + x0) * embedDimsOpt], embedDimsOpt);
                                    } else if (isInRange(x1, w)) {
                                        DataCopy(valueLocal[valueLocalOffset + numPointsAlignOpt * embedDimsOptTriple],
                                            valueGm[(valueOffset + y1 * w + x1) * embedDimsOpt], embedDimsOpt);
                                    }
                                }
                            }
                            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);

                            Sub(xLocal[tmpOffset2], floatOneLocal, paramLocal[tmpOffset2], numPointsAlignOpt);

                            Mul(weightLocal, paramLocal[tmpOffset2],
                                paramLocal[tmpOffset2 + miniBatch * numPointsAlignOpt], numPointsAlignOpt);
                            Sub(weightLocal[numPointsAlignOpt], paramLocal[tmpOffset2], weightLocal, numPointsAlignOpt);
                            Sub(weightLocal[numPointsAlignOpt * 2],
                                paramLocal[tmpOffset2 + miniBatch * numPointsAlignOpt], weightLocal, numPointsAlignOpt);
                            Sub(weightLocal[numPointsAlignOpt * 3], xLocal[tmpOffset2],
                                weightLocal[numPointsAlignOpt * 2], numPointsAlignOpt);

                            Mul(weightLocal, weightLocal, attentionWeightLocal[tmpOffset2], numPointsAlignOpt, 4,
                                {1, 1, 1, 1, 1, 0});

                            BroadCast<DTYPE_VALUE, 2, 1>(cornerWeightLocal, weightLocal, dstShape_, srcShape_);

                            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);

                            Mul(valueLocal, valueLocal, cornerWeightLocal, 4 * numPointsAlignOpt * embedDimsOpt);
                            Add(tmpResLocal, valueLocal, valueLocal[numPointsAlignOpt * embedDimsOpt * 2],
                                numPointsAlignOpt * embedDimsOpt * 2);
                            Add(tmpResLocal2[srcOffset], tmpResLocal, tmpResLocal[numPointsAlignOpt * embedDimsOpt],
                                numPointsAlignOpt * embedDimsOpt);

                            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                            dstOffset = moveOffset + (queryBase + curQuery) * numHeads * embedDimsOpt;
                            for (uint32_t point = 0; point < numPointsAlignOpt; point++) {
                                DataCopy(
                                    outputGm[dstOffset], tmpResLocal2[srcOffset + point * embedDimsOpt], embedDimsOpt);
                            }
                        }
                    }
                }
            }
        }
        SetAtomicNone();
    }

    template<uint32_t NUM_POINTS>
    __aicore__ inline void ComputePointOpt()
    {
        event_t eventIdVToMte3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        event_t eventIdVToMte2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>());
        event_t eventIdMte2ToV_0 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        event_t eventIdMte2ToV_1 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        event_t eventIdMte2ToV_2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());

        LocalTensor<DTYPE_VALUE> locationLocal = locationQueue.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> attentionWeightLocal = attentionWeightQueue.Get<DTYPE_VALUE>();

        LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> shapesLocal = shapeQueue.Get<DTYPE_VALUE_SPATIAL_SHAPES>();
        LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> offsetLocal = offsetQueue.Get<DTYPE_VALUE_SPATIAL_SHAPES>();
        LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> tmpIntLocal = tmpIntUb.Get<DTYPE_VALUE_SPATIAL_SHAPES>();
        LocalTensor<DTYPE_VALUE> tmpFloatLocal = tmpFloatUb.Get<DTYPE_VALUE>();

        DataCopy(shapesLocal, valueSpatialShapesGm, AlignUp(numLevels * 2, dataAlign));
        DataCopy(offsetLocal, valueLevelStartIndexGm, numLevelsAlign);

        LocalTensor<DTYPE_VALUE> valueLocal = valueUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> cornerWeightLocal = cornerWeightUb.Get<DTYPE_VALUE>();

        LocalTensor<DTYPE_VALUE> floatOneLocal = floatOneUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> halfLocal = halfUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> locLocal = locUb.Get<DTYPE_VALUE>();

        Duplicate<DTYPE_VALUE>(floatOneLocal, (DTYPE_VALUE)1, miniBatch * NUM_POINTS * 2);
        Duplicate<DTYPE_VALUE>(halfLocal, (DTYPE_VALUE)0.5, miniBatch * NUM_POINTS * 2);

        LocalTensor<DTYPE_VALUE> weightLocal = weightQueue.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> xLocal = tmpXUb.Get<DTYPE_VALUE>();

        LocalTensor<DTYPE_VALUE> tmpResLocal = tmpResUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> tmpResLocal2 = tmpResUb2.Get<DTYPE_VALUE>();

        LocalTensor<DTYPE_VALUE> paramLocal = tmpParamUb.Get<DTYPE_VALUE>();

        uint32_t srcShape_[2] = {4 * miniBatch * NUM_POINTS, 1};
        uint32_t dstShape_[2] = {4 * miniBatch * NUM_POINTS, embedDimsOpt};

        DataCopyExtParams copyParams {2, uint32_t(embedDimsOpt * sizeof(DTYPE_VALUE)), 0,
            uint32_t((NUM_POINTS * embedDimsOptTwice - embedDimsOpt) * sizeof(DTYPE_VALUE) / 32), 0};
        DataCopyPadExtParams<DTYPE_VALUE> padParams {false, 0, 0, 0};

        SetAtomicAdd<DTYPE_VALUE>();
        for (uint32_t level = 0; level < numLevels; level++) {
            h = shapesLocal.GetValue(level * 2);
            w = shapesLocal.GetValue(level * 2 + 1);
            dataOffset = offsetLocal.GetValue(level);

            Duplicate<DTYPE_VALUE>(locLocal, (DTYPE_VALUE)w, miniBatch * NUM_POINTS);
            Duplicate<DTYPE_VALUE>(locLocal[miniBatch * NUM_POINTS], (DTYPE_VALUE)h, miniBatch * NUM_POINTS);
            for (uint32_t batch = 0; batch < batchSize; batch++) {
                for (uint32_t head = 0; head < numHeads; head++) {
                    baseOffset = batch * numHeads * numLevels * numQueries * NUM_POINTS +
                                 head * numLevels * numQueries * NUM_POINTS + level * numQueries * NUM_POINTS;
                    valueOffset = batch * numHeads * numKeys + head * numKeys + dataOffset;
                    moveOffset = (batch * numQueries * numHeads + head) * embedDimsOpt;
                    for (uint32_t queryloop = startLoop; queryloop < endLoop; queryloop++) {
                        queryBase = queryloop * miniBatch;
                        weightOffset = baseOffset + queryBase * NUM_POINTS;
                        locationOffset = baseOffset * 2 + queryBase * NUM_POINTS;

                        DataCopy(locationLocal, locationGm[locationOffset], miniBatch * NUM_POINTS);
                        DataCopy(locationLocal[miniBatch * NUM_POINTS],
                            locationGm[locationOffset + numQueries * NUM_POINTS], miniBatch * NUM_POINTS);
                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);
                        DataCopy(attentionWeightLocal, attentionWeightsGm[weightOffset], miniBatch * NUM_POINTS);
                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_2);

                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);
                        Mul(locationLocal, locationLocal, locLocal, 2 * miniBatch * NUM_POINTS);
                        Add(tmpFloatLocal, locationLocal, halfLocal, 2 * miniBatch * NUM_POINTS);
                        Cast(tmpIntLocal, tmpFloatLocal, RoundMode::CAST_FLOOR, 2 * miniBatch * NUM_POINTS);

                        Sub(tmpFloatLocal[miniBatch * NUM_POINTS * 2], tmpFloatLocal, floatOneLocal,
                            miniBatch * NUM_POINTS * 2);
                        Cast(tmpFloatLocal, tmpIntLocal, RoundMode::CAST_NONE, miniBatch * NUM_POINTS * 2);

                        Sub(paramLocal, tmpFloatLocal, tmpFloatLocal[miniBatch * NUM_POINTS * 2],
                            miniBatch * NUM_POINTS * 2);
                        Mul(weightLocal, paramLocal, paramLocal[miniBatch * NUM_POINTS], miniBatch * NUM_POINTS);

                        Sub(xLocal, floatOneLocal, paramLocal, miniBatch * NUM_POINTS);
                        Sub(weightLocal[miniBatch * NUM_POINTS], paramLocal, weightLocal, miniBatch * NUM_POINTS);
                        Sub(weightLocal[miniBatch * NUM_POINTS * 2], paramLocal[miniBatch * NUM_POINTS], weightLocal,
                            miniBatch * NUM_POINTS);
                        Sub(weightLocal[miniBatch * NUM_POINTS * 3], xLocal, weightLocal[miniBatch * NUM_POINTS * 2],
                            miniBatch * NUM_POINTS);

                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_2);
                        Mul(weightLocal, weightLocal, attentionWeightLocal, miniBatch * NUM_POINTS);
                        Mul(weightLocal[miniBatch * NUM_POINTS], weightLocal[miniBatch * NUM_POINTS],
                            attentionWeightLocal, miniBatch * NUM_POINTS);
                        Mul(weightLocal[miniBatch * NUM_POINTS * 2], weightLocal[miniBatch * NUM_POINTS * 2],
                            attentionWeightLocal, miniBatch * NUM_POINTS);
                        Mul(weightLocal[miniBatch * NUM_POINTS * 3], weightLocal[miniBatch * NUM_POINTS * 3],
                            attentionWeightLocal, miniBatch * NUM_POINTS);

                        BroadCast<DTYPE_VALUE, 2, 1>(cornerWeightLocal, weightLocal, dstShape_, srcShape_);

                        for (uint32_t curQuery = 0; curQuery < miniBatch; curQuery++) {
                            query = queryloop * miniBatch + curQuery;
                            dstOffset = moveOffset + query * numHeads * embedDimsOpt;
                            srcOffset = curQuery * NUM_POINTS * embedDimsOpt;

                            Duplicate<DTYPE_VALUE>(valueLocal, DTYPE_VALUE(0), 4 * NUM_POINTS * embedDimsOpt);
                            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                            for (uint32_t point = 0; point < NUM_POINTS; point++) {
                                valueLocalOffset = point * embedDimsOpt;
                                y1 = tmpIntLocal.GetValue(curQuery * NUM_POINTS + miniBatch * NUM_POINTS + point);
                                x1 = tmpIntLocal.GetValue(curQuery * NUM_POINTS + point);

                                x0 = x1 - 1;
                                y0 = y1 - 1;

                                if (isInRange(y0, h)) {
                                    if (0 < x1 && x1 < w) {
                                        DataCopyPad(valueLocal[valueLocalOffset],
                                            valueGm[(valueOffset + y0 * w + x0) * embedDimsOpt], copyParams, padParams);
                                    } else if (isInRange(x0, w)) {
                                        DataCopy(valueLocal[valueLocalOffset],
                                            valueGm[(valueOffset + y0 * w + x0) * embedDimsOpt], embedDimsOpt);
                                    } else if (isInRange(x1, w)) {
                                        DataCopy(valueLocal[valueLocalOffset + NUM_POINTS * embedDimsOptTwice],
                                            valueGm[(valueOffset + y0 * w + x1) * embedDimsOpt], embedDimsOpt);
                                    }
                                }
                                if (isInRange(y1, h)) {
                                    if (0 < x1 && x1 < w) {
                                        DataCopyPad(valueLocal[valueLocalOffset + NUM_POINTS * embedDimsOpt],
                                            valueGm[(valueOffset + y1 * w + x0) * embedDimsOpt], copyParams, padParams);
                                    } else if (isInRange(x0, w)) {
                                        DataCopy(valueLocal[valueLocalOffset + NUM_POINTS * embedDimsOpt],
                                            valueGm[(valueOffset + y1 * w + x0) * embedDimsOpt], embedDimsOpt);
                                    } else if (isInRange(x1, w)) {
                                        DataCopy(valueLocal[valueLocalOffset + NUM_POINTS * embedDimsOptTriple],
                                            valueGm[(valueOffset + y1 * w + x1) * embedDimsOpt], embedDimsOpt);
                                    }
                                }
                            }
                            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);

                            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);
                            Mul(valueLocal, valueLocal, cornerWeightLocal[curQuery * NUM_POINTS * embedDimsOpt],
                                NUM_POINTS * embedDimsOpt);
                            Mul(valueLocal[NUM_POINTS * embedDimsOpt], valueLocal[NUM_POINTS * embedDimsOpt],
                                cornerWeightLocal[(miniBatch + curQuery) * NUM_POINTS * embedDimsOpt],
                                NUM_POINTS * embedDimsOpt);
                            Mul(valueLocal[NUM_POINTS * embedDimsOpt * 2], valueLocal[NUM_POINTS * embedDimsOpt * 2],
                                cornerWeightLocal[(miniBatch * 2 + curQuery) * NUM_POINTS * embedDimsOpt],
                                NUM_POINTS * embedDimsOpt);
                            Mul(valueLocal[NUM_POINTS * embedDimsOpt * 3], valueLocal[NUM_POINTS * embedDimsOpt * 3],
                                cornerWeightLocal[(miniBatch * 3 + curQuery) * NUM_POINTS * embedDimsOpt],
                                NUM_POINTS * embedDimsOpt);

                            Add(tmpResLocal, valueLocal, valueLocal[NUM_POINTS * embedDimsOpt * 2],
                                NUM_POINTS * embedDimsOpt * 2);
                            Add(tmpResLocal2[srcOffset], tmpResLocal, tmpResLocal[NUM_POINTS * embedDimsOpt],
                                NUM_POINTS * embedDimsOpt);
                            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                            for (uint32_t point = 0; point < NUM_POINTS; point++) {
                                DataCopy(
                                    outputGm[dstOffset], tmpResLocal2[srcOffset + point * embedDimsOpt], embedDimsOpt);
                            }
                        }
                    }
                }
            }
        }
        SetAtomicNone();
    }

    __aicore__ inline void ComputePointCommon()
    {
        event_t eventIdVToMte3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        event_t eventIdVToMte2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE2>());
        event_t eventIdMte2ToV_0 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        event_t eventIdMte2ToV_1 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        event_t eventIdMte2ToV_2 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());

        LocalTensor<DTYPE_VALUE> locationLocal = locationQueue.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> attentionWeightLocal = attentionWeightQueue.Get<DTYPE_VALUE>();

        LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> shapesLocal = shapeQueue.Get<DTYPE_VALUE_SPATIAL_SHAPES>();
        LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> offsetLocal = offsetQueue.Get<DTYPE_VALUE_SPATIAL_SHAPES>();
        LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> tmpIntLocal = tmpIntUb.Get<DTYPE_VALUE_SPATIAL_SHAPES>();
        LocalTensor<DTYPE_VALUE> tmpFloatLocal = tmpFloatUb.Get<DTYPE_VALUE>();

        DataCopy(shapesLocal, valueSpatialShapesGm, AlignUp(numLevels * 2, dataAlign));
        DataCopy(offsetLocal, valueLevelStartIndexGm, numLevelsAlign);

        LocalTensor<DTYPE_VALUE> valueLocal = valueUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> cornerWeightLocal = cornerWeightUb.Get<DTYPE_VALUE>();

        LocalTensor<DTYPE_VALUE> floatOneLocal = floatOneUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> halfLocal = halfUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> locLocal = locUb.Get<DTYPE_VALUE>();

        Duplicate<DTYPE_VALUE>(floatOneLocal, (DTYPE_VALUE)1, miniBatch * numPoints * 2);
        Duplicate<DTYPE_VALUE>(halfLocal, (DTYPE_VALUE)0.5, miniBatch * numPoints * 2);

        LocalTensor<DTYPE_VALUE> weightLocal = weightQueue.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> xLocal = tmpXUb.Get<DTYPE_VALUE>();

        LocalTensor<DTYPE_VALUE> tmpResLocal = tmpResUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> tmpResLocal2 = tmpResUb2.Get<DTYPE_VALUE>();

        LocalTensor<DTYPE_VALUE> paramLocal = tmpParamUb.Get<DTYPE_VALUE>();

        uint32_t srcShape_[2] = {4 * miniBatch * numPoints, 1};
        uint32_t dstShape_[2] = {4 * miniBatch * numPoints, embedDims};

        SetAtomicAdd<DTYPE_VALUE>();

        for (uint32_t level = 0; level < numLevels; level++) {
            h = shapesLocal.GetValue(level * 2);
            w = shapesLocal.GetValue(level * 2 + 1);
            dataOffset = offsetLocal.GetValue(level);

            Duplicate<DTYPE_VALUE>(locLocal, (DTYPE_VALUE)w, miniBatch * numPoints);
            Duplicate<DTYPE_VALUE>(locLocal[miniBatch * numPoints], (DTYPE_VALUE)h, miniBatch * numPoints);
            for (uint32_t batch = 0; batch < batchSize; batch++) {
                for (uint32_t head = 0; head < numHeads; head++) {
                    baseOffset = batch * numHeads * numLevels * numQueries * numPoints +
                                 head * numLevels * numQueries * numPoints + level * numQueries * numPoints;
                    valueOffset = batch * numHeads * numKeys + head * numKeys + dataOffset;
                    moveOffset = (batch * numQueries * numHeads + head) * embedDims;
                    for (uint32_t queryloop = startLoop; queryloop < endLoop; queryloop++) {
                        queryBase = queryloop * miniBatch;
                        weightOffset = baseOffset + queryBase * numPoints;
                        locationOffset = baseOffset * 2 + queryBase * numPoints;

                        DataCopy(locationLocal, locationGm[locationOffset], miniBatch * numPoints);
                        DataCopy(locationLocal[miniBatch * numPoints],
                            locationGm[locationOffset + numQueries * numPoints], miniBatch * numPoints);
                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);
                        DataCopy(attentionWeightLocal, attentionWeightsGm[weightOffset], miniBatch * numPoints);
                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_2);

                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);
                        Mul(locationLocal, locationLocal, locLocal, 2 * miniBatch * numPoints);
                        Add(tmpFloatLocal, locationLocal, halfLocal, 2 * miniBatch * numPoints);
                        Cast(tmpIntLocal, tmpFloatLocal, RoundMode::CAST_FLOOR, 2 * miniBatch * numPoints);

                        Sub(tmpFloatLocal[miniBatch * numPoints * 2], tmpFloatLocal, floatOneLocal,
                            miniBatch * numPoints * 2);
                        Cast(tmpFloatLocal, tmpIntLocal, RoundMode::CAST_NONE, miniBatch * numPoints * 2);

                        Sub(paramLocal, tmpFloatLocal, tmpFloatLocal[miniBatch * numPoints * 2],
                            miniBatch * numPoints * 2);
                        Mul(weightLocal, paramLocal, paramLocal[miniBatch * numPoints], miniBatch * numPoints);

                        Sub(xLocal, floatOneLocal, paramLocal, miniBatch * numPoints);
                        Sub(weightLocal[miniBatch * numPoints], paramLocal, weightLocal, miniBatch * numPoints);
                        Sub(weightLocal[miniBatch * numPoints * 2], paramLocal[miniBatch * numPoints], weightLocal,
                            miniBatch * numPoints);
                        Sub(weightLocal[miniBatch * numPoints * 3], xLocal, weightLocal[miniBatch * numPoints * 2],
                            miniBatch * numPoints);

                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_2);
                        Mul(weightLocal, weightLocal, attentionWeightLocal, miniBatch * numPoints);
                        Mul(weightLocal[miniBatch * numPoints], weightLocal[miniBatch * numPoints],
                            attentionWeightLocal, miniBatch * numPoints);
                        Mul(weightLocal[miniBatch * numPoints * 2], weightLocal[miniBatch * numPoints * 2],
                            attentionWeightLocal, miniBatch * numPoints);
                        Mul(weightLocal[miniBatch * numPoints * 3], weightLocal[miniBatch * numPoints * 3],
                            attentionWeightLocal, miniBatch * numPoints);

                        BroadCast<DTYPE_VALUE, 2, 1>(cornerWeightLocal, weightLocal, dstShape_, srcShape_);

                        for (uint32_t curQuery = 0; curQuery < miniBatch; curQuery++) {
                            query = queryloop * miniBatch + curQuery;
                            dstOffset = moveOffset + query * numHeads * embedDims;
                            srcOffset = curQuery * numPoints * embedDims;

                            Duplicate<DTYPE_VALUE>(valueLocal, DTYPE_VALUE(0), 4 * numPoints * embedDims);
                            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                            for (uint32_t point = 0; point < numPoints; point++) {
                                y1 = tmpIntLocal.GetValue(curQuery * numPoints + miniBatch * numPoints + point);
                                x1 = tmpIntLocal.GetValue(curQuery * numPoints + point);

                                x0 = x1 - 1;
                                y0 = y1 - 1;

                                if (isInRange(y0, h)) {
                                    if (isInRange(x0, w)) {
                                        DataCopy(valueLocal[point * embedDims],
                                            valueGm[(valueOffset + y0 * w + x0) * embedDims], embedDims);
                                    }
                                    if (isInRange(x1, w)) {
                                        DataCopy(valueLocal[point * embedDims + numPoints * embedDims * 2],
                                            valueGm[(valueOffset + y0 * w + x1) * embedDims], embedDims);
                                    }
                                }
                                if (isInRange(y1, h)) {
                                    if (isInRange(x0, w)) {
                                        DataCopy(valueLocal[(point + numPoints) * embedDims],
                                            valueGm[(valueOffset + y1 * w + x0) * embedDims], embedDims);
                                    }
                                    if (isInRange(x1, w)) {
                                        DataCopy(valueLocal[point * embedDims + numPoints * embedDims * 3],
                                            valueGm[(valueOffset + y1 * w + x1) * embedDims], embedDims);
                                    }
                                }
                            }
                            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);
                            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);

                            Mul(valueLocal, valueLocal, cornerWeightLocal[curQuery * numPoints * embedDims],
                                numPoints * embedDims);
                            Mul(valueLocal[numPoints * embedDims], valueLocal[numPoints * embedDims],
                                cornerWeightLocal[(miniBatch + curQuery) * numPoints * embedDims],
                                numPoints * embedDims);
                            Mul(valueLocal[numPoints * embedDims * 2], valueLocal[numPoints * embedDims * 2],
                                cornerWeightLocal[(miniBatch * 2 + curQuery) * numPoints * embedDims],
                                numPoints * embedDims);
                            Mul(valueLocal[numPoints * embedDims * 3], valueLocal[numPoints * embedDims * 3],
                                cornerWeightLocal[(miniBatch * 3 + curQuery) * numPoints * embedDims],
                                numPoints * embedDims);
                            pipe_barrier(PIPE_ALL);
                            Add(tmpResLocal, valueLocal, valueLocal[numPoints * embedDims * 2],
                                numPoints * embedDims * 2);
                            Add(tmpResLocal2[srcOffset], tmpResLocal, tmpResLocal[numPoints * embedDims],
                                numPoints * embedDims);
                            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                            for (uint32_t point = 0; point < numPoints; point++) {
                                DataCopy(outputGm[dstOffset], tmpResLocal2[srcOffset + point * embedDims], embedDims);
                            }
                        }
                    }
                }
            }
        }
        SetAtomicNone();
    }

private:
    TPipe* pipe;
    GlobalTensor<DTYPE_VALUE> valueGm, locationGm, attentionWeightsGm, outputGm;
    GlobalTensor<DTYPE_VALUE_SPATIAL_SHAPES> valueSpatialShapesGm, valueLevelStartIndexGm;

    TBuf<TPosition::VECCALC> locationQueue, attentionWeightQueue, shapeQueue, offsetQueue;
    TBuf<TPosition::VECCALC> outputQueue;

    TBuf<TPosition::VECCALC> tmpResUb, tmpResUb2, tmpResUb3, tmpXUb, tmpParamUb, tmpIntUb, tmpFloatUb;
    TBuf<TPosition::VECCALC> floatOneUb, weightQueue;
    TBuf<TPosition::VECCALC> valueUb, tmpValueUb, cornerWeightUb, halfUb, locUb;

    uint32_t batchSize;
    uint32_t numKeys;
    uint32_t numHeads;
    uint32_t embedDims;
    uint32_t tailNum;

    uint32_t numLevels;
    uint32_t numQueries;
    uint32_t numPoints;
    uint32_t coreNum;

    uint32_t numPointsAlign;
    uint32_t numLevelsAlign;
    uint32_t embedDimsOpt = 32;
    uint32_t embedDimsOptHalf = 16;
    uint32_t embedDimsOptTwice = 64;
    uint32_t embedDimsOptTriple = 96;
    uint32_t numPointsAlignOpt = 8;
    uint32_t numPointsFourOpt = 4;
    uint32_t numPointsAlignOptTwice = 16;
    uint32_t numPointsAlignOptTriple = 24;
    uint32_t batch;
    uint32_t query;
    uint32_t head;

    uint32_t taskNum;
    uint32_t taskNumPerCore;
    uint32_t curBlockIdx;
    uint32_t startLoop;
    uint32_t endLoop;
    uint32_t dataAlign;
    uint32_t blockNum = 32;
    uint32_t innerClean = 0;
    uint32_t miniBatch = 16;

    DTYPE_VALUE_SPATIAL_SHAPES tmpOffset1, tmpOffset2, baseOffset, valueOffset, weightOffset, dataOffset,
        locationOffset, moveOffset, batchOffset, dstOffset, srcOffset, valueLocalOffset, queryBase;
    DTYPE_VALUE tmp1, tmp2, leftTopWeight, rightTopWeight, leftBottomWeight, rightBottomWeight, attnWeight;
    DTYPE_VALUE_SPATIAL_SHAPES h, w, x0, y0, x1, y1;
};

extern "C" __global__ __aicore__ void multi_scale_deformable_attn(GM_ADDR value, GM_ADDR value_spatial_shapes,
    GM_ADDR value_level_start_index, GM_ADDR sampling_locations, GM_ADDR attention_weights, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    KernelMultiScaleDeformableAttn op;
    op.Init(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, output,
        &tiling_data, &pipe);
    op.Process();
}
