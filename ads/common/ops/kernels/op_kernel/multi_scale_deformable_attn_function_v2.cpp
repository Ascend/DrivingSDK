/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * This sample is a very basic sample that implements vector add on Ascend plaform.
 */
#include "kernel_operator.h"
using namespace AscendC;

class KernelMultiScaleDeformableAttnFunctionV2 {
public:
    __aicore__ inline KernelMultiScaleDeformableAttnFunctionV2() {}
    __aicore__ inline void Init(GM_ADDR value, GM_ADDR valueSpatialShapes, GM_ADDR valuLevelStartIndex,
        GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output,
        const MultiScaleDeformableAttnFunctionV2TilingData* tiling_data, TPipe* tmpPipe)
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
        taskNumPerCore = DivCeil(taskNum, coreNum);

        numPointsAlign = AlignUp(numPoints, dataAlign);
        numLevelsAlign = AlignUp(numLevels, dataAlign);

        batchOffset = numPoints * embedDims;

        curBlockIdx = GetBlockIdx();
        startOffset = curBlockIdx * taskNumPerCore;
        endOffset = (curBlockIdx + 1) * taskNumPerCore;
        if (endOffset > taskNum) {
            endOffset = taskNum;
        }

        valueGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_VALUE*>(value), batchSize * numKeys * numHeads * embedDims);
        locationGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE*>(samplingLocations),
            batchSize * numQueries * numHeads * numLevels * numPoints * 2);
        attentionWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE*>(attentionWeights),
            batchSize * numQueries * numHeads * numLevels * numPoints);
        outputGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_VALUE*>(output), batchSize * numQueries * numHeads * embedDims);

        valueSpatialShapesGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_VALUE_SPATIAL_SHAPES*>(valueSpatialShapes), numLevels * 2);
        valueLevelStartIndexGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_VALUE_SPATIAL_SHAPES*>(valuLevelStartIndex), numLevels);

        pipe->InitBuffer(shapeQueue, AlignUp(numLevels * 2, dataAlign) * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(offsetQueue, numLevelsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(locationQueue, AlignUp(numHeads * numLevels * numPoints * 2, dataAlign) * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(
            attentionWeightsUb, AlignUp(numHeads * numLevels * numPoints, dataAlign) * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(outputQueue, embedDims * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(emptyUb, embedDims * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(intOneUb, numPointsAlign * sizeof(DTYPE_VALUE_SPATIAL_SHAPES));
        pipe->InitBuffer(floatOneUb, numPointsAlign * 2 * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(tmpXUb, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpYUb, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpParamUb, numPointsAlign * 2 * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(tmpIntUb, 4 * numPointsAlign * sizeof(DTYPE_VALUE_SPATIAL_SHAPES));
        pipe->InitBuffer(tmpFloatUb, 4 * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpUb, 2 * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(halfUb, 2 * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(locUb, 2 * numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(weightQueue, 4 * numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(valueUb, batchOffset * 8 * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(cornerWeightUb, batchOffset * 8 * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(tmpResUb, batchOffset * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpResUb2, batchOffset * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpResUb3, numHeads * batchOffset * sizeof(DTYPE_VALUE));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t taskIdx = startOffset; taskIdx < endOffset; taskIdx++) {
            Compute(taskIdx);
        }
    }

private:
    __aicore__ inline bool isInRange(DTYPE_VALUE_SPATIAL_SHAPES x, DTYPE_VALUE_SPATIAL_SHAPES upper)
    {
        return -1 < x && x < upper;
    }

    __aicore__ inline void Compute(uint32_t query)
    {
        LocalTensor<DTYPE_VALUE> locationLocal = locationQueue.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> attentionWeightLocal = attentionWeightsUb.Get<DTYPE_VALUE>();

        LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> shapesLocal = shapeQueue.Get<DTYPE_VALUE_SPATIAL_SHAPES>();
        LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> offsetLocal = offsetQueue.Get<DTYPE_VALUE_SPATIAL_SHAPES>();

        DataCopy(shapesLocal, valueSpatialShapesGm, AlignUp(numLevels * 2, dataAlign));
        DataCopy(offsetLocal, valueLevelStartIndexGm, numLevelsAlign);

        LocalTensor<DTYPE_VALUE> valueLocal = valueUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> cornerWeightLocal = cornerWeightUb.Get<DTYPE_VALUE>();

        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        event_t eventIdMte2ToV_0 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        event_t eventIdMte2ToV_1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        event_t eventIdMte2ToV_2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());

        LocalTensor<DTYPE_VALUE> emptyUbLocal = emptyUb.Get<DTYPE_VALUE>();

        LocalTensor<DTYPE_VALUE> tmpLocal = tmpUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> intOneLocal = intOneUb.Get<DTYPE_VALUE_SPATIAL_SHAPES>();
        LocalTensor<DTYPE_VALUE> floatOneLocal = floatOneUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> halfLocal = halfUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> locLocal = locUb.Get<DTYPE_VALUE>();

        Duplicate<DTYPE_VALUE>(emptyUbLocal, DTYPE_VALUE(0), embedDims);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        Duplicate<DTYPE_VALUE_SPATIAL_SHAPES>(intOneLocal, (DTYPE_VALUE_SPATIAL_SHAPES)1, numPointsAlign);
        Duplicate<DTYPE_VALUE>(floatOneLocal, (DTYPE_VALUE)1, numPointsAlign * 2);
        Duplicate<DTYPE_VALUE>(halfLocal, (DTYPE_VALUE)0.5, numPointsAlign * 2);

        for (uint32_t batch = 0; batch < batchSize; batch++) {
            LocalTensor<DTYPE_VALUE> weightLocal = weightQueue.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> xLocal = tmpXUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> yLocal = tmpYUb.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE> tmpResLocal = tmpResUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> tmpResLocal2 = tmpResUb2.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> tmpResLocal3 = tmpResUb3.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE> paramLocal = tmpParamUb.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> tmpIntLocal = tmpIntUb.Get<DTYPE_VALUE_SPATIAL_SHAPES>();
            LocalTensor<DTYPE_VALUE> tmpFloatLocal = tmpFloatUb.Get<DTYPE_VALUE>();

            baseOffset = batch * numHeads * numKeys;
            moveOffset = (batch * numQueries + query) * numHeads * embedDims;
            dataOffset = (batch * numQueries + query) * numHeads * numLevels * numPoints;

            for (uint32_t head = 0; head < numHeads; head++) {
                DataCopy(outputGm[moveOffset + head * embedDims], emptyUbLocal, embedDims);
            }
            pipe_barrier(PIPE_ALL);

            for (uint32_t level = 0; level < numLevels; level++) {
                h = shapesLocal.GetValue(level * 2);
                w = shapesLocal.GetValue(level * 2 + 1);
                oriOffset = baseOffset + offsetLocal.GetValue(level);

                Duplicate<DTYPE_VALUE>(locLocal, (DTYPE_VALUE)w, numPointsAlign);
                Duplicate<DTYPE_VALUE>(locLocal[numPointsAlign], (DTYPE_VALUE)h, numPointsAlign);

                SetAtomicAdd<DTYPE_VALUE>();
                weightOffset = dataOffset + level * numPoints;

                DataCopy(tmpLocal, locationGm[weightOffset * 2], numPointsAlign);
                DataCopy(tmpLocal[numPointsAlign], locationGm[weightOffset * 2 + numPoints], numPointsAlign);
                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_2);

                DataCopy(attentionWeightLocal, attentionWeightsGm[weightOffset], numPointsAlign);
                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);

                for (uint32_t head = 0; head < numHeads; head++) {
                    Duplicate<DTYPE_VALUE>(valueLocal[4 * batchOffset], DTYPE_VALUE(0), 4 * batchOffset);
                    srcOffset = head * batchOffset;
                    dstOffset = moveOffset + head * embedDims;
                    valueOffset = (oriOffset + head * numKeys) * embedDims;

                    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_2);
                    Mul(tmpLocal, tmpLocal, locLocal, 2 * numPointsAlign);
                    Add(tmpFloatLocal, tmpLocal, halfLocal, 2 * numPointsAlign);
                    Cast(tmpIntLocal, tmpFloatLocal, RoundMode::CAST_FLOOR, 2 * numPointsAlign);

                    for (uint32_t point = 0; point < numPoints; point++) {
                        y1 = tmpIntLocal.GetValue(point + numPointsAlign);
                        x1 = tmpIntLocal.GetValue(point);

                        x0 = x1 - 1;
                        y0 = y1 - 1;

                        if (isInRange(y0, h)) {
                            if (0 < x1 && x1 < w) {
                                DataCopy(valueLocal[batchOffset * 4 + point * embedDims * 2],
                                    valueGm[valueOffset + (y0 * w + x0) * embedDims], 2 * embedDims);
                            } else if (isInRange(x0, w)) {
                                DataCopy(valueLocal[batchOffset * 4 + point * embedDims * 2],
                                    valueGm[valueOffset + (y0 * w + x0) * embedDims], embedDims);
                            } else if (isInRange(x1, w)) {
                                DataCopy(valueLocal[batchOffset * 4 + point * embedDims * 2 + embedDims],
                                    valueGm[valueOffset + (y0 * w + x1) * embedDims], embedDims);
                            }
                        }
                        if (isInRange(y1, h)) {
                            if (0 < x1 && x1 < w) {
                                DataCopy(valueLocal[batchOffset * 6 + point * embedDims * 2],
                                    valueGm[valueOffset + (y1 * w + x0) * embedDims], 2 * embedDims);
                            } else if (isInRange(x0, w)) {
                                DataCopy(valueLocal[batchOffset * 6 + point * embedDims * 2],
                                    valueGm[valueOffset + (y1 * w + x0) * embedDims], embedDims);
                            } else if (isInRange(x1, w)) {
                                DataCopy(valueLocal[batchOffset * 6 + point * embedDims * 2 + embedDims],
                                    valueGm[valueOffset + (y1 * w + x1) * embedDims], embedDims);
                            }
                        }
                    }
                    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);

                    Sub(tmpFloatLocal[numPointsAlign * 2], tmpFloatLocal, floatOneLocal, 2 * numPointsAlign);
                    Cast(tmpFloatLocal, tmpIntLocal, RoundMode::CAST_NONE, 2 * numPointsAlign);

                    weightOffset = dataOffset + ((head + 1) * numLevels + level) * numPoints;
                    if (head < numHeads - 1) {
                        locationOffset = weightOffset * 2;
                        DataCopy(tmpLocal, locationGm[locationOffset], numPointsAlign);
                        DataCopy(tmpLocal[numPointsAlign], locationGm[locationOffset + numPoints], numPointsAlign);
                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_2);
                    }

                    Sub(paramLocal, tmpFloatLocal, tmpFloatLocal[numPointsAlign * 2], 2 * numPointsAlign);
                    Mul(weightLocal[numPointsAlign * 3], paramLocal, paramLocal[numPointsAlign], numPointsAlign);

                    Sub(xLocal, floatOneLocal, paramLocal, numPointsAlign);
                    Sub(weightLocal[numPointsAlign * 2], paramLocal, weightLocal[numPointsAlign * 3], numPointsAlign);
                    Sub(weightLocal[numPointsAlign], paramLocal[numPointsAlign], weightLocal[numPointsAlign * 3],
                        numPointsAlign);
                    Sub(weightLocal, xLocal, weightLocal[numPointsAlign], numPointsAlign);

                    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);
                    Mul(weightLocal, weightLocal, attentionWeightLocal, numPointsAlign, 4,
                        {1, 1, 1, uint8_t(numPointsAlign / dataAlign), uint8_t(numPointsAlign / dataAlign), 0});
                    if (head < numHeads - 1) {
                        DataCopy(attentionWeightLocal, attentionWeightsGm[weightOffset], numPointsAlign);
                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);
                    }
                    for (uint32_t point = 0; point < numPoints; point++) {
                        tmpOffset1 = 2 * point * embedDims;
                        tmpOffset2 = batchOffset * 2 + tmpOffset1;

                        leftTopWeight = weightLocal.GetValue(numPointsAlign * 3 + point);
                        rightTopWeight = weightLocal.GetValue(numPointsAlign + point);

                        leftBottomWeight = weightLocal.GetValue(numPointsAlign * 2 + point);
                        rightBottomWeight = weightLocal.GetValue(point);

                        Duplicate<DTYPE_VALUE>(cornerWeightLocal[tmpOffset1], leftTopWeight, embedDims);
                        Duplicate<DTYPE_VALUE>(cornerWeightLocal[tmpOffset1 + embedDims], rightTopWeight, embedDims);
                        Duplicate<DTYPE_VALUE>(cornerWeightLocal[tmpOffset2], leftBottomWeight, embedDims);
                        Duplicate<DTYPE_VALUE>(cornerWeightLocal[tmpOffset2 + embedDims], rightBottomWeight, embedDims);
                    }

                    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);
                    Mul(valueLocal, valueLocal[batchOffset * 4], cornerWeightLocal, 4 * batchOffset);

                    if (embedDims != 32) {
                        pipe_barrier(PIPE_ALL);
                    }

                    Add(tmpResLocal, valueLocal, valueLocal[batchOffset], batchOffset);
                    Add(tmpResLocal2, valueLocal[batchOffset * 2], valueLocal[batchOffset * 3], batchOffset);
                    Add(tmpResLocal3[srcOffset], tmpResLocal, tmpResLocal2, batchOffset);

                    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                    for (uint32_t point = 0; point < numPoints; point++) {
                        DataCopy(outputGm[dstOffset], tmpResLocal3[srcOffset + point * embedDims], embedDims);
                    }
                }
                SetAtomicNone();
            }
        }
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV_0);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV_1);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV_2);
    }

private:
    TPipe* pipe;
    GlobalTensor<DTYPE_VALUE> valueGm, locationGm, attentionWeightsGm, outputGm;
    GlobalTensor<DTYPE_VALUE_SPATIAL_SHAPES> valueSpatialShapesGm, valueLevelStartIndexGm;

    TBuf<TPosition::VECCALC> locationQueue, attentionWeightsUb, shapeQueue, offsetQueue;
    TBuf<TPosition::VECCALC> outputQueue;

    TBuf<TPosition::VECCALC> tmpResUb, tmpResUb2, tmpResUb3, tmpXUb, tmpYUb, tmpParamUb, tmpIntUb, tmpFloatUb;
    TBuf<TPosition::VECCALC> intOneUb, floatOneUb, weightQueue, emptyUb, tmpUb;
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

    uint32_t batch;
    uint32_t query;
    uint32_t head;

    uint32_t taskNum;
    uint32_t taskNumPerCore;
    uint32_t curBlockIdx;
    uint32_t startOffset;
    uint32_t endOffset;
    uint32_t dataAlign;
    uint32_t blockNum = 32;

    DTYPE_VALUE_SPATIAL_SHAPES tmpOffset1, tmpOffset2, baseOffset, valueOffset, weightOffset, oriOffset, pointOffset,
        dataOffset, locationOffset, moveOffset, batchOffset, dstOffset, srcOffset, headOffset;
    DTYPE_VALUE tmp1, tmp2, leftTopWeight, rightTopWeight, leftBottomWeight, rightBottomWeight, attnWeight;
    DTYPE_VALUE_SPATIAL_SHAPES h, w, x0, y0, x1, y1;
};

extern "C" __global__ __aicore__ void multi_scale_deformable_attn_function_v2(GM_ADDR value,
    GM_ADDR value_spatial_shapes, GM_ADDR value_level_start_index, GM_ADDR sampling_locations,
    GM_ADDR attention_weights, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    KernelMultiScaleDeformableAttnFunctionV2 op;
    op.Init(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, output,
        &tiling_data, &pipe);
    op.Process();
}
