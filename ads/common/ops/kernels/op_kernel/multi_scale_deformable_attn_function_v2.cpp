/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * This sample is a very basic sample that implements vector add on Ascend plaform.
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;

class KernelMultiScaleDeformableAttnFunctionV2
{
public:
    __aicore__ inline KernelMultiScaleDeformableAttnFunctionV2() {}
    __aicore__ inline void Init(GM_ADDR value,
                                GM_ADDR value_spatial_shapes,
                                GM_ADDR value_level_start_index,
                                GM_ADDR sampling_locations,
                                GM_ADDR attention_weights,
                                GM_ADDR output,
                                const MultiScaleDeformableAttnFunctionV2TilingData *tiling_data,
                                TPipe *tmpPipe)
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
        if (endOffset > taskNum)
        {
            endOffset = taskNum;
        }

        valueGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(value), batchSize * numKeys * numHeads * embedDims);
        locationGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(sampling_locations), batchSize * numQueries * numHeads * numLevels * numPoints * 2);
        attentionWeightsGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(attention_weights), batchSize * numQueries * numHeads * numLevels * numPoints);
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(output), batchSize * numQueries * numHeads * embedDims);

        valueSpatialShapesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE_SPATIAL_SHAPES *>(value_spatial_shapes), numLevels * 2);
        valueLevelStartIndexGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE_SPATIAL_SHAPES *>(value_level_start_index), numLevels);

        pipe->InitBuffer(shapeQueue, BUFFER_NUM, AlignUp(numLevels * 2, dataAlign) * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(offsetQueue, BUFFER_NUM, numLevelsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(locationQueue, BUFFER_NUM, AlignUp(numHeads * numLevels * numPoints * 2, dataAlign) * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(attentionWeightsUb, BUFFER_NUM, AlignUp(numHeads * numLevels * numPoints, dataAlign) * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(outputQueue, BUFFER_NUM, embedDims * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(emptyUb, BUFFER_NUM, embedDims * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(intOneUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE_SPATIAL_SHAPES));
        pipe->InitBuffer(floatOneUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(tmpXUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpYUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpParam0Ub, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpParam1Ub, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(tmpIntUb, BUFFER_NUM, 4 * numPointsAlign * sizeof(DTYPE_VALUE_SPATIAL_SHAPES));
        pipe->InitBuffer(tmpFloatUb, BUFFER_NUM, 4 * numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(weightQueue, BUFFER_NUM, 4 * numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(valueUb, BUFFER_NUM, batchOffset * 4 * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpResUb, BUFFER_NUM, batchOffset * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpResUb2, BUFFER_NUM, batchOffset * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpResUb3, BUFFER_NUM, numHeads * batchOffset * sizeof(DTYPE_VALUE));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t taskIdx = startOffset; taskIdx < endOffset; taskIdx++)
        {
            Compute(taskIdx);
        }
    }

private:
    __aicore__ inline bool isInRange(DTYPE_VALUE_SPATIAL_SHAPES x, DTYPE_VALUE_SPATIAL_SHAPES upper)
    {
        return 0 <= x && x < upper;
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

        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());

        for (uint32_t batch = 0; batch < batchSize; batch++)
        {
            LocalTensor<DTYPE_VALUE> emptyUbLocal = emptyUb.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE> weightLocal = weightQueue.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE> xLocal = tmpXUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> yLocal = tmpYUb.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE> tmpResLocal = tmpResUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> tmpResLocal2 = tmpResUb2.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> tmpResLocal3 = tmpResUb3.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE> param0Local = tmpParam0Ub.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> param1Local = tmpParam1Ub.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> tmpIntLocal = tmpIntUb.Get<DTYPE_VALUE_SPATIAL_SHAPES>();
            LocalTensor<DTYPE_VALUE> tmpFloatLocal = tmpFloatUb.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> intOneLocal = intOneUb.Get<DTYPE_VALUE_SPATIAL_SHAPES>();
            LocalTensor<DTYPE_VALUE> floatOneLocal = floatOneUb.Get<DTYPE_VALUE>();

            Duplicate<DTYPE_VALUE_SPATIAL_SHAPES>(intOneLocal, (DTYPE_VALUE_SPATIAL_SHAPES)1, numPointsAlign);
            Duplicate<DTYPE_VALUE>(floatOneLocal, (DTYPE_VALUE)1, numPointsAlign);

            Duplicate<DTYPE_VALUE>(emptyUbLocal, DTYPE_VALUE(0), embedDims);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            moveOffset = (batch * numQueries + query) * numHeads * embedDims;
            dataOffset = (batch * numQueries + query) * numHeads * numLevels * numPoints;
            DataCopy(locationLocal, locationGm[dataOffset * 2], AlignUp(numHeads * numLevels * numPoints * 2, dataAlign));

            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            for (uint32_t head = 0; head < numHeads; head++)
            {
                DataCopy(outputGm[moveOffset + head * embedDims], emptyUbLocal, embedDims);
            }
            pipe_barrier(PIPE_ALL);

            for (uint32_t level = 0; level < numLevels; level++)
            {
                h = shapesLocal.GetValue(level * 2);
                w = shapesLocal.GetValue(level * 2 + 1);
                oriOffset = (batch * numHeads * numKeys + offsetLocal.GetValue(level)) * embedDims;

                SetAtomicAdd<DTYPE_VALUE>();
                for (uint32_t head = 0; head < numHeads; head++)
                {
                    srcOffset = head * batchOffset;
                    dstOffset = moveOffset + head * embedDims;

                    weightOffset = (head * numLevels + level) * numPoints;
                    DataCopy(attentionWeightLocal, attentionWeightsGm[dataOffset + weightOffset], AlignUp(numPoints, dataAlign));
                    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                    locationOffset = weightOffset * 2;
                    valueOffset = oriOffset + (head * numKeys) * embedDims;
                    for (uint32_t point = 0; point < numPoints; point++)
                    {
                        tmp1 = locationLocal.GetValue(locationOffset + point * 2) * (DTYPE_VALUE)w;
                        tmp2 = locationLocal.GetValue(locationOffset + point * 2 + 1) * (DTYPE_VALUE)h;

                        tmpFloatLocal.SetValue(point, tmp1 + (DTYPE_VALUE)0.5);
                        tmpFloatLocal.SetValue(point + numPointsAlign, tmp1 - (DTYPE_VALUE)0.5 + (DTYPE_VALUE)1e-5);

                        tmpFloatLocal.SetValue(point + numPointsAlign * 2, tmp2 + (DTYPE_VALUE)0.5);
                        tmpFloatLocal.SetValue(point + numPointsAlign * 3, tmp2 - (DTYPE_VALUE)0.5 + (DTYPE_VALUE)1e-5);
                    }

                    Cast(tmpIntLocal, tmpFloatLocal, RoundMode::CAST_FLOOR, 4 * numPointsAlign);
                    Cast(xLocal, tmpIntLocal, RoundMode::CAST_NONE, numPointsAlign);
                    Cast(yLocal, tmpIntLocal[numPointsAlign * 2], RoundMode::CAST_NONE, numPointsAlign);

                    Sub(param0Local, xLocal, tmpFloatLocal[numPointsAlign], numPointsAlign);
                    Sub(param1Local, yLocal, tmpFloatLocal[numPointsAlign * 3], numPointsAlign);
                    Mul(weightLocal[numPointsAlign * 3], param0Local, param1Local, numPointsAlign);

                    Sub(xLocal, floatOneLocal, param0Local, numPointsAlign);
                    Sub(weightLocal[numPointsAlign * 2], param0Local, weightLocal[numPointsAlign * 3], numPointsAlign);
                    Sub(weightLocal[numPointsAlign], param1Local, weightLocal[numPointsAlign * 3], numPointsAlign);
                    Sub(weightLocal, xLocal, weightLocal[numPointsAlign], numPointsAlign);

                    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                    Mul(weightLocal, weightLocal, attentionWeightLocal, numPointsAlign, 4, {1, 1, 1, uint8_t(numPointsAlign / dataAlign), uint8_t(numPointsAlign / dataAlign), 0});

                    Duplicate<DTYPE_VALUE>(valueLocal, DTYPE_VALUE(0), 4 * batchOffset);

                    for (uint32_t point = 0; point < numPoints; point++)
                    {
                        x0 = tmpIntLocal.GetValue(point);
                        x1 = tmpIntLocal.GetValue(point + numPointsAlign);
                        y0 = tmpIntLocal.GetValue(point + numPointsAlign * 2);
                        y1 = tmpIntLocal.GetValue(point + numPointsAlign * 3);

                        if (isInRange(x0, w))
                        {
                            if (isInRange(y0, h))
                            {
                                DataCopy(valueLocal[point * embedDims], valueGm[valueOffset + (y0 * w + x0) * embedDims], embedDims);
                            }
                            if (isInRange(y1, h))
                            {
                                DataCopy(valueLocal[batchOffset + point * embedDims], valueGm[valueOffset + (y1 * w + x0) * embedDims], embedDims);
                            }
                            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                            leftTopWeight = weightLocal.GetValue(point);
                            leftBottomWeight = weightLocal.GetValue(numPointsAlign + point);
                            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                            Muls(valueLocal[point * embedDims], valueLocal[point * embedDims], leftTopWeight, embedDims);
                            Muls(valueLocal[batchOffset + point * embedDims], valueLocal[batchOffset + point * embedDims], leftBottomWeight, embedDims);
                        }
                        if (isInRange(x1, w))
                        {
                            if (isInRange(y0, h))
                            {
                                DataCopy(valueLocal[batchOffset * 2 + point * embedDims], valueGm[valueOffset + (y0 * w + x1) * embedDims], embedDims);
                            }
                            if (isInRange(y1, h))
                            {
                                DataCopy(valueLocal[batchOffset * 3 + point * embedDims], valueGm[valueOffset + (y1 * w + x1) * embedDims], embedDims);
                            }
                            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                            rightTopWeiight = weightLocal.GetValue(numPointsAlign * 2 + point);
                            rightBottomWeight = weightLocal.GetValue(numPointsAlign * 3 + point);
                            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                            Muls(valueLocal[batchOffset * 2 + point * embedDims], valueLocal[batchOffset * 2 + point * embedDims], rightTopWeiight, embedDims);
                            Muls(valueLocal[batchOffset * 3 + point * embedDims], valueLocal[batchOffset * 3 + point * embedDims], rightBottomWeight, embedDims);
                        }
                    }

                    if (embedDims != 32) {
                        pipe_barrier(PIPE_ALL);
                    }

                    Add(tmpResLocal, valueLocal, valueLocal[batchOffset], batchOffset);
                    Add(tmpResLocal2, valueLocal[batchOffset * 2], valueLocal[batchOffset * 3], batchOffset);
                    Add(tmpResLocal3[srcOffset], tmpResLocal, tmpResLocal2, batchOffset);

                    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                    for (uint32_t point = 0; point < numPoints; point++)
                    {
                        DataCopy(outputGm[dstOffset], tmpResLocal3[srcOffset + point * embedDims], embedDims);
                    }
                }
                SetAtomicNone();
            }
        }
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
    }

private:
    TPipe *pipe;
    GlobalTensor<DTYPE_VALUE> valueGm, locationGm, attentionWeightsGm, outputGm;
    GlobalTensor<DTYPE_VALUE_SPATIAL_SHAPES> valueSpatialShapesGm, valueLevelStartIndexGm;

    TBuf<TPosition::VECCALC> locationQueue, attentionWeightsUb, shapeQueue, offsetQueue;
    TBuf<TPosition::VECCALC> outputQueue;

    TBuf<TPosition::VECCALC> tmpResUb, tmpResUb2, tmpResUb3, tmpXUb, tmpYUb, tmpParam0Ub, tmpParam1Ub, tmpIntUb, tmpFloatUb;
    TBuf<TPosition::VECCALC> intOneUb, floatOneUb, weightQueue, emptyUb;
    TBuf<TPosition::VECCALC> valueUb;

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

    DTYPE_VALUE tmp1, tmp2, leftTopWeight, rightTopWeiight, leftBottomWeight, rightBottomWeight, attnWeight;
    DTYPE_VALUE_SPATIAL_SHAPES h, w, x0, y0, x1, y1;
    DTYPE_VALUE_SPATIAL_SHAPES valueOffset, weightOffset, oriOffset, dataOffset, locationOffset, moveOffset, batchOffset, dstOffset, srcOffset, headOffset;
};

extern "C" __global__ __aicore__ void multi_scale_deformable_attn_function_v2(GM_ADDR value,
                                                                              GM_ADDR value_spatial_shapes,
                                                                              GM_ADDR value_level_start_index,
                                                                              GM_ADDR sampling_locations,
                                                                              GM_ADDR attention_weights,
                                                                              GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    KernelMultiScaleDeformableAttnFunctionV2 op;
    op.Init(value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights, output, &tiling_data, &pipe);
    op.Process();
}