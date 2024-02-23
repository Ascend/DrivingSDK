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
                                MultiScaleDeformableAttnFunctionV2TilingData *tiling_data,
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

        embedDimsAlign = AlignUp(embedDims, dataAlign);
        numPointsAlign = AlignUp(numPoints, dataAlign);
        numLevelsAlign = AlignUp(numLevels, dataAlign);

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
        pipe->InitBuffer(outputQueue, BUFFER_NUM, embedDimsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(emptyUb, BUFFER_NUM, embedDimsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(tmpUb1, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpUb2, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpUb3, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpUb4, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(intOneUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE_SPATIAL_SHAPES));
        pipe->InitBuffer(floatOneUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(tmpXUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpYUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpParam0Ub, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpParam1Ub, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(tmpIntX0Ub, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE_SPATIAL_SHAPES));
        pipe->InitBuffer(tmpIntY0Ub, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE_SPATIAL_SHAPES));
        pipe->InitBuffer(tmpIntX1Ub, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE_SPATIAL_SHAPES));
        pipe->InitBuffer(tmpIntY1Ub, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE_SPATIAL_SHAPES));

        pipe->InitBuffer(leftTopWieightQueue, BUFFER_NUM, 4 * numPointsAlign * sizeof(DTYPE_VALUE));

        pipe->InitBuffer(valueUb, BUFFER_NUM, numPoints * 4 * embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpResUb, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpResUb2, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(tmpResUb3, BUFFER_NUM, numPoints * embedDimsAlign * sizeof(DTYPE_VALUE));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t taskIdx = startOffset; taskIdx < endOffset; taskIdx++)
        {
            SetAtomicAdd<DTYPE_VALUE>();
            Compute(taskIdx);
            SetAtomicNone();
        }
    }

private:
    __aicore__ inline bool isInRange(DTYPE_VALUE_SPATIAL_SHAPES x, DTYPE_VALUE_SPATIAL_SHAPES upper)
    {
        return 0 <= x && x < upper;
    }

    __aicore__ inline void Compute(uint32_t query)
    {
        LocalTensor<DTYPE_VALUE> locationLocal = locationQueue.AllocTensor<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> attentionWeightLocal = attentionWeightsUb.AllocTensor<DTYPE_VALUE>();

        LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> shapesLocal = shapeQueue.AllocTensor<DTYPE_VALUE_SPATIAL_SHAPES>();
        LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> offsetLocal = offsetQueue.AllocTensor<DTYPE_VALUE_SPATIAL_SHAPES>();

        DataCopy(shapesLocal, valueSpatialShapesGm, AlignUp(numLevels * 2, dataAlign));
        DataCopy(offsetLocal, valueLevelStartIndexGm, numLevelsAlign);

        DataCopyParams copyParams{1, (uint16_t)(embedDims * sizeof(DTYPE_VALUE)), 0, 0};

        LocalTensor<DTYPE_VALUE> valueLocal = valueUb.Get<DTYPE_VALUE>();

        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());

        for (uint32_t batch = 0; batch < batchSize; batch++)
        {
            LocalTensor<DTYPE_VALUE> emptyUbLocal = emptyUb.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE> weightLocal = leftTopWieightQueue.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE> xLocal = tmpXUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> yLocal = tmpYUb.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE> tmpResLocal = tmpResUb.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> tmpResLocal2 = tmpResUb2.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> tmpResLocal3 = tmpResUb3.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE> param0Local = tmpParam0Ub.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> param1Local = tmpParam1Ub.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> x1Local = tmpIntX1Ub.Get<DTYPE_VALUE_SPATIAL_SHAPES>();
            LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> y1Local = tmpIntY1Ub.Get<DTYPE_VALUE_SPATIAL_SHAPES>();

            LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> x0Local = tmpIntX0Ub.Get<DTYPE_VALUE_SPATIAL_SHAPES>();
            LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> y0Local = tmpIntY0Ub.Get<DTYPE_VALUE_SPATIAL_SHAPES>();

            LocalTensor<DTYPE_VALUE> tmpLocal1 = tmpUb1.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> tmpLocal2 = tmpUb2.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> tmpLocal3 = tmpUb3.Get<DTYPE_VALUE>();
            LocalTensor<DTYPE_VALUE> tmpLocal4 = tmpUb4.Get<DTYPE_VALUE>();

            LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> intOneLocal = intOneUb.Get<DTYPE_VALUE_SPATIAL_SHAPES>();
            LocalTensor<DTYPE_VALUE> floatOneLocal = floatOneUb.Get<DTYPE_VALUE>();

            Duplicate<DTYPE_VALUE_SPATIAL_SHAPES>(intOneLocal, (DTYPE_VALUE_SPATIAL_SHAPES)1, numPointsAlign);
            Duplicate<DTYPE_VALUE>(floatOneLocal, (DTYPE_VALUE)1, numPointsAlign);

            Duplicate<DTYPE_VALUE>(emptyUbLocal, DTYPE_VALUE(0), embedDimsAlign);
            moveOffset = batch * numQueries * numHeads * embedDims + query * numHeads * embedDims;

            for (uint32_t head = 0; head < numHeads; head++)
            {
                DataCopyPad(outputGm[moveOffset + head * embedDims], emptyUbLocal, copyParams);
            }

            weightOffset = (batch * numQueries * numHeads * numLevels + query * numHeads * numLevels) * numPoints;

            DataCopy(locationLocal, locationGm[weightOffset * 2], AlignUp(numHeads * numLevels * numPoints * 2, dataAlign));
            DataCopy(attentionWeightLocal, attentionWeightsGm[weightOffset], AlignUp(numHeads * numLevels * numPoints, dataAlign));

            for (uint32_t head = 0; head < numHeads; head++)
            {
                for (uint32_t level = 0; level < numLevels; level++)
                {
                    h = shapesLocal.GetValue(level * 2);
                    w = shapesLocal.GetValue(level * 2 + 1);

                    weightOffset = (head * numLevels + level) * numPoints;
                    locationOffset = weightOffset * 2;
                    for (uint32_t point = 0; point < numPoints; point++)
                    {
                        xLocal.SetValue(point, locationLocal.GetValue(locationOffset + point * 2));
                        yLocal.SetValue(point, locationLocal.GetValue(locationOffset + point * 2 + 1));
                    }

                    Muls(tmpLocal1, xLocal, (DTYPE_VALUE)w, numPointsAlign);
                    Muls(tmpLocal2, yLocal, (DTYPE_VALUE)h, numPointsAlign);

                    Adds(param0Local, tmpLocal1, (DTYPE_VALUE)0.5, numPointsAlign);
                    Adds(param1Local, tmpLocal2, (DTYPE_VALUE)0.5, numPointsAlign);

                    Cast(x1Local, param0Local, RoundMode::CAST_FLOOR, numPointsAlign);
                    Cast(y1Local, param1Local, RoundMode::CAST_FLOOR, numPointsAlign);

                    Adds(tmpLocal3, param0Local, (DTYPE_VALUE)-1, numPointsAlign);
                    Adds(tmpLocal4, param1Local, (DTYPE_VALUE)-1, numPointsAlign);

                    Sub(x0Local, x1Local, intOneLocal, numPointsAlign);
                    Sub(y0Local, y1Local, intOneLocal, numPointsAlign);

                    Cast(xLocal, x0Local, RoundMode::CAST_NONE, numPointsAlign);
                    Cast(yLocal, y0Local, RoundMode::CAST_NONE, numPointsAlign);

                    Sub(tmpLocal1, tmpLocal3, xLocal, numPointsAlign);
                    Sub(tmpLocal2, tmpLocal4, yLocal, numPointsAlign);

                    Abs(param0Local, tmpLocal1, numPointsAlign);
                    Abs(param1Local, tmpLocal2, numPointsAlign);

                    Sub(xLocal, floatOneLocal, param0Local, numPointsAlign);
                    Sub(yLocal, floatOneLocal, param1Local, numPointsAlign);

                    Mul(weightLocal, xLocal, yLocal, numPointsAlign);
                    Mul(weightLocal[numPointsAlign], xLocal, param1Local, numPointsAlign);
                    Mul(weightLocal[numPointsAlign * 2], param0Local, yLocal, numPointsAlign);
                    Mul(weightLocal[numPointsAlign * 3], param0Local, param1Local, numPointsAlign);

                    Mul(weightLocal, weightLocal, attentionWeightLocal[weightOffset], numPointsAlign);
                    Mul(weightLocal[numPointsAlign], weightLocal[numPointsAlign], attentionWeightLocal[weightOffset], numPointsAlign);
                    Mul(weightLocal[numPointsAlign * 2], weightLocal[numPointsAlign * 2], attentionWeightLocal[weightOffset], numPointsAlign);
                    Mul(weightLocal[numPointsAlign * 3], weightLocal[numPointsAlign * 3], attentionWeightLocal[weightOffset], numPointsAlign);

                    valueOffset = (batch * numKeys * numHeads + offsetLocal.GetValue(level) * numHeads + head) * embedDims;

                    Duplicate<DTYPE_VALUE>(valueLocal, DTYPE_VALUE(0), 4 * numPoints * embedDimsAlign);
                    for (uint32_t point = 0; point < numPoints; point++)
                    {
                        x0 = x0Local.GetValue(point);
                        y0 = y0Local.GetValue(point);
                        x1 = x1Local.GetValue(point);
                        y1 = y1Local.GetValue(point);

                        if (isInRange(x0, w))
                        {
                            if (isInRange(y0, h))
                            {
                                DataCopy(valueLocal[point * embedDimsAlign * 4], valueGm[valueOffset + (y0 * w + x0) * tailNum], embedDimsAlign);
                            }
                            if (isInRange(y1, h))
                            {
                                DataCopy(valueLocal[point * embedDimsAlign * 4 + embedDimsAlign], valueGm[valueOffset + (y1 * w + x0) * tailNum], embedDimsAlign);
                            }
                        }
                        if (isInRange(x1, w))
                        {
                            if (isInRange(y0, h))
                            {
                                DataCopy(valueLocal[point * embedDimsAlign * 4 + embedDimsAlign * 2], valueGm[valueOffset + (y0 * w + x1) * tailNum], embedDimsAlign);
                            }
                            if (isInRange(y1, h))
                            {
                                DataCopy(valueLocal[point * embedDimsAlign * 4 + embedDimsAlign * 3], valueGm[valueOffset + (y1 * w + x1) * tailNum], embedDimsAlign);
                            }
                        }
                    }
                    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                    for (uint32_t point = 0; point < numPoints; point++)
                    {
                        leftTopWeight = weightLocal.GetValue(point);
                        leftBottomWeight = weightLocal.GetValue(numPointsAlign + point);
                        rightTopWeiight = weightLocal.GetValue(numPointsAlign * 2 + point);
                        rightBottomWeight = weightLocal.GetValue(numPointsAlign * 3 + point);

                        Muls(valueLocal[point * embedDimsAlign * 4], valueLocal[point * embedDimsAlign * 4], leftTopWeight, embedDimsAlign);
                        Muls(valueLocal[point * embedDimsAlign * 4 + embedDimsAlign], valueLocal[point * embedDimsAlign * 4 + embedDimsAlign], leftBottomWeight, embedDimsAlign);
                        Muls(valueLocal[point * embedDimsAlign * 4 + embedDimsAlign * 2], valueLocal[point * embedDimsAlign * 4 + embedDimsAlign * 2], rightTopWeiight, embedDimsAlign);
                        Muls(valueLocal[point * embedDimsAlign * 4 + embedDimsAlign * 3], valueLocal[point * embedDimsAlign * 4 + embedDimsAlign * 3], rightBottomWeight, embedDimsAlign);

                        Add(tmpResLocal[point * embedDimsAlign], valueLocal[point * embedDimsAlign * 4], valueLocal[point * embedDimsAlign * 4 + embedDimsAlign], embedDimsAlign);
                        Add(tmpResLocal2[point * embedDimsAlign], valueLocal[point * embedDimsAlign * 4 + embedDimsAlign * 2], valueLocal[point * embedDimsAlign * 4 + embedDimsAlign * 3], embedDimsAlign);
                        Add(tmpResLocal3[point * embedDimsAlign], tmpResLocal[point * embedDimsAlign], tmpResLocal2[point * embedDimsAlign], embedDimsAlign);

                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                        DataCopyPad(outputGm[moveOffset + head * embedDims], tmpResLocal3[point * embedDimsAlign], copyParams);
                    }
                }
            }
        }
        locationQueue.FreeTensor(locationLocal);
        attentionWeightsUb.FreeTensor(attentionWeightLocal);

        shapeQueue.FreeTensor(shapesLocal);
        offsetQueue.FreeTensor(offsetLocal);

        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventIdVToMte3);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
    }

private:
    TPipe *pipe;
    GlobalTensor<DTYPE_VALUE> valueGm, locationGm, attentionWeightsGm, outputGm;
    GlobalTensor<DTYPE_VALUE_SPATIAL_SHAPES> valueSpatialShapesGm, valueLevelStartIndexGm;

    TQue<QuePosition::VECIN, BUFFER_NUM> locationQueue, attentionWeightsUb, shapeQueue, offsetQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;

    TBuf<TPosition::VECCALC> tmpResUb, tmpResUb2, tmpResUb3, tmpXUb, tmpYUb, tmpParam0Ub, tmpParam1Ub, tmpIntX0Ub, tmpIntY0Ub, tmpIntX1Ub, tmpIntY1Ub, tmpUb1, tmpUb2, tmpUb3, tmpUb4;
    TBuf<TPosition::VECCALC> intOneUb, floatOneUb, leftTopWieightQueue, emptyUb;
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
    uint32_t blockNum = 32;

    DTYPE_VALUE leftTopWeight, rightTopWeiight, leftBottomWeight, rightBottomWeight, attnWeight;
    DTYPE_VALUE_SPATIAL_SHAPES h, w, x0, y0, x1, y1, valueOffset, weightOffset, locationOffset, moveOffset;
};

extern "C" __global__ __aicore__ void multi_scale_deformable_attn_function_v2(GM_ADDR value,
                                                                              GM_ADDR value_spatial_shapes,
                                                                              GM_ADDR value_level_start_index,
                                                                              GM_ADDR sampling_locations,
                                                                              GM_ADDR attention_weights,
                                                                              GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe; //
    GET_TILING_DATA(tiling_data, tiling);
    KernelMultiScaleDeformableAttnFunctionV2 op;
    op.Init(value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights, output, &tiling_data, &pipe);
    op.Process();
}
