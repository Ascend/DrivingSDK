
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 *
 * This sample is a very basic sample that implements vector add on Ascend plaform.
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelMultiScaleDeformableAttnFunctionV2
{
public:
    __aicore__ inline KernelMultiScaleDeformableAttnFunctionV2() {}
    __aicore__ inline void Init(GM_ADDR value,
                                GM_ADDR value_spatial_shapes,
                                GM_ADDR value_level_start_index,
                                GM_ADDR sampling_locations,
                                GM_ADDR attention_weights,
                                GM_ADDR output, MultiScaleDeformableAttnFunctionV2TilingData *tiling_data)
    {
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

        taskNum = batchSize * numQueries;
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

        pipe.InitBuffer(shapeQueue, BUFFER_NUM, AlignUp(numLevels * 2, dataAlign) * sizeof(DTYPE_VALUE));
        pipe.InitBuffer(offsetQueue, BUFFER_NUM, numLevelsAlign * sizeof(DTYPE_VALUE));

        pipe.InitBuffer(locationQueue, BUFFER_NUM, AlignUp(numLevels * numPoints * 2, dataAlign) * sizeof(DTYPE_VALUE));
        pipe.InitBuffer(attentionWeightsUb, BUFFER_NUM, AlignUp(numLevels * numPoints, dataAlign) * sizeof(DTYPE_VALUE));
        pipe.InitBuffer(outputQueue, BUFFER_NUM, embedDimsAlign * sizeof(DTYPE_VALUE));

        pipe.InitBuffer(tmpUb1, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe.InitBuffer(tmpUb2, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe.InitBuffer(tmpUb3, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe.InitBuffer(tmpUb4, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));

        pipe.InitBuffer(tmpResUb, BUFFER_NUM, embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe.InitBuffer(tmpResUb2, BUFFER_NUM, embedDimsAlign * sizeof(DTYPE_VALUE));

        pipe.InitBuffer(intOneUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE_SPATIAL_SHAPES));
        pipe.InitBuffer(floatOneUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));

        pipe.InitBuffer(tmpXUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe.InitBuffer(tmpYUb, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe.InitBuffer(tmpParam0Ub, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe.InitBuffer(tmpParam1Ub, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));

        pipe.InitBuffer(tmpIntX0Ub, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE_SPATIAL_SHAPES));
        pipe.InitBuffer(tmpIntY0Ub, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE_SPATIAL_SHAPES));
        pipe.InitBuffer(tmpIntX1Ub, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE_SPATIAL_SHAPES));
        pipe.InitBuffer(tmpIntY1Ub, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE_SPATIAL_SHAPES));

        pipe.InitBuffer(leftTopWieightQueue, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe.InitBuffer(leftBottomWieightQueue, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe.InitBuffer(rightTopWieightQueue, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));
        pipe.InitBuffer(rightBottomWieightQueue, BUFFER_NUM, numPointsAlign * sizeof(DTYPE_VALUE));

        pipe.InitBuffer(leftTopValueUb, BUFFER_NUM, embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe.InitBuffer(leftBottomValueUb, BUFFER_NUM, embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe.InitBuffer(rightTopValueUb, BUFFER_NUM, embedDimsAlign * sizeof(DTYPE_VALUE));
        pipe.InitBuffer(rightBottomValueUb, BUFFER_NUM, embedDimsAlign * sizeof(DTYPE_VALUE));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t taskIdx = startOffset; taskIdx < endOffset; taskIdx++)
        {
            batch = taskIdx / numQueries;
            query = taskIdx % numQueries;
            pipe_barrier(PIPE_ALL);
            Compute(batch, query);
        }
    }

private:
    __aicore__ inline bool isInRange(DTYPE_VALUE_SPATIAL_SHAPES x, DTYPE_VALUE_SPATIAL_SHAPES upper)
    {
        return 0 <= x && x < upper;
    }

    __aicore__ inline void Compute(uint32_t batch, uint32_t query)
    {
        LocalTensor<DTYPE_VALUE> tmpResLocal = tmpResUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> tmpResLocal2 = tmpResUb2.Get<DTYPE_VALUE>();

        LocalTensor<DTYPE_VALUE> leftTopValueLocal = leftTopValueUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> leftBottomValueUbLocal = leftBottomValueUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> rightTopValueUbLocal = rightTopValueUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> rightBottomValueUbLocal = rightBottomValueUb.Get<DTYPE_VALUE>();

        LocalTensor<DTYPE_VALUE> leftTopWeiightLocal = leftTopWieightQueue.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> leftBottomWeightLocal = leftBottomWieightQueue.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> rightTopWeiightLocal = rightTopWieightQueue.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> rightBottomWeightLocal = rightBottomWieightQueue.Get<DTYPE_VALUE>();

        LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> shapesLocal = shapeQueue.AllocTensor<DTYPE_VALUE_SPATIAL_SHAPES>();
        LocalTensor<DTYPE_VALUE_SPATIAL_SHAPES> offsetLocal = offsetQueue.AllocTensor<DTYPE_VALUE_SPATIAL_SHAPES>();

        LocalTensor<DTYPE_VALUE> locationLocal = locationQueue.AllocTensor<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> attentionWeightLocal = attentionWeightsUb.AllocTensor<DTYPE_VALUE>();

        LocalTensor<DTYPE_VALUE> resLocal = outputQueue.AllocTensor<DTYPE_VALUE>();

        LocalTensor<DTYPE_VALUE> xLocal = tmpXUb.Get<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> yLocal = tmpYUb.Get<DTYPE_VALUE>();

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
        DataCopyParams copyParams{1, (uint16_t)(embedDims * sizeof(DTYPE_VALUE)), 0, 0};

        DataCopy(shapesLocal, valueSpatialShapesGm, AlignUp(numLevels * 2, dataAlign));
        DataCopy(offsetLocal, valueLevelStartIndexGm, numLevelsAlign);
        Duplicate<DTYPE_VALUE>(resLocal, DTYPE_VALUE(0), embedDimsAlign);
        moveOffset = batch * numQueries * numHeads * embedDims + query * numHeads * embedDims;
        pipe_barrier(PIPE_ALL);

        for (uint32_t head = 0; head < numHeads; head++)
        {
            DataCopyPad(outputGm[moveOffset + head * embedDims], resLocal, copyParams);
        }
        pipe_barrier(PIPE_ALL);

        for (uint32_t head = 0; head < numHeads; head++)
        {
            weightOffset = (batch * numQueries * numHeads * numLevels + query * numHeads * numLevels + head * numLevels) * numPoints;

            pipe_barrier(PIPE_ALL);

            DataCopy(locationLocal, locationGm[weightOffset * 2], AlignUp(numLevels * numPoints * 2, dataAlign));
            DataCopy(attentionWeightLocal, attentionWeightsGm[weightOffset], AlignUp(numLevels * numPoints, dataAlign));

            pipe_barrier(PIPE_ALL);
            for (uint32_t level = 0; level < numLevels; level++)
            {
                h = shapesLocal.GetValue(level * 2);
                w = shapesLocal.GetValue(level * 2 + 1);
                for (uint32_t point = 0; point < numPoints; point++)
                {
                    locationOffset = (level * numPoints + point) * 2;
                    xLocal.SetValue(point, locationLocal.GetValue(locationOffset));
                    yLocal.SetValue(point, locationLocal.GetValue(locationOffset + 1));
                }

                pipe_barrier(PIPE_ALL);

                Muls(tmpLocal1, xLocal, (DTYPE_VALUE)w, numPointsAlign);
                Muls(tmpLocal2, yLocal, (DTYPE_VALUE)h, numPointsAlign);
                pipe_barrier(PIPE_ALL);

                Adds(param0Local, tmpLocal1, (DTYPE_VALUE)0.5, numPointsAlign);
                Adds(param1Local, tmpLocal2, (DTYPE_VALUE)0.5, numPointsAlign);
                pipe_barrier(PIPE_ALL);

                Cast(x1Local, param0Local, RoundMode::CAST_FLOOR, numPointsAlign);
                Cast(y1Local, param1Local, RoundMode::CAST_FLOOR, numPointsAlign);
                pipe_barrier(PIPE_ALL);

                Adds(tmpLocal3, param0Local, (DTYPE_VALUE)-1, numPointsAlign);
                Adds(tmpLocal4, param1Local, (DTYPE_VALUE)-1, numPointsAlign);
                pipe_barrier(PIPE_ALL);

                Sub(x0Local, x1Local, intOneLocal, numPointsAlign);
                Sub(y0Local, y1Local, intOneLocal, numPointsAlign);
                pipe_barrier(PIPE_ALL);

                Cast(xLocal, x0Local, RoundMode::CAST_NONE, numPointsAlign);
                Cast(yLocal, y0Local, RoundMode::CAST_NONE, numPointsAlign);
                pipe_barrier(PIPE_ALL);

                Sub(tmpLocal1, tmpLocal3, xLocal, numPointsAlign);
                Sub(tmpLocal2, tmpLocal4, yLocal, numPointsAlign);
                pipe_barrier(PIPE_ALL);

                Abs(param0Local, tmpLocal1, numPointsAlign);
                Abs(param1Local, tmpLocal2, numPointsAlign);
                pipe_barrier(PIPE_ALL);

                Sub(xLocal, floatOneLocal, param0Local, numPointsAlign);
                Sub(yLocal, floatOneLocal, param1Local, numPointsAlign);
                pipe_barrier(PIPE_ALL);

                Mul(leftTopWeiightLocal, xLocal, yLocal, numPointsAlign);
                Mul(leftBottomWeightLocal, xLocal, param1Local, numPointsAlign);
                Mul(rightTopWeiightLocal, param0Local, yLocal, numPointsAlign);
                Mul(rightBottomWeightLocal, param0Local, param1Local, numPointsAlign);
                pipe_barrier(PIPE_ALL);

                Duplicate<DTYPE_VALUE>(resLocal, DTYPE_VALUE(0), embedDimsAlign);

                for (uint32_t point = 0; point < numPoints; point++)
                {
                    Duplicate<DTYPE_VALUE>(leftTopValueLocal, DTYPE_VALUE(0), embedDimsAlign);
                    Duplicate<DTYPE_VALUE>(leftBottomValueUbLocal, DTYPE_VALUE(0), embedDimsAlign);
                    Duplicate<DTYPE_VALUE>(rightTopValueUbLocal, DTYPE_VALUE(0), embedDimsAlign);
                    Duplicate<DTYPE_VALUE>(rightBottomValueUbLocal, DTYPE_VALUE(0), embedDimsAlign);

                    x0 = x0Local.GetValue(point);
                    y0 = y0Local.GetValue(point);
                    x1 = x1Local.GetValue(point);
                    y1 = y1Local.GetValue(point);

                    valueOffset = batch * numKeys * numHeads + offsetLocal.GetValue(level) * numHeads + head;
                    pipe_barrier(PIPE_ALL);

                    if (isInRange(x0, w))
                    {
                        if (isInRange(y0, h))
                        {
                            DataCopy(leftTopValueLocal, valueGm[(valueOffset + (y0 * w + x0) * numHeads) * embedDims], embedDimsAlign);
                        }
                        if (isInRange(y1, h))
                        {
                            DataCopy(leftBottomValueUbLocal, valueGm[(valueOffset + (y1 * w + x0) * numHeads) * embedDims], embedDimsAlign);
                        }
                    }
                    if (isInRange(x1, w))
                    {
                        if (isInRange(y0, h))
                        {
                            DataCopy(rightTopValueUbLocal, valueGm[(valueOffset + (y0 * w + x1) * numHeads) * embedDims], embedDimsAlign);
                        }
                        if (isInRange(y1, h))
                        {
                            DataCopy(rightBottomValueUbLocal, valueGm[(valueOffset + (y1 * w + x1) * numHeads) * embedDims], embedDimsAlign);
                        }
                    }
                    pipe_barrier(PIPE_ALL);

                    Muls(leftTopValueLocal, leftTopValueLocal, leftTopWeiightLocal.GetValue(point), embedDimsAlign);
                    Muls(rightTopValueUbLocal, rightTopValueUbLocal, rightTopWeiightLocal.GetValue(point), embedDimsAlign);
                    Muls(leftBottomValueUbLocal, leftBottomValueUbLocal, leftBottomWeightLocal.GetValue(point), embedDimsAlign);
                    Muls(rightBottomValueUbLocal, rightBottomValueUbLocal, rightBottomWeightLocal.GetValue(point), embedDimsAlign);
                    pipe_barrier(PIPE_ALL);
                    Add(tmpResLocal, leftTopValueLocal, rightTopValueUbLocal, embedDimsAlign);
                    Add(tmpResLocal2, leftBottomValueUbLocal, rightBottomValueUbLocal, embedDimsAlign);
                    pipe_barrier(PIPE_ALL);
                    Add(tmpResLocal, tmpResLocal, tmpResLocal2, embedDimsAlign);
                    pipe_barrier(PIPE_ALL);
                    Muls(tmpResLocal, tmpResLocal, attentionWeightLocal.GetValue(level * numPoints + point), embedDimsAlign);
                    pipe_barrier(PIPE_ALL);
                    Add(resLocal, resLocal, tmpResLocal, embedDimsAlign);
                }
                pipe_barrier(PIPE_ALL);

                SetAtomicAdd<DTYPE_VALUE>();
                DataCopyPad(outputGm[moveOffset + head * embedDims], resLocal, copyParams);
                SetAtomicNone();
            }
        }
        locationQueue.FreeTensor(locationLocal);
        attentionWeightsUb.FreeTensor(attentionWeightLocal);
        outputQueue.FreeTensor(resLocal);
        shapeQueue.FreeTensor(shapesLocal);
        offsetQueue.FreeTensor(offsetLocal);
    }

private:
    TPipe pipe;
    GlobalTensor<DTYPE_VALUE> valueGm, locationGm, attentionWeightsGm, outputGm;
    GlobalTensor<DTYPE_VALUE_SPATIAL_SHAPES> valueSpatialShapesGm, valueLevelStartIndexGm;

    TQue<QuePosition::VECIN, BUFFER_NUM> locationQueue, attentionWeightsUb, shapeQueue, offsetQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;

    TBuf<TPosition::VECCALC> tmpResUb, tmpResUb2, tmpXUb, tmpYUb, tmpParam0Ub, tmpParam1Ub, tmpIntX0Ub, tmpIntY0Ub, tmpIntX1Ub, tmpIntY1Ub, tmpUb1, tmpUb2, tmpUb3, tmpUb4;
    TBuf<TPosition::VECCALC> intOneUb, floatOneUb, leftTopValueUb, leftBottomValueUb, rightTopValueUb, rightBottomValueUb;
    TBuf<TPosition::VECCALC> leftTopWieightQueue, leftBottomWieightQueue, rightTopWieightQueue, rightBottomWieightQueue;

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
    uint32_t blockNum = 32;

    DTYPE_VALUE_SPATIAL_SHAPES h, w, x0, y0, x1, y1, valueOffset, weightOffset, locationOffset, moveOffset;
};

extern "C" __global__ __aicore__ void multi_scale_deformable_attn_function_v2(GM_ADDR value,
                                                                              GM_ADDR value_spatial_shapes,
                                                                              GM_ADDR value_level_start_index,
                                                                              GM_ADDR sampling_locations,
                                                                              GM_ADDR attention_weights,
                                                                              GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelMultiScaleDeformableAttnFunctionV2 op;
    op.Init(value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights, output, &tiling_data);
    op.Process();
}
