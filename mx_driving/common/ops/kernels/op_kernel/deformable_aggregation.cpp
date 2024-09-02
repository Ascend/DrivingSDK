/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;


template<typename DTYPE_F, typename DTYPE_I>
class KernelDeformableAggregation {
public:
    __aicore__ inline KernelDeformableAggregation() {}
    __aicore__ inline void Init(GM_ADDR mc_ms_feat, GM_ADDR spatial_shape, GM_ADDR scale_start_index,
        GM_ADDR sampling_location, GM_ADDR weights, GM_ADDR out, const DeformableAggregationTilingData* tiling_data)
    {
        bs = tiling_data->bs;
        numFeats = tiling_data->numFeats;
        numEmbeds = tiling_data->numEmbeds;
        numAnchor = tiling_data->numAnchor;
        numPoints = tiling_data->numPoints;
        numCams = tiling_data->numCams;
        numScale = tiling_data->numScale;
        numGroups = tiling_data->numGroups;
        cAligned = tiling_data->cAligned;
        singleAligned = tiling_data->singleAligned;
        average = tiling_data->average;
        taskLast = tiling_data->taskLast;
        usedCoreNum = tiling_data->usedCoreNum;
        groupAligned = tiling_data->groupAligned;

        CopyParamasInit();

        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        mcMsFeatGmLength = bs * numFeats * numEmbeds;
        spatialShapeGmLength = numCams * numScale * 2;
        scaleStartIndexLength = numCams * numScale;
        samplingLocationGmLength = bs * numAnchor * numPoints * numCams * 2;
        weightsGmLength = bs * numAnchor * numPoints * numCams * numScale * numGroups;
        outGmLength = bs * numAnchor * numEmbeds;

        mcMsFeatGm.SetGlobalBuffer((__gm__ DTYPE_F*)mc_ms_feat, mcMsFeatGmLength);
        samplingLocationGm.SetGlobalBuffer((__gm__ DTYPE_F*)sampling_location, samplingLocationGmLength);
        weightsGm.SetGlobalBuffer((__gm__ DTYPE_F*)weights, weightsGmLength);
        outGm.SetGlobalBuffer((__gm__ DTYPE_F*)out, outGmLength);
        spatialShapesGm.SetGlobalBuffer((__gm__ DTYPE_I*)spatial_shape, spatialShapeGmLength);
        scaleStartIndexGm.SetGlobalBuffer((__gm__ DTYPE_I*)scale_start_index, scaleStartIndexLength);

        pipe.InitBuffer(inQueueWeights, groupAligned * sizeof(DTYPE_F));
        pipe.InitBuffer(inQueueLocation, singleAligned * sizeof(DTYPE_F));
        pipe.InitBuffer(inQueueScaleStart, singleAligned * sizeof(DTYPE_I));
        pipe.InitBuffer(inQueueSpatialShape, singleAligned * sizeof(DTYPE_I));
        pipe.InitBuffer(inQueueFloat, singleAligned * sizeof(DTYPE_F));
        pipe.InitBuffer(inQueueInt, singleAligned * sizeof(DTYPE_I));
        pipe.InitBuffer(inQueueFeat, cAligned * sizeof(DTYPE_F));
        pipe.InitBuffer(inQueueV1, cAligned * sizeof(DTYPE_F));
        pipe.InitBuffer(inQueueV2, cAligned * sizeof(DTYPE_F));
        pipe.InitBuffer(inQueueV3, cAligned * sizeof(DTYPE_F));
        pipe.InitBuffer(inQueueV4, cAligned * sizeof(DTYPE_F));
        pipe.InitBuffer(inQueueWightMul, cAligned * sizeof(DTYPE_F));
    }

    __aicore__ inline void CopyParamasInit()
    {
        copyParamsOut.blockCount = 1;
        copyParamsOut.blockLen = static_cast<uint32_t>(numEmbeds * sizeof(DTYPE_F));
        copyParamsOut.srcStride = 0;
        copyParamsOut.dstStride = 0;
        copyParamsOut.rsv = 0;
    }

    __aicore__ inline void Process()
    {
        int32_t tmp = average;
        if (GetBlockIdx() < taskLast) {
            tmp = tmp + 1;
        }
        for (int32_t i = 0; i < tmp; i++) {
            ComputeAndCopyOut(i);
        }
    }

    __aicore__ inline void ComputeAndCopyOut(int32_t i)
    {
        LocalTensor<DTYPE_F> featLocal = inQueueFeat.Get<DTYPE_F>();
        LocalTensor<DTYPE_F> weightLocal = inQueueWeights.Get<DTYPE_F>();
        LocalTensor<DTYPE_F> loactionLocal = inQueueLocation.Get<DTYPE_F>();
        LocalTensor<DTYPE_I> scaleStartLocal = inQueueScaleStart.Get<DTYPE_I>();
        LocalTensor<DTYPE_I> spatialShapeLocal = inQueueSpatialShape.Get<DTYPE_I>();
        LocalTensor<DTYPE_F> floatLocal = inQueueFloat.Get<DTYPE_F>();
        LocalTensor<DTYPE_I> intLocal = inQueueInt.Get<DTYPE_I>();
        LocalTensor<DTYPE_F> v1Local = inQueueV1.Get<DTYPE_F>();
        LocalTensor<DTYPE_F> v2Local = inQueueV2.Get<DTYPE_F>();
        LocalTensor<DTYPE_F> v3Local = inQueueV3.Get<DTYPE_F>();
        LocalTensor<DTYPE_F> v4Local = inQueueV4.Get<DTYPE_F>();
        LocalTensor<DTYPE_F> weightMulLocal = inQueueWightMul.Get<DTYPE_F>();

        int32_t offset = average * GetBlockIdx() + taskLast;
        if (GetBlockIdx() < taskLast) {
            offset = (average + 1) * GetBlockIdx();
        }

        int32_t idx = offset + i;
        int32_t chanenlOffset = 0;
        int32_t weightsOffset = idx;

        int32_t scaleIndex = idx % numScale;
        idx = idx / numScale;

        int32_t camIndex = idx % numCams;
        idx = idx / numCams;

        int32_t ptsIndex = idx % numPoints;
        idx = idx / numPoints;

        int32_t anchorIndex = idx % numAnchor;
        idx = idx / numAnchor;

        int32_t batchIndex = idx % bs;
        idx = idx / batchIndex;

        int32_t loactionOffset = batchIndex * numAnchor * numPoints * numCams * 2 +
                                 anchorIndex * numPoints * numCams * 2 + ptsIndex * numCams * 2 + camIndex * 2;

        DataCopy(loactionLocal, samplingLocationGm[loactionOffset], singleAligned);

        float locW = loactionLocal.GetValue(0);
        float locH = loactionLocal.GetValue(1);

        if (locW <= 0 || locW >= 1) {
            return;
        }
        if (locH <= 0 || locH >= 1) {
            return;
        }

        int32_t scaleStartOffset = camIndex * numScale + scaleIndex;

        DataCopy(scaleStartLocal, scaleStartIndexGm[scaleStartOffset], singleAligned);
        int32_t scaleStartIdx = scaleStartLocal.GetValue(0);
        int32_t valueOffset = batchIndex * numFeats * numEmbeds + scaleStartIdx * numEmbeds;

        int32_t spatialShapeOffset = camIndex * numScale * 2 + scaleIndex * 2;
        DataCopy(spatialShapeLocal, spatialShapesGm[spatialShapeOffset], singleAligned);

        int32_t h = spatialShapeLocal.GetValue(0);
        int32_t w = spatialShapeLocal.GetValue(1);

        float hIm = locH * h - float(0.5);
        float wIm = locW * w - float(0.5);

        floatLocal.SetValue(0, hIm);
        floatLocal.SetValue(1, wIm);

        Cast(intLocal, floatLocal, RoundMode::CAST_FLOOR, 8);

        int32_t hLow = intLocal.GetValue(0);
        int32_t wLow = intLocal.GetValue(1);
        int32_t hHigh = hLow + 1;
        int32_t wHigh = wLow + 1;

        float lh = hIm - hLow;
        float lw = wIm - wLow;
        float hh = 1 - lh;
        float hw = 1 - lw;

        int32_t wStride = numEmbeds;
        int32_t hStride = w * wStride;
        int32_t hLowPtrOffset = hLow * hStride;
        int32_t hHighPtrOffset = hLowPtrOffset + hStride;
        int32_t wLowPtrOffset = wLow * wStride;
        int32_t wHighPtrOffset = wLowPtrOffset + wStride;

        float w1 = hh * hw;
        float w2 = hh * lw;
        float w3 = lh * hw;
        float w4 = lh * lw;

        Duplicate(v1Local, static_cast<DTYPE_F>(0), cAligned);
        if (hLow >= 0 && wLow >= 0) {
            int32_t ptr1 = valueOffset + hLowPtrOffset + wLowPtrOffset + chanenlOffset;
            DataCopy(v1Local, mcMsFeatGm[ptr1], cAligned);
        }

        Duplicate(v2Local, static_cast<DTYPE_F>(0), cAligned);
        if (hLow >= 0 && wHigh <= w - 1) {
            int32_t ptr2 = valueOffset + hLowPtrOffset + wHighPtrOffset + chanenlOffset;
            DataCopy(v2Local, mcMsFeatGm[ptr2], cAligned);
        }

        Duplicate(v3Local, static_cast<DTYPE_F>(0), cAligned);
        if (hHigh <= h - 1 && wLow >= 0) {
            int32_t ptr3 = valueOffset + hHighPtrOffset + wLowPtrOffset + chanenlOffset;
            DataCopy(v3Local, mcMsFeatGm[ptr3], cAligned);
        }

        Duplicate(v4Local, static_cast<DTYPE_F>(0), cAligned);
        if (hHigh <= h - 1 && wHigh <= w - 1) {
            int32_t ptr4 = valueOffset + hHighPtrOffset + wHighPtrOffset + chanenlOffset;
            DataCopy(v4Local, mcMsFeatGm[ptr4], cAligned);
        }

        pipe_barrier(PIPE_ALL);
        Muls(v1Local, v1Local, w1, cAligned);
        Muls(v2Local, v2Local, w2, cAligned);
        Muls(v3Local, v3Local, w3, cAligned);
        Muls(v4Local, v4Local, w4, cAligned);

        Add(v1Local, v1Local, v2Local, cAligned);
        Add(v1Local, v1Local, v3Local, cAligned);
        Add(v1Local, v1Local, v4Local, cAligned);

        int32_t weightIdx = weightsOffset * numGroups;
        DataCopy(weightLocal, weightsGm[weightIdx], groupAligned);

        Duplicate(weightMulLocal, static_cast<DTYPE_F>(0), cAligned);

        for (int32_t groupIdx = 0; groupIdx < numGroups; groupIdx++) {
            int32_t offset = groupIdx * (numEmbeds / numGroups);
            float weight = weightLocal.GetValue(groupIdx);
            if (numEmbeds / numGroups % BloclAlign == 0) {
                Duplicate(weightMulLocal[offset], static_cast<DTYPE_F>(weight), numEmbeds / numGroups);
            } else {
                for (int32_t idx = 0; idx < numEmbeds / numGroups; idx++) {
                    weightMulLocal.SetValue(offset + idx, weight);
                }
            }
        }

        Mul(v1Local, v1Local, weightMulLocal, cAligned);
        pipe_barrier(PIPE_ALL);

        SetAtomicAdd<DTYPE_F>();
        DataCopyPad(outGm[batchIndex * numAnchor * numEmbeds + anchorIndex * numEmbeds], v1Local, copyParamsOut);
        pipe_barrier(PIPE_ALL);
        SetAtomicNone();
    }

private:
    TPipe pipe;

    TBuf<TPosition::VECCALC> inQueueFeat, inQueueWeights, inQueueLocation, inQueueScaleStart, inQueueSpatialShape;
    TBuf<TPosition::VECCALC> inQueueFloat, inQueueInt, inQueueWightMul;
    TBuf<TPosition::VECCALC> inQueueV1, inQueueV2, inQueueV3, inQueueV4;

    GlobalTensor<DTYPE_F> mcMsFeatGm, samplingLocationGm, weightsGm, outGm;
    GlobalTensor<DTYPE_I> spatialShapesGm, scaleStartIndexGm;

    uint32_t mcMsFeatGmLength;
    uint32_t spatialShapeGmLength;
    uint32_t scaleStartIndexLength;
    uint32_t samplingLocationGmLength;
    uint32_t weightsGmLength;
    uint32_t outGmLength;
    uint32_t bs;
    uint32_t numFeats;
    uint32_t numEmbeds;
    uint32_t numAnchor;
    uint32_t numPoints;
    uint32_t numCams;
    uint32_t numScale;
    uint32_t numGroups;
    uint32_t cAligned;
    uint32_t singleAligned;
    uint32_t average;
    uint32_t taskLast;
    uint32_t usedCoreNum;
    uint32_t groupAligned;
    uint32_t BloclAlign;

    DataCopyExtParams copyParamsOut;
};

extern "C" __global__ __aicore__ void deformable_aggregation(GM_ADDR mc_ms_feat, GM_ADDR spatial_shape,
    GM_ADDR scale_start_index, GM_ADDR sampling_location, GM_ADDR weights, GM_ADDR out, GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelDeformableAggregation<float, int32_t> op;
    op.Init(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights, out, &tiling_data);
    op.Process();
}