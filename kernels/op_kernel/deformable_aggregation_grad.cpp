/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;


template<typename DTYPE_F, typename DTYPE_I>
class KernelDeformableAggregationGrad {
public:
    __aicore__ inline KernelDeformableAggregationGrad() {}

    __aicore__ inline void Init(
        GM_ADDR mc_ms_feat,
        GM_ADDR spatial_shape,
        GM_ADDR scale_start_index,
        GM_ADDR sampling_location,
        GM_ADDR weights,
        GM_ADDR grad_output,
        GM_ADDR grad_mc_ms_feat,
        GM_ADDR grad_sampling_location,
        GM_ADDR grad_weights,
        const DeformableAggregationGradTilingData* tiling_data)
    {
        batchSize = tiling_data->batchSize;
        numFeat = tiling_data->numFeat;
        numEmbeds = tiling_data->numEmbeds;
        numAnchors = tiling_data->numAnchors;
        numPoints = tiling_data->numPoints;
        numCams = tiling_data->numCams;
        numScale = tiling_data->numScale;
        numGroups = tiling_data->numGroups;

        cAligned = tiling_data->cAligned;
        singleAligned = tiling_data->singleAligned;

        average = tiling_data->average;
        taskLast = tiling_data->taskLast;
        usedCoreNum = tiling_data->usedCoreNum;
        splitNum = numEmbeds / numGroups;

        CopyParamasInit();

        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        mcMsFeatGmLength = static_cast<uint64_t>(batchSize) * numFeat * numEmbeds;
        spatialShapeGmLength = static_cast<uint64_t>(numCams) * numScale * 2;
        scaleStartIndexLength = static_cast<uint64_t>(numCams) * numScale;
        samplingLocationGmLength = static_cast<uint64_t>(batchSize) * numAnchors * numPoints * numCams * 2;
        weightsGmLength = static_cast<uint64_t>(batchSize) * numAnchors * numPoints * numCams * numScale * numGroups;
        gradOutputGmLength = static_cast<uint64_t>(batchSize) * numAnchors * numEmbeds;
        gradMcMsFeatGmLength = static_cast<uint64_t>(mcMsFeatGmLength);
        gradSamplingLocationGmLength = static_cast<uint64_t>(samplingLocationGmLength);
        gradWeightsGmLength = static_cast<uint64_t>(weightsGmLength);

        mcMsFeatGm.SetGlobalBuffer((__gm__ DTYPE_F*)mc_ms_feat, mcMsFeatGmLength);
        spatialShapeGm.SetGlobalBuffer((__gm__ DTYPE_I*)spatial_shape, spatialShapeGmLength);
        scaleStartIndexGm.SetGlobalBuffer((__gm__ DTYPE_I*)scale_start_index, scaleStartIndexLength);
        samplingLocationGm.SetGlobalBuffer((__gm__ DTYPE_F*)sampling_location, samplingLocationGmLength);
        weightsGm.SetGlobalBuffer((__gm__ DTYPE_F*)weights, weightsGmLength);
        gradOutputGm.SetGlobalBuffer((__gm__ DTYPE_F*)grad_output, gradOutputGmLength);
        gradMcMsFeatGm.SetGlobalBuffer((__gm__ DTYPE_F*)grad_mc_ms_feat, gradMcMsFeatGmLength);
        gradSamplingLocationGm.SetGlobalBuffer((__gm__ DTYPE_F*)grad_sampling_location, gradSamplingLocationGmLength);
        gradWeightsGm.SetGlobalBuffer((__gm__ DTYPE_F*)grad_weights, gradWeightsGmLength);

        pipe.InitBuffer(inQueueFeat, cAligned * sizeof(DTYPE_F));
        pipe.InitBuffer(inQueueSpatialShape, singleAligned * sizeof(DTYPE_I));
        pipe.InitBuffer(inQueueScaleStart, singleAligned * sizeof(DTYPE_I));
        pipe.InitBuffer(inQueueLocation, singleAligned * sizeof(DTYPE_F));
        pipe.InitBuffer(inQueueWeights, singleAligned * sizeof(DTYPE_F));
        pipe.InitBuffer(inQueueGradOutput, cAligned * sizeof(DTYPE_F));

        pipe.InitBuffer(inQueueFloat, singleAligned * sizeof(DTYPE_F));
        pipe.InitBuffer(inQueueInt, singleAligned * sizeof(DTYPE_I));

        pipe.InitBuffer(inQueueV1, cAligned * sizeof(DTYPE_F));
        pipe.InitBuffer(inQueueV2, cAligned * sizeof(DTYPE_F));
        pipe.InitBuffer(inQueueV3, cAligned * sizeof(DTYPE_F));
        pipe.InitBuffer(inQueueV4, cAligned * sizeof(DTYPE_F));

        pipe.InitBuffer(tmpTopGradMcMsFeat, cAligned * sizeof(DTYPE_F));
        pipe.InitBuffer(tmpTopGradMcMsFeatMulsWx, cAligned * sizeof(DTYPE_F));
        pipe.InitBuffer(tmpGradHweight, cAligned * sizeof(DTYPE_F));
        pipe.InitBuffer(tmpGradWweight, cAligned * sizeof(DTYPE_F));
        pipe.InitBuffer(tmpPointGradHweight, cAligned * sizeof(DTYPE_F));
        pipe.InitBuffer(tmpPointGradWweight, cAligned * sizeof(DTYPE_F));
    }

    __aicore__ inline void CopyParamasInit()
    {
        copyParamsOut.blockCount = 1;
        copyParamsOut.blockLen = static_cast<uint32_t>((numEmbeds / numGroups) * sizeof(DTYPE_F));
        copyParamsOut.srcStride = 0;
        copyParamsOut.dstStride = 0;
        copyParamsOut.rsv = 0;

        copyParamsOutOneNum.blockCount = 1;
        copyParamsOutOneNum.blockLen = static_cast<uint32_t>(1 * sizeof(DTYPE_F));
        copyParamsOutOneNum.srcStride = 0;
        copyParamsOutOneNum.dstStride = 0;
        copyParamsOutOneNum.rsv = 0;
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
        LocalTensor<DTYPE_I> spatialShapeLocal = inQueueSpatialShape.Get<DTYPE_I>();
        LocalTensor<DTYPE_I> scaleStartLocal = inQueueScaleStart.Get<DTYPE_I>();
        LocalTensor<DTYPE_F> locationLocal = inQueueLocation.Get<DTYPE_F>();
        LocalTensor<DTYPE_F> weightLocal = inQueueWeights.Get<DTYPE_F>();
        LocalTensor<DTYPE_F> gradOutputLocal = inQueueGradOutput.Get<DTYPE_F>();

        LocalTensor<DTYPE_F> floatLocal = inQueueFloat.Get<DTYPE_F>();
        LocalTensor<DTYPE_I> intLocal = inQueueInt.Get<DTYPE_I>();

        LocalTensor<DTYPE_F> v1Local = inQueueV1.Get<DTYPE_F>();
        LocalTensor<DTYPE_F> v2Local = inQueueV2.Get<DTYPE_F>();
        LocalTensor<DTYPE_F> v3Local = inQueueV3.Get<DTYPE_F>();
        LocalTensor<DTYPE_F> v4Local = inQueueV4.Get<DTYPE_F>();

        LocalTensor<DTYPE_F> topGradMcMsFeatLocal = tmpTopGradMcMsFeat.Get<DTYPE_F>();
        LocalTensor<DTYPE_F> topGradMcMsFeatMulsWxLocal = tmpTopGradMcMsFeatMulsWx.Get<DTYPE_F>();
        LocalTensor<DTYPE_F> gradHweightLocal = tmpGradHweight.Get<DTYPE_F>();
        LocalTensor<DTYPE_F> gradWweightLocal = tmpGradWweight.Get<DTYPE_F>();
        LocalTensor<DTYPE_F> PointGradHweightLocal = tmpPointGradHweight.Get<DTYPE_F>();
        LocalTensor<DTYPE_F> PointGradWweightLocal = tmpPointGradWweight.Get<DTYPE_F>();

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

        int32_t anchorIndex = idx % numAnchors;
        idx = idx / numAnchors;

        int32_t batchIndex = idx % batchSize;

        uint64_t locationOffset = static_cast<uint64_t>(batchIndex) * numAnchors * numPoints * numCams * 2 + anchorIndex * numPoints * numCams * 2 + ptsIndex * numCams * 2 + camIndex * 2;

        DataCopy(locationLocal, samplingLocationGm[locationOffset], singleAligned);

        float locW = locationLocal.GetValue(0);
        float locH = locationLocal.GetValue(1);
        if (locW <= 0 || locW >= 1) {
            return ;
        }
        if (locH <= 0 || locH >= 1) {
            return ;
        }

        uint64_t scaleStartOffset = static_cast<uint64_t>(camIndex) * numScale + scaleIndex;
        DataCopy(scaleStartLocal, scaleStartIndexGm[scaleStartOffset], singleAligned);

        int32_t scaleStartIdx = scaleStartLocal.GetValue(0);
        int32_t valueOffset = batchIndex * numFeat * numEmbeds + scaleStartIdx * numEmbeds;

        uint64_t spatialShapeOffset = static_cast<uint64_t>(camIndex) * numScale * 2 + scaleIndex * 2;
        DataCopy(spatialShapeLocal, spatialShapeGm[spatialShapeOffset], singleAligned);

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

        for (int32_t groupIdx = 0; groupIdx < numGroups; groupIdx++) {
            uint64_t weightIdx = static_cast<uint64_t>(weightsOffset) * numGroups + groupIdx;

            DataCopy(weightLocal, weightsGm[weightIdx], singleAligned);
            float weight = weightLocal.GetValue(0);

            uint64_t gradOutputOffset = static_cast<uint64_t>(batchIndex) * numAnchors * numEmbeds +  anchorIndex * numEmbeds + chanenlOffset;

            DataCopy(gradOutputLocal, gradOutputGm[gradOutputOffset], cAligned);
            pipe_barrier(PIPE_ALL);

            Muls(topGradMcMsFeatLocal, gradOutputLocal, weight, cAligned);

            Duplicate(gradHweightLocal, (DTYPE_F)0, cAligned);
            Duplicate(gradWweightLocal, (DTYPE_F)0, cAligned);

            Duplicate(v1Local, (DTYPE_F)0, cAligned);
            if (hLow >= 0 && wLow >=0) {
                uint64_t ptr1 = static_cast<uint64_t>(valueOffset) + hLowPtrOffset + wLowPtrOffset + chanenlOffset;
                DataCopy(v1Local, mcMsFeatGm[ptr1], cAligned);
                pipe_barrier(PIPE_ALL);

                Muls(PointGradHweightLocal, v1Local, hw, cAligned);
                gradHweightLocal = gradHweightLocal - PointGradHweightLocal;
                Muls(PointGradWweightLocal, v1Local, hh, cAligned);
                gradWweightLocal = gradWweightLocal - PointGradWweightLocal;

                Muls(PointGradHweightLocal, topGradMcMsFeatLocal, w1, cAligned);
                SetAtomicAdd<DTYPE_F>();
                DataCopyPad(gradMcMsFeatGm[ptr1], PointGradHweightLocal, copyParamsOut);
                SetAtomicNone();
            }

            Duplicate(v2Local, (DTYPE_F)0, cAligned);
            if (hLow >= 0 && wHigh <= w - 1) {
                uint64_t ptr2 = static_cast<uint64_t>(valueOffset) + hLowPtrOffset + wHighPtrOffset + chanenlOffset;
                DataCopy(v2Local, mcMsFeatGm[ptr2], cAligned);
                pipe_barrier(PIPE_ALL);

                Muls(PointGradHweightLocal, v2Local, lw, cAligned);
                gradHweightLocal = gradHweightLocal - PointGradHweightLocal;
                Muls(PointGradWweightLocal, v2Local, hh, cAligned);
                gradWweightLocal = gradWweightLocal + PointGradWweightLocal;

                Muls(PointGradHweightLocal, topGradMcMsFeatLocal, w2, cAligned);
                SetAtomicAdd<DTYPE_F>();
                DataCopyPad(gradMcMsFeatGm[ptr2], PointGradHweightLocal, copyParamsOut);
                SetAtomicNone();
            }

            Duplicate(v3Local, (DTYPE_F)0, cAligned);
            if (hHigh <= h - 1 && wLow >= 0) {
                uint64_t ptr3 = static_cast<uint64_t>(valueOffset) + hHighPtrOffset + wLowPtrOffset + chanenlOffset;
                DataCopy(v3Local, mcMsFeatGm[ptr3], cAligned);
                pipe_barrier(PIPE_ALL);

                Muls(PointGradHweightLocal, v3Local, hw, cAligned);
                gradHweightLocal = gradHweightLocal + PointGradHweightLocal;
                Muls(PointGradWweightLocal, v3Local, lh, cAligned);
                gradWweightLocal = gradWweightLocal - PointGradWweightLocal;

                Muls(PointGradHweightLocal, topGradMcMsFeatLocal, w3, cAligned);
                SetAtomicAdd<DTYPE_F>();
                DataCopyPad(gradMcMsFeatGm[ptr3], PointGradHweightLocal, copyParamsOut);
                SetAtomicNone();
            }

            Duplicate(v4Local, (DTYPE_F)0, cAligned);
            if (hHigh <= h - 1 && wHigh <= w - 1) {
                uint64_t ptr4 = static_cast<uint64_t>(valueOffset) + hHighPtrOffset + wHighPtrOffset + chanenlOffset;
                DataCopy(v4Local, mcMsFeatGm[ptr4], cAligned);
                pipe_barrier(PIPE_ALL);

                Muls(PointGradHweightLocal, v4Local, lw, cAligned);
                gradHweightLocal = gradHweightLocal + PointGradHweightLocal;
                Muls(PointGradWweightLocal, v4Local, lh, cAligned);
                gradWweightLocal = gradWweightLocal + PointGradWweightLocal;

                Muls(PointGradHweightLocal, topGradMcMsFeatLocal, w4, cAligned);
                SetAtomicAdd<DTYPE_F>();
                DataCopyPad(gradMcMsFeatGm[ptr4], PointGradHweightLocal, copyParamsOut);
                SetAtomicNone();
            }

            pipe_barrier(PIPE_ALL);

            Muls(v1Local, v1Local, w1, cAligned);
            Muls(v2Local, v2Local, w2, cAligned);
            Muls(v3Local, v3Local, w3, cAligned);
            Muls(v4Local, v4Local, w4, cAligned);

            Add(v1Local, v1Local, v2Local, cAligned);
            Add(v1Local, v1Local, v3Local, cAligned);
            Add(v1Local, v1Local, v4Local, cAligned);
            
            Mul(gradOutputLocal, v1Local, gradOutputLocal, cAligned);
            ReduceSum(gradOutputLocal, gradOutputLocal, gradOutputLocal, splitNum);

            SetAtomicAdd<DTYPE_F>();
            DataCopyPad(gradWeightsGm[weightIdx], gradOutputLocal, copyParamsOutOneNum);
            pipe_barrier(PIPE_ALL);
            SetAtomicNone();

            gradWweightLocal = gradWweightLocal * topGradMcMsFeatLocal;
            Muls(gradWweightLocal, gradWweightLocal, (float)w, cAligned);
            ReduceSum(gradWweightLocal, gradWweightLocal, gradWweightLocal, splitNum);

            SetAtomicAdd<DTYPE_F>();
            DataCopyPad(gradSamplingLocationGm[locationOffset], gradWweightLocal, copyParamsOutOneNum);
            pipe_barrier(PIPE_ALL);
            SetAtomicNone();

            gradHweightLocal = gradHweightLocal * topGradMcMsFeatLocal;
            Muls(gradHweightLocal, gradHweightLocal, (float)h, cAligned);
            ReduceSum(gradHweightLocal, gradHweightLocal, gradHweightLocal, splitNum);

            SetAtomicAdd<DTYPE_F>();
            DataCopyPad(gradSamplingLocationGm[locationOffset+1], gradHweightLocal, copyParamsOutOneNum);
            pipe_barrier(PIPE_ALL);
            SetAtomicNone();

            chanenlOffset += numEmbeds / numGroups;
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> inQueueFeat;
    TBuf<TPosition::VECCALC> inQueueWeights;
    TBuf<TPosition::VECCALC> inQueueLocation;
    TBuf<TPosition::VECCALC> inQueueScaleStart;
    TBuf<TPosition::VECCALC> inQueueSpatialShape;
    TBuf<TPosition::VECCALC> inQueueGradOutput;

    TBuf<TPosition::VECCALC> inQueueFloat;
    TBuf<TPosition::VECCALC> inQueueInt;

    TBuf<TPosition::VECCALC> inQueueV1;
    TBuf<TPosition::VECCALC> inQueueV2;
    TBuf<TPosition::VECCALC> inQueueV3;
    TBuf<TPosition::VECCALC> inQueueV4;

    TBuf<QuePosition::VECCALC> tmpTopGradMcMsFeat;
    TBuf<QuePosition::VECCALC> tmpTopGradMcMsFeatMulsWx;
    TBuf<QuePosition::VECCALC> tmpGradHweight;
    TBuf<QuePosition::VECCALC> tmpGradWweight;
    TBuf<QuePosition::VECCALC> tmpPointGradHweight;
    TBuf<QuePosition::VECCALC> tmpPointGradWweight;

    GlobalTensor<DTYPE_F> mcMsFeatGm;
    GlobalTensor<DTYPE_F> samplingLocationGm;
    GlobalTensor<DTYPE_F> weightsGm;
    GlobalTensor<DTYPE_F> gradOutputGm;
    GlobalTensor<DTYPE_F> gradMcMsFeatGm;
    GlobalTensor<DTYPE_F> gradSamplingLocationGm;
    GlobalTensor<DTYPE_F> gradWeightsGm;

    GlobalTensor<DTYPE_I> spatialShapeGm;
    GlobalTensor<DTYPE_I> scaleStartIndexGm;

    uint64_t mcMsFeatGmLength;
    uint64_t spatialShapeGmLength;
    uint64_t scaleStartIndexLength;
    uint64_t samplingLocationGmLength;
    uint64_t weightsGmLength;
    uint64_t gradOutputGmLength;

    uint64_t gradMcMsFeatGmLength;
    uint64_t gradSamplingLocationGmLength;
    uint64_t gradWeightsGmLength;

    uint32_t batchSize;
    uint32_t numFeat;
    uint32_t numEmbeds;
    uint32_t numAnchors;
    uint32_t numPoints;
    uint32_t numCams;
    uint32_t numScale;
    uint32_t numGroups;
    uint32_t splitNum;

    uint32_t cAligned;
    uint32_t singleAligned;

    uint32_t average;
    uint32_t taskLast;
    uint32_t usedCoreNum;

    DataCopyExtParams copyParamsOut;
    DataCopyExtParams copyParamsOutOneNum;
};

extern "C" __global__ __aicore__ void deformable_aggregation_grad(
    GM_ADDR mc_ms_feat,
    GM_ADDR spatial_shape,
    GM_ADDR scale_start_index,
    GM_ADDR sampling_location,
    GM_ADDR weights,
    GM_ADDR grad_output,
    GM_ADDR grad_mc_ms_feat,
    GM_ADDR grad_sampling_location,
    GM_ADDR grad_weights,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelDeformableAggregationGrad<float, int32_t> op;
    op.Init(
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
        grad_output,
        grad_mc_ms_feat,
        grad_sampling_location,
        grad_weights,
        &tiling_data
    );
    op.Process();
}
