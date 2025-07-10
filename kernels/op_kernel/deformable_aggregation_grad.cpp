/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

template<typename DTYPE_F>
class KernelDeformableAggregationGrad {
public:
    __aicore__ inline KernelDeformableAggregationGrad() = delete;

    __aicore__ inline KernelDeformableAggregationGrad(
        GM_ADDR mc_ms_feat,
        GM_ADDR spatial_shape,
        GM_ADDR scale_start_index,
        GM_ADDR sampling_location,
        GM_ADDR weights,
        GM_ADDR grad_output,
        GM_ADDR grad_mc_ms_feat,
        GM_ADDR grad_sampling_location,
        GM_ADDR grad_weights,
        const DeformableAggregationGradTilingData& tiling_data,
        TPipe* pipe)
        : pipe_(pipe)
    {
        InitTask(tiling_data);
        InitGM(mc_ms_feat, spatial_shape, scale_start_index,
            sampling_location, weights, grad_output,
            grad_mc_ms_feat, grad_sampling_location, grad_weights);
        InitBuffer();
    }

    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTask(const DeformableAggregationGradTilingData& tiling)
    {
        usedCoreNum_ = tiling.usedCoreNum;
        avgWeightNum_ = tiling.avgWeightNum;
        tailWeightNum_ = tiling.tailWeightNum;
        coreId = GetBlockIdx();
        taskOffset = coreId * avgWeightNum_;
        totalTaskNum_ = avgWeightNum_;
        if (coreId == usedCoreNum_ - 1) {
            totalTaskNum_ = tailWeightNum_;
        }
        singleProcessTaskLen_ = min(tiling.singleProcessTaskLen, totalTaskNum_);
        singleProcessTaskLen_ = max(singleProcessTaskLen_, (uint32_t)1);
        taskRepeatTimes = (totalTaskNum_ - 1) / singleProcessTaskLen_ + 1;
        pts_ = tiling.numPoints;
        cam_  = tiling.numCams;
        scale_ = tiling.numScale;
        group_ = tiling.numGroups;
        numEmbeds = tiling.numEmbeds;
        numFeat = tiling.numFeat;
        numAnchors = tiling.numAnchors;
        totalGroups = numEmbeds / group_;

        blockSize_ = 32;
        blockDataNum_ = blockSize_ / sizeof(DTYPE_F);
    }

    __aicore__ inline void InitGM(GM_ADDR mc_ms_feat, GM_ADDR spatial_shape, GM_ADDR scale_start_index,
                                GM_ADDR sampling_location, GM_ADDR weights, GM_ADDR grad_output,
                                GM_ADDR grad_mc_ms_feat, GM_ADDR grad_sampling_location, GM_ADDR grad_weights)
    {
        int64_t samplingLocationOffset = taskOffset * pts_ * cam_ * 2;
        int64_t weightOffset = taskOffset * pts_ * cam_ * scale_ * group_;
        mcMsFeatGm.SetGlobalBuffer((__gm__ DTYPE_F*)(mc_ms_feat));
        spatialShapeGm.SetGlobalBuffer((__gm__ int32_t*)(spatial_shape));
        scaleStartLocationGm.SetGlobalBuffer((__gm__ int32_t*)(scale_start_index));
        samplingLocationGm.SetGlobalBuffer((__gm__ DTYPE_F*)(sampling_location) + samplingLocationOffset);
        weightGm.SetGlobalBuffer((__gm__ DTYPE_F*)(weights) + weightOffset);
        outputGradGm.SetGlobalBuffer((__gm__ DTYPE_F*)(grad_output) + taskOffset * numEmbeds);
        gradMcMsFeatGm.SetGlobalBuffer((__gm__ DTYPE_F*)(grad_mc_ms_feat));
        gradSamplingLocalGm.SetGlobalBuffer((__gm__ DTYPE_F*)(grad_sampling_location) + samplingLocationOffset);
        gradWeightsGm.SetGlobalBuffer((__gm__ DTYPE_F*)(grad_weights) + weightOffset);
    }

    __aicore__ inline void InitBuffer()
    {
        uint64_t singleWeightOffset = scale_ * group_;
        uint64_t samplingOffset = pts_ * cam_ * 2;
        pipe_->InitBuffer(weightQue_, AlignUp(singleWeightOffset, blockDataNum_) * sizeof(DTYPE_F));
        pipe_->InitBuffer(gradOutputQue_, singleProcessTaskLen_ * numEmbeds * sizeof(DTYPE_F));
        pipe_->InitBuffer(scaleStartLocationQue_, AlignUp(cam_ * scale_, B32_DATA_NUM_PER_BLOCK) * sizeof(int32_t));
        pipe_->InitBuffer(samplingLocationQue_, AlignUp(samplingOffset, blockDataNum_) * sizeof(DTYPE_F));
        pipe_->InitBuffer(spatialShapeQue_, AlignUp(cam_ * scale_ * 2, B32_DATA_NUM_PER_BLOCK) * sizeof(int32_t));
        pipe_->InitBuffer(topGradMcMsFeatQue_, numEmbeds * sizeof(DTYPE_F));
        pipe_->InitBuffer(gradValueQue_, 4 * numEmbeds * sizeof(DTYPE_F));
        pipe_->InitBuffer(vQue_, 4 * numEmbeds * sizeof(DTYPE_F));
        pipe_->InitBuffer(featureQue_, scale_ * numEmbeds * sizeof(DTYPE_F));
        pipe_->InitBuffer(gradWeightsQue_, scale_ * group_ * sizeof(DTYPE_F));
        pipe_->InitBuffer(pointGradWeightQue_, 4 * numEmbeds * sizeof(DTYPE_F));
        pipe_->InitBuffer(gradSamplingQue_, blockDataNum_ * sizeof(DTYPE_F));
        pipe_->InitBuffer(pointGradQue_, 2 * numEmbeds * sizeof(DTYPE_F));
        pipe_->InitBuffer(weightBrobQue_, scale_ * numEmbeds * sizeof(DTYPE_F));
    }

    __aicore__ inline void Prepare()
    {
        int32_t scaleStartNum = AlignUp(cam_ * scale_, B32_DATA_NUM_PER_BLOCK);
        int32_t spatialShapeNum = AlignUp(cam_ * scale_ * 2, B32_DATA_NUM_PER_BLOCK);
        scaleStartLocation = scaleStartLocationQue_.Get<int32_t>();
        spatialShape = spatialShapeQue_.Get<int32_t>();
        weight = weightQue_.Get<DTYPE_F>();
        gradOutput = gradOutputQue_.Get<DTYPE_F>();
        samplingLocation = samplingLocationQue_.Get<DTYPE_F>();

        gradWeightsLocal = gradWeightsQue_.Get<DTYPE_F>();
        gradSamplingLocal = gradSamplingQue_.Get<DTYPE_F>();
        gradValueLocal = gradValueQue_.Get<DTYPE_F>();

        topGradMcMsFeatLocal = topGradMcMsFeatQue_.Get<DTYPE_F>();
        vLocal = vQue_.Get<DTYPE_F>();
        featureLocal = featureQue_.Get<DTYPE_F>();
        pointGradWeightLocal = pointGradWeightQue_.Get<DTYPE_F>();
        pointGradSum = pointGradQue_.Get<DTYPE_F>();
        weightBrobLocal = weightBrobQue_.Get<DTYPE_F>();

        Duplicate(pointGradSum, (DTYPE_F)0, 2 * numEmbeds);
        Duplicate(featureLocal, (DTYPE_F)0, scale_ * numEmbeds);
        Duplicate(vLocal, (DTYPE_F)0, numEmbeds * 4);

        DataCopy(scaleStartLocation, scaleStartLocationGm, scaleStartNum);
        DataCopy(spatialShape, spatialShapeGm, spatialShapeNum);
    }

    __aicore__ inline void ProcessSingle(uint64_t taskIdx, uint32_t actualWeightNum)
    {
        uint64_t singleWeightOffset = scale_ * group_;
        uint32_t weightCopyLen = AlignUp(singleWeightOffset, blockDataNum_);
        int32_t gradOuputNum = AlignUp(actualWeightNum * numEmbeds, blockDataNum_);
        int32_t samplingLocationNum = AlignUp(pts_ * cam_ * 2, blockDataNum_);
        uint64_t gradOutputOffset = taskIdx * singleProcessTaskLen_ * numEmbeds;

        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        DataCopy(gradOutput, outputGradGm[gradOutputOffset], gradOuputNum);

        for (int32_t weightNumId = 0; weightNumId < actualWeightNum; weightNumId++) {
            int64_t curBatch = (taskOffset + taskIdx * singleProcessTaskLen_ + weightNumId)  / numAnchors;
            int64_t featOffset = curBatch * numFeat * numEmbeds;
            uint64_t samplingLocationOffset = (taskIdx * singleProcessTaskLen_ + weightNumId) * pts_ * cam_ * 2;
            DataCopy(samplingLocation, samplingLocationGm[samplingLocationOffset], samplingLocationNum);
            for (int32_t ptsId = 0; ptsId < pts_; ptsId++) {
                for (int32_t camId = 0; camId < cam_; camId++) {
                    int32_t locOffset = ptsId * cam_ + camId;
                    float locW = samplingLocation.GetValue(locOffset * 2);
                    float locH = samplingLocation.GetValue(locOffset * 2 + 1);
                    if (locW <= 0 || locW >= 1 ||  locH <=0 || locH >=1) {
                        continue;
                    }
                    uint64_t weightGmOffset = (((taskIdx * singleProcessTaskLen_ + weightNumId) * pts_ + ptsId) * cam_ + camId) * singleWeightOffset;
                    uint64_t samplingLocationCopyOutOffset = samplingLocationOffset + (ptsId * cam_ + camId) * 2;
                    DataCopy(weight, weightGm[weightGmOffset], weightCopyLen);
                    SetFlag<HardEvent::MTE2_V>(0);
                    WaitFlag<HardEvent::MTE2_V>(0);
                    uint32_t dstShape_[2] = {scale_ * group_, totalGroups};
                    uint32_t srcShape_[2] = {scale_ * group_, 1};
                    BroadCast<DTYPE_F, 2, 1>(weightBrobLocal, weight, dstShape_, srcShape_);
                    SetFlag<HardEvent::V_MTE2>(0);
                    WaitFlag<HardEvent::V_MTE2>(0);
                    for (int32_t scaleId = 0; scaleId < scale_; scaleId++) {
                        int32_t scaleStartOffset = camId * scale_ + scaleId;
                        int32_t scaleStartIdx = scaleStartLocation.GetValue(scaleStartOffset);
                        int64_t featureOffset = (int64_t)scaleStartIdx * numEmbeds;
                        int32_t h =  spatialShape.GetValue(scaleStartOffset * 2);
                        int32_t w =  spatialShape.GetValue(scaleStartOffset * 2 + 1);
                        float hIm = locH * h - (float)0.5;
                        float wIm = locW * w - (float)0.5;
                        int32_t hLow = ScalarCast<float, int32_t, AscendC::RoundMode::CAST_FLOOR>(hIm);
                        int32_t wLow = ScalarCast<float, int32_t, AscendC::RoundMode::CAST_FLOOR>(wIm);
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
                        uint64_t ptr1 = featureOffset + hLowPtrOffset + wLowPtrOffset;
                        uint64_t ptr2 = featureOffset + hLowPtrOffset + wHighPtrOffset;
                        uint64_t ptr3 = featureOffset + hHighPtrOffset + wLowPtrOffset;
                        uint64_t ptr4 = featureOffset + hHighPtrOffset + wHighPtrOffset;

                        uint64_t weightOffset = scaleId * numEmbeds;
                        uint64_t gradOuputBaseOffset = weightNumId * numEmbeds;

                        SetFlag<HardEvent::MTE3_V>(0);
                        WaitFlag<HardEvent::MTE3_V>(0);

                        Mul(topGradMcMsFeatLocal, weightBrobLocal[weightOffset], gradOutput[gradOuputBaseOffset], numEmbeds);
                        Muls(gradValueLocal, topGradMcMsFeatLocal, static_cast<DTYPE_F>(w1), numEmbeds);
                        Muls(gradValueLocal[numEmbeds * 1], topGradMcMsFeatLocal, static_cast<DTYPE_F>(w2), numEmbeds);
                        Muls(gradValueLocal[numEmbeds * 2], topGradMcMsFeatLocal, static_cast<DTYPE_F>(w3), numEmbeds);
                        Muls(gradValueLocal[numEmbeds * 3], topGradMcMsFeatLocal, static_cast<DTYPE_F>(w4), numEmbeds);

                        SetFlag<HardEvent::V_MTE3>(0);
                        WaitFlag<HardEvent::V_MTE3>(0);

                        SetAtomicAdd<DTYPE_F>();
                        if (hLow >= 0 && wLow >=0) {
                            DataCopy(gradMcMsFeatGm[featOffset + ptr1], gradValueLocal, numEmbeds);
                            DataCopy(vLocal, mcMsFeatGm[featOffset + ptr1], numEmbeds);
                        }
                        if (hLow >= 0 && wHigh <= w - 1) {
                            DataCopy(gradMcMsFeatGm[featOffset + ptr2], gradValueLocal[numEmbeds * 1], numEmbeds);
                            DataCopy(vLocal[numEmbeds], mcMsFeatGm[featOffset + ptr2], numEmbeds);
                        }
                        if (hHigh <= h - 1 && wLow >= 0) {
                            DataCopy(gradMcMsFeatGm[featOffset + ptr3], gradValueLocal[numEmbeds * 2], numEmbeds);
                            DataCopy(vLocal[numEmbeds * 2], mcMsFeatGm[featOffset + ptr3], numEmbeds);
                        }
                        if (hHigh <= h - 1 && wHigh <= w - 1) {
                            DataCopy(gradMcMsFeatGm[featOffset + ptr4], gradValueLocal[numEmbeds * 3], numEmbeds);
                            DataCopy(vLocal[numEmbeds * 3], mcMsFeatGm[featOffset + ptr4], numEmbeds);
                        }
                        SetAtomicNone();

                        SetFlag<HardEvent::MTE2_V>(0);
                        WaitFlag<HardEvent::MTE2_V>(0);

                        Muls(featureLocal[weightOffset], vLocal, static_cast<DTYPE_F>(w1), numEmbeds);
                        Axpy(featureLocal[weightOffset], vLocal[numEmbeds], static_cast<DTYPE_F>(w2), numEmbeds);
                        Axpy(featureLocal[weightOffset], vLocal[numEmbeds * 2], static_cast<DTYPE_F>(w3), numEmbeds);
                        Axpy(featureLocal[weightOffset], vLocal[numEmbeds * 3], static_cast<DTYPE_F>(w4), numEmbeds);
                        Mul(featureLocal[weightOffset], featureLocal[weightOffset], gradOutput[gradOuputBaseOffset], numEmbeds);

                        Sub(pointGradWeightLocal, vLocal[numEmbeds * 1], vLocal, numEmbeds);
                        Sub(pointGradWeightLocal[numEmbeds * 2], vLocal[numEmbeds * 3], vLocal[numEmbeds * 2], numEmbeds);

                        Sub(pointGradWeightLocal[numEmbeds * 1], vLocal[numEmbeds * 2], vLocal, numEmbeds);
                        Sub(pointGradWeightLocal[numEmbeds * 3], vLocal[numEmbeds * 3], vLocal[numEmbeds * 1], numEmbeds);
                        Duplicate(vLocal, (DTYPE_F)0, numEmbeds * 4);

                        SetFlag<HardEvent::V_MTE2>(0);
                        WaitFlag<HardEvent::V_MTE2>(0);

                        Muls(pointGradWeightLocal, pointGradWeightLocal, static_cast<DTYPE_F>(hh), numEmbeds);
                        Axpy(pointGradWeightLocal, pointGradWeightLocal[numEmbeds * 2], static_cast<DTYPE_F>(lh), numEmbeds);

                        Muls(pointGradWeightLocal[numEmbeds * 1], pointGradWeightLocal[numEmbeds * 1], static_cast<DTYPE_F>(hw), numEmbeds);
                        Axpy(pointGradWeightLocal[numEmbeds * 1], pointGradWeightLocal[numEmbeds * 3], static_cast<DTYPE_F>(lw), numEmbeds);

                        Mul(pointGradWeightLocal, pointGradWeightLocal, topGradMcMsFeatLocal, numEmbeds);
                        Mul(pointGradWeightLocal[numEmbeds], pointGradWeightLocal[numEmbeds], topGradMcMsFeatLocal, numEmbeds);
                        Muls(pointGradWeightLocal, pointGradWeightLocal, (DTYPE_F)w, numEmbeds);
                        Muls(pointGradWeightLocal[numEmbeds], pointGradWeightLocal[numEmbeds], (DTYPE_F)h, numEmbeds);

                        Add(pointGradSum, pointGradSum, pointGradWeightLocal, numEmbeds * 2);
                    }
                    SetFlag<HardEvent::MTE3_V>(0);
                    WaitFlag<HardEvent::MTE3_V>(0);
                    Sum(gradWeightsLocal, featureLocal, {scale_ * group_, totalGroups, totalGroups});
                    Sum(gradSamplingLocal, pointGradSum, {2, numEmbeds, numEmbeds});
                    SetFlag<HardEvent::V_MTE3>(0);
                    WaitFlag<HardEvent::V_MTE3>(0);
                    Duplicate(featureLocal, (DTYPE_F)0, scale_ * numEmbeds);
                    Duplicate(pointGradSum, (DTYPE_F)0, 2 * numEmbeds);
                    DataCopyExtParams locationCopyParams {1, (uint32_t)(2 * sizeof(DTYPE_F)), 0, 0, 0};
                    DataCopyExtParams weightsCopyParams {1, (uint32_t)(scale_ * group_ * sizeof(DTYPE_F)), 0, 0, 0};
                    DataCopyPad(gradSamplingLocalGm[samplingLocationCopyOutOffset], gradSamplingLocal, locationCopyParams);
                    DataCopyPad(gradWeightsGm[weightGmOffset], gradWeightsLocal, weightsCopyParams);
                }
            }
        }
    }

private:
    TPipe* pipe_;
    GlobalTensor<DTYPE_F> mcMsFeatGm, samplingLocationGm, weightGm, outputGradGm;
    GlobalTensor<DTYPE_F> gradMcMsFeatGm, gradSamplingLocalGm, gradWeightsGm;
    GlobalTensor<int32_t> spatialShapeGm, scaleStartLocationGm;
    TBuf<TPosition::VECCALC> weightQue_, gradOutputQue_, samplingLocationQue_, scaleStartLocationQue_, spatialShapeQue_;
    TBuf<TPosition::VECCALC> gradWeightsQue_, gradSamplingQue_, gradValueQue_;
    TBuf<TPosition::VECCALC> topGradMcMsFeatQue_, vQue_, featureQue_, pointGradWeightQue_, pointGradQue_, weightBrobQue_;
    LocalTensor<int32_t> scaleStartLocation, spatialShape;
    LocalTensor<DTYPE_F> weight, gradOutput, samplingLocation;
    LocalTensor<DTYPE_F> gradWeightsLocal, gradSamplingLocal, gradValueLocal;
    LocalTensor<DTYPE_F> topGradMcMsFeatLocal, vLocal, featureLocal, pointGradWeightLocal, pointGradSum, weightBrobLocal;
    uint32_t usedCoreNum_, avgWeightNum_, tailWeightNum_, coreId;
    uint32_t totalTaskNum_, singleProcessTaskLen_, taskRepeatTimes;
    uint32_t pts_, cam_, scale_, group_, numEmbeds, numFeat, numAnchors, totalGroups;
    uint32_t blockSize_, blockDataNum_;
    int64_t taskOffset;
};

template<typename DTYPE_F>
__aicore__ inline void KernelDeformableAggregationGrad<DTYPE_F>::Process()
{
    Prepare();
    for (uint32_t i = 0; i < taskRepeatTimes; ++i) {
        uint32_t actualWeightNum = singleProcessTaskLen_;
        if (unlikely(i == taskRepeatTimes - 1)) {
            actualWeightNum = (totalTaskNum_ - 1) % singleProcessTaskLen_ + 1;
        }
        ProcessSingle(i, actualWeightNum);
    }
}

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
    TPipe pipe;
    KernelDeformableAggregationGrad<DTYPE_MC_MS_FEAT> op(
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
        grad_output,
        grad_mc_ms_feat,
        grad_sampling_location,
        grad_weights,
        tiling_data,
        &pipe
    );
    op.Process();
}
