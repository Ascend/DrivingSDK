/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

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
        InitEvent();
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
        singleProcessTaskLen_ = tiling.singleProcessTaskLen;
        taskRepeatTimes = (totalTaskNum_ - 1) / singleProcessTaskLen_ + 1;
        pts_ = tiling.numPoints;
        cam_  = tiling.numCams;
        scale_ = tiling.numScale;
        group_ = tiling.numGroups;
        numEmbeds = tiling.numEmbeds;
        numFeat = tiling.numFeat;
        numAnchors = tiling.numAnchors;
        totalGroups = numEmbeds / group_;
    }

    __aicore__ inline void InitGM(GM_ADDR mc_ms_feat, GM_ADDR spatial_shape, GM_ADDR scale_start_index,
                                GM_ADDR sampling_location, GM_ADDR weights, GM_ADDR grad_output,
                                GM_ADDR grad_mc_ms_feat, GM_ADDR grad_sampling_location, GM_ADDR grad_weights)
    {
        int64_t samplingLocationOffset = taskOffset * pts_ * cam_ * 2;
        int64_t weightOffset = taskOffset * pts_ * cam_ * scale_ * group_;
        mcMsFeatGm.SetGlobalBuffer((__gm__ float*)(mc_ms_feat));
        spatialShapeGm.SetGlobalBuffer((__gm__ int32_t*)(spatial_shape));
        scaleStartLocationGm.SetGlobalBuffer((__gm__ int32_t*)(scale_start_index));
        samplingLocationGm.SetGlobalBuffer((__gm__ float*)(sampling_location) + samplingLocationOffset);
        weightGm.SetGlobalBuffer((__gm__ float*)(weights) + weightOffset);
        outputGradGm.SetGlobalBuffer((__gm__ float*)(grad_output) + taskOffset * numEmbeds);
        gradMcMsFeatGm.SetGlobalBuffer((__gm__ float*)(grad_mc_ms_feat));
        gradSamplingLocationGm.SetGlobalBuffer((__gm__ float*)(grad_sampling_location) + samplingLocationOffset * 4);
        gradWeightsGm.SetGlobalBuffer((__gm__ float*)(grad_weights) + weightOffset);
    }

    __aicore__ inline void InitBuffer()
    {
        uint64_t singleWeightOffset = cam_ * scale_ * group_;
        uint64_t samplingOffset = pts_ * cam_ * 2;
        pipe_->InitBuffer(weightQue_, AlignUp(singleWeightOffset, B32_DATA_NUM_PER_BLOCK) * sizeof(float));
        pipe_->InitBuffer(gradOutputQue_, singleProcessTaskLen_ * numEmbeds * sizeof(float));
        pipe_->InitBuffer(scaleStartLocationQue_, AlignUp(cam_ * scale_, B32_DATA_NUM_PER_BLOCK) * sizeof(int32_t));
        pipe_->InitBuffer(samplingLocationQue_, AlignUp(samplingOffset, B32_DATA_NUM_PER_BLOCK) * sizeof(float));
        pipe_->InitBuffer(spatialShapeQue_, AlignUp(cam_ * scale_* 2, B32_DATA_NUM_PER_BLOCK) * sizeof(int32_t));
        pipe_->InitBuffer(topGradMcMsFeatQue_, 5 * numEmbeds * sizeof(float));
        pipe_->InitBuffer(vQue_, 4 * numEmbeds * sizeof(float));
        pipe_->InitBuffer(featureQue_, 4 * numEmbeds * sizeof(float));
        pipe_->InitBuffer(featureQue__, numEmbeds * sizeof(float));
        pipe_->InitBuffer(pointGradWeightQue_, 8 * numEmbeds * sizeof(float));
        pipe_->InitBuffer(gradSamplingQue_, 4 * samplingOffset * sizeof(float));
        pipe_->InitBuffer(sumTmp_, 8 * sizeof(float));
    }

    __aicore__ inline void InitEvent()
    {
        cpInEvtID_ = pipe_->FetchEventID(HardEvent::MTE2_V);
        cpOutEvtID_ = pipe_->FetchEventID(HardEvent::MTE3_MTE2);
        vToOutEvtID_ = pipe_->FetchEventID(HardEvent::V_MTE3);
        vToMTE2EvtID_ = pipe_->FetchEventID(HardEvent::V_MTE2);
        mte3ToVEvtID_ = pipe_->FetchEventID(HardEvent::MTE3_V);
    }

    __aicore__ inline void ProcessSingle(uint64_t taskIdx, uint32_t actualWeightNum)
    {
        uint64_t singleWeightOffset = cam_ * scale_ * group_;
        uint32_t weightCopyLen = AlignUp(singleWeightOffset, B32_DATA_NUM_PER_BLOCK);
        int32_t gradOuputNum = AlignUp(actualWeightNum * numEmbeds, B32_DATA_NUM_PER_BLOCK);
        int32_t samplingLocationNum = AlignUp(pts_ * cam_ * 2, B32_DATA_NUM_PER_BLOCK);
        int32_t scaleStartNum = AlignUp(cam_ * scale_, B32_DATA_NUM_PER_BLOCK);
        int32_t spatialShapeNum = AlignUp(cam_ * scale_ * 2, B32_DATA_NUM_PER_BLOCK);
        uint64_t gradOutputOffset = taskIdx * singleProcessTaskLen_ * numEmbeds;

        LocalTensor<float> weight = weightQue_.Get<float>();
        LocalTensor<float> gradOutput = gradOutputQue_.Get<float>();
        LocalTensor<float> samplingLocation = samplingLocationQue_.Get<float>();
        LocalTensor<int32_t> scaleStartLocation = scaleStartLocationQue_.Get<int32_t>();
        LocalTensor<int32_t> spatialShape = spatialShapeQue_.Get<int32_t>();

        LocalTensor<float> topGradMcMsFeatLocal = topGradMcMsFeatQue_.Get<float>();
        LocalTensor<float> vLocal = vQue_.Get<float>();
        LocalTensor<float> featureLocal = featureQue_.Get<float>();
        LocalTensor<float> featureLocal_ = featureQue__.Get<float>();
        LocalTensor<float> pointGradWeightLocal = pointGradWeightQue_.Get<float>();
        LocalTensor<float> gradSamplingLocation = gradSamplingQue_.Get<float>();
        LocalTensor<float> tmpLocation = sumTmp_.Get<float>();

        SetFlag<HardEvent::V_MTE2>(vToMTE2EvtID_);
        WaitFlag<HardEvent::V_MTE2>(vToMTE2EvtID_);
        DataCopy(gradOutput, outputGradGm[gradOutputOffset], gradOuputNum);
        
        DataCopy(scaleStartLocation, scaleStartLocationGm, scaleStartNum);
        DataCopy(spatialShape, spatialShapeGm, spatialShapeNum);
        for (int32_t weightNumId = 0; weightNumId < actualWeightNum; weightNumId++) {
            int64_t curBatch = (taskOffset + taskIdx * singleProcessTaskLen_ + weightNumId)  / numAnchors;
            int64_t featOffset = curBatch * numFeat * numEmbeds;
            uint64_t samplingLocationOffset = (taskIdx * singleProcessTaskLen_ + weightNumId) * pts_ * cam_ * 2;
            DataCopy(samplingLocation, samplingLocationGm[samplingLocationOffset], samplingLocationNum);
            SetFlag<HardEvent::MTE3_V>(mte3ToVEvtID_);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVEvtID_);
            Duplicate(gradSamplingLocation, (float)0, pts_ * cam_ * 8);
            for (int32_t ptsId = 0; ptsId < pts_; ptsId++) {
                uint64_t weightGmOffset = ((taskIdx * singleProcessTaskLen_ + weightNumId) * pts_ + ptsId) * singleWeightOffset;
                SetFlag<HardEvent::V_MTE2>(vToMTE2EvtID_);
                WaitFlag<HardEvent::V_MTE2>(vToMTE2EvtID_);
                DataCopy(weight, weightGm[weightGmOffset], weightCopyLen);
                SetFlag<HardEvent::MTE2_V>(cpInEvtID_);
                WaitFlag<HardEvent::MTE2_V>(cpInEvtID_);
                for (int32_t camId = 0; camId < cam_; camId++) {
                    int32_t locOffset = ptsId * cam_ + camId;
                    float locW = samplingLocation.GetValue(locOffset * 2);
                    float locH = samplingLocation.GetValue(locOffset * 2 + 1);
                    if (locW <= 0 || locW >= 1 ||  locH <=0 || locH >=1) {
                        continue;
                    }
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

                        uint64_t weightOffset = (camId * scale_ + scaleId) * group_;
                        uint64_t gradOuputBaseOffset = weightNumId * numEmbeds;
                        uint32_t dstShape_[2] = {group_, totalGroups};
                        uint32_t srcShape_[2] = {group_, 1};

                        Duplicate(vLocal, (float)0, numEmbeds * 4);

                        SetFlag<HardEvent::V_MTE2>(vToMTE2EvtID_);
                        WaitFlag<HardEvent::V_MTE2>(vToMTE2EvtID_);

                        SetFlag<HardEvent::MTE3_V>(mte3ToVEvtID_);
                        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvtID_);

                        BroadCast<float, 2, 1>(topGradMcMsFeatLocal, weight[weightOffset], dstShape_, srcShape_);
                        Mul(topGradMcMsFeatLocal, topGradMcMsFeatLocal, gradOutput[gradOuputBaseOffset], numEmbeds);
                        Muls(topGradMcMsFeatLocal[numEmbeds], topGradMcMsFeatLocal, w1, numEmbeds);
                        Muls(topGradMcMsFeatLocal[numEmbeds * 2], topGradMcMsFeatLocal, w2, numEmbeds);
                        Muls(topGradMcMsFeatLocal[numEmbeds * 3], topGradMcMsFeatLocal, w3, numEmbeds);
                        Muls(topGradMcMsFeatLocal[numEmbeds * 4], topGradMcMsFeatLocal, w4, numEmbeds);

                        SetFlag<HardEvent::V_MTE3>(vToOutEvtID_);
                        WaitFlag<HardEvent::V_MTE3>(vToOutEvtID_);

                        SetAtomicAdd<float>();
                        if (hLow >= 0 && wLow >=0) {
                            DataCopy(gradMcMsFeatGm[featOffset + ptr1], topGradMcMsFeatLocal[numEmbeds], numEmbeds);
                            DataCopy(vLocal, mcMsFeatGm[featOffset + ptr1], numEmbeds);
                        }
                        if (hLow >= 0 && wHigh <= w - 1) {
                            DataCopy(gradMcMsFeatGm[featOffset + ptr2], topGradMcMsFeatLocal[numEmbeds * 2], numEmbeds);
                            DataCopy(vLocal[numEmbeds], mcMsFeatGm[featOffset + ptr2], numEmbeds);
                        }
                        if (hHigh <= h - 1 && wLow >= 0) {
                            DataCopy(gradMcMsFeatGm[featOffset + ptr3], topGradMcMsFeatLocal[numEmbeds * 3], numEmbeds);
                            DataCopy(vLocal[numEmbeds * 2], mcMsFeatGm[featOffset + ptr3], numEmbeds);
                        }
                        if (hHigh <= h - 1 && wHigh <= w - 1) {
                            DataCopy(gradMcMsFeatGm[featOffset + ptr4], topGradMcMsFeatLocal[numEmbeds * 4], numEmbeds);
                            DataCopy(vLocal[numEmbeds * 3], mcMsFeatGm[featOffset + ptr4], numEmbeds);
                        }
                        SetAtomicNone();

                        SetFlag<HardEvent::MTE2_V>(cpInEvtID_);
                        WaitFlag<HardEvent::MTE2_V>(cpInEvtID_);

                        Muls(featureLocal, vLocal, w1, numEmbeds);
                        Muls(featureLocal[numEmbeds], vLocal[numEmbeds], w2, numEmbeds);
                        Muls(featureLocal[numEmbeds * 2], vLocal[numEmbeds * 2], w3, numEmbeds);
                        Muls(featureLocal[numEmbeds * 3], vLocal[numEmbeds * 3], w4, numEmbeds);
                        Add(featureLocal, featureLocal, featureLocal[numEmbeds], numEmbeds);
                        Add(featureLocal[numEmbeds * 2], featureLocal[numEmbeds * 2], featureLocal[numEmbeds * 3], numEmbeds);
                        Add(featureLocal, featureLocal, featureLocal[numEmbeds * 2], numEmbeds);
                        Mul(featureLocal, featureLocal, gradOutput[gradOuputBaseOffset], numEmbeds);

                        SetFlag<HardEvent::MTE3_V>(mte3ToVEvtID_);
                        WaitFlag<HardEvent::MTE3_V>(mte3ToVEvtID_);

                        Sum(featureLocal_, featureLocal, {group_, totalGroups, totalGroups});

                        SetFlag<HardEvent::V_MTE3>(vToOutEvtID_);
                        WaitFlag<HardEvent::V_MTE3>(vToOutEvtID_);

                        SetAtomicAdd<float>();
                        DataCopy(gradWeightsGm[weightGmOffset + weightOffset], featureLocal_, group_);
                        SetAtomicNone();

                        Muls(pointGradWeightLocal, vLocal, hw, numEmbeds);
                        Muls(pointGradWeightLocal[numEmbeds * 2], vLocal[numEmbeds], lw, numEmbeds);
                        Muls(pointGradWeightLocal[numEmbeds * 4], vLocal[numEmbeds * 2], hw, numEmbeds);
                        Muls(pointGradWeightLocal[numEmbeds * 6], vLocal[numEmbeds * 3], lw, numEmbeds);
                        Muls(pointGradWeightLocal[numEmbeds], vLocal, hh, numEmbeds);
                        Muls(pointGradWeightLocal[numEmbeds * 3], vLocal[numEmbeds], hh, numEmbeds);
                        Muls(pointGradWeightLocal[numEmbeds * 5], vLocal[numEmbeds * 2], lh, numEmbeds);
                        Muls(pointGradWeightLocal[numEmbeds * 7], vLocal[numEmbeds * 3], lh, numEmbeds);
                        Sub(pointGradWeightLocal[numEmbeds * 4], pointGradWeightLocal[numEmbeds * 4], pointGradWeightLocal, numEmbeds);
                        Sub(pointGradWeightLocal[numEmbeds * 6], pointGradWeightLocal[numEmbeds * 6], pointGradWeightLocal[numEmbeds * 2], numEmbeds);
                        Sub(pointGradWeightLocal[numEmbeds * 3], pointGradWeightLocal[numEmbeds * 3], pointGradWeightLocal[numEmbeds], numEmbeds);
                        Sub(pointGradWeightLocal[numEmbeds * 7], pointGradWeightLocal[numEmbeds * 7], pointGradWeightLocal[numEmbeds * 5], numEmbeds);
                        Add(pointGradWeightLocal[numEmbeds], pointGradWeightLocal[numEmbeds * 4], pointGradWeightLocal[numEmbeds * 6], numEmbeds);
                        Add(pointGradWeightLocal, pointGradWeightLocal[numEmbeds * 3], pointGradWeightLocal[numEmbeds * 7], numEmbeds);
                        Mul(pointGradWeightLocal, pointGradWeightLocal, topGradMcMsFeatLocal, numEmbeds);
                        Mul(pointGradWeightLocal[numEmbeds], pointGradWeightLocal[numEmbeds], topGradMcMsFeatLocal, numEmbeds);
                        Muls(pointGradWeightLocal, pointGradWeightLocal, (float)w, numEmbeds);
                        Muls(pointGradWeightLocal[numEmbeds], pointGradWeightLocal[numEmbeds], (float)h, numEmbeds);
                        Sum(tmpLocation, pointGradWeightLocal, {2, numEmbeds, numEmbeds});
                        Add(gradSamplingLocation[locOffset * 8], gradSamplingLocation[locOffset * 8], tmpLocation, 8);
                    }
                }
            }
            SetFlag<HardEvent::V_MTE3>(vToOutEvtID_);
            WaitFlag<HardEvent::V_MTE3>(vToOutEvtID_);
            DataCopyExtParams copyParams {1, (uint32_t)(pts_ * cam_ * 8 * sizeof(float)), 0, 0, 0};
            DataCopyPad(gradSamplingLocationGm[samplingLocationOffset * 4], gradSamplingLocation, copyParams);
        }
    }

private:
    TPipe* pipe_;
    GlobalTensor<float> mcMsFeatGm, samplingLocationGm, weightGm, outputGradGm;
    GlobalTensor<float> gradMcMsFeatGm, gradSamplingLocationGm, gradWeightsGm;
    GlobalTensor<int32_t> spatialShapeGm, scaleStartLocationGm;
    TBuf<TPosition::VECCALC> weightQue_, gradOutputQue_, samplingLocationQue_, scaleStartLocationQue_, spatialShapeQue_;
    TBuf<TPosition::VECCALC> topGradMcMsFeatQue_, vQue_, featureQue_, featureQue__, pointGradWeightQue_, gradSamplingQue_, sumTmp_;
    uint32_t usedCoreNum_, avgWeightNum_, tailWeightNum_, coreId;
    uint32_t totalTaskNum_, singleProcessTaskLen_, taskRepeatTimes;
    uint32_t pts_, cam_, scale_, group_, numEmbeds, numFeat, numAnchors, totalGroups;
    int64_t taskOffset;
    TEventID cpInEvtID_, cpOutEvtID_, vToOutEvtID_, vToMTE2EvtID_, mte3ToVEvtID_;
};

__aicore__ inline void KernelDeformableAggregationGrad::Process()
{
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
    KernelDeformableAggregationGrad op(
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
 