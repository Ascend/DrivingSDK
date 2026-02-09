/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

namespace {
    constexpr uint32_t BLOCK_SIZE = 32;
    constexpr uint32_t INT32_BYTE = 4;
    constexpr uint32_t INT32_BLOCK_NUM = BLOCK_SIZE / INT32_BYTE;
    constexpr uint32_t REPEAT_SIZE = 256;
    constexpr uint32_t BLOCK_PER_REPEAT = REPEAT_SIZE / BLOCK_SIZE;
    constexpr uint32_t DATA_NUM_PER_MASK = 8;
    constexpr uint32_t MASK_NUM_PER_BLOCK = DATA_NUM_PER_MASK * BLOCK_SIZE;

    constexpr uint32_t FOUR_BUFFER = 4;
}

template<typename DTYPE_F>
class KernelDeformableAggregationGrad {
public:
    __aicore__ inline KernelDeformableAggregationGrad() = delete;

    __aicore__ inline KernelDeformableAggregationGrad(const DeformableAggregationGradTilingData* tilingData, TPipe* pipe) 
    : pipe_(pipe), tilingData_(tilingData) {}

    __aicore__ inline void Init(GM_ADDR mc_ms_feat, GM_ADDR spatial_shape, GM_ADDR scale_start_index, GM_ADDR sampling_location, 
        GM_ADDR weights, GM_ADDR grad_output, GM_ADDR grad_mc_ms_feat, GM_ADDR grad_sampling_location, GM_ADDR grad_weights)
    {
        blockIdx_ = GetBlockIdx();
        InitTiling(tilingData_);
        InitGm(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights, grad_output, grad_mc_ms_feat, 
            grad_sampling_location, grad_weights);
        InitBuffer();
    }

    __aicore__ inline void Process()
    {
        Duplicate(vLocal, (DTYPE_F)0, numEmbeds * FOUR_BUFFER);
        DataCopy(scaleStartLocation, scaleStartLocationGm, scaleStartNum_);
        DataCopy(spatialShape, spatialShapeGm, spatialShapeNum_);
        for (uint32_t i = blockIdx_ * singleProcessTaskNum_; i < totalTaskNum_; i += usedCoreNum_ * singleProcessTaskNum_) {
            ProcessSingleUB(i, min(singleProcessTaskNum_, totalTaskNum_ - i));
        }
    }

protected:
    TPipe* pipe_;
    const DeformableAggregationGradTilingData* tilingData_;
    GlobalTensor<DTYPE_F> mcMsFeatGm, samplingLocationGm, weightGm, outputGradGm;
    GlobalTensor<DTYPE_F> gradMcMsFeatGm, gradSamplingLocalGm, gradWeightsGm;
    GlobalTensor<int32_t> spatialShapeGm, scaleStartLocationGm;

    TBuf<TPosition::VECCALC> weightBuf_, gradOutputBuf_, samplingLocationBuf_, scaleStartLocationBuf_, spatialShapeBuf_;
    TBuf<TPosition::VECCALC> gradWeightsBuf_, gradSamplingBuf_, gradValueBuf_;
    TBuf<TPosition::VECCALC> topGradMcMsFeatBuf_, vBuf_, featureBuf_, pointGradWeightBuf_, pointGradBuf_, weightBrobBuf_, validMaskBuf_;
    
    LocalTensor<int32_t> scaleStartLocation, spatialShape;
    LocalTensor<DTYPE_F> weight, gradOutput, samplingLocation;
    LocalTensor<DTYPE_F> gradWeightsLocal, gradSamplingLocal, gradValueLocal;
    LocalTensor<DTYPE_F> topGradMcMsFeatLocal, vLocal, featureLocal, pointGradWeightLocal, pointGradSum, weightBrobLocal;
    LocalTensor<uint8_t> validMaskLocal;

    uint32_t usedCoreNum_, blockIdx_, dataSize_, blockDataNum_, totalTaskNum_, singleProcessTaskNum_;
    uint32_t pts_, cam_, scale_, group_, numEmbeds, numFeat, numAnchors;
    uint32_t embedsPerGroup, scaleEmbeds_, pointsCam_, camScale_, scaleGroup_, scaleEmbedsSize_, scaleStartNum_, spatialShapeNum_;
    uint32_t taskRpt_, alignedOneTaskNum_, validMaskLen_, repeatDataNum_, scaleGroupAlign_;
    uint32_t v1Offset_ = 0, v2Offset_ = 1, v3Offset_ = 2, v4Offset_ = 3;

    GatherMaskParams gatherParams_;

private:
     __aicore__ inline void InitTiling(const DeformableAggregationGradTilingData* tiling);

     __aicore__ inline void InitGm(GM_ADDR mc_ms_feat, GM_ADDR spatial_shape, GM_ADDR scale_start_index, GM_ADDR sampling_location, 
        GM_ADDR weights, GM_ADDR grad_output, GM_ADDR grad_mc_ms_feat, GM_ADDR grad_sampling_location, GM_ADDR grad_weights);
    
    __aicore__ inline void InitBuffer();

    __aicore__ inline void ProcessSingleUB(uint64_t taskIdx, uint32_t actualTaskNum);

    __aicore__ inline void ComputeMulVF(LocalTensor<DTYPE_F> gradOutLocal, LocalTensor<DTYPE_F> weightLocal, LocalTensor<DTYPE_F> gradValueLocal, 
        LocalTensor<DTYPE_F> featureLocal, LocalTensor<DTYPE_F> gradWeightLocal, const DTYPE_F lh, const DTYPE_F lw, int32_t h, int32_t w);
};

template<typename DTYPE_F>
__aicore__ inline void KernelDeformableAggregationGrad<DTYPE_F>::InitTiling(const DeformableAggregationGradTilingData* tiling)
{
    usedCoreNum_ = tiling->usedCoreNum;
    pts_ = tiling->numPoints;
    cam_  = tiling->numCams;
    scale_ = tiling->numScale;
    group_ = tiling->numGroups;
    numEmbeds = tiling->numEmbeds;
    numFeat = tiling->numFeat;
    numAnchors = tiling->numAnchors;
    totalTaskNum_ = tiling->batchSize * numAnchors;
    singleProcessTaskNum_ = tiling->singleProcessTaskNum;

    dataSize_ = sizeof(DTYPE_F);
    blockDataNum_ = BLOCK_SIZE / dataSize_;
    repeatDataNum_ = BLOCK_PER_REPEAT * blockDataNum_;
    embedsPerGroup = numEmbeds / group_;
    scaleEmbeds_ = scale_ * numEmbeds;
    pointsCam_ = pts_ * cam_;
    camScale_ = cam_ * scale_;
    scaleGroup_ = scale_ * group_;
    scaleEmbedsSize_ = scaleEmbeds_ * dataSize_;
    
    scaleGroupAlign_ = AlignUp(scaleGroup_, blockDataNum_);
    taskRpt_ = DivCeil(singleProcessTaskNum_ * pointsCam_, repeatDataNum_);
    alignedOneTaskNum_ = taskRpt_ * repeatDataNum_;
    validMaskLen_ = DivCeil(alignedOneTaskNum_, MASK_NUM_PER_BLOCK) * BLOCK_SIZE; // 1B for one number
    scaleStartNum_ = AlignUp(camScale_, INT32_BLOCK_NUM);
    spatialShapeNum_ = AlignUp(camScale_ * 2, INT32_BLOCK_NUM);

    v1Offset_ = 0 * scaleEmbeds_;
    v2Offset_ = 1 * scaleEmbeds_;
    v3Offset_ = 2 * scaleEmbeds_;
    v4Offset_ = 3 * scaleEmbeds_;

    gatherParams_.repeatTimes = taskRpt_ * 2;
}  

template<typename DTYPE_F>
__aicore__ inline void KernelDeformableAggregationGrad<DTYPE_F>::InitGm(GM_ADDR mc_ms_feat, GM_ADDR spatial_shape, 
    GM_ADDR scale_start_index, GM_ADDR sampling_location, GM_ADDR weights, GM_ADDR grad_output, GM_ADDR grad_mc_ms_feat, 
    GM_ADDR grad_sampling_location, GM_ADDR grad_weights)
{
    mcMsFeatGm.SetGlobalBuffer((__gm__ DTYPE_F*)(mc_ms_feat));
    spatialShapeGm.SetGlobalBuffer((__gm__ int32_t*)(spatial_shape));
    scaleStartLocationGm.SetGlobalBuffer((__gm__ int32_t*)(scale_start_index));
    samplingLocationGm.SetGlobalBuffer((__gm__ DTYPE_F*)(sampling_location));
    weightGm.SetGlobalBuffer((__gm__ DTYPE_F*)(weights));
    outputGradGm.SetGlobalBuffer((__gm__ DTYPE_F*)(grad_output));
    gradMcMsFeatGm.SetGlobalBuffer((__gm__ DTYPE_F*)(grad_mc_ms_feat));
    gradSamplingLocalGm.SetGlobalBuffer((__gm__ DTYPE_F*)(grad_sampling_location));
    gradWeightsGm.SetGlobalBuffer((__gm__ DTYPE_F*)(grad_weights));
}

template<typename DTYPE_F>
__aicore__ inline void KernelDeformableAggregationGrad<DTYPE_F>::InitBuffer()
{
    pipe_->InitBuffer(weightBuf_, scaleGroupAlign_ * dataSize_);
    pipe_->InitBuffer(gradOutputBuf_, singleProcessTaskNum_ * numEmbeds * dataSize_);
    pipe_->InitBuffer(scaleStartLocationBuf_, scaleStartNum_ * sizeof(int32_t));
    pipe_->InitBuffer(samplingLocationBuf_, 4 * alignedOneTaskNum_ * dataSize_);
    pipe_->InitBuffer(spatialShapeBuf_, spatialShapeNum_ * sizeof(int32_t));

    pipe_->InitBuffer(gradValueBuf_, 21 * scaleEmbedsSize_);

    pipe_->InitBuffer(gradWeightsBuf_, scaleGroupAlign_ * dataSize_);
    pipe_->InitBuffer(gradSamplingBuf_, BLOCK_SIZE);
    pipe_->InitBuffer(validMaskBuf_, 4 * validMaskLen_);

    scaleStartLocation = scaleStartLocationBuf_.Get<int32_t>();
    spatialShape = spatialShapeBuf_.Get<int32_t>();
    weight = weightBuf_.Get<DTYPE_F>();
    gradOutput = gradOutputBuf_.Get<DTYPE_F>();
    samplingLocation = samplingLocationBuf_.Get<DTYPE_F>();

    gradWeightsLocal = gradWeightsBuf_.Get<DTYPE_F>();
    gradSamplingLocal = gradSamplingBuf_.Get<DTYPE_F>();
    validMaskLocal = validMaskBuf_.Get<uint8_t>();  

    gradValueLocal = gradValueBuf_.Get<DTYPE_F>();
    vLocal = gradValueLocal[4 * scaleEmbeds_];
    featureLocal = vLocal[4 * scaleEmbeds_];
    pointGradWeightLocal = featureLocal[4 * scaleEmbeds_];
    pointGradSum = pointGradWeightLocal[4 * scaleEmbeds_];
    weightBrobLocal = pointGradSum[4 * scaleEmbeds_];
}

template<typename DTYPE_F>
__aicore__ inline void KernelDeformableAggregationGrad<DTYPE_F>::ProcessSingleUB(uint64_t taskIdx, uint32_t actualTaskNum)
{
    uint32_t actualCompNum = actualTaskNum * pointsCam_;
    uint32_t outerLoops = DivCeil(actualCompNum, 64);
    uint64_t baseOffset = taskIdx * pointsCam_;
    int32_t gradOuputNum = AlignUp(actualTaskNum * numEmbeds, blockDataNum_);
    int32_t samplingLocationNum = AlignUp(actualTaskNum * pointsCam_ * 2, blockDataNum_);
    uint64_t gradOutputOffset = taskIdx * numEmbeds;

    SetFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(0);
    DataCopy(gradOutput, outputGradGm[gradOutputOffset], gradOuputNum);
    DataCopy(samplingLocation[2 * alignedOneTaskNum_], samplingLocationGm[baseOffset * 2], samplingLocationNum);

    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);
    uint64_t rsvdCnt = 0;
    GatherMask(samplingLocation, samplingLocation[2 * alignedOneTaskNum_], 1, false, 0, gatherParams_, rsvdCnt);
    GatherMask(samplingLocation[alignedOneTaskNum_], samplingLocation[2 * alignedOneTaskNum_], 2, false, 0, gatherParams_, rsvdCnt);
    CompareScalar(validMaskLocal, samplingLocation, static_cast<DTYPE_F>(0.f),
        CMPMODE::GT, repeatDataNum_, taskRpt_, {1, 1, 8, 8});
    CompareScalar(validMaskLocal[validMaskLen_], samplingLocation[alignedOneTaskNum_], static_cast<DTYPE_F>(0.f),
        CMPMODE::GT, repeatDataNum_, taskRpt_, {1, 1, 8, 8});
    CompareScalar(validMaskLocal[2 * validMaskLen_], samplingLocation, static_cast<DTYPE_F>(1.0f),
        CMPMODE::LT, repeatDataNum_, taskRpt_, {1, 1, 8, 8});
    CompareScalar(validMaskLocal[3 * validMaskLen_], samplingLocation[alignedOneTaskNum_], static_cast<DTYPE_F>(1.0f),
        CMPMODE::LT, repeatDataNum_, taskRpt_, {1, 1, 8, 8});

    And(validMaskLocal.ReinterpretCast<uint16_t>(), validMaskLocal.ReinterpretCast<uint16_t>(),
        validMaskLocal[2 * validMaskLen_].ReinterpretCast<uint16_t>(), validMaskLen_);
    And(validMaskLocal.ReinterpretCast<uint16_t>(), validMaskLocal.ReinterpretCast<uint16_t>(),
        validMaskLocal[validMaskLen_].ReinterpretCast<uint16_t>(), validMaskLen_ / 2);

    SetFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE2>(1);
    SetFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::MTE3_V>(1);
    for (uint32_t outerIndex = 0; outerIndex < outerLoops; ++outerIndex) {
        uint64_t valid = validMaskLocal.ReinterpretCast<uint64_t>().GetValue(outerIndex);
        uint32_t innerLoops = outerIndex == (outerLoops - 1) ? actualCompNum - 64 * outerIndex : 64;
        for (uint32_t innerIndex = ScalarGetSFFValue<1>(valid); innerIndex < innerLoops && innerIndex >= 0; innerIndex = ScalarGetSFFValue<1>(valid)) {
            valid = sbitset0(valid, innerIndex);
            uint32_t indexOffset = outerIndex * 64 + innerIndex;
            uint32_t taskOffset = indexOffset / (pointsCam_);
            uint32_t batchIdx = (taskIdx + taskOffset) / numAnchors;
            uint32_t camId = indexOffset % cam_;
            uint64_t weightOffset = (baseOffset + indexOffset) * scaleGroup_;
            uint64_t samplingLocationCopyOutOffset = (baseOffset + indexOffset) * 2;
            uint64_t featOffset = batchIdx * numFeat * numEmbeds;
            uint64_t gradOuputBaseOffset = taskOffset * numEmbeds;

            float locW = samplingLocation.GetValue(indexOffset);
            float locH = samplingLocation.GetValue(alignedOneTaskNum_ + indexOffset);
            WaitFlag<HardEvent::V_MTE2>(0);
            DataCopy(weight, weightGm[weightOffset], scaleGroupAlign_);
            SetFlag<HardEvent::MTE2_V>(0);

            WaitFlag<HardEvent::MTE2_V>(0);
            uint32_t dstShape_[2] = {scaleGroup_, embedsPerGroup};
            uint32_t srcShape_[2] = {scaleGroup_, 1};
            BroadCast<DTYPE_F, 2, 1>(weightBrobLocal, weight, dstShape_, srcShape_);
            SetFlag<HardEvent::V_MTE2>(0);

            WaitFlag<HardEvent::V_MTE2>(1);
            WaitFlag<HardEvent::MTE3_V>(0);
            for (int32_t scaleId = 0; scaleId < scale_; scaleId++) {
                int32_t scaleStartOffset = camId * scale_ + scaleId;
                int32_t scaleStartIdx = scaleStartLocation.GetValue(scaleStartOffset);
                int64_t featureOffset = featOffset + (int64_t)scaleStartIdx * numEmbeds;
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

                int32_t wStride = numEmbeds;
                int32_t hStride = w * wStride;
                int32_t hLowPtrOffset = hLow * hStride;
                int32_t hHighPtrOffset = hLowPtrOffset + hStride;
                int32_t wLowPtrOffset = wLow * wStride;
                int32_t wHighPtrOffset = wLowPtrOffset + wStride;

                uint64_t ptr1 = featureOffset + hLowPtrOffset + wLowPtrOffset;
                uint64_t ptr2 = featureOffset + hLowPtrOffset + wHighPtrOffset;
                uint64_t ptr3 = featureOffset + hHighPtrOffset + wLowPtrOffset;
                uint64_t ptr4 = featureOffset + hHighPtrOffset + wHighPtrOffset;

                uint64_t localOffset = scaleId * numEmbeds;
                uint64_t localPtr1_ = v1Offset_ + localOffset;
                uint64_t localPtr2_ = v2Offset_ + localOffset;
                uint64_t localPtr3_ = v3Offset_ + localOffset;
                uint64_t localPtr4_ = v4Offset_ + localOffset;
                ComputeMulVF(gradOutput[gradOuputBaseOffset], weightBrobLocal[localOffset], gradValueLocal[localOffset], featureLocal[localOffset], 
                    pointGradWeightLocal[localOffset], lh, lw, h, w);
                
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);

                SetAtomicAdd<DTYPE_F>();
                if (hLow >= 0 && wLow >=0) {
                    DataCopy(gradMcMsFeatGm[ptr1], gradValueLocal[localPtr1_], numEmbeds);
                    DataCopy(vLocal[localPtr1_], mcMsFeatGm[ptr1], numEmbeds);
                }
                if (hLow >= 0 && wHigh <= w - 1) {
                    DataCopy(gradMcMsFeatGm[ptr2], gradValueLocal[localPtr2_], numEmbeds);
                    DataCopy(vLocal[localPtr2_], mcMsFeatGm[ptr2], numEmbeds);
                }
                if (hHigh <= h - 1 && wLow >= 0) {
                    DataCopy(gradMcMsFeatGm[ptr3], gradValueLocal[localPtr3_], numEmbeds);
                    DataCopy(vLocal[localPtr3_], mcMsFeatGm[ptr3], numEmbeds);
                }
                if (hHigh <= h - 1 && wHigh <= w - 1) {
                    DataCopy(gradMcMsFeatGm[ptr4], gradValueLocal[localPtr4_], numEmbeds);
                    DataCopy(vLocal[localPtr4_], mcMsFeatGm[ptr4], numEmbeds);
                }
                SetAtomicNone();
            }
            SetFlag<HardEvent::MTE2_V>(1);
            SetFlag<HardEvent::MTE3_V>(0);

            WaitFlag<HardEvent::MTE2_V>(1);
            Mul(featureLocal, featureLocal, vLocal, 4 * scaleEmbeds_);
            Sub(pointGradSum[v1Offset_], vLocal[v2Offset_], vLocal[v1Offset_], scaleEmbeds_);
            Sub(pointGradSum[v3Offset_], vLocal[v4Offset_], vLocal[v3Offset_], scaleEmbeds_);
            Sub(pointGradSum[v2Offset_], vLocal[v3Offset_], vLocal[v1Offset_], scaleEmbeds_);
            Sub(pointGradSum[v4Offset_], vLocal[v4Offset_], vLocal[v2Offset_], scaleEmbeds_);
            Duplicate(vLocal, (DTYPE_F)0, 4 * scaleEmbeds_);
            SetFlag<HardEvent::V_MTE2>(1);

            Add(featureLocal, featureLocal, featureLocal[2 * scaleEmbeds_], 2 * scaleEmbeds_);
            Add(featureLocal, featureLocal, featureLocal[scaleEmbeds_], scaleEmbeds_);
            Mul(pointGradSum, pointGradSum, pointGradWeightLocal, 4 * scaleEmbeds_);
            Add(pointGradSum, pointGradSum, pointGradSum[2 * scaleEmbeds_], 2 * scaleEmbeds_);
            WaitFlag<HardEvent::MTE3_V>(1);
            Sum(gradWeightsLocal, featureLocal, {scaleGroup_, AlignUp(embedsPerGroup, blockDataNum_), embedsPerGroup});
            Sum(gradSamplingLocal, pointGradSum, {2, scaleEmbeds_, scaleEmbeds_});
            SetFlag<HardEvent::V_MTE3>(1);

            WaitFlag<HardEvent::V_MTE3>(1);
            DataCopyExtParams locationCopyParams {1, (uint32_t)(2 * dataSize_), 0, 0, 0};
            DataCopyExtParams weightsCopyParams {1, (uint32_t)(scaleGroup_ * dataSize_), 0, 0, 0};
            DataCopyPad(gradSamplingLocalGm[samplingLocationCopyOutOffset], gradSamplingLocal, locationCopyParams);
            DataCopyPad(gradWeightsGm[weightOffset], gradWeightsLocal, weightsCopyParams);
            SetFlag<HardEvent::MTE3_V>(1);
        }
    }
    WaitFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(1);
    WaitFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(1);
}

template<typename DTYPE_F>
__aicore__ inline void KernelDeformableAggregationGrad<DTYPE_F>::ComputeMulVF(LocalTensor<DTYPE_F> gradOutLocal, LocalTensor<DTYPE_F> weightLocal, 
    LocalTensor<DTYPE_F> gradValueLocal, LocalTensor<DTYPE_F> featureLocal, LocalTensor<DTYPE_F> gradWeightLocal, const DTYPE_F lh, const DTYPE_F lw, int32_t h, int32_t w)
{    
    float hh = 1 - lh;
    float hw = 1 - lw;
    float w1 = hh * hw;
    float w2 = hh * lw;
    float w3 = lh * hw;
    float w4 = lh * lw;
    uint16_t oneRepeatSize = GetVecLen() / dataSize_;
    uint16_t repeatTimes = CeilDivision(numEmbeds, oneRepeatSize);

    __local_mem__ DTYPE_F* gradOutPtr = (__local_mem__ DTYPE_F*) gradOutLocal.GetPhyAddr();
    __local_mem__ DTYPE_F* weightPtr = (__local_mem__ DTYPE_F*) weightLocal.GetPhyAddr();

    __local_mem__ DTYPE_F* gradValuePtr = (__local_mem__ DTYPE_F*) gradValueLocal.GetPhyAddr();
    __local_mem__ DTYPE_F* featurePtr = (__local_mem__ DTYPE_F*) featureLocal.GetPhyAddr();
    __local_mem__ DTYPE_F* gradWeightPtr = (__local_mem__ DTYPE_F*) gradWeightLocal.GetPhyAddr();

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<DTYPE_F> gradOutReg;
        MicroAPI::RegTensor<DTYPE_F> weightReg;
        MicroAPI::RegTensor<DTYPE_F> topGradFeatReg;

        MicroAPI::RegTensor<DTYPE_F> topGradReg;
        MicroAPI::RegTensor<DTYPE_F> bottomGradReg;
        MicroAPI::RegTensor<DTYPE_F> leftGradReg;
        MicroAPI::RegTensor<DTYPE_F> rightGradReg;

        MicroAPI::RegTensor<DTYPE_F> topFeatureReg;
        MicroAPI::RegTensor<DTYPE_F> bottomFeatureReg;
        MicroAPI::RegTensor<DTYPE_F> leftFeatureReg;
        MicroAPI::RegTensor<DTYPE_F> rightFeatureReg;

        MicroAPI::RegTensor<DTYPE_F> topGradWeightReg;
        MicroAPI::RegTensor<DTYPE_F> bottomGradWeightReg;
        MicroAPI::RegTensor<DTYPE_F> leftGradWeightReg;
        MicroAPI::RegTensor<DTYPE_F> rightGradWeightReg;

        // MicroAPI::MaskReg mask = MicroAPI::CreateMask<DTYPE_F, MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            MicroAPI::MaskReg mask = MicroAPI::UpdateMask<DTYPE_F>(numEmbeds);
            MicroAPI::DataCopy(gradOutReg, gradOutPtr + i * oneRepeatSize);
            MicroAPI::DataCopy(weightReg, weightPtr + i * oneRepeatSize);

            MicroAPI::Mul(topGradFeatReg, weightReg, gradOutReg, mask);
            MicroAPI::Muls(topGradReg, topGradFeatReg, static_cast<DTYPE_F>(w1), mask);
            MicroAPI::Muls(bottomGradReg, topGradFeatReg, static_cast<DTYPE_F>(w2), mask);
            MicroAPI::Muls(leftGradReg, topGradFeatReg, static_cast<DTYPE_F>(w3), mask);
            MicroAPI::Muls(rightGradReg, topGradFeatReg, static_cast<DTYPE_F>(w4), mask);

            MicroAPI::Muls(topFeatureReg, gradOutReg, static_cast<DTYPE_F>(w1), mask);
            MicroAPI::Muls(bottomFeatureReg, gradOutReg, static_cast<DTYPE_F>(w2), mask);
            MicroAPI::Muls(leftFeatureReg, gradOutReg, static_cast<DTYPE_F>(w3), mask);
            MicroAPI::Muls(rightFeatureReg, gradOutReg, static_cast<DTYPE_F>(w4), mask);

            MicroAPI::Muls(topGradWeightReg, topGradFeatReg, static_cast<DTYPE_F>(hh * w), mask);
            MicroAPI::Muls(bottomGradWeightReg, topGradFeatReg, static_cast<DTYPE_F>(hw * h), mask);
            MicroAPI::Muls(leftGradWeightReg, topGradFeatReg, static_cast<DTYPE_F>(lh * w), mask);
            MicroAPI::Muls(rightGradWeightReg, topGradFeatReg, static_cast<DTYPE_F>(lw * h), mask);

            MicroAPI::DataCopy(gradValuePtr + i * oneRepeatSize, topGradReg, mask);
            MicroAPI::DataCopy(gradValuePtr + 1 * scaleEmbeds_ + i * oneRepeatSize, bottomGradReg, mask);
            MicroAPI::DataCopy(gradValuePtr + 2 * scaleEmbeds_ + i * oneRepeatSize, leftGradReg, mask);
            MicroAPI::DataCopy(gradValuePtr + 3 * scaleEmbeds_ + i * oneRepeatSize, rightGradReg, mask);

            MicroAPI::DataCopy(featurePtr + i * oneRepeatSize, topFeatureReg, mask);
            MicroAPI::DataCopy(featurePtr + 1 * scaleEmbeds_ + i * oneRepeatSize, bottomFeatureReg, mask);
            MicroAPI::DataCopy(featurePtr + 2 * scaleEmbeds_ + i * oneRepeatSize, leftFeatureReg, mask);
            MicroAPI::DataCopy(featurePtr + 3 * scaleEmbeds_ + i * oneRepeatSize, rightFeatureReg, mask);

            MicroAPI::DataCopy(gradWeightPtr + i * oneRepeatSize, topGradWeightReg, mask);
            MicroAPI::DataCopy(gradWeightPtr + 1 * scaleEmbeds_ + i * oneRepeatSize, bottomGradWeightReg, mask);
            MicroAPI::DataCopy(gradWeightPtr + 2 * scaleEmbeds_ + i * oneRepeatSize, leftGradWeightReg, mask);
            MicroAPI::DataCopy(gradWeightPtr + 3 * scaleEmbeds_ + i * oneRepeatSize, rightGradWeightReg, mask);
        }
    }
}

extern "C" __global__ __aicore__ void deformable_aggregation_grad(GM_ADDR mc_ms_feat, GM_ADDR spatial_shape, GM_ADDR scale_start_index, 
    GM_ADDR sampling_location, GM_ADDR weights, GM_ADDR grad_output, GM_ADDR grad_mc_ms_feat, GM_ADDR grad_sampling_location, GM_ADDR grad_weights, 
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    KernelDeformableAggregationGrad<DTYPE_MC_MS_FEAT> op(&tiling_data, &pipe);
    op.Init(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights, grad_output, grad_mc_ms_feat, 
        grad_sampling_location, grad_weights);
    op.Process();
}
