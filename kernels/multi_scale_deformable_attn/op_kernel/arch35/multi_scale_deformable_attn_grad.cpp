/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 */

#include "kernel_operator.h"
#include "msda.h"

using namespace AscendC;
using namespace MicroAPI;

constexpr uint32_t taskOffset_ = 1024;
constexpr uint16_t taskRpt_ = taskOffset_ / B32_DATA_NUM_PER_REPEAT;
constexpr uint32_t threadNum_ = 1024;
constexpr uint32_t warpSize_ = 32;
constexpr uint32_t warpCount_ = threadNum_ / warpSize_;


template<typename T, typename U>
__aicore__ inline void ComputeGradVF(const LocalTensor<T> locFloat, const LocalTensor<T> shapeFloat,
    const LocalTensor<T> attentionWeight, const LocalTensor<T> weight, const LocalTensor<T> gradAttentionWeights,
    const LocalTensor<T> gradLocation)
{
    __local_mem__ T* locFloatPtr = (__local_mem__ T*) locFloat.GetPhyAddr();
    __local_mem__ T* shapeFloatPtr = (__local_mem__ T*) shapeFloat.GetPhyAddr();
    __local_mem__ T* attentionWeightPtr = (__local_mem__ T*) attentionWeight.GetPhyAddr();
    __local_mem__ T* weightPtr = (__local_mem__ T*) weight.GetPhyAddr();
    __local_mem__ T* gradAttentionWeightsPtr = (__local_mem__ T*) gradAttentionWeights.GetPhyAddr();
    __local_mem__ T* gradLocationPtr = (__local_mem__ T*) gradLocation.GetPhyAddr();

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<T> locationXY1Reg, locationXY2Reg, shapeInput1Reg, shapeInput2Reg;
        MicroAPI::RegTensor<T> locationXReg, locationYReg;
        MicroAPI::RegTensor<U> locationXIntReg, locationYIntReg;
        MicroAPI::RegTensor<T> widthFloatReg, heightFloatReg;
        MicroAPI::RegTensor<T> attentionWeightReg;
        MicroAPI::RegTensor<T> locationWidthLowReg, locationWidthHighReg, locationHeightLowReg, locationHeightHighReg;
        MicroAPI::RegTensor<T> gradLocW1Reg, gradLocW2Reg, gradLocW3Reg, gradLocW4Reg;
        MicroAPI::RegTensor<T> weight1Reg, weight2Reg, weight3Reg, weight4Reg;
        MicroAPI::RegTensor<T> gradWeight1Reg, gradWeight2Reg, gradWeight3Reg, gradWeight4Reg;
        MicroAPI::RegTensor<T> gradAttnReg, gradLocationXY1Reg, gradLocationXY2Reg;

        MicroAPI::MaskReg mask = MicroAPI::CreateMask<T,AscendC::MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg validMask = MicroAPI::CreateMask<T>();
        MicroAPI::MaskReg tmpMask = MicroAPI::CreateMask<T>();

        static constexpr AscendC::MicroAPI::CastTrait castF2ITrait = 
            {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};
        static constexpr AscendC::MicroAPI::CastTrait castI2FTrait = 
            {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        for (uint16_t taskIdx = 0; taskIdx < taskRpt_; ++taskIdx) {
            uint32_t localOffset = taskIdx * B32_DATA_NUM_PER_REPEAT;

            MicroAPI::DataCopy(locationXY1Reg, locFloatPtr + 2 * localOffset);
            MicroAPI::DataCopy(locationXY2Reg, locFloatPtr + 2 * localOffset + B32_DATA_NUM_PER_REPEAT);
            MicroAPI::DataCopy(shapeInput1Reg, shapeFloatPtr + 2 * localOffset);
            MicroAPI::DataCopy(shapeInput2Reg, shapeFloatPtr + 2 * localOffset + B32_DATA_NUM_PER_REPEAT);
            MicroAPI::DataCopy(attentionWeightReg, attentionWeightPtr + localOffset);

            MicroAPI::DataCopy(weight1Reg, weightPtr + 4 * localOffset);
            MicroAPI::DataCopy(weight2Reg, weightPtr + 4 * localOffset + B32_DATA_NUM_PER_REPEAT);
            MicroAPI::DataCopy(weight3Reg, weightPtr + 4 * localOffset + 2 * B32_DATA_NUM_PER_REPEAT);
            MicroAPI::DataCopy(weight4Reg, weightPtr + 4 * localOffset + 3 * B32_DATA_NUM_PER_REPEAT);

            // [N, 2] -> [2, N]
            MicroAPI::DeInterleave(locationXReg, locationYReg, locationXY1Reg, locationXY2Reg);
            MicroAPI::DeInterleave(widthFloatReg, heightFloatReg, shapeInput1Reg, shapeInput2Reg);

            // [N, 4] -> [4, N]
            MicroAPI::DeInterleave(gradWeight1Reg, gradWeight2Reg, weight1Reg, weight2Reg);
            MicroAPI::DeInterleave(gradWeight3Reg, gradWeight4Reg, weight3Reg, weight4Reg);
            MicroAPI::DeInterleave(weight1Reg, weight3Reg, gradWeight1Reg, gradWeight3Reg);
            MicroAPI::DeInterleave(weight2Reg, weight4Reg, gradWeight2Reg, gradWeight4Reg);

            MicroAPI::Compares<T, CMPMODE::GT>(validMask, locationXReg, -1.0f, mask);
            MicroAPI::Compares<T, CMPMODE::GT>(tmpMask, locationYReg, -1.0f, mask);
            MicroAPI::And(validMask, validMask, tmpMask, mask);
            MicroAPI::Compare<T, CMPMODE::LT>(tmpMask, locationXReg, widthFloatReg, mask);
            MicroAPI::And(validMask, validMask, tmpMask, mask);
            MicroAPI::Compare<T, CMPMODE::LT>(tmpMask, locationYReg, heightFloatReg, mask);
            MicroAPI::And(validMask, validMask, tmpMask, mask);

            MicroAPI::Cast<U, T, castF2ITrait>(locationXIntReg, locationXReg, mask);
            MicroAPI::Cast<U, T, castF2ITrait>(locationYIntReg, locationYReg, mask);
            MicroAPI::Cast<T, U, castI2FTrait>(locationWidthLowReg, locationXIntReg, mask);
            MicroAPI::Cast<T, U, castI2FTrait>(locationHeightLowReg, locationYIntReg, mask);

            MicroAPI::Sub(locationWidthLowReg, locationXReg, locationWidthLowReg, mask);
            MicroAPI::Sub(locationHeightLowReg, locationYReg, locationHeightLowReg, mask);
            MicroAPI::Duplicate(locationWidthHighReg, 1.0f, mask);
            MicroAPI::Duplicate(locationHeightHighReg, 1.0f, mask);
            MicroAPI::Sub(locationWidthHighReg, locationWidthHighReg, locationWidthLowReg, mask);
            MicroAPI::Sub(locationHeightHighReg, locationHeightHighReg, locationHeightLowReg, mask);

            MicroAPI::Mul(gradWeight1Reg, locationHeightHighReg, locationWidthHighReg, mask);
            MicroAPI::Mul(gradWeight2Reg, locationHeightHighReg, locationWidthLowReg, mask);
            MicroAPI::Mul(gradWeight3Reg, locationHeightLowReg, locationWidthHighReg, mask);
            MicroAPI::Mul(gradWeight4Reg, locationHeightLowReg, locationWidthLowReg, mask);

            MicroAPI::Mul(gradWeight1Reg, weight1Reg, gradWeight1Reg, mask);
            MicroAPI::Mul(gradWeight2Reg, weight2Reg, gradWeight2Reg, mask);
            MicroAPI::Mul(gradWeight3Reg, weight3Reg, gradWeight3Reg, mask);
            MicroAPI::Mul(gradWeight4Reg, weight4Reg, gradWeight4Reg, mask);

            MicroAPI::Add(gradAttnReg, gradWeight1Reg, gradWeight2Reg, mask);
            MicroAPI::Add(gradAttnReg, gradAttnReg, gradWeight3Reg, mask);
            MicroAPI::Add<T, MaskMergeMode::ZEROING>(gradAttnReg, gradAttnReg, gradWeight4Reg, validMask);

            MicroAPI::DataCopy(gradAttentionWeightsPtr + localOffset, gradAttnReg, mask);

            MicroAPI::Sub(gradLocW1Reg, weight4Reg, weight3Reg, mask);
            MicroAPI::Sub(gradLocW2Reg, weight4Reg, weight2Reg, mask);
            MicroAPI::Sub(gradLocW3Reg, weight2Reg, weight1Reg, mask);
            MicroAPI::Sub(gradLocW4Reg, weight3Reg, weight1Reg, mask);

            MicroAPI::Mul(gradLocW1Reg, gradLocW1Reg, locationHeightLowReg, mask);
            MicroAPI::Mul(gradLocW2Reg, gradLocW2Reg, locationWidthLowReg, mask);
            MicroAPI::Mul(gradLocW3Reg, gradLocW3Reg, locationHeightHighReg, mask);
            MicroAPI::Mul(gradLocW4Reg, gradLocW4Reg, locationWidthHighReg, mask);

            MicroAPI::Add(gradLocW1Reg, gradLocW1Reg, gradLocW3Reg, mask);
            MicroAPI::Add(gradLocW2Reg, gradLocW2Reg, gradLocW4Reg, mask);

            MicroAPI::Mul(gradLocW1Reg, gradLocW1Reg, attentionWeightReg, mask);
            MicroAPI::Mul(gradLocW2Reg, gradLocW2Reg, attentionWeightReg, mask);

            MicroAPI::Mul<T, MaskMergeMode::ZEROING>(gradLocW1Reg, gradLocW1Reg, widthFloatReg, validMask);
            MicroAPI::Mul<T, MaskMergeMode::ZEROING>(gradLocW2Reg, gradLocW2Reg, heightFloatReg, validMask);

            MicroAPI::Interleave(gradLocationXY1Reg, gradLocationXY2Reg, gradLocW1Reg, gradLocW2Reg);
            MicroAPI::DataCopy(gradLocationPtr + 2 * localOffset, gradLocationXY1Reg, mask);
            MicroAPI::DataCopy(gradLocationPtr + 2 * localOffset + B32_DATA_NUM_PER_REPEAT, gradLocationXY2Reg, mask);
        }
    }
}


__simt_vf__ __aicore__ LAUNCH_BOUND(threadNum_) inline void MSDASimtComputeGradSmallByHead(
    __gm__ float* gradValueGm_, __gm__ float* valueGm_, __gm__ float* gradOutGm_,
    __ubuf__ float2* locFloat, __ubuf__ float2* shapeFloat, __ubuf__ int2* locationInt,
    __ubuf__ float* attnWeight, __ubuf__ float4* weight, uint32_t count, uint32_t outDims_,
    uint32_t embedDims_, uint32_t oneHeadNum_)
{
    uint32_t channelIdx = threadIdx.x;
    for (uint32_t headIdx = blockDim.y * threadIdx.z + threadIdx.y; headIdx < count; headIdx += blockDim.y * blockDim.z) {
        if (channelIdx >= embedDims_) {
            continue;
        }
        uint32_t outOffset = headIdx * embedDims_ + channelIdx;
        float grad = gradOutGm_[outOffset];
        for (uint32_t point = 0; point < oneHeadNum_; point++) {
            uint32_t pointIdx = headIdx * oneHeadNum_ + point;
            float2 image = shapeFloat[pointIdx];
            float2 location = locFloat[pointIdx];
            if (!(location.x > -1 && location.y > -1 && location.x < image.x && location.y < image.y)) {
                continue;
            }

            int2 gmOffset = locationInt[pointIdx];
            gmOffset.x = gmOffset.x + channelIdx;
            gmOffset.y = gmOffset.y + channelIdx;

            float v1 = (location.y >= 0 && location.x >= 0) ? valueGm_[gmOffset.x] : 0;
            float v2 = (location.y >= 0 && location.x < image.x - 1) ? valueGm_[gmOffset.x + outDims_] : 0;
            float v3 = (location.y < image.y - 1 && location.x >= 0) ? valueGm_[gmOffset.y] : 0;
            float v4 = (location.y < image.y - 1 && location.x < image.x - 1) ? valueGm_[gmOffset.y + outDims_] : 0;

            float attn = attnWeight[pointIdx];
            float gradValueMul = grad * attn;

            float lh = location.y - Simt::Floor(location.y);
            float lw = location.x - Simt::Floor(location.x);
            float hh = 1 - lh;
            float hw = 1 - lw;

            float w1 = hh * hw;
            float w2 = hh * lw;
            float w3 = lh * hw;
            float w4 = lh * lw;

            if (location.y >= 0 && location.x >= 0) {
                Simt::AtomicAdd(gradValueGm_ + gmOffset.x, w1 * gradValueMul);
            }
            if (location.y >= 0 && location.x < image.x - 1) {
                Simt::AtomicAdd(gradValueGm_ + gmOffset.x + outDims_, w2 * gradValueMul);
            }
            if (location.y < image.y - 1 && location.x >= 0) {
                Simt::AtomicAdd(gradValueGm_ + gmOffset.y, w3 * gradValueMul);
            }
            if (location.y < image.y - 1 && location.x < image.x - 1) {
                Simt::AtomicAdd(gradValueGm_ + gmOffset.y + outDims_, w4 * gradValueMul);
            }

            for (uint32_t reduceIdx = 0; reduceIdx < blockDim.y; reduceIdx++) {
                bool reduceFlag = reduceIdx == threadIdx.y;
                float gradWeight1 = reduceFlag ? v1 * grad : 0;
                float gradWeight2 = reduceFlag ? v2 * grad : 0;
                float gradWeight3 = reduceFlag ? v3 * grad : 0;
                float gradWeight4 = reduceFlag ? v4 * grad : 0;

                float4 results;
                results.x = Simt::WarpReduceAddSync(gradWeight1);
                results.y = Simt::WarpReduceAddSync(gradWeight2);
                results.z = Simt::WarpReduceAddSync(gradWeight3);
                results.w = Simt::WarpReduceAddSync(gradWeight4);

                if (reduceFlag & (channelIdx == 0)) {
                    weight[pointIdx] = results;
                }
            }
        }
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(threadNum_) inline void MSDASimtComputeGradSmall(
    __gm__ float* gradValueGm_, __gm__ float* valueGm_, __gm__ float* gradOutGm_,
    __ubuf__ float2* locFloat, __ubuf__ float2* shapeFloat, __ubuf__ int2* locationInt,
    __ubuf__ float* attnWeight, __ubuf__ float4* weight, uint32_t count, uint32_t outDims_,
    uint32_t embedDims_, uint32_t magic, uint32_t shift)
{
    uint32_t channelIdx = threadIdx.x;
    for (uint32_t pointIdx = blockDim.y * threadIdx.z + threadIdx.y; pointIdx < count; pointIdx += blockDim.y * blockDim.z) {
        if (channelIdx >= embedDims_) {
            continue;
        }
        float2 image = shapeFloat[pointIdx];
        float2 location = locFloat[pointIdx];
        if (!(location.x > -1 && location.y > -1 && location.x < image.x && location.y < image.y)) {
            continue;
        }

        int2 gmOffset = locationInt[pointIdx];
        gmOffset.x = gmOffset.x + channelIdx;
        gmOffset.y = gmOffset.y + channelIdx;

        float v1 = (location.y >= 0 && location.x >= 0) ? valueGm_[gmOffset.x] : 0;
        float v2 = (location.y >= 0 && location.x < image.x - 1) ? valueGm_[gmOffset.x + outDims_] : 0;
        float v3 = (location.y < image.y - 1 && location.x >= 0) ? valueGm_[gmOffset.y] : 0;
        float v4 = (location.y < image.y - 1 && location.x < image.x - 1) ? valueGm_[gmOffset.y + outDims_] : 0;

        uint32_t outOffset = Simt::UintDiv(pointIdx, magic, shift) * embedDims_ + channelIdx;
        float grad = gradOutGm_[outOffset];
        float attn = attnWeight[pointIdx];
        float gradValueMul = grad * attn;

        float lh = location.y - Simt::Floor(location.y);
        float lw = location.x - Simt::Floor(location.x);
        float hh = 1 - lh;
        float hw = 1 - lw;

        float w1 = hh * hw;
        float w2 = hh * lw;
        float w3 = lh * hw;
        float w4 = lh * lw;

        if (location.y >= 0 && location.x >= 0) {
            Simt::AtomicAdd(gradValueGm_ + gmOffset.x, w1 * gradValueMul);
        }
        if (location.y >= 0 && location.x < image.x - 1) {
            Simt::AtomicAdd(gradValueGm_ + gmOffset.x + outDims_, w2 * gradValueMul);
        }
        if (location.y < image.y - 1 && location.x >= 0) {
            Simt::AtomicAdd(gradValueGm_ + gmOffset.y, w3 * gradValueMul);
        }
        if (location.y < image.y - 1 && location.x < image.x - 1) {
            Simt::AtomicAdd(gradValueGm_ + gmOffset.y + outDims_, w4 * gradValueMul);
        }

        for (uint32_t reduceIdx = 0; reduceIdx < blockDim.y; reduceIdx++) {
            bool reduceFlag = reduceIdx == threadIdx.y;
            float gradWeight1 = reduceFlag ? v1 * grad : 0;
            float gradWeight2 = reduceFlag ? v2 * grad : 0;
            float gradWeight3 = reduceFlag ? v3 * grad : 0;
            float gradWeight4 = reduceFlag ? v4 * grad : 0;

            float4 results;
            results.x = Simt::WarpReduceAddSync(gradWeight1);
            results.y = Simt::WarpReduceAddSync(gradWeight2);
            results.z = Simt::WarpReduceAddSync(gradWeight3);
            results.w = Simt::WarpReduceAddSync(gradWeight4);

            if (reduceFlag & (channelIdx == 0)) {
                weight[pointIdx] = results;
            }
        }
    }
}


__simt_vf__ __aicore__ LAUNCH_BOUND(threadNum_) inline void MSDASimtComputeGradLarge(
    __gm__ float* gradValueGm_, __gm__ float* valueGm_, __gm__ float* gradOutGm_,
    __ubuf__ float2* locFloat, __ubuf__ float2* shapeFloat, __ubuf__ int2* locationInt,
    __ubuf__ float* attnWeight, __ubuf__ float4* weight, uint32_t count,
    uint32_t outDims_, uint32_t embedDims_, uint32_t magic, uint32_t shift)
{
    uint32_t channelIdx = threadIdx.x;
    for (uint32_t pointIdx = threadIdx.y; pointIdx < count; pointIdx += blockDim.y) {
        float2 image = shapeFloat[pointIdx];
        float2 location = locFloat[pointIdx];

        if (!(location.x > -1 && location.y > -1 && location.x < image.x && location.y < image.y)) {
            continue;
        }

        float attn = attnWeight[pointIdx];
        int2 gmOffset = locationInt[pointIdx];
        gmOffset.x = gmOffset.x + channelIdx;
        gmOffset.y = gmOffset.y + channelIdx;

        float lh = location.y - Simt::Floor(location.y);
        float lw = location.x - Simt::Floor(location.x);
        float hh = 1 - lh;
        float hw = 1 - lw;

        float w1 = hh * hw;
        float w2 = hh * lw;
        float w3 = lh * hw;
        float w4 = lh * lw;

        float value1 = 0, value2 = 0, value3 = 0, value4 = 0;
        for (uint32_t channelOffset = 0; channelOffset < embedDims_; channelOffset += Simt::GetWarpSize()) {
            if ((channelIdx + channelOffset) >= embedDims_) {
                continue;
            }
            float v1 = (location.y >= 0 && location.x >= 0) ? valueGm_[gmOffset.x + channelOffset] : 0;
            float v2 = (location.y >= 0 && location.x < image.x - 1) ? valueGm_[gmOffset.x + outDims_ + channelOffset] : 0;
            float v3 = (location.y < image.y - 1 && location.x >= 0) ? valueGm_[gmOffset.y + channelOffset] : 0;
            float v4 = (location.y < image.y - 1 && location.x < image.x - 1) ? valueGm_[gmOffset.y + outDims_ + channelOffset] : 0;

            uint32_t outOffset = Simt::UintDiv(pointIdx, magic, shift) * embedDims_ + channelIdx + channelOffset;
            float grad = gradOutGm_[outOffset];
            float gradValueMul = grad * attn;

            if (location.y >= 0 && location.x >= 0) {
                Simt::AtomicAdd(gradValueGm_ + gmOffset.x + channelOffset, w1 * gradValueMul);
            }
            if (location.y >= 0 && location.x < image.x - 1) {
                Simt::AtomicAdd(gradValueGm_ + gmOffset.x + outDims_ + channelOffset, w2 * gradValueMul);
            }
            if (location.y < image.y - 1 && location.x >= 0) {
                Simt::AtomicAdd(gradValueGm_ + gmOffset.y + channelOffset, w3 * gradValueMul);
            }
            if (location.y < image.y - 1 && location.x < image.x - 1) {
                Simt::AtomicAdd(gradValueGm_ + gmOffset.y + outDims_ + channelOffset, w4 * gradValueMul);
            }

            value1 = value1 + v1 * grad;
            value2 = value2 + v2 * grad;
            value3 = value3 + v3 * grad;
            value4 = value4 + v4 * grad;
        }

        float4 results;
        results.x = Simt::WarpReduceAddSync(value1);
        results.y = Simt::WarpReduceAddSync(value2);
        results.z = Simt::WarpReduceAddSync(value3);
        results.w = Simt::WarpReduceAddSync(value4);

        if (channelIdx == 0) {
            weight[pointIdx] = results;
        }
    }
}

class MultiScaleDeformableAttnGradKernel {
public:
    __aicore__ inline MultiScaleDeformableAttnGradKernel() = delete;

    __aicore__ inline MultiScaleDeformableAttnGradKernel(GM_ADDR value, GM_ADDR valueSpatialShapes,
        GM_ADDR valueLevelStartIndex, GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR gradOutput,
        GM_ADDR gradValue, GM_ADDR gradSamplingLocations, GM_ADDR gradAttentionWeights,
        MultiScaleDeformableAttnTilingData* tilingData, TPipe* pipe)
        : pipe_(pipe), blkIdx_(GetBlockIdx())
    {
        InitTiling(tilingData);
        InitGM(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations, attentionWeights,
            gradOutput, gradValue, gradSamplingLocations, gradAttentionWeights);
        InitBuffer();
        ResetMask();
        SetAtomicNone();
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> locationFloat = locationQue_.template Get<float>();
        LocalTensor<int32_t> locationInt = gmOffsetbuf_.template Get<int32_t>();
        LocalTensor<float> attentionWeights = attentionWeightsQue_.template Get<float>();
        LocalTensor<int32_t> shapes = shapeQue_.template Get<int32_t>();
        LocalTensor<int32_t> offset = offsetQue_.template Get<int32_t>();
        LocalTensor<float> shapeFloat = shapeFloatBuf_.template Get<float>();
        LocalTensor<int32_t> shapeInt = shapeFloatBuf_.template Get<int32_t>();
        LocalTensor<int32_t> offsetInt = offsetIntBuf_.template Get<int32_t>();
        LocalTensor<int32_t> validMask = validMaskBuf_.template Get<int32_t>();
        LocalTensor<float> weight = weightBuf_.template Get<float>();
        LocalTensor<float> gradLocation = gradLocationQue_.template Get<float>();
        LocalTensor<float> gradAttentionWeights = gradAttentionWeightsQue_.template Get<float>();

        PrepareShape(shapes, shapeInt, shapeFloat, offset, offsetInt);

        for (uint32_t taskIdx = blkIdx_ * compTaskNum_; taskIdx < batchSize_ * numQueries_; taskIdx += compTaskNum_ * coreNum_) {
            uint32_t baseNum = (taskIdx / numQueries_ + 1) * numQueries_ - taskIdx;
            uint32_t taskNum = min(compTaskNum_, batchSize_ * numQueries_ - taskIdx);
            uint32_t baseSrcOffset = taskIdx / numQueries_ * numKeys_ * numHeads_;
            uint32_t nextSrcOffset = baseSrcOffset + numKeys_ * numHeads_;

            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
            CopyInSample(locationFloat[2 * alignedOneTaskNum_], attentionWeights, taskIdx, taskNum);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            Duplicate(weight, 0.f, 4 * alignedOneTaskNum_);
            ComputeGmOffsetVF<float, int32_t>(taskRpt_, numHeads_, embedDims_, baseSrcOffset, nextSrcOffset, baseNum * oneQueryNum_,
                locationFloat, shapeFloat, offsetInt, locationInt, validMask);
            pipe_barrier(PIPE_ALL);

            CallMSDASimtFunc(taskIdx, taskNum, locationFloat, shapeFloat, locationInt, attentionWeights, weight);
            pipe_barrier(PIPE_ALL);

            ComputeGradVF<float, int32_t>(locationFloat, shapeFloat, attentionWeights, weight, gradAttentionWeights, gradLocation);
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            CopyOutGrad(gradLocation, gradAttentionWeights, taskIdx, taskNum);
        }
    }

    __aicore__ inline void CallMSDASimtFunc(uint32_t taskIdx, uint32_t taskNum, const LocalTensor<float>& locationFloat, const LocalTensor<float>& shapeFloat,
        const LocalTensor<int32_t>& locationInt, const LocalTensor<float>& attentionWeights, const LocalTensor<float>& weight)
    {
        if (embedDims_ > warpSize_) {
            Simt::VF_CALL<MSDASimtComputeGradLarge>(Simt::Dim3(warpSize_, warpCount_),
                (__gm__ float*)gradValueGm_.GetPhyAddr(), (__gm__ float*)valueGm_.GetPhyAddr(), (__gm__ float*)gradOutGm_[taskIdx * outDims_].GetPhyAddr(),
                (__ubuf__ float2*)locationFloat.GetPhyAddr(), (__ubuf__ float2*)shapeFloat.GetPhyAddr(), (__ubuf__ int2*)locationInt.GetPhyAddr(),
                (__ubuf__ float*)attentionWeights.GetPhyAddr(), (__ubuf__ float4*)weight.GetPhyAddr(), taskNum * oneQueryNum_, outDims_, embedDims_, magic, shift);
        } else if (oneHeadNum_ <= embedDims_) {
            // if oneHeadNum_ > embedDims_, the thread in this branch can not reach 1024
            Simt::VF_CALL<MSDASimtComputeGradSmallByHead>(Simt::Dim3(embedDimsAlign_, warpGroupSize_, warpCount_),
                (__gm__ float*)gradValueGm_.GetPhyAddr(), (__gm__ float*)valueGm_.GetPhyAddr(), (__gm__ float*)gradOutGm_[taskIdx * outDims_].GetPhyAddr(),
                (__ubuf__ float2*)locationFloat.GetPhyAddr(), (__ubuf__ float2*)shapeFloat.GetPhyAddr(), (__ubuf__ int2*)locationInt.GetPhyAddr(),
                (__ubuf__ float*)attentionWeights.GetPhyAddr(), (__ubuf__ float4*)weight.GetPhyAddr(), taskNum * numHeads_, outDims_, embedDims_, oneHeadNum_);
        } else {
            Simt::VF_CALL<MSDASimtComputeGradSmall>(Simt::Dim3(embedDimsAlign_, warpGroupSize_, warpCount_),
                (__gm__ float*)gradValueGm_.GetPhyAddr(), (__gm__ float*)valueGm_.GetPhyAddr(), (__gm__ float*)gradOutGm_[taskIdx * outDims_].GetPhyAddr(),
                (__ubuf__ float2*)locationFloat.GetPhyAddr(), (__ubuf__ float2*)shapeFloat.GetPhyAddr(), (__ubuf__ int2*)locationInt.GetPhyAddr(),
                (__ubuf__ float*)attentionWeights.GetPhyAddr(), (__ubuf__ float4*)weight.GetPhyAddr(), taskNum * oneQueryNum_, outDims_, embedDims_, magic, shift);
        }
    }

protected:
    __aicore__ inline void InitTiling(const MultiScaleDeformableAttnTilingData* tilingData)
    {
        batchSize_ = tilingData->batchSize;
        numKeys_ = tilingData->numKeys;
        numHeads_ = tilingData->numHeads;
        embedDims_ = tilingData->embedDims;
        numLevels_ = tilingData->numLevels;
        numQueries_ = tilingData->numQueries;
        numPoints_ = tilingData->numPoints;
        coreNum_ = tilingData->coreNum;
        realLevels_ = tilingData->realLevels;

        oneQueryNum_ = numHeads_ * numLevels_ * numPoints_;
        oneHeadNum_ = numLevels_ * numPoints_;
        outDims_ = numHeads_ * embedDims_;

        compTaskNum_ = taskOffset_ / oneQueryNum_;
        compTaskNum_ = min(numQueries_, compTaskNum_);
        alignedOneTaskNum_ = taskOffset_;

        if (embedDims_ <= warpSize_) {
            int64_t leadingZeros = ScalarCountLeadingZero((uint64_t)embedDims_);
            int32_t power = 63 - leadingZeros;
            if ((embedDims_ & (embedDims_ - 1)) != 0) {
                power += 1;
            }
            embedDimsAlign_ = 1 << power;
            warpGroupSize_ = warpSize_ / embedDimsAlign_;
        }

        GetUintDivMagicAndShift(magic, shift, oneHeadNum_);
    }

    __aicore__ inline void InitGM(GM_ADDR value, GM_ADDR valueSpatialShapes, GM_ADDR valueLevelStartIndex,
        GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR gradOut, GM_ADDR gradValue,
        GM_ADDR gradSamplingLocations, GM_ADDR gradAttentionWeights)
    {
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(value));
        locationGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(samplingLocations));
        attentionWeightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(attentionWeights));

        valueSpatialShapesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valueSpatialShapes));
        valueLevelStartIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valueLevelStartIndex));

        gradOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradOut));
        gradValueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradValue));
        gradLocGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradSamplingLocations));
        gradAttentionWeightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradAttentionWeights));
    }

    __aicore__ inline void InitBuffer()
    {
        pipe_->InitBuffer(shapeQue_, AlignUp(numLevels_ * 2, B32_DATA_NUM_PER_BLOCK) * B32_BYTE_SIZE);
        pipe_->InitBuffer(offsetQue_, AlignUp(numLevels_, B32_DATA_NUM_PER_BLOCK) * B32_BYTE_SIZE);
        pipe_->InitBuffer(shapeFloatBuf_, 2 * alignedOneTaskNum_ * B32_BYTE_SIZE); // w, h
        pipe_->InitBuffer(offsetIntBuf_, alignedOneTaskNum_ * B32_BYTE_SIZE);      // offsetInt
        pipe_->InitBuffer(locationQue_, 4 * alignedOneTaskNum_ * B32_BYTE_SIZE);   // x, y
        pipe_->InitBuffer(gmOffsetbuf_, 2 * alignedOneTaskNum_ * B32_BYTE_SIZE);   // x, y
        pipe_->InitBuffer(validMaskBuf_, alignedOneTaskNum_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(attentionWeightsQue_, alignedOneTaskNum_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(weightBuf_, 4 * alignedOneTaskNum_ * B32_BYTE_SIZE);     // w1-w4
        pipe_->InitBuffer(gradLocationQue_, 2 * alignedOneTaskNum_ * B32_BYTE_SIZE); // x, y
        pipe_->InitBuffer(gradAttentionWeightsQue_, alignedOneTaskNum_ * B32_BYTE_SIZE);
    }

    __aicore__ inline void PrepareShape(const LocalTensor<int32_t>& shapes, const LocalTensor<int32_t>& shapeInt,
        const LocalTensor<float>& shapeFloat, const LocalTensor<int32_t>& offset, const LocalTensor<int32_t>& offsetInt)
    {
        DataCopy(shapes, valueSpatialShapesGm_,
            {1, static_cast<uint16_t>(DivCeil(2 * numLevels_, B32_DATA_NUM_PER_BLOCK)), 0, 0});
        DataCopy(offset, valueLevelStartIndexGm_,
            {1, static_cast<uint16_t>(DivCeil(numLevels_, B32_DATA_NUM_PER_BLOCK)), 0, 0});
        // broadcast to [head*level, POINT]
        for (uint32_t query = 0; query < compTaskNum_; ++query) {
            for (uint32_t head = 0; head < numHeads_; ++head) {
                uint32_t idx = (query * numHeads_ + head) * oneHeadNum_;
                for (uint32_t level = 0; level < numLevels_; ++level) {
                    int32_t w = shapes.GetValue(2 * level + 1);
                    int32_t h = shapes.GetValue(2 * level);
                    int32_t o = offset.GetValue(level);
                    for (uint32_t point = 0; point < numPoints_; ++point) {
                        int32_t xIdx = 2 * idx;
                        int32_t yIdx = 2 * idx + 1;
                        shapeInt.SetValue(xIdx, w);
                        shapeInt.SetValue(yIdx, h);
                        offsetInt.SetValue(idx, o * numHeads_ + head);
                        ++idx;
                    }
                }
            }
        }
        Cast<float, int32_t>(shapeFloat, shapeInt, RoundMode::CAST_NONE, 2 * alignedOneTaskNum_);
    }

    __aicore__ inline void CopyInSample(
        const LocalTensor<float>& location, const LocalTensor<float>& attentionWeights, uint32_t taskIdx, uint32_t taskNum)
    {
        if (unlikely(numLevels_ != realLevels_)) {
            uint64_t sampleOffset = taskIdx * numHeads_ * realLevels_ * numPoints_;
            DataCopyPad<float, PaddingMode::Compact>(location, locationGm_[sampleOffset * 2],
                {static_cast<uint16_t>(taskNum), 2 * oneQueryNum_ * B32_BYTE_SIZE, 2 * numHeads_ * (realLevels_ - numLevels_) * numPoints_ * B32_BYTE_SIZE, 0, 0}, {});
            DataCopyPad<float, PaddingMode::Compact>(attentionWeights, attentionWeightsGm_[sampleOffset],
                {static_cast<uint16_t>(taskNum), oneQueryNum_ * B32_BYTE_SIZE, numHeads_ * (realLevels_ - numLevels_) * numPoints_ * B32_BYTE_SIZE, 0, 0}, {});
        } else {
            uint64_t sampleOffset = taskIdx * oneQueryNum_;
            uint32_t sampleNum = taskNum * oneQueryNum_ * B32_BYTE_SIZE;
            DataCopyPad(location, locationGm_[sampleOffset * 2], {1, 2 * sampleNum, 0, 0, 0}, {});
            DataCopyPad(attentionWeights, attentionWeightsGm_[sampleOffset], {1, sampleNum, 0, 0, 0}, {});
        }
    }

    __aicore__ inline void CopyOutGrad(
        const LocalTensor<float>& gradLocation, const LocalTensor<float>& gradAttentionWeights, uint32_t taskIdx, uint32_t taskNum)
    {
        if (unlikely(numLevels_ != realLevels_)) {
            uint64_t sampleOffset = taskIdx * numHeads_ * realLevels_ * numPoints_;
            DataCopyPad<float, PaddingMode::Compact>(gradLocGm_[sampleOffset * 2], gradLocation,
                {static_cast<uint16_t>(taskNum), 2 * oneQueryNum_ * B32_BYTE_SIZE, 0, 2 * numHeads_ * (realLevels_ - numLevels_) * numPoints_ * B32_BYTE_SIZE, 0});
            DataCopyPad<float, PaddingMode::Compact>(gradAttentionWeightsGm_[sampleOffset], gradAttentionWeights,
                {static_cast<uint16_t>(taskNum), oneQueryNum_ * B32_BYTE_SIZE, 0, numHeads_ * (realLevels_ - numLevels_) * numPoints_ * B32_BYTE_SIZE, 0});
        } else {
            uint64_t sampleOffset = taskIdx * oneQueryNum_;
            uint32_t sampleNum = taskNum * oneQueryNum_ * B32_BYTE_SIZE;
            DataCopyPad(gradLocGm_[sampleOffset * 2], gradLocation, {1, 2 * sampleNum, 0, 0, 0});
            DataCopyPad(gradAttentionWeightsGm_[sampleOffset], gradAttentionWeights, {1, sampleNum, 0, 0, 0});
        }
    }

protected:
    TPipe* pipe_;
    GlobalTensor<float> valueGm_, locationGm_, attentionWeightsGm_;
    GlobalTensor<float> gradOutGm_, gradValueGm_, gradAttentionWeightsGm_, gradLocGm_;
    GlobalTensor<int32_t> valueSpatialShapesGm_, valueLevelStartIndexGm_;

    TBuf<TPosition::VECCALC> locationQue_, attentionWeightsQue_, shapeQue_, offsetQue_;
    TBuf<TPosition::VECCALC> shapeFloatBuf_, offsetIntBuf_, weightBuf_, gmOffsetbuf_, validMaskBuf_;
    TBuf<TPosition::VECCALC> gradLocationQue_, gradAttentionWeightsQue_;

    int32_t blkIdx_;
    // const values
    uint32_t coreNum_, compTaskNum_;
    uint32_t batchSize_, numKeys_, numHeads_, embedDims_, outDims_, numLevels_, numQueries_, numPoints_, realLevels_;
    uint32_t alignedOneTaskNum_;
    uint32_t oneHeadNum_, oneQueryNum_;
    uint32_t warpGroupSize_, embedDimsAlign_;
    uint32_t magic, shift;
};


extern "C" __global__ __aicore__ void multi_scale_deformable_attn_grad(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm,
    GM_ADDR level_start_index_gm, GM_ADDR sampling_loc_gm, GM_ADDR attn_weight_gm, GM_ADDR grad_output_gm,
    GM_ADDR grad_value_gm, GM_ADDR grad_sampling_loc_gm, GM_ADDR grad_attn_weight_gm, GM_ADDR workspace,
    GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    MultiScaleDeformableAttnGradKernel op(value_gm, spatial_shapes_gm, level_start_index_gm, sampling_loc_gm,
        attn_weight_gm, grad_output_gm, grad_value_gm, grad_sampling_loc_gm, grad_attn_weight_gm,
        &tilingData, &pipe);
    op.Process();
}