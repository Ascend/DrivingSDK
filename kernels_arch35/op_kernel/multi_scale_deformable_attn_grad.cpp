/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 */

#include "kernel_utils.h"
#include "kernel_operator.h"
#include "kernel_tpipe_impl.h"
#include "msda.h"

using namespace AscendC;
using namespace MicroAPI;

constexpr uint32_t taskOffset_ = 1024;
constexpr uint16_t taskRpt_ = taskOffset_ / B32_DATA_NUM_PER_REPEAT;


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
        MicroAPI::RegTensor<T> locationXReg;
        MicroAPI::RegTensor<T> locationYReg;
        MicroAPI::RegTensor<T> widthFloatReg;
        MicroAPI::RegTensor<T> heightFloatReg;
        MicroAPI::RegTensor<T> attentionWeightReg;

        MicroAPI::RegTensor<U> locationXIntReg;
        MicroAPI::RegTensor<U> locationYIntReg;

        MicroAPI::RegTensor<T> locationWidthLowReg;
        MicroAPI::RegTensor<T> locationWidthHighReg;
        MicroAPI::RegTensor<T> locationHeightLowReg;
        MicroAPI::RegTensor<T> locationHeightHighReg;

        MicroAPI::RegTensor<T> gradLocW1Reg;
        MicroAPI::RegTensor<T> gradLocW2Reg;
        MicroAPI::RegTensor<T> gradLocW3Reg;
        MicroAPI::RegTensor<T> gradLocW4Reg;

        MicroAPI::RegTensor<T> weight1Reg;
        MicroAPI::RegTensor<T> weight2Reg;
        MicroAPI::RegTensor<T> weight3Reg;
        MicroAPI::RegTensor<T> weight4Reg;

        MicroAPI::RegTensor<T> gradWeight1Reg;
        MicroAPI::RegTensor<T> gradWeight2Reg;
        MicroAPI::RegTensor<T> gradWeight3Reg;
        MicroAPI::RegTensor<T> gradWeight4Reg;

        MicroAPI::RegTensor<T> gradAttnReg;
        MicroAPI::RegTensor<T> gradLocationXY1Reg;
        MicroAPI::RegTensor<T> gradLocationXY2Reg;

        MicroAPI::MaskReg mask = MicroAPI::CreateMask<T,AscendC::MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg validMask = MicroAPI::CreateMask<T>();
        MicroAPI::MaskReg tmpMask = MicroAPI::CreateMask<T>();

        static constexpr AscendC::MicroAPI::CastTrait castF2ITrait = 
            {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};
        static constexpr AscendC::MicroAPI::CastTrait castI2FTrait = 
            {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        for (uint16_t taskIdx = 0; taskIdx < taskRpt_; ++taskIdx) {
            uint32_t localOffset = taskIdx * B32_DATA_NUM_PER_REPEAT;

            MicroAPI::DataCopy(locationXReg, locFloatPtr + localOffset);
            MicroAPI::DataCopy(locationYReg, locFloatPtr + localOffset + taskOffset_);
            MicroAPI::DataCopy(widthFloatReg, shapeFloatPtr + localOffset);
            MicroAPI::DataCopy(heightFloatReg, shapeFloatPtr + localOffset + taskOffset_);
            MicroAPI::DataCopy(attentionWeightReg, attentionWeightPtr + localOffset);

            MicroAPI::DataCopy(weight1Reg, weightPtr + localOffset);
            MicroAPI::DataCopy(weight2Reg, weightPtr + localOffset + taskOffset_);
            MicroAPI::DataCopy(weight3Reg, weightPtr + localOffset + 2 * taskOffset_);
            MicroAPI::DataCopy(weight4Reg, weightPtr + localOffset + 3 * taskOffset_);

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


template<typename T, typename U, U embedDims_>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void MSDASimtComputeGradSmallTemplate(
    __gm__ T* gradValueGm_, __gm__ T* valueGm_, __gm__ T* gradOutGm_,
    __ubuf__ T* locFloat, __ubuf__ T* shapeFloat, __ubuf__ U* locationInt,
    __ubuf__ T* attnWeight, __ubuf__ T* weight, U baseOffset, U count,
    U oneHeadNum_, U outDims_)
{
    U reduceGroupCount = Simt::GetWarpSize() / embedDims_;
    for (U idx = Simt::GetThreadIdx(); idx < count * embedDims_; idx += Simt::GetThreadNum()) {
        U pointIdx = idx / embedDims_;
        U channelIdx = idx % embedDims_;

        T x = locFloat[pointIdx];
        T y = locFloat[pointIdx + taskOffset_];
        T w = shapeFloat[pointIdx];
        T h = shapeFloat[pointIdx + taskOffset_];

        if (!(x > -1 && y > -1 && x < w && y < h)) {
            continue;
        }

        T lh = y - Simt::Floor(y);
        T lw = x - Simt::Floor(x);
        T hh = 1 - lh;
        T hw = 1 - lw;

        T w1 = hh * hw;
        T w2 = hh * lw;
        T w3 = lh * hw;
        T w4 = lh * lw;

        U gmOffset1 = locationInt[pointIdx] + channelIdx;
        U gmOffset2 = gmOffset1 + outDims_;
        U gmOffset3 = locationInt[pointIdx + taskOffset_] + channelIdx;
        U gmOffset4 = gmOffset3 + outDims_;

        T v1 = (y >= 0 && x >= 0) ? valueGm_[gmOffset1] : 0;
        T v2 = (y >= 0 && x < w - 1) ? valueGm_[gmOffset2] : 0;
        T v3 = (y < h - 1 && x >= 0) ? valueGm_[gmOffset3] : 0;
        T v4 = (y < h - 1 && x < w - 1) ? valueGm_[gmOffset4] : 0;

        U outOffset = pointIdx / oneHeadNum_ * embedDims_ + channelIdx;
        T grad = gradOutGm_[baseOffset + outOffset];
        T attn = attnWeight[pointIdx];
        T gradValueMul = grad * attn;

        if (y >= 0 && x >= 0) {
            Simt::AtomicAdd(gradValueGm_ + gmOffset1, w1 * gradValueMul);
        }
        if (y >= 0 && x < w - 1) {
            Simt::AtomicAdd(gradValueGm_ + gmOffset2, w2 * gradValueMul);
        }
        if (y < h - 1 && x >= 0) {
            Simt::AtomicAdd(gradValueGm_ + gmOffset3, w3 * gradValueMul);
        }
        if (y < h - 1 && x < w - 1) {
            Simt::AtomicAdd(gradValueGm_ + gmOffset4, w4 * gradValueMul);
        }

        T val1, val2, val3, val4;
        U groupIdx = (idx % Simt::GetWarpSize()) / embedDims_;
        for (U groupLoop = 0; groupLoop < reduceGroupCount; groupLoop++) {
            T gradWeight1 = groupIdx == groupLoop ? v1 * grad : 0;
            T gradWeight2 = groupIdx == groupLoop ? v2 * grad : 0;
            T gradWeight3 = groupIdx == groupLoop ? v3 * grad : 0;
            T gradWeight4 = groupIdx == groupLoop ? v4 * grad : 0;

            T sum1 = Simt::WarpReduceAddSync(gradWeight1);
            T sum2 = Simt::WarpReduceAddSync(gradWeight2);
            T sum3 = Simt::WarpReduceAddSync(gradWeight3);
            T sum4 = Simt::WarpReduceAddSync(gradWeight4);

            if (groupIdx == groupLoop) {
                val1 = sum1, val2 = sum2, val3 = sum3, val4 = sum4;
            }
        }

        if (idx % embedDims_ == 0) {
            weight[pointIdx] = val1;
            weight[pointIdx + taskOffset_] = val2;
            weight[pointIdx + 2 * taskOffset_] = val3;
            weight[pointIdx + 3 * taskOffset_] = val4;
        }
    }
}


template<typename T, typename U, U embedDims_>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void MSDASimtComputeGradLargeTemplate(
    __gm__ T* gradValueGm_, __gm__ T* valueGm_, __gm__ T* gradOutGm_,
    __ubuf__ T* locFloat, __ubuf__ T* shapeFloat, __ubuf__ U* locationInt,
    __ubuf__ T* attnWeight, __ubuf__ T* weight, U baseOffset, U count,
    U oneHeadNum_, U outDims_)
{
    U innerLoops_ = embedDims_ / Simt::GetWarpSize();
    for (U idx = Simt::GetThreadIdx(); idx < count * embedDims_; idx += Simt::GetThreadNum()) {
        U pointIdx = idx / Simt::GetWarpSize();
        U channelIdx = idx % Simt::GetWarpSize();

        T x = locFloat[pointIdx];
        T y = locFloat[pointIdx + taskOffset_];
        T w = shapeFloat[pointIdx];
        T h = shapeFloat[pointIdx + taskOffset_];

        if (!(x > -1 && y > -1 && x < w && y < h)) {
            continue;
        }

        T lh = y - Simt::Floor(y);
        T lw = x - Simt::Floor(x);
        T hh = 1 - lh;
        T hw = 1 - lw;

        T w1 = hh * hw;
        T w2 = hh * lw;
        T w3 = lh * hw;
        T w4 = lh * lw;

        U gmOffset1 = locationInt[pointIdx] + channelIdx;
        U gmOffset2 = gmOffset1 + outDims_;
        U gmOffset3 = locationInt[pointIdx + taskOffset_] + channelIdx;
        U gmOffset4 = gmOffset3 + outDims_;
        T attn = attnWeight[pointIdx];

        T value1 = 0, value2 = 0, value3 = 0, value4 = 0;
        for (U innerIdx = 0; innerIdx < innerLoops_; innerIdx++) {
            U channelOffset = Simt::GetWarpSize() * innerIdx;

            T v1 = (y >= 0 && x >= 0) ? valueGm_[gmOffset1 + channelOffset] : 0;
            T v2 = (y >= 0 && x < w - 1) ? valueGm_[gmOffset2 + channelOffset] : 0;
            T v3 = (y < h - 1 && x >= 0) ? valueGm_[gmOffset3 + channelOffset] : 0;
            T v4 = (y < h - 1 && x < w - 1) ? valueGm_[gmOffset4 + channelOffset] : 0;

            T grad = gradOutGm_[baseOffset + pointIdx / oneHeadNum_ * embedDims_ + channelIdx + channelOffset];
            T gradValueMul = grad * attn;

            if (y >= 0 && x >= 0) {
                Simt::AtomicAdd(gradValueGm_ + gmOffset1 + channelOffset, w1 * gradValueMul);
            }
            if (y >= 0 && x < w - 1) {
                Simt::AtomicAdd(gradValueGm_ + gmOffset2 + channelOffset, w2 * gradValueMul);
            }
            if (y < h - 1 && x >= 0) {
                Simt::AtomicAdd(gradValueGm_ + gmOffset3 + channelOffset, w3 * gradValueMul);
            }
            if (y < h - 1 && x < w - 1) {
                Simt::AtomicAdd(gradValueGm_ + gmOffset4 + channelOffset, w4 * gradValueMul);
            }
            value1 = value1 + v1 * grad;
            value2 = value2 + v2 * grad;
            value3 = value3 + v3 * grad;
            value4 = value4 + v4 * grad;
        }

        value1 = Simt::WarpReduceAddSync(value1);
        value2 = Simt::WarpReduceAddSync(value2);
        value3 = Simt::WarpReduceAddSync(value3);
        value4 = Simt::WarpReduceAddSync(value4);

        if (idx % Simt::GetWarpSize()) {
            weight[pointIdx] = value1;
            weight[pointIdx + taskOffset_] = value2;
            weight[pointIdx + 2 * taskOffset_] = value3;
            weight[pointIdx + 3 * taskOffset_] = value4;
        }
    }
}


template<typename T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void MSDASimtComputeGradSmall(
    __gm__ T* gradValueGm_, __gm__ T* valueGm_, __gm__ T* gradOutGm_,
    __ubuf__ T* locFloat, __ubuf__ T* shapeFloat, __ubuf__ U* locationInt,
    __ubuf__ T* attnWeight, __ubuf__ T* weight, U baseOffset, U count,
    U oneHeadNum_, U outDims_, U embedDims_)
{
    for (U idx = Simt::GetThreadIdx(); idx < count * Simt::GetWarpSize(); idx += Simt::GetThreadNum()) {
        U pointIdx = idx / Simt::GetWarpSize();
        U channelIdx = idx % Simt::GetWarpSize();
        if (channelIdx >= embedDims_) {
            continue;
        }

        T x = locFloat[pointIdx];
        T y = locFloat[pointIdx + taskOffset_];
        T w = shapeFloat[pointIdx];
        T h = shapeFloat[pointIdx + taskOffset_];

        if (!(x > -1 && y > -1 && x < w && y < h)) {
            continue;
        }

        T lh = y - Simt::Floor(y);
        T lw = x - Simt::Floor(x);
        T hh = 1 - lh;
        T hw = 1 - lw;

        T w1 = hh * hw;
        T w2 = hh * lw;
        T w3 = lh * hw;
        T w4 = lh * lw;

        U gmOffset1 = locationInt[pointIdx] + channelIdx;
        U gmOffset2 = gmOffset1 + outDims_;
        U gmOffset3 = locationInt[pointIdx + taskOffset_] + channelIdx;
        U gmOffset4 = gmOffset3 + outDims_;

        T v1 = (y >= 0 && x >= 0) ? valueGm_[gmOffset1] : 0;
        T v2 = (y >= 0 && x < w - 1) ? valueGm_[gmOffset2] : 0;
        T v3 = (y < h - 1 && x >= 0) ? valueGm_[gmOffset3] : 0;
        T v4 = (y < h - 1 && x < w - 1) ? valueGm_[gmOffset4] : 0;

        U outOffset = pointIdx / oneHeadNum_ * embedDims_ + channelIdx;
        T grad = gradOutGm_[baseOffset + outOffset];
        T attn = attnWeight[pointIdx];
        T gradValueMul = grad * attn;

        if (y >= 0 && x >= 0) {
            Simt::AtomicAdd(gradValueGm_ + gmOffset1, w1 * gradValueMul);
        }
        if (y >= 0 && x < w - 1) {
            Simt::AtomicAdd(gradValueGm_ + gmOffset2, w2 * gradValueMul);
        }
        if (y < h - 1 && x >= 0) {
            Simt::AtomicAdd(gradValueGm_ + gmOffset3, w3 * gradValueMul);
        }
        if (y < h - 1 && x < w - 1) {
            Simt::AtomicAdd(gradValueGm_ + gmOffset4, w4 * gradValueMul);
        }

        T val1 = Simt::WarpReduceAddSync(v1 * grad);
        T val2 = Simt::WarpReduceAddSync(v2 * grad);
        T val3 = Simt::WarpReduceAddSync(v3 * grad);
        T val4 = Simt::WarpReduceAddSync(v4 * grad);

        if (idx % Simt::GetWarpSize() == 0) {
            weight[pointIdx] = val1;
            weight[pointIdx + taskOffset_] = val2;
            weight[pointIdx + 2 * taskOffset_] = val3;
            weight[pointIdx + 3 * taskOffset_] = val4;
        }
    }
}

template<typename T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void MSDASimtComputeGradLarge(
    __gm__ T* gradValueGm_, __gm__ T* valueGm_, __gm__ T* gradOutGm_,
    __ubuf__ T* locFloat, __ubuf__ T* shapeFloat, __ubuf__ U* locationInt,
    __ubuf__ T* attnWeight, __ubuf__ T* weight, U baseOffset, U count,
    U oneHeadNum_, U outDims_, U embedDims_)
{
    for (U idx = Simt::GetThreadIdx(); idx < count * Simt::GetWarpSize(); idx += Simt::GetThreadNum()) {
        U pointIdx = idx / Simt::GetWarpSize();
        U channelIdx = idx % Simt::GetWarpSize();

        T x = locFloat[pointIdx];
        T y = locFloat[pointIdx + taskOffset_];
        T w = shapeFloat[pointIdx];
        T h = shapeFloat[pointIdx + taskOffset_];

        if (!(x > -1 && y > -1 && x < w && y < h)) {
            continue;
        }

        T lh = y - Simt::Floor(y);
        T lw = x - Simt::Floor(x);
        T hh = 1 - lh;
        T hw = 1 - lw;

        T w1 = hh * hw;
        T w2 = hh * lw;
        T w3 = lh * hw;
        T w4 = lh * lw;

        U gmOffset1 = locationInt[pointIdx];
        U gmOffset2 = gmOffset1 + outDims_;
        U gmOffset3 = locationInt[pointIdx + taskOffset_];
        U gmOffset4 = gmOffset3 + outDims_;
        T attn = attnWeight[pointIdx];

        T value1 = 0, value2 = 0, value3 = 0, value4 = 0;
        for (; channelIdx < embedDims_; channelIdx += Simt::GetWarpSize()) {
            T v1 = (y >= 0 && x >= 0) ? valueGm_[gmOffset1 + channelIdx] : 0;
            T v2 = (y >= 0 && x < w - 1) ? valueGm_[gmOffset2 + channelIdx] : 0;
            T v3 = (y < h - 1 && x >= 0) ? valueGm_[gmOffset3 + channelIdx] : 0;
            T v4 = (y < h - 1 && x < w - 1) ? valueGm_[gmOffset4 + channelIdx] : 0;

            T grad = gradOutGm_[baseOffset + pointIdx / oneHeadNum_ * embedDims_ + channelIdx];
            T gradValueMul = grad * attn;

            if (y >= 0 && x >= 0) {
                Simt::AtomicAdd(gradValueGm_ + gmOffset1 + channelIdx, w1 * gradValueMul);
            }
            if (y >= 0 && x < w - 1) {
                Simt::AtomicAdd(gradValueGm_ + gmOffset2 + channelIdx, w2 * gradValueMul);
            }
            if (y < h - 1 && x >= 0) {
                Simt::AtomicAdd(gradValueGm_ + gmOffset3 + channelIdx, w3 * gradValueMul);
            }
            if (y < h - 1 && x < w - 1) {
                Simt::AtomicAdd(gradValueGm_ + gmOffset4 + channelIdx, w4 * gradValueMul);
            }

            value1 = value1 + v1 * grad;
            value2 = value2 + v2 * grad;
            value3 = value3 + v3 * grad;
            value4 = value4 + v4 * grad;
        }

        value1 = Simt::WarpReduceAddSync(value1);
        value2 = Simt::WarpReduceAddSync(value2);
        value3 = Simt::WarpReduceAddSync(value3);
        value4 = Simt::WarpReduceAddSync(value4);

        if (idx % Simt::GetWarpSize()) {
            weight[pointIdx] = value1;
            weight[pointIdx + taskOffset_] = value2;
            weight[pointIdx + 2 * taskOffset_] = value3;
            weight[pointIdx + 3 * taskOffset_] = value4;
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
            ComputeGmOffsetVF<float, int32_t>(taskRpt_, numHeads_, embedDims_, baseSrcOffset, nextSrcOffset, baseNum * oneQueryNum_, locationFloat, shapeFloat, offsetInt, locationInt);
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
        switch (embedDims_) {
            case 8:
                AscendC::Simt::VF_CALL<MSDASimtComputeGradSmallTemplate<float, int32_t, 8>>(AscendC::Simt::Dim3{1024},
                    (__gm__ float*)gradValueGm_.GetPhyAddr(), (__gm__ float*)valueGm_.GetPhyAddr(), (__gm__ float*)gradOutGm_.GetPhyAddr(),
                    (__ubuf__ float*)locationFloat.GetPhyAddr(), (__ubuf__ float*)shapeFloat.GetPhyAddr(), (__ubuf__ int32_t*)locationInt.GetPhyAddr(),
                    (__ubuf__ float*)attentionWeights.GetPhyAddr(), (__ubuf__ float*)weight.GetPhyAddr(),
                    taskIdx * outDims_, taskNum * oneQueryNum_, oneHeadNum_, outDims_);
                break;
            case 16:
                AscendC::Simt::VF_CALL<MSDASimtComputeGradSmallTemplate<float, int32_t, 16>>(AscendC::Simt::Dim3{1024},
                    (__gm__ float*)gradValueGm_.GetPhyAddr(), (__gm__ float*)valueGm_.GetPhyAddr(), (__gm__ float*)gradOutGm_.GetPhyAddr(),
                    (__ubuf__ float*)locationFloat.GetPhyAddr(), (__ubuf__ float*)shapeFloat.GetPhyAddr(), (__ubuf__ int32_t*)locationInt.GetPhyAddr(),
                    (__ubuf__ float*)attentionWeights.GetPhyAddr(), (__ubuf__ float*)weight.GetPhyAddr(),
                    taskIdx * outDims_, taskNum * oneQueryNum_, oneHeadNum_, outDims_);
                break;
            case 32:
                AscendC::Simt::VF_CALL<MSDASimtComputeGradSmallTemplate<float, int32_t, 32>>(AscendC::Simt::Dim3{1024},
                    (__gm__ float*)gradValueGm_.GetPhyAddr(), (__gm__ float*)valueGm_.GetPhyAddr(), (__gm__ float*)gradOutGm_.GetPhyAddr(),
                    (__ubuf__ float*)locationFloat.GetPhyAddr(), (__ubuf__ float*)shapeFloat.GetPhyAddr(), (__ubuf__ int32_t*)locationInt.GetPhyAddr(),
                    (__ubuf__ float*)attentionWeights.GetPhyAddr(), (__ubuf__ float*)weight.GetPhyAddr(),
                    taskIdx * outDims_, taskNum * oneQueryNum_, oneHeadNum_, outDims_);
                break;
            case 64:
                AscendC::Simt::VF_CALL<MSDASimtComputeGradLargeTemplate<float, int32_t, 64>>(AscendC::Simt::Dim3{1024},
                    (__gm__ float*)gradValueGm_.GetPhyAddr(), (__gm__ float*)valueGm_.GetPhyAddr(), (__gm__ float*)gradOutGm_.GetPhyAddr(),
                    (__ubuf__ float*)locationFloat.GetPhyAddr(), (__ubuf__ float*)shapeFloat.GetPhyAddr(), (__ubuf__ int32_t*)locationInt.GetPhyAddr(),
                    (__ubuf__ float*)attentionWeights.GetPhyAddr(), (__ubuf__ float*)weight.GetPhyAddr(),
                    taskIdx * outDims_, taskNum * oneQueryNum_, oneHeadNum_, outDims_);
                break;
            case 128:
                AscendC::Simt::VF_CALL<MSDASimtComputeGradLargeTemplate<float, int32_t, 128>>(AscendC::Simt::Dim3{1024},
                    (__gm__ float*)gradValueGm_.GetPhyAddr(), (__gm__ float*)valueGm_.GetPhyAddr(), (__gm__ float*)gradOutGm_.GetPhyAddr(),
                    (__ubuf__ float*)locationFloat.GetPhyAddr(), (__ubuf__ float*)shapeFloat.GetPhyAddr(), (__ubuf__ int32_t*)locationInt.GetPhyAddr(),
                    (__ubuf__ float*)attentionWeights.GetPhyAddr(), (__ubuf__ float*)weight.GetPhyAddr(),
                    taskIdx * outDims_, taskNum * oneQueryNum_, oneHeadNum_, outDims_);
                break;
            case 256:
                AscendC::Simt::VF_CALL<MSDASimtComputeGradLargeTemplate<float, int32_t, 256>>(AscendC::Simt::Dim3{1024},
                    (__gm__ float*)gradValueGm_.GetPhyAddr(), (__gm__ float*)valueGm_.GetPhyAddr(), (__gm__ float*)gradOutGm_.GetPhyAddr(),
                    (__ubuf__ float*)locationFloat.GetPhyAddr(), (__ubuf__ float*)shapeFloat.GetPhyAddr(), (__ubuf__ int32_t*)locationInt.GetPhyAddr(),
                    (__ubuf__ float*)attentionWeights.GetPhyAddr(), (__ubuf__ float*)weight.GetPhyAddr(),
                    taskIdx * outDims_, taskNum * oneQueryNum_, oneHeadNum_, outDims_);
                break;
            default:
                if (embedDims_ <= 32) {
                    AscendC::Simt::VF_CALL<MSDASimtComputeGradSmall<float, int32_t>>(AscendC::Simt::Dim3{1024},
                        (__gm__ float*)gradValueGm_.GetPhyAddr(), (__gm__ float*)valueGm_.GetPhyAddr(), (__gm__ float*)gradOutGm_.GetPhyAddr(),
                        (__ubuf__ float*)locationFloat.GetPhyAddr(), (__ubuf__ float*)shapeFloat.GetPhyAddr(), (__ubuf__ int32_t*)locationInt.GetPhyAddr(),
                        (__ubuf__ float*)attentionWeights.GetPhyAddr(), (__ubuf__ float*)weight.GetPhyAddr(),
                        taskIdx * outDims_, taskNum * oneQueryNum_, oneHeadNum_, outDims_, embedDims_);
                } else {
                    AscendC::Simt::VF_CALL<MSDASimtComputeGradLarge<float, int32_t>>(AscendC::Simt::Dim3{1024},
                        (__gm__ float*)gradValueGm_.GetPhyAddr(), (__gm__ float*)valueGm_.GetPhyAddr(), (__gm__ float*)gradOutGm_.GetPhyAddr(),
                        (__ubuf__ float*)locationFloat.GetPhyAddr(), (__ubuf__ float*)shapeFloat.GetPhyAddr(), (__ubuf__ int32_t*)locationInt.GetPhyAddr(),
                        (__ubuf__ float*)attentionWeights.GetPhyAddr(), (__ubuf__ float*)weight.GetPhyAddr(),
                        taskIdx * outDims_, taskNum * oneQueryNum_, oneHeadNum_, outDims_, embedDims_);
                }
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
                        shapeInt.SetValue(idx, w);
                        shapeInt.SetValue(idx + alignedOneTaskNum_, h);
                        offsetInt.SetValue(idx, o * numHeads_ + head);
                        ++idx;
                    }
                }
            }
        }
        Cast<float, int32_t, false>(
            shapeFloat, shapeInt, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 2 * taskRpt_, {1, 1, 8, 8});
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
    TBuf<TPosition::VECCALC> shapeFloatBuf_, offsetIntBuf_, weightBuf_, gmOffsetbuf_;
    TBuf<TPosition::VECCALC> gradLocationQue_, gradAttentionWeightsQue_;

    int32_t blkIdx_;
    // const values
    uint32_t coreNum_, compTaskNum_;
    uint32_t batchSize_, numKeys_, numHeads_, embedDims_, outDims_, numLevels_, numQueries_, numPoints_, realLevels_;
    uint32_t alignedOneTaskNum_;
    uint32_t oneHeadNum_, oneQueryNum_;
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