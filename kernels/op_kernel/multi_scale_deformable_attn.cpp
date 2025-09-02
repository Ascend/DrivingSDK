/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 */

#include "kernel_utils.h"
#include "msda.h"

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnKernel<aligned, fastMode>::UpdateParams(uint32_t tailCompNum)
{
    this->compTaskNum_ = tailCompNum;
    if constexpr (fastMode) {
        this->outerLoops_ = this->compTaskNum_;
    } else {
        this->outerLoops_ = this->compTaskNum_ * this->numHeads_;
    }
    this->cpOutParams_.blockCount = this->compTaskNum_ * this->numHeads_;
    if (fastMode) {
        this->cpSampleParams_.blockCount = this->compTaskNum_;
        this->cpDoubleSampleParams_.blockCount = this->compTaskNum_;
    } else {
        this->cpSampleParams_.blockCount = this->compTaskNum_ * this->numHeads_;
        this->cpDoubleSampleParams_.blockCount = this->compTaskNum_ * this->numHeads_;
    }
}

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnKernel<aligned, fastMode>::CopyFullPoint(
    const LocalTensor<int32_t>& location, const LocalTensor<float>& value,
    uint64_t valid, uint32_t baseIdx, uint32_t innerLoops)
{
    for (int32_t i = ScalarGetSFFValue<1>(valid); i < innerLoops && i >= 0;
        i = ScalarGetSFFValue<1>(valid)) {
        valid = sbitset0(valid, i);
        uint32_t idx = baseIdx + i;
        // WARN: dangerous!
        int32_t gmY0Offset = location.GetValue(idx);
        int32_t gmY1Offset = location.GetValue(idx + this->alignedOneTaskNum_);
        this->CopyInValue(value[i * this->alignedEmbedDims_], this->valueGm_[gmY0Offset], this->cpRowDoubleParams_);
        this->CopyInValue(value[i * this->alignedEmbedDims_ + 2 * this->alignedCornerEmbedDims_],
            this->valueGm_[gmY1Offset], this->cpRowDoubleParams_);
    }
}

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnKernel<aligned, fastMode>::CopyBorderPoint(
    const LocalTensor<int32_t>& location, const LocalTensor<float>& value,
    const LocalTensor<int32_t>& shapeInt, const LocalTensor<int32_t>& loc,
    uint64_t valid, uint32_t baseIdx, uint32_t innerLoops)
{
    for (int32_t i = ScalarGetSFFValue<0>(valid); i < innerLoops && i >= 0;
        i = ScalarGetSFFValue<0>(valid)) {
        valid = sbitset1(valid, i);
        uint32_t idx = baseIdx + i;
        int32_t w = shapeInt.GetValue(idx);
        int32_t x = loc.GetValue(idx);
        // WARN: dangerous!
        int32_t gmOffset = location.GetValue(idx);
        if (x != -1) {
            this->CopyInValue(value[i * this->alignedEmbedDims_], this->valueGm_[gmOffset], this->cpOneValParams_);
        }
        if (x != w - 1) {
            this->CopyInValue(value[i * this->alignedEmbedDims_ + this->alignedCornerEmbedDims_],
                this->valueGm_[gmOffset + this->outDims_], this->cpOneValParams_);
        }
    }
}

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnKernel<aligned, fastMode>::ComputeBilinearInterpolation(
    const LocalTensor<uint64_t>& validFlag, const LocalTensor<int32_t>& shapeInt, const LocalTensor<int32_t>& location,
    const LocalTensor<int32_t>& loc, const LocalTensor<float>& shapeFloat, const LocalTensor<float>& production,
    const LocalTensor<float>& value, const LocalTensor<float>& locFloat, const LocalTensor<float>& weight,
    const LocalTensor<float>& attentionWeight, const LocalTensor<float>& cornerWeightBrc, const LocalTensor<float>& output)
{
    WaitFlag<HardEvent::V_MTE2>(this->biEvt_);
    for (uint32_t head = 0; head < this->outerLoops_; ++head) {
        uint64_t baseIdx = head * this->innerLoopsAligned_;
        uint64_t headOffset = baseIdx / B32_DATA_NUM_PER_REPEAT;
        uint64_t byteOffset = baseIdx - headOffset * B32_DATA_NUM_PER_REPEAT;
        uint64_t valid = validFlag.GetValue(headOffset) >> byteOffset;
        uint64_t bottomValid = validFlag.GetValue(headOffset + 2 * this->validFlagMaskLen_ / 8) >> byteOffset;
        uint64_t topValid = validFlag.GetValue(headOffset + 3 * this->validFlagMaskLen_ / 8) >> byteOffset;
        uint32_t outOffset = fastMode ? head * this->numHeads_ * this->alignedEmbedDims_ : head * this->alignedEmbedDims_;
        uint32_t bufferFlag = head % 2;
        LocalTensor<float> valueSrc = value[bufferFlag * this->cornerRpt_ * B32_DATA_NUM_PER_REPEAT];

        WaitFlag<HardEvent::V_MTE2>(bufferFlag);
        CopyFullPoint(location, valueSrc, valid, baseIdx, this->innerLoops_);
        if (head == 0) {
            this->ComputeWeight(locFloat, shapeFloat, production, weight, attentionWeight);
        }
        for (uint32_t i = 0; i < 4; ++i) {
            Brcb(cornerWeightBrc[i * this->alignedCornerEmbedDims_], weight[baseIdx + i * this->alignedOneTaskNum_],
                (fastMode ? this->alignedOneQueryNum_ : this->alignedOneHeadNum_) / B32_DATA_NUM_PER_BLOCK,
                {this->embedBlk_, static_cast<uint16_t>(8 * this->embedBlk_)});
        }
        CopyBorderPoint(location, valueSrc, shapeInt, loc, bottomValid, baseIdx, this->innerLoops_);
        CopyBorderPoint(location[this->alignedOneTaskNum_], valueSrc[2 * this->alignedCornerEmbedDims_],
            shapeInt, loc, topValid, baseIdx, this->innerLoops_);
        SetFlag<HardEvent::MTE2_V>(bufferFlag);
        for (uint32_t i = 1; i < this->embedBlk_; ++i) {
            Adds<float, false>(cornerWeightBrc[i * B32_DATA_NUM_PER_BLOCK], cornerWeightBrc, 0.f, MASK_PLACEHOLDER,
                this->brcRpt_,
                {this->embedBlk_, this->embedBlk_, static_cast<uint8_t>(8 * this->embedBlk_),
                    static_cast<uint8_t>(8 * this->embedBlk_)});
        }
        WaitFlag<HardEvent::MTE2_V>(bufferFlag);

        if (unlikely(this->cornerRpt_ > MAX_REPEAT_TIMES)) {
            Mul<float, false>(
                cornerWeightBrc, valueSrc, cornerWeightBrc, MASK_PLACEHOLDER, this->cornerRpt_ / 2, {1, 1, 1, 8, 8, 8});
            Duplicate<float, false>(valueSrc, 0.f, MASK_PLACEHOLDER, this->cornerRpt_ / 2, 1, 8);
            Mul<float, false>(cornerWeightBrc[this->cornerRpt_ / 2 * B32_DATA_NUM_PER_REPEAT],
                valueSrc[this->cornerRpt_ / 2 * B32_DATA_NUM_PER_REPEAT],
                cornerWeightBrc[this->cornerRpt_ / 2 * B32_DATA_NUM_PER_REPEAT], MASK_PLACEHOLDER, this->cornerRpt_ / 2,
                {1, 1, 1, 8, 8, 8});
            Duplicate<float, false>(valueSrc[this->cornerRpt_ / 2 * B32_DATA_NUM_PER_REPEAT], 0.f, MASK_PLACEHOLDER,
                this->cornerRpt_ / 2, 1, 8);
        } else {
            Mul<float, false>(
                cornerWeightBrc, valueSrc, cornerWeightBrc, MASK_PLACEHOLDER, this->cornerRpt_, {1, 1, 1, 8, 8, 8});
            Duplicate<float, false>(valueSrc, 0.f, MASK_PLACEHOLDER, this->cornerRpt_, 1, 8);
        }
        SetFlag<HardEvent::V_MTE2>(bufferFlag);

        Add<float>(cornerWeightBrc, cornerWeightBrc[2 * this->alignedCornerEmbedDims_], cornerWeightBrc,
            2 * this->alignedCornerEmbedDims_);
        Add<float>(cornerWeightBrc, cornerWeightBrc[this->alignedCornerEmbedDims_], cornerWeightBrc,
            this->alignedCornerEmbedDims_);

        if (unlikely(head == 0)) {
            WaitFlag<HardEvent::MTE3_V>(0);
            Duplicate<float>(output, 0.f, this->compTaskNum_ * this->numHeads_ * this->alignedEmbedDims_);
        }
        SetVectorMask<float>(0, this->embedMask_);
        if (fastMode) {
            for (uint32_t i = 0; i < this->numHeads_; ++i) {
                Add<float, false>(output[outOffset], cornerWeightBrc[i * this->oneHeadNum_ * this->alignedEmbedDims_], output[outOffset],
                    MASK_PLACEHOLDER, this->oneHeadNum_, {1, 1, 1, 0, static_cast<uint8_t>(this->embedBlk_), 0});
                outOffset += this->alignedEmbedDims_;
            }
        } else {
            Add<float, false>(output[outOffset], cornerWeightBrc, output[outOffset], MASK_PLACEHOLDER,
                this->oneHeadNum_, {1, 1, 1, 0, static_cast<uint8_t>(this->embedBlk_), 0});
            outOffset += this->alignedEmbedDims_;
        }
        ResetMask();
    }
    SetFlag<HardEvent::V_MTE3>(0);
}

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnKernel<aligned, fastMode>::Process()
{
    LocalTensor<float> locationFloat = this->locationQue_.template Get<float>();
    LocalTensor<int32_t> locationInt = this->locationQue_.template Get<int32_t>();
    LocalTensor<float> attentionWeight = this->attentionWeightsQue_.template Get<float>();
    LocalTensor<int32_t> shapes = this->shapeQue_.template Get<int32_t>();
    LocalTensor<int32_t> offset = this->offsetQue_.template Get<int32_t>();
    LocalTensor<float> shapeFloat = this->shapeFloatBuf_.template Get<float>();
    LocalTensor<int32_t> shapeInt = this->shapeIntBuf_.template Get<int32_t>();
    LocalTensor<int32_t> offsetInt = this->offsetIntBuf_.template Get<int32_t>();
    LocalTensor<float> value = this->valueQue_.template Get<float>();
    LocalTensor<float> cornerWeightBrc = this->cornerWeightBrcBuf_.template Get<float>();
    LocalTensor<float> output = this->outputQue_.template Get<float>();
    LocalTensor<uint64_t> validFlag = this->validFlagBuf_.template Get<uint64_t>();

    LocalTensor<int32_t> locInt = this->locIntBuf_.template Get<int32_t>();
    LocalTensor<float> locFloat = this->locFloatBuf_.template Get<float>();
    LocalTensor<float> production = this->productionBuf_.template Get<float>();
    LocalTensor<float> weight = this->weightBuf_.template Get<float>();

    this->PrepareShape(shapes, shapeInt, shapeFloat, offset, offsetInt);
    // note that the repeat times can be 256 when one head num comes to 64 and embeddims comes to 64
    if (unlikely(this->cornerRpt_ > MAX_REPEAT_TIMES)) {
        Duplicate<float, false>(value, 0.f, MASK_PLACEHOLDER, this->cornerRpt_ / 2, 1, 8);
        Duplicate<float, false>(value[this->cornerRpt_ / 2 * B32_DATA_NUM_PER_REPEAT], 0.f, MASK_PLACEHOLDER, this->cornerRpt_ / 2, 1, 8);
        Duplicate<float, false>(value[this->cornerRpt_ * B32_DATA_NUM_PER_REPEAT], 0.f, MASK_PLACEHOLDER, this->cornerRpt_ / 2, 1, 8);
        Duplicate<float, false>(value[this->cornerRpt_ * 3 / 2 * B32_DATA_NUM_PER_REPEAT], 0.f, MASK_PLACEHOLDER, this->cornerRpt_ / 2, 1, 8);
    } else {
        Duplicate<float, false>(value, 0.f, MASK_PLACEHOLDER, this->cornerRpt_, 1, 8);
        Duplicate<float, false>(value[this->cornerRpt_ * B32_DATA_NUM_PER_REPEAT], 0.f, MASK_PLACEHOLDER, this->cornerRpt_, 1, 8);
    }

    SetFlag<HardEvent::V_MTE2>(this->copyEvt_);
    SetFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE2>(1);
    SetFlag<HardEvent::MTE3_V>(0);

    for (uint32_t taskIdx = this->startOffset_; taskIdx < this->endOffset_; taskIdx+=this->compTaskNum_) {
        if (unlikely(taskIdx + this->compTaskNum_ > this->endOffset_)) {
            UpdateParams(this->endOffset_ - taskIdx);
        }
        this->CopyInSample(locationFloat[2 * this->alignedOneTaskNum_], attentionWeight, taskIdx);
        this->ComputeLocation(taskIdx, locationFloat, attentionWeight, locationInt, shapeFloat, shapeInt, locFloat, locInt, offsetInt,
            validFlag.ReinterpretCast<uint8_t>());
        ComputeBilinearInterpolation(validFlag, shapeInt, locationInt, locInt, shapeFloat, production, value, locFloat,
            weight, attentionWeight, cornerWeightBrc, output);
        CopyOut(output, taskIdx);
    }
    WaitFlag<HardEvent::V_MTE2>(this->copyEvt_);
    WaitFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(1);
    WaitFlag<HardEvent::MTE3_V>(0);
}

extern "C" __global__ __aicore__ void multi_scale_deformable_attn(GM_ADDR value, GM_ADDR valueSpatialShapes,
    GM_ADDR valueLevelStartIndex, GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(11)) {
        MultiScaleDeformableAttnKernel<true, true> op(value, valueSpatialShapes, valueLevelStartIndex,
            samplingLocations, attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(01)) {
        MultiScaleDeformableAttnKernel<false, true> op(value, valueSpatialShapes, valueLevelStartIndex,
            samplingLocations, attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(10)) {
        MultiScaleDeformableAttnKernel<true, false> op(value, valueSpatialShapes, valueLevelStartIndex,
            samplingLocations, attentionWeights, output, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(00)) {
        MultiScaleDeformableAttnKernel<false, false> op(value, valueSpatialShapes, valueLevelStartIndex,
            samplingLocations, attentionWeights, output, &tilingData, &pipe);
        op.Process();
    }
}
