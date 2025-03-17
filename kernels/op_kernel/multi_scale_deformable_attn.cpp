/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 */

#include "kernel_utils.h"
#include "msda.h"

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnKernel<aligned, fastMode>::ComputeBilinearInterpolation(
    const LocalTensor<uint64_t>& validFlag, const LocalTensor<int32_t>& shapeInt, const LocalTensor<int32_t>& location,
    const LocalTensor<int32_t>& loc, const LocalTensor<float>& shapeFloat, const LocalTensor<float>& production,
    const LocalTensor<float>& value, const LocalTensor<float>& locFloat, const LocalTensor<float>& weight,
    const LocalTensor<float>& attentionWeight, const LocalTensor<float>& cornerWeightBrc,
    const LocalTensor<float>& output)
{
    WaitFlag<HardEvent::V_MTE2>(this->biEvt_);
    for (uint32_t head = 0; head < this->outerLoops_; ++head) {
        uint64_t valid = validFlag.GetValue(head);
        uint64_t bottomInvalid = validFlag.GetValue(head + 2 * this->validFlagMaskLen_ / 8);
        uint64_t topInvalid = validFlag.GetValue(head + 3 * this->validFlagMaskLen_ / 8);
        uint32_t outOffset = head * this->alignedEmbedDims_;
        uint32_t baseIdx = head * this->alignedOneHeadNum_;
        WaitFlag<HardEvent::V_MTE2>(0);
        for (int32_t i = ScalarGetSFFValue<1>(valid); i < this->innerLoops_ && i >= 0;
            i = ScalarGetSFFValue<1>(valid)) {
            valid = sbitset0(valid, i);
            uint32_t idx = baseIdx + i;
            int32_t w = shapeInt.GetValue(idx);
            // WARN: dangerous!
            uint64_t gmOffset = static_cast<uint64_t>(location.GetValue(idx));
            this->CopyInValue(value[i * this->alignedEmbedDims_], this->valueGm_[gmOffset], this->cpRowDoubleParams_);
            this->CopyInValue(value[i * this->alignedEmbedDims_ + 2 * this->alignedCornerEmbedDims_],
                this->valueGm_[gmOffset + w * this->outDims_], this->cpRowDoubleParams_);
        }
        if (head == 0) {
            this->ComputeWeight(locFloat, shapeFloat, production, weight, attentionWeight);
        }
        for (uint32_t i = 0; i < 4; ++i) {
            Brcb(cornerWeightBrc[i * this->alignedCornerEmbedDims_], weight[baseIdx + i * this->alignedOneQueryNum_],
                (fastMode ? this->alignedOneQueryNum_ : this->alignedOneHeadNum_) / B32_DATA_NUM_PER_BLOCK,
                {this->embedBlk_, static_cast<uint16_t>(8 * this->embedBlk_)});
        }
        for (int32_t i = ScalarGetSFFValue<0>(bottomInvalid); i < this->innerLoops_ && i >= 0;
            i = ScalarGetSFFValue<0>(bottomInvalid)) {
            bottomInvalid = sbitset1(bottomInvalid, i);
            uint32_t idx = baseIdx + i;
            int32_t w = shapeInt.GetValue(idx);
            int32_t x = loc.GetValue(idx);
            // WARN: dangerous!
            uint64_t gmOffset = static_cast<uint64_t>(location.GetValue(idx));
            if (x != -1) {
                this->CopyInValue(value[i * this->alignedEmbedDims_], this->valueGm_[gmOffset], this->cpOneValParams_);
            }
            if (x != w - 1) {
                this->CopyInValue(value[i * this->alignedEmbedDims_ + this->alignedCornerEmbedDims_],
                    this->valueGm_[gmOffset + this->outDims_], this->cpOneValParams_);
            }
        }
        for (int32_t i = ScalarGetSFFValue<0>(topInvalid); i < this->innerLoops_ && i >= 0;
            i = ScalarGetSFFValue<0>(topInvalid)) {
            topInvalid = sbitset1(topInvalid, i);
            uint32_t idx = baseIdx + i;
            int32_t w = shapeInt.GetValue(idx);
            int32_t x = loc.GetValue(idx);
            // WARN: dangerous!
            uint64_t gmOffset = static_cast<uint64_t>(location.GetValue(idx));
            if (x != -1) {
                this->CopyInValue(value[i * this->alignedEmbedDims_ + 2 * this->alignedCornerEmbedDims_],
                    this->valueGm_[gmOffset + w * this->outDims_], this->cpOneValParams_);
            }
            if (x != w - 1) {
                this->CopyInValue(value[i * this->alignedEmbedDims_ + 3 * this->alignedCornerEmbedDims_],
                    this->valueGm_[gmOffset + w * this->outDims_ + this->outDims_], this->cpOneValParams_);
            }
        }
        SetFlag<HardEvent::MTE2_V>(0);
        for (uint32_t i = 1; i < this->embedBlk_; ++i) {
            Adds<float, false>(cornerWeightBrc[i * B32_DATA_NUM_PER_BLOCK], cornerWeightBrc, 0.f, MASK_PLACEHOLDER,
                this->brcRpt_,
                {this->embedBlk_, this->embedBlk_, static_cast<uint8_t>(8 * this->embedBlk_),
                    static_cast<uint8_t>(8 * this->embedBlk_)});
        }
        WaitFlag<HardEvent::MTE2_V>(0);

        if (unlikely(this->cornerRpt_ > MAX_REPEAT_TIMES)) {
            Mul<float, false>(
                cornerWeightBrc, value, cornerWeightBrc, MASK_PLACEHOLDER, this->cornerRpt_ / 2, {1, 1, 1, 8, 8, 8});
            Duplicate<float, false>(value, 0.f, MASK_PLACEHOLDER, this->cornerRpt_ / 2, 1, 8);
            Mul<float, false>(cornerWeightBrc[this->cornerRpt_ / 2 * B32_DATA_NUM_PER_REPEAT],
                value[this->cornerRpt_ / 2 * B32_DATA_NUM_PER_REPEAT],
                cornerWeightBrc[this->cornerRpt_ / 2 * B32_DATA_NUM_PER_REPEAT], MASK_PLACEHOLDER, this->cornerRpt_ / 2,
                {1, 1, 1, 8, 8, 8});
            Duplicate<float, false>(value[this->cornerRpt_ / 2 * B32_DATA_NUM_PER_REPEAT], 0.f, MASK_PLACEHOLDER,
                this->cornerRpt_ / 2, 1, 8);
        } else {
            Mul<float, false>(
                cornerWeightBrc, value, cornerWeightBrc, MASK_PLACEHOLDER, this->cornerRpt_, {1, 1, 1, 8, 8, 8});
            Duplicate<float, false>(value, 0.f, MASK_PLACEHOLDER, this->cornerRpt_, 1, 8);
        }
        SetFlag<HardEvent::V_MTE2>(0);

        Add<float>(cornerWeightBrc, cornerWeightBrc[2 * this->alignedCornerEmbedDims_], cornerWeightBrc,
            2 * this->alignedCornerEmbedDims_);
        Add<float>(cornerWeightBrc, cornerWeightBrc[this->alignedCornerEmbedDims_], cornerWeightBrc,
            this->alignedCornerEmbedDims_);

        SetVectorMask<float>(0, this->embedMask_);
        if (unlikely(head == 0)) {
            WaitFlag<HardEvent::MTE3_V>(0);
            Duplicate<float, false>(
                output, 0.f, MASK_PLACEHOLDER, this->numHeads_, 1, static_cast<uint8_t>(this->embedBlk_));
        }
        if (fastMode) {
            for (uint32_t i = 0; i < this->numHeads_; ++i) {
                Add<float, false>(output[outOffset], cornerWeightBrc[outOffset * this->oneHeadNum_], output[outOffset],
                    MASK_PLACEHOLDER, this->oneHeadNum_, {1, 1, 1, 0, static_cast<uint8_t>(this->embedBlk_), 0});
                outOffset += this->alignedEmbedDims_;
            }
        } else {
            Add<float, false>(output[outOffset], cornerWeightBrc, output[outOffset], MASK_PLACEHOLDER,
                this->oneHeadNum_, {1, 1, 1, 0, static_cast<uint8_t>(this->embedBlk_), 0});
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
        Duplicate<float, false>(
            value[this->cornerRpt_ / 2 * B32_DATA_NUM_PER_REPEAT], 0.f, MASK_PLACEHOLDER, this->cornerRpt_ / 2, 1, 8);
    } else {
        Duplicate<float, false>(value, 0.f, MASK_PLACEHOLDER, this->cornerRpt_, 1, 8);
    }

    SetFlag<HardEvent::V_MTE2>(this->copyEvt_);
    SetFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::MTE3_V>(0);

    for (uint32_t taskIdx = this->startOffset_; taskIdx < this->endOffset_; ++taskIdx) {
        this->CopyInSample(locationFloat[2 * this->alignedOneQueryNum_], attentionWeight, taskIdx);
        this->ComputeLocation(taskIdx, locationFloat, locationInt, shapeFloat, shapeInt, locFloat, locInt, offsetInt,
            validFlag.ReinterpretCast<uint8_t>());
        ComputeBilinearInterpolation(validFlag, shapeInt, locationInt, locInt, shapeFloat, production, value, locFloat,
            weight, attentionWeight, cornerWeightBrc, output);
        CopyOut(output, taskIdx);
    }
    WaitFlag<HardEvent::V_MTE2>(this->copyEvt_);
    WaitFlag<HardEvent::V_MTE2>(0);
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
