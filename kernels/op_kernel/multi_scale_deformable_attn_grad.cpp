/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file multi_scale_deformable_attn_grad.cpp
 * \brief msdagrad operator
 */

#include "kernel_common.h"
#include "msda.h"

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnGradKernel<aligned, fastMode>::UpdateParams(uint32_t tailCompNum)
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

    if constexpr (fastMode) {
        cpGradSampleParams_.blockCount = this->compTaskNum_;
        cpGradDoubleSampleParams_.blockCount = this->compTaskNum_;
    } else {
        cpGradSampleParams_.blockCount = this->numHeads_ * this->compTaskNum_;
        cpGradDoubleSampleParams_.blockCount = this->numHeads_ * this->compTaskNum_;
    }
}

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnGradKernel<aligned, fastMode>::CopyFullPoint(
    const LocalTensor<int32_t>& location, const LocalTensor<float>& value,
    const LocalTensor<float>& cornerWeightBrc,
    uint64_t& valid, uint32_t baseIdx, uint32_t innerLoops)
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
        this->CopyOutValue(
            gradValueGm_[gmY0Offset], cornerWeightBrc[i * this->alignedEmbedDims_], cpGradRowDoubleParams_);
        this->CopyOutValue(gradValueGm_[gmY1Offset],
            cornerWeightBrc[i * this->alignedEmbedDims_ + 2 * this->alignedCornerEmbedDims_],
            cpGradRowDoubleParams_);
    }
}

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnGradKernel<aligned, fastMode>::CopyBorderPoint(
    const LocalTensor<int32_t>& location, const LocalTensor<float>& value,
    const LocalTensor<float>& cornerWeightBrc,
    const LocalTensor<int32_t>& shapeInt, const LocalTensor<int32_t>& loc,
    uint64_t& valid, uint32_t baseIdx, uint32_t innerLoops)
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
            this->CopyOutValue(
                gradValueGm_[gmOffset], cornerWeightBrc[i * this->alignedEmbedDims_], cpGradOneValParams_);
        }
        if (x != w - 1) {
            this->CopyInValue(value[i * this->alignedEmbedDims_ + this->alignedCornerEmbedDims_],
                this->valueGm_[gmOffset + this->outDims_], this->cpOneValParams_);
            this->CopyOutValue(gradValueGm_[gmOffset + this->outDims_],
                cornerWeightBrc[i * this->alignedEmbedDims_ + this->alignedCornerEmbedDims_], cpGradOneValParams_);
        }
    }
}

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnGradKernel<aligned, fastMode>::PrepareOutTensor(
    const LocalTensor<float>& weight, const LocalTensor<float>& cornerWeightBrc,
    const LocalTensor<float>& gradOut, uint32_t baseIdx, uint32_t outOffset)
{
    ResetMask();
    for (uint32_t i = 0; i < 4; ++i) {
        Brcb(cornerWeightBrc[i * this->alignedCornerEmbedDims_], weight[baseIdx + i * this->alignedOneTaskNum_],
            (fastMode ? this->alignedOneQueryNum_ : this->alignedOneHeadNum_) / B32_DATA_NUM_PER_BLOCK,
            {this->embedBlk_, static_cast<uint16_t>(8 * this->embedBlk_)});
    }
    for (uint32_t i = 1; i < this->embedBlk_; ++i) {
        Adds<float, false>(cornerWeightBrc[i * B32_DATA_NUM_PER_BLOCK], cornerWeightBrc, 0.f, MASK_PLACEHOLDER,
            this->brcRpt_,
            {this->embedBlk_, this->embedBlk_, static_cast<uint8_t>(8 * this->embedBlk_),
                static_cast<uint8_t>(8 * this->embedBlk_)});
    }
    SetVectorMask<float>(0, this->embedMask_);
    GradMul(cornerWeightBrc, cornerWeightBrc, gradOut, outOffset);
}

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnGradKernel<aligned, fastMode>::ComputeBilinearInterpolation(
    const LocalTensor<uint64_t>& validFlag, const LocalTensor<int32_t>& shapeInt, const LocalTensor<int32_t>& location,
    const LocalTensor<int32_t>& loc, const LocalTensor<float>& shapeFloat, const LocalTensor<float>& production,
    const LocalTensor<float>& value, const LocalTensor<float>& locFloat, const LocalTensor<float>& weight,
    const LocalTensor<float>& attentionWeight, const LocalTensor<float>& cornerWeightBrc,
    const LocalTensor<float>& gradOut, const LocalTensor<float>& gradMulTmp)
{
    WaitFlag<HardEvent::V_MTE2>(this->biEvt_);
    this->ComputeWeight(locFloat, shapeFloat, production, weight, attentionWeight);
    WaitFlag<HardEvent::MTE2_V>(1);
    SetVectorMask<float>(0, this->embedMask_);
    for (uint32_t head = 0; head < this->outerLoops_; ++head) {
        uint64_t baseIdx = head * this->innerLoopsAligned_;
        uint64_t headOffset = baseIdx / B32_DATA_NUM_PER_REPEAT;
        uint64_t byteOffset = baseIdx - headOffset * B32_DATA_NUM_PER_REPEAT;
        uint64_t valid = validFlag.GetValue(headOffset) >> byteOffset;
        uint64_t bottomInvalid = validFlag.GetValue(headOffset + 2 * this->validFlagMaskLen_ / 8) >> byteOffset;
        uint64_t topInvalid = validFlag.GetValue(headOffset + 3 * this->validFlagMaskLen_ / 8) >> byteOffset;
        uint32_t outOffset = fastMode ? head * this->numHeads_ * this->alignedEmbedDims_ : head * this->alignedEmbedDims_;

        uint32_t nextOffset = fastMode ? (head + 1) * this->numHeads_ * this->alignedEmbedDims_ : (head + 1) * this->alignedEmbedDims_;
        uint32_t nextIdx = (head + 1) * this->innerLoopsAligned_;

        if (head == 0) {
            WaitFlag<HardEvent::MTE3_V>(0);
            PrepareOutTensor(weight, cornerWeightBrc, gradOut, baseIdx, outOffset);
            SetFlag<HardEvent::V_MTE3>(0);
        }

        WaitFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
        SetAtomicAdd<float>();
        CopyFullPoint(location, value, cornerWeightBrc, valid, baseIdx, this->innerLoops_);
        CopyBorderPoint(location, value, cornerWeightBrc, shapeInt, loc, bottomInvalid, baseIdx, this->innerLoops_);
        CopyBorderPoint(location[this->alignedOneTaskNum_], value[2 * this->alignedCornerEmbedDims_],
            cornerWeightBrc[2 * this->alignedCornerEmbedDims_], shapeInt, loc, topInvalid, baseIdx, this->innerLoops_);
        SetAtomicNone();
        SetFlag<HardEvent::MTE2_V>(0);
        SetFlag<HardEvent::MTE3_V>(0);

        WaitFlag<HardEvent::MTE2_V>(0);
        GradMul(gradMulTmp, value, gradOut, outOffset);
        if (head == this->outerLoops_ - 1) {
            SetFlag<HardEvent::V_MTE2>(1);
        }

        ResetMask();
        if (unlikely(this->cornerRpt_ > MAX_REPEAT_TIMES)) {
            Duplicate<float, false>(value, 0.f, MASK_PLACEHOLDER, this->cornerRpt_ / 2, 1, 8);
            Duplicate<float, false>(value[this->cornerRpt_ / 2 * B32_DATA_NUM_PER_REPEAT], 0.f, MASK_PLACEHOLDER,
                this->cornerRpt_ / 2, 1, 8);
        } else {
            Duplicate<float, false>(value, 0.f, MASK_PLACEHOLDER, this->cornerRpt_, 1, 8);
        }
        SetVectorMask<float>(0, this->embedMask_);
        SetFlag<HardEvent::V_MTE2>(0);
        
        if (head != this->outerLoops_ - 1) {
            WaitFlag<HardEvent::MTE3_V>(0);
            PrepareOutTensor(weight, cornerWeightBrc, gradOut, nextIdx, nextOffset);
            SetFlag<HardEvent::V_MTE3>(0);
        }

        for (uint32_t i = 0; i < 4; ++i) {
            WholeReduceSum<float, false>(weight[baseIdx + i * this->alignedOneTaskNum_],
                gradMulTmp[i * this->alignedCornerEmbedDims_], MASK_PLACEHOLDER, this->innerLoops_, 1, 1, this->embedBlk_);
        }
    }
    ResetMask();
}

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnGradKernel<aligned, fastMode>::ComputeGrad(
    const LocalTensor<float>& production, const LocalTensor<float>& locFloat, const LocalTensor<float>& weight,
    const LocalTensor<float>& attentionWeight, const LocalTensor<float>& gradLocation,
    const LocalTensor<float>& gradAttentionWeight, const LocalTensor<uint32_t>& gatherOffset, uint32_t taskIdx)
{
    uint64_t sampleOffset = taskIdx * this->oneQueryNum_;
    Mul<float, false>(production, weight, production, MASK_PLACEHOLDER, 4 * this->taskRpt_, {1, 1, 1, 8, 8, 8});
    Add<float, false>(production, production, production[2 * this->alignedOneTaskNum_], MASK_PLACEHOLDER,
        2 * this->taskRpt_, {1, 1, 1, 8, 8, 8});
    WaitFlag<HardEvent::MTE3_V>(1);
    Add<float, false>(gradAttentionWeight, production, production[this->alignedOneTaskNum_], MASK_PLACEHOLDER,
        this->taskRpt_, {1, 1, 1, 8, 8, 8});

    Sub<float, false>(gradLocation, weight[3 * this->alignedOneTaskNum_], weight[this->alignedOneTaskNum_],
        MASK_PLACEHOLDER, this->taskRpt_, {1, 1, 1, 8, 8, 8});
    Sub<float, false>(gradLocation[this->alignedOneTaskNum_], weight[3 * this->alignedOneTaskNum_],
        weight[2 * this->alignedOneTaskNum_], MASK_PLACEHOLDER, this->taskRpt_, {1, 1, 1, 8, 8, 8});
    Sub<float, false>(gradLocation[2 * this->alignedOneTaskNum_], weight[2 * this->alignedOneTaskNum_], weight,
        MASK_PLACEHOLDER, this->taskRpt_, {1, 1, 1, 8, 8, 8});
    Sub<float, false>(gradLocation[3 * this->alignedOneTaskNum_], weight[this->alignedOneTaskNum_], weight,
        MASK_PLACEHOLDER, this->taskRpt_, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(gradLocation, locFloat, gradLocation, MASK_PLACEHOLDER, 4 * this->taskRpt_, {1, 1, 1, 8, 8, 8});
    Add<float, false>(gradLocation[2 * this->alignedOneTaskNum_], gradLocation,
        gradLocation[2 * this->alignedOneTaskNum_], MASK_PLACEHOLDER, 2 * this->taskRpt_, {1, 1, 1, 8, 8, 8});
    Gather(gradLocation, gradLocation[2 * this->alignedOneTaskNum_], gatherOffset, 0, 64, 2 * this->taskRpt_, 8);

    SetFlag<HardEvent::V_MTE3>(1);
    WaitFlag<HardEvent::V_MTE3>(1);
    DataCopyPad(gradLocGm_[sampleOffset * 2], gradLocation, cpGradDoubleSampleParams_);
    DataCopyPad(gradAttentionWeightsGm_[sampleOffset], gradAttentionWeight, cpGradSampleParams_);
    SetFlag<HardEvent::MTE3_V>(1);
}

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnGradKernel<aligned, fastMode>::Process()
{
    LocalTensor<uint32_t> gatherOffset = this->gatherOffsetBuf_.template Get<uint32_t>();
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
    LocalTensor<float> gradOut = this->outputQue_.template Get<float>();
    LocalTensor<uint64_t> validFlag = this->validFlagBuf_.template Get<uint64_t>();

    LocalTensor<int32_t> locInt = this->locIntBuf_.template Get<int32_t>();
    LocalTensor<float> locFloat = this->locFloatBuf_.template Get<float>();
    LocalTensor<float> production = this->productionBuf_.template Get<float>();
    LocalTensor<float> weight = this->weightBuf_.template Get<float>();
    LocalTensor<float> gradMulTmp = this->gradMulTmpBuf_.template Get<float>();
    LocalTensor<float> gradLocation = this->gradLocationQue_.template Get<float>();
    LocalTensor<float> gradAttentionWeight = this->gradAttentionWeightsQue_.template Get<float>();

    PrepareGatherOffset(gatherOffset);
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
    SetFlag<HardEvent::V_MTE2>(1);
    SetFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::MTE3_V>(1);

    for (uint32_t taskIdx = this->startOffset_; taskIdx < this->endOffset_; taskIdx += this->compTaskNum_) {
        if (unlikely(taskIdx + this->compTaskNum_ > this->endOffset_)) {
            UpdateParams(this->endOffset_ - taskIdx);
        }
        this->CopyInSample(locationFloat[2 * this->alignedOneTaskNum_], attentionWeight, taskIdx);
        CopyInGradOut(gradOut, taskIdx);
        this->ComputeLocation(taskIdx, locationFloat, locationInt, shapeFloat, shapeInt, locFloat, locInt, offsetInt,
            validFlag.ReinterpretCast<uint8_t>());
        ComputeBilinearInterpolation(validFlag, shapeInt, locationInt, locInt, shapeFloat, production, value, locFloat,
            weight, attentionWeight, cornerWeightBrc, gradOut, gradMulTmp);
        ComputeGrad(
            production, locFloat, weight, attentionWeight, gradLocation, gradAttentionWeight, gatherOffset, taskIdx);
    }
    WaitFlag<HardEvent::V_MTE2>(this->copyEvt_);
    WaitFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(1);
    WaitFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(1);
}

// core func
extern "C" __global__ __aicore__ void multi_scale_deformable_attn_grad(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm,
    GM_ADDR level_start_index_gm, GM_ADDR sampling_loc_gm, GM_ADDR attn_weight_gm, GM_ADDR grad_output_gm,
    GM_ADDR grad_value_gm, GM_ADDR grad_sampling_loc_gm, GM_ADDR grad_attn_weight_gm, GM_ADDR workspace,
    GM_ADDR tiling_data)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_datas, tiling_data);
    if (TILING_KEY_IS(10)) {
        MultiScaleDeformableAttnGradKernel<true, false> op(value_gm, spatial_shapes_gm, level_start_index_gm,
            sampling_loc_gm, attn_weight_gm, grad_output_gm, grad_value_gm, grad_sampling_loc_gm, grad_attn_weight_gm,
            &tiling_datas, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(00)) {
        MultiScaleDeformableAttnGradKernel<false, false> op(value_gm, spatial_shapes_gm, level_start_index_gm,
            sampling_loc_gm, attn_weight_gm, grad_output_gm, grad_value_gm, grad_sampling_loc_gm, grad_attn_weight_gm,
            &tiling_datas, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(11)) {
        MultiScaleDeformableAttnGradKernel<true, true> op(value_gm, spatial_shapes_gm, level_start_index_gm,
            sampling_loc_gm, attn_weight_gm, grad_output_gm, grad_value_gm, grad_sampling_loc_gm, grad_attn_weight_gm,
            &tiling_datas, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(01)) {
        MultiScaleDeformableAttnGradKernel<false, true> op(value_gm, spatial_shapes_gm, level_start_index_gm,
            sampling_loc_gm, attn_weight_gm, grad_output_gm, grad_value_gm, grad_sampling_loc_gm, grad_attn_weight_gm,
            &tiling_datas, &pipe);
        op.Process();
    }
}
