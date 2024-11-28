/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
 * \file multi_scale_deformable_attention_grad_generic_v2.h
 * \brief
 */

#ifndef MS_DEFORM_ATTN_GRAD_HIGH_PERF_H_
#define MS_DEFORM_ATTN_GRAD_HIGH_PERF_H_


#include "kernel_operator.h"

using namespace AscendC;

template<int32_t num_points, int32_t embed_dims>
class KernelMultiScaleDeformableAttnGradOpt {
public:
    __aicore__ inline KernelMultiScaleDeformableAttnGradOpt() = delete;

    __aicore__ inline KernelMultiScaleDeformableAttnGradOpt(GM_ADDR value, GM_ADDR valueSpatialShapes,
        GM_ADDR valueLevelStartIndex, GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR gradOutput,
        GM_ADDR gradValue, GM_ADDR gradSamplingLocations, GM_ADDR gradAttentionWeights,
        const MultiScaleDeformableAttnGradTilingData* tilingData, TPipe* pipe)
        : pipe_(pipe), blkIdx_(GetBlockIdx())
    {
        InitTiling(tilingData);
        InitTask();
        InitGM(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations, attentionWeights, gradOutput,
            gradValue, gradSamplingLocations, gradAttentionWeights);
        InitBuffer();
        InitEvent();

        SetVectorMask<float>(FULL_MASK, FULL_MASK);
    }

    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTask()
    {
        uint32_t avgTasks = numQueries_ / coreNum_;
        uint32_t remainTasks = numQueries_ % coreNum_;
        startOffset_ = avgTasks * blkIdx_ + (blkIdx_ < remainTasks ? blkIdx_ : remainTasks);
        endOffset_ = startOffset_ + avgTasks + (blkIdx_ < remainTasks ? 1 : 0);
    }

    __aicore__ inline void InitTiling(const MultiScaleDeformableAttnGradTilingData* tilingData)
    {
        batchSize_ = tilingData->batchSize;
        numKeys_ = tilingData->numKeys;
        numHeads_ = tilingData->numHeads;
        embedDims_ = embed_dims;
        numLevels_ = tilingData->numLevels;
        numQueries_ = tilingData->numQueries;
        numPoints_ = tilingData->numPoints;
        coreNum_ = tilingData->coreNum;
        pointLoops_ = tilingData->pointLoops;
        realLevels_ = tilingData->realLevels;

        oneQueryNum_ = realLevels_ * numHeads_ * numPoints_;

        alignedNumPoints_ = AlignUp(num_points, B32_DATA_NUM_PER_BLOCK);
        alignedOneHeadNum_ = numLevels_ * alignedNumPoints_;
        alignedOneQueryNum_ = AlignUp(numHeads_ * alignedOneHeadNum_, B32_DATA_NUM_PER_REPEAT);
        alignedEmbedDims_ = AlignUp(embedDims_, B32_DATA_NUM_PER_BLOCK);
        alignedCornerEmbedDims_ = AlignUp(4 * num_points * alignedEmbedDims_, B32_DATA_NUM_PER_REPEAT);

        embedBlk_ = alignedEmbedDims_ / B32_DATA_NUM_PER_BLOCK;
        outDims_ = numHeads_ * embedDims_;
        outBlk_ = numHeads_ * embedBlk_;
        pointBlk_ = alignedNumPoints_ / B32_DATA_NUM_PER_BLOCK;
        queryBlk_ = alignedOneQueryNum_ / B32_DATA_NUM_PER_BLOCK;
        rptTimes_ = alignedOneQueryNum_ / B32_DATA_NUM_PER_REPEAT;
        valRptTimes4_ = alignedCornerEmbedDims_ / B32_DATA_NUM_PER_REPEAT;
        valRptTimes1_ = DivCeil(num_points * alignedEmbedDims_, B32_DATA_NUM_PER_REPEAT);

        if (num_points == 8 && pointLoops_ == 1) {
            cpSampleParams_.blockLen = DivCeil(numLevels_ * numHeads_ * num_points, B32_DATA_NUM_PER_BLOCK);
            cpDoubleSampleParams_.blockLen = DivCeil(2 * numLevels_ * numHeads_ * num_points, B32_DATA_NUM_PER_BLOCK);
        } else {
            cpSampleParams_.blockCount = numLevels_ * numHeads_;
            cpSampleParams_.blockLen = num_points * B32_BYTE_SIZE;
            cpSampleParams_.srcStride = (numPoints_ - num_points) * B32_BYTE_SIZE;
            cpDoubleSampleParams_.blockCount = numLevels_ * numHeads_;
            cpDoubleSampleParams_.blockLen = 2 * num_points * B32_BYTE_SIZE;
            cpDoubleSampleParams_.srcStride = 2 * (numPoints_ - num_points) * B32_BYTE_SIZE;
            cpDoubleSampleParams_.dstStride = num_points == 8 ? 0 : 1;
        }

        cpGradOutParams_.blockLen = numHeads_ * embedBlk_;

        cpOneValParams_.blockLen = embedBlk_;
        cpDoubleValParams_.blockLen = embedBlk_;
        cpDoubleValParams_.srcStride = outBlk_ - embedBlk_;
        cpDoubleValParams_.dstStride = num_points * embedBlk_ - embedBlk_;
        cpGradValueParams_.blockLen = embedBlk_;
        cpGradValueParams_.srcStride = num_points * embedBlk_ - embedBlk_;
        cpGradValueParams_.dstStride = outBlk_ - embedBlk_;

        gatherParams_.repeatTimes = rptTimes_ * 2;

        dstRptStride_ = num_points * embedBlk_;
    }

    __aicore__ inline void InitGM(GM_ADDR value, GM_ADDR valueSpatialShapes, GM_ADDR valueLevelStartIndex,
        GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR gradOutput, GM_ADDR gradValue,
        GM_ADDR gradSamplingLocations, GM_ADDR gradAttentionWeights)
    {
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(value));
        locationGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(samplingLocations));
        attentionWeightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(attentionWeights));
        valueSpatialShapesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valueSpatialShapes));
        valueLevelStartIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valueLevelStartIndex));
        gradOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradOutput));
        gradValueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradValue));
        gradLocGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradSamplingLocations));
        gradAttentionWeightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradAttentionWeights));
    }

    __aicore__ inline void InitBuffer()
    {
        pipe_->InitBuffer(
            gatherOffsetBuf_, 16 * B32_BYTE_SIZE); // [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]
        pipe_->InitBuffer(shapeQue_, AlignUp(numLevels_ * 2, B32_DATA_NUM_PER_BLOCK) * B32_BYTE_SIZE);
        pipe_->InitBuffer(offsetQue_, AlignUp(numLevels_, B32_DATA_NUM_PER_BLOCK) * B32_BYTE_SIZE);
        pipe_->InitBuffer(locationQue_, 4 * alignedOneQueryNum_ * B32_BYTE_SIZE); // x, y
        pipe_->InitBuffer(attentionWeightsQue_, alignedOneQueryNum_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(valueQue_, 2 * alignedCornerEmbedDims_ * B32_BYTE_SIZE); // 2 for double buffer
        pipe_->InitBuffer(gradValueQue_, 2 * alignedCornerEmbedDims_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(gradOutQue_, numHeads_ * alignedEmbedDims_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(gradAttentionWeightsQue_, numLevels_ * alignedNumPoints_ * B32_BYTE_SIZE);

        pipe_->InitBuffer(shapeBrcBuf_, 2 * alignedOneQueryNum_ * B32_BYTE_SIZE);   // w, h
        pipe_->InitBuffer(locIntBuf_, 4 * alignedOneQueryNum_ * B32_BYTE_SIZE);     // x0, y0, x1, y1
        pipe_->InitBuffer(locFloatBuf_, 4 * alignedOneQueryNum_ * B32_BYTE_SIZE);   // lw, lh
        pipe_->InitBuffer(productionBuf_, 4 * alignedOneQueryNum_ * B32_BYTE_SIZE); // lh * lw
        pipe_->InitBuffer(weightBuf_, 4 * alignedOneQueryNum_ * B32_BYTE_SIZE);     // w1-w4
        pipe_->InitBuffer(cornerWeightBuf_, 4 * alignedNumPoints_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(reducedValueBuf_, 4 * alignedNumPoints_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(valueDiffBuf_, 4 * alignedNumPoints_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(gradLocQue_, numLevels_ * 32 * B32_BYTE_SIZE);
    }

    __aicore__ inline void InitEvent()
    {
        calEvt_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        copyEvt_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
    }

    __aicore__ inline void PrepareGatherOffset(const LocalTensor<uint32_t>& gatherOffset);

    __aicore__ inline void PrepareShape(
        const LocalTensor<int32_t>& shapes, const LocalTensor<int32_t>& offset, LocalTensor<float>& shapeBrc);

    __aicore__ inline void CopyInSample(const LocalTensor<float>& location, const LocalTensor<float>& attentionWeight,
        uint32_t batch, uint32_t query, uint32_t pl);

    __aicore__ inline void CopyInGradOut(const LocalTensor<float>& gradOut, uint32_t batch, uint32_t query);

    __aicore__ inline void ComputeLocation(const LocalTensor<float>& location, const LocalTensor<float>& shapes,
        const LocalTensor<int32_t>& locInt, const LocalTensor<float>& locFloat);

    __aicore__ inline void ComputeWeight(const LocalTensor<int32_t>& locInt, const LocalTensor<float>& locFloat,
        const LocalTensor<float>& shapes, const LocalTensor<float>& production, const LocalTensor<float>& weight,
        const LocalTensor<float>& attentionWeight);

    __aicore__ inline void ComputeBilinearInterpolation(const LocalTensor<int32_t>& shapes,
        const LocalTensor<int32_t>& offset, const LocalTensor<int32_t>& locInt, const LocalTensor<float>& locFloat,
        const LocalTensor<float>& value, const LocalTensor<float>& production, const LocalTensor<float>& weight,
        const LocalTensor<float>& gradOut, const LocalTensor<float>& gradValue, const LocalTensor<float>& cornerWeight,
        const LocalTensor<float>& reducedValue, const LocalTensor<float>& valueDiff, const LocalTensor<float>& gradLoc,
        const LocalTensor<float>& gradWeight, const LocalTensor<uint32_t> gatherOffset);

private:
    TPipe* pipe_;
    GlobalTensor<float> valueGm_, locationGm_, attentionWeightsGm_, gradOutGm_, gradValueGm_, gradLocGm_,
        gradAttentionWeightsGm_;
    GlobalTensor<int32_t> valueSpatialShapesGm_, valueLevelStartIndexGm_;

    TBuf<TPosition::VECCALC> locationQue_, attentionWeightsQue_, shapeQue_, offsetQue_, valueQue_, gradOutQue_;
    TBuf<TPosition::VECCALC> gradValueQue_, gradLocQue_, gradAttentionWeightsQue_;

    TBuf<TPosition::VECCALC> locIntBuf_, locFloatBuf_, shapeBrcBuf_, productionBuf_, weightBuf_, cornerWeightBuf_,
        reducedValueBuf_, valueDiffBuf_, gatherOffsetBuf_;

    int32_t blkIdx_;

    uint32_t batchSize_, numKeys_, numHeads_, embedDims_, outDims_, numLevels_, numQueries_, numPoints_, coreNum_,
        pointLoops_, realLevels_;
    uint32_t startOffset_, endOffset_;
    uint32_t alignedNumPoints_, alignedOneHeadNum_, alignedOneQueryNum_, alignedEmbedDims_, alignedCornerEmbedDims_;
    uint32_t oneQueryNum_;
    uint16_t pointBlk_, headBlk_, queryBlk_, embedBlk_, outBlk_, dstRptStride_;
    uint16_t rptTimes_, valRptTimes4_, valRptTimes1_;

    TEventID calEvt_, copyEvt_;

    uint32_t baseSrcOffset_, baseDstOffset_, srcOffset_, weightOffset_;

    DataCopyParams cpOneValParams_, cpDoubleValParams_ {2, 0, 0, 0}, cpSampleParams_,
        cpDoubleSampleParams_ {1, 0, 0, 0}, cpGradOutParams_, cpGradValueParams_ {2, 0, 0, 0};
    GatherMaskParams gatherParams_;
};

template<int32_t num_points, int32_t embed_dims>
__aicore__ inline void KernelMultiScaleDeformableAttnGradOpt<num_points, embed_dims>::PrepareGatherOffset(
    const LocalTensor<uint32_t>& gatherOffset)
{
    for (uint32_t i = 0; i < 8; ++i) {
        gatherOffset.SetValue(2 * i, (i + 8) * 4);
        gatherOffset.SetValue(2 * i + 1, i * 4);
    }
}
template<int32_t num_points, int32_t embed_dims>
__aicore__ inline void KernelMultiScaleDeformableAttnGradOpt<num_points, embed_dims>::PrepareShape(
    const LocalTensor<int32_t>& shapes, const LocalTensor<int32_t>& offset, LocalTensor<float>& shapeBrc)
{
    DataCopy(shapes, valueSpatialShapesGm_,
        {1, static_cast<uint16_t>(DivCeil(2 * numLevels_, B32_DATA_NUM_PER_BLOCK)), 0, 0});
    DataCopy(
        offset, valueLevelStartIndexGm_, {1, static_cast<uint16_t>(DivCeil(numLevels_, B32_DATA_NUM_PER_BLOCK)), 0, 0});
    SetFlag<HardEvent::MTE2_V>(copyEvt_);
    WaitFlag<HardEvent::MTE2_V>(copyEvt_);
    // broadcast to [head*level, 8]
    for (uint32_t k = 0; k < 2; ++k) {
        for (uint32_t i = 0; i < numLevels_; ++i) {
            shapeBrc.SetValue(i + k * alignedOneQueryNum_, shapes.GetValue(2 * i + 1 - k));
        }
        Brcb(shapeBrc[k * alignedOneQueryNum_], shapeBrc[k * alignedOneQueryNum_], 1, {1, 8});
        Copy<float, false>(shapeBrc[k * alignedOneQueryNum_ + numLevels_ * 8], shapeBrc[k * alignedOneQueryNum_],
            MASK_PLACEHOLDER, numHeads_ - 1, {1, 1, static_cast<uint16_t>(numLevels_), 0});
    }
}

template<int32_t num_points, int32_t embed_dims>
__aicore__ inline void KernelMultiScaleDeformableAttnGradOpt<num_points, embed_dims>::CopyInSample(
    const LocalTensor<float>& location, const LocalTensor<float>& attentionWeight, uint32_t batch, uint32_t query,
    uint32_t pl)
{
    uint32_t sampleOffset = (batch * numQueries_ + query) * oneQueryNum_;
    weightOffset_ = sampleOffset + pl * num_points;
    WaitFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(1);
    if (num_points == 8 && pointLoops_ == 1) {
        DataCopy(location, locationGm_[weightOffset_ * 2], cpDoubleSampleParams_);
        DataCopy(attentionWeight, attentionWeightsGm_[weightOffset_], cpSampleParams_);
    } else {
        DataCopyPad(location, locationGm_[weightOffset_ * 2], cpDoubleSampleParams_, {});
        DataCopyPad(attentionWeight, attentionWeightsGm_[weightOffset_], cpSampleParams_, {});
    }
}

template<int32_t num_points, int32_t embed_dims>
__aicore__ inline void KernelMultiScaleDeformableAttnGradOpt<num_points, embed_dims>::CopyInGradOut(
    const LocalTensor<float>& gradOut, uint32_t batch, uint32_t query)
{
    uint32_t gradOffset = (batch * numQueries_ + query) * numHeads_ * embedDims_;
    DataCopy(gradOut, gradOutGm_[gradOffset], cpGradOutParams_);
    SetFlag<HardEvent::MTE2_V>(copyEvt_);
}

template<int32_t num_points, int32_t embed_dims>
__aicore__ inline void KernelMultiScaleDeformableAttnGradOpt<num_points, embed_dims>::ComputeLocation(
    const LocalTensor<float>& location, const LocalTensor<float>& shapes, const LocalTensor<int32_t>& locInt,
    const LocalTensor<float>& locFloat)
{
    uint64_t cnt;
    WaitFlag<HardEvent::MTE2_V>(copyEvt_);

    GatherMask(location, location[2 * alignedOneQueryNum_], 1, false, MASK_PLACEHOLDER, gatherParams_, cnt);
    GatherMask(location[alignedOneQueryNum_], location[2 * alignedOneQueryNum_], 2, false, MASK_PLACEHOLDER,
        gatherParams_, cnt);
    SetVectorMask<float>(FULL_MASK, FULL_MASK);

    Mul<float, false>(location, location, shapes, MASK_PLACEHOLDER, 2 * rptTimes_, {1, 1, 1, 8, 8, 8});
    Adds<float, false>(locFloat, location, 0.5f, MASK_PLACEHOLDER, 2 * rptTimes_, {1, 1, 8, 8});
    Cast<int32_t, float, false>(locInt, locFloat, RoundMode::CAST_FLOOR, MASK_PLACEHOLDER, 2 * rptTimes_, {1, 1, 8, 8});
    SetFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE2>(1);
}

template<int32_t num_points, int32_t embed_dims>
__aicore__ inline void KernelMultiScaleDeformableAttnGradOpt<num_points, embed_dims>::ComputeWeight(
    const LocalTensor<int32_t>& locInt, const LocalTensor<float>& locFloat, const LocalTensor<float>& shapes,
    const LocalTensor<float>& production, const LocalTensor<float>& weight, const LocalTensor<float>& attentionWeight)
{
    Cast<float, int32_t, false>(
        locFloat[2 * alignedOneQueryNum_], locInt, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 2 * rptTimes_, {1, 1, 8, 8});
    Sub<float, false>(locFloat, locFloat, locFloat[2 * alignedOneQueryNum_], MASK_PLACEHOLDER, 2 * rptTimes_,
        {1, 1, 1, 8, 8, 8}); // lw, lh
    Mul<float, false>(production[3 * alignedOneQueryNum_], locFloat, locFloat[alignedOneQueryNum_], MASK_PLACEHOLDER,
        rptTimes_, {1, 1, 1, 8, 8, 8}); // lh * lw
    Duplicate<float, false>(production, 1.f, MASK_PLACEHOLDER, rptTimes_, 1, 8);
    Sub<float, false>(
        locFloat[2 * alignedOneQueryNum_], production, locFloat, MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8}); // hw
    Sub<float, false>(locFloat[3 * alignedOneQueryNum_], production, locFloat[alignedOneQueryNum_], MASK_PLACEHOLDER,
        rptTimes_, {1, 1, 1, 8, 8, 8}); // hh

    Mul<float, false>(production, locFloat[2 * alignedOneQueryNum_], locFloat[3 * alignedOneQueryNum_],
        MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8}); // hw * hh
    Mul<float, false>(production[alignedOneQueryNum_], locFloat, locFloat[3 * alignedOneQueryNum_], MASK_PLACEHOLDER,
        rptTimes_, {1, 1, 1, 8, 8, 8}); // lw * hh
    Mul<float, false>(production[2 * alignedOneQueryNum_], locFloat[alignedOneQueryNum_],
        locFloat[2 * alignedOneQueryNum_], MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8}); // hw * lh
    Mul<float, false>(production[3 * alignedOneQueryNum_], locFloat[alignedOneQueryNum_], locFloat, MASK_PLACEHOLDER,
        rptTimes_, {1, 1, 1, 8, 8, 8}); // lw * lh
    Mul<float, false>(weight, production, attentionWeight, MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(weight[alignedOneQueryNum_], production[alignedOneQueryNum_], attentionWeight, MASK_PLACEHOLDER,
        rptTimes_, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(weight[2 * alignedOneQueryNum_], production[2 * alignedOneQueryNum_], attentionWeight,
        MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(weight[3 * alignedOneQueryNum_], production[3 * alignedOneQueryNum_], attentionWeight,
        MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(
        locFloat, locFloat, shapes[alignedOneQueryNum_], MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8}); // lw * h
    Mul<float, false>(locFloat[alignedOneQueryNum_], locFloat[alignedOneQueryNum_], shapes, MASK_PLACEHOLDER, rptTimes_,
        {1, 1, 1, 8, 8, 8}); // lh * w
    Mul<float, false>(locFloat[2 * alignedOneQueryNum_], locFloat[2 * alignedOneQueryNum_], shapes[alignedOneQueryNum_],
        MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8}); // hw * h
    Mul<float, false>(locFloat[3 * alignedOneQueryNum_], locFloat[3 * alignedOneQueryNum_], shapes, MASK_PLACEHOLDER,
        rptTimes_, {1, 1, 1, 8, 8, 8}); // hh * w
    Mul<float, false>(locFloat, locFloat, attentionWeight, MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(locFloat[alignedOneQueryNum_], locFloat[alignedOneQueryNum_], attentionWeight, MASK_PLACEHOLDER,
        rptTimes_, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(locFloat[2 * alignedOneQueryNum_], locFloat[2 * alignedOneQueryNum_], attentionWeight,
        MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(locFloat[3 * alignedOneQueryNum_], locFloat[3 * alignedOneQueryNum_], attentionWeight,
        MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
}

template<int32_t num_points, int32_t embed_dims>
__aicore__ inline void KernelMultiScaleDeformableAttnGradOpt<num_points, embed_dims>::ComputeBilinearInterpolation(
    const LocalTensor<int32_t>& shapes, const LocalTensor<int32_t>& offset, const LocalTensor<int32_t>& locInt,
    const LocalTensor<float>& locFloat, const LocalTensor<float>& value, const LocalTensor<float>& production,
    const LocalTensor<float>& weight, const LocalTensor<float>& gradOut, const LocalTensor<float>& gradValue,
    const LocalTensor<float>& cornerWeight, const LocalTensor<float>& reducedValue, const LocalTensor<float>& valueDiff,
    const LocalTensor<float>& gradLoc, const LocalTensor<float>& gradWeight, const LocalTensor<uint32_t> gatherOffset)
{
    uint8_t ping = 0;

#pragma bisheng auto_sync parallel
    for (uint32_t head = 0; head < numHeads_; ++head) {
        uint32_t valueOffset = (baseSrcOffset_ + head) * embedDims_;
        uint32_t outOffset = head * alignedEmbedDims_;
        uint32_t weightOffset = weightOffset_ + head * realLevels_ * numPoints_;

        for (uint32_t level = 0; level < numLevels_; ++level) {
            if (embed_dims < 64) {
                SetVectorMask<float>(0, (1UL << embedDims_) - 1);
            } else {
                SetVectorMask<float>(FULL_MASK, FULL_MASK);
            }

            int32_t h = shapes.GetValue(level * 2);
            int32_t w = shapes.GetValue(level * 2 + 1);
            srcOffset_ = valueOffset + offset.GetValue(level) * outDims_;

            uint32_t sx = head * alignedOneHeadNum_ + level * alignedNumPoints_;
            uint32_t sy = sx + alignedOneQueryNum_;

            uint32_t pingOffset = ping * alignedCornerEmbedDims_;
            WaitFlag<HardEvent::V_MTE2>(ping);

            for (uint32_t point = 0; point < num_points; ++point) {
                int32_t px = point + sx;
                int32_t py = point + sy;
                int32_t y1 = locInt.GetValue(py);
                int32_t x1 = locInt.GetValue(px);
                int32_t y0 = y1 - 1;
                int32_t x0 = x1 - 1;

                if (0 <= y0 && y0 < h) {
                    if (0 < x1 && x1 < w) {
                        uint32_t ubOffset = pingOffset + point * alignedEmbedDims_;
                        uint32_t gmOffset = srcOffset_ + (y0 * w + x0) * outDims_;
                        DataCopy(value[ubOffset], valueGm_[gmOffset], cpDoubleValParams_);
                        Muls<float, false>(gradValue[ubOffset], gradOut[outOffset], weight.GetValue(px),
                            MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
                        Muls<float, false>(gradValue[ubOffset + num_points * alignedEmbedDims_], gradOut[outOffset],
                            weight.GetValue(py), MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
                        SetFlag<HardEvent::V_MTE3>(calEvt_);
                        WaitFlag<HardEvent::V_MTE3>(calEvt_);
                        SetAtomicAdd<float>();
                        DataCopy(gradValueGm_[gmOffset], gradValue[ubOffset], cpGradValueParams_);
                        SetAtomicNone();
                    } else if (0 <= x0 && x0 < w) {
                        uint32_t ubOffset = pingOffset + point * alignedEmbedDims_;
                        uint32_t gmOffset = srcOffset_ + (y0 * w + x0) * outDims_;
                        DataCopy(value[ubOffset], valueGm_[gmOffset], cpOneValParams_);
                        Muls<float, false>(gradValue[ubOffset], gradOut[outOffset], weight.GetValue(px),
                            MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
                        SetFlag<HardEvent::V_MTE3>(calEvt_);
                        WaitFlag<HardEvent::V_MTE3>(calEvt_);
                        SetAtomicAdd<float>();
                        DataCopy(gradValueGm_[gmOffset], gradValue[ubOffset], cpOneValParams_);
                        SetAtomicNone();
                    } else if (0 <= x1 && x1 < w) {
                        uint32_t ubOffset = pingOffset + (point + num_points) * alignedEmbedDims_;
                        uint32_t gmOffset = srcOffset_ + (y0 * w + x1) * outDims_;
                        DataCopy(value[ubOffset], valueGm_[gmOffset], cpOneValParams_);
                        Muls<float, false>(gradValue[ubOffset], gradOut[outOffset], weight.GetValue(py),
                            MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
                        SetFlag<HardEvent::V_MTE3>(calEvt_);
                        WaitFlag<HardEvent::V_MTE3>(calEvt_);
                        SetAtomicAdd<float>();
                        DataCopy(gradValueGm_[gmOffset], gradValue[ubOffset], cpOneValParams_);
                        SetAtomicNone();
                    }
                }
                if (0 <= y1 && y1 < h) {
                    if (0 < x1 && x1 < w) {
                        uint32_t ubOffset = pingOffset + (point + 2 * num_points) * alignedEmbedDims_;
                        uint32_t gmOffset = srcOffset_ + (y1 * w + x0) * outDims_;
                        DataCopy(value[ubOffset], valueGm_[gmOffset], cpDoubleValParams_);
                        Muls<float, false>(gradValue[ubOffset], gradOut[outOffset],
                            weight.GetValue(px + 2 * alignedOneQueryNum_), MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
                        Muls<float, false>(gradValue[ubOffset + num_points * alignedEmbedDims_], gradOut[outOffset],
                            weight.GetValue(py + 2 * alignedOneQueryNum_), MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
                        SetFlag<HardEvent::V_MTE3>(calEvt_);
                        WaitFlag<HardEvent::V_MTE3>(calEvt_);
                        SetAtomicAdd<float>();
                        DataCopy(gradValueGm_[gmOffset], gradValue[ubOffset], cpGradValueParams_);
                        SetAtomicNone();
                    } else if (0 <= x0 && x0 < w) {
                        uint32_t ubOffset = pingOffset + (point + 2 * num_points) * alignedEmbedDims_;
                        uint32_t gmOffset = srcOffset_ + (y1 * w + x0) * outDims_;
                        DataCopy(value[ubOffset], valueGm_[gmOffset], cpOneValParams_);
                        Muls<float, false>(gradValue[ubOffset], gradOut[outOffset],
                            weight.GetValue(px + 2 * alignedOneQueryNum_), MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
                        SetFlag<HardEvent::V_MTE3>(calEvt_);
                        WaitFlag<HardEvent::V_MTE3>(calEvt_);
                        SetAtomicAdd<float>();
                        DataCopy(gradValueGm_[gmOffset], gradValue[ubOffset], cpOneValParams_);
                        SetAtomicNone();
                    } else if (0 <= x1 && x1 < w) {
                        uint32_t ubOffset = pingOffset + (point + 3 * num_points) * alignedEmbedDims_;
                        uint32_t gmOffset = srcOffset_ + (y1 * w + x1) * outDims_;
                        DataCopy(value[ubOffset], valueGm_[gmOffset], cpOneValParams_);
                        Muls<float, false>(gradValue[ubOffset], gradOut[outOffset],
                            weight.GetValue(py + 2 * alignedOneQueryNum_), MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
                        SetFlag<HardEvent::V_MTE3>(calEvt_);
                        WaitFlag<HardEvent::V_MTE3>(calEvt_);
                        SetAtomicAdd<float>();
                        DataCopy(gradValueGm_[gmOffset], gradValue[ubOffset], cpOneValParams_);
                        SetAtomicNone();
                    }
                }
            }
            SetFlag<HardEvent::MTE2_V>(copyEvt_);
            SetFlag<HardEvent::MTE3_V>(ping);
            WaitFlag<HardEvent::MTE3_V>(ping);

            SetVectorMask<float>(0, 0xffffffff);
            Copy<float, false>(cornerWeight, production[sx], MASK_PLACEHOLDER, 1, {1, queryBlk_, 8, 8});

            WaitFlag<HardEvent::MTE2_V>(copyEvt_);
            if (embed_dims < 64) {
                SetVectorMask<float>(0, (1UL << embedDims_) - 1);
            } else {
                SetVectorMask<float>(FULL_MASK, FULL_MASK);
            }
            Mul<float, false>(value[pingOffset], value[pingOffset], gradOut[outOffset], MASK_PLACEHOLDER,
                num_points * 4, {1, 1, 1, static_cast<uint8_t>(embedBlk_), static_cast<uint8_t>(embedBlk_), 0});
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < 4; ++i) {
                WholeReduceSum<float, false>(reducedValue[i * alignedNumPoints_],
                    value[pingOffset + i * num_points * alignedEmbedDims_], MASK_PLACEHOLDER, num_points, 1, 1,
                    embedBlk_); // dstRepStride Unit: 4 bytes
            }
            PipeBarrier<PIPE_V>();
            Duplicate<float, false>(value[pingOffset], 0.f, MASK_PLACEHOLDER, num_points * 4, 1, embedBlk_);
            SetFlag<HardEvent::V_MTE2>(ping);
            ping = 1 - ping;

            SetVectorMask<float>(0, 0xff);
            PipeBarrier<PIPE_V>();
            Mul<float, false>(cornerWeight, reducedValue, cornerWeight, MASK_PLACEHOLDER, 4,
                {1, 1, 1, 1, 1, 1}); // [4*numPoints,] * [4*numPoints,]

            PipeBarrier<PIPE_V>();
            Add<float, false>(cornerWeight, cornerWeight, cornerWeight[2 * alignedNumPoints_], MASK_PLACEHOLDER, 2,
                {1, 1, 1, 1, 1, 1});
            PipeBarrier<PIPE_V>();
            Add<float, false>(gradWeight[level * alignedNumPoints_], cornerWeight, cornerWeight[alignedNumPoints_],
                MASK_PLACEHOLDER, 1, {1, 1, 1, 1, 1, 1});
            SetFlag<HardEvent::V_MTE3>(calEvt_);
            WaitFlag<HardEvent::V_MTE3>(calEvt_);
            if (num_points == 8) {
                DataCopy(gradAttentionWeightsGm_[weightOffset], gradWeight[level * alignedNumPoints_], {1, 1, 0, 0});
            } else {
                DataCopyPad(gradAttentionWeightsGm_[weightOffset], gradWeight[level * alignedNumPoints_],
                    {1, static_cast<uint16_t>(num_points * B32_BYTE_SIZE), 0, 0});
            }

            Sub<float, false>(valueDiff, reducedValue[3 * alignedNumPoints_], reducedValue[alignedNumPoints_],
                MASK_PLACEHOLDER, 2, {1, 1, 1, 1, 0, 1});
            PipeBarrier<PIPE_V>();
            Sub<float, false>(valueDiff[2 * alignedNumPoints_], reducedValue[2 * alignedNumPoints_], reducedValue,
                MASK_PLACEHOLDER, 1, {1, 1, 1, 1, 1, 0});
            PipeBarrier<PIPE_V>();
            Sub<float, false>(valueDiff[3 * alignedNumPoints_], reducedValue[alignedNumPoints_], reducedValue,
                MASK_PLACEHOLDER, 1, {1, 1, 1, 1, 1, 0});

            SetVectorMask<float>(0, 0xffffffff);
            Copy<float, false>(reducedValue, locFloat[sx], MASK_PLACEHOLDER, 1, {1, queryBlk_, 8, 8});
            PipeBarrier<PIPE_V>();
            Mul<float, false>(reducedValue, reducedValue, valueDiff, MASK_PLACEHOLDER, 1, {1, 1, 1, 1, 1, 1});
            PipeBarrier<PIPE_V>();
            Add<float, false>(reducedValue, reducedValue, reducedValue[2 * alignedNumPoints_], MASK_PLACEHOLDER, 1,
                {1, 1, 1, 1, 1, 1});
            PipeBarrier<PIPE_V>();
            Gather(gradLoc[level * 32], reducedValue, gatherOffset, 0, 16);
            SetFlag<HardEvent::V_MTE3>(calEvt_);
            WaitFlag<HardEvent::V_MTE3>(calEvt_);
            if (num_points >= 4) { // has padded
                DataCopy(gradLocGm_[weightOffset * 2], gradLoc[level * 32],
                    {1, static_cast<uint16_t>(num_points * 2 / B32_DATA_NUM_PER_BLOCK), 0, 0});
            } else {
                DataCopyPad(gradLocGm_[weightOffset * 2], gradLoc[level * 32],
                    {1, static_cast<uint16_t>(2 * num_points * B32_BYTE_SIZE), 0, 0});
            }
            weightOffset += numPoints_;
        }
    }

    SetVectorMask<float>(FULL_MASK, FULL_MASK);
}


template<int32_t num_points, int32_t embed_dims>
__aicore__ inline void KernelMultiScaleDeformableAttnGradOpt<num_points, embed_dims>::Process()
{
    LocalTensor<uint32_t> gatherOffset = gatherOffsetBuf_.Get<uint32_t>();
    LocalTensor<float> location = locationQue_.Get<float>();
    LocalTensor<float> attentionWeight = attentionWeightsQue_.Get<float>();
    LocalTensor<int32_t> shapes = shapeQue_.Get<int32_t>();
    LocalTensor<int32_t> offset = offsetQue_.Get<int32_t>();
    LocalTensor<float> value = valueQue_.Get<float>();
    LocalTensor<float> cornerWeight = cornerWeightBuf_.Get<float>();
    LocalTensor<float> reducedValue = reducedValueBuf_.Get<float>();
    LocalTensor<float> valueDiff = valueDiffBuf_.Get<float>();
    LocalTensor<float> gradOut = gradOutQue_.Get<float>();
    LocalTensor<float> gradValue = gradValueQue_.Get<float>();
    LocalTensor<float> gradLoc = gradLocQue_.Get<float>();
    LocalTensor<float> gradWeight = gradAttentionWeightsQue_.Get<float>();

    LocalTensor<float> shapeBrc = shapeBrcBuf_.Get<float>();
    LocalTensor<int32_t> locInt = locIntBuf_.Get<int32_t>();
    LocalTensor<float> locFloat = locFloatBuf_.Get<float>();
    LocalTensor<float> production = productionBuf_.Get<float>();
    LocalTensor<float> weight = weightBuf_.Get<float>();

    PrepareGatherOffset(gatherOffset);
    PrepareShape(shapes, offset, shapeBrc);
    Duplicate<float, false>(value, 0.f, MASK_PLACEHOLDER, 2 * valRptTimes4_, 1, 8);
    SetFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE2>(1);

    for (uint32_t batch = 0; batch < batchSize_; ++batch) {
        for (uint32_t query = startOffset_; query < endOffset_; ++query) {
            for (uint32_t pl = 0; pl < pointLoops_; ++pl) {
                baseSrcOffset_ = batch * numHeads_ * numKeys_;
                baseDstOffset_ = (batch * numQueries_ + query) * numHeads_ * embedDims_;
                CopyInSample(location[2 * alignedOneQueryNum_], attentionWeight, batch, query, pl);
                CopyInGradOut(gradOut, batch, query);
                ComputeLocation(location, shapeBrc, locInt, locFloat);
                ComputeWeight(locInt, locFloat, shapeBrc, production, weight, attentionWeight);
                ComputeBilinearInterpolation(shapes, offset, locInt, locFloat, value, production, weight, gradOut,
                    gradValue, cornerWeight, reducedValue, valueDiff, gradLoc, gradWeight, gatherOffset);
            }
        }
    }
    WaitFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(1);
    PipeBarrier<PIPE_ALL>();
}

#endif // MS_DEFORM_ATTN_GRAD_HIGH_PERF_H_
