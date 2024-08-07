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

#ifndef MS_DEFORM_ATTN_HIGH_PERF_H_
#define MS_DEFORM_ATTN_HIGH_PERF_H_

#include "kernel_operator.h"

using namespace AscendC;

template<int32_t num_points>
class KernelMultiScaleDeformableAttnOpt {
public:
    __aicore__ inline KernelMultiScaleDeformableAttnOpt() = delete;

    __aicore__ inline KernelMultiScaleDeformableAttnOpt(GM_ADDR value, GM_ADDR valueSpatialShapes,
        GM_ADDR valueLevelStartIndex, GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output,
        const MultiScaleDeformableAttnTilingData* tilingData, TPipe* pipe)
        : pipe_(pipe), blkIdx_(GetBlockIdx())
    {
        InitTiling(tilingData);
        InitTask();
        InitGM(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations, attentionWeights, output);
        InitBuffer();
        InitEvent();

        SetVectorMask<float>(FULL_MASK, FULL_MASK);
        SetAtomicAdd<float>();
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

    __aicore__ inline void InitTiling(const MultiScaleDeformableAttnTilingData* tilingData)
    {
        batchSize_ = tilingData->batchSize;
        numKeys_ = tilingData->numKeys;
        numHeads_ = tilingData->numHeads;
        embedDims_ = 32;
        numLevels_ = tilingData->numLevels;
        numQueries_ = tilingData->numQueries;
        numPoints_ = tilingData->numPoints;
        coreNum_ = tilingData->coreNum;

        oneQueryNum_ = numLevels_ * numHeads_ * numPoints_;
        oneQueryBlk_ = DivCeil(oneQueryNum_, B32_DATA_NUM_PER_BLOCK);

        alignedNumPoints_ = AlignUp(numPoints_, B32_DATA_NUM_PER_BLOCK);
        alignedOneLevelNum_ = numHeads_ * alignedNumPoints_;
        alignedOneQueryNum_ = AlignUp(numLevels_ * alignedOneLevelNum_, B32_DATA_NUM_PER_REPEAT);
        alignedEmbedDims_ = AlignUp(embedDims_, B32_DATA_NUM_PER_BLOCK);
        alignedHeadEmbedDims_ = AlignUp(4 * numPoints_ * alignedEmbedDims_, B32_DATA_NUM_PER_REPEAT);

        embedBlk_ = alignedEmbedDims_ / B32_DATA_NUM_PER_BLOCK;
        pointBlk_ = alignedNumPoints_ / B32_DATA_NUM_PER_BLOCK;
        headBlk_ = numHeads_ * pointBlk_;
        rptTimes_ = alignedOneQueryNum_ / B32_DATA_NUM_PER_REPEAT;
        valRptTimes4_ = alignedHeadEmbedDims_ / B32_DATA_NUM_PER_REPEAT;
        valRptTimes3_ = DivCeil(3 * numPoints_ * alignedEmbedDims_, B32_DATA_NUM_PER_REPEAT);
        valRptTimes2_ = DivCeil(2 * numPoints_ * alignedEmbedDims_, B32_DATA_NUM_PER_REPEAT);
        valRptTimes1_ = DivCeil(numPoints_ * alignedEmbedDims_, B32_DATA_NUM_PER_REPEAT);

        cpDoubleSampleParams_.blockLen = oneQueryBlk_;
        cpDoubleSampleParams_.dstStride = alignedOneQueryNum_ / B32_DATA_NUM_PER_BLOCK - oneQueryBlk_;
        cpSampleParams_.blockCount = numLevels_ * numHeads_;
        cpSampleParams_.blockLen = B32_BYTE_SIZE * numPoints_;
        cpSamplePadParams_.rightPadding = alignedNumPoints_ - numPoints_;

        cpOneValParams_.blockLen = embedBlk_;
        cpDoubleValParams_.blockLen = embedBlk_;
        cpDoubleValParams_.dstStride = 2 * numPoints_ * embedBlk_ - embedBlk_;

        cpValParams_.blockLen = B32_BYTE_SIZE * embedDims_;
        cpValParams_.dstStride = 2 * numPoints_ * embedBlk_ - embedBlk_;
        cpValPadParams_.rightPadding = alignedEmbedDims_ - embedDims_;

        dstRptStride_ = 8 * embedBlk_;
    }

    __aicore__ inline void InitGM(GM_ADDR value, GM_ADDR valueSpatialShapes, GM_ADDR valueLevelStartIndex,
        GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output)
    {
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(value));
        locationGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(samplingLocations));
        attentionWeightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(attentionWeights));

        valueSpatialShapesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valueSpatialShapes));
        valueLevelStartIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valueLevelStartIndex));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    }

    __aicore__ inline void InitBuffer()
    {
        pipe_->InitBuffer(shapeQue_, AlignUp(numLevels_ * 2, B32_DATA_NUM_PER_BLOCK) * B32_BYTE_SIZE);
        pipe_->InitBuffer(offsetQue_, AlignUp(numLevels_, B32_DATA_NUM_PER_BLOCK) * B32_BYTE_SIZE);
        pipe_->InitBuffer(locationQue_, 2 * alignedOneQueryNum_ * B32_BYTE_SIZE); // x, y
        pipe_->InitBuffer(attentionWeightsQue_, alignedOneQueryNum_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(valueQue_, 2 * alignedHeadEmbedDims_ * B32_BYTE_SIZE); // 2 for double buffer
        pipe_->InitBuffer(
            outputQue_, numHeads_ * alignedEmbedDims_ * B32_BYTE_SIZE + B32_DATA_NUM_PER_REPEAT * B32_BYTE_SIZE);

        pipe_->InitBuffer(shapeBrcBuf_, 2 * alignedOneQueryNum_ * B32_BYTE_SIZE); // w, h
        pipe_->InitBuffer(locIntBuf_, 4 * alignedOneQueryNum_ * B32_BYTE_SIZE);   // x0, y0, x1, y1
        pipe_->InitBuffer(locFloatBuf_, 4 * alignedOneQueryNum_ * B32_BYTE_SIZE); // lw, lh
        pipe_->InitBuffer(weightBuf_, 4 * alignedOneQueryNum_ * B32_BYTE_SIZE);   // w1-w4
        pipe_->InitBuffer(cornerWeightBuf_, 4 * alignedNumPoints_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(cornerWeightBrcBuf_, 4 * numPoints_ * alignedEmbedDims_ * B32_BYTE_SIZE);
    }

    __aicore__ inline void InitEvent()
    {
        calEvt_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        copyEvt_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
    }

    __aicore__ inline void PrepareShape(
        const LocalTensor<int32_t>& shapes, const LocalTensor<int32_t>& offset, LocalTensor<float>& shapeBrc);

    __aicore__ inline void CopyInSample(
        const LocalTensor<float>& location, const LocalTensor<float>& attentionWeight, uint32_t batch, uint32_t query);

    __aicore__ inline void ComputeLocation(const LocalTensor<float>& location, const LocalTensor<float>& shapes,
        const LocalTensor<int32_t>& locInt, const LocalTensor<float>& locFloat);

    __aicore__ inline void ComputeWeight(const LocalTensor<int32_t>& locInt, const LocalTensor<float>& locFloat,
        const LocalTensor<float>& weight, const LocalTensor<float>& attentionWeight);

    __aicore__ inline void ComputeBilinearInterpolation(const LocalTensor<int32_t>& shapes,
        const LocalTensor<int32_t>& offset, const LocalTensor<int32_t>& locInt, const LocalTensor<float>& value,
        const LocalTensor<float>& weight, const LocalTensor<float>& cornerWeight,
        const LocalTensor<float>& cornerWeightBrc, const LocalTensor<float>& output);

private:
    TPipe* pipe_;
    GlobalTensor<float> valueGm_, locationGm_, attentionWeightsGm_, outputGm_;
    GlobalTensor<int32_t> valueSpatialShapesGm_, valueLevelStartIndexGm_;

    TBuf<TPosition::VECCALC> locationQue_, attentionWeightsQue_, shapeQue_, offsetQue_, valueQue_;
    TBuf<TPosition::VECCALC> outputQue_;

    TBuf<TPosition::VECCALC> locIntBuf_, locFloatBuf_, shapeBrcBuf_, weightBuf_, cornerWeightBuf_, cornerWeightBrcBuf_;

    int32_t blkIdx_;

    uint32_t batchSize_, numKeys_, numHeads_, embedDims_, numLevels_, numQueries_, numPoints_, coreNum_;
    uint32_t startOffset_, endOffset_;
    uint32_t alignedNumPoints_, alignedOneLevelNum_, alignedOneQueryNum_, alignedEmbedDims_, alignedHeadEmbedDims_;
    uint32_t oneQueryNum_;
    uint16_t pointBlk_, headBlk_, oneQueryBlk_, embedBlk_, dstRptStride_;
    uint16_t rptTimes_, valRptTimes4_, valRptTimes3_, valRptTimes2_, valRptTimes1_;

    TEventID calEvt_, copyEvt_;

    uint32_t baseSrcOffset_, baseDstOffset_, srcOffset_, dstOffset_;

    DataCopyParams cpOneValParams_, cpDoubleValParams_ {2, 0, 0, 0}, cpDoubleSampleParams_ {2, 0, 0, 0};
    DataCopyExtParams cpSampleParams_, cpValParams_ {2, 0, 0, 0, 0};
    DataCopyPadExtParams<float> cpSamplePadParams_, cpValPadParams_;
};

template<int32_t num_points>
__aicore__ inline void KernelMultiScaleDeformableAttnOpt<num_points>::PrepareShape(
    const LocalTensor<int32_t>& shapes, const LocalTensor<int32_t>& offset, LocalTensor<float>& shapeBrc)
{
    DataCopy(shapes, valueSpatialShapesGm_,
        {1, static_cast<uint16_t>(DivCeil(2 * numLevels_, B32_DATA_NUM_PER_BLOCK)), 0, 0});
    DataCopy(
        offset, valueLevelStartIndexGm_, {1, static_cast<uint16_t>(DivCeil(numLevels_, B32_DATA_NUM_PER_BLOCK)), 0, 0});
    SetFlag<HardEvent::MTE2_V>(copyEvt_);
    WaitFlag<HardEvent::MTE2_V>(copyEvt_);
    for (uint32_t k = 0; k < 2; ++k) {
        for (uint32_t i = 0; i < numLevels_; ++i) {
            shapeBrc.SetValue(i + k * alignedOneQueryNum_, shapes.GetValue(2 * i + 1 - k));
        }
        Brcb(shapeBrc[k * alignedOneQueryNum_], shapeBrc[k * alignedOneQueryNum_], 1, {headBlk_, 8});
        for (uint16_t i = 1; i < headBlk_; ++i) {
            Copy<float, false>(shapeBrc[k * alignedOneQueryNum_ + i * B32_DATA_NUM_PER_BLOCK],
                shapeBrc[k * alignedOneQueryNum_], MASK_PLACEHOLDER, 1,
                {headBlk_, headBlk_, static_cast<uint16_t>(8 * headBlk_), static_cast<uint16_t>(8 * headBlk_)});
        }
    }
}

template<int32_t num_points>
__aicore__ inline void KernelMultiScaleDeformableAttnOpt<num_points>::CopyInSample(
    const LocalTensor<float>& location, const LocalTensor<float>& attentionWeight, uint32_t batch, uint32_t query)
{
    uint32_t sampleOffset = (batch * numQueries_ + query) * oneQueryNum_;
    WaitFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(1);
    if (num_points == 8) {
        DataCopy(location, locationGm_[sampleOffset * 2], cpDoubleSampleParams_);
        DataCopy(attentionWeight, attentionWeightsGm_[sampleOffset], {1, oneQueryBlk_, 0, 0});
    } else {
        DataCopyPad(location, locationGm_[sampleOffset * 2], cpSampleParams_, cpSamplePadParams_);
        DataCopyPad(location[alignedOneQueryNum_], locationGm_[sampleOffset * 2 + oneQueryNum_], cpSampleParams_,
            cpSamplePadParams_);
        DataCopyPad(attentionWeight, attentionWeightsGm_[sampleOffset], cpSampleParams_, cpSamplePadParams_);
    }

    SetFlag<HardEvent::MTE2_V>(copyEvt_);
}

template<int32_t num_points>
__aicore__ inline void KernelMultiScaleDeformableAttnOpt<num_points>::ComputeLocation(
    const LocalTensor<float>& location, const LocalTensor<float>& shapes, const LocalTensor<int32_t>& locInt,
    const LocalTensor<float>& locFloat)
{
    WaitFlag<HardEvent::MTE2_V>(copyEvt_);
    Mul<float, false>(location, location, shapes, MASK_PLACEHOLDER, 2 * rptTimes_, {1, 1, 1, 8, 8, 8});
    Adds<float, false>(locFloat, location, 0.5f, MASK_PLACEHOLDER, 2 * rptTimes_, {1, 1, 8, 8});
    Cast<int32_t, float, false>(locInt, locFloat, RoundMode::CAST_FLOOR, MASK_PLACEHOLDER, 2 * rptTimes_, {1, 1, 8, 8});
    SetFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE2>(1);
}

template<int32_t num_points>
__aicore__ inline void KernelMultiScaleDeformableAttnOpt<num_points>::ComputeWeight(const LocalTensor<int32_t>& locInt,
    const LocalTensor<float>& locFloat, const LocalTensor<float>& weight, const LocalTensor<float>& attentionWeight)
{
    Cast<float, int32_t, false>(
        locFloat[2 * alignedOneQueryNum_], locInt, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 2 * rptTimes_, {1, 1, 8, 8});
    Sub<float, false>(locFloat, locFloat, locFloat[2 * alignedOneQueryNum_], MASK_PLACEHOLDER, 2 * rptTimes_,
        {1, 1, 1, 8, 8, 8}); // lh, lw
    Mul<float, false>(weight[3 * alignedOneQueryNum_], locFloat, locFloat[alignedOneQueryNum_], MASK_PLACEHOLDER,
        rptTimes_, {1, 1, 1, 8, 8, 8}); // lh * lw
    Duplicate<float, false>(weight, 1.f, MASK_PLACEHOLDER, rptTimes_, 1, 8);
    Sub<float, false>(weight, weight, locFloat, MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
    Sub<float, false>(weight, weight, locFloat[alignedOneQueryNum_], MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
    Add<float, false>(weight, weight, weight[3 * alignedOneQueryNum_], MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
    Sub<float, false>(weight[alignedOneQueryNum_], locFloat[alignedOneQueryNum_], weight[3 * alignedOneQueryNum_],
        MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
    Sub<float, false>(weight[2 * alignedOneQueryNum_], locFloat, weight[3 * alignedOneQueryNum_], MASK_PLACEHOLDER,
        rptTimes_, {1, 1, 1, 8, 8, 8});

    Mul<float, false>(weight, weight, attentionWeight, MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(weight[alignedOneQueryNum_], weight[alignedOneQueryNum_], attentionWeight, MASK_PLACEHOLDER,
        rptTimes_, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(weight[2 * alignedOneQueryNum_], weight[2 * alignedOneQueryNum_], attentionWeight,
        MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(weight[3 * alignedOneQueryNum_], weight[3 * alignedOneQueryNum_], attentionWeight,
        MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
}

template<int32_t num_points>
__aicore__ inline void KernelMultiScaleDeformableAttnOpt<num_points>::ComputeBilinearInterpolation(
    const LocalTensor<int32_t>& shapes, const LocalTensor<int32_t>& offset, const LocalTensor<int32_t>& locInt,
    const LocalTensor<float>& value, const LocalTensor<float>& weight, const LocalTensor<float>& cornerWeight,
    const LocalTensor<float>& cornerWeightBrc, const LocalTensor<float>& output)
{
    uint8_t ping = 0;
    for (uint32_t level = 0; level < numLevels_; ++level) {
        uint32_t valueOffset = baseSrcOffset_ + offset.GetValue(level);
        int32_t h = shapes.GetValue(level * 2);
        int32_t w = shapes.GetValue(level * 2 + 1);

        for (uint32_t head = 0; head < numHeads_; ++head) {
            uint32_t outOffset = head * alignedEmbedDims_;
            srcOffset_ = (valueOffset + head * numKeys_) * embedDims_;
            dstOffset_ = baseDstOffset_ + head * embedDims_;

            uint32_t sx = level * alignedOneLevelNum_ + head * alignedNumPoints_;
            uint32_t sy = sx + alignedOneQueryNum_;
            uint32_t pingOffset = ping * alignedHeadEmbedDims_;

            WaitFlag<HardEvent::V_MTE2>(ping);
            for (uint32_t point = 0; point < numPoints_; ++point) {
                int32_t y1 = locInt.GetValue(point + sy);
                int32_t x1 = locInt.GetValue(point + sx);
                int32_t y0 = y1 - 1;
                int32_t x0 = x1 - 1;

                if (0 <= y0 && y0 < h) {
                    if (0 < x1 && x1 < w) {
                        DataCopy(value[pingOffset + point * alignedEmbedDims_],
                            valueGm_[srcOffset_ + (y0 * w + x0) * embedDims_], cpDoubleValParams_);
                    } else if (0 <= x0 && x0 < w) {
                        DataCopy(value[pingOffset + point * alignedEmbedDims_],
                            valueGm_[srcOffset_ + (y0 * w + x0) * embedDims_], cpOneValParams_);
                    } else if (0 <= x1 && x1 < w) {
                        DataCopy(value[pingOffset + point * alignedEmbedDims_ + 2 * numPoints_ * alignedEmbedDims_],
                            valueGm_[srcOffset_ + (y0 * w + x1) * embedDims_], cpOneValParams_);
                    }
                }
                if (0 <= y1 && y1 < h) {
                    if (0 < x1 && x1 < w) {
                        DataCopy(value[pingOffset + point * alignedEmbedDims_ + numPoints_ * alignedEmbedDims_],
                            valueGm_[srcOffset_ + (y1 * w + x0) * embedDims_], cpDoubleValParams_);
                    } else if (0 <= x0 && x0 < w) {
                        DataCopy(value[pingOffset + point * alignedEmbedDims_ + numPoints_ * alignedEmbedDims_],
                            valueGm_[srcOffset_ + (y1 * w + x0) * embedDims_], cpOneValParams_);
                    } else if (0 <= x1 && x1 < w) {
                        DataCopy(value[pingOffset + point * alignedEmbedDims_ + 3 * numPoints_ * alignedEmbedDims_],
                            valueGm_[srcOffset_ + (y1 * w + x1) * embedDims_], cpOneValParams_);
                    }
                }
            }
            SetFlag<HardEvent::MTE2_V>(copyEvt_);

            Copy<float, false>(
                cornerWeight, weight[sx], MASK_PLACEHOLDER, 1, {1, static_cast<uint16_t>(headBlk_ * numLevels_), 8, 8});
            for (uint32_t i = 0; i < 4; ++i) {
                Brcb(cornerWeightBrc[i * numPoints_ * alignedEmbedDims_], cornerWeight[i * alignedNumPoints_], 1,
                    {embedBlk_, dstRptStride_});
            }
            for (uint32_t i = 1; i < embedBlk_; ++i) {
                Copy<float, false>(cornerWeightBrc[i * B32_DATA_NUM_PER_BLOCK], cornerWeightBrc, MASK_PLACEHOLDER, 4,
                    {embedBlk_, embedBlk_, dstRptStride_, dstRptStride_});
            }

            WaitFlag<HardEvent::MTE2_V>(copyEvt_);
            Mul<float, false>(cornerWeightBrc, value[pingOffset], cornerWeightBrc, MASK_PLACEHOLDER, valRptTimes4_,
                {1, 1, 1, 8, 8, 8});

            Duplicate<float, false>(value[pingOffset], 0.f, MASK_PLACEHOLDER, valRptTimes4_, 1, 8);
            SetFlag<HardEvent::V_MTE2>(ping);
            ping = 1 - ping;

            Add<float, false>(cornerWeightBrc, cornerWeightBrc, cornerWeightBrc[2 * numPoints_ * alignedEmbedDims_],
                MASK_PLACEHOLDER, valRptTimes2_, {1, 1, 1, 8, 8, 8});
            Add<float, false>(cornerWeightBrc, cornerWeightBrc, cornerWeightBrc[numPoints_ * alignedEmbedDims_],
                MASK_PLACEHOLDER, valRptTimes1_, {1, 1, 1, 8, 8, 8});
            if (num_points == 8) {
                Add<float, false>(
                    cornerWeightBrc, cornerWeightBrc, cornerWeightBrc[128], MASK_PLACEHOLDER, 2, {1, 1, 1, 8, 8, 8});
            }
            if (num_points >= 4) {
                Add<float, false>(
                    cornerWeightBrc, cornerWeightBrc, cornerWeightBrc[64], MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
            }
            if (num_points >= 2) {
                Add<float, false>(
                    output[outOffset], cornerWeightBrc, cornerWeightBrc[32], MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
            }
            SetFlag<HardEvent::V_MTE3>(calEvt_);
            WaitFlag<HardEvent::V_MTE3>(calEvt_);
            DataCopy(outputGm_[dstOffset_], output[outOffset], {1, embedBlk_, 0, 0});
            PipeBarrier<PIPE_MTE3>();
        }
    }
}


template<int32_t num_points>
__aicore__ inline void KernelMultiScaleDeformableAttnOpt<num_points>::Process()
{
    LocalTensor<float> location = locationQue_.Get<float>();
    LocalTensor<float> attentionWeight = attentionWeightsQue_.Get<float>();
    LocalTensor<int32_t> shapes = shapeQue_.Get<int32_t>();
    LocalTensor<int32_t> offset = offsetQue_.Get<int32_t>();
    LocalTensor<float> value = valueQue_.Get<float>();
    LocalTensor<float> cornerWeight = cornerWeightBuf_.Get<float>();
    LocalTensor<float> cornerWeightBrc = cornerWeightBrcBuf_.Get<float>();
    LocalTensor<float> output = outputQue_.Get<float>();

    LocalTensor<float> shapeBrc = shapeBrcBuf_.Get<float>();
    LocalTensor<int32_t> locInt = locIntBuf_.Get<int32_t>();
    LocalTensor<float> locFloat = locFloatBuf_.Get<float>();
    LocalTensor<float> weight = weightBuf_.Get<float>();

    PrepareShape(shapes, offset, shapeBrc);
    Duplicate<float, false>(value, 0.f, MASK_PLACEHOLDER, 2 * valRptTimes4_, 1, 8);
    SetFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE2>(1);

    for (uint32_t batch = 0; batch < batchSize_; ++batch) {
        for (uint32_t query = startOffset_; query < endOffset_; ++query) {
            baseSrcOffset_ = batch * numHeads_ * numKeys_;
            baseDstOffset_ = (batch * numQueries_ + query) * numHeads_ * embedDims_;

            CopyInSample(location, attentionWeight, batch, query);
            ComputeLocation(location, shapeBrc, locInt, locFloat);
            ComputeWeight(locInt, locFloat, weight, attentionWeight);
            ComputeBilinearInterpolation(shapes, offset, locInt, value, weight, cornerWeight, cornerWeightBrc, output);
        }
    }
    WaitFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(1);
    SetAtomicNone();
}
#endif // MS_DEFORM_ATTN_HIGH_PERF_H_
