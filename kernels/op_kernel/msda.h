#ifndef MSDA_H
#define MSDA_H

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
 * \file msda.h
 * \brief msda & msda_grad operator
 */

#include "kernel_operator.h"
#include "kernel_tpipe_impl.h"
#include "kernel_utils.h"

using namespace AscendC;

template<bool aligned, bool forward, bool fastMode>
class MSDABaseKernel {
public:
    __aicore__ inline MSDABaseKernel() = delete;

    __aicore__ inline MSDABaseKernel(GM_ADDR value, GM_ADDR valueSpatialShapes, GM_ADDR valueLevelStartIndex,
        GM_ADDR samplingLocations, GM_ADDR attentionWeights, const MultiScaleDeformableAttnTilingData* tilingData,
        TPipe* pipe)
        : pipe_(pipe), blkIdx_(GetBlockIdx())
    {
        InitTiling(tilingData);
        InitTask();
        InitGM(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations, attentionWeights);
        InitBuffer();
        ResetMask();
        SetAtomicNone();
    }

protected:
    __aicore__ inline void InitTask()
    {
        uint32_t avgTasks = (batchSize_ * numQueries_) / coreNum_;
        uint32_t remainTasks = (batchSize_ * numQueries_) % coreNum_;
        startOffset_ = avgTasks * blkIdx_ + (blkIdx_ < remainTasks ? blkIdx_ : remainTasks);
        endOffset_ = startOffset_ + avgTasks + (blkIdx_ < remainTasks ? 1 : 0);
    }

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

        oneQueryNum_ = numHeads_ * realLevels_ * numPoints_;

        oneHeadNum_ = numLevels_ * numPoints_;
        alignedEmbedDims_ = AlignUp(embedDims_, B32_DATA_NUM_PER_BLOCK);
        if constexpr (fastMode) {
            alignedOneHeadNum_ = oneHeadNum_;
            alignedCornerEmbedDims_ = numHeads_ * oneHeadNum_ * alignedEmbedDims_;
            qryRpt_ = 1;
            brcRpt_ = DivCeil(4 * numHeads_ * oneHeadNum_ * B32_DATA_NUM_PER_BLOCK, B32_DATA_NUM_PER_REPEAT);
            outerLoops_ = 1;
            innerLoops_ = numHeads_ * oneHeadNum_;
        } else {
            alignedOneHeadNum_ = B32_DATA_NUM_PER_REPEAT;
            alignedCornerEmbedDims_ = oneHeadNum_ * alignedEmbedDims_;
            qryRpt_ = numHeads_;
            brcRpt_ = DivCeil(4 * oneHeadNum_ * B32_DATA_NUM_PER_BLOCK, B32_DATA_NUM_PER_REPEAT);
            outerLoops_ = numHeads_;
            innerLoops_ = oneHeadNum_;
        }
        alignedOneQueryNum_ = AlignUp(numHeads_ * alignedOneHeadNum_, B32_DATA_NUM_PER_REPEAT);
        alignedHeadEmbedDims_ = numHeads_ * alignedEmbedDims_;
        outDims_ = numHeads_ * embedDims_;
        embedBlk_ = DivCeil(embedDims_, B32_DATA_NUM_PER_BLOCK);
        outBlk_ = numHeads_ * embedBlk_;
        embedMask_ = embedDims_ < 64 ? (1UL << embedDims_) - 1 : FULL_MASK;
        queryBlk_ = alignedOneQueryNum_ / B32_DATA_NUM_PER_BLOCK;
        cornerRpt_ = DivCeil(4 * alignedCornerEmbedDims_, B32_DATA_NUM_PER_REPEAT);

        cpRowDoubleParams_.dstStride =
            alignedCornerEmbedDims_ / B32_DATA_NUM_PER_BLOCK - DivCeil(embedDims_, B32_DATA_NUM_PER_BLOCK);
        cpOutParams_.blockCount = numHeads_;
        if constexpr (aligned) {
            cpOneValParams_.blockLen = embedBlk_;
            cpRowDoubleParams_.blockLen = embedBlk_;
            cpRowDoubleParams_.srcStride = outBlk_ - embedBlk_;
            cpOutParams_.blockLen = embedBlk_;
        } else {
            cpOneValParams_.blockLen = embedDims_ * B32_BYTE_SIZE;
            cpRowDoubleParams_.blockLen = embedDims_ * B32_BYTE_SIZE;
            cpRowDoubleParams_.srcStride = (outDims_ - embedDims_) * B32_BYTE_SIZE;
            cpOutParams_.blockLen = embedDims_ * B32_BYTE_SIZE;
        }

        if (fastMode) {
            cpSampleParams_.blockCount = 1;
            cpSampleParams_.blockLen = numHeads_ * oneHeadNum_ * B32_BYTE_SIZE;
            cpDoubleSampleParams_.blockCount = 1;
            cpDoubleSampleParams_.blockLen = 2 * numHeads_ * oneHeadNum_ * B32_BYTE_SIZE;
        } else {
            cpSampleParams_.blockCount = numHeads_;
            cpSampleParams_.blockLen = oneHeadNum_ * B32_BYTE_SIZE;
            cpSampleParams_.dstStride =
                alignedOneHeadNum_ / B32_DATA_NUM_PER_BLOCK - DivCeil(oneHeadNum_, B32_DATA_NUM_PER_BLOCK);
            cpDoubleSampleParams_.blockCount = numHeads_;
            cpDoubleSampleParams_.blockLen = 2 * oneHeadNum_ * B32_BYTE_SIZE;
            cpDoubleSampleParams_.dstStride =
                2 * alignedOneHeadNum_ / B32_DATA_NUM_PER_BLOCK - DivCeil(2 * oneHeadNum_, B32_DATA_NUM_PER_BLOCK);
        }

        gatherParams_.repeatTimes = qryRpt_ * 2;
    }

    __aicore__ inline void InitGM(GM_ADDR value, GM_ADDR valueSpatialShapes, GM_ADDR valueLevelStartIndex,
        GM_ADDR samplingLocations, GM_ADDR attentionWeights)
    {
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(value));
        locationGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(samplingLocations));
        attentionWeightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(attentionWeights));

        valueSpatialShapesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valueSpatialShapes));
        valueLevelStartIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(valueLevelStartIndex));
    }

    __aicore__ inline void InitBuffer()
    {
        if constexpr (!forward) {
            pipe_->InitBuffer(gatherOffsetBuf_, 2 * alignedOneQueryNum_ * B32_BYTE_SIZE);
            pipe_->InitBuffer(gradLocationQue_, 4 * alignedOneQueryNum_ * B32_BYTE_SIZE); // x, y
            pipe_->InitBuffer(gradAttentionWeightsQue_, alignedOneQueryNum_ * B32_BYTE_SIZE);
        }
        pipe_->InitBuffer(shapeQue_, AlignUp(numLevels_ * 2, B32_DATA_NUM_PER_BLOCK) * B32_BYTE_SIZE);
        pipe_->InitBuffer(offsetQue_, AlignUp(numLevels_, B32_DATA_NUM_PER_BLOCK) * B32_BYTE_SIZE);
        pipe_->InitBuffer(shapeIntBuf_, 2 * alignedOneQueryNum_ * B32_BYTE_SIZE);   // w, h
        pipe_->InitBuffer(shapeFloatBuf_, 2 * alignedOneQueryNum_ * B32_BYTE_SIZE); // w, h
        pipe_->InitBuffer(offsetIntBuf_, alignedOneQueryNum_ * B32_BYTE_SIZE);      // offsetInt
        pipe_->InitBuffer(locIntBuf_, 2 * alignedOneQueryNum_ * B32_BYTE_SIZE);     // x0, y0
        pipe_->InitBuffer(locFloatBuf_, 6 * alignedOneQueryNum_ * B32_BYTE_SIZE);   // lw, lh
        pipe_->InitBuffer(validFlagBuf_, 8 * validFlagMaskLen_);                    // 16blocks
        pipe_->InitBuffer(productionBuf_, 4 * alignedOneQueryNum_ * B32_BYTE_SIZE); // lh * lw
        pipe_->InitBuffer(weightBuf_, 4 * alignedOneQueryNum_ * B32_BYTE_SIZE);     // w1-w4
        pipe_->InitBuffer(locationQue_, 4 * alignedOneQueryNum_ * B32_BYTE_SIZE);   // x, y
        pipe_->InitBuffer(attentionWeightsQue_, alignedOneQueryNum_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(valueQue_, cornerRpt_ * B32_DATA_NUM_PER_REPEAT * B32_BYTE_SIZE);
        pipe_->InitBuffer(outputQue_, alignedHeadEmbedDims_ * B32_BYTE_SIZE);
        // WARN: cornerWeightBrcBuf_ must be at the end of the buffer!
        pipe_->InitBuffer(cornerWeightBrcBuf_, cornerRpt_ * B32_DATA_NUM_PER_REPEAT * B32_BYTE_SIZE);
    }

    __aicore__ inline void PrepareShape(const LocalTensor<int32_t>& shapes, const LocalTensor<int32_t>& shapeInt,
        const LocalTensor<float>& shapeFloat, const LocalTensor<int32_t>& offset, const LocalTensor<int32_t>& offsetInt)
    {
        DataCopy(shapes, valueSpatialShapesGm_,
            {1, static_cast<uint16_t>(DivCeil(2 * numLevels_, B32_DATA_NUM_PER_BLOCK)), 0, 0});
        DataCopy(offset, valueLevelStartIndexGm_,
            {1, static_cast<uint16_t>(DivCeil(numLevels_, B32_DATA_NUM_PER_BLOCK)), 0, 0});
        // broadcast to [head*level, POINT]
        for (uint32_t head = 0; head < numHeads_; ++head) {
            uint32_t idx = head * alignedOneHeadNum_;
            for (uint32_t level = 0; level < numLevels_; ++level) {
                int32_t w = shapes.GetValue(2 * level + 1);
                int32_t h = shapes.GetValue(2 * level);
                int32_t o = offset.GetValue(level);
                for (uint32_t point = 0; point < numPoints_; ++point) {
                    shapeInt.SetValue(idx, w);
                    shapeInt.SetValue(idx + alignedOneQueryNum_, h);
                    offsetInt.SetValue(idx, o * numHeads_ + head);
                    ++idx;
                }
            }
        }
        Cast<float, int32_t, false>(
            shapeFloat, shapeInt, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 2 * qryRpt_, {1, 1, 8, 8});
    }

    __aicore__ inline void CopyInSample(
        const LocalTensor<float>& location, const LocalTensor<float>& attentionWeight, uint32_t taskIdx)
    {
        uint64_t sampleOffset = taskIdx * oneQueryNum_;
        WaitFlag<HardEvent::V_MTE2>(copyEvt_);
        DataCopyPad(location, locationGm_[sampleOffset * 2], cpDoubleSampleParams_, {});
        DataCopyPad(attentionWeight, attentionWeightsGm_[sampleOffset], cpSampleParams_, {});
        SetFlag<HardEvent::MTE2_V>(copyEvt_);
    }

    __aicore__ inline void ComputeLocation(uint32_t taskIdx, const LocalTensor<float>& locationFloat,
        const LocalTensor<int32_t>& locationInt, const LocalTensor<float>& shapeFloat,
        const LocalTensor<int32_t>& shapeInt, const LocalTensor<float>& locFloat, const LocalTensor<int32_t>& locInt,
        const LocalTensor<int32_t>& offsetInt, const LocalTensor<uint8_t>& validFlag);

    __aicore__ inline void ComputeWeight(const LocalTensor<float>& locFloat, const LocalTensor<float>& shapes,
        const LocalTensor<float>& production, const LocalTensor<float>& weight,
        const LocalTensor<float>& attentionWeight);

    __aicore__ inline void CopyInValue(
        const LocalTensor<float>& dst, const GlobalTensor<float>& src, const DataCopyParams& cpParams)
    {
        if constexpr (aligned) {
            DataCopy(dst, src, cpParams);
        } else {
            DataCopyPad(dst, src, cpParams, {});
        }
    }

    __aicore__ inline void CopyOutValue(
        const GlobalTensor<float>& dst, const LocalTensor<float>& src, const DataCopyParams& cpParams)
    {
        if constexpr (aligned) {
            DataCopy(dst, src, cpParams);
        } else {
            DataCopyPad(dst, src, cpParams);
        }
    }

protected:
    TPipe* pipe_;
    GlobalTensor<float> valueGm_, locationGm_, attentionWeightsGm_;
    GlobalTensor<int32_t> valueSpatialShapesGm_, valueLevelStartIndexGm_;

    TBuf<TPosition::VECCALC> locationQue_, attentionWeightsQue_, shapeQue_, offsetQue_, valueQue_;
    TBuf<TPosition::VECCALC> outputQue_;

    TBuf<TPosition::VECCALC> locIntBuf_, locFloatBuf_, shapeIntBuf_, shapeFloatBuf_, offsetIntBuf_, productionBuf_,
        weightBuf_, cornerWeightBrcBuf_, validFlagBuf_, gatherOffsetBuf_;

    TBuf<TPosition::VECCALC> gradLocationQue_, gradAttentionWeightsQue_;

    int32_t blkIdx_;

    // const values
    uint32_t coreNum_;
    uint32_t startOffset_, endOffset_;
    uint64_t batchSize_, numKeys_, numHeads_, embedDims_, outDims_, numLevels_, numQueries_, numPoints_, realLevels_;
    uint32_t alignedOneHeadNum_, alignedOneQueryNum_, alignedEmbedDims_, alignedCornerEmbedDims_, alignedHeadEmbedDims_;
    uint32_t oneHeadNum_, oneQueryNum_;
    uint32_t outerLoops_, innerLoops_;
    uint16_t tailBrcBlk_, queryBlk_, embedBlk_, outBlk_;
    uint16_t brcRpt_, qryRpt_, cornerRpt_;
    uint64_t embedMask_;
    uint32_t validFlagMaskLen_ {64};
    TEventID copyEvt_ {2}, biEvt_ {3}; // biEvt_ is used for bilinear interpolation
    DataCopyParams cpOneValParams_, cpRowDoubleParams_ {2, 0, 0, 0}, cpSampleParams_, cpDoubleSampleParams_,
        cpOutParams_;
    GatherMaskParams gatherParams_;
};

template<bool aligned, bool forward, bool fastMode>
__aicore__ inline void MSDABaseKernel<aligned, forward, fastMode>::ComputeLocation(uint32_t taskIdx,
    const LocalTensor<float>& locationFloat, const LocalTensor<int32_t>& locationInt,
    const LocalTensor<float>& shapeFloat, const LocalTensor<int32_t>& shapeInt, const LocalTensor<float>& locFloat,
    const LocalTensor<int32_t>& locInt, const LocalTensor<int32_t>& offsetInt, const LocalTensor<uint8_t>& validFlag)
{
    uint64_t cnt;
    int32_t baseSrcOffset = taskIdx / numQueries_ * numKeys_ * numHeads_;
    WaitFlag<HardEvent::MTE2_V>(copyEvt_);

    GatherMask(locationFloat, locationFloat[2 * alignedOneQueryNum_], 1, false, MASK_PLACEHOLDER, gatherParams_, cnt);
    GatherMask(locationFloat[alignedOneQueryNum_], locationFloat[2 * alignedOneQueryNum_], 2, false, MASK_PLACEHOLDER,
        gatherParams_, cnt);
    ResetMask();

    Mul<float, false>(locationFloat, locationFloat, shapeFloat, MASK_PLACEHOLDER, 2 * qryRpt_, {1, 1, 1, 8, 8, 8});
    Adds<float, false>(locFloat, locationFloat, 0.5f, MASK_PLACEHOLDER, 2 * qryRpt_, {1, 1, 8, 8});
    Cast<int32_t, float, false>(locInt, locFloat, RoundMode::CAST_FLOOR, MASK_PLACEHOLDER, 2 * qryRpt_, {1, 1, 8, 8});
    // fix the precesion issue of the floor operation(0.9999f -> 1.0f)
    Cast<float, int32_t, false>(
        locFloat[2 * alignedOneQueryNum_], locInt, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 2 * qryRpt_, {1, 1, 8, 8});
    Compare<float, uint8_t, false>(validFlag, locFloat[2 * alignedOneQueryNum_], locFloat, CMPMODE::GT,
        MASK_PLACEHOLDER, 2 * qryRpt_, {1, 1, 1, 8, 8, 8});
    Adds<int32_t, false>(locFloat[2 * alignedOneQueryNum_].ReinterpretCast<int32_t>(), locInt, 0, MASK_PLACEHOLDER,
        2 * qryRpt_, {1, 1, 8, 8});
    Adds<int32_t, false>(locInt, locInt, -1, MASK_PLACEHOLDER, 2 * qryRpt_, {1, 1, 8, 8});
    Select<float, uint8_t, false>(locInt.ReinterpretCast<float>(), validFlag, locInt.ReinterpretCast<float>(),
        locFloat[2 * alignedOneQueryNum_], SELMODE::VSEL_TENSOR_TENSOR_MODE, 64, 2 * qryRpt_, {1, 1, 1, 8, 8, 8});
    // fix end
    Cast<float, int32_t, false>(
        locFloat[2 * alignedOneQueryNum_], locInt, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 2 * qryRpt_, {1, 1, 8, 8});
    Adds<int32_t, false>(locInt, locInt, -1, MASK_PLACEHOLDER, 2 * qryRpt_, {1, 1, 8, 8});

    Mul<int32_t, false>(
        locationInt, locInt[alignedOneQueryNum_], shapeInt, MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
    Add<int32_t, false>(locationInt, locInt, locationInt, MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
    Muls<int32_t, false>(locationInt, locationInt, numHeads_, MASK_PLACEHOLDER, qryRpt_, {1, 1, 8, 8});
    Add<int32_t, false>(locationInt, locationInt, offsetInt, MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
    Adds<int32_t, false>(locationInt, locationInt, baseSrcOffset, MASK_PLACEHOLDER, qryRpt_, {1, 1, 8, 8});
    // WARN: it's dangerous to use int32_t type for global memory address.
    Muls<int32_t, false>(locationInt, locationInt, embedDims_, MASK_PLACEHOLDER, qryRpt_, {1, 1, 8, 8});
    Adds<float, false>(locFloat[4 * alignedOneQueryNum_], locFloat[2 * alignedOneQueryNum_], -1.f, MASK_PLACEHOLDER,
        2 * qryRpt_, {1, 1, 8, 8});

    CompareScalar<float, uint8_t, false>(
        validFlag, locFloat[4 * alignedOneQueryNum_], 0.f, CMPMODE::GE, MASK_PLACEHOLDER, qryRpt_, {1, 1, 8, 8});
    CompareScalar<float, uint8_t, false>(validFlag[validFlagMaskLen_], locFloat[5 * alignedOneQueryNum_], 0.f,
        CMPMODE::GE, MASK_PLACEHOLDER, qryRpt_, {1, 1, 8, 8});
    Compare<float, uint8_t, false>(validFlag[2 * validFlagMaskLen_], locFloat[2 * alignedOneQueryNum_], shapeFloat,
        CMPMODE::LT, MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
    Compare<float, uint8_t, false>(validFlag[3 * validFlagMaskLen_], locFloat[3 * alignedOneQueryNum_],
        shapeFloat[alignedOneQueryNum_], CMPMODE::LT, MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
    And<uint16_t, false>(validFlag.ReinterpretCast<uint16_t>(), validFlag.ReinterpretCast<uint16_t>(),
        validFlag[2 * validFlagMaskLen_].ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
    And<uint16_t, false>(validFlag.ReinterpretCast<uint16_t>(), validFlag.ReinterpretCast<uint16_t>(),
        validFlag[validFlagMaskLen_].ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});

    Compare<float, uint8_t, false>(validFlag[validFlagMaskLen_], locFloat[4 * alignedOneQueryNum_], shapeFloat,
        CMPMODE::GE, MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
    CompareScalar<float, uint8_t, false>(validFlag[2 * validFlagMaskLen_], locFloat[2 * alignedOneQueryNum_], 0.f,
        CMPMODE::LT, MASK_PLACEHOLDER, qryRpt_, {1, 1, 8, 8});
    Or<uint16_t, false>(validFlag[validFlagMaskLen_].ReinterpretCast<uint16_t>(),
        validFlag[validFlagMaskLen_].ReinterpretCast<uint16_t>(),
        validFlag[2 * validFlagMaskLen_].ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
    Or<uint16_t, false>(validFlag[validFlagMaskLen_].ReinterpretCast<uint16_t>(),
        validFlag[validFlagMaskLen_].ReinterpretCast<uint16_t>(), validFlag.ReinterpretCast<uint16_t>(),
        MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});

    Compare<float, uint8_t, false>(validFlag[2 * validFlagMaskLen_], locFloat[5 * alignedOneQueryNum_],
        shapeFloat[alignedOneQueryNum_], CMPMODE::GE, MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
    CompareScalar<float, uint8_t, false>(validFlag[3 * validFlagMaskLen_], locFloat[5 * alignedOneQueryNum_], 0.f,
        CMPMODE::LT, MASK_PLACEHOLDER, qryRpt_, {1, 1, 8, 8});
    Or<uint16_t, false>(validFlag[2 * validFlagMaskLen_].ReinterpretCast<uint16_t>(),
        validFlag[2 * validFlagMaskLen_].ReinterpretCast<uint16_t>(),
        validFlag[3 * validFlagMaskLen_].ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
    Or<uint16_t, false>(validFlag[2 * validFlagMaskLen_].ReinterpretCast<uint16_t>(),
        validFlag[2 * validFlagMaskLen_].ReinterpretCast<uint16_t>(),
        validFlag[validFlagMaskLen_].ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});

    Compare<float, uint8_t, false>(validFlag[3 * validFlagMaskLen_], locFloat[3 * alignedOneQueryNum_],
        shapeFloat[alignedOneQueryNum_], CMPMODE::GE, MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
    CompareScalar<float, uint8_t, false>(validFlag[4 * validFlagMaskLen_], locFloat[3 * alignedOneQueryNum_], 0.f,
        CMPMODE::LT, MASK_PLACEHOLDER, qryRpt_, {1, 1, 8, 8});
    Or<uint16_t, false>(validFlag[3 * validFlagMaskLen_].ReinterpretCast<uint16_t>(),
        validFlag[3 * validFlagMaskLen_].ReinterpretCast<uint16_t>(),
        validFlag[4 * validFlagMaskLen_].ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
    Or<uint16_t, false>(validFlag[3 * validFlagMaskLen_].ReinterpretCast<uint16_t>(),
        validFlag[3 * validFlagMaskLen_].ReinterpretCast<uint16_t>(),
        validFlag[validFlagMaskLen_].ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
    SetFlag<HardEvent::V_MTE2>(biEvt_);
}

template<bool aligned, bool forward, bool fastMode>
__aicore__ inline void MSDABaseKernel<aligned, forward, fastMode>::ComputeWeight(const LocalTensor<float>& locFloat,
    const LocalTensor<float>& shapes, const LocalTensor<float>& production, const LocalTensor<float>& weight,
    const LocalTensor<float>& attentionWeight)
{
    Sub<float, false>(locFloat, locFloat, locFloat[2 * alignedOneQueryNum_], MASK_PLACEHOLDER, 2 * qryRpt_,
        {1, 1, 1, 8, 8, 8}); // lw, lh

    Mul<float, false>(production[3 * alignedOneQueryNum_], locFloat, locFloat[alignedOneQueryNum_], MASK_PLACEHOLDER,
        qryRpt_, {1, 1, 1, 8, 8, 8}); // lw * lh
    Duplicate<float, false>(production, 1.f, MASK_PLACEHOLDER, 2 * qryRpt_, 1, 8);
    // hw, hh
    Sub<float, false>(
        locFloat[2 * alignedOneQueryNum_], production, locFloat, MASK_PLACEHOLDER, 2 * qryRpt_, {1, 1, 1, 8, 8, 8});
    // hw*hh
    Mul<float, false>(production, locFloat[2 * alignedOneQueryNum_], locFloat[3 * alignedOneQueryNum_],
        MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
    // lw*hh
    Mul<float, false>(production[alignedOneQueryNum_], locFloat, locFloat[3 * alignedOneQueryNum_], MASK_PLACEHOLDER,
        qryRpt_, {1, 1, 1, 8, 8, 8}); // lw * hh
    // hw*lh
    Mul<float, false>(production[2 * alignedOneQueryNum_], locFloat[alignedOneQueryNum_],
        locFloat[2 * alignedOneQueryNum_], MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8}); // hw * lh
    // lw*lh
    Mul<float, false>(production[3 * alignedOneQueryNum_], locFloat[alignedOneQueryNum_], locFloat, MASK_PLACEHOLDER,
        qryRpt_, {1, 1, 1, 8, 8, 8}); // lw * lh

    Mul<float, false>(weight, production, attentionWeight, MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(weight[alignedOneQueryNum_], production[alignedOneQueryNum_], attentionWeight, MASK_PLACEHOLDER,
        qryRpt_, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(weight[2 * alignedOneQueryNum_], production[2 * alignedOneQueryNum_], attentionWeight,
        MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(weight[3 * alignedOneQueryNum_], production[3 * alignedOneQueryNum_], attentionWeight,
        MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
    if constexpr (!forward) {
        Mul<float, false>(
            locFloat, locFloat, shapes[alignedOneQueryNum_], MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8}); // lw * h
        Mul<float, false>(locFloat[alignedOneQueryNum_], locFloat[alignedOneQueryNum_], shapes, MASK_PLACEHOLDER,
            qryRpt_, {1, 1, 1, 8, 8, 8}); // lh * w
        Mul<float, false>(locFloat[2 * alignedOneQueryNum_], locFloat[2 * alignedOneQueryNum_],
            shapes[alignedOneQueryNum_], MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8}); // hw * h
        Mul<float, false>(locFloat[3 * alignedOneQueryNum_], locFloat[3 * alignedOneQueryNum_], shapes,
            MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8}); // hh * w
        Mul<float, false>(locFloat, locFloat, attentionWeight, MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
        Mul<float, false>(locFloat[alignedOneQueryNum_], locFloat[alignedOneQueryNum_], attentionWeight,
            MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
        Mul<float, false>(locFloat[2 * alignedOneQueryNum_], locFloat[2 * alignedOneQueryNum_], attentionWeight,
            MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
        Mul<float, false>(locFloat[3 * alignedOneQueryNum_], locFloat[3 * alignedOneQueryNum_], attentionWeight,
            MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
    }
    SetFlag<HardEvent::V_MTE2>(copyEvt_);
}

template<bool aligned, bool fastMode>
class MultiScaleDeformableAttnKernel : MSDABaseKernel<aligned, true, fastMode> {
public:
    __aicore__ inline MultiScaleDeformableAttnKernel() = delete;

    __aicore__ inline MultiScaleDeformableAttnKernel(GM_ADDR value, GM_ADDR valueSpatialShapes,
        GM_ADDR valueLevelStartIndex, GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output,
        const MultiScaleDeformableAttnTilingData* tilingData, TPipe* pipe)
        : MSDABaseKernel<aligned, true, fastMode>(
              value, valueSpatialShapes, valueLevelStartIndex, samplingLocations, attentionWeights, tilingData, pipe)
    {
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
    }

    __aicore__ inline void Process();

private:
    GlobalTensor<float> outputGm_;

    __aicore__ inline void CopyOut(const LocalTensor<float>& output, uint32_t taskIdx)
    {
        WaitFlag<HardEvent::V_MTE3>(0);
        if constexpr (aligned) {
            DataCopy(outputGm_[taskIdx * this->outDims_], output, this->cpOutParams_);
        } else {
            DataCopyPad(outputGm_[taskIdx * this->outDims_], output, this->cpOutParams_);
        }
        SetFlag<HardEvent::MTE3_V>(0);
    }
    __aicore__ inline void ComputeBilinearInterpolation(const LocalTensor<uint64_t>& validFlag,
        const LocalTensor<int32_t>& shapeInt, const LocalTensor<int32_t>& location, const LocalTensor<int32_t>& loc,
        const LocalTensor<float>& shapeFloat, const LocalTensor<float>& production, const LocalTensor<float>& value,
        const LocalTensor<float>& locFloat, const LocalTensor<float>& weight, const LocalTensor<float>& attentionWeight,
        const LocalTensor<float>& cornerWeightBrc, const LocalTensor<float>& output);
};

template<bool aligned, bool fastMode>
class MultiScaleDeformableAttnGradKernel : MSDABaseKernel<aligned, false, fastMode> {
public:
    __aicore__ inline MultiScaleDeformableAttnGradKernel() = delete;

    __aicore__ inline MultiScaleDeformableAttnGradKernel(GM_ADDR value, GM_ADDR valueSpatialShapes,
        GM_ADDR valueLevelStartIndex, GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR gradOutput,
        GM_ADDR gradValue, GM_ADDR gradSamplingLocations, GM_ADDR gradAttentionWeights,
        const MultiScaleDeformableAttnTilingData* tilingData, TPipe* pipe)
        : MSDABaseKernel<aligned, false, fastMode>(
              value, valueSpatialShapes, valueLevelStartIndex, samplingLocations, attentionWeights, tilingData, pipe)
    {
        gradOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradOutput));
        gradValueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradValue));
        gradLocGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradSamplingLocations));
        gradAttentionWeightsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradAttentionWeights));

        cpGradRowDoubleParams_.srcStride =
            this->alignedCornerEmbedDims_ / B32_DATA_NUM_PER_BLOCK - DivCeil(this->embedDims_, B32_DATA_NUM_PER_BLOCK);
        if constexpr (aligned) {
            cpGradOneValParams_.blockLen = this->embedBlk_;
            cpGradRowDoubleParams_.blockLen = this->embedBlk_;
            cpGradRowDoubleParams_.dstStride = this->outBlk_ - this->embedBlk_;
        } else {
            cpGradOneValParams_.blockLen = this->embedDims_ * B32_BYTE_SIZE;
            cpGradRowDoubleParams_.blockLen = this->embedDims_ * B32_BYTE_SIZE;
            cpGradRowDoubleParams_.dstStride = (this->outDims_ - this->embedDims_) * B32_BYTE_SIZE;
        }

        if (fastMode) {
            cpGradSampleParams_.blockCount = 1;
            cpGradSampleParams_.blockLen = this->numHeads_ * this->oneHeadNum_ * B32_BYTE_SIZE;
            cpGradDoubleSampleParams_.blockCount = 1;
            cpGradDoubleSampleParams_.blockLen = 2 * this->numHeads_ * this->oneHeadNum_ * B32_BYTE_SIZE;
        } else {
            cpGradSampleParams_.blockCount = this->numHeads_;
            cpGradSampleParams_.blockLen = this->oneHeadNum_ * B32_BYTE_SIZE;
            cpGradSampleParams_.srcStride =
                this->alignedOneHeadNum_ / B32_DATA_NUM_PER_BLOCK - DivCeil(this->oneHeadNum_, B32_DATA_NUM_PER_BLOCK);
            cpGradDoubleSampleParams_.blockCount = this->numHeads_;
            cpGradDoubleSampleParams_.blockLen = 2 * this->oneHeadNum_ * B32_BYTE_SIZE;
            cpGradDoubleSampleParams_.srcStride = 2 * this->alignedOneHeadNum_ / B32_DATA_NUM_PER_BLOCK -
                                                  DivCeil(2 * this->oneHeadNum_, B32_DATA_NUM_PER_BLOCK);
        }
    }

    __aicore__ inline void Process();

private:
    GlobalTensor<float> gradOutGm_, gradValueGm_, gradAttentionWeightsGm_, gradLocGm_;
    DataCopyParams cpGradOneValParams_, cpGradRowDoubleParams_ {2, 0, 0, 0}, cpGradSampleParams_,
        cpGradDoubleSampleParams_;

    __aicore__ inline void PrepareGatherOffset(const LocalTensor<uint32_t>& gatherOffset)
    {
        for (uint32_t i = 0; i < this->alignedOneQueryNum_; ++i) {
            gatherOffset.SetValue(2 * i, (i + this->alignedOneQueryNum_) * 4);
            gatherOffset.SetValue(2 * i + 1, i * 4);
        }
    }

    __aicore__ inline void CopyInGradOut(const LocalTensor<float>& gradOut, uint32_t taskIdx)
    {
        WaitFlag<HardEvent::V_MTE2>(1);
        if constexpr (aligned) {
            DataCopy(gradOut, gradOutGm_[taskIdx * this->outDims_], this->cpOutParams_, {});
        } else {
            DataCopyPad(gradOut, gradOutGm_[taskIdx * this->outDims_], this->cpOutParams_, {});
        }
        SetFlag<HardEvent::MTE2_V>(1);
    }

    __aicore__ inline void GradMul(const LocalTensor<float>& dst, const LocalTensor<float>& gradOut, uint32_t outOffset)
    {
        for (uint32_t i = 0; i < 4; ++i) {
            uint32_t outerOffset = i * this->alignedCornerEmbedDims_;
            uint32_t offset = outOffset;
            if (fastMode) {
                for (uint32_t j = 0; j < this->numHeads_; ++j) {
                    uint32_t innerOffset = outerOffset + offset * this->oneHeadNum_;
                    Mul<float, false>(dst[innerOffset], dst[innerOffset], gradOut[offset], MASK_PLACEHOLDER,
                        this->oneHeadNum_,
                        {1, 1, 1, static_cast<uint8_t>(this->embedBlk_), static_cast<uint8_t>(this->embedBlk_), 0});
                    offset += this->alignedEmbedDims_;
                }
            } else {
                Mul<float, false>(dst[outerOffset], dst[outerOffset], gradOut[outOffset], MASK_PLACEHOLDER,
                    this->oneHeadNum_,
                    {1, 1, 1, static_cast<uint8_t>(this->embedBlk_), static_cast<uint8_t>(this->embedBlk_), 0});
            }
        }
    }

    __aicore__ inline void ComputeBilinearInterpolation(const LocalTensor<uint64_t>& validFlag,
        const LocalTensor<int32_t>& shapeInt, const LocalTensor<int32_t>& location, const LocalTensor<int32_t>& loc,
        const LocalTensor<float>& shapeFloat, const LocalTensor<float>& production, const LocalTensor<float>& value,
        const LocalTensor<float>& locFloat, const LocalTensor<float>& weight, const LocalTensor<float>& attentionWeight,
        const LocalTensor<float>& cornerWeightBrc, const LocalTensor<float>& gradOut);

    __aicore__ inline void ComputeGrad(const LocalTensor<float>& production, const LocalTensor<float>& locFloat,
        const LocalTensor<float>& weight, const LocalTensor<float>& attentionWeight,
        const LocalTensor<float>& gradLocation, const LocalTensor<float>& gradAttentionWeight,
        const LocalTensor<uint32_t>& gatherOffset, uint32_t taskIdx);
};
#endif // MSDA_H
