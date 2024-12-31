/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 */

#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t EMBED_DIMS = 64;
constexpr uint32_t POINTS = 8;

#define ADD_MSDA_CASE_ALIGNED(num_points)                                                                    \
    if (TILING_KEY_IS(num_points##1)) {                                                                      \
        MultiScaleDeformableAttnKernel<num_points, true> op(value, valueSpatialShapes, valueLevelStartIndex, \
            samplingLocations, attentionWeights, output, &tilingData, &pipe);                                \
        op.Process();                                                                                        \
        return;                                                                                              \
    }

#define ADD_MSDA_CASE_UNALIGNED(num_points)                                                                   \
    if (TILING_KEY_IS(num_points##0)) {                                                                       \
        MultiScaleDeformableAttnKernel<num_points, false> op(value, valueSpatialShapes, valueLevelStartIndex, \
            samplingLocations, attentionWeights, output, &tilingData, &pipe);                                 \
        op.Process();                                                                                         \
        return;                                                                                               \
    }

template<uint32_t num_points, bool aligned>
class MultiScaleDeformableAttnKernel {
public:
    __aicore__ inline MultiScaleDeformableAttnKernel() = delete;

    __aicore__ inline MultiScaleDeformableAttnKernel(GM_ADDR value, GM_ADDR valueSpatialShapes,
        GM_ADDR valueLevelStartIndex, GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output,
        const MultiScaleDeformableAttnTilingData* tilingData, TPipe* pipe)
        : pipe_(pipe), blkIdx_(GetBlockIdx())
    {
        InitTiling(tilingData);
        InitTask();
        InitGM(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations, attentionWeights, output);
        InitBuffer();
        SetVectorMask<float>(FULL_MASK, FULL_MASK);
        SetAtomicNone();
    }

    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTask()
    {
        uint32_t avgTasks = (batchSize_ * numQueries_) / coreNum_;
        uint32_t remainTasks = (batchSize_ * numQueries_) % coreNum_;
        startOffset_ = avgTasks * blkIdx_ + (blkIdx_ < remainTasks ? blkIdx_ : remainTasks);
        endOffset_ = startOffset_ + avgTasks + (blkIdx_ < remainTasks ? 1 : 0);
        batch_ = startOffset_ / numQueries_;
        query_ = startOffset_ % numQueries_;
        baseSrcOffset_ = batch_ * numKeys_ * numHeads_;
        baseDstOffset_ = startOffset_ * numHeads_ * embedDims_;
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
        pointLoops_ = tilingData->pointLoops;
        realLevels_ = tilingData->realLevels;

        oneQueryNum_ = numHeads_ * realLevels_ * numPoints_;

        alignedOneHeadNum_ = numLevels_ * POINTS;
        alignedOneQueryNum_ = AlignUp(numHeads_ * alignedOneHeadNum_, B32_DATA_NUM_PER_REPEAT);
        alignedCornerEmbedDims_ = alignedOneHeadNum_ * EMBED_DIMS;
        alignedHeadEmbedDims_ = numHeads_ * EMBED_DIMS;
        outDims_ = numHeads_ * embedDims_;

        queryBlk_ = alignedOneQueryNum_ / B32_DATA_NUM_PER_BLOCK;
        qryRpt_ = alignedOneQueryNum_ / B32_DATA_NUM_PER_REPEAT;
        cornerRpt_ = 4 * alignedOneHeadNum_;

        if constexpr (aligned) {
            embedBlk_ = embedDims_ / B32_DATA_NUM_PER_BLOCK;
        } else {
            embedBlk_ = embedDims_ * B32_BYTE_SIZE;
        }
        outBlk_ = numHeads_ * embedBlk_;
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
            cpDoubleSampleParams_.dstStride = num_points > 4 ? 0 : 1;
        }

        cpOneValParams_.blockLen = embedBlk_;
        cpRowDoubleParams_.blockLen = embedBlk_;
        cpRowDoubleParams_.srcStride = outBlk_ - embedBlk_;
        cpRowDoubleParams_.dstStride =
            alignedCornerEmbedDims_ / B32_DATA_NUM_PER_BLOCK - DivCeil(embedDims_, B32_DATA_NUM_PER_BLOCK);
        cpColDoubleParams_.blockLen = embedBlk_;
        cpColDoubleParams_.dstStride =
            2 * alignedCornerEmbedDims_ / B32_DATA_NUM_PER_BLOCK - DivCeil(embedDims_, B32_DATA_NUM_PER_BLOCK);
        cpOutParams_.blockCount = numHeads_;
        cpOutParams_.blockLen = embedBlk_;
        cpOutParams_.srcStride = 8 - DivCeil(embedDims_, B32_DATA_NUM_PER_BLOCK);
        gatherParams_.repeatTimes = qryRpt_ * 2;
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
        pipe_->InitBuffer(locationQue_, 4 * alignedOneQueryNum_ * B32_BYTE_SIZE); // x, y
        pipe_->InitBuffer(attentionWeightsQue_, alignedOneQueryNum_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(valueQue_, BUFFER_NUM * 4 * alignedCornerEmbedDims_ * B32_BYTE_SIZE); // 2 for double buffer
        pipe_->InitBuffer(outputQue_, BUFFER_NUM * alignedHeadEmbedDims_ * B32_BYTE_SIZE);

        pipe_->InitBuffer(shapeBrcBuf_, 2 * alignedOneQueryNum_ * B32_BYTE_SIZE); // w, h
        pipe_->InitBuffer(locIntBuf_, 4 * alignedOneQueryNum_ * B32_BYTE_SIZE);   // x0, y0, x1, y1
        pipe_->InitBuffer(locFloatBuf_, 4 * alignedOneQueryNum_ * B32_BYTE_SIZE); // lw, lh
        pipe_->InitBuffer(weightBuf_, 4 * alignedOneQueryNum_ * B32_BYTE_SIZE);   // w1-w4
        // NOTE: cornerWeightBrcBuf must be at the tail of ub
        pipe_->InitBuffer(cornerWeightBrcBuf_, 4 * (alignedCornerEmbedDims_ + alignedOneHeadNum_ * 8) * B32_BYTE_SIZE);
    }

    __aicore__ inline void PrepareShape(
        const LocalTensor<int32_t>& shapes, const LocalTensor<int32_t>& offset, LocalTensor<float>& shapeBrc);

    __aicore__ inline void CopyInSample(
        const LocalTensor<float>& location, const LocalTensor<float>& attentionWeight, uint32_t taskIdx, uint32_t pl);

    __aicore__ inline void ComputeLocation(const LocalTensor<float>& location, const LocalTensor<float>& shapes,
        const LocalTensor<int32_t>& locInt, const LocalTensor<float>& locFloat);

    __aicore__ inline void ComputeWeight(const LocalTensor<float>& locFloat, const LocalTensor<float>& weight,
        const LocalTensor<float>& attentionWeight);

    __aicore__ inline void ComputeBilinearInterpolation(uint32_t pl, uint8_t outerPing,
        const LocalTensor<int32_t>& shapes, const LocalTensor<int32_t>& offset, const LocalTensor<int32_t>& locX,
        const LocalTensor<int32_t>& locY, const LocalTensor<float>& value, const LocalTensor<float>& weight,
        const LocalTensor<float>& cornerWeightBrc, const LocalTensor<float>& output);

    __aicore__ inline void DataCopyValue(
        const LocalTensor<float>& dst, const GlobalTensor<float>& src, const DataCopyParams& cpParams)
    {
        if constexpr (aligned) {
            DataCopy(dst, src, cpParams);
        } else {
            DataCopyPad(dst, src, cpParams, {});
        }
    }

private:
    TPipe* pipe_;
    GlobalTensor<float> valueGm_, locationGm_, attentionWeightsGm_, outputGm_;
    GlobalTensor<int32_t> valueSpatialShapesGm_, valueLevelStartIndexGm_;

    TBuf<TPosition::VECCALC> locationQue_, attentionWeightsQue_, shapeQue_, offsetQue_, valueQue_;
    TBuf<TPosition::VECCALC> outputQue_;

    TBuf<TPosition::VECCALC> locIntBuf_, locFloatBuf_, shapeBrcBuf_, weightBuf_, cornerWeightBrcBuf_;

    int32_t blkIdx_;

    uint64_t batchSize_, numKeys_, numHeads_, embedDims_, outDims_, numLevels_, numQueries_, numPoints_, realLevels_;
    uint32_t coreNum_, pointLoops_;
    uint32_t startOffset_, endOffset_;
    uint32_t alignedOneHeadNum_, alignedOneQueryNum_, alignedCornerEmbedDims_, alignedHeadEmbedDims_;
    uint32_t oneQueryNum_;
    uint16_t queryBlk_, embedBlk_, outBlk_;
    uint16_t qryRpt_, cornerRpt_;
    uint32_t query_, batch_;
    TEventID copyEvt_ {3};

    uint64_t baseSrcOffset_, baseDstOffset_, srcOffset_;

    DataCopyParams cpOneValParams_, cpRowDoubleParams_ {2, 0, 0, 0}, cpColDoubleParams_ {2, 0, 0, 0}, cpSampleParams_,
        cpDoubleSampleParams_, cpOutParams_;
    GatherMaskParams gatherParams_;
};

template<uint32_t num_points, bool aligned>
__aicore__ inline void MultiScaleDeformableAttnKernel<num_points, aligned>::PrepareShape(
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

template<uint32_t num_points, bool aligned>
__aicore__ inline void MultiScaleDeformableAttnKernel<num_points, aligned>::CopyInSample(
    const LocalTensor<float>& location, const LocalTensor<float>& attentionWeight, uint32_t taskIdx, uint32_t pl)
{
    uint64_t sampleOffset = taskIdx * oneQueryNum_;
    WaitFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(1);
    if (num_points == 8 && pointLoops_ == 1) {
        DataCopy(location, locationGm_[sampleOffset * 2], cpDoubleSampleParams_);
        DataCopy(attentionWeight, attentionWeightsGm_[sampleOffset], cpSampleParams_);
    } else {
        DataCopyPad(location, locationGm_[sampleOffset * 2 + pl * num_points * 2], cpDoubleSampleParams_, {});
        DataCopyPad(attentionWeight, attentionWeightsGm_[sampleOffset + pl * num_points], cpSampleParams_, {});
    }

    SetFlag<HardEvent::MTE2_V>(copyEvt_);
}

template<uint32_t num_points, bool aligned>
__aicore__ inline void MultiScaleDeformableAttnKernel<num_points, aligned>::ComputeLocation(
    const LocalTensor<float>& location, const LocalTensor<float>& shapes, const LocalTensor<int32_t>& locInt,
    const LocalTensor<float>& locFloat)
{
    uint64_t cnt;
    WaitFlag<HardEvent::MTE2_V>(copyEvt_);

    GatherMask(location, location[2 * alignedOneQueryNum_], 1, false, MASK_PLACEHOLDER, gatherParams_, cnt);
    GatherMask(location[alignedOneQueryNum_], location[2 * alignedOneQueryNum_], 2, false, MASK_PLACEHOLDER,
        gatherParams_, cnt);
    SetVectorMask<float>(FULL_MASK, FULL_MASK);

    Mul<float, false>(location, location, shapes, MASK_PLACEHOLDER, 2 * qryRpt_, {1, 1, 1, 8, 8, 8});
    Adds<float, false>(locFloat, location, 0.5f, MASK_PLACEHOLDER, 2 * qryRpt_, {1, 1, 8, 8});
    Cast<int32_t, float, false>(locInt, locFloat, RoundMode::CAST_FLOOR, MASK_PLACEHOLDER, 2 * qryRpt_, {1, 1, 8, 8});
    Cast<float, int32_t, false>(
        locFloat[2 * alignedOneQueryNum_], locInt, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 2 * qryRpt_, {1, 1, 8, 8});
    Adds<int32_t, false>(locInt, locInt, -1, MASK_PLACEHOLDER, 2 * qryRpt_, {1, 1, 8, 8});
    SetFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE2>(1);
}

template<uint32_t num_points, bool aligned>
__aicore__ inline void MultiScaleDeformableAttnKernel<num_points, aligned>::ComputeWeight(
    const LocalTensor<float>& locFloat, const LocalTensor<float>& weight, const LocalTensor<float>& attentionWeight)
{
    Sub<float, false>(locFloat, locFloat, locFloat[2 * alignedOneQueryNum_], MASK_PLACEHOLDER, 2 * qryRpt_,
        {1, 1, 1, 8, 8, 8}); // lw, lh
    Mul<float, false>(weight[3 * alignedOneQueryNum_], locFloat, locFloat[alignedOneQueryNum_], MASK_PLACEHOLDER,
        qryRpt_, {1, 1, 1, 8, 8, 8}); // lw * lh
    Duplicate<float, false>(weight, 1.f, MASK_PLACEHOLDER, qryRpt_, 1, 8);
    Sub<float, false>(weight, weight, locFloat, MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
    Sub<float, false>(weight, weight, locFloat[alignedOneQueryNum_], MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
    Add<float, false>(weight, weight, weight[3 * alignedOneQueryNum_], MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
    Sub<float, false>(weight[alignedOneQueryNum_], locFloat, weight[3 * alignedOneQueryNum_], MASK_PLACEHOLDER, qryRpt_,
        {1, 1, 1, 8, 8, 8}); // lw - lw * lh, lw*hh
    Sub<float, false>(weight[2 * alignedOneQueryNum_], locFloat[alignedOneQueryNum_], weight[3 * alignedOneQueryNum_],
        MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8}); // hw*lh

    Mul<float, false>(weight, weight, attentionWeight, MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(weight[alignedOneQueryNum_], weight[alignedOneQueryNum_], attentionWeight, MASK_PLACEHOLDER,
        qryRpt_, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(weight[2 * alignedOneQueryNum_], weight[2 * alignedOneQueryNum_], attentionWeight,
        MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
    Mul<float, false>(weight[3 * alignedOneQueryNum_], weight[3 * alignedOneQueryNum_], attentionWeight,
        MASK_PLACEHOLDER, qryRpt_, {1, 1, 1, 8, 8, 8});
}

template<uint32_t num_points, bool aligned>
__aicore__ inline void MultiScaleDeformableAttnKernel<num_points, aligned>::ComputeBilinearInterpolation(uint32_t pl,
    uint8_t outerPing, const LocalTensor<int32_t>& shapes, const LocalTensor<int32_t>& offset,
    const LocalTensor<int32_t>& locX, const LocalTensor<int32_t>& locY, const LocalTensor<float>& value,
    const LocalTensor<float>& weight, const LocalTensor<float>& cornerWeightBrc, const LocalTensor<float>& output)
{
    uint8_t innerPing = 0;

    for (uint32_t head = 0; head < numHeads_; ++head) {
        uint32_t valueOffset = (baseSrcOffset_ + head) * embedDims_;
        uint32_t outOffset = head * EMBED_DIMS;
        uint32_t baseIdx = head * alignedOneHeadNum_;
        LocalTensor<float> valuePing = value[innerPing * 4 * alignedCornerEmbedDims_];
        for (uint32_t level = 0; level < numLevels_; ++level) {
            int32_t h = shapes.GetValue(level * 2);
            int32_t w = shapes.GetValue(level * 2 + 1);
            srcOffset_ = valueOffset + offset.GetValue(level) * outDims_;
            uint32_t idx = baseIdx + level * POINTS;
            LocalTensor<float> valueLevel = valuePing[level * POINTS * EMBED_DIMS];
            for (uint32_t point = 0; point < num_points; ++point) {
                int32_t x = locX.GetValue(idx + point);
                int32_t y = locY.GetValue(idx + point);
                if (unlikely(level == 0 && point == 0)) {
                    WaitFlag<HardEvent::V_MTE2>(innerPing);
                }
                if (0 <= y && y < h - 1) {
                    if (0 <= x && x < w - 1) {
                        DataCopyValue(valueLevel[point * EMBED_DIMS], valueGm_[srcOffset_ + (y * w + x) * outDims_],
                            cpRowDoubleParams_);
                        DataCopyValue(valueLevel[point * EMBED_DIMS + 2 * alignedCornerEmbedDims_],
                            valueGm_[srcOffset_ + ((y + 1) * w + x) * outDims_], cpRowDoubleParams_);
                    } else if (x == w - 1) {
                        cpColDoubleParams_.srcStride = w * outBlk_ - embedBlk_;
                        DataCopyValue(valueLevel[point * EMBED_DIMS], valueGm_[srcOffset_ + (y * w + x) * outDims_],
                            cpColDoubleParams_);
                    } else if (x == -1) {
                        cpColDoubleParams_.srcStride = w * outBlk_ - embedBlk_;
                        DataCopyValue(valueLevel[point * EMBED_DIMS + alignedCornerEmbedDims_],
                            valueGm_[srcOffset_ + y * w * outDims_], cpColDoubleParams_);
                    }
                } else if (y == h - 1) {
                    if (0 <= x && x < w - 1) {
                        DataCopyValue(valueLevel[point * EMBED_DIMS], valueGm_[srcOffset_ + (y * w + x) * outDims_],
                            cpRowDoubleParams_);
                    } else if (x == w - 1) {
                        DataCopyValue(valueLevel[point * EMBED_DIMS], valueGm_[srcOffset_ + (y * w + x) * outDims_],
                            cpOneValParams_);
                    } else if (x == -1) {
                        DataCopyValue(valueLevel[point * EMBED_DIMS + alignedCornerEmbedDims_],
                            valueGm_[srcOffset_ + y * w * outDims_], cpOneValParams_);
                    }
                } else if (y == -1) {
                    if (0 <= x && x < w - 1) {
                        DataCopyValue(valueLevel[point * EMBED_DIMS + 2 * alignedCornerEmbedDims_],
                            valueGm_[srcOffset_ + x * outDims_], cpRowDoubleParams_);
                    } else if (x == w - 1) {
                        DataCopyValue(valueLevel[point * EMBED_DIMS + 2 * alignedCornerEmbedDims_],
                            valueGm_[srcOffset_ + x * outDims_], cpOneValParams_);
                    } else if (x == -1) {
                        DataCopyValue(valueLevel[point * EMBED_DIMS + 3 * alignedCornerEmbedDims_],
                            valueGm_[srcOffset_], cpOneValParams_);
                    }
                }
            }
        }
        SetFlag<HardEvent::MTE2_V>(innerPing);

        for (uint32_t i = 0; i < 4; ++i) {
            Brcb(cornerWeightBrc[4 * alignedCornerEmbedDims_ + i * alignedOneHeadNum_ * 8],
                weight[baseIdx + i * alignedOneQueryNum_], numLevels_, {1, 8});
        }
        Brcb(cornerWeightBrc, cornerWeightBrc[4 * alignedCornerEmbedDims_], cornerRpt_, {1, 8});
        WaitFlag<HardEvent::MTE2_V>(innerPing);
        Mul<float, false>(
            cornerWeightBrc, valuePing, cornerWeightBrc, MASK_PLACEHOLDER, cornerRpt_, {1, 1, 1, 8, 8, 8});
        Duplicate<float, false>(valuePing, 0.f, MASK_PLACEHOLDER, cornerRpt_, 1, 8);
        SetFlag<HardEvent::V_MTE2>(innerPing);
        innerPing = 1 - innerPing;
        if (unlikely(head == 0 && pl == 0)) {
            WaitFlag<HardEvent::MTE3_V>(outerPing);
            Duplicate<float, false>(output, 0.f, MASK_PLACEHOLDER, numHeads_, 1, 8);
        }
        Add<float, false>(
            output[outOffset], cornerWeightBrc, output[outOffset], MASK_PLACEHOLDER, cornerRpt_, {1, 1, 1, 0, 8, 0});
    }
}


template<uint32_t num_points, bool aligned>
__aicore__ inline void MultiScaleDeformableAttnKernel<num_points, aligned>::Process()
{
    LocalTensor<float> location = locationQue_.Get<float>();
    LocalTensor<float> attentionWeight = attentionWeightsQue_.Get<float>();
    LocalTensor<int32_t> shapes = shapeQue_.Get<int32_t>();
    LocalTensor<int32_t> offset = offsetQue_.Get<int32_t>();
    LocalTensor<float> value = valueQue_.Get<float>();
    LocalTensor<float> cornerWeightBrc = cornerWeightBrcBuf_.Get<float>();
    LocalTensor<float> output = outputQue_.Get<float>();

    LocalTensor<float> shapeBrc = shapeBrcBuf_.Get<float>();
    LocalTensor<int32_t> locInt = locIntBuf_.Get<int32_t>();
    LocalTensor<float> locFloat = locFloatBuf_.Get<float>();
    LocalTensor<float> weight = weightBuf_.Get<float>();

    PrepareShape(shapes, offset, shapeBrc);
    Duplicate<float, false>(value, 0.f, MASK_PLACEHOLDER, cornerRpt_, 1, 8);
    Duplicate<float, false>(value[4 * alignedCornerEmbedDims_], 0.f, MASK_PLACEHOLDER, cornerRpt_, 1, 8);
    SetFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE2>(1);
    SetFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::MTE3_V>(1);

    uint8_t outerPing = 0;
    for (uint32_t taskIdx = startOffset_; taskIdx < endOffset_; ++taskIdx) {
        for (uint32_t pl = 0; pl < pointLoops_; ++pl) {
            CopyInSample(location[2 * alignedOneQueryNum_], attentionWeight, taskIdx, pl);
            ComputeLocation(location, shapeBrc, locInt, locFloat);
            ComputeWeight(locFloat, weight, attentionWeight);
            ComputeBilinearInterpolation(pl, outerPing, shapes, offset, locInt, locInt[alignedOneQueryNum_], value,
                weight, cornerWeightBrc, output[outerPing * alignedHeadEmbedDims_]);
        }
        SetFlag<HardEvent::V_MTE3>(outerPing);
        WaitFlag<HardEvent::V_MTE3>(outerPing);
        if constexpr (aligned) {
            DataCopy(outputGm_[baseDstOffset_], output[outerPing * alignedHeadEmbedDims_], cpOutParams_);
        } else {
            DataCopyPad(outputGm_[baseDstOffset_], output[outerPing * alignedHeadEmbedDims_], cpOutParams_);
        }
        SetFlag<HardEvent::MTE3_V>(outerPing);

        outerPing = 1 - outerPing;
        baseDstOffset_ += outDims_;
        ++query_;
        if (unlikely(query_ == numQueries_)) {
            query_ = 0;
            batch_++;
            baseSrcOffset_ += numKeys_ * numHeads_;
        }
    }
    WaitFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(1);
    WaitFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(1);
}

extern "C" __global__ __aicore__ void multi_scale_deformable_attn(GM_ADDR value, GM_ADDR valueSpatialShapes,
    GM_ADDR valueLevelStartIndex, GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    ADD_MSDA_CASE_ALIGNED(1)
    ADD_MSDA_CASE_ALIGNED(2)
    ADD_MSDA_CASE_ALIGNED(3)
    ADD_MSDA_CASE_ALIGNED(4)
    ADD_MSDA_CASE_ALIGNED(5)
    ADD_MSDA_CASE_ALIGNED(6)
    ADD_MSDA_CASE_ALIGNED(7)
    ADD_MSDA_CASE_ALIGNED(8)
    ADD_MSDA_CASE_UNALIGNED(1)
    ADD_MSDA_CASE_UNALIGNED(2)
    ADD_MSDA_CASE_UNALIGNED(3)
    ADD_MSDA_CASE_UNALIGNED(4)
    ADD_MSDA_CASE_UNALIGNED(5)
    ADD_MSDA_CASE_UNALIGNED(6)
    ADD_MSDA_CASE_UNALIGNED(7)
    ADD_MSDA_CASE_UNALIGNED(8)
}
