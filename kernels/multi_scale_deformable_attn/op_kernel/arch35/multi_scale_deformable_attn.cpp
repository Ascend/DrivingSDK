/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 */

#include "kernel_operator.h"
#include "msda.h"

using namespace AscendC;
using namespace MicroAPI;

constexpr uint32_t taskOffset_ = 2048;
constexpr uint16_t taskRpt_ = taskOffset_ / B32_DATA_NUM_PER_REPEAT;
constexpr uint32_t threadNum_ = 1024;


__simt_callee__ __aicore__ __attribute__ ((always_inline)) inline int32_t GetValidPoint(
    uint32_t oneHeadNum_, uint32_t headIdx, uint32_t& point, __ubuf__ int32_t* validMask)
{
    for (; point < oneHeadNum_; ++point) {
        uint32_t pointIdx = headIdx * oneHeadNum_ + point;
        int32_t mask = validMask[pointIdx];
        if (mask > 0) {
            return mask;
        }
    }
    return 0;
}


__simt_vf__ __aicore__ LAUNCH_BOUND(threadNum_) inline void MSDASimtCompute(
    __gm__ float* valueGm_, __gm__ float* outputGm_, __ubuf__ float2* locationFloat,
    __ubuf__ int32_t* validMask, __ubuf__ int2* locationInt, __ubuf__ float* attnWeight,
    uint32_t count, uint32_t baseOffset, uint32_t oneHeadNum_, uint32_t outDims_)
{
    uint32_t channelIdx = threadIdx.x;
    for (uint32_t headIdx = threadIdx.y; headIdx < count; headIdx += blockDim.y) {
        float value = 0;
        for (uint32_t point = 0; point < oneHeadNum_; ++point) {
            int32_t mask = GetValidPoint(oneHeadNum_, headIdx, point, validMask);
            if (mask == 0) {
                break;
            }

            uint32_t pointIdx = headIdx * oneHeadNum_ + point;
            int2 gmOffset = locationInt[pointIdx];
            gmOffset.x = gmOffset.x + channelIdx;
            gmOffset.y = gmOffset.y + channelIdx;

            float v1 = (mask & 1) ? valueGm_[gmOffset.x] : 0;
            float v2 = (mask & 2) ? valueGm_[gmOffset.x + outDims_] : 0;
            float v3 = (mask & 4) ? valueGm_[gmOffset.y] : 0;
            float v4 = (mask & 8) ? valueGm_[gmOffset.y + outDims_] : 0;

            float2 location = locationFloat[pointIdx];
            float lh = location.y - Simt::Floor(location.y);
            float lw = location.x - Simt::Floor(location.x);
            float hh = 1 - lh;
            float hw = 1 - lw;

            float w1 = hh * hw;
            float w2 = hh * lw;
            float w3 = lh * hw;
            float w4 = lh * lw;

            float val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
            float w = attnWeight[pointIdx];
            value = value + w * val;
        }
        uint32_t idx = headIdx * blockDim.x + channelIdx;
        outputGm_[baseOffset + idx] = value;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(threadNum_) inline void MSDASimtComputeDoubleEmbed(
    __gm__ float2* valueGm_, __gm__ float2* outputGm_, __ubuf__ float2* locationFloat,
    __ubuf__ int32_t* validMask, __ubuf__ int2* locationInt, __ubuf__ float* attnWeight,
    uint32_t count, uint32_t baseOffset, uint32_t oneHeadNum_, uint32_t outDims_)
{
    uint32_t channelIdx = threadIdx.x * 2;
    const float2 zero = {0, 0};
    for (uint32_t headIdx = threadIdx.y; headIdx < count; headIdx += blockDim.y) {
        float2 value = {0, 0};
        for (uint32_t point = 0; point < oneHeadNum_; ++point) {
            int32_t mask = GetValidPoint(oneHeadNum_, headIdx, point, validMask);
            if (mask == 0) {
                break;
            }

            uint32_t pointIdx = headIdx * oneHeadNum_ + point;
            int2 gmOffset = locationInt[pointIdx];
            gmOffset.x = gmOffset.x + channelIdx;
            gmOffset.y = gmOffset.y + channelIdx;

            float2 v1 = (mask & 1) ? valueGm_[gmOffset.x >> 1] : zero;
            float2 v2 = (mask & 2) ? valueGm_[(gmOffset.x + outDims_) >> 1] : zero;
            float2 v3 = (mask & 4) ? valueGm_[gmOffset.y >> 1] : zero;
            float2 v4 = (mask & 8) ? valueGm_[(gmOffset.y + outDims_) >> 1] : zero;

            float2 location = locationFloat[pointIdx];
            float lh = location.y - Simt::Floor(location.y);
            float lw = location.x - Simt::Floor(location.x);
            float hh = 1 - lh;
            float hw = 1 - lw;

            float w1 = hh * hw;
            float w2 = hh * lw;
            float w3 = lh * hw;
            float w4 = lh * lw;

            float val1 = w1 * v1.x + w2 * v2.x + w3 * v3.x + w4 * v4.x;
            float val2 = w1 * v1.y + w2 * v2.y + w3 * v3.y + w4 * v4.y;

            float w = attnWeight[pointIdx];
            value.x =  value.x + w * val1;
            value.y =  value.y + w * val2;
        }
        uint32_t idx = headIdx * blockDim.x * 2 + channelIdx;
        outputGm_[(baseOffset + idx) >> 1] = value;
    }
}


class MultiScaleDeformableAttnKernel {
public:
    __aicore__ inline MultiScaleDeformableAttnKernel() = delete;

    __aicore__ inline MultiScaleDeformableAttnKernel(GM_ADDR value, GM_ADDR valueSpatialShapes, GM_ADDR valueLevelStartIndex,
        GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output, const MultiScaleDeformableAttnTilingData* tilingData,
        TPipe* pipe)
        : pipe_(pipe), blkIdx_(GetBlockIdx())
    {
        InitTiling(tilingData);
        InitGM(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations, attentionWeights, output);
        InitBuffer();
        ResetMask();
        SetAtomicNone();
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> locationFloat = locationQue_.template Get<float>();
        LocalTensor<int32_t> locationInt = gmOffsetBuf_.template Get<int32_t>();
        LocalTensor<float> attentionWeight = attentionWeightsQue_.template Get<float>();
        LocalTensor<int32_t> shapes = shapeQue_.template Get<int32_t>();
        LocalTensor<int32_t> offset = offsetQue_.template Get<int32_t>();
        LocalTensor<float> shapeFloat = shapeFloatBuf_.template Get<float>();
        LocalTensor<int32_t> shapeInt = shapeFloatBuf_.template Get<int32_t>();
        LocalTensor<int32_t> offsetInt = offsetIntBuf_.template Get<int32_t>();
        LocalTensor<int32_t> validMask = validMaskBuf_.template Get<int32_t>();

        PrepareShape(shapes, shapeInt, shapeFloat, offset, offsetInt);

        for (uint32_t taskIdx = blkIdx_ * compTaskNum_; taskIdx < batchSize_ * numQueries_; taskIdx += compTaskNum_ * coreNum_) {
            uint32_t baseNum = (taskIdx / numQueries_ + 1) * numQueries_ - taskIdx;
            uint32_t taskNum = min(compTaskNum_, batchSize_ * numQueries_ - taskIdx);
            uint32_t baseSrcOffset = taskIdx / numQueries_ * numKeys_ * numHeads_;
            uint32_t nextSrcOffset = baseSrcOffset + numKeys_ * numHeads_;
            CopyInSample(locationFloat[2 * alignedOneTaskNum_], attentionWeight, taskIdx, taskNum);
            PipeBarrier<PIPE_ALL>();

            ComputeGmOffsetVF<float, int32_t>(taskRpt_, numHeads_, embedDims_, baseSrcOffset, nextSrcOffset, baseNum * oneQueryNum_,
                locationFloat, shapeFloat, offsetInt, locationInt, validMask);
            PipeBarrier<PIPE_ALL>();

            CallMSDASimtFunc(taskIdx, taskNum, locationFloat, shapeFloat, locationInt, attentionWeight, validMask);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void CallMSDASimtFunc(uint32_t taskIdx, uint32_t taskNum, const LocalTensor<float>& locationFloat,
        const LocalTensor<float>& shapeFloat, const LocalTensor<int32_t>& locationInt, const LocalTensor<float>& attentionWeight,
        const LocalTensor<int32_t>& validMask)
    {
        bool doubleEmbedFlag = (embedDims_ % 2) == 0;
        if (doubleEmbedFlag) {
            uint32_t embedDimThreads = embedDims_ / 2;
            uint32_t headThreads = threadNum_ / embedDimThreads;
            Simt::VF_CALL<MSDASimtComputeDoubleEmbed>(Simt::Dim3(embedDimThreads, headThreads),
                (__gm__ float2*)valueGm_.GetPhyAddr(), (__gm__ float2*)outputGm_.GetPhyAddr(),
                (__ubuf__ float2*)locationFloat.GetPhyAddr(), (__ubuf__ int32_t*)validMask.GetPhyAddr(),
                (__ubuf__ int2*)locationInt.GetPhyAddr(), (__ubuf__ float*)attentionWeight.GetPhyAddr(),
                taskNum * numHeads_, taskIdx * outDims_, oneHeadNum_, outDims_);
        } else {
            uint32_t embedDimThreads = embedDims_;
            uint32_t headThreads = threadNum_ / embedDimThreads;
            Simt::VF_CALL<MSDASimtCompute>(Simt::Dim3(embedDimThreads, headThreads),
                (__gm__ float*)valueGm_.GetPhyAddr(), (__gm__ float*)outputGm_.GetPhyAddr(),
                (__ubuf__ float2*)locationFloat.GetPhyAddr(), (__ubuf__ int32_t*)validMask.GetPhyAddr(),
                (__ubuf__ int2*)locationInt.GetPhyAddr(), (__ubuf__ float*)attentionWeight.GetPhyAddr(),
                taskNum * numHeads_, taskIdx * outDims_, oneHeadNum_, outDims_);
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
        pipe_->InitBuffer(shapeFloatBuf_, 2 * alignedOneTaskNum_ * B32_BYTE_SIZE); // w, h
        pipe_->InitBuffer(offsetIntBuf_, alignedOneTaskNum_ * B32_BYTE_SIZE);      // offsetInt
        pipe_->InitBuffer(locationQue_, 4 * alignedOneTaskNum_ * B32_BYTE_SIZE);   // x, y
        pipe_->InitBuffer(gmOffsetBuf_, 2 * alignedOneTaskNum_ * B32_BYTE_SIZE);   // x, y
        pipe_->InitBuffer(attentionWeightsQue_, alignedOneTaskNum_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(validMaskBuf_, alignedOneTaskNum_ * B32_BYTE_SIZE);
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

protected:
    TPipe* pipe_;
    GlobalTensor<float> valueGm_, locationGm_, attentionWeightsGm_, outputGm_;
    GlobalTensor<int32_t> valueSpatialShapesGm_, valueLevelStartIndexGm_;

    TBuf<TPosition::VECCALC> locationQue_, attentionWeightsQue_, shapeQue_, offsetQue_;
    TBuf<TPosition::VECCALC> shapeFloatBuf_, offsetIntBuf_, gmOffsetBuf_, validMaskBuf_;

    int32_t blkIdx_;
    // const values
    uint32_t coreNum_, compTaskNum_;
    uint32_t batchSize_, numKeys_, numHeads_, embedDims_, outDims_, numLevels_, numQueries_, numPoints_, realLevels_;
    uint32_t alignedOneTaskNum_;
    uint32_t oneHeadNum_, oneQueryNum_;
};


extern "C" __global__ __aicore__ void multi_scale_deformable_attn(GM_ADDR value, GM_ADDR valueSpatialShapes,
    GM_ADDR valueLevelStartIndex, GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    MultiScaleDeformableAttnKernel op(value, valueSpatialShapes, valueLevelStartIndex, samplingLocations, attentionWeights, output, &tilingData, &pipe);
    op.Process();
}