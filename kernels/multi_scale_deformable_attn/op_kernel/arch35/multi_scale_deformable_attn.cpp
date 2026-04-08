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

constexpr uint32_t taskOffset_ = 2048;
constexpr uint16_t taskRpt_ = taskOffset_ / B32_DATA_NUM_PER_REPEAT;


template<typename T, typename U>
__aicore__ __attribute__ ((always_inline)) inline bool GetValidPoint(
    __ubuf__ T* locationFloat, __ubuf__ T* shapeFloat, U oneHeadNum_,
    U headIdx, U& point, T& x, T& y, T& height, T& width)
{
    for (; point < oneHeadNum_; ++point) {
        U pointIdx = headIdx * oneHeadNum_ + point;
        x = locationFloat[pointIdx];
        y = locationFloat[pointIdx + taskOffset_];
        width = shapeFloat[pointIdx];
        height = shapeFloat[pointIdx + taskOffset_];

        if ((x > -1 && y > -1 && x < width && y < height)) {
            return true;
        }
    }
    return false;
}


template<typename T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void MSDASimtCompute(
    __gm__ T* valueGm_, __gm__ T* outputGm_,
    __ubuf__ T* locationFloat, __ubuf__ T* shapeFloat,
    __ubuf__ U* locationInt, __ubuf__ T* attnWeight,
    U count, U baseOffset, U oneHeadNum_, U embedDims_, U outDims_)
{
    for (U idx = Simt::GetThreadIdx(); idx < count; idx += AscendC::Simt::GetThreadNum()) {
        U channelIdx = idx % embedDims_;
        U headIdx = idx / embedDims_;
        T value = 0;
        for (U point = 0; point < oneHeadNum_; ++point) {
            T x, y, height, width;
            if (!GetValidPoint(locationFloat, shapeFloat, oneHeadNum_, headIdx, point, x, y, height, width)) {
                continue;
            }

            U pointIdx = headIdx * oneHeadNum_ + point;
            U gmOffset1 = locationInt[pointIdx] + channelIdx;
            U gmOffset2 = gmOffset1 + outDims_;
            U gmOffset3 = locationInt[pointIdx + taskOffset_] + channelIdx;
            U gmOffset4 = gmOffset3 + outDims_;

            T v1 = (y >= 0 && x >= 0) ? valueGm_[gmOffset1] : 0;
            T v2 = (y >= 0 && x < width - 1) ? valueGm_[gmOffset2] : 0;
            T v3 = (y < height - 1 && x >= 0) ? valueGm_[gmOffset3] : 0;
            T v4 = (y < height - 1 && x < width - 1) ? valueGm_[gmOffset4] : 0;

            T lh = y - Simt::Floor(y);
            T lw = x - Simt::Floor(x);
            T hh = 1 - lh;
            T hw = 1 - lw;

            T w1 = hh * hw;
            T w2 = hh * lw;
            T w3 = lh * hw;
            T w4 = lh * lw;

            T val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
            T w = attnWeight[pointIdx];
            value = value + w * val;
        }
        outputGm_[baseOffset + idx] = value;
    }
}


template<typename T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void MSDASimtComputeDoubleEmbed(
    __gm__ T* valueGm_, __gm__ T* outputGm_,
    __ubuf__ T* locationFloat, __ubuf__ T* shapeFloat,
    __ubuf__ U* locationInt, __ubuf__ T* attnWeight,
    U count, U baseOffset, U oneHeadNum_, U embedDims_, U outDims_)
{
    for (U idx = Simt::GetThreadIdx() * 2; idx < count; idx += AscendC::Simt::GetThreadNum() * 2) {
        U channelIdx = idx % embedDims_;
        U headIdx = idx / embedDims_;

        T value1 = 0;
        T value2 = 0;
        for (U point = 0; point < oneHeadNum_; ++point) {
            T x, y, height, width;
            if (!GetValidPoint(locationFloat, shapeFloat, oneHeadNum_, headIdx, point, x, y, height, width)) {
                continue;
            }

            U pointIdx = headIdx * oneHeadNum_ + point;
            U gmOffset1 = locationInt[pointIdx] + channelIdx;
            U gmOffset2 = gmOffset1 + outDims_;
            U gmOffset3 = locationInt[pointIdx + taskOffset_] + channelIdx;
            U gmOffset4 = gmOffset3 + outDims_;

            T v1 = (y >= 0 && x >= 0) ? valueGm_[gmOffset1] : 0;
            T v2 = (y >= 0 && x < width - 1) ? valueGm_[gmOffset2] : 0;
            T v3 = (y < height - 1 && x >= 0) ? valueGm_[gmOffset3] : 0;
            T v4 = (y < height - 1 && x < width - 1) ? valueGm_[gmOffset4] : 0;

            T v5 = (y >= 0 && x >= 0) ? valueGm_[gmOffset1 + 1] : 0;
            T v6 = (y >= 0 && x < width - 1) ? valueGm_[gmOffset2 + 1] : 0;
            T v7 = (y < height - 1 && x >= 0) ? valueGm_[gmOffset3 + 1] : 0;
            T v8 = (y < height - 1 && x < width - 1) ? valueGm_[gmOffset4 + 1] : 0;

            T lh = y - Simt::Floor(y);
            T lw = x - Simt::Floor(x);
            T hh = 1 - lh;
            T hw = 1 - lw;

            T w1 = hh * hw;
            T w2 = hh * lw;
            T w3 = lh * hw;
            T w4 = lh * lw;

            T val1 = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
            T val2 = w1 * v5 + w2 * v6 + w3 * v7 + w4 * v8;

            T w = attnWeight[pointIdx];

            value1 = value1 + w * val1;
            value2 = value2 + w * val2;
        }
        outputGm_[baseOffset + idx] = value1;
        outputGm_[baseOffset + idx + 1] = value2;
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

        PrepareShape(shapes, shapeInt, shapeFloat, offset, offsetInt);

        for (uint32_t taskIdx = blkIdx_ * compTaskNum_; taskIdx < batchSize_ * numQueries_; taskIdx += compTaskNum_ * coreNum_) {
            uint32_t baseNum = (taskIdx / numQueries_ + 1) * numQueries_ - taskIdx;
            uint32_t taskNum = min(compTaskNum_, batchSize_ * numQueries_ - taskIdx);
            uint32_t baseSrcOffset = taskIdx / numQueries_ * numKeys_ * numHeads_;
            uint32_t nextSrcOffset = baseSrcOffset + numKeys_ * numHeads_;
            CopyInSample(locationFloat[2 * alignedOneTaskNum_], attentionWeight, taskIdx, taskNum);
            pipe_barrier(PIPE_ALL);

            ComputeGmOffsetVF<float, int32_t>(taskRpt_, numHeads_, embedDims_, baseSrcOffset, nextSrcOffset, baseNum * oneQueryNum_, locationFloat, shapeFloat, offsetInt, locationInt);
            pipe_barrier(PIPE_ALL);

            CallMSDASimtFunc(taskIdx, taskNum, locationFloat, shapeFloat, locationInt, attentionWeight);
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void CallMSDASimtFunc(uint32_t taskIdx, uint32_t taskNum, const LocalTensor<float>& locationFloat,
        const LocalTensor<float>& shapeFloat, const LocalTensor<int32_t>& locationInt, const LocalTensor<float>& attentionWeight)
    {
        if ((embedDims_ % 2 == 0) && (taskNum * numHeads_ * embedDims_ > 1024)) {
            AscendC::Simt::VF_CALL<MSDASimtComputeDoubleEmbed<float, int32_t>>(AscendC::Simt::Dim3{1024}, (__gm__ float*)valueGm_.GetPhyAddr(),
                (__gm__ float*)outputGm_.GetPhyAddr(), (__ubuf__ float*)locationFloat.GetPhyAddr(), (__ubuf__ float*)shapeFloat.GetPhyAddr(),
                (__ubuf__ int32_t*)locationInt.GetPhyAddr(), (__ubuf__ float*)attentionWeight.GetPhyAddr(),
                taskNum * numHeads_ * embedDims_, taskIdx * outDims_, oneHeadNum_, embedDims_, outDims_);
        } else {
            AscendC::Simt::VF_CALL<MSDASimtCompute<float, int32_t>>(AscendC::Simt::Dim3{1024}, (__gm__ float*)valueGm_.GetPhyAddr(),
                (__gm__ float*)outputGm_.GetPhyAddr(), (__ubuf__ float*)locationFloat.GetPhyAddr(), (__ubuf__ float*)shapeFloat.GetPhyAddr(),
                (__ubuf__ int32_t*)locationInt.GetPhyAddr(), (__ubuf__ float*)attentionWeight.GetPhyAddr(),
                taskNum * numHeads_ * embedDims_, taskIdx * outDims_, oneHeadNum_, embedDims_, outDims_);
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

protected:
    TPipe* pipe_;
    GlobalTensor<float> valueGm_, locationGm_, attentionWeightsGm_, outputGm_;
    GlobalTensor<int32_t> valueSpatialShapesGm_, valueLevelStartIndexGm_;

    TBuf<TPosition::VECCALC> locationQue_, attentionWeightsQue_, shapeQue_, offsetQue_;
    TBuf<TPosition::VECCALC> shapeFloatBuf_, offsetIntBuf_, gmOffsetBuf_;

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