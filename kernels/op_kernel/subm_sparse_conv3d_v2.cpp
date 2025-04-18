/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "kernel_operator.h"
using namespace AscendC;
 
namespace {
    constexpr int32_t INT32_BYTE_SIZE = 4;
    constexpr int32_t FLOAT_BYTE_SIZE = 4;
    constexpr int32_t ALIGNED_BYTE_SIZE = 32;
    constexpr int32_t REPEAT_BYTE_SIZE = 256;
    constexpr int32_t SPATIAL_SHAPE_THRESHOLD = 200000000;
    constexpr int32_t INDICES_TASK_SIZE = 4;
    constexpr int32_t SPATIAL_0_LOCAL_IDX = 1;
    constexpr int32_t SPATIAL_1_LOCAL_IDX = 2;
    constexpr int32_t SPATIAL_2_LOCAL_IDX = 3;
    constexpr uint8_t SRC_PARTTEN_0 = 3;
    constexpr uint8_t SRC_PARTTEN_1 = 4;
    constexpr uint8_t SRC_PARTTEN_2 = 5;
    constexpr uint8_t SRC_PARTTEN_3 = 6;
};

class KernelSubmSparseConv3dV2 {
public:
   __aicore__ inline KernelSubmSparseConv3dV2() {}
   __aicore__ inline void InitTiling(SubmSparseConv3dV2TilingData *tilingData)
   {
        k0_ = tilingData->k0;
        k1_ = tilingData->k1;
        k2_ = tilingData->k2;
        k12_ = k1_ * k2_;
        kernelSize_ = k0_ * k12_;
        kernelSizeAligned_ = AlignUp(kernelSize_, ALIGNED_BYTE_SIZE / FLOAT_BYTE_SIZE);
        batchSize_ = tilingData->batchSize;
        inChannels_ = tilingData->inChannels;
        inChannelsAligned_ = AlignUp(inChannels_, ALIGNED_BYTE_SIZE / FLOAT_BYTE_SIZE);
        outputOneLineElementCount_ = kernelSize_ * inChannels_;
        outputHalfLineElementCount_ = (kernelSize_ / TWO) * inChannels_;
        spatialShape0_ = tilingData->spatialShape0;
        spatialShape1_ = tilingData->spatialShape1;
        spatialShape2_ = tilingData->spatialShape2;
        spatialShape01_ = spatialShape0_ * spatialShape1_;
        spatialShape12_ = spatialShape1_ * spatialShape2_;
        totalSpatialShape_ = (int64_t)spatialShape01_ * spatialShape2_;
        useTwolevelMap_ = totalSpatialShape_ * batchSize_ >= SPATIAL_SHAPE_THRESHOLD;

        if (blkIdx_ < tilingData->bigCoreCount) {
            globalTaskOffset_ = (tilingData->coreTaskCount + 1) * blkIdx_;
            coreTaskCount_ = tilingData->coreTaskCount + 1;
        } else {
            globalTaskOffset_ = (tilingData->coreTaskCount + 1) * tilingData->bigCoreCount +
                                tilingData->coreTaskCount * (blkIdx_ - tilingData->bigCoreCount);
            coreTaskCount_ = tilingData->coreTaskCount;
        }
        singleLoopTask_ = tilingData->singleLoopTask;
        singleLoopTaskAligned_ = AlignUp(singleLoopTask_, ALIGNED_BYTE_SIZE / FLOAT_BYTE_SIZE);
        copyInOffset_ = 0;
        tmpBufLength_ = singleLoopTask_;
    }

    __aicore__ inline void InitGM(GM_ADDR feature, GM_ADDR indices, GM_ADDR map1, GM_ADDR map2,
        GM_ADDR feature_out, GM_ADDR indices_offset)
    {
        inputFeatureGM_.SetGlobalBuffer((__gm__ float*) feature);
        indicesGM_.SetGlobalBuffer((__gm__ int32_t*) indices);
        map1GM_.SetGlobalBuffer((__gm__ int32_t*) map1);
        outputFeatureGM_.SetGlobalBuffer((__gm__ float*) feature_out);
        indicesOffsetGM_.SetGlobalBuffer((__gm__ int32_t*) indices_offset);
        if (useTwolevelMap_) {
            map2GM_.SetGlobalBuffer((__gm__ int32_t*) map2);
        }
    }

    __aicore__ inline void InitUB()
    {
        pipe_->InitBuffer(inputIndicesBuf_, INDICES_TASK_SIZE * singleLoopTaskAligned_ * INT32_BYTE_SIZE);
        pipe_->InitBuffer(totalIndicesBuf_, INDICES_TASK_SIZE * singleLoopTaskAligned_ * INT32_BYTE_SIZE);
        pipe_->InitBuffer(tmpFeatureBuf_, singleLoopTaskAligned_ * inChannelsAligned_ * FLOAT_BYTE_SIZE);

        inputIndicesLocal_ = inputIndicesBuf_.Get<int32_t>();
        tmpFeatureLocal_ = tmpFeatureBuf_.Get<float>();

        batchIdxLocal_ = totalIndicesBuf_.Get<int32_t>();
        spatial0Local_ = batchIdxLocal_[singleLoopTaskAligned_ * SPATIAL_0_LOCAL_IDX];
        spatial1Local_ = batchIdxLocal_[singleLoopTaskAligned_ * SPATIAL_1_LOCAL_IDX];
        spatial2Local_ = batchIdxLocal_[singleLoopTaskAligned_ * SPATIAL_2_LOCAL_IDX];
    }

    __aicore__ inline void Init(TPipe *pipe, GM_ADDR feature, GM_ADDR indices, GM_ADDR map1, GM_ADDR map2,
        GM_ADDR feature_out, GM_ADDR indices_offset, SubmSparseConv3dV2TilingData *tilingData)
    {
        pipe_ = pipe;
        blkIdx_ = GetBlockIdx();
        InitTiling(tilingData);
        InitGM(feature, indices, map1, map2, feature_out, indices_offset);
        InitUB();
        eventMTE2ToMTE3_ = pipe_->AllocEventID<HardEvent::MTE2_MTE3>();
    }

    __aicore__ inline void Process()
    {
        for (int32_t taskOffset = 0; taskOffset < coreTaskCount_;
                taskOffset += singleLoopTask_, globalTaskOffset_ += singleLoopTask_) {
            uint32_t taskCount = min(singleLoopTask_, coreTaskCount_ - taskOffset);

            // CopyIn
            DataCopyPad(inputIndicesLocal_, indicesGM_[globalTaskOffset_ * INDICES_TASK_SIZE],
                {1, static_cast<uint32_t>(4 * singleLoopTask_ * INT32_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();

            uint32_t mask = 0;
            uint64_t rsvdCnt = 0;
            uint16_t repeatTimes = Ceil(singleLoopTask_ * 4, REPEAT_BYTE_SIZE / INT32_BYTE_SIZE);
            GatherMask(batchIdxLocal_, inputIndicesLocal_, SRC_PARTTEN_0, false, mask, { 1, repeatTimes, 8, 0 }, rsvdCnt);
            GatherMask(spatial0Local_, inputIndicesLocal_, SRC_PARTTEN_1, false, mask, { 1, repeatTimes, 8, 0 }, rsvdCnt);
            GatherMask(spatial1Local_, inputIndicesLocal_, SRC_PARTTEN_2, false, mask, { 1, repeatTimes, 8, 0 }, rsvdCnt);
            GatherMask(spatial2Local_, inputIndicesLocal_, SRC_PARTTEN_3, false, mask, { 1, repeatTimes, 8, 0 }, rsvdCnt);

            Adds(spatial0Local_, spatial0Local_, - k0_ / (int32_t)TWO, taskCount);
            Adds(spatial1Local_, spatial1Local_, - k1_ / (int32_t)TWO, taskCount);
            Adds(spatial2Local_, spatial2Local_, - k2_ / (int32_t)TWO, taskCount);

            if (useTwolevelMap_) {
                ProcessOneLoopForTwoLevelMap(taskOffset, taskCount);
            } else {
                ProcessOneLoopForOneLevelMap(taskOffset, taskCount);
            }
        }
    }

    __aicore__ inline void ProcessOneLoopForOneLevelMap(int32_t taskOffset, uint32_t taskCount)
    {
        for (int32_t i = 0; i < taskCount; i++) {
            int32_t batchIdx = batchIdxLocal_.GetValue(i);
            int32_t spatial0BaseIdx = spatial0Local_.GetValue(i);
            int32_t spatial1BaseIdx = spatial1Local_.GetValue(i);
            int32_t spatial2BaseIdx = spatial2Local_.GetValue(i);

            for (int32_t k = 0; k < kernelSize_; k++) {
                int32_t k2Idx = k % k2_;
                int32_t k1Idx = (k % k12_) / k2_;
                int32_t k0Idx = k / k12_;
                
                int32_t spatial0Idx = spatial0BaseIdx + k0Idx;
                int32_t spatial1Idx = spatial1BaseIdx + k1Idx;
                int32_t spatial2Idx = spatial2BaseIdx + k2Idx;

                if (spatial0Idx < 0 || spatial1Idx < 0 || spatial2Idx < 0 ||
                    spatial0Idx >= spatialShape0_ || spatial1Idx >= spatialShape1_ || spatial2Idx >= spatialShape2_) {
                    continue;
                }

                int32_t map1Idx = batchIdx * totalSpatialShape_ + spatial0Idx * spatialShape12_ + spatial1Idx * spatialShape2_ + spatial2Idx;
                int32_t map1Val = map1GM_.GetValue(map1Idx);
                if (map1Val == -1) {
                    continue;
                }

                int32_t outputIdx = (globalTaskOffset_ + i) * kernelSize_ + k;
                indicesOffsetGM_.SetValue(outputIdx, map1Val);
                DataCopyPad(tmpFeatureLocal_[copyInOffset_ * inChannelsAligned_], inputFeatureGM_[map1Val * inChannels_],
                    {1, static_cast<uint32_t>(inChannels_ * FLOAT_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});
                
                SetFlag<HardEvent::MTE2_MTE3>(eventMTE2ToMTE3_);
                WaitFlag<HardEvent::MTE2_MTE3>(eventMTE2ToMTE3_);

                DataCopyPad(outputFeatureGM_[outputIdx * inChannels_], tmpFeatureLocal_[copyInOffset_ * inChannelsAligned_],
                    {1, static_cast<uint32_t>(inChannels_ * FLOAT_BYTE_SIZE), 0, 0, 0});

                copyInOffset_ = (copyInOffset_ + 1) % tmpBufLength_;
            }
        }
    }

    __aicore__ inline void ProcessOneLoopForTwoLevelMap(int32_t taskOffset, uint32_t taskCount)
    {
        for (int32_t i = 0; i < taskCount; i++) {
            int32_t batchIdx = batchIdxLocal_.GetValue(i);
            int32_t spatial0BaseIdx = spatial0Local_.GetValue(i);
            int32_t spatial1BaseIdx = spatial1Local_.GetValue(i);
            int32_t spatial2BaseIdx = spatial2Local_.GetValue(i);
            
            for (int32_t k = 0; k < kernelSize_; k++) {
                int32_t k2Idx = k % k2_;
                int32_t k1Idx = (k % k12_) / k2_;
                int32_t k0Idx = k / k12_;
                
                int32_t spatial0Idx = spatial0BaseIdx + k0Idx;
                int32_t spatial1Idx = spatial1BaseIdx + k1Idx;
                int32_t spatial2Idx = spatial2BaseIdx + k2Idx;

                if (spatial0Idx < 0 || spatial1Idx < 0 || spatial2Idx < 0 ||
                    spatial0Idx >= spatialShape0_ || spatial1Idx >= spatialShape1_ || spatial2Idx >= spatialShape2_) {
                    continue;
                }

                int32_t map1Idx = batchIdx * spatialShape01_ + spatial0Idx * spatialShape1_ + spatial1Idx;
                int32_t map1Val = map1GM_.GetValue(map1Idx);
                if (map1Val == -1) {
                    continue;
                }

                int32_t map2Idx = map1Val * spatialShape2_ + spatial2Idx;
                int32_t map2Val = map2GM_.GetValue(map2Idx);
                if (map2Val == -1) {
                    continue;
                }

                int32_t outputIdx = (globalTaskOffset_ + i) * kernelSize_ + k;
                indicesOffsetGM_.SetValue(outputIdx, map2Val);
                DataCopyPad(tmpFeatureLocal_[copyInOffset_ * inChannelsAligned_], inputFeatureGM_[map2Val * inChannels_],
                    {1, static_cast<uint32_t>(inChannels_ * FLOAT_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});
                
                SetFlag<HardEvent::MTE2_MTE3>(eventMTE2ToMTE3_);
                WaitFlag<HardEvent::MTE2_MTE3>(eventMTE2ToMTE3_);

                DataCopyPad(outputFeatureGM_[outputIdx * inChannels_], tmpFeatureLocal_[copyInOffset_ * inChannelsAligned_],
                    {1, static_cast<uint32_t>(inChannels_ * FLOAT_BYTE_SIZE), 0, 0, 0});

                copyInOffset_ = (copyInOffset_ + 1) % tmpBufLength_;
            }
        }
    }

private:
    bool useTwolevelMap_;
    int32_t blkIdx_;
    int32_t k0_, k1_, k2_, k12_, kernelSize_, batchSize_, inChannels_, tmpBufLength_, spatialShape0_, spatialShape1_,
        spatialShape2_, spatialShape01_, spatialShape12_, coreTaskCount_, singleLoopTask_, singleLoopTaskAligned_, globalTaskOffset_,
        inChannelsAligned_, copyInOffset_, kernelSizeAligned_, outputOneLineElementCount_, outputHalfLineElementCount_;
    int64_t totalSpatialShape_;
    GlobalTensor<float> inputFeatureGM_, outputFeatureGM_;
    GlobalTensor<int32_t> indicesGM_, map1GM_, map2GM_, indicesOffsetGM_;
    LocalTensor<float> tmpFeatureLocal_;
    LocalTensor<int32_t> inputIndicesLocal_, batchIdxLocal_, spatial0Local_, spatial1Local_, spatial2Local_;
    TBuf<TPosition::VECCALC> inputIndicesBuf_, totalIndicesBuf_, tmpFeatureBuf_, mapValBuf_;
    int32_t eventMTE2ToMTE3_;
    TPipe* pipe_;
};
 
extern "C" __global__ __aicore__ void subm_sparse_conv3d_v2(GM_ADDR feature, GM_ADDR indices, GM_ADDR map1, GM_ADDR map2,
                                                            GM_ADDR feature_out, GM_ADDR indices_offset, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelSubmSparseConv3dV2 op;
    TPipe pipe;
    op.Init(&pipe, feature, indices, map1, map2, feature_out, indices_offset, &tiling_data);
    op.Process();
}