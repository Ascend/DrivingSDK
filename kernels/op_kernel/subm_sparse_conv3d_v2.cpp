/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "kernel_operator.h"
using namespace AscendC;
 
namespace {
constexpr int64_t SPATIAL_SHAPE_THRESHOLD = 200000000;
constexpr int32_t INT32_BYTE_SIZE = 4;
constexpr int32_t FLOAT_BYTE_SIZE = 4;
constexpr int32_t ALIGNED_BYTE_SIZE = 32;
constexpr int32_t REPEAT_BYTE_SIZE = 256;
constexpr int32_t INDICES_TASK_SIZE = 4;
constexpr int32_t SPATIAL_0_LOCAL_IDX = 1;
constexpr int32_t SPATIAL_1_LOCAL_IDX = 2;
constexpr int32_t SPATIAL_2_LOCAL_IDX = 3;
constexpr int32_t WORK_LOCAL_IDX = 2;
constexpr uint8_t SRC_PARTTEN_0 = 3;
constexpr uint8_t SRC_PARTTEN_1 = 4;
constexpr uint8_t SRC_PARTTEN_2 = 5;
constexpr uint8_t SRC_PARTTEN_3 = 6;
constexpr uint8_t MAP_VAL_FLOAT_BUF_LENGTH = 3;
constexpr uint8_t K2_SIZE_3 = 3;
constexpr uint8_t K2_SIZE_5 = 5;
constexpr int8_t K2_IDX_0 = 0;
constexpr int8_t K2_IDX_1 = 1;
constexpr int8_t K2_IDX_2 = 2;
constexpr int8_t K2_IDX_3 = 3;
constexpr int8_t K2_IDX_4 = 4;
constexpr int32_t MAP2_OFFSET_1 = 1;
constexpr int32_t MAP2_OFFSET_2 = 2;
constexpr int32_t MAP2_OFFSET_3 = 3;
constexpr int32_t MAP2_OFFSET_4 = 4;
constexpr float SPARSE_THRESHOLD = 1e-4;
};

class KernelSubmSparseConv3dV2 {
public:
   __aicore__ inline KernelSubmSparseConv3dV2() {}
   __aicore__ inline void InitTiling(SubmSparseConv3dV2TilingData *tilingData)
   {
        k0_ = tilingData->k0;
        k1_ = tilingData->k1;
        k2_ = tilingData->k2;

        halfk0_ = k0_ / TWO;
        halfk1_ = k1_ / TWO;
        halfk2_ = k2_ / TWO;

        k12_ = k1_ * k2_;
        kernelSize_ = k0_ * k12_;
        kernelSizeAligned_ = AlignUp(kernelSize_, ALIGNED_BYTE_SIZE / FLOAT_BYTE_SIZE);
        k1Aligned_ = AlignUp(k1_, ALIGNED_BYTE_SIZE / FLOAT_BYTE_SIZE);
        k2Aligned_ = AlignUp(k2_, ALIGNED_BYTE_SIZE / FLOAT_BYTE_SIZE);
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
        sparseRate = tilingData->sparseRate;
        useTwolevelMap_ = (totalSpatialShape_ * (int64_t)batchSize_ >= SPATIAL_SHAPE_THRESHOLD) && (sparseRate < SPARSE_THRESHOLD);
        copyByteSize_ = inChannels_ * FLOAT_BYTE_SIZE;
        copyOutOneChannel_ = tilingData->copyOutOneChannel;

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
        tmpBufLength_ = singleLoopTask_;
        tmpBufIdx_ = 0;
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
        if (copyOutOneChannel_ == 0) {
            pipe_->InitBuffer(tmpFeatureBuf_, singleLoopTask_ * kernelSize_ * inChannelsAligned_ * FLOAT_BYTE_SIZE);
        } else {
            pipe_->InitBuffer(tmpFeatureBuf_, singleLoopTask_ * inChannelsAligned_ * FLOAT_BYTE_SIZE);
        }
        pipe_->InitBuffer(mapValBuf_, k0_ * k1_ * k2Aligned_ * INT32_BYTE_SIZE);
        pipe_->InitBuffer(mapValFloatBuf_, MAP_VAL_FLOAT_BUF_LENGTH * k0_ * k1_ * k2Aligned_ * FLOAT_BYTE_SIZE);
        pipe_->InitBuffer(indicesOffsetBuf_, singleLoopTaskAligned_ * kernelSizeAligned_ * INT32_BYTE_SIZE);

        inputIndicesLocal_ = inputIndicesBuf_.Get<int32_t>();
        tmpFeatureLocal_ = tmpFeatureBuf_.Get<float>();

        batchIdxLocal_ = totalIndicesBuf_.Get<int32_t>();
        spatial0Local_ = batchIdxLocal_[singleLoopTaskAligned_ * SPATIAL_0_LOCAL_IDX];
        spatial1Local_ = batchIdxLocal_[singleLoopTaskAligned_ * SPATIAL_1_LOCAL_IDX];
        spatial2Local_ = batchIdxLocal_[singleLoopTaskAligned_ * SPATIAL_2_LOCAL_IDX];
        mapValLocal_ = mapValBuf_.Get<int32_t>();
        mapValFloatLocal_ = mapValFloatBuf_.Get<float>();
        mapValFloatLocalBak_ = mapValFloatLocal_[k0_ * k1_ * k2Aligned_];
        workLocal_ = mapValFloatLocal_[WORK_LOCAL_IDX * k0_ * k1_ * k2Aligned_];
        indicesOffsetLocal_ = indicesOffsetBuf_.Get<int32_t>();
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
                {1, static_cast<uint32_t>(INDICES_TASK_SIZE * taskCount * INT32_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
            
            uint32_t mask = 0;
            uint64_t rsvdCnt = 0;
            uint16_t repeatTimes = Ceil(taskCount * 4, REPEAT_BYTE_SIZE / INT32_BYTE_SIZE);
            GatherMask(batchIdxLocal_, inputIndicesLocal_, SRC_PARTTEN_0, false, mask, { 1, repeatTimes, 8, 0 }, rsvdCnt);
            GatherMask(spatial0Local_, inputIndicesLocal_, SRC_PARTTEN_1, false, mask, { 1, repeatTimes, 8, 0 }, rsvdCnt);
            GatherMask(spatial1Local_, inputIndicesLocal_, SRC_PARTTEN_2, false, mask, { 1, repeatTimes, 8, 0 }, rsvdCnt);
            GatherMask(spatial2Local_, inputIndicesLocal_, SRC_PARTTEN_3, false, mask, { 1, repeatTimes, 8, 0 }, rsvdCnt);
            Duplicate(indicesOffsetLocal_, static_cast<int32_t>(-1), taskCount * kernelSizeAligned_);
            if (copyOutOneChannel_ == 0) {
                Duplicate(tmpFeatureLocal_, static_cast<float>(0), taskCount * kernelSize_ * inChannelsAligned_);
            } else {
                Duplicate(tmpFeatureLocal_, static_cast<float>(0), taskCount * inChannelsAligned_);
            }
            
            Adds(spatial0Local_, spatial0Local_, - halfk0_, taskCount);
            Adds(spatial1Local_, spatial1Local_, - halfk0_, taskCount);
            Adds(spatial2Local_, spatial2Local_, - halfk0_, taskCount);

            if (useTwolevelMap_) {
                ProcessOneLoopForTwoLevelMap(taskCount);
            } else {
                if (sparseRate < SPARSE_THRESHOLD) {
                    ProcessOneLoopForOneLevelMap(taskCount);
                } else {
                    ProcessOneLoopForOneLevelMapDense(taskCount);
                }
            }

            DataCopyPad(indicesOffsetGM_[globalTaskOffset_ * kernelSize_], indicesOffsetLocal_,
                {static_cast<uint16_t>(taskCount), static_cast<uint32_t>(kernelSize_ * INT32_BYTE_SIZE), 0, 0, 0});

            if (copyOutOneChannel_ == 0) {
                SetFlag<HardEvent::MTE2_MTE3>(eventMTE2ToMTE3_);
                WaitFlag<HardEvent::MTE2_MTE3>(eventMTE2ToMTE3_);

                DataCopyPad(outputFeatureGM_[((globalTaskOffset_) * kernelSize_) * inChannels_], tmpFeatureLocal_,
                            {static_cast<uint16_t>(taskCount * kernelSize_), copyByteSize_, 0, 0, 0});
            }
        }
    }

    __aicore__ inline void ProcessOnePoint(const int16_t &i, const int8_t &k0Idx, const int8_t &k1Idx, const int8_t &k2Idx, const int32_t &mapVal)
    {
        if (mapVal == -1) {
            return;
        }

        int32_t k = k0Idx * k12_ + k1Idx * k2_ + k2Idx;
        indicesOffsetLocal_.SetValue(i * kernelSizeAligned_ + k, mapVal);
        if (copyOutOneChannel_ == 0) {
            DataCopyPad(tmpFeatureLocal_[(i * kernelSize_ + k) * inChannelsAligned_], inputFeatureGM_[mapVal * inChannels_], {1, copyByteSize_, 0, 0, 0}, {false, 0, 0, 0});
        } else {
            DataCopyPad(tmpFeatureLocal_[tmpBufIdx_ * inChannelsAligned_], inputFeatureGM_[mapVal * inChannels_], {1, copyByteSize_, 0, 0, 0}, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_MTE3>(0);
            WaitFlag<HardEvent::MTE2_MTE3>(0);
            DataCopyPad(outputFeatureGM_[(globalTaskOffset_ + i) * kernelSize_ * inChannels_ + k * inChannels_], tmpFeatureLocal_[tmpBufIdx_ * inChannelsAligned_], {1, copyByteSize_, 0, 0, 0});
            tmpBufIdx_ = (tmpBufIdx_ + 1) % tmpBufLength_;
        }
    }

    __aicore__ inline void ProcessOneLoopForOneLevelMap(uint32_t taskCount)
    {
        for (int16_t i = 0; i < taskCount; i++) {
            int16_t batchIdx = batchIdxLocal_.GetValue(i);
            int16_t spatial0BaseIdx = spatial0Local_.GetValue(i);
            int16_t spatial1BaseIdx = spatial1Local_.GetValue(i);
            int16_t spatial2BaseIdx = spatial2Local_.GetValue(i);

            int32_t batchOffset = batchIdx * totalSpatialShape_;

            for (int16_t k0Idx = spatial0BaseIdx; k0Idx < k0_ + spatial0BaseIdx; k0Idx++) {
                int32_t mapOffset = batchOffset + k0Idx * spatialShape12_ + spatial1BaseIdx * spatialShape2_ + spatial2BaseIdx;
                DataCopyPad(mapValLocal_[(k0Idx - spatial0BaseIdx) * k1_ * k2Aligned_], map1GM_[mapOffset],
                    {static_cast<uint16_t>(k1_), static_cast<uint32_t>(k2_ * INT32_BYTE_SIZE), static_cast<uint16_t>((spatialShape2_ - k2_) * INT32_BYTE_SIZE), 0, 0},
                    {true, 0, static_cast<uint8_t>(k2Aligned_ - k2_), -2});
            }
            PipeBarrier<PIPE_ALL>();
            
            Cast(mapValFloatLocalBak_, mapValLocal_, RoundMode::CAST_ROUND, k0_ * k1_ * k2Aligned_);

            do {
                ReduceMax<float>(mapValFloatLocal_, mapValFloatLocalBak_, workLocal_, k0_ * k1_ * k2Aligned_, true);
                int32_t mapVal = static_cast<int32_t>(mapValFloatLocal_.GetValue(0));
                if (mapVal < 0) {
                    break;
                }
                
                float mapIdxFloat = mapValFloatLocal_.GetValue(1);
                int32_t mapIdx = *reinterpret_cast<int32_t*>(&mapIdxFloat);

                ProcessOnePoint(i, mapIdx / (k2Aligned_ * k1_), (mapIdx % (k2Aligned_ * k1_)) / k2Aligned_, mapIdx % k2Aligned_, mapVal);
                mapValFloatLocalBak_.SetValue(mapIdx, -2.0f);
            } while (true);
        }
    }

    __aicore__ inline void ProcessOneLoopForOneLevelMapDense(uint32_t taskCount)
    {
        for (int16_t i = 0; i < taskCount; i++) {
            int16_t batchIdx = batchIdxLocal_.GetValue(i);
            int16_t spatial0BaseIdx = spatial0Local_.GetValue(i);
            int16_t spatial1BaseIdx = spatial1Local_.GetValue(i);
            int16_t spatial2BaseIdx = spatial2Local_.GetValue(i);

            int32_t batchOffset = batchIdx * totalSpatialShape_;
            for (int16_t k0Idx = spatial0BaseIdx; k0Idx < k0_ + spatial0BaseIdx; k0Idx++) {
                int32_t mapOffset = batchOffset + k0Idx * spatialShape12_ + spatial1BaseIdx * spatialShape2_ + spatial2BaseIdx;
                DataCopyPad(mapValLocal_[(k0Idx - spatial0BaseIdx) * k1_ * k2Aligned_], map1GM_[mapOffset],
                    {static_cast<uint16_t>(k1_), static_cast<uint32_t>(k2_ * INT32_BYTE_SIZE), static_cast<uint16_t>((spatialShape2_ - k2_) * INT32_BYTE_SIZE), 0, 0},
                    {true, 0, static_cast<uint8_t>(k2Aligned_ - k2_), -2});
            }
            PipeBarrier<PIPE_ALL>();

            int32_t innerKernelOffset = 0;
            for (int8_t k0Idx = 0; k0Idx < k0_; k0Idx++) {
                innerKernelOffset = k0Idx * k1_ * k2Aligned_;
                for (int8_t k1Idx = 0; k1Idx < k1_; k1Idx++) {
                    if (k2_ == K2_SIZE_3) {
                        ProcessOnePoint(i, k0Idx, k1Idx, K2_IDX_0, mapValLocal_.GetValue(innerKernelOffset));
                        ProcessOnePoint(i, k0Idx, k1Idx, K2_IDX_1, mapValLocal_.GetValue(innerKernelOffset + MAP2_OFFSET_1));
                        ProcessOnePoint(i, k0Idx, k1Idx, K2_IDX_2, mapValLocal_.GetValue(innerKernelOffset + MAP2_OFFSET_2));
                    } else if (k2_ == K2_SIZE_5) {
                        ProcessOnePoint(i, k0Idx, k1Idx, K2_IDX_0, mapValLocal_.GetValue(innerKernelOffset));
                        ProcessOnePoint(i, k0Idx, k1Idx, K2_IDX_1, mapValLocal_.GetValue(innerKernelOffset + MAP2_OFFSET_1));
                        ProcessOnePoint(i, k0Idx, k1Idx, K2_IDX_2, mapValLocal_.GetValue(innerKernelOffset + MAP2_OFFSET_2));
                        ProcessOnePoint(i, k0Idx, k1Idx, K2_IDX_3, mapValLocal_.GetValue(innerKernelOffset + MAP2_OFFSET_3));
                        ProcessOnePoint(i, k0Idx, k1Idx, K2_IDX_4, mapValLocal_.GetValue(innerKernelOffset + MAP2_OFFSET_4));
                    }
                    innerKernelOffset += k2Aligned_;
                }
            }
        }
    }

    __aicore__ inline void ProcessOneLoopForTwoLevelMap(uint32_t taskCount)
    {
        for (int16_t i = 0; i < taskCount; i++) {
            int16_t batchIdx = batchIdxLocal_.GetValue(i);
            int16_t spatial0BaseIdx = spatial0Local_.GetValue(i);
            int16_t spatial1BaseIdx = spatial1Local_.GetValue(i);
            int16_t spatial2BaseIdx = spatial2Local_.GetValue(i);

            int32_t batchOffset = batchIdx * spatialShape01_;
            int32_t mapOffset = batchOffset + spatial0BaseIdx * spatialShape1_ + spatial1BaseIdx;
            DataCopyPad(mapValLocal_, map1GM_[mapOffset],
                {static_cast<uint16_t>(k0_), static_cast<uint32_t>(k1_ * INT32_BYTE_SIZE), static_cast<uint16_t>((spatialShape1_ - k1_) * INT32_BYTE_SIZE), 0, 0},
                {true, 0, static_cast<uint8_t>(k1Aligned_ - k1_), -2});
            PipeBarrier<PIPE_ALL>();

            Cast(mapValFloatLocalBak_, mapValLocal_, RoundMode::CAST_ROUND, k0_ * k1Aligned_);

            do {
                ReduceMax<float>(mapValFloatLocal_, mapValFloatLocalBak_, workLocal_, k0_ * k1Aligned_, true);
                int32_t map1Val = static_cast<int32_t>(mapValFloatLocal_.GetValue(0));
                if (map1Val < 0) {
                    break;
                }

                float mapIdxFloat = mapValFloatLocal_.GetValue(1);
                int32_t mapIdx = *reinterpret_cast<int32_t*>(&mapIdxFloat);
                int8_t k0Idx = mapIdx / k1Aligned_;
                int8_t k1Idx = mapIdx % k1Aligned_;

                int32_t map2Offset = map1Val * spatialShape2_ + spatial2BaseIdx;
                if (k2_ == K2_SIZE_3) {
                    ProcessOnePoint(i, k0Idx, k1Idx, K2_IDX_0, map2GM_.GetValue(map2Offset));
                    ProcessOnePoint(i, k0Idx, k1Idx, K2_IDX_1, map2GM_.GetValue(map2Offset + MAP2_OFFSET_1));
                    ProcessOnePoint(i, k0Idx, k1Idx, K2_IDX_2, map2GM_.GetValue(map2Offset + MAP2_OFFSET_2));
                } else if (k2_ == K2_SIZE_5) {
                    ProcessOnePoint(i, k0Idx, k1Idx, K2_IDX_0, map2GM_.GetValue(map2Offset));
                    ProcessOnePoint(i, k0Idx, k1Idx, K2_IDX_1, map2GM_.GetValue(map2Offset + MAP2_OFFSET_1));
                    ProcessOnePoint(i, k0Idx, k1Idx, K2_IDX_2, map2GM_.GetValue(map2Offset + MAP2_OFFSET_2));
                    ProcessOnePoint(i, k0Idx, k1Idx, K2_IDX_3, map2GM_.GetValue(map2Offset + MAP2_OFFSET_3));
                    ProcessOnePoint(i, k0Idx, k1Idx, K2_IDX_4, map2GM_.GetValue(map2Offset + MAP2_OFFSET_4));
                }
                mapValFloatLocalBak_.SetValue(mapIdx, -2.0f);
            } while (true);
        }
    }

private:
    bool useTwolevelMap_;
    uint32_t blkIdx_, copyByteSize_, copyOutOneChannel_, tmpBufIdx_;
    int32_t k0_, k1_, k2_, k12_, halfk0_, halfk1_, halfk2_, k1Aligned_, k2Aligned_, kernelSize_, batchSize_, inChannels_, tmpBufLength_, spatialShape0_, spatialShape1_,
        spatialShape2_, spatialShape01_, spatialShape12_, coreTaskCount_, singleLoopTask_, singleLoopTaskAligned_, globalTaskOffset_,
        inChannelsAligned_, kernelSizeAligned_, outputOneLineElementCount_, outputHalfLineElementCount_;
    int32_t eventMTE2ToMTE3_;
    float sparseRate;
    int64_t totalSpatialShape_;
    GlobalTensor<float> inputFeatureGM_, outputFeatureGM_;
    GlobalTensor<int32_t> indicesGM_, map1GM_, map2GM_, indicesOffsetGM_;
    LocalTensor<float> tmpFeatureLocal_, mapValFloatLocal_, mapValFloatLocalBak_, workLocal_;
    LocalTensor<int32_t> inputIndicesLocal_, batchIdxLocal_, spatial0Local_, spatial1Local_, spatial2Local_, mapValLocal_, indicesOffsetLocal_;
    TBuf<TPosition::VECCALC> inputIndicesBuf_, totalIndicesBuf_, tmpFeatureBuf_, mapValBuf_, mapValFloatBuf_, indicesOffsetBuf_;
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