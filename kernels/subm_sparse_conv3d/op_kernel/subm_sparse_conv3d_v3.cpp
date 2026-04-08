/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
 
namespace {
constexpr int64_t SPATIAL_SHAPE_THRESHOLD = 400000000;
constexpr int32_t INT32_BYTE_SIZE = 4;
constexpr int32_t INDICES_ELEMENTS_COUNT = 4;
constexpr int32_t REPEAT_BYTE_SIZE = 256;
constexpr uint8_t SRC_PARTTEN_0 = 3;
constexpr uint8_t SRC_PARTTEN_1 = 4;
constexpr uint8_t SRC_PARTTEN_2 = 5;
constexpr uint8_t SRC_PARTTEN_3 = 6;
constexpr int32_t BYTE_SIZE_PER_BLOCK = 32;
constexpr int32_t NUM_TWO = 2;
constexpr int32_t INT64_BIT_SIZE = 64;
constexpr MatmulConfig SUBM_SPARSE_CONV3D_CFG = GetIBShareNormConfig();
};


template<typename T>
class KernelSubmSparseConv3dV3 {
public:
    using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    matmul::Matmul<AType, BType, CType, CType, SUBM_SPARSE_CONV3D_CFG> mm0_;
    matmul::Matmul<AType, BType, CType, CType, SUBM_SPARSE_CONV3D_CFG> mm1_;
    
   __aicore__ inline KernelSubmSparseConv3dV3() {}
   __aicore__ inline void InitTiling(SubmSparseConv3dV3TilingData *tilingData)
   {
        byteSizePerElements_ = sizeof(T);
        k0_ = tilingData->k0;
        k1_ = tilingData->k1;
        k2_ = tilingData->k2;
        batchSize_ = tilingData->batchSize;
        inChannels_ = tilingData->inChannels;
        outChannels_ = tilingData->outChannels;
        spatialShape0_ = tilingData->spatialShape0;
        spatialShape1_ = tilingData->spatialShape1;
        spatialShape2_ = tilingData->spatialShape2;
        singleLoopTask_ = tilingData->singleLoopTask;
        totalTaskCount_ = tilingData->totalTaskCount;
        availableUBSize_ = tilingData->availableUBSize;
        gatherBufLen_ = tilingData->gatherBufLen;
        scatterBufLen_ = tilingData->scatterBufLen;
        stage2SingleLoopTask_ = tilingData->stage2SingleLoopTask;
        withKey_ = tilingData->withKey;

        kernelSize_ = k0_ * k1_ * k2_;
        spatialShape0_times_1_ = spatialShape0_ * spatialShape1_;
        spatialShape1_times_2_ = spatialShape1_ * spatialShape2_;
        totalSpatialShape_ = (int64_t)spatialShape0_times_1_ * spatialShape2_;
        useTwolevelMap_ = (totalSpatialShape_ >= SPATIAL_SHAPE_THRESHOLD);
        kernelSizeAligned_ = AlignUp(kernelSize_, BYTE_SIZE_PER_BLOCK / INT32_BYTE_SIZE);
        inChannelsAligned_ = AlignUp(inChannels_, BYTE_SIZE_PER_BLOCK / byteSizePerElements_);
        outChannelsAligned_ = AlignUp(outChannels_, BYTE_SIZE_PER_BLOCK / byteSizePerElements_);
        singleLoopTaskAligned_ = AlignUp(singleLoopTask_, BYTE_SIZE_PER_BLOCK / INT32_BYTE_SIZE);
        k2Aligned_ = AlignUp(k2_, BYTE_SIZE_PER_BLOCK / INT32_BYTE_SIZE);
        k1Aligned_ = AlignUp(k1_, BYTE_SIZE_PER_BLOCK / INT32_BYTE_SIZE);
        mapValBufSize_ = AlignUp(k0_ * k1_ * k2Aligned_, BYTE_SIZE_PER_BLOCK / INT32_BYTE_SIZE);

        if (blkIdx_ < tilingData->bigCoreCount) {
            taskStartOffset_ = (tilingData->coreTaskCount + 1) * blkIdx_;
            coreTaskCount_ = tilingData->coreTaskCount + 1;
        } else {
            taskStartOffset_ = (tilingData->coreTaskCount + 1) * tilingData->bigCoreCount +
                                tilingData->coreTaskCount * (blkIdx_ - tilingData->bigCoreCount);
            coreTaskCount_ = tilingData->coreTaskCount;
        }
        stage2SingleLoopTaskAligned_ = AlignUp(stage2SingleLoopTask_, BYTE_SIZE_PER_BLOCK / INT32_BYTE_SIZE);
   }

    __aicore__ inline void InitGM(GM_ADDR feature, GM_ADDR weight, GM_ADDR indices, GM_ADDR indices_offset, GM_ADDR map1, GM_ADDR map2,
        GM_ADDR feature_out, GM_ADDR out_indices_offset, GM_ADDR workspace)
    {
        inputFeatureGM_.SetGlobalBuffer((__gm__ T*) feature);
        weightGM_.SetGlobalBuffer((__gm__ T*) weight);
        indicesGM_.SetGlobalBuffer((__gm__ int32_t*) indices);

        if (withKey_) {
            indicesOffsetGM_.SetGlobalBuffer((__gm__ int32_t*) indices_offset);
        } else {
            map1GM_.SetGlobalBuffer((__gm__ int32_t*) map1);
            if (useTwolevelMap_) {
                map2GM_.SetGlobalBuffer((__gm__ int32_t*) map2);
            }
            indicesOffsetGM_.SetGlobalBuffer((__gm__ int32_t*) out_indices_offset);
        }
        
        outputFeatureGM_.SetGlobalBuffer((__gm__ T*) feature_out);
        mmFeatureGM1_.SetGlobalBuffer(((__gm__ T*) workspace) + taskStartOffset_ * inChannels_);
        mmFeatureGM2_ = mmFeatureGM1_[totalTaskCount_ * inChannels_];

        matmulResultGM1_.SetGlobalBuffer(((__gm__ T*) workspace) + 2 * totalTaskCount_ * inChannels_ +
            (taskStartOffset_ * outChannels_));
        matmulResultGM2_ = matmulResultGM1_[totalTaskCount_ * outChannels_];
    }

    __aicore__ inline void InitUB()
    {
        pipe_->InitBuffer(ubBuf_, availableUBSize_);

        if (!withKey_) {
            // for stage1 compute indicesOffset
            inputIndicesLocal_ = ubBuf_.Get<int32_t>();
            batchIdxLocal_ = inputIndicesLocal_[INT32_BYTE_SIZE * singleLoopTaskAligned_];
            spatial0Local_ = batchIdxLocal_[singleLoopTaskAligned_];
            spatial1Local_ = spatial0Local_[singleLoopTaskAligned_];
            spatial2Local_ = spatial1Local_[singleLoopTaskAligned_];
            indicesOffsetLocal_ = spatial2Local_[singleLoopTaskAligned_];
            map1ValLocal_ = indicesOffsetLocal_[kernelSizeAligned_ * singleLoopTaskAligned_];
            mapValLocal_ = map1ValLocal_[k0_ * k1Aligned_];
            gatherOffsetLocal_ = mapValLocal_[mapValBufSize_ * singleLoopTaskAligned_].template ReinterpretCast<uint32_t>();

            // initialize gatherOffsetLocal_
            for (int32_t k = 0; k < kernelSize_; k++) {
                int32_t innerKernelOffset = (k / k2_) * k2Aligned_ + k % k2_;
                for (int i = 0; i < singleLoopTaskAligned_; i++) {
                    gatherOffsetLocal_.SetValue(k * singleLoopTaskAligned_ + i, INT32_BYTE_SIZE * (i * mapValBufSize_ + innerKernelOffset));
                }
            }
        }
        
        // for stage2 copy feature and sparseMatmul
        gatherFeatureLocal1_ = ubBuf_.Get<T>();
        gatherFeatureLocal2_ = gatherFeatureLocal1_[NUM_TWO * gatherBufLen_ * inChannelsAligned_];
        validIndicesForGatherFeatureLocal_ = gatherFeatureLocal2_[NUM_TWO * gatherBufLen_ * inChannelsAligned_].template ReinterpretCast<int32_t>();
        maskForGatherLocal_ = validIndicesForGatherFeatureLocal_[stage2SingleLoopTaskAligned_];

        scatterFeatureLocal1_ = ubBuf_.Get<T>();
        scatterFeatureLocal2_ = scatterFeatureLocal1_[NUM_TWO * scatterBufLen_ * outChannelsAligned_];
        validIndicesForScatterFeatureLocal_ = scatterFeatureLocal2_[NUM_TWO * scatterBufLen_ * outChannelsAligned_].template ReinterpretCast<int32_t>();
        maskForScatterLocal_ = validIndicesForScatterFeatureLocal_[stage2SingleLoopTaskAligned_];
    }

    __aicore__ inline void Init(TPipe *pipe, GM_ADDR feature, GM_ADDR weight, GM_ADDR indices, GM_ADDR indices_offset, GM_ADDR map1, GM_ADDR map2,
        GM_ADDR feature_out, GM_ADDR out_indices_offset, SubmSparseConv3dV3TilingData *tilingData, GM_ADDR workspace)
    {
        pipe_ = pipe;
        aicNum_ = GetBlockNum();
        blkIdx_ = GetBlockIdx();
        InitTiling(tilingData);
        InitGM(feature, weight, indices, indices_offset, map1, map2, feature_out, out_indices_offset, workspace);
        InitUB();
    }

    __aicore__ inline void ProcessCube(const int16_t &k, const uint32_t &taskCount, const uint32_t &offset)
    {
        if (taskCount <= 0) {
            return;
        }
        
        int16_t k1 = kernelSize_ - k - 1;

        mm0_.SetTensorA(mmFeatureGM1_[offset * inChannels_]);
        mm0_.SetTensorB(weightGM_[k * inChannels_ * outChannels_]);
        mm0_.SetSingleShape(taskCount, outChannels_, inChannels_);
        
        mm0_.template IterateAll<false>(matmulResultGM1_[offset * outChannels_], 0, false, true);

        mm1_.SetTensorA(mmFeatureGM2_[offset * inChannels_]);
        mm1_.SetTensorB(weightGM_[k1 * inChannels_ * outChannels_]);
        mm1_.SetSingleShape(taskCount, outChannels_, inChannels_);
        
        mm1_.template IterateAll<false>(matmulResultGM2_[offset * outChannels_], 0, false, true);

        ++matmulCount0_;
        ++matmulCount1_;
    }

    __aicore__ inline void MatmulCenterPoint()
    {
        if (coreTaskCount_ <= 0) {
            return ;
        }

        mm0_.SetTensorA(inputFeatureGM_[taskStartOffset_ * inChannels_]);
        mm0_.SetTensorB(weightGM_[kernelSize_ / NUM_TWO * inChannels_ * outChannels_]);
        mm0_.SetSingleShape(coreTaskCount_, outChannels_, inChannels_);
        
        mm0_.template IterateAll<false>(outputFeatureGM_[taskStartOffset_ * outChannels_], 1, false, true);
        ++matmulCount0_;
    }

    __aicore__ inline void ScatterAdd(const int32_t &k, const int32_t &validTaskCount) {   
        if (validTaskCount <= 0)
            return;
        
        int32_t singleLoopTaskStart = 0;
        for (int32_t taskOffset = 0; taskOffset < coreTaskCount_; taskOffset += stage2SingleLoopTask_) {
            uint32_t taskCount = min(stage2SingleLoopTask_, coreTaskCount_ - taskOffset);
            DataCopyPad(validIndicesForScatterFeatureLocal_, indicesOffsetGM_[k * totalTaskCount_ + taskStartOffset_ + taskOffset],
                {1, static_cast<uint32_t>(taskCount * INT32_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            CompareScalar(maskForScatterLocal_.ReinterpretCast<uint8_t>(), validIndicesForScatterFeatureLocal_,
                static_cast<int32_t>(-1), CMPMODE::EQ, stage2SingleLoopTaskAligned_);
            curLoopValidTask_ = 0;
            if (matmulCount0_-- > 0) {
                mm1_.WaitIterateAll();
            }
            if (matmulCount1_-- > 0) {
                mm0_.WaitIterateAll();
            }
            for (int32_t i = 0; i < taskCount; i+=INT64_BIT_SIZE) {
                uint64_t validIdxMask = maskForScatterLocal_.ReinterpretCast<uint64_t>().GetValue(i / INT64_BIT_SIZE);
                int32_t validIdx = ScalarGetSFFValue<0>(validIdxMask);
                int32_t curTaskIdx = validIdx + i;
                while(validIdx != -1 && (curTaskIdx < taskCount)) {
                    int32_t taskOffset2 = validIndicesForScatterFeatureLocal_.GetValue(curTaskIdx);
                    if (curLoopValidTask_ % scatterBufLen_ == 0) {
                        int32_t copyInTaskCount = validTaskCount - curLoopValidTask_ < scatterBufLen_? validTaskCount - curLoopValidTask_ : scatterBufLen_;
                        WaitFlag<HardEvent::MTE3_MTE2>(ping2_);
                        CopyInFeatures((singleLoopTaskStart + curLoopValidTask_) * outChannels_, copyInTaskCount);
                    }
                    SetAtomicAdd<T>();
                    DataCopyPad(outputFeatureGM_[(curTaskIdx + taskStartOffset_ + taskOffset) * outChannels_], scatterFeatureLocal1_[(curLoopValidTask_ % scatterBufLen_ + ping2_ * scatterBufLen_) * outChannelsAligned_],
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(outChannels_ * byteSizePerElements_), 0, 0, 0});
                    DataCopyPad(outputFeatureGM_[taskOffset2 * outChannels_], scatterFeatureLocal2_[(curLoopValidTask_ % scatterBufLen_ + ping2_ * scatterBufLen_) * outChannelsAligned_],
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(outChannels_ * byteSizePerElements_), 0, 0, 0});
                    SetAtomicNone();
                    curLoopValidTask_ += 1;
                    if (curLoopValidTask_ % scatterBufLen_ == 0) {
                        SetFlag<HardEvent::MTE3_MTE2>(ping2_);
                        ping2_ = 1 - ping2_;
                    }
                    validIdxMask += (static_cast<uint64_t>(1) << validIdx);
                    validIdx = ScalarGetSFFValue<0>(validIdxMask);
                    curTaskIdx = validIdx + i;
                }
            }
            if (curLoopValidTask_ % scatterBufLen_ != 0) {
                SetFlag<HardEvent::MTE3_MTE2>(ping2_);
            }
            singleLoopTaskStart += curLoopValidTask_;
        }
    }

    __aicore__ inline void ComputeValidPositionTwoMap(const uint32_t &taskOffset, const uint32_t &taskCount)
    {
        Duplicate(mapValLocal_, static_cast<int32_t>(-1), mapValBufSize_ * singleLoopTaskAligned_);
        Muls(batchIdxLocal_, batchIdxLocal_, spatialShape0_times_1_, taskCount);
        Muls(spatial0Local_, spatial0Local_, spatialShape1_, taskCount);
        Add(batchIdxLocal_, batchIdxLocal_, spatial0Local_, taskCount);
        Add(batchIdxLocal_, batchIdxLocal_, spatial1Local_, taskCount);

        PipeBarrier<PIPE_ALL>();
        
        for (int16_t i = 0; i < taskCount; i++) {
            DataCopyPad(map1ValLocal_, map1GM_[batchIdxLocal_.GetValue(i)],
                {static_cast<uint16_t>(k0_), static_cast<uint32_t>(k1_ * INT32_BYTE_SIZE), static_cast<uint16_t>((spatialShape1_ - k1_) * INT32_BYTE_SIZE), 0, 0},
                {false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();

            for (int8_t k0Idx = 0; k0Idx < k0_; k0Idx++) {
                for (int8_t k1Idx = 0; k1Idx < k1_; k1Idx++) {
                    int32_t mapVal1 = map1ValLocal_.GetValue(k0Idx * k1Aligned_ + k1Idx);
                    if (mapVal1 < 0) {
                        continue;
                    }
                    int32_t map2Idx = mapVal1 * spatialShape2_ + spatial2Local_.GetValue(i);

                    DataCopyPad(mapValLocal_[i * mapValBufSize_ + (k0Idx * k1_ + k1Idx) * k2Aligned_], map2GM_[map2Idx],
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(k2_ * INT32_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});
                }
            }
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);

        Gather(indicesOffsetLocal_, mapValLocal_, gatherOffsetLocal_, 0u, kernelSize_ * singleLoopTaskAligned_);
    }

    __aicore__ inline void ComputeValidPositionOneMap(const uint32_t &taskOffset, const uint32_t &taskCount)
    {
        Muls(batchIdxLocal_, batchIdxLocal_, static_cast<int32_t>(totalSpatialShape_), taskCount);
        Muls(spatial1Local_, spatial1Local_, spatialShape2_, taskCount);
        Add(batchIdxLocal_, batchIdxLocal_, spatial1Local_, taskCount);
        Add(batchIdxLocal_, batchIdxLocal_, spatial2Local_, taskCount);
        PipeBarrier<PIPE_ALL>();

        for (int16_t i = 0; i < taskCount; i++) {
            int32_t spatial0BaseIdx = spatial0Local_.GetValue(i);
            for (int16_t k0Idx = 0; k0Idx < k0_; k0Idx++) {
                DataCopyPad(mapValLocal_[i * mapValBufSize_ + k0Idx * k1_ * k2Aligned_], map1GM_[batchIdxLocal_.GetValue(i) + (k0Idx + spatial0BaseIdx) * spatialShape1_times_2_],
                    {static_cast<uint16_t>(k1_), static_cast<uint32_t>(k2_ * INT32_BYTE_SIZE), static_cast<uint16_t>((spatialShape2_ - k2_) * INT32_BYTE_SIZE), 0, 0},
                    {true, 0, static_cast<uint8_t>(k2Aligned_ - k2_), -1});
            }
        }

        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);

        Gather(indicesOffsetLocal_, mapValLocal_, gatherOffsetLocal_, 0u, kernelSize_ * singleLoopTaskAligned_);
    }

    __aicore__ inline void ComputeIndicesOffset()
    {
        SetFlag<HardEvent::MTE3_V>(0);

        // compute offset
        for (int32_t taskOffset = 0; taskOffset < coreTaskCount_; taskOffset += singleLoopTask_)
        {
            uint32_t taskCount = min(singleLoopTask_, coreTaskCount_ - taskOffset);
            uint32_t taskCountAligned = AlignUp(taskCount, BYTE_SIZE_PER_BLOCK / INT32_BYTE_SIZE);
            
            DataCopyPad(inputIndicesLocal_, indicesGM_[(taskStartOffset_ + taskOffset) * INDICES_ELEMENTS_COUNT],
                {1, static_cast<uint32_t>(4 * taskCount * INT32_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});

            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);

            uint32_t mask = 0;
            uint64_t rsvdCnt = 0;
            uint16_t repeatTimes = Ceil(taskCount * 4, REPEAT_BYTE_SIZE / INT32_BYTE_SIZE);
            GatherMask(batchIdxLocal_, inputIndicesLocal_, SRC_PARTTEN_0, false, mask, { 1, repeatTimes, 8, 0 }, rsvdCnt);
            GatherMask(spatial0Local_, inputIndicesLocal_, SRC_PARTTEN_1, false, mask, { 1, repeatTimes, 8, 0 }, rsvdCnt);
            GatherMask(spatial1Local_, inputIndicesLocal_, SRC_PARTTEN_2, false, mask, { 1, repeatTimes, 8, 0 }, rsvdCnt);
            GatherMask(spatial2Local_, inputIndicesLocal_, SRC_PARTTEN_3, false, mask, { 1, repeatTimes, 8, 0 }, rsvdCnt);

            if (useTwolevelMap_) {
                ComputeValidPositionTwoMap(taskOffset, taskCount);
            } else {
                ComputeValidPositionOneMap(taskOffset, taskCount);
            }

            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);

            DataCopyPad(indicesOffsetGM_[taskStartOffset_ + taskOffset], indicesOffsetLocal_,
                {static_cast<uint16_t>(kernelSize_), static_cast<uint32_t>(taskCount * INT32_BYTE_SIZE),
                static_cast<uint32_t>((singleLoopTaskAligned_ - taskCountAligned) / (BYTE_SIZE_PER_BLOCK / INT32_BYTE_SIZE)),
                static_cast<uint32_t>((totalTaskCount_ - taskCount) * INT32_BYTE_SIZE), 0});
            
            SetFlag<HardEvent::MTE3_V>(0);
        }
        WaitFlag<HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void CopyOutFeatures(int32_t copyOutOffset, int32_t copyOutTaskCount)
    {
        if (copyOutTaskCount <= 0)
            return;
            
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);
        
        if (inChannels_ != inChannelsAligned_) {
            DataCopyPad(mmFeatureGM1_[copyOutOffset], gatherFeatureLocal1_[ping1_ * gatherBufLen_ * inChannelsAligned_],
                {static_cast<uint16_t>(copyOutTaskCount), static_cast<uint32_t>(inChannels_ * byteSizePerElements_), 0, 0, 0});
            DataCopyPad(mmFeatureGM2_[copyOutOffset], gatherFeatureLocal2_[ping1_ * gatherBufLen_ * inChannelsAligned_],
                {static_cast<uint16_t>(copyOutTaskCount), static_cast<uint32_t>(inChannels_ * byteSizePerElements_), 0, 0, 0});
        } else {
            DataCopyPad(mmFeatureGM1_[copyOutOffset], gatherFeatureLocal1_[ping1_ * gatherBufLen_ * inChannelsAligned_],
                {static_cast<uint16_t>(2), static_cast<uint32_t>(copyOutTaskCount * inChannels_ * byteSizePerElements_),
                static_cast<uint32_t>((2 * gatherBufLen_ - copyOutTaskCount) * inChannels_ / (BYTE_SIZE_PER_BLOCK / byteSizePerElements_)),
                static_cast<uint32_t>((totalTaskCount_ - copyOutTaskCount) * inChannels_ * byteSizePerElements_), 0});
        }
    }

    __aicore__ inline void CopyInFeatures(int32_t copyInOffset, int32_t copyInTaskCount)
    {
        if (copyInTaskCount <= 0)
            return;
        
        if (outChannels_ != outChannelsAligned_) {
            DataCopyPad(scatterFeatureLocal1_[ping2_ * scatterBufLen_ * outChannelsAligned_], matmulResultGM1_[copyInOffset],
                {static_cast<uint16_t>(copyInTaskCount), static_cast<uint32_t>(outChannels_ * byteSizePerElements_), 0, 0, 0}, {false, 0, 0, 0});
            DataCopyPad(scatterFeatureLocal2_[ping2_ * scatterBufLen_ * outChannelsAligned_], matmulResultGM2_[copyInOffset],
                {static_cast<uint16_t>(copyInTaskCount), static_cast<uint32_t>(outChannels_ * byteSizePerElements_), 0, 0, 0}, {false, 0, 0, 0});
        } else {
            DataCopyPad(scatterFeatureLocal1_[ping2_ * scatterBufLen_ * outChannelsAligned_], matmulResultGM1_[copyInOffset],
                {static_cast<uint16_t>(2), static_cast<uint32_t>(copyInTaskCount * outChannels_ * byteSizePerElements_),
                static_cast<uint32_t>((totalTaskCount_ - copyInTaskCount) * outChannels_ * byteSizePerElements_),
                static_cast<uint32_t>(((2 * scatterBufLen_ - copyInTaskCount) * outChannels_) / (BYTE_SIZE_PER_BLOCK / byteSizePerElements_)), 0},
                {false, 0, 0, 0});
        }

        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);
    }

    __aicore__ inline void GatherFeature(const int32_t &taskOffset, const int32_t &taskCount, const int32_t &curPosValidTaskCount)
    {
        curLoopValidTask_ = 0;
        for (int32_t i = 0; i < taskCount; i+=INT64_BIT_SIZE) {
            uint64_t validIdxMask = maskForGatherLocal_.ReinterpretCast<uint64_t>().GetValue(i / INT64_BIT_SIZE);
            int32_t validIdx = ScalarGetSFFValue<0>(validIdxMask);
            int32_t curTaskIdx = validIdx + i;

            while(validIdx != -1 && (curTaskIdx < taskCount)) {
                int32_t mapVal1 = validIndicesForGatherFeatureLocal_.GetValue(curTaskIdx);
                int32_t mapVal2 = curTaskIdx + taskStartOffset_ + taskOffset;

                if (curLoopValidTask_ % gatherBufLen_ == 0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(ping1_);
                }
                DataCopyPad(gatherFeatureLocal1_[(ping1_ * gatherBufLen_ + curLoopValidTask_ % gatherBufLen_) * inChannelsAligned_], inputFeatureGM_[mapVal1 * inChannels_],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(inChannels_ * byteSizePerElements_), 0, 0, 0}, {false, 0, 0, 0});
                DataCopyPad(gatherFeatureLocal2_[(ping1_ * gatherBufLen_ + curLoopValidTask_ % gatherBufLen_) * inChannelsAligned_], inputFeatureGM_[mapVal2 * inChannels_],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(inChannels_ * byteSizePerElements_), 0, 0, 0}, {false, 0, 0, 0});

                curLoopValidTask_ += 1;
                if (curLoopValidTask_ % gatherBufLen_ == 0) {
                    int32_t copyOutOffset = (curPosValidTaskCount + curLoopValidTask_ - gatherBufLen_) * inChannels_;
                    CopyOutFeatures(copyOutOffset, gatherBufLen_);
                    SetFlag<HardEvent::MTE3_MTE2>(ping1_);
                    ping1_ = 1 - ping1_;
                }

                validIdxMask += (static_cast<uint64_t>(1) << validIdx);
                validIdx = ScalarGetSFFValue<0>(validIdxMask);
                curTaskIdx = validIdx + i;
            }
        }

        if (curLoopValidTask_ % gatherBufLen_ != 0) {
            int32_t copyOutTaskCount = curLoopValidTask_ % gatherBufLen_;
            int32_t copyOutOffset = (curPosValidTaskCount + AlignUp(curLoopValidTask_, gatherBufLen_) - gatherBufLen_) * inChannels_;
            CopyOutFeatures(copyOutOffset, copyOutTaskCount);
            SetFlag<HardEvent::MTE3_MTE2>(ping1_);
        }
    }

    __aicore__ inline void ProcessSparseMatmul(const int32_t &k)
    {
        matmulCount0_ = false;
        matmulCount1_ = false;
        int32_t curPosValidTaskCount = 0;
        for (int32_t taskOffset = 0; taskOffset < coreTaskCount_; taskOffset += stage2SingleLoopTask_) {
            uint32_t taskCount = min(stage2SingleLoopTask_, coreTaskCount_ - taskOffset);
            DataCopyPad(validIndicesForGatherFeatureLocal_, indicesOffsetGM_[k * totalTaskCount_ + taskStartOffset_ + taskOffset],
                {1, static_cast<uint32_t>(taskCount * INT32_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            CompareScalar(maskForGatherLocal_.ReinterpretCast<uint8_t>(), validIndicesForGatherFeatureLocal_,
                static_cast<int32_t>(-1), CMPMODE::EQ, stage2SingleLoopTaskAligned_);

            GatherFeature(taskOffset, taskCount, curPosValidTaskCount);

            ProcessCube(k, curLoopValidTask_, curPosValidTaskCount);
            
            curPosValidTaskCount += curLoopValidTask_;
        }

        if (matmulCount0_ != matmulCount1_) {
            mm0_.WaitIterateAll();
            matmulCount0_ -= 1;
        }

        ScatterAdd(k, curPosValidTaskCount);
        
        matmulCount0_ = 0;
        matmulCount1_ = 0;
    }

    __aicore__ inline void Process()
    {
        MatmulCenterPoint();
        if (!withKey_) {
            ComputeIndicesOffset();
        }
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);

        SetFlag<HardEvent::MTE3_MTE2>(0);
        SetFlag<HardEvent::MTE3_MTE2>(1);

        for (int32_t k = kernelSize_ / 2 + 1; k < kernelSize_; k++) {
            ProcessSparseMatmul(k);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(1);
    }

private:
    bool useTwolevelMap_;
    uint8_t ping1_ = 0, ping2_ = 0, ping3_ = 0;
    uint32_t blkIdx_, aicNum_;
    int32_t k0_, k1_, k2_, kernelSize_, batchSize_, inChannels_, outChannels_, spatialShape0_, spatialShape1_, byteSizePerElements_, totalTaskCount_, stage2SingleLoopTask_, withKey_,
        spatialShape2_, spatialShape0_times_1_, spatialShape1_times_2_, coreTaskCount_, stage2SingleLoopTaskAligned_, singleLoopTask_, inChannelsAligned_, outChannelsAligned_, kernelSizeAligned_,
        taskStartOffset_, singleLoopTaskAligned_, k1Aligned_, k2Aligned_, mapValBufSize_, gatherBufLen_, scatterBufLen_, availableUBSize_, curLoopValidTask_, preLoopValidTask_, matmulCount0_ = 0, matmulCount1_ = 0;
    int64_t totalSpatialShape_;

    GlobalTensor<T> inputFeatureGM_, outputFeatureGM_, weightGM_, mmFeatureGM1_, mmFeatureGM2_, matmulResultGM1_, matmulResultGM2_;
    GlobalTensor<int32_t> indicesGM_, map1GM_, map2GM_, indicesOffsetGM_;

    TBuf<TPosition::VECCALC> ubBuf_;

    LocalTensor<int32_t> inputIndicesLocal_, indicesOffsetLocal_, batchIdxLocal_, spatial0Local_, validIndicesForGatherFeatureLocal_,
        spatial1Local_, spatial2Local_, mapValLocal_, map1ValLocal_, maskForGatherLocal_, validIndicesForScatterFeatureLocal_, maskForScatterLocal_;
    LocalTensor<T> gatherFeatureLocal1_, gatherFeatureLocal2_, scatterFeatureLocal1_, scatterFeatureLocal2_;
    LocalTensor<uint32_t> gatherOffsetLocal_;
    TPipe* pipe_;
};
 
extern "C" __global__ __aicore__ void subm_sparse_conv3d_v3(GM_ADDR feature, GM_ADDR weight, GM_ADDR indices, GM_ADDR indices_offset, GM_ADDR map1, GM_ADDR map2,
                                                            GM_ADDR feature_out, GM_ADDR out_indices_offset, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }

    KernelSubmSparseConv3dV3<DTYPE_FEATURE> op;
    TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm0_, &(tiling_data.mm0TilingData), op.mm1_, &(tiling_data.mm1TilingData));
    op.Init(&pipe, feature, weight, indices, indices_offset, map1, map2, feature_out, out_indices_offset, &tiling_data, usrWorkspace);
    op.Process();
}