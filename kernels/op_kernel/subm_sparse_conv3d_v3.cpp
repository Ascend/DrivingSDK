/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
 
namespace {
constexpr int64_t SPATIAL_SHAPE_THRESHOLD = 400000000;
constexpr int32_t INT32_BYTE_SIZE = 4;
constexpr int32_t REPEAT_BYTE_SIZE = 256;
constexpr uint8_t SRC_PARTTEN_0 = 3;
constexpr uint8_t SRC_PARTTEN_1 = 4;
constexpr uint8_t SRC_PARTTEN_2 = 5;
constexpr uint8_t SRC_PARTTEN_3 = 6;
constexpr int32_t BYTE_SIZE_PER_BLOCK = 32;
constexpr int32_t NUM_TWO = 2;
constexpr MatmulConfig SUBM_SPARSE_CONV3D_CFG = GetIBShareNormConfig();
};


template<typename T>
class KernelSubmSparseConv3dV3 {
public:
    using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    matmul::Matmul<AType, BType, CType, CType, SUBM_SPARSE_CONV3D_CFG> mm0_;
    
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
        inputBufferLen_ = tilingData->featureBufLen;
        outputBufferLen_ = tilingData->featureBufLen;
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
        mmFeatureGM1_.SetGlobalBuffer(((__gm__ T*) workspace) + (blkIdx_ * stage2SingleLoopTask_ * inChannels_));
        mmFeatureGM2_ = mmFeatureGM1_[(NUM_TWO * aicNum_ * stage2SingleLoopTask_ * inChannels_)];

        matmulResultGM1_.SetGlobalBuffer(((__gm__ T*) workspace) + NUM_TWO * (NUM_TWO * aicNum_ * stage2SingleLoopTask_ * inChannels_) +
            (blkIdx_ * stage2SingleLoopTask_ * outChannels_));
        matmulResultGM2_ = matmulResultGM1_[(NUM_TWO * aicNum_ * stage2SingleLoopTask_ * outChannels_)];
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
        featureQueLocal1_ = ubBuf_.Get<T>();
        featureQueLocal2_ = featureQueLocal1_[inputBufferLen_ * inChannelsAligned_];
        outFeatureQueLocal1_ = featureQueLocal2_[inputBufferLen_ * inChannelsAligned_];
        outFeatureQueLocal2_ = outFeatureQueLocal1_[outputBufferLen_ * outChannelsAligned_];
        validIndicesLocal_ = outFeatureQueLocal2_[outputBufferLen_ * outChannelsAligned_].template ReinterpretCast<int32_t>();
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

    __aicore__ inline void ProcessCube(const int16_t &k, const uint32_t &taskCount)
    {
        int16_t k1 = kernelSize_ - k - 1;

        mm0_.SetTensorA(mmFeatureGM1_);
        mm0_.SetTensorB(weightGM_[k * inChannels_ * outChannels_]);
        mm0_.SetSingleShape(taskCount, outChannels_, inChannels_);
        
        mm0_.template IterateAll<true>(matmulResultGM1_);

        mm0_.SetTensorA(mmFeatureGM2_);
        mm0_.SetTensorB(weightGM_[k1 * inChannels_ * outChannels_]);
        mm0_.SetSingleShape(taskCount, outChannels_, inChannels_);
        
        mm0_.template IterateAll<true>(matmulResultGM2_);
    }

    __aicore__ inline void MatmulCenterPoint()
    {
        if (coreTaskCount_ <= 0) {
            return ;
        }

        mm0_.SetTensorA(inputFeatureGM_[taskStartOffset_ * inChannels_]);
        mm0_.SetTensorB(weightGM_[kernelSize_ / NUM_TWO * inChannels_ * outChannels_]);
        mm0_.SetSingleShape(coreTaskCount_, outChannels_, inChannels_);
        
        mm0_.template IterateAll<false>(outputFeatureGM_[taskStartOffset_ * outChannels_], 1);
    }

    __aicore__ inline void ScatterAdd(const int16_t &k, const int32_t &taskCount) {
        for (int32_t i = 0; i < taskCount; i += outputBufferLen_) {
            int32_t copyInTaskCount = min(outputBufferLen_, taskCount - i);
            
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
            
            if (outChannels_ != outChannelsAligned_) {
               // channels is not Aligned
                DataCopyPad(outFeatureQueLocal1_, matmulResultGM1_[i * outChannels_],
                {static_cast<uint16_t>(copyInTaskCount), static_cast<uint32_t>(outChannels_ * byteSizePerElements_), 0, 0, 0}, {false, 0, 0, 0});
                DataCopyPad(outFeatureQueLocal2_, matmulResultGM2_[i * outChannels_],
                {static_cast<uint16_t>(copyInTaskCount), static_cast<uint32_t>(outChannels_ * byteSizePerElements_), 0, 0, 0}, {false, 0, 0, 0});
            } else {
                DataCopyPad(outFeatureQueLocal1_, matmulResultGM1_[i * outChannels_],
                    {static_cast<uint16_t>(2), static_cast<uint32_t>(copyInTaskCount * outChannels_ * byteSizePerElements_),
                    static_cast<uint32_t>((2 * aicNum_ * stage2SingleLoopTask_ - copyInTaskCount) * outChannels_ * byteSizePerElements_),
                    static_cast<uint32_t>((outputBufferLen_ * outChannels_ - copyInTaskCount * outChannels_) / (BYTE_SIZE_PER_BLOCK / byteSizePerElements_)), 0},
                    {false, 0, 0, 0});
            }

            SetFlag<HardEvent::MTE2_MTE3>(0);
            WaitFlag<HardEvent::MTE2_MTE3>(0);

            SetAtomicAdd<T>();
            for (int j = 0; j < copyInTaskCount; j++) {
                int32_t taskOffset1 = validIndicesLocal_.GetValue((i + j));
                int32_t taskOffset2 = validIndicesLocal_.GetValue((i + j) + stage2SingleLoopTaskAligned_);

                DataCopyPad(outputFeatureGM_[taskOffset1 * outChannels_], outFeatureQueLocal1_[j * outChannelsAligned_],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(outChannels_ * byteSizePerElements_), 0, 0, 0});
                DataCopyPad(outputFeatureGM_[taskOffset2 * outChannels_], outFeatureQueLocal2_[j * outChannelsAligned_],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(outChannels_ * byteSizePerElements_), 0, 0, 0});
            }
            SetAtomicNone();
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
                    {true, 0, static_cast<uint8_t>(k2Aligned_ - k2_), -2});
            }
        }

        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        
        Gather(indicesOffsetLocal_, mapValLocal_, gatherOffsetLocal_, 0u, kernelSize_ * singleLoopTaskAligned_);
    }

    __aicore__ inline void ComputeIndicesOffset()
    {
        // compute offset
        for (int32_t taskOffset = 0; taskOffset < coreTaskCount_; taskOffset += singleLoopTask_)
        {
            uint32_t taskCount = min(singleLoopTask_, coreTaskCount_ - taskOffset);
            uint32_t taskCountAligned = AlignUp(taskCount, BYTE_SIZE_PER_BLOCK / INT32_BYTE_SIZE);

            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
            
            DataCopyPad(inputIndicesLocal_, indicesGM_[(taskStartOffset_ + taskOffset) * INT32_BYTE_SIZE],
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
        }
    }

    __aicore__ inline void CopyFeatures(int32_t copyOutOffset, int32_t copyOutTaskCount)
    {
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);
        
        if (inChannels_ != inChannelsAligned_) {
            DataCopyPad(mmFeatureGM1_[copyOutOffset], featureQueLocal1_,
                {static_cast<uint16_t>(copyOutTaskCount), static_cast<uint32_t>(inChannels_ * byteSizePerElements_), 0, 0, 0});
            DataCopyPad(mmFeatureGM2_[copyOutOffset], featureQueLocal2_,
                {static_cast<uint16_t>(copyOutTaskCount), static_cast<uint32_t>(inChannels_ * byteSizePerElements_), 0, 0, 0});
        } else {
            DataCopyPad(mmFeatureGM1_[copyOutOffset], featureQueLocal1_,
                {static_cast<uint16_t>(2), static_cast<uint32_t>(copyOutTaskCount * inChannels_ * byteSizePerElements_),
                static_cast<uint32_t>((inputBufferLen_ - copyOutTaskCount) * inChannels_ / (BYTE_SIZE_PER_BLOCK / byteSizePerElements_)),
                static_cast<uint32_t>((2 * aicNum_ * stage2SingleLoopTask_ - copyOutTaskCount) * inChannels_ * byteSizePerElements_), 0});
        }

        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }

    __aicore__ inline void ProcessSparseMatmul(int32_t k)
    {
        int32_t curLoopValidTask = 0;
        for (int32_t taskOffset = 0; taskOffset < coreTaskCount_; taskOffset += stage2SingleLoopTask_) {
            uint32_t taskCount = min(stage2SingleLoopTask_, coreTaskCount_ - taskOffset);
            curLoopValidTask = 0;

            DataCopyPad(validIndicesLocal_[stage2SingleLoopTaskAligned_], indicesOffsetGM_[k * totalTaskCount_ + taskStartOffset_ + taskOffset],
                {1, static_cast<uint32_t>(taskCount * INT32_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});

            for (int32_t i = 0; i < taskCount; i++) {
                int32_t mapVal1 = validIndicesLocal_.GetValue(stage2SingleLoopTaskAligned_ + i);

                if (mapVal1 < 0)
                    continue;
                
                int32_t mapVal2 = taskStartOffset_ + taskOffset + i;

                DataCopyPad(featureQueLocal1_[(curLoopValidTask % inputBufferLen_) * inChannelsAligned_], inputFeatureGM_[mapVal1 * inChannels_],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(inChannels_ * byteSizePerElements_), 0, 0, 0}, {false, 0, 0, 0});
                DataCopyPad(featureQueLocal2_[(curLoopValidTask % inputBufferLen_) * inChannelsAligned_], inputFeatureGM_[mapVal2 * inChannels_],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(inChannels_ * byteSizePerElements_), 0, 0, 0}, {false, 0, 0, 0});
                
                validIndicesLocal_.SetValue(curLoopValidTask + stage2SingleLoopTaskAligned_, mapVal1);
                validIndicesLocal_.SetValue(curLoopValidTask, mapVal2);

                curLoopValidTask += 1;
                if (curLoopValidTask % inputBufferLen_ == 0) {
                    int32_t copyOutOffset = (curLoopValidTask - inputBufferLen_) * inChannels_;
                    CopyFeatures(copyOutOffset, inputBufferLen_);
                }
            }

            // tail
            if (curLoopValidTask <= 0)
                continue;

            if (curLoopValidTask % inputBufferLen_ != 0) {
                int32_t copyOutTaskCount = curLoopValidTask % inputBufferLen_;
                int32_t copyOutOffset = (AlignUp(curLoopValidTask, inputBufferLen_) - inputBufferLen_) * inChannels_;
                CopyFeatures(copyOutOffset, copyOutTaskCount);
            }

            ProcessCube(k, curLoopValidTask);
            ScatterAdd(k, curLoopValidTask);
        }
    }

    __aicore__ inline void Process()
    {
        MatmulCenterPoint();
        if (!withKey_) {
            ComputeIndicesOffset();
        }
        
        PipeBarrier<PIPE_ALL>();

        for (int32_t k = kernelSize_ / 2 + 1; k < kernelSize_; k++) {
            ProcessSparseMatmul(k);
        }
    }

private:
    bool useTwolevelMap_;
    uint32_t blkIdx_, aicNum_;
    int32_t k0_, k1_, k2_, kernelSize_, batchSize_, inChannels_, outChannels_, spatialShape0_, spatialShape1_, byteSizePerElements_, totalTaskCount_, stage2SingleLoopTask_, withKey_,
        spatialShape2_, spatialShape0_times_1_, spatialShape1_times_2_, coreTaskCount_, stage2SingleLoopTaskAligned_, singleLoopTask_, inChannelsAligned_, outChannelsAligned_, kernelSizeAligned_,
        taskStartOffset_, singleLoopTaskAligned_, k1Aligned_, k2Aligned_, mapValBufSize_, inputBufferLen_, outputBufferLen_, availableUBSize_;
    int64_t totalSpatialShape_;

    GlobalTensor<T> inputFeatureGM_, outputFeatureGM_, weightGM_, mmFeatureGM1_, mmFeatureGM2_, matmulResultGM1_, matmulResultGM2_;
    GlobalTensor<int32_t> indicesGM_, map1GM_, map2GM_, indicesOffsetGM_;

    TBuf<TPosition::VECCALC> ubBuf_;

    LocalTensor<int32_t> inputIndicesLocal_, indicesOffsetLocal_, validIndicesLocal_, batchIdxLocal_, spatial0Local_,
        spatial1Local_, spatial2Local_, mapValLocal_, map1ValLocal_;
    LocalTensor<T> featureQueLocal1_, featureQueLocal2_, outFeatureQueLocal1_, outFeatureQueLocal2_;
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
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm0_, &(tiling_data.mm0TilingData));
    op.Init(&pipe, feature, weight, indices, indices_offset, map1, map2, feature_out, out_indices_offset, &tiling_data, usrWorkspace);
    op.Process();
}