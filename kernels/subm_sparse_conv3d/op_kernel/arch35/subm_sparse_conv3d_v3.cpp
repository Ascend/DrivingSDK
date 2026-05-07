/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "basic_api/kernel_operator_intf.h"
#include "basic_api/kernel_tensor.h"
#include "basic_api/kernel_tpipe.h"

#include "simt_api/common_functions.h"
#include "lib/matmul_intf.h"
#include "subm_sparse_conv3d_v3.h"

using namespace AscendC;
 
namespace {
constexpr int64_t SPATIAL_SHAPE_THRESHOLD = 40000000;
constexpr int32_t INT32_BYTE_SIZE = 4;
constexpr int32_t FP32_BYTE_SIZE = 4;
constexpr int32_t INDICES_ELEMENTS_COUNT = 4;
constexpr int32_t REPEAT_BYTE_SIZE = 256;
constexpr uint8_t SRC_PARTTEN_0 = 3;
constexpr uint8_t SRC_PARTTEN_1 = 4;
constexpr uint8_t SRC_PARTTEN_2 = 5;
constexpr uint8_t SRC_PARTTEN_3 = 6;
constexpr int32_t NUM_TWO = 2;
constexpr int32_t INT64_BIT_SIZE = 64;
constexpr int32_t VECTOR_CORE_COUNT_PER_AI_CORE = 2;

constexpr MatmulConfig SUBM_SPARSE_CONV3D_CFG = GetMDLConfig(false, false, 0, true, false, false, true);
};


template<typename T>
class KernelSubmSparseConv3dV3 {
public:
   using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
   using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
   using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
   matmul::MatmulImpl<AType, BType, CType, CType, SUBM_SPARSE_CONV3D_CFG> mm_;

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
        outChannelsAligned_ = AlignUp(outChannels_, BYTE_SIZE_PER_BLOCK / FP32_BYTE_SIZE);
        singleLoopTaskAligned_ = AlignUp(singleLoopTask_, BYTE_SIZE_PER_BLOCK / INT32_BYTE_SIZE);
        k2Aligned_ = AlignUp(k2_, BYTE_SIZE_PER_BLOCK / INT32_BYTE_SIZE);
        k1Aligned_ = AlignUp(k1_, BYTE_SIZE_PER_BLOCK / INT32_BYTE_SIZE);
        mapValBufSize_ = AlignUp(k0_ * k1_ * k2Aligned_, BYTE_SIZE_PER_BLOCK / INT32_BYTE_SIZE);
        totalTaskCountAligned_ = AlignUp(totalTaskCount_, VECTOR_CORE_COUNT_PER_AI_CORE * singleLoopTask_);

        if ASCEND_IS_AIC {
            // 需要计算 center matmul 阶段，每个 aic 需要计算的任务量
            // blkIdx_: aic 的 索引
            int32_t aicTaskCount = totalTaskCount_ / aicNum_;
            int32_t aicBigCoreCount = totalTaskCount_ % aicNum_;
            if (blkIdx_ < aicBigCoreCount) {
                taskStartOffset_ = (aicTaskCount + 1) * blkIdx_;
                coreTaskCount_ = aicTaskCount + 1;
            } else {
                taskStartOffset_ = (aicTaskCount + 1) * aicBigCoreCount +
                                    aicTaskCount * (blkIdx_ - aicBigCoreCount);
                coreTaskCount_ = aicTaskCount;
            }
        }
   }

    __aicore__ inline void InitGM(GM_ADDR feature, GM_ADDR weight, GM_ADDR indices, GM_ADDR indices_offset, GM_ADDR map1, GM_ADDR map2,
        GM_ADDR feature_out, GM_ADDR out_indices_offset, GM_ADDR workspace)
    {
        inputFeatureGM_.SetGlobalBuffer((__gm__ T*) feature, totalTaskCount_ * inChannels_);
        weightGM_.SetGlobalBuffer((__gm__ T*) weight);
        indicesGM_.SetGlobalBuffer((__gm__ int32_t*) indices);
        indicesGM_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        if (withKey_) {
            indicesOffsetGM_.SetGlobalBuffer((__gm__ int32_t*) indices_offset);
        } else {
            map1GM_.SetGlobalBuffer((__gm__ int32_t*) map1);
            if (useTwolevelMap_) {
                map2GM_.SetGlobalBuffer((__gm__ int32_t*) map2);
            }
            indicesOffsetGM_.SetGlobalBuffer((__gm__ int32_t*) out_indices_offset);
        }
        
        outputFeatureGM_.SetGlobalBuffer((__gm__ float*) feature_out);
        mmFeatureGM1_.SetGlobalBuffer(((__gm__ T*) workspace));
        mmFeatureGM2_ = mmFeatureGM1_[totalTaskCount_ * inChannels_];

        matmulResultGM1_.SetGlobalBuffer((__gm__ float*) (workspace + 2 * totalTaskCount_ * inChannels_ * byteSizePerElements_));
        matmulResultGM2_ = matmulResultGM1_[totalTaskCount_ * outChannels_];
        validTaskCountGM_ = matmulResultGM2_[totalTaskCount_ * outChannels_].template ReinterpretCast<int32_t>();
    }

    __aicore__ inline void InitUB()
    {
        if ASCEND_IS_AIV {
            pipe_->InitBuffer(ubBuf_, availableUBSize_);

            reOrderedIndicesLocal_ = ubBuf_.Get<int32_t>();
            validTaskCountLocal_ = reOrderedIndicesLocal_[singleLoopTaskAligned_];
            validIndicesForGatherFeatureLocal1_ = validTaskCountLocal_[8];
            validIndicesForGatherFeatureLocal2_ = validIndicesForGatherFeatureLocal1_[singleLoopTaskAligned_];

            scatterFeatureLocal1_ = validIndicesForGatherFeatureLocal2_[singleLoopTaskAligned_].template ReinterpretCast<float>();
            scatterFeatureLocal2_ = scatterFeatureLocal1_[2 * scatterBufLen_ * outChannelsAligned_];

            gatherFeatureLocal1_ = validIndicesForGatherFeatureLocal2_[singleLoopTaskAligned_].template ReinterpretCast<T>();
            gatherFeatureLocal2_ = gatherFeatureLocal1_[2 * gatherBufLen_ * inChannelsAligned_];
        }
    }

    __aicore__ inline void Init(TPipe *pipe, GM_ADDR feature, GM_ADDR weight, GM_ADDR indices, GM_ADDR indices_offset, GM_ADDR map1, GM_ADDR map2,
        GM_ADDR feature_out, GM_ADDR out_indices_offset, SubmSparseConv3dV3TilingData *tilingData, GM_ADDR workspace)
    {
        pipe_ = pipe;
        aicNum_ = GetBlockNum();
        aivNum_ = aicNum_ * VECTOR_CORE_COUNT_PER_AI_CORE;
        blkIdx_ = GetBlockIdx();

        InitTiling(tilingData);
        InitGM(feature, weight, indices, indices_offset, map1, map2, feature_out, out_indices_offset, workspace);
        InitUB();
    }

    __aicore__ inline void GatherFeature(int32_t taskOffset, int32_t vfTaskCount)
    {
        if (vfTaskCount <= 0) {
            return;
        }

        __ubuf__ int32_t* validIndices1Local = (__ubuf__ int32_t*) validIndicesForGatherFeatureLocal1_.GetPhyAddr();
        __ubuf__ int32_t* validIndices2Local = (__ubuf__ int32_t*) validIndicesForGatherFeatureLocal2_.GetPhyAddr();
        
        int8_t ping = 0;
        SetFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::MTE3_V>(1);
        for (int32_t i = 0; i < vfTaskCount; i += gatherBufLen_) {
            int32_t processTaskCountCurLoop = min(vfTaskCount - i, gatherBufLen_);

            WaitFlag<HardEvent::MTE3_V>(ping);
            if (inChannels_ % (BYTE_SIZE_PER_BLOCK / byteSizePerElements_) == 0) {
                int32_t elementsCountPerVectorDtype = std::is_same<T, float>::value ? 4 : 8;
                int32_t vectorDtypeCountPerChannel = inChannels_ / elementsCountPerVectorDtype;
                uint32_t threadNum = min(inChannels_ * processTaskCountCurLoop / elementsCountPerVectorDtype, THREAD_NUM);
                uint32_t xThreadNum = inChannels_ / elementsCountPerVectorDtype;
                uint32_t yThreadNum = threadNum / xThreadNum;
                using FEATURE_DTYPE = float4;
                __gm__ volatile FEATURE_DTYPE* inputFeatureGlobal = (__gm__  volatile FEATURE_DTYPE*) inputFeatureGM_.GetPhyAddr();
                __ubuf__ FEATURE_DTYPE* gatheredFeature1Local = (__ubuf__ FEATURE_DTYPE*) gatherFeatureLocal1_[ping * gatherBufLen_ * inChannelsAligned_].GetPhyAddr();
                __ubuf__ FEATURE_DTYPE* gatheredFeature2Local = (__ubuf__ FEATURE_DTYPE*) gatherFeatureLocal2_[ping * gatherBufLen_ * inChannelsAligned_].GetPhyAddr();
                Simt::VF_CALL<GatherFeatureVectorDTypeSimt<T, FEATURE_DTYPE>>(Simt::Dim3{xThreadNum, yThreadNum}, validIndices1Local + i, validIndices2Local + i, inputFeatureGlobal,
                    gatheredFeature1Local, gatheredFeature2Local, vectorDtypeCountPerChannel, processTaskCountCurLoop, taskOffset);
            } else {
                uint32_t threadNum = min(inChannels_ * processTaskCountCurLoop, THREAD_NUM);
                __gm__ volatile T* inputFeatureGlobal = (__gm__ volatile T*) inputFeatureGM_.GetPhyAddr();
                __ubuf__ T* gatheredFeature1Local = (__ubuf__ T*) gatherFeatureLocal1_[ping * gatherBufLen_ * inChannelsAligned_].GetPhyAddr();
                __ubuf__ T* gatheredFeature2Local = (__ubuf__ T*) gatherFeatureLocal2_[ping * gatherBufLen_ * inChannelsAligned_].GetPhyAddr();
                Simt::VF_CALL<GatherFeatureSimt<T>>(Simt::Dim3{threadNum}, validIndices1Local + i, validIndices2Local + i, inputFeatureGlobal,
                    gatheredFeature1Local, gatheredFeature2Local, inChannels_, processTaskCountCurLoop, taskOffset);
            }
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);

            DataCopyPad(mmFeatureGM1_[(taskOffset + i) * inChannels_], gatherFeatureLocal1_[ping * gatherBufLen_ * inChannelsAligned_],
                {static_cast<uint16_t>(processTaskCountCurLoop), static_cast<uint32_t>(inChannels_  * byteSizePerElements_), 0, 0, 0});
            DataCopyPad(mmFeatureGM2_[(taskOffset + i) * inChannels_], gatherFeatureLocal2_[ping * gatherBufLen_ * inChannelsAligned_],
                {static_cast<uint16_t>(processTaskCountCurLoop), static_cast<uint32_t>(inChannels_  * byteSizePerElements_), 0, 0, 0});

            SetFlag<HardEvent::MTE3_V>(ping);
            ping = 1 - ping;
        }
        WaitFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(1);
    }

    __aicore__ inline void ScatterFeature(const int32_t &taskOffset, const int32_t &vfTaskCount)
    {
        if (vfTaskCount <= 0) {
            return;
        }

        __ubuf__ int32_t* validIndices1Local = (__ubuf__ int32_t*) validIndicesForGatherFeatureLocal1_.GetPhyAddr();
        __ubuf__ int32_t* validIndices2Local = (__ubuf__ int32_t*) validIndicesForGatherFeatureLocal2_.GetPhyAddr();
        __gm__ float* outputFeatureGlobal = (__gm__ float*) outputFeatureGM_.GetPhyAddr();

        uint32_t xThreadCount = outChannels_ / 2;
        uint32_t yThreadCount = THREAD_NUM / xThreadCount;

        int8_t ping = 0;
        SetFlag<HardEvent::V_MTE2>(0);
        SetFlag<HardEvent::V_MTE2>(1);
        for (int32_t i = 0; i < vfTaskCount; i += scatterBufLen_) {
            const int32_t processTaskCountCurLoop = min(vfTaskCount - i, scatterBufLen_);

            __ubuf__ float* matmulRes1Local = (__ubuf__ float*) scatterFeatureLocal1_[ping * scatterBufLen_ * outChannelsAligned_].GetPhyAddr();
            __ubuf__ float* matmulRes2Local = (__ubuf__ float*) scatterFeatureLocal2_[ping * scatterBufLen_ * outChannelsAligned_].GetPhyAddr();

            WaitFlag<HardEvent::V_MTE2>(ping);
            DataCopyPad(scatterFeatureLocal1_[ping * scatterBufLen_ * outChannelsAligned_], matmulResultGM1_[(taskOffset + i) * outChannels_],
                {static_cast<uint16_t>(processTaskCountCurLoop), static_cast<uint32_t>(outChannels_ * FP32_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});
            DataCopyPad(scatterFeatureLocal2_[ping * scatterBufLen_ * outChannelsAligned_], matmulResultGM2_[(taskOffset + i) * outChannels_],
                {static_cast<uint16_t>(processTaskCountCurLoop), static_cast<uint32_t>(outChannels_ * FP32_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});

            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);
            Simt::VF_CALL<ScatterFeatureSimt<T>>(Simt::Dim3{xThreadCount, yThreadCount}, validIndices1Local + i, validIndices2Local + i, outputFeatureGlobal,
                matmulRes1Local, matmulRes2Local, outChannels_, processTaskCountCurLoop, taskOffset);
            SetFlag<HardEvent::V_MTE2>(ping);
            ping = 1 - ping;
        }
        WaitFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(1);
    }

    __aicore__ inline void ComputeIndicesOffset()
    {
        if ASCEND_IS_AIV {
            if (useTwolevelMap_) {
                Simt::VF_CALL<CopyInIndicesTwoMapNoKeySimt>(Simt::Dim3{THREAD_NUM, 1, 1},
                    (__gm__ volatile int32_t*) indicesOffsetGM_.GetPhyAddr(), (__gm__ volatile int4*) indicesGM_.GetPhyAddr(),
                    (__gm__ volatile int32_t*) map1GM_.GetPhyAddr(), (__gm__ volatile int32_t*) map2GM_.GetPhyAddr(),
                    blkIdx_, aivNum_, totalTaskCount_, spatialShape0_, spatialShape1_, spatialShape2_, k0_, k1_, k2_);
            } else {
                Simt::VF_CALL<CopyInIndicesOneMapNoKeySimt>(Simt::Dim3{THREAD_NUM, 1, 1},
                    (__gm__ volatile int32_t*) indicesOffsetGM_.GetPhyAddr(), (__gm__ volatile int4*) indicesGM_.GetPhyAddr(),
                    (__gm__ volatile int32_t*) map1GM_.GetPhyAddr(), blkIdx_, aivNum_, totalTaskCount_, spatialShape0_,
                    spatialShape1_, spatialShape2_, k0_, k1_, k2_);
            }

            DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(indicesOffsetGM_);
            CrossCoreSetFlag<0x0, PIPE_V>(0x8);
            CrossCoreWaitFlag<0x0>(0x8);
        }
    }

    __aicore__ inline void ProcessCube(const int32_t &k, const int32_t &taskOffset, const int32_t &taskCount)
    {
        if (taskCount <= 0) {
            return ;
        }

        int16_t k1 = kernelSize_ - k - 1;

        if (byteSizePerElements_ == FP32_BYTE_SIZE) {
            mm_.SetHF32(true, 1);
        }

        mm_.SetTensorA(mmFeatureGM1_[taskOffset * inChannels_]);
        mm_.SetTensorB(weightGM_[k * inChannels_ * outChannels_]);
        mm_.SetSingleShape(taskCount, outChannels_, inChannels_);
        mm_.template IterateAll<false>(matmulResultGM1_[taskOffset * outChannels_], 0, false, true);
        mm_.End();
        
        if (byteSizePerElements_ == FP32_BYTE_SIZE) {
            mm_.SetHF32(true, 1);
        }
        mm_.SetTensorA(mmFeatureGM2_[taskOffset * inChannels_]);
        mm_.SetTensorB(weightGM_[k1 * inChannels_ * outChannels_]);
        mm_.SetSingleShape(taskCount, outChannels_, inChannels_);
        mm_.template IterateAll<false>(matmulResultGM2_[taskOffset * outChannels_], 0, false, true);
        mm_.End();
    }

    __aicore__ inline void CenterPositionMatmul()
    {
        if ASCEND_IS_AIC {
            if (coreTaskCount_ <= 0) {
                return ;
            }

            mm_.SetTensorA(inputFeatureGM_[taskStartOffset_ * inChannels_]);
            mm_.SetTensorB(weightGM_[kernelSize_ / NUM_TWO * inChannels_ * outChannels_]);
            if (byteSizePerElements_ == FP32_BYTE_SIZE) {
                mm_.SetHF32(true, 1);
            }
            mm_.SetSingleShape(coreTaskCount_, outChannels_, inChannels_);
            mm_.template IterateAll<false>(outputFeatureGM_[taskStartOffset_ * outChannels_], 1);
            mm_.End();
        }
    }

    __aicore__ inline void CopyInIndicesAndSort(int32_t k, int32_t taskOffset, int32_t taskCount)
    {
        if (taskCount <= 0) {
            return;
        }

        DataCopyPad(reOrderedIndicesLocal_, indicesOffsetGM_[k * totalTaskCount_ + taskOffset],
            {1, static_cast<uint32_t>(taskCount * INT32_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        static constexpr SortConfig config = {SortType::RADIX_SORT, true};
        LocalTensor<uint32_t> tmpLocal = validIndicesForGatherFeatureLocal2_.template ReinterpretCast<uint32_t>();
        Sort<int32_t, true, config>(validIndicesForGatherFeatureLocal1_, tmpLocal, reOrderedIndicesLocal_, taskCount);
    }

    __aicore__ inline void ComputeValidPointCount(int32_t k, int32_t &validTaskCount, int32_t taskCount)
    {
        // taskCount 个元素需要256Byte对齐
        if (taskCount <= 0) {
            return;
        }
            
        int32_t taskCountAligned = AlignUp(taskCount, 256 / INT32_BYTE_SIZE);
        LocalTensor<uint8_t> validMaskLocal = reOrderedIndicesLocal_.ReinterpretCast<uint8_t>();
        CompareScalar(validMaskLocal, validIndicesForGatherFeatureLocal1_, static_cast<int32_t>(-1), CMPMODE::NE, taskCountAligned);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        validTaskCount = 0;
        for (int32_t i = 0; i < taskCount; i += INT64_BIT_SIZE) {
            uint64_t validBit = min(INT64_BIT_SIZE, taskCount - i);
            uint64_t validMask = ((uint64_t)(1) << validBit) - 1;
            validMask = validBit == INT64_BIT_SIZE? UINT64_MAX : validMask;
            
            uint64_t curValidMask = validMaskLocal.ReinterpretCast<uint64_t>().GetValue(i / 64);
            curValidMask = curValidMask & validMask;

            validTaskCount += ScalarGetCountOfValue<1>(curValidMask);
        }
    }

    __aicore__ inline void ProcessSparseMatmul()
    {
        if ASCEND_IS_AIV {
            int32_t aivTaskStartOffset = (blkIdx_ / 2) * (2 * singleLoopTask_);
            for (int32_t k = 0; k < kernelSize_ / 2; k++) {
                int8_t ping = 0;
                SetFlag<HardEvent::MTE3_S>(0);
                for (int32_t taskOffset = aivTaskStartOffset; taskOffset < totalTaskCount_; taskOffset += singleLoopTask_ * aivNum_) {
                    int32_t aivTaskOffset = taskOffset + (blkIdx_ % 2) * singleLoopTask_;
                    int32_t aivTaskCount = min(singleLoopTask_, totalTaskCount_ - aivTaskOffset);
                    int32_t validTaskCount = 0;
                    CopyInIndicesAndSort(k, aivTaskOffset, aivTaskCount);
                    ComputeValidPointCount(k, validTaskCount, aivTaskCount);
                    WaitFlag<HardEvent::MTE3_S>(0);
                    validTaskCountLocal_.SetValue(0, validTaskCount);
                    SetFlag<HardEvent::S_MTE3>(0);
                    GatherFeature(aivTaskOffset, validTaskCount);
                    WaitFlag<HardEvent::S_MTE3>(0);
                    DataCopyPad(validTaskCountGM_[aivTaskOffset + k * totalTaskCountAligned_], validTaskCountLocal_,
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(1  * INT32_BYTE_SIZE), 0, 0, 0});
                    SetFlag<HardEvent::MTE3_S>(0);
                    CrossCoreSetFlag<0x2, PIPE_MTE3>(ping);
                    CrossCoreSetFlag<0x2, PIPE_V>(ping);
                    ping = (ping + 1) % 10;
                }
                WaitFlag<HardEvent::MTE3_S>(0);
                
                ping = 0;
                for (int32_t taskOffset = aivTaskStartOffset; taskOffset < totalTaskCount_; taskOffset += singleLoopTask_ * aivNum_) {
                    int32_t aivTaskOffset = taskOffset + (blkIdx_ % 2) * singleLoopTask_;
                    int32_t aivTaskCount = min(singleLoopTask_, totalTaskCount_ - aivTaskOffset);
                    int32_t validTaskCount = 0;
                    CopyInIndicesAndSort(k, aivTaskOffset, aivTaskCount);
                    ComputeValidPointCount(k, validTaskCount, aivTaskCount);
                    CrossCoreWaitFlag<0x2>(ping);
                    ScatterFeature(aivTaskOffset, validTaskCount);
                    ping = (ping + 1) % 10;
                }
            }
        }

        if ASCEND_IS_AIC {
            int32_t aicTaskStartOffset = blkIdx_ * (2 * singleLoopTask_);
            for (int32_t k = 0; k < kernelSize_ / 2; k++) {
                int8_t ping = 0;
                for (int32_t aicTaskOffset = aicTaskStartOffset; aicTaskOffset < totalTaskCount_; aicTaskOffset += (2 * singleLoopTask_) * aicNum_) {
                    CrossCoreWaitFlag<0x2>(ping);
                    CrossCoreWaitFlag<0x2>(ping);
                    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(validTaskCountGM_[aicTaskOffset + k * totalTaskCountAligned_]);
                    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(validTaskCountGM_[aicTaskOffset + k * totalTaskCountAligned_ + singleLoopTask_]);
                    int32_t validTaskCount1 = validTaskCountGM_.GetValue(k * totalTaskCountAligned_ + aicTaskOffset);
                    int32_t validTaskCount2 = validTaskCountGM_.GetValue(k * totalTaskCountAligned_ + aicTaskOffset + singleLoopTask_);

                    ProcessCube(k, aicTaskOffset, validTaskCount1);
                    ProcessCube(k, aicTaskOffset + singleLoopTask_, validTaskCount2);
                    CrossCoreSetFlag<0x2, PIPE_FIX>(ping);
                    ping = (ping + 1) % 10;
                }
            }
        }
    }

    __aicore__ inline void Process()
    {
        if (!withKey_) {
            ComputeIndicesOffset();
        }
        CenterPositionMatmul();
        ProcessSparseMatmul();
    }

private:
    bool useTwolevelMap_;
    uint8_t ping1_ = 0, ping2_ = 0, ping3_ = 0;
    uint32_t blkIdx_, aicNum_, aivNum_;
    int32_t k0_, k1_, k2_, kernelSize_, batchSize_, inChannels_, outChannels_, spatialShape0_, spatialShape1_, byteSizePerElements_, totalTaskCount_, totalTaskCountAligned_, stage2SingleLoopTask_, withKey_,
        spatialShape2_, spatialShape0_times_1_, spatialShape1_times_2_, coreTaskCount_, stage2SingleLoopTaskAligned_, singleLoopTask_, inChannelsAligned_, outChannelsAligned_, kernelSizeAligned_,
        taskStartOffset_, singleLoopTaskAligned_, k1Aligned_, k2Aligned_, mapValBufSize_, gatherBufLen_, scatterBufLen_, availableUBSize_, curLoopValidTask_, preLoopValidTask_, matmulCount0_ = 0, matmulCount1_ = 0;
    int64_t totalSpatialShape_;

    GlobalTensor<float> matmulResultGM1_, matmulResultGM2_;
    GlobalTensor<T> inputFeatureGM_, weightGM_, mmFeatureGM1_, mmFeatureGM2_;
    GlobalTensor<float> outputFeatureGM_;
    GlobalTensor<int32_t> indicesGM_, map1GM_, map2GM_, indicesOffsetGM_, validTaskCountGM_;

    TBuf<TPosition::VECCALC> ubBuf_, ubBuf1_;

    LocalTensor<int32_t> validTaskCountLocal_, validTaskCountLL1Local_;
    LocalTensor<int32_t> inputIndicesLocal_, indicesOffsetLocal_, batchIdxLocal_, spatial0Local_, validIndicesForGatherFeatureLocal1_, validIndicesForGatherFeatureLocal2_, reOrderedIndicesLocal_,
        spatial1Local_, spatial2Local_, mapValLocal_, map1ValLocal_, maskForGatherLocal_, validIndicesForScatterFeatureLocal_, maskForScatterLocal_;
    LocalTensor<T> gatherFeatureLocal1_, gatherFeatureLocal2_;
    LocalTensor<float> scatterFeatureLocal1_, scatterFeatureLocal2_;
    LocalTensor<uint32_t> gatherOffsetLocal_;
    TPipe* pipe_;
};
 
extern "C" __global__ __aicore__ void subm_sparse_conv3d_v3(GM_ADDR feature, GM_ADDR weight, GM_ADDR indices, GM_ADDR indices_offset, GM_ADDR map1, GM_ADDR map2,
                                                            GM_ADDR feature_out, GM_ADDR out_indices_offset, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    KernelSubmSparseConv3dV3<DTYPE_FEATURE> op;
    TPipe pipe;

    op.mm_.SetSubBlockIdx(0);
    op.mm_.Init(&tilingData.mmTilingData, &pipe);

    op.Init(&pipe, feature, weight, indices, indices_offset, map1, map2, feature_out, out_indices_offset, &tilingData, usrWorkspace);
    op.Process();
}