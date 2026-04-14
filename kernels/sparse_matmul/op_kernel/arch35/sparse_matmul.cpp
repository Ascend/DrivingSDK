/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include "kernel_utils.h"
#include "kernel_operator.h"
#include "kernel_tpipe_impl.h"
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace MicroAPI;

namespace {
constexpr int32_t INT32_BYTE_SIZE = 4;
constexpr int32_t BYTE_SIZE_PER_BLOCK = 32;
constexpr MatmulConfig SPARSE_MATMUL_CFG = GetMDLConfig();
constexpr int32_t THREAD_NUM = 1024;
constexpr int32_t AIC_FLAG_RANGE = 10;
constexpr int32_t AIC_FLAG_OFFSET = 16;
constexpr int32_t AIC_AIV_RATIO = 2;
};


template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ZeroInitSimt(
    __gm__ T* img2colGM,
    int32_t GMSize
)
{
    for (int32_t i = AscendC::Simt::GetThreadIdx(); i < GMSize; i += AscendC::Simt::GetThreadNum()) {
        img2colGM[i] = 0;
    }
}


template<typename T>
class KernelSparseMatmul {
public:
    using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    matmul::MatmulImpl<AType, BType, CType, CType, SPARSE_MATMUL_CFG> mm0_;

    __aicore__ inline KernelSparseMatmul() {}
    __aicore__ inline void Init(GM_ADDR features, GM_ADDR weight, GM_ADDR unique_indices_offset, GM_ADDR former_sorted_indices,
        GM_ADDR indices, GM_ADDR sparse_value, GM_ADDR sparse_indices, GM_ADDR workspace, SparseMatmulTilingData *tilingData, TPipe *pipe)
    {
        pipe_ = pipe;
        blkIdx_ = GetBlockIdx();
        aicNum_ = GetBlockNum();
        aivNum_ = aicNum_ * AIC_AIV_RATIO;

        InitTiling(tilingData);
        InitGM(features, weight, unique_indices_offset, former_sorted_indices,
            indices, sparse_value, sparse_indices, workspace);
        InitUB();
    }

    __aicore__ inline void InitTiling(SparseMatmulTilingData *tilingData)
    {
        byteSizePerElements_ = sizeof(T);
        k0_ = tilingData->k0;
        k1_ = tilingData->k1;
        k2_ = tilingData->k2;
        kernelSize_ = k0_ * k1_ * k2_;
        inChannels_ = tilingData->inChannels;
        outChannels_ = tilingData->outChannels;
        availableUBSize_ = tilingData->availableUBSize;
        outputCoreTaskCount_ = tilingData->outputCoreTaskCount;
        outputBigCoreCount_ = tilingData->outputBigCoreCount;
        singleLoopTask_ = tilingData->singleLoopTask;
        outputTaskCount_ = tilingData->outputTaskCount;
        matmulTaskPerIter_ = tilingData->matmulTaskPerIter;
        GLOBAL_BUFFER_NUM = tilingData->globalBufferNum;

        if ASCEND_IS_AIC {
            aicTaskOffset_ = blkIdx_ * singleLoopTask_ * 2;
            taskStartOffset_ = blkIdx_ * singleLoopTask_ * 2;
        }
        if ASCEND_IS_AIV {
            aivTaskOffset_ = (blkIdx_ / 2) * (2 * singleLoopTask_);
            taskStartOffset_ = (blkIdx_ / 2) * (2 * singleLoopTask_);
        }

        singleLoopTaskAligned_ = AlignUp(singleLoopTask_, BYTE_SIZE_PER_BLOCK / INT32_BYTE_SIZE);
        singleLoopTaskPlusOneAligned_ = AlignUp(singleLoopTask_ + 1, BYTE_SIZE_PER_BLOCK / INT32_BYTE_SIZE);
        inChannelsAligned_ = AlignUp(inChannels_, BYTE_SIZE_PER_BLOCK / byteSizePerElements_);
        outChannelsAligned_ = AlignUp(outChannels_, BYTE_SIZE_PER_BLOCK / byteSizePerElements_);

        indicesBufSize_ = (1 + kernelSize_) * singleLoopTaskPlusOneAligned_;
        featuresBufSize_ = inChannelsAligned_ * singleLoopTask_;
        img2colSize_ = inChannels_ * kernelSize_ * singleLoopTask_;
    }
    
    __aicore__ inline void InitGM(GM_ADDR features, GM_ADDR weight, GM_ADDR unique_indices_offset, GM_ADDR former_sorted_indices,
        GM_ADDR indices, GM_ADDR sparse_value, GM_ADDR sparse_indices, GM_ADDR workspace)
    {
        inputFeatureGM_.SetGlobalBuffer((__gm__ T*) features);
        sparseValueGM_.SetGlobalBuffer((__gm__ T*) sparse_value);
        weightGM_.SetGlobalBuffer((__gm__ T*) weight);

        uniqueIndicesGM_.SetGlobalBuffer((__gm__ int32_t*) unique_indices_offset);
        sortedIndicesGM_.SetGlobalBuffer((__gm__ int32_t*) former_sorted_indices);
        indicesGM_.SetGlobalBuffer((__gm__ int32_t*) indices);
        sparseIndicesGM_.SetGlobalBuffer((__gm__ int32_t*) sparse_indices);

        if ASCEND_IS_AIC {
            img2colGM_.SetGlobalBuffer((__gm__ T*) workspace + static_cast<int64_t>(blkIdx_) * img2colSize_ * 2 * GLOBAL_BUFFER_NUM);
        }
        if ASCEND_IS_AIV {
            img2colGM_.SetGlobalBuffer((__gm__ T*) workspace + static_cast<int64_t>(blkIdx_) * img2colSize_ * GLOBAL_BUFFER_NUM);
            // aiv need zero init, fp16 need syncall to ensure precision
            Fill(img2colGM_, img2colSize_ * GLOBAL_BUFFER_NUM, (T)0.0f);
        }
        SyncAll<false>();
    }

    __aicore__ inline void InitUB()
    {
        pipe_->InitBuffer(indicesBuf_, indicesBufSize_ * INT32_BYTE_SIZE);
        pipe_->InitBuffer(featuresBuf_, featuresBufSize_ * byteSizePerElements_);

        uniqueIndicesLocal_ = indicesBuf_.Get<int32_t>();
        sortedIndicesLocal_ = uniqueIndicesLocal_[singleLoopTaskPlusOneAligned_];
        copyoutFeatureLocal_ = featuresBuf_.Get<T>();
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            AivProcess();
        }

        if ASCEND_IS_AIC {
            AicProcess();
        }
    }

    __aicore__ inline void AivProcess()
    {
        uint16_t flagIdx = 0;

        for (int32_t idx = taskStartOffset_; idx < outputTaskCount_; idx += GLOBAL_BUFFER_NUM * aivNum_ * singleLoopTask_) {
            GatherFeature(idx, flagIdx);
            ZeroInitGM(idx, flagIdx);

            PipeBarrier<PIPE_ALL>();

            flagIdx = (flagIdx + GLOBAL_BUFFER_NUM) % AIC_FLAG_RANGE;
        }
    }

    __aicore__ inline void GatherFeature(int32_t idx, uint16_t flagIdx)
    {
        for (int32_t globalBufIdx = 0; globalBufIdx < GLOBAL_BUFFER_NUM; globalBufIdx++) {
            int32_t taskOffset = idx + globalBufIdx * aivNum_ * singleLoopTask_ + (blkIdx_ % 2) * singleLoopTask_;
            int32_t taskCount = min(singleLoopTask_, outputTaskCount_ - taskOffset);
            if (taskCount <= 0) {
                continue;
            }

            // CopyIn UniqueIndices
            DataCopyPad(uniqueIndicesLocal_, uniqueIndicesGM_[taskOffset],
                {1, static_cast<uint32_t>((taskCount + 1) * INT32_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});

            SetFlag<HardEvent::MTE2_S>(0);
            WaitFlag<HardEvent::MTE2_S>(0);

            int32_t copyOffset = uniqueIndicesLocal_.GetValue(0);
            int32_t copyLength = uniqueIndicesLocal_.GetValue(taskCount) - copyOffset;

            DataCopyPad(sortedIndicesLocal_, sortedIndicesGM_[copyOffset],
                {1, static_cast<uint32_t>(copyLength * INT32_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});
            
            CopyFeatures(taskCount, img2colGM_[globalBufIdx * img2colSize_]);

            CrossCoreSetFlag<0x4, PIPE_MTE3>(flagIdx);

            flagIdx = (flagIdx + 1) % AIC_FLAG_RANGE;
        }
    }

    __aicore__ inline void ZeroInitGM(int32_t idx, uint16_t flagIdx)
    {
        for (int32_t globalBufIdx = 0; globalBufIdx < GLOBAL_BUFFER_NUM; globalBufIdx++) {
            int32_t taskOffset = idx + globalBufIdx * aivNum_ * singleLoopTask_ + (blkIdx_ % 2) * singleLoopTask_;
            int32_t taskCount = min(singleLoopTask_, outputTaskCount_ - taskOffset);
            if (taskCount <= 0) {
                continue;
            }

            CrossCoreWaitFlag<0x4>(flagIdx);

            AscendC::Simt::VF_CALL<ZeroInitSimt<T>>(
                AscendC::Simt::Dim3 {
                    THREAD_NUM
                },
                (__gm__ T*) img2colGM_[globalBufIdx * img2colSize_].GetPhyAddr(),
                taskCount * kernelSize_ * inChannels_
            );

            flagIdx = (flagIdx + 1) % AIC_FLAG_RANGE;
        }
    }

    __aicore__ inline void CopyFeatures(const int32_t &taskCount, GlobalTensor<T> img2colGM)
    {
        int32_t head = uniqueIndicesLocal_.GetValue(0);
        int32_t tail = uniqueIndicesLocal_.GetValue(taskCount);

        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
        int32_t featureBufIdx = 0;
        for (int32_t i = 0; i < taskCount; i++) {
            int32_t cur = uniqueIndicesLocal_.GetValue(i + 1);
            int32_t pre = uniqueIndicesLocal_.GetValue(i);
            int32_t length = cur - pre;
            int32_t start = pre - head;

            for (int32_t sortIndicesOffset = 0; sortIndicesOffset < length; sortIndicesOffset++) {
                int32_t sortIndicesVal = sortedIndicesLocal_.GetValue(sortIndicesOffset + start);

                DataCopyPad(copyoutFeatureLocal_[featureBufIdx * inChannelsAligned_], inputFeatureGM_[(sortIndicesVal / kernelSize_) * inChannels_],
                    {1, static_cast<uint32_t>(inChannels_ * byteSizePerElements_), 0, 0, 0}, {false, 0, 0, 0});

                SetFlag<HardEvent::MTE2_MTE3>(0);
                WaitFlag<HardEvent::MTE2_MTE3>(0);

                DataCopyPad(img2colGM[(i * kernelSize_ + sortIndicesVal % kernelSize_) * inChannels_], copyoutFeatureLocal_[featureBufIdx * inChannelsAligned_],
                    {1, static_cast<uint32_t>(inChannels_ * byteSizePerElements_), 0, 0, 0});

                featureBufIdx = (featureBufIdx + 1) % singleLoopTask_;
                if (featureBufIdx == 0) {
                    SetFlag<HardEvent::MTE3_MTE2>(0);
                    WaitFlag<HardEvent::MTE3_MTE2>(0);
                }
            }
        }
    }

    __aicore__ inline void AicProcess()
    {
        uint16_t flagIdx = 0;

        for (int32_t idx = taskStartOffset_; idx < outputTaskCount_; idx += GLOBAL_BUFFER_NUM * aivNum_ * singleLoopTask_) {
            ProcessMatmul(idx, flagIdx);

            flagIdx = (flagIdx + GLOBAL_BUFFER_NUM) % AIC_FLAG_RANGE;
        }
    }

    __aicore__ inline void ProcessMatmul(int32_t idx, uint16_t flagIdx)
    {
        for (int32_t globalBufIdx = 0; globalBufIdx < GLOBAL_BUFFER_NUM; globalBufIdx++) {
            int32_t taskOffset0 = idx + globalBufIdx * aivNum_ * singleLoopTask_;
            int32_t taskOffset1 = idx + globalBufIdx * aivNum_ * singleLoopTask_ + singleLoopTask_;
            int32_t taskAiv0 = min(singleLoopTask_, outputTaskCount_ - taskOffset0);
            int32_t taskAiv1 = min(singleLoopTask_, outputTaskCount_ - taskOffset1);

            Img2colMatmul(taskOffset0, taskAiv0, img2colGM_[globalBufIdx * img2colSize_], flagIdx);
            Img2colMatmul(taskOffset1, taskAiv1, img2colGM_[(GLOBAL_BUFFER_NUM + globalBufIdx) * img2colSize_], flagIdx + AIC_FLAG_OFFSET);

            flagIdx = (flagIdx + 1) % AIC_FLAG_RANGE;
        }
    }

    __aicore__ inline void Img2colMatmul(int32_t taskOffset, int32_t taskCount, GlobalTensor<T> img2colGM, uint16_t flagIdx)
    {
        if (taskCount <= 0) {
            return;
        }
        CrossCoreWaitFlag<0x4>(flagIdx);

        mm0_.SetTensorA(img2colGM);
        mm0_.SetTensorB(weightGM_);
        mm0_.SetSingleShape(taskCount, outChannels_, inChannels_ * kernelSize_);
        
        mm0_.template IterateAll<false>(sparseValueGM_[taskOffset * outChannels_]);
        mm0_.End();

        CrossCoreSetFlag<0x4, PIPE_FIX>(flagIdx);
    }

private:
    TPipe* pipe_;
    int32_t k0_, k1_, k2_, kernelSize_,
        blkIdx_, aivNum_, aicNum_, byteSizePerElements_,
        inChannels_, inChannelsAligned_, outChannels_, outChannelsAligned_,
        singleLoopTaskPlusOneAligned_, singleLoopTaskAligned_, singleLoopTask_,
        availableUBSize_, indicesBufSize_, featuresBufSize_,
        outputCoreTaskCount_, outputBigCoreCount_,
        outputTaskCount_, inputCoreTaskCount_,
        matmulTaskPerIter_,
        aivTaskOffset_, aicTaskOffset_, taskStartOffset_,
        GLOBAL_BUFFER_NUM, img2colSize_;

    GlobalTensor<T> inputFeatureGM_, sparseValueGM_, weightGM_, img2colGM_;
    GlobalTensor<int32_t> uniqueIndicesGM_, sortedIndicesGM_, indicesGM_, sparseIndicesGM_;
    LocalTensor<int32_t> uniqueIndicesLocal_, sortedIndicesLocal_;
    LocalTensor<T> copyInFeatureLocal_, copyoutFeatureLocal_;
    TBuf<TPosition::VECCALC> indicesBuf_, featuresBuf_;
};

extern "C" __global__ __aicore__ void sparse_matmul(GM_ADDR features, GM_ADDR weight, GM_ADDR unique_indices_offset, GM_ADDR former_sorted_indices,
    GM_ADDR indices, GM_ADDR sparse_value, GM_ADDR sparse_indices, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }
    TPipe pipe;
    KernelSparseMatmul<DTYPE_FEATURES> op;

    op.mm0_.SetSubBlockIdx(0);
    op.mm0_.Init(&(tiling_data.mm0TilingData), &pipe);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    op.Init(features, weight, unique_indices_offset, former_sorted_indices, indices, sparse_value, sparse_indices, workspace, &tiling_data, &pipe);
    op.Process();
}