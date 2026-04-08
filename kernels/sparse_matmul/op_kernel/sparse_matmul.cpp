#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;

namespace {
constexpr int32_t INT32_BYTE_SIZE = 4;
constexpr int32_t BYTE_SIZE_PER_BLOCK = 32;
constexpr MatmulConfig SPARSE_MATMUL_CFG = GetNormalConfig();
};

template<typename T>
class KernelSparseMatmul {
public:
    using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    matmul::Matmul<AType, BType, CType, CType, SPARSE_MATMUL_CFG> mm0_;

    __aicore__ inline KernelSparseMatmul() {}
    __aicore__ inline void Init(GM_ADDR features, GM_ADDR weight, GM_ADDR unique_indices_offset, GM_ADDR former_sorted_indices,
        GM_ADDR indices, GM_ADDR sparse_value, GM_ADDR sparse_indices, GM_ADDR workspace, SparseMatmulTilingData *tilingData, TPipe *pipe)
    {
        pipe_ = pipe;
        blkIdx_ = GetBlockIdx();
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
        aivNum_ = tilingData->aivNum;
        featureBufLen_ = tilingData->featureBufLen;
        outputCoreTaskCount_ = tilingData->outputCoreTaskCount;
        outputBigCoreCount_ = tilingData->outputBigCoreCount;
        outputSingleLoopTask_ = tilingData->outputSingleLoopTask;
        outputTaskCount_ = tilingData->outputTaskCount;
        matmulTaskPerIter_ = tilingData->matmulTaskPerIter;

        if (blkIdx_ < tilingData->outputBigCoreCount) {
            outputTaskStartOffset_ = (tilingData->outputCoreTaskCount + 1) * blkIdx_;
            outputCoreTaskCount_ = tilingData->outputCoreTaskCount + 1;
        } else {
            outputTaskStartOffset_ = (tilingData->outputCoreTaskCount + 1) * tilingData->outputBigCoreCount +
                                tilingData->outputCoreTaskCount * (blkIdx_ - tilingData->outputBigCoreCount);
            outputCoreTaskCount_ = tilingData->outputCoreTaskCount;
        }

        outputSingleLoopTaskAligned_ = AlignUp(outputSingleLoopTask_, BYTE_SIZE_PER_BLOCK / INT32_BYTE_SIZE);
        outputSingleLoopTaskPlusOneAligned_ = AlignUp(outputSingleLoopTask_ + 1, BYTE_SIZE_PER_BLOCK / INT32_BYTE_SIZE);
        inChannelsAligned_ = AlignUp(inChannels_, BYTE_SIZE_PER_BLOCK / byteSizePerElements_);
        outChannelsAligned_ = AlignUp(outChannels_, BYTE_SIZE_PER_BLOCK / byteSizePerElements_);
    }
    
    __aicore__ inline void InitGM(GM_ADDR features, GM_ADDR weight, GM_ADDR unique_indices_offset, GM_ADDR former_sorted_indices,
        GM_ADDR indices, GM_ADDR sparse_value, GM_ADDR sparse_indices, GM_ADDR workspace)
    {
        inputFeatureGM_.SetGlobalBuffer((__gm__ T*) features);
        sparseValueGM_.SetGlobalBuffer((__gm__ T*) sparse_value);
        weightGM_.SetGlobalBuffer((__gm__ T*) weight);

        img2colGM_.SetGlobalBuffer((__gm__ T*) workspace + static_cast<int64_t>(blkIdx_) * inChannels_ * kernelSize_ * matmulTaskPerIter_);
        uniqueIndicesGM_.SetGlobalBuffer((__gm__ int32_t*) unique_indices_offset);
        sortedIndicesGM_.SetGlobalBuffer((__gm__ int32_t*) former_sorted_indices);
        indicesGM_.SetGlobalBuffer((__gm__ int32_t*) indices);
        sparseIndicesGM_.SetGlobalBuffer((__gm__ int32_t*) sparse_indices);
    }

    __aicore__ inline void InitUB()
    {
        pipe_->InitBuffer(UbBuf_, availableUBSize_);

        uniqueIndicesLocal_ = UbBuf_.Get<int32_t>();
        sortedIndicesLocal_ = uniqueIndicesLocal_[outputSingleLoopTaskPlusOneAligned_];
        zerosLocal_ = sortedIndicesLocal_[outputSingleLoopTaskAligned_ * kernelSize_].template ReinterpretCast<T>();
        copyoutFeatureLocal_ = zerosLocal_[k2_ * inChannelsAligned_];

        Duplicate(zerosLocal_, static_cast<T>(0), k2_ * inChannelsAligned_);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process()
    {
        for (int32_t j = 0; j < matmulTaskPerIter_; j++) {
            for (int32_t k = 0; k < k0_ * k1_; k++) {
                DataCopyPad(img2colGM_[(j * kernelSize_ + k * k2_) * inChannels_], zerosLocal_,
                    {static_cast<uint16_t>(k2_), static_cast<uint32_t>(inChannels_ * byteSizePerElements_), 0, 0, 0});
            }
        }

        matmulIdx_ = 0;
        for (int32_t taskOffset = 0; taskOffset < outputCoreTaskCount_; taskOffset += outputSingleLoopTask_) {
            uint32_t taskCount = min(outputSingleLoopTask_, outputCoreTaskCount_ - taskOffset);

            // CopyIn UniqueIndices
            DataCopyPad(uniqueIndicesLocal_, uniqueIndicesGM_[(outputTaskStartOffset_ + taskOffset)],
                {1, static_cast<uint32_t>((taskCount + 1) * INT32_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});

            int32_t copyOffset = uniqueIndicesLocal_.GetValue(0);
            int32_t copyLength = uniqueIndicesLocal_.GetValue(taskCount) - copyOffset;

            DataCopyPad(sortedIndicesLocal_, sortedIndicesGM_[copyOffset],
                {1, static_cast<uint32_t>(copyLength * INT32_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});
            
            PipeBarrier<PIPE_ALL>();

            CopyFeatures(taskCount, taskOffset);
        }

        if (matmulIdx_ > 0) {
            Img2colMatmul(matmulIdx_, outputTaskStartOffset_ + outputCoreTaskCount_ - matmulIdx_);
        }
    }

    __aicore__ inline void CopyFeatures(const int32_t &taskCount, const int32_t &taskOffset)
    {
        // CopyOut zeros
        int32_t start = 0;
        int32_t featureBufIdx = 0;
        for (int32_t i = 0; i < taskCount; i++) {
            int32_t length = uniqueIndicesLocal_.GetValue(i + 1) - uniqueIndicesLocal_.GetValue(i);

            PipeBarrier<PIPE_MTE3>();

            for (int32_t sortIndicesOffset = 0; sortIndicesOffset < length; sortIndicesOffset++) {
                int32_t sortIndicesVal = sortedIndicesLocal_.GetValue(sortIndicesOffset + start);

                DataCopyPad(copyoutFeatureLocal_[featureBufIdx * inChannelsAligned_], inputFeatureGM_[(sortIndicesVal / kernelSize_) * inChannels_],
                    {1, static_cast<uint32_t>(inChannels_ * byteSizePerElements_), 0, 0, 0}, {false, 0, 0, 0});

                SetFlag<HardEvent::MTE2_MTE3>(0);
                WaitFlag<HardEvent::MTE2_MTE3>(0);

                DataCopyPad(img2colGM_[(matmulIdx_ * kernelSize_ + sortIndicesVal % kernelSize_) * inChannels_], copyoutFeatureLocal_[featureBufIdx * inChannelsAligned_],
                    {1, static_cast<uint32_t>(inChannels_ * byteSizePerElements_), 0, 0, 0});
                
                featureBufIdx = (featureBufIdx + 1) % featureBufLen_;
            }
            matmulIdx_ += 1;
            start += length;

            if (matmulIdx_ == matmulTaskPerIter_) {
                Img2colMatmul(matmulTaskPerIter_, outputTaskStartOffset_ + taskOffset + i + 1 - matmulTaskPerIter_);
                matmulIdx_ = 0;

                for (int32_t j = 0; j < matmulTaskPerIter_; j++) {
                    for (int32_t k = 0; k < k0_ * k1_; k++) {
                        DataCopyPad(img2colGM_[(j * kernelSize_ + k * k2_) * inChannels_], zerosLocal_,
                            {static_cast<uint16_t>(k2_), static_cast<uint32_t>(inChannels_ * byteSizePerElements_), 0, 0, 0});
                    }
                }
            }
        }
    }

    __aicore__ inline void Img2colMatmul(const int32_t &taskCount, const int32_t &globalTaskOffset)
    {
        if (taskCount <= 0) {
            return;
        }

        mm0_.SetTensorA(img2colGM_);
        mm0_.SetTensorB(weightGM_);
        mm0_.SetSingleShape(taskCount, outChannels_, inChannels_ * kernelSize_);
        
        mm0_.template IterateAll<true>(sparseValueGM_[globalTaskOffset * outChannels_]);
    }

private:
    TPipe* pipe_;
    int8_t k0_, k1_, k2_;
    int32_t blkIdx_, aivNum_, byteSizePerElements_, kernelSize_, inChannels_, inChannelsAligned_, outChannels_, outChannelsAligned_, outputSingleLoopTaskPlusOneAligned_, outputSingleLoopTaskAligned_,
        availableUBSize_, inputTaskStartOffset_, outputTaskStartOffset_, featureChannelsSize_, featureBufLen_, featureBufIdx_, outputCoreTaskCount_, outputBigCoreCount_,
        outputSingleLoopTask_, outputTaskCount_, inputCoreTaskCount_, inputBigCoreCount_, inputSingleLoopTask_, inputTaskCount_, inputSingleLoopTaskAligned_, matmulTaskPerIter_, matmulIdx_;

    GlobalTensor<T> inputFeatureGM_, sparseValueGM_, weightGM_, mmFeatureGM_, img2colGM_;
    GlobalTensor<int32_t> uniqueIndicesGM_, sortedIndicesGM_, indicesGM_, sparseIndicesGM_, reOrderedindicesGM_;
    LocalTensor<int32_t> uniqueIndicesLocal_, sortedIndicesLocal_, reOrderedIndicesLocal_, validIndiceslLocal_, kLocalPosLocal_, kGlobalPosLocal_;
    LocalTensor<T> copyInFeatureLocal_, copyoutFeatureLocal_, zerosLocal_;
    TBuf<TPosition::VECCALC> UbBuf_;
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
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm0_, &tiling_data.mm0TilingData);
    op.Init(features, weight, unique_indices_offset, former_sorted_indices, indices, sparse_value, sparse_indices, workspace, &tiling_data, &pipe);
    op.Process();
}