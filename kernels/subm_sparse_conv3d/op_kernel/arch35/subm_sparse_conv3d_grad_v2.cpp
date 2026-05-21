#include "kernel_operator.h"
#include "kernel_tpipe_impl.h"
#include "kernel_utils.h"
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace MicroAPI;

namespace {
constexpr int32_t BYTE_SIZE_PER_BLOCK = 32;
constexpr int32_t INT32_BYTE_SIZE = 4;
constexpr int32_t FLOAT32_BYTE_SIZE = 4;
constexpr int32_t FLOAT16_BYTE_SIZE = 2;
constexpr MatmulConfig SUBM_SPARSE_CONV3D_CFG = GetNormalConfig(); // normal比MDL更好
constexpr int32_t INT64_BIT_SIZE = 64;
constexpr int32_t SINGLE_LOOP_COMPARE_UB = 256;
constexpr int32_t FLOAT4_ELEM = 4;
constexpr int32_t HALF2_ELEM = 2;
constexpr uint32_t THREAD_NUM = 2048;
constexpr int32_t UNROLL_NUM = 2;
constexpr int32_t AIC_FLAG_RANGE = 10;
constexpr int32_t AIC_FLAG_OFFSET = 16;
constexpr int32_t AIC_AIV_RATIO = 2;
constexpr int32_t SIMT_CHANNEL_THRES = 128;
} // namespace

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void getSparseFeatureSIMT(volatile __gm__ T *inputFeaturesGM,
    __ubuf__ T *sparseFeaturesLocal0, __ubuf__ T *sparseFeaturesLocal1, __ubuf__ int32_t *sparseIndicesLocal0,
    __ubuf__ int32_t *sparseIndicesLocal1, int32_t sparseNum, int32_t channels, int32_t channelsAligned) {
    int32_t xId = threadIdx.x;
    int32_t yId = threadIdx.y;
    int32_t yBlockDim = blockDim.y;

    for (int32_t i = sparseNum - yId - 1; i >= 0; i -= yBlockDim) {
        int32_t sparseIndices0 = sparseIndicesLocal0[i];
        int32_t sparseIndices1 = sparseIndicesLocal1[i];
        int32_t featureOffset0 = sparseIndices0 * channels + xId;
        int32_t featureOffset1 = sparseIndices1 * channels + xId;
        int32_t sparseOffset = i * channelsAligned + xId;

        T inputFeature0 = inputFeaturesGM[featureOffset0];
        T inputFeature1 = inputFeaturesGM[featureOffset1];

        sparseFeaturesLocal0[sparseOffset] = inputFeature0;
        sparseFeaturesLocal1[sparseOffset] = inputFeature1;
    }
}

template <typename T, int32_t channels, int32_t channelsAligned>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void getSparseFeatureFastSIMT(volatile __gm__ T *inputFeaturesGM,
    __ubuf__ T *sparseFeaturesLocal0, // offset
    __ubuf__ T *sparseFeaturesLocal1, // offset
    __ubuf__ int32_t *sparseIndicesLocal0, __ubuf__ int32_t *sparseIndicesLocal1, int32_t sparseNum) {
    int32_t xId = threadIdx.x;
    int32_t yId = threadIdx.y;
    int32_t yBlockDim = blockDim.y;

    for (int32_t i = sparseNum - yId - 1; i >= 0; i -= yBlockDim) {
        int32_t sparseIndices0 = sparseIndicesLocal0[i];
        int32_t sparseIndices1 = sparseIndicesLocal1[i];
        int32_t featureOffset0 = sparseIndices0 * channels + xId;
        int32_t featureOffset1 = sparseIndices1 * channels + xId;
        int32_t sparseOffset = i * channelsAligned + xId;

        T inputFeature0 = inputFeaturesGM[featureOffset0];
        T inputFeature1 = inputFeaturesGM[featureOffset1];

        sparseFeaturesLocal0[sparseOffset] = inputFeature0;
        sparseFeaturesLocal1[sparseOffset] = inputFeature1;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void scatterAddFeatureSIMT(__gm__ float *outputFeaturesGradGM,
    __ubuf__ int32_t *tmpSparseIndicesLocal0, __ubuf__ int32_t *tmpSparseIndicesLocal1,
    __ubuf__ float *scatterFeatureLocal0, __ubuf__ float *scatterFeatureLocal1, int32_t sparseNum, int32_t channels,
    int32_t channelsAligned) {
    int32_t xId = threadIdx.x;
    int32_t yId = threadIdx.y;
    int32_t yBlockDim = blockDim.y;

    for (int32_t i = sparseNum - yId - 1; i >= 0; i -= yBlockDim) {
        int32_t sparseIndices0 = tmpSparseIndicesLocal0[i];
        int32_t sparseIndices1 = tmpSparseIndicesLocal1[i];
        int32_t outputOffset0 = sparseIndices0 * channels + xId;
        int32_t outputOffset1 = sparseIndices1 * channels + xId;
        int32_t featureOffset = i * channelsAligned + xId;

        float featureGrad0 = scatterFeatureLocal0[featureOffset];
        float featureGrad1 = scatterFeatureLocal1[featureOffset];

        Simt::AtomicAdd(outputFeaturesGradGM + outputOffset0, featureGrad0);
        Simt::AtomicAdd(outputFeaturesGradGM + outputOffset1, featureGrad1);
    }
}

template <int32_t channels, int32_t channelsAligned>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void scatterAddFeatureFastSIMT(
    __gm__ float *outputFeaturesGradGM, __ubuf__ int32_t *tmpSparseIndicesLocal0,
    __ubuf__ int32_t *tmpSparseIndicesLocal1, __ubuf__ float *scatterFeatureLocal0,
    __ubuf__ float *scatterFeatureLocal1, int32_t sparseNum) {
    int32_t xId = threadIdx.x;
    int32_t yId = threadIdx.y;
    int32_t yBlockDim = blockDim.y;

    for (int32_t i = sparseNum - yId - 1; i >= 0; i -= yBlockDim) {
        int32_t sparseIndices0 = tmpSparseIndicesLocal0[i];
        int32_t sparseIndices1 = tmpSparseIndicesLocal1[i];
        int32_t outputOffset0 = sparseIndices0 * channels + xId;
        int32_t outputOffset1 = sparseIndices1 * channels + xId;
        int32_t featureOffset = i * channelsAligned + xId;

        float featureGrad0 = scatterFeatureLocal0[featureOffset];
        float featureGrad1 = scatterFeatureLocal1[featureOffset];

        Simt::AtomicAdd(outputFeaturesGradGM + outputOffset0, featureGrad0);
        Simt::AtomicAdd(outputFeaturesGradGM + outputOffset1, featureGrad1);
    }
}

template <typename T> class SubmSparseConv3dGradV2 {
  public:
    using weightMatType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using featureMatType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using gradOutFeaturesMatType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using weightGradMatType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using featureGradMatType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;

    // matmul是vector的api，matmulImpl是cube的api
    matmul::MatmulImpl<gradOutFeaturesMatType, weightMatType, featureGradMatType, featureGradMatType,
        SUBM_SPARSE_CONV3D_CFG>
        featureMatmul_;
    matmul::MatmulImpl<featureMatType, gradOutFeaturesMatType, weightGradMatType, weightGradMatType,
        SUBM_SPARSE_CONV3D_CFG>
        weightMatmul_;

    using FLOAT_DTYPE = float4;
    using HALF_DTYPE = half2;

    __aicore__ inline SubmSparseConv3dGradV2(){};

    __aicore__ inline void Init(TPipe *pipe, GM_ADDR features, GM_ADDR weight, GM_ADDR grad_out_features,
        GM_ADDR indices_offset, GM_ADDR features_grad, GM_ADDR weight_grad, GM_ADDR usrWorkspace,
        SubmConv3dGradV2TillingData *tilingData) {
        pipe_ = pipe;
        aicNum_ = GetBlockNum();
        aivNum_ = AIC_AIV_RATIO * aicNum_;

        blockIdx_ = GetBlockIdx();
        InitTiling(tilingData);
        InitGM(features, weight, grad_out_features, indices_offset, features_grad, weight_grad, usrWorkspace);
        InitUB();
    }

    // process one batch with `singleLoopTask=S` tasks
    // features: [S, C1]
    // indices_offset: [S*K*K*K]
    // weight: [K, K, K, C1, C2]
    // grad_out_feature: [S, C2]
    __aicore__ inline void Process() {
        // feature和weight部分可以提前算中间区域，提高计算效率
        if ASCEND_IS_AIC {
            calCenterFeatureMatmul();
            calCenterWeightMatmul();
        }
        // true: vector syncall; false: aiv aic syncall
        SyncAll<false>();

        // vector部分首先获取稀疏点的特征，随后等待cube计算，再做scatterAdd
        if ASCEND_IS_AIV {
            for (int32_t k = 0; k < halfK_; k++) {
                getSparseData(k);
                scatterAddSparseFeatures(k);
            }
        }

        // cube部分首先等待vector获取点，然后计算结果
        if ASCEND_IS_AIC {
            for (int32_t k = 0; k < halfK_; k++) {
                processSparseMatmul(k);
            }
        }
    }

  private:
    __aicore__ inline void InitTiling(SubmConv3dGradV2TillingData *tilingData) {
        byteSizePerElement_ = sizeof(T);
        k0_ = tilingData->k0;
        k1_ = tilingData->k1;
        k2_ = tilingData->k2;
        inChannels_ = tilingData->inChannels;
        outChannels_ = tilingData->outChannels;
        totalTaskCount_ = tilingData->totalTaskCount;
        coreTaskCount_ = tilingData->coreTaskCount;
        bigCoreCount_ = tilingData->bigCoreCount;
        singleLoopTask_ = tilingData->singleLoopTask;
        INT_SPACE_NUM = tilingData->intSpaceNum;
        PROCESS_NUM_PER_STEP = tilingData->processNumPerStep;
        innerLoopTask_ = tilingData->innerLoopTask;
        BUFFER_NUM = tilingData->bufferNum;

        k12_ = k1_ * k2_;
        kernelSize_ = k0_ * k12_;
        halfK_ = kernelSize_ / TWO;
        inChannelsAligned_ = AlignUp(inChannels_, BYTE_SIZE_PER_BLOCK / byteSizePerElement_);
        inChannelsFP32Align_ = AlignUp(inChannels_, BYTE_SIZE_PER_BLOCK / FLOAT32_BYTE_SIZE);
        outChannelsAligned_ = AlignUp(outChannels_, BYTE_SIZE_PER_BLOCK / byteSizePerElement_);
        singleLoopTaskAligned_ = AlignUp(singleLoopTask_, BYTE_SIZE_PER_BLOCK / INT32_BYTE_SIZE);
        totalTaskAligned_ = AlignUp(totalTaskCount_, BYTE_SIZE_PER_BLOCK / byteSizePerElement_);
        indicesBufSize_ = AlignUp(singleLoopTaskAligned_, SINGLE_LOOP_COMPARE_UB / INT32_BYTE_SIZE);

        featureBufSize_ = innerLoopTask_ * inChannelsAligned_;
        gradOutFeatBufSize_ = innerLoopTask_ * outChannelsAligned_;
        scatterFP32BufSize_ = innerLoopTask_ * inChannelsFP32Align_;

        if ASCEND_IS_AIC {
            if (blockIdx_ < bigCoreCount_) {
                coreTaskCount_ = coreTaskCount_ + 1;
                matmulTaskOffset_ = coreTaskCount_ * blockIdx_;
            } else {
                matmulTaskOffset_ = (coreTaskCount_ + 1) * bigCoreCount_ + coreTaskCount_ * (blockIdx_ - bigCoreCount_);
            }

            aicTaskOffset_ = blockIdx_ * singleLoopTask_ * 2;
        }
        if ASCEND_IS_AIV {
            aivTaskOffset_ = (blockIdx_ / 2) * (2 * singleLoopTask_);
        }
    }

    __aicore__ inline void InitGM(GM_ADDR features, GM_ADDR weight, GM_ADDR grad_out_features, GM_ADDR indices_offset,
        GM_ADDR features_grad, GM_ADDR weight_grad, GM_ADDR usrWorkspace) {
        inputFeaturesGM_.SetGlobalBuffer((__gm__ T *)features);
        inputWeightGM_.SetGlobalBuffer((__gm__ T *)weight);
        inputGradOutFeaturesGM_.SetGlobalBuffer((__gm__ T *)grad_out_features);
        inputIndicesOffsetGM_.SetGlobalBuffer((__gm__ int32_t *)indices_offset);
        outputFeaturesGradGM_.SetGlobalBuffer((__gm__ float *)features_grad);
        outputWeightGradGM_.SetGlobalBuffer((__gm__ float *)weight_grad);

        int64_t upperByteRatio = INT32_BYTE_SIZE / byteSizePerElement_;
        int64_t offset0 = 0;
        int64_t offset1 = offset0 + totalTaskCount_ * inChannels_;
        int64_t offset2 = offset1 + totalTaskCount_ * inChannels_;
        int64_t offset3 = offset2 + totalTaskCount_ * outChannels_;
        int64_t offset4 = (offset3 + totalTaskCount_ * outChannels_ + upperByteRatio - 1) / upperByteRatio;
        int64_t offset5 = offset4 + totalTaskCount_ * inChannels_;
        int64_t offset6 = offset5 + totalTaskCount_ * inChannels_;
        int64_t offset7 = offset6 + totalTaskCount_ * halfK_;
        int64_t offset8 = offset7 + totalTaskCount_ * halfK_;

        tmpSparseFeaturesGM0_.SetGlobalBuffer((__gm__ T *)(usrWorkspace) + offset0);
        tmpSparseFeaturesGM1_.SetGlobalBuffer((__gm__ T *)(usrWorkspace) + offset1);
        tmpSparseGradOutFeatGM0_.SetGlobalBuffer((__gm__ T *)(usrWorkspace) + offset2);
        tmpSparseGradOutFeatGM1_.SetGlobalBuffer((__gm__ T *)(usrWorkspace) + offset3);
        tmpMatmulResGM0_.SetGlobalBuffer((__gm__ float *)(usrWorkspace) + offset4);
        tmpMatmulResGM1_.SetGlobalBuffer((__gm__ float *)(usrWorkspace) + offset5);
        tmpSparseIndicesGM0_.SetGlobalBuffer((__gm__ int32_t *)(usrWorkspace) + offset6);
        tmpSparseIndicesGM1_.SetGlobalBuffer((__gm__ int32_t *)(usrWorkspace) + offset7);
        tmpSparseNumGM_.SetGlobalBuffer((__gm__ int32_t *)(usrWorkspace) + offset8);
    }

    __aicore__ inline void InitUB() {
        pipe_->InitBuffer(indicesBuf_, INT_SPACE_NUM * indicesBufSize_ * INT32_BYTE_SIZE);
        pipe_->InitBuffer(featureLocalBuf_, BUFFER_NUM * PROCESS_NUM_PER_STEP * featureBufSize_ * byteSizePerElement_);
        pipe_->InitBuffer(
            gradOutFeatLocalBuf_, BUFFER_NUM * PROCESS_NUM_PER_STEP * gradOutFeatBufSize_ * byteSizePerElement_);
        pipe_->InitBuffer(
            scatterLocalBuf_, BUFFER_NUM * PROCESS_NUM_PER_STEP * scatterFP32BufSize_ * FLOAT32_BYTE_SIZE);

        indicesLocal_ = indicesBuf_.Get<int32_t>();
        sparseIndicesLocal0_ = indicesLocal_[indicesBufSize_];
        sparseIndicesLocal1_ = sparseIndicesLocal0_[indicesBufSize_];

        scatterIndicesLocal0_ = sparseIndicesLocal1_[indicesBufSize_];
        scatterIndicesLocal1_ = scatterIndicesLocal0_[indicesBufSize_];

        gatherFeatureLocal0_ = featureLocalBuf_.Get<T>();
        gatherFeatureLocal1_ = gatherFeatureLocal0_[BUFFER_NUM * featureBufSize_];

        gatherGradOutFeatLocal0_ = gradOutFeatLocalBuf_.Get<T>();
        gatherGradOutFeatLocal1_ = gatherGradOutFeatLocal0_[BUFFER_NUM * gradOutFeatBufSize_];

        scatterFeatureLocal0_ = scatterLocalBuf_.Get<float>();
        scatterFeatureLocal1_ = scatterFeatureLocal0_[BUFFER_NUM * scatterFP32BufSize_];
    }

    __aicore__ inline void calCenterFeatureMatmul() {
        // inputGradOutFeaturesGM_: [S, C2]
        // inputWeightGM_: [C1, C2]
        // outputFeaturesGradGM_: [S, C1]
        if (coreTaskCount_ == 0) {
            return;
        }

        featureMatmul_.SetTensorA(inputGradOutFeaturesGM_[matmulTaskOffset_ * outChannels_]);
        featureMatmul_.SetTensorB(inputWeightGM_[halfK_ * inChannels_ * outChannels_], true);
        featureMatmul_.SetOrgShape(coreTaskCount_, inChannels_, outChannels_);
        featureMatmul_.SetSingleShape(coreTaskCount_, inChannels_, outChannels_);

        if (byteSizePerElement_ == FLOAT32_BYTE_SIZE) {
            featureMatmul_.SetHF32(true);
        }
        featureMatmul_.template IterateAll<false>(outputFeaturesGradGM_[matmulTaskOffset_ * inChannels_], 0);
        featureMatmul_.End();
    }

    __aicore__ inline void calCenterWeightMatmul() {
        // calculate center dense weight matmul
        // inputFeaturesGM_: [S, C1].T
        // inputGradOutFeaturesGM_: [S, C2]
        // outputWeightGradGM_: [C1, C2]
        if (coreTaskCount_ == 0) {
            return;
        }

        weightMatmul_.SetTensorA(inputFeaturesGM_[matmulTaskOffset_ * inChannels_], true);
        weightMatmul_.SetTensorB(inputGradOutFeaturesGM_[matmulTaskOffset_ * outChannels_]);
        weightMatmul_.SetOrgShape(inChannels_, outChannels_, coreTaskCount_);
        weightMatmul_.SetSingleShape(inChannels_, outChannels_, coreTaskCount_);

        if (byteSizePerElement_ == FLOAT32_BYTE_SIZE) {
            weightMatmul_.SetHF32(true);
        }
        weightMatmul_.template IterateAll<false>(outputWeightGradGM_[halfK_ * inChannels_ * outChannels_], 1);
        weightMatmul_.End();
    }

    __aicore__ inline void getSparseData(int32_t k) {
        uint16_t flagIdx = 0;

        for (int32_t idx = aivTaskOffset_; idx < totalTaskCount_; idx += singleLoopTask_ * aivNum_) {
            int32_t taskOffset = idx + (blockIdx_ % 2) * singleLoopTask_;
            int32_t curTaskCount = min(singleLoopTask_, totalTaskCount_ - taskOffset);
            if (curTaskCount <= 0) {
                continue;
            }

            copyInSortIndices(k, taskOffset, curTaskCount);
            int32_t sparseNum = getSparseNum(curTaskCount);
            copyOutSparseIndices(k, taskOffset, sparseNum);
            copyOutSparseFeature(k, taskOffset, sparseNum);

            CrossCoreSetFlag<0x4, PIPE_MTE3>(flagIdx);

            flagIdx = (flagIdx + 1) % AIC_FLAG_RANGE;
        }
    }

    __aicore__ inline void copyInSortIndices(int32_t k, int32_t taskOffset, int32_t taskCount) {
        if (taskCount <= 0) {
            return;
        }

        DataCopyPad(indicesLocal_, inputIndicesOffsetGM_[k * totalTaskCount_ + taskOffset],
            {1, static_cast<uint32_t>(taskCount * INT32_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});

        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        LocalTensor<uint32_t> tmpLocal = sparseIndicesLocal1_.template ReinterpretCast<uint32_t>();
        Sort<int32_t, true, sortConfig_>(sparseIndicesLocal0_, tmpLocal, indicesLocal_, taskCount);
        Adds(sparseIndicesLocal1_, sparseIndicesLocal1_, taskOffset, taskCount);

        SetFlag<HardEvent::V_MTE3>(0);
    }

    __aicore__ inline int32_t getSparseNum(int32_t taskCount) {
        if (taskCount <= 0) {
            return 0;
        }

        int32_t compTaskAligned = AlignUp(taskCount, SINGLE_LOOP_COMPARE_UB / INT32_BYTE_SIZE);
        LocalTensor<uint8_t> validMaskLocal = indicesLocal_.ReinterpretCast<uint8_t>();
        CompareScalar(validMaskLocal, sparseIndicesLocal0_, static_cast<int32_t>(-1), CMPMODE::NE, compTaskAligned);

        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);

        int32_t sparseNum = 0;
        for (int32_t i = 0; i < taskCount; i += INT64_BIT_SIZE) {
            uint64_t validBit = min(64, taskCount - i);
            uint64_t validMask = validBit == INT64_BIT_SIZE ? UINT64_MAX : ((uint64_t)(1) << validBit) - 1;

            uint64_t curValidMask = validMaskLocal.ReinterpretCast<uint64_t>().GetValue(i / INT64_BIT_SIZE);
            curValidMask = curValidMask & validMask;

            sparseNum += ScalarGetCountOfValue<1>(curValidMask);
        }

        return sparseNum;
    }

    __aicore__ inline void copyOutSparseIndices(int32_t k, int32_t taskOffset, int32_t sparseNum) {
        if (taskOffset >= totalTaskCount_) {
            return;
        }

        indicesLocal_.SetValue(0, sparseNum);
        DataCopyPad(tmpSparseNumGM_[k * totalTaskCount_ + taskOffset], indicesLocal_,
            {static_cast<uint16_t>(1), static_cast<uint32_t>(1 * INT32_BYTE_SIZE), 0, 0, 0});

        // 等待copyInSortIndices
        WaitFlag<HardEvent::V_MTE3>(0);
        if (sparseNum == 0) {
            return;
        }
        DataCopyPad(tmpSparseIndicesGM0_[k * totalTaskCount_ + taskOffset], sparseIndicesLocal0_,
            {static_cast<uint16_t>(1), static_cast<uint32_t>(sparseNum * INT32_BYTE_SIZE), 0, 0, 0});
        DataCopyPad(tmpSparseIndicesGM1_[k * totalTaskCount_ + taskOffset], sparseIndicesLocal1_,
            {static_cast<uint16_t>(1), static_cast<uint32_t>(sparseNum * INT32_BYTE_SIZE), 0, 0, 0});
    }

    __aicore__ inline void copyOutSparseFeature(int32_t k, int32_t taskOffset, int32_t sparseNum) {
        if (taskOffset >= totalTaskCount_) {
            return;
        }
        if (sparseNum == 0) {
            return;
        }

        for (uint8_t i = 0; i < BUFFER_NUM; i++) {
            SetFlag<HardEvent::MTE3_V>(i);
            SetFlag<HardEvent::MTE3_MTE2>(i);
        }

        uint8_t flagIdx = 0;
        for (int32_t innerIdx = 0; innerIdx < sparseNum; innerIdx += innerLoopTask_) {
            int32_t innerTasks = min(innerLoopTask_, sparseNum - innerIdx);
            WaitFlag<HardEvent::MTE3_V>(flagIdx);
            WaitFlag<HardEvent::MTE3_MTE2>(flagIdx);

            getSparseFeature(inputFeaturesGM_, gatherFeatureLocal0_[flagIdx * featureBufSize_],
                gatherFeatureLocal1_[flagIdx * featureBufSize_], sparseIndicesLocal0_[innerIdx],
                sparseIndicesLocal1_[innerIdx], innerTasks, inChannels_, inChannelsAligned_);

            getSparseFeature(inputGradOutFeaturesGM_, gatherGradOutFeatLocal0_[flagIdx * gradOutFeatBufSize_],
                gatherGradOutFeatLocal1_[flagIdx * gradOutFeatBufSize_],
                sparseIndicesLocal1_[innerIdx], // gradOutFeature的indices和inputFeatures的indices相反
                sparseIndicesLocal0_[innerIdx], innerTasks, outChannels_, outChannelsAligned_);

            SetFlag<HardEvent::V_MTE3>(flagIdx);
            WaitFlag<HardEvent::V_MTE3>(flagIdx);

            SetFlag<HardEvent::MTE2_MTE3>(flagIdx);
            WaitFlag<HardEvent::MTE2_MTE3>(flagIdx);

            DataCopyPad(tmpSparseFeaturesGM0_[(taskOffset + innerIdx) * inChannels_],
                gatherFeatureLocal0_[flagIdx * featureBufSize_],
                {static_cast<uint16_t>(innerTasks), static_cast<uint32_t>(inChannels_ * byteSizePerElement_), 0, 0, 0});
            DataCopyPad(tmpSparseFeaturesGM1_[(taskOffset + innerIdx) * inChannels_],
                gatherFeatureLocal1_[flagIdx * featureBufSize_],
                {static_cast<uint16_t>(innerTasks), static_cast<uint32_t>(inChannels_ * byteSizePerElement_), 0, 0, 0});

            DataCopyPad(tmpSparseGradOutFeatGM0_[(taskOffset + innerIdx) * outChannels_],
                gatherGradOutFeatLocal0_[flagIdx * gradOutFeatBufSize_],
                {static_cast<uint16_t>(innerTasks), static_cast<uint32_t>(outChannels_ * byteSizePerElement_), 0, 0,
                    0});
            DataCopyPad(tmpSparseGradOutFeatGM1_[(taskOffset + innerIdx) * outChannels_],
                gatherGradOutFeatLocal1_[flagIdx * gradOutFeatBufSize_],
                {static_cast<uint16_t>(innerTasks), static_cast<uint32_t>(outChannels_ * byteSizePerElement_), 0, 0,
                    0});

            SetFlag<HardEvent::V_S>(flagIdx);
            WaitFlag<HardEvent::V_S>(flagIdx);

            SetFlag<HardEvent::MTE3_V>(flagIdx);
            SetFlag<HardEvent::MTE3_MTE2>(flagIdx);

            flagIdx = (flagIdx + 1) % BUFFER_NUM;
        }

        for (uint8_t i = 0; i < BUFFER_NUM; i++) {
            WaitFlag<HardEvent::MTE3_V>(i);
            WaitFlag<HardEvent::MTE3_MTE2>(i);
        }
    }

    __aicore__ inline void getSparseFeature(GlobalTensor<T> inputFeaturesGM, LocalTensor<T> sparseFeaturesLocal0,
        LocalTensor<T> sparseFeaturesLocal1, LocalTensor<int32_t> sparseIndicesLocal0,
        LocalTensor<int32_t> sparseIndicesLocal1, int32_t sparseNum, int32_t channels, int32_t channelsAligned) {
        if (channels >= SIMT_CHANNEL_THRES) {
            for (int32_t idx = sparseNum - 1; idx >= 0; idx--) {
                int32_t indiceVal0 = sparseIndicesLocal0.GetValue(idx);
                int32_t indiceVal1 = sparseIndicesLocal1.GetValue(idx);

                DataCopyPad(sparseFeaturesLocal0[idx * channelsAligned], inputFeaturesGM[indiceVal0 * channels],
                    {1, static_cast<uint32_t>(channels * byteSizePerElement_), 0, 0, 0}, {false, 0, 0, 0});
                DataCopyPad(sparseFeaturesLocal1[idx * channelsAligned], inputFeaturesGM[indiceVal1 * channels],
                    {1, static_cast<uint32_t>(channels * byteSizePerElement_), 0, 0, 0}, {false, 0, 0, 0});
            }
            return;
        }

        if (byteSizePerElement_ == FLOAT32_BYTE_SIZE && channels % FLOAT4_ELEM == 0) {
            uint32_t inputChannels = channels / FLOAT4_ELEM;
            uint32_t inputChannelsAligned = channelsAligned / FLOAT4_ELEM;

            callGatherSIMTFunc<FLOAT_DTYPE>((__gm__ FLOAT_DTYPE *)inputFeaturesGM.GetPhyAddr(),
                (__ubuf__ FLOAT_DTYPE *)sparseFeaturesLocal0.GetPhyAddr(),
                (__ubuf__ FLOAT_DTYPE *)sparseFeaturesLocal1.GetPhyAddr(),
                (__ubuf__ int32_t *)sparseIndicesLocal0.GetPhyAddr(),
                (__ubuf__ int32_t *)sparseIndicesLocal1.GetPhyAddr(), sparseNum, inputChannels, inputChannelsAligned);
        } else if (byteSizePerElement_ == FLOAT16_BYTE_SIZE && channels % HALF2_ELEM == 0) {
            uint32_t inputChannels = channels / HALF2_ELEM;
            uint32_t inputChannelsAligned = channelsAligned / HALF2_ELEM;

            callGatherSIMTFunc<HALF_DTYPE>((__gm__ HALF_DTYPE *)inputFeaturesGM.GetPhyAddr(),
                (__ubuf__ HALF_DTYPE *)sparseFeaturesLocal0.GetPhyAddr(),
                (__ubuf__ HALF_DTYPE *)sparseFeaturesLocal1.GetPhyAddr(),
                (__ubuf__ int32_t *)sparseIndicesLocal0.GetPhyAddr(),
                (__ubuf__ int32_t *)sparseIndicesLocal1.GetPhyAddr(), sparseNum, inputChannels, inputChannelsAligned);
        } else {
            AscendC::Simt::VF_CALL<getSparseFeatureSIMT<T>>(
                AscendC::Simt::Dim3{static_cast<uint32_t>(channels), THREAD_NUM / channels},
                (__gm__ T *)inputFeaturesGM.GetPhyAddr(), (__ubuf__ T *)sparseFeaturesLocal0.GetPhyAddr(),
                (__ubuf__ T *)sparseFeaturesLocal1.GetPhyAddr(), (__ubuf__ int32_t *)sparseIndicesLocal0.GetPhyAddr(),
                (__ubuf__ int32_t *)sparseIndicesLocal1.GetPhyAddr(), sparseNum, channels, channelsAligned);
        }
    }

    template <typename D>
    __aicore__ inline void callGatherSIMTFunc(__gm__ D *inputFeaturesGM, __ubuf__ D *sparseFeaturesLocal0,
        __ubuf__ D *sparseFeaturesLocal1, __ubuf__ int32_t *sparseIndicesLocal0, __ubuf__ int32_t *sparseIndicesLocal1,
        int32_t sparseNum, uint32_t inputChannels, uint32_t inputChannelsAligned) {
        // 2**3 -- 2**7
        switch (inputChannels) {
        case 4:
            if (byteSizePerElement_ == FLOAT16_BYTE_SIZE) {
                Simt::VF_CALL<getSparseFeatureFastSIMT<D, 4, 8>>(Simt::Dim3{inputChannels, THREAD_NUM / inputChannels},
                    inputFeaturesGM, sparseFeaturesLocal0, sparseFeaturesLocal1, sparseIndicesLocal0,
                    sparseIndicesLocal1, sparseNum);
            } else {
                Simt::VF_CALL<getSparseFeatureFastSIMT<D, 4, 4>>(Simt::Dim3{inputChannels, THREAD_NUM / inputChannels},
                    inputFeaturesGM, sparseFeaturesLocal0, sparseFeaturesLocal1, sparseIndicesLocal0,
                    sparseIndicesLocal1, sparseNum);
            }
            break;
        case 8:
            Simt::VF_CALL<getSparseFeatureFastSIMT<D, 8, 8>>(Simt::Dim3{inputChannels, THREAD_NUM / inputChannels},
                inputFeaturesGM, sparseFeaturesLocal0, sparseFeaturesLocal1, sparseIndicesLocal0, sparseIndicesLocal1,
                sparseNum);
            break;
        case 16:
            Simt::VF_CALL<getSparseFeatureFastSIMT<D, 16, 16>>(Simt::Dim3{inputChannels, THREAD_NUM / inputChannels},
                inputFeaturesGM, sparseFeaturesLocal0, sparseFeaturesLocal1, sparseIndicesLocal0, sparseIndicesLocal1,
                sparseNum);
            break;
        case 32:
            Simt::VF_CALL<getSparseFeatureFastSIMT<D, 32, 32>>(Simt::Dim3{inputChannels, THREAD_NUM / inputChannels},
                inputFeaturesGM, sparseFeaturesLocal0, sparseFeaturesLocal1, sparseIndicesLocal0, sparseIndicesLocal1,
                sparseNum);
            break;
        default:
            Simt::VF_CALL<getSparseFeatureSIMT<D>>(Simt::Dim3{inputChannels, THREAD_NUM / inputChannels},
                inputFeaturesGM, sparseFeaturesLocal0, sparseFeaturesLocal1, sparseIndicesLocal0, sparseIndicesLocal1,
                sparseNum, inputChannels, inputChannelsAligned);
            break;
        }
    }

    __aicore__ inline void processSparseMatmul(int32_t k) {
        uint16_t flagIdx = 0;
        for (int32_t taskOffset = aicTaskOffset_; taskOffset < totalTaskCount_;
             taskOffset += AIC_AIV_RATIO * singleLoopTask_ * aicNum_) {
            calSparseMatmul(k, taskOffset, flagIdx);
            calSparseMatmul(k, taskOffset + singleLoopTask_, flagIdx + AIC_FLAG_OFFSET);

            flagIdx = (flagIdx + 1) % AIC_FLAG_RANGE;
        }
    }

    __aicore__ inline void calSparseMatmul(int32_t k, int32_t taskOffset, uint16_t flagIdx) {
        if (taskOffset >= totalTaskCount_) {
            return;
        }

        CrossCoreWaitFlag<0x4>(flagIdx);

        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            tmpSparseNumGM_[k * totalTaskCount_ + taskOffset]);
        int32_t sparseNum = tmpSparseNumGM_.GetValue(k * totalTaskCount_ + taskOffset);

        calSparseFeatureMatmul(k, taskOffset, sparseNum);
        calSpraseWeightMatmul(k, taskOffset, sparseNum);

        CrossCoreSetFlag<0x4, PIPE_FIX>(flagIdx);
    }

    __aicore__ inline void calSparseFeatureMatmul(int32_t k, int32_t taskOffset, int32_t sparseNum) {
        // tmpSparseGradOutFeatGM0_: [T, C2]
        // inputWeightGM_: [C1, C2]
        // tmpMatmulResGM0_: [S, C2] x [C1, C2].T = [S, C1]
        if (sparseNum == 0) {
            return;
        }

        if (byteSizePerElement_ == FLOAT32_BYTE_SIZE) {
            featureMatmul_.SetHF32(true);
        }

        // calculate k/2 - i column
        featureMatmul_.SetOrgShape(sparseNum, inChannels_, outChannels_);
        featureMatmul_.SetTensorA(tmpSparseGradOutFeatGM0_[taskOffset * outChannels_]);
        featureMatmul_.SetTensorB(inputWeightGM_[k * inChannels_ * outChannels_], true);
        featureMatmul_.SetSingleShape(sparseNum, inChannels_, outChannels_);

        featureMatmul_.template IterateAll<false>(tmpMatmulResGM0_[taskOffset * inChannels_], 0, false, true);
        featureMatmul_.End();

        // calculate symmetryk = k/2 + i column
        int32_t symmK = kernelSize_ - k - 1;
        featureMatmul_.SetTensorA(tmpSparseGradOutFeatGM1_[taskOffset * outChannels_]);
        featureMatmul_.SetTensorB(inputWeightGM_[symmK * inChannels_ * outChannels_], true);
        featureMatmul_.SetSingleShape(sparseNum, inChannels_, outChannels_);

        featureMatmul_.template IterateAll<false>(tmpMatmulResGM1_[taskOffset * inChannels_], 0, false, true);
        featureMatmul_.End();
    }

    __aicore__ inline void calSpraseWeightMatmul(int32_t k, int32_t taskOffset, int32_t sparseNum) {
        if (sparseNum == 0) {
            return;
        }

        if (byteSizePerElement_ == FLOAT32_BYTE_SIZE) {
            weightMatmul_.SetHF32(true);
        }

        // calculate k/2-i column
        weightMatmul_.SetOrgShape(inChannels_, outChannels_, sparseNum);
        weightMatmul_.SetTensorA(tmpSparseFeaturesGM0_[taskOffset * inChannels_], true);
        weightMatmul_.SetTensorB(tmpSparseGradOutFeatGM0_[taskOffset * outChannels_]);
        weightMatmul_.SetSingleShape(inChannels_, outChannels_, sparseNum);

        weightMatmul_.template IterateAll<false>(outputWeightGradGM_[k * inChannels_ * outChannels_], 1, false, true);
        weightMatmul_.End();

        // calculate symmetry k =  k/2+i column
        int32_t symmK = kernelSize_ - k - 1;
        weightMatmul_.SetTensorA(tmpSparseFeaturesGM1_[taskOffset * inChannels_], true);
        weightMatmul_.SetTensorB(tmpSparseGradOutFeatGM1_[taskOffset * outChannels_]);
        weightMatmul_.SetSingleShape(inChannels_, outChannels_, sparseNum);

        weightMatmul_.template IterateAll<false>(
            outputWeightGradGM_[symmK * inChannels_ * outChannels_], 1, false, true);
        weightMatmul_.End();
    }

    __aicore__ inline void scatterAddSparseFeatures(int32_t k) {
        uint16_t flagIdx = 0;

        for (int32_t idx = aivTaskOffset_; idx < totalTaskCount_; idx += singleLoopTask_ * aivNum_) {
            int32_t taskOffset = idx + (blockIdx_ % 2) * singleLoopTask_;
            if (taskOffset >= totalTaskCount_) {
                continue;
            }

            CrossCoreWaitFlag<0x4>(flagIdx);

            calScatterAdd(k, taskOffset);

            flagIdx = (flagIdx + 1) % AIC_FLAG_RANGE;
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void calScatterAdd(int32_t k, int32_t taskOffset) {
        if (taskOffset >= totalTaskCount_) {
            return;
        }

        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            tmpSparseNumGM_[k * totalTaskCount_ + taskOffset]);
        int32_t sparseNum = tmpSparseNumGM_.GetValue(k * totalTaskCount_ + taskOffset);
        if (sparseNum == 0) {
            return;
        }

        DataCopyPad(scatterIndicesLocal0_, tmpSparseIndicesGM0_[k * totalTaskCount_ + taskOffset],
            {static_cast<uint16_t>(1), static_cast<uint32_t>(sparseNum * INT32_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});
        DataCopyPad(scatterIndicesLocal1_, tmpSparseIndicesGM1_[k * totalTaskCount_ + taskOffset],
            {static_cast<uint16_t>(1), static_cast<uint32_t>(sparseNum * INT32_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});

        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);

        for (uint8_t i = 0; i < BUFFER_NUM; i++) {
            SetFlag<HardEvent::V_MTE2>(i);
            SetFlag<HardEvent::MTE3_MTE2>(i);
        }

        uint8_t flagIdx = 0;
        for (int32_t innerIdx = 0; innerIdx < sparseNum; innerIdx += innerLoopTask_) {
            int32_t innerTasks = min(innerLoopTask_, sparseNum - innerIdx);

            WaitFlag<HardEvent::V_MTE2>(flagIdx);
            WaitFlag<HardEvent::MTE3_MTE2>(flagIdx);
            DataCopyPad(scatterFeatureLocal0_[flagIdx * scatterFP32BufSize_],
                tmpMatmulResGM0_[(taskOffset + innerIdx) * inChannels_],
                {static_cast<uint16_t>(innerTasks), static_cast<uint32_t>(inChannels_ * FLOAT32_BYTE_SIZE), 0, 0, 0},
                {false, 0, 0, 0});
            DataCopyPad(scatterFeatureLocal1_[flagIdx * scatterFP32BufSize_],
                tmpMatmulResGM1_[(taskOffset + innerIdx) * inChannels_],
                {static_cast<uint16_t>(innerTasks), static_cast<uint32_t>(inChannels_ * FLOAT32_BYTE_SIZE), 0, 0, 0},
                {false, 0, 0, 0});

            SetFlag<HardEvent::MTE2_V>(flagIdx);
            WaitFlag<HardEvent::MTE2_V>(flagIdx);

            SetFlag<HardEvent::MTE2_MTE3>(flagIdx);
            WaitFlag<HardEvent::MTE2_MTE3>(flagIdx);

            callScatterFunc(outputFeaturesGradGM_, scatterIndicesLocal0_[innerIdx], scatterIndicesLocal1_[innerIdx],
                scatterFeatureLocal0_[flagIdx * scatterFP32BufSize_],
                scatterFeatureLocal1_[flagIdx * scatterFP32BufSize_], innerTasks);

            SetFlag<HardEvent::V_S>(flagIdx + BUFFER_NUM); // BUFFER_NUM <= 8
            WaitFlag<HardEvent::V_S>(flagIdx + BUFFER_NUM); // BUFFER_NUM <= 8

            SetFlag<HardEvent::V_MTE2>(flagIdx);
            SetFlag<HardEvent::MTE3_MTE2>(flagIdx);

            flagIdx = (flagIdx + 1) % BUFFER_NUM;
        }

        for (uint8_t i = 0; i < BUFFER_NUM; i++) {
            WaitFlag<HardEvent::V_MTE2>(i);
            WaitFlag<HardEvent::MTE3_MTE2>(i);
        }
    }

    __aicore__ inline void callScatterFunc(GlobalTensor<float> outputFeaturesGradGM,
        LocalTensor<int32_t> scatterIndicesLocal0, LocalTensor<int32_t> scatterIndicesLocal1,
        LocalTensor<float> scatterFeatureLocal0, LocalTensor<float> scatterFeatureLocal1, int32_t innerTasks) {
        if (inChannels_ >= SIMT_CHANNEL_THRES) {
            SetAtomicAdd<float>();
            for (int32_t idx = innerTasks - 1; idx >= 0; idx--) {
                int32_t indiceVal0 = scatterIndicesLocal0.GetValue(idx);
                int32_t indiceVal1 = scatterIndicesLocal1.GetValue(idx);

                DataCopyPad(outputFeaturesGradGM[indiceVal0 * inChannels_],
                    scatterFeatureLocal0[idx * inChannelsFP32Align_],
                    {1, static_cast<uint32_t>(inChannels_ * FLOAT32_BYTE_SIZE), 0, 0, 0});
                DataCopyPad(outputFeaturesGradGM[indiceVal1 * inChannels_],
                    scatterFeatureLocal1[idx * inChannelsFP32Align_],
                    {1, static_cast<uint32_t>(inChannels_ * FLOAT32_BYTE_SIZE), 0, 0, 0});
            }
            SetAtomicNone();

            return;
        }

        callScatterSIMTFunc((__gm__ float *)outputFeaturesGradGM.GetPhyAddr(),
            (__ubuf__ int32_t *)scatterIndicesLocal0.GetPhyAddr(),
            (__ubuf__ int32_t *)scatterIndicesLocal1.GetPhyAddr(), (__ubuf__ float *)scatterFeatureLocal0.GetPhyAddr(),
            (__ubuf__ float *)scatterFeatureLocal1.GetPhyAddr(), innerTasks);
    }

    __aicore__ inline void callScatterSIMTFunc(__gm__ float *outputFeaturesGradGM,
        __ubuf__ int32_t *scatterIndicesLocal0, __ubuf__ int32_t *scatterIndicesLocal1,
        __ubuf__ float *scatterFeatureLocal0, __ubuf__ float *scatterFeatureLocal1, int32_t innerTasks) {
        // 2**3 -- 2**6
        switch (inChannels_) {
        case 8:
            if (byteSizePerElement_ == FLOAT16_BYTE_SIZE) {
                Simt::VF_CALL<scatterAddFeatureFastSIMT<8, 16>>(Simt::Dim3{inChannels_, THREAD_NUM / inChannels_},
                    outputFeaturesGradGM, scatterIndicesLocal0, scatterIndicesLocal1, scatterFeatureLocal0,
                    scatterFeatureLocal1, innerTasks);
            } else {
                Simt::VF_CALL<scatterAddFeatureFastSIMT<8, 8>>(Simt::Dim3{inChannels_, THREAD_NUM / inChannels_},
                    outputFeaturesGradGM, scatterIndicesLocal0, scatterIndicesLocal1, scatterFeatureLocal0,
                    scatterFeatureLocal1, innerTasks);
            }
            break;
        case 16:
            Simt::VF_CALL<scatterAddFeatureFastSIMT<16, 16>>(Simt::Dim3{inChannels_, THREAD_NUM / inChannels_},
                outputFeaturesGradGM, scatterIndicesLocal0, scatterIndicesLocal1, scatterFeatureLocal0,
                scatterFeatureLocal1, innerTasks);
            break;
        case 32:
            Simt::VF_CALL<scatterAddFeatureFastSIMT<32, 32>>(Simt::Dim3{inChannels_, THREAD_NUM / inChannels_},
                outputFeaturesGradGM, scatterIndicesLocal0, scatterIndicesLocal1, scatterFeatureLocal0,
                scatterFeatureLocal1, innerTasks);
            break;
        case 64:
            Simt::VF_CALL<scatterAddFeatureFastSIMT<64, 64>>(Simt::Dim3{inChannels_, THREAD_NUM / inChannels_},
                outputFeaturesGradGM, scatterIndicesLocal0, scatterIndicesLocal1, scatterFeatureLocal0,
                scatterFeatureLocal1, innerTasks);
            break;
        default:
            Simt::VF_CALL<scatterAddFeatureSIMT>(Simt::Dim3{inChannels_, THREAD_NUM / inChannels_},
                outputFeaturesGradGM, scatterIndicesLocal0, scatterIndicesLocal1, scatterFeatureLocal0,
                scatterFeatureLocal1, innerTasks, inChannels_, inChannelsFP32Align_);
            break;
        }
    }

  protected:
    int32_t aivNum_, aicNum_, k0_, k1_, k2_, k12_, halfK_, kernelSize_, inChannels_, inChannelsAligned_,
        inChannelsFP32Align_, outChannels_, outChannelsAligned_, byteSizePerElement_, coreTaskCount_, bigCoreCount_,
        singleLoopTask_, singleLoopTaskAligned_, blockIdx_, totalTaskCount_, totalTaskAligned_, aicTaskOffset_,
        aivTaskOffset_, matmulTaskOffset_, indicesBufSize_, innerLoopTask_, featureBufSize_, gradOutFeatBufSize_,
        scatterFP32BufSize_;

    TBuf<TPosition::VECCALC> indicesBuf_, featureLocalBuf_, scatterLocalBuf_, gradOutFeatLocalBuf_;

    GlobalTensor<T> inputFeaturesGM_, inputWeightGM_, inputGradOutFeaturesGM_, tmpSparseGradOutFeatGM0_,
        tmpSparseGradOutFeatGM1_, tmpSparseFeaturesGM0_, tmpSparseFeaturesGM1_;

    GlobalTensor<int32_t> inputIndicesOffsetGM_, tmpSparseIndicesGM0_, tmpSparseIndicesGM1_, tmpSparseNumGM_;

    GlobalTensor<float> outputWeightGradGM_, outputFeaturesGradGM_, tmpMatmulResGM0_, tmpMatmulResGM1_;

    LocalTensor<int32_t> indicesLocal_, sparseIndicesLocal0_, sparseIndicesLocal1_, scatterIndicesLocal0_,
        scatterIndicesLocal1_;

    LocalTensor<T> gatherFeatureLocal0_, gatherFeatureLocal1_, gatherGradOutFeatLocal0_, gatherGradOutFeatLocal1_;

    LocalTensor<float> scatterFeatureLocal0_, scatterFeatureLocal1_;

    TPipe *pipe_;

    int32_t INT_SPACE_NUM, PROCESS_NUM_PER_STEP, BUFFER_NUM;

    static constexpr SortConfig sortConfig_ = {SortType::RADIX_SORT, true};
};

extern "C" __global__ __aicore__ void subm_sparse_conv3d_grad_v2(GM_ADDR features, GM_ADDR weight,
    GM_ADDR grad_out_features, GM_ADDR indices_offset, GM_ADDR features_grad, GM_ADDR weight_grad, GM_ADDR workspace,
    GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }

    SubmSparseConv3dGradV2<DTYPE_FEATURES> op;
    TPipe pipe;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    // must register matmul object if using matmul ops
    op.featureMatmul_.SetSubBlockIdx(0);
    op.featureMatmul_.Init(&(tiling_data.featureMatmulTilingData), &pipe);
    op.weightMatmul_.SetSubBlockIdx(0);
    op.weightMatmul_.Init(&(tiling_data.weightMatmulTilingData), &pipe);

    op.Init(&pipe, features, weight, grad_out_features, indices_offset, features_grad, weight_grad, usrWorkspace,
        &tiling_data);
    op.Process();
}
