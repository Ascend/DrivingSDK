#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "sparse_conv3d_grad_simt.h"
using namespace AscendC;

namespace {
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t INT32_SIZE = 4;
constexpr int32_t INT32_BLOCK_NUM = BLOCK_SIZE / INT32_SIZE;
constexpr int32_t INT8_PER_LOOP = 256;

constexpr MatmulConfig SPARSE_CONV3D_CFG = GetNormalConfig(); // 替换config
} // namespace

template<typename T>
class SparseConv3dGrad {
public:
    using weightMatType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using imgToColMatType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using gradOutMatType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using weightGradMatType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using featureGradMatType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>; // for scatteradd

    matmul::MatmulImpl<gradOutMatType, weightMatType, featureGradMatType, featureGradMatType, SPARSE_CONV3D_CFG>
        featureMatmul_;
    matmul::MatmulImpl<imgToColMatType, gradOutMatType, weightGradMatType, weightGradMatType, SPARSE_CONV3D_CFG>
        weightMatmul_;

    __aicore__ inline SparseConv3dGrad() {};

    __aicore__ inline void Init(TPipe* pipe, GM_ADDR features, GM_ADDR weight, GM_ADDR grad_out_features,
        GM_ADDR former_sorted_indices, GM_ADDR indices_offset, GM_ADDR features_grad, GM_ADDR weight_grad,
        GM_ADDR usrWorkspace, SparseConv3dGradTilingData* tilingData)
    {
        pipe_ = pipe;
        InitTiling(tilingData);
        InitBuffer(features, weight, grad_out_features, former_sorted_indices, indices_offset, features_grad,
            weight_grad, usrWorkspace);
    }

    __aicore__ inline void Process();

protected:
    TPipe* pipe_;
    GlobalTensor<T> featuresGM_, weightGM_, gradOutFeaturesGM_, featuresGradGM_, weightGradGM_;
    GlobalTensor<T> featureWsp_, gradOutWsp_, tempGradFeatureWsp_;
    GlobalTensor<int32_t> sortedIndicesGM_, indicesOffsetGM_;
    GlobalTensor<int32_t> inputIdxWsp_, outIdxWsp_, sparseNumWsp_;
    GlobalTensor<uint8_t> kIdxWsp_;

    TBuf<TPosition::VECCALC> tmpInputBuf_, tmpGradFeaturesBuf_, indexBuf_, kIdxBuf_;

    LocalTensor<T> tmpGradOutLocal_, tmpFeaturesLocal_;
    LocalTensor<T> tmpGradFeaturesLocal_;
    LocalTensor<int32_t> inputIdxLocal_, outIdxLocal_, inputIdxPtrLocal_, idxInfoLocal_;
    LocalTensor<uint8_t> kIdxLocal_, maskLocal_;

    uint32_t blockIdx_, aicNum_, aivNum_, featureByteSize_, blockDataNum_, loopPointCount_;
    uint32_t usedVectorNum_, kernelSize_, inChannels_, outChannels_, totalTaskNum_, sparseRatio_, ubMaxTaskNum_;
    int32_t totalPointsCount_, startOffset_;
    uint64_t featureWspOffset_;
    uint32_t sparseWspOffset_;

private:
    __aicore__ inline void InitTiling(SparseConv3dGradTilingData* tilingData);

    __aicore__ inline void InitBuffer(GM_ADDR features, GM_ADDR weight, GM_ADDR grad_out_features,
        GM_ADDR former_sorted_indices, GM_ADDR indices_offset, GM_ADDR features_grad, GM_ADDR weight_grad,
        GM_ADDR usrWorkspace);

    __aicore__ inline void CopyInHashMap(uint32_t startOffset, int32_t pointCount);

    __aicore__ inline void CopyInFeature(uint32_t bitLoopNum, const int32_t pointCount, int32_t& totalSparseM);

    __aicore__ inline void CalGradFeaturesMatmul(uint8_t k, int32_t sparseM, uint8_t subBlockIdx);

    __aicore__ inline void CalGradWeightMatmul(uint8_t k, int32_t sparseM, uint8_t subBlockIdx);

    __aicore__ inline void GradFeaturesScatterAdd(uint32_t sparseM);
};

template<typename T>
__aicore__ inline void SparseConv3dGrad<T>::InitTiling(SparseConv3dGradTilingData* tilingData)
{
    blockIdx_ = GetBlockIdx();
    aicNum_ = GetBlockNum();
    aivNum_ = aicNum_ * 2;
    featureByteSize_ = sizeof(T);

    blockDataNum_ = BLOCK_SIZE / featureByteSize_;
    usedVectorNum_ = tilingData->usedVectorNum;
    kernelSize_ = tilingData->kernelSize;
    inChannels_ = tilingData->inChannels;
    outChannels_ = tilingData->outChannels;
    totalTaskNum_ = tilingData->totalTaskNum;
    totalPointsCount_ = tilingData->totalPointsCount;
    startOffset_ = tilingData->startOffset;

    ubMaxTaskNum_ = tilingData->ubMaxTaskNum;
    loopPointCount_ = tilingData->loopPointCount;
    featureWspOffset_ = tilingData->featureWspOffset;
    sparseWspOffset_ = tilingData->sparseWspOffset;
}

template<typename T>
__aicore__ inline void SparseConv3dGrad<T>::InitBuffer(GM_ADDR features, GM_ADDR weight, GM_ADDR grad_out_features,
    GM_ADDR former_sorted_indices, GM_ADDR indices_offset, GM_ADDR features_grad, GM_ADDR weight_grad,
    GM_ADDR usrWorkspace)
{
    featuresGM_.SetGlobalBuffer((__gm__ T*)features);
    weightGM_.SetGlobalBuffer((__gm__ T*)weight);
    gradOutFeaturesGM_.SetGlobalBuffer((__gm__ T*)grad_out_features);
    sortedIndicesGM_.SetGlobalBuffer((__gm__ int32_t*)former_sorted_indices);
    indicesOffsetGM_.SetGlobalBuffer((__gm__ int32_t*)indices_offset);

    featuresGradGM_.SetGlobalBuffer((__gm__ T*)features_grad);
    weightGradGM_.SetGlobalBuffer((__gm__ T*)weight_grad);

    pipe_->InitBuffer(tmpInputBuf_, ubMaxTaskNum_ * (outChannels_ + inChannels_) * featureByteSize_);
    pipe_->InitBuffer(tmpGradFeaturesBuf_, ubMaxTaskNum_ * inChannels_ * featureByteSize_);
    pipe_->InitBuffer(indexBuf_, 3 * loopPointCount_ * INT32_SIZE + BLOCK_SIZE);

    tmpGradOutLocal_ = tmpInputBuf_.Get<T>();
    tmpFeaturesLocal_ = tmpGradOutLocal_[ubMaxTaskNum_ * outChannels_];
    tmpGradFeaturesLocal_ = tmpGradFeaturesBuf_.Get<T>();

    inputIdxLocal_ = indexBuf_.Get<int32_t>();
    outIdxLocal_ = inputIdxLocal_[loopPointCount_];
    inputIdxPtrLocal_ = outIdxLocal_[loopPointCount_];
    idxInfoLocal_ = inputIdxPtrLocal_[loopPointCount_]; // record totalPointsCount

    uint32_t maskAlignLen_ = AlignUp(loopPointCount_, INT8_PER_LOOP); // align 256B for CompareScalar, 1B for uint8_ts
    pipe_->InitBuffer(kIdxBuf_, maskAlignLen_ + maskAlignLen_ / 8);
    kIdxLocal_ = kIdxBuf_.Get<uint8_t>();
    maskLocal_ = kIdxLocal_[maskAlignLen_];

    if ASCEND_IS_AIV {
        featureWsp_.SetGlobalBuffer((__gm__ T*)(usrWorkspace) + blockIdx_ * featureWspOffset_);
    }
    if ASCEND_IS_AIC {
        featureWsp_.SetGlobalBuffer((__gm__ T*)(usrWorkspace) + blockIdx_ * 2 * featureWspOffset_);
    }
    tempGradFeatureWsp_ = featureWsp_[loopPointCount_ * inChannels_];
    gradOutWsp_ = tempGradFeatureWsp_[loopPointCount_ * inChannels_];

    inputIdxWsp_.SetGlobalBuffer(
        (__gm__ int32_t*)(usrWorkspace) + usedVectorNum_ * featureWspOffset_); // only true for float32
    outIdxWsp_ = inputIdxWsp_[totalPointsCount_];
    sparseNumWsp_ = outIdxWsp_[totalPointsCount_];
    
    kIdxWsp_ = sparseNumWsp_[sparseWspOffset_].template ReinterpretCast<uint8_t>();
}

template<typename T>
__aicore__ inline void SparseConv3dGrad<T>::Process()
{
    if ASCEND_IS_AIV {
        Simt::VF_CALL<PrepareGlobalHashMap>(Simt::Dim3 {THREAD_NUM}, (__gm__ int32_t*)sortedIndicesGM_.GetPhyAddr(),
            (__gm__ int32_t*)indicesOffsetGM_.GetPhyAddr(), (__gm__ int32_t*)inputIdxWsp_.GetPhyAddr(),
            (__gm__ int32_t*)outIdxWsp_.GetPhyAddr(), (__gm__ uint8_t*)kIdxWsp_.GetPhyAddr(), totalTaskNum_,
            kernelSize_, startOffset_);
        SyncAll();

        for (int32_t pointIdx = blockIdx_ * loopPointCount_; pointIdx < totalPointsCount_;
            pointIdx += loopPointCount_ * usedVectorNum_) {
            int32_t ubPointCount = min(totalPointsCount_ - pointIdx, (int32_t)(loopPointCount_));

            CopyInHashMap(pointIdx, ubPointCount);
            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);

            uint32_t maskAlign = AlignUp(ubPointCount, INT8_PER_LOOP);
            uint32_t bitLoopCount = DivCeil(ubPointCount, 64);
            for (uint8_t kIdx = 0; kIdx < kernelSize_; ++kIdx) { // uint8_t, kernelSize_ must less than 128
                uint16_t flagId = kIdx % 8;
                CompareScalar(maskLocal_, kIdxLocal_, kIdx, CMPMODE::EQ, maskAlign);
                SetFlag<HardEvent::V_S>(0);
                WaitFlag<HardEvent::V_S>(0);

                int32_t totalSparseNum = 0;
                CopyInFeature(bitLoopCount, ubPointCount, totalSparseNum);

                idxInfoLocal_.SetValue(0, totalSparseNum);
                SetFlag<HardEvent::S_MTE3>(flagId);
                WaitFlag<HardEvent::S_MTE3>(flagId);
                DataCopyPad(sparseNumWsp_[blockIdx_ * kernelSize_ + kIdx], idxInfoLocal_,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(1 * INT32_SIZE), 0, 0, 0});
                CrossCoreSetFlag<0x4, PIPE_MTE3>(flagId);

                CrossCoreWaitFlag<0x4>(flagId);
                GradFeaturesScatterAdd(totalSparseNum); // wait featurematmul before scatter add
            }
        }
    }

    if ASCEND_IS_AIC {
        for (uint32_t aicTaskOffset = 2 * loopPointCount_ * blockIdx_; aicTaskOffset < totalPointsCount_;
            aicTaskOffset += 2 * loopPointCount_ * aicNum_) {
            bool sub1Used = (totalPointsCount_ - aicTaskOffset > loopPointCount_) ? true : false;
            for (uint8_t kIdx = 0; kIdx < kernelSize_; ++kIdx) { // uint8_t, kernelSize_ must less than 128
                uint16_t sub0FlagId = kIdx % 8;
                uint16_t sub1FlagId = sub0FlagId + 16;

                CrossCoreWaitFlag<0x4>(sub0FlagId);
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
                    sparseNumWsp_[2 * blockIdx_ * kernelSize_ + kIdx]);
                int32_t sub0SparseNum = sparseNumWsp_.GetValue(2 * blockIdx_ * kernelSize_ + kIdx);

                CalGradFeaturesMatmul(kIdx, sub0SparseNum, 0);
                CalGradWeightMatmul(kIdx, sub0SparseNum, 0);
                CrossCoreSetFlag<0x4, PIPE_FIX>(sub0FlagId);

                if (sub1Used) {
                    CrossCoreWaitFlag<0x4>(sub1FlagId);
                    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
                        sparseNumWsp_[(2 * blockIdx_ + 1) * kernelSize_ + kIdx]);
                    int32_t sub1SparseNum = sparseNumWsp_.GetValue((2 * blockIdx_ + 1) * kernelSize_ + kIdx);
                    CalGradFeaturesMatmul(kIdx, sub1SparseNum, 1);
                    CalGradWeightMatmul(kIdx, sub1SparseNum, 1);
                    CrossCoreSetFlag<0x4, PIPE_FIX>(sub1FlagId);
                }
            }
        }
    }
}
template<typename T>
__aicore__ inline void SparseConv3dGrad<T>::CopyInHashMap(uint32_t startOffset, int32_t pointCount)
{
    uint32_t moveAlign = (pointCount == loopPointCount_) ? loopPointCount_ : AlignUp(pointCount, INT32_BLOCK_NUM);
    uint32_t kIdxAlign = AlignUp(moveAlign, INT8_PER_LOOP);

    DataCopy(kIdxLocal_, kIdxWsp_[startOffset], kIdxAlign);
    DataCopy(inputIdxLocal_, inputIdxWsp_[startOffset], moveAlign);
    DataCopy(outIdxLocal_, outIdxWsp_[startOffset], moveAlign);
}

template<typename T>
__aicore__ inline void SparseConv3dGrad<T>::CopyInFeature(
    uint32_t bitLoopNum, const int32_t pointCount, int32_t& totalSparseM)
{
    int32_t sparseM = 0;
    for (uint32_t outerIdx = 0; outerIdx < bitLoopNum; ++outerIdx) {
        uint64_t maskValue = maskLocal_.ReinterpretCast<uint64_t>().GetValue(outerIdx);
        uint32_t innerLoopCount = outerIdx == (bitLoopNum - 1) ? (pointCount - 64 * outerIdx) : 64;
        for (uint32_t innerIdx = ScalarGetSFFValue<1>(maskValue); innerIdx < innerLoopCount && innerIdx >= 0;
            innerIdx = ScalarGetSFFValue<1>(maskValue)) {
            maskValue = sbitset0(maskValue, innerIdx);
            uint32_t bitIdx = outerIdx * 64 + innerIdx;
            uint32_t inputIdx = inputIdxLocal_.GetValue(bitIdx);
            uint32_t outIdx = outIdxLocal_.GetValue(bitIdx);

            inputIdxPtrLocal_.SetValue(totalSparseM + sparseM, inputIdx);
            DataCopy(tmpGradOutLocal_[sparseM * outChannels_], gradOutFeaturesGM_[outIdx * outChannels_], outChannels_);
            DataCopy(tmpFeaturesLocal_[sparseM * inChannels_], featuresGM_[inputIdx * inChannels_], inChannels_);
            sparseM++;

            if (sparseM == ubMaxTaskNum_) {
                SetFlag<HardEvent::MTE2_MTE3>(0);
                WaitFlag<HardEvent::MTE2_MTE3>(0);

                DataCopy(gradOutWsp_[totalSparseM * outChannels_], tmpGradOutLocal_, sparseM * outChannels_);
                DataCopy(featureWsp_[totalSparseM * inChannels_], tmpFeaturesLocal_, sparseM * inChannels_);
                totalSparseM += sparseM;
                sparseM = 0;

                SetFlag<HardEvent::MTE3_MTE2>(0);
                WaitFlag<HardEvent::MTE3_MTE2>(0);
            }
        }
    }
    if (sparseM > 0) {
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);
        DataCopy(gradOutWsp_[totalSparseM * outChannels_], tmpGradOutLocal_, sparseM * outChannels_);
        DataCopy(featureWsp_[totalSparseM * inChannels_], tmpFeaturesLocal_, sparseM * inChannels_);
        totalSparseM += sparseM;
    }
}

template<typename T>
__aicore__ inline void SparseConv3dGrad<T>::GradFeaturesScatterAdd(uint32_t m)
{
    if (m <= 0) {
        return;
    }
    SetFlag<HardEvent::MTE3_MTE2>(1);
    for (int32_t idxM = 0; idxM < m; idxM += ubMaxTaskNum_) {
        uint32_t loopCount = min(ubMaxTaskNum_, m - idxM);
        WaitFlag<HardEvent::MTE3_MTE2>(1);
        DataCopy(tmpGradFeaturesLocal_, tempGradFeatureWsp_[idxM * inChannels_], loopCount * inChannels_);

        SetFlag<HardEvent::MTE2_MTE3>(1);
        WaitFlag<HardEvent::MTE2_MTE3>(1);
        for (uint32_t i = 0; i < loopCount; ++i) {
            uint32_t inpIdx = inputIdxPtrLocal_.GetValue(idxM + i);

            SetAtomicAdd<T>();
            DataCopy(featuresGradGM_[inpIdx * inChannels_], tmpGradFeaturesLocal_[i * inChannels_], inChannels_);
            SetAtomicNone();
        }
        SetFlag<HardEvent::MTE3_MTE2>(1);
    }
    WaitFlag<HardEvent::MTE3_MTE2>(1);
}

template<typename T>
__aicore__ inline void SparseConv3dGrad<T>::CalGradFeaturesMatmul(uint8_t k, int32_t sparseM, uint8_t subBlockIdx)
{
    // gradOutWorkSpace[T, outChannels_] @ weightGM_[inChannels_, outChannels_].T = tmpGradFeaturesWorkSpace[T, inChannels_]
    if (sparseM <= 0) {
        return;
    }
    featureMatmul_.SetOrgShape(sparseM, inChannels_, outChannels_);
    featureMatmul_.SetTensorA(gradOutWsp_[featureWspOffset_ * subBlockIdx]);
    featureMatmul_.SetTensorB(weightGM_[k * inChannels_ * outChannels_], true);

    featureMatmul_.SetSingleShape(sparseM, inChannels_, outChannels_);
    featureMatmul_.template IterateAll<false>(tempGradFeatureWsp_[featureWspOffset_ * subBlockIdx], 0);
    featureMatmul_.End();
}

template<typename T>
__aicore__ inline void SparseConv3dGrad<T>::CalGradWeightMatmul(uint8_t k, int32_t sparseM, uint8_t subBlockIdx)
{
    if (sparseM <= 0) {
        return;
    }
    // @ featuresWorkSpace[T, inChannels_].T @ gradOutWorkSpace[T, outChannels_] = weightGradGM_[inChannels_, outChannel_]
    weightMatmul_.SetOrgShape(inChannels_, outChannels_, sparseM);
    weightMatmul_.SetTensorA(featureWsp_[featureWspOffset_ * subBlockIdx], true);
    weightMatmul_.SetTensorB(gradOutWsp_[featureWspOffset_ * subBlockIdx]);
    weightMatmul_.SetSingleShape(inChannels_, outChannels_, sparseM);

    weightMatmul_.template IterateAll<false>(weightGradGM_[k * inChannels_ * outChannels_], 1);
    weightMatmul_.End();
}

extern "C" __global__ __aicore__ void sparse_conv3d_grad(GM_ADDR features, GM_ADDR weight, GM_ADDR grad_out_features,
    GM_ADDR former_sorted_indices, GM_ADDR indices_offset, GM_ADDR features_grad, GM_ADDR weight_grad,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }
    TPipe pipe;
    SparseConv3dGrad<DTYPE_FEATURES> op;

    op.featureMatmul_.SetSubBlockIdx(0);
    op.featureMatmul_.Init(&tiling_data.featureMatmulTilingData, &pipe);
    op.weightMatmul_.SetSubBlockIdx(0);
    op.weightMatmul_.Init(&tiling_data.weightMatmulTilingData, &pipe);
    op.Init(&pipe, features, weight, grad_out_features, former_sorted_indices, indices_offset, features_grad,
        weight_grad, usrWorkspace, &tiling_data);
    op.Process();
}