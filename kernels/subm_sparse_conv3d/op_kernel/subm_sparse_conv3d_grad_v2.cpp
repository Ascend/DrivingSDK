#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;

namespace {
    constexpr int32_t BYTE_SIZE_PER_BLOCK = 32;
    constexpr int32_t INT32_BYTE_SIZE = 4;
    constexpr int32_t FLOAT32_BYTE_SIZE = 4;
    constexpr int32_t FLOAT16_BYTE_SIZE = 2;
    constexpr MatmulConfig SUBM_SPARSE_CONV3D_CFG = GetNormalConfig();
    const int32_t INT_SPACE_NUM = 3;
    const int32_t INCHANNELS_BUF_NUM = 2;

    const int32_t SPARSE_CONTIGUOUS_IDX = 2;
}

template<typename T>
class SubmSparseConv3dGradV2{

public:
    using weightMatType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using imgToColMatType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using gradOutFeaturesMatType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using weightGradMatType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using gradFeatureIndicesMatType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;

    matmul::Matmul<gradOutFeaturesMatType, weightMatType, gradFeatureIndicesMatType, gradFeatureIndicesMatType, SUBM_SPARSE_CONV3D_CFG> featureMatmul_;
    matmul::Matmul<imgToColMatType, gradOutFeaturesMatType, weightGradMatType, weightGradMatType, SUBM_SPARSE_CONV3D_CFG> weightMatmul_;

    __aicore__ inline SubmSparseConv3dGradV2() {};

    __aicore__ inline void Init(TPipe *pipe, GM_ADDR features, GM_ADDR weight, GM_ADDR grad_out_features, GM_ADDR indices_offset, 
        GM_ADDR features_grad, GM_ADDR weight_grad, GM_ADDR usrWorkspace, SubmConv3dGradV2TillingData *tilingData)
    {
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        InitTiling(tilingData);
        InitGM(features, weight, grad_out_features, indices_offset, features_grad, weight_grad, usrWorkspace);
        InitUB();
    }

    __aicore__ inline void Process() {
        calCenterFeatureMatmul();
        calCenterWeightMatmul();

        for (int32_t k = 0; k< kernelSize_; k++) {
            if (k == centerK_) {
                continue;
            }
            // get sparse indices for each batch
            getSparseIndices(k);
            // calculate sparse feature matmul
            calSparseFeatureMatmul(k);
            // calculate sparse weight matmul
            calSpraseWeightMatmul(k);
            // wait featurematmul before scatter add
            if (sparseNum_ > 0) {
                featureMatmul_.WaitIterateAll();
            }
            // scatter_add and get features_grad
            scatterAddSparseFeatures();
            // wait weightmatmul before get sparse indices
            if (sparseNum_ > 0) {
                weightMatmul_.WaitIterateAll();
            }
        }
        weightMatmul_.End();
        featureMatmul_.End();
    }
    

private:

    __aicore__ inline void InitTiling(SubmConv3dGradV2TillingData *tilingData) {
        byteSizePerElement_ = sizeof(T);
        aivNum_ = tilingData->aivNum;
        k0_ = tilingData->k0;
        k1_ = tilingData->k1;
        k2_ = tilingData->k2;
        inChannels_ = tilingData->inChannels;
        outChannels_ = tilingData->outChannels;
        totalTaskCount_ = tilingData->totalTaskCount;
        coreTaskCount_ = tilingData->coreTaskCount;
        bigCoreCount_ = tilingData->bigCoreCount;
        singleLoopTask_ = tilingData->singleLoopTask;

        k12_ = k1_ * k2_;
        kernelSize_ = k0_ * k12_;
        centerK_ = kernelSize_ / 2;
        inChannelsAligned_ = AlignUp(inChannels_, BYTE_SIZE_PER_BLOCK / byteSizePerElement_);
        outChannelsAlinged_ = AlignUp(outChannels_, BYTE_SIZE_PER_BLOCK / byteSizePerElement_);
        singleLoopTaskAligned_ = AlignUp(singleLoopTask_, BYTE_SIZE_PER_BLOCK / INT32_BYTE_SIZE);
        totalTaskAligned_ = AlignUp(totalTaskCount_, BYTE_SIZE_PER_BLOCK / byteSizePerElement_);

        if (blockIdx_ < bigCoreCount_) {
            globalTaskOffset_ = (coreTaskCount_ + 1) * blockIdx_;
            coreTaskCount_ = coreTaskCount_ + 1;
        } else {
            globalTaskOffset_ = (coreTaskCount_ + 1) * bigCoreCount_ + coreTaskCount_ * (blockIdx_ - bigCoreCount_);
        }
        tmpFeaturesBufIdx_ = 0;
        tmpSparseBufIdx_ = 0;
        globalTaskStartIdx_ = globalTaskOffset_;
        globalTaskEndIdx_ = globalTaskStartIdx_ + coreTaskCount_;

    }

    __aicore__ inline void InitGM(GM_ADDR features, GM_ADDR weight, GM_ADDR grad_out_features, GM_ADDR indices_offset,
        GM_ADDR features_grad, GM_ADDR weight_grad, GM_ADDR usrWorkspace)
    {
        inputFeaturesGM_.SetGlobalBuffer((__gm__ T*) features);
        inputWeightGM_.SetGlobalBuffer((__gm__ T*) weight);
        inputGradOutFeaturesGM_.SetGlobalBuffer((__gm__ T*) grad_out_features);
        inputIndicesOffsetGM_.SetGlobalBuffer((__gm__ int32_t*) indices_offset);
        outputFeaturesGradGM_.SetGlobalBuffer((__gm__ T*) features_grad);
        outputWeightGradGM_.SetGlobalBuffer((__gm__ float*) weight_grad);
        
        int64_t offset1 = globalTaskStartIdx_ * inChannels_;
        int64_t offset2 = (globalTaskStartIdx_ + totalTaskCount_) * inChannels_;
        int64_t offset3 = 2 * totalTaskCount_ * inChannels_ + globalTaskStartIdx_ * outChannels_;
        int64_t offset4 = (totalTaskCount_ * (2 * inChannels_ + outChannels_) + INT32_BYTE_SIZE / byteSizePerElement_ -1) / 
            (INT32_BYTE_SIZE / byteSizePerElement_) + globalTaskStartIdx_;

        tmpSparseFeaturesGM_.SetGlobalBuffer((__gm__ T*)(usrWorkspace) + offset1, coreTaskCount_ * inChannels_);
        tmpFeatureMatmulResGM_.SetGlobalBuffer((__gm__ T*)(usrWorkspace) + offset2, coreTaskCount_ * inChannels_);
        tmpSparseGradOutFeaturesGM_.SetGlobalBuffer((__gm__ T*)(usrWorkspace) + offset3, coreTaskCount_ * outChannels_);
        tmpSparseIndicesGM_.SetGlobalBuffer((__gm__ int32_t*)(usrWorkspace) + offset4, coreTaskCount_ );
    }

    __aicore__ inline void InitUB() {
        pipe_->InitBuffer(tmpFeaturesBuf_, INCHANNELS_BUF_NUM * singleLoopTask_ * inChannelsAligned_ * byteSizePerElement_);
        pipe_->InitBuffer(tmpSparseGradOutFeaturesBuf_, singleLoopTask_ * outChannelsAlinged_ * byteSizePerElement_);
        pipe_->InitBuffer(tmpSparseIndicesBuf_, INT_SPACE_NUM * singleLoopTaskAligned_ * INT32_BYTE_SIZE);
        
        tmpFeaturesLocal_ = tmpFeaturesBuf_.Get<T>();
        tmpFeaturesLocalBak_ = tmpFeaturesLocal_[singleLoopTask_ * inChannelsAligned_];

        tmpSparseGradOutFeaturesLocal_ = tmpSparseGradOutFeaturesBuf_.Get<T>();

        tmpSparseIndicesLocal_ = tmpSparseIndicesBuf_.Get<int32_t>();
        tmpSparseIndicesLocalBak_ = tmpSparseIndicesLocal_[singleLoopTaskAligned_];
        tmpSparseIndicesLocalContiguous_ = tmpSparseIndicesLocal_[SPARSE_CONTIGUOUS_IDX * singleLoopTaskAligned_];

    }


    __aicore__ inline void getSparseIndices(int32_t k) {

        sparseNum_ = 0;
        tmpSparseBufIdx_ = 0;
        tmpFeaturesBufIdx_ = 0;
        
        for (int32_t idx = globalTaskStartIdx_; idx < coreTaskCount_ + globalTaskStartIdx_; idx += singleLoopTask_) {
            int32_t curTaskCount = min(singleLoopTask_, globalTaskStartIdx_ + coreTaskCount_ - idx);

            DataCopyPad<int32_t>(tmpSparseIndicesLocal_, inputIndicesOffsetGM_[k * totalTaskCount_ + idx],
                {1, static_cast<uint32_t>(curTaskCount * INT32_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});

            SetFlag<HardEvent::MTE2_V>(0);
            WaitFlag<HardEvent::MTE2_V>(0);

            for (int taskIdx = 0; taskIdx < curTaskCount; taskIdx++) {
                int32_t indiceVal = tmpSparseIndicesLocal_.GetValue(taskIdx);

                if (indiceVal >= 0) {

                    SetFlag<HardEvent::V_S>(0);
                    WaitFlag<HardEvent::V_S>(0);

                    SetFlag<HardEvent::V_MTE2>(0);
                    WaitFlag<HardEvent::V_MTE2>(0);

                    sparseNum_++;
                    getSparseFeatures(indiceVal, taskIdx + idx);
                    
                    if (sparseNum_ - tmpSparseBufIdx_ == singleLoopTask_) {
                        SetFlag<HardEvent::V_MTE3>(0);
                        WaitFlag<HardEvent::V_MTE3>(0);

                        SetFlag<HardEvent::MTE2_MTE3>(0);
                        WaitFlag<HardEvent::MTE2_MTE3>(0);

                        copyInSparseData(singleLoopTask_);
                        tmpSparseBufIdx_ += singleLoopTask_;

                        SetFlag<HardEvent::MTE3_MTE2>(0);
                        WaitFlag<HardEvent::MTE3_MTE2>(0);

                        SetFlag<HardEvent::MTE3_V>(0);
                        WaitFlag<HardEvent::MTE3_V>(0);
                    }
                }
            }
        }

        if (sparseNum_ > tmpSparseBufIdx_) {
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);

            SetFlag<HardEvent::MTE2_MTE3>(0);
            WaitFlag<HardEvent::MTE2_MTE3>(0);

            copyInSparseData(sparseNum_ - tmpSparseBufIdx_);
            tmpSparseBufIdx_ += sparseNum_ - tmpSparseBufIdx_;

        }

    }


    __aicore__ inline void getSparseFeatures(int32_t indiceVal, int32_t taskIdx) {
        tmpSparseIndicesLocalContiguous_.SetValue(tmpFeaturesBufIdx_, indiceVal);

        DataCopyPad(tmpSparseGradOutFeaturesLocal_[tmpFeaturesBufIdx_ * outChannelsAlinged_], inputGradOutFeaturesGM_[taskIdx * outChannels_],
            {1, static_cast<uint32_t>(outChannels_ * byteSizePerElement_), 0, 0, 0}, {false, 0, 0, 0});
        DataCopyPad(tmpFeaturesLocal_[tmpFeaturesBufIdx_ * inChannelsAligned_], inputFeaturesGM_[indiceVal * inChannels_],
            {1, static_cast<uint32_t>(inChannels_ * byteSizePerElement_), 0, 0, 0}, {false, 0, 0, 0});
        
        tmpFeaturesBufIdx_ = (tmpFeaturesBufIdx_ + 1) % singleLoopTask_;
    }

    __aicore__ inline void copyInSparseData(int32_t taskNum) {
        DataCopyPad(tmpSparseGradOutFeaturesGM_[tmpSparseBufIdx_ * outChannels_], tmpSparseGradOutFeaturesLocal_,
            {static_cast<uint16_t>(taskNum), static_cast<uint32_t>(outChannels_ * byteSizePerElement_), 0, 0, 0});
        DataCopyPad(tmpSparseFeaturesGM_[tmpSparseBufIdx_ * inChannels_], tmpFeaturesLocal_,
            {static_cast<uint16_t>(taskNum), static_cast<uint32_t>(inChannels_ * byteSizePerElement_), 0, 0, 0});
        DataCopyPad(tmpSparseIndicesGM_[tmpSparseBufIdx_], tmpSparseIndicesLocalContiguous_,
            {1, static_cast<uint32_t>(taskNum * INT32_BYTE_SIZE), 0, 0, 0});
    }

    __aicore__ inline void calCenterWeightMatmul() {
        if (coreTaskCount_ == 0) {
            return ;
        }
        weightMatmul_.SetTensorA(inputFeaturesGM_[globalTaskStartIdx_ * inChannels_], true);
        weightMatmul_.SetTensorB(inputGradOutFeaturesGM_[globalTaskStartIdx_ * outChannels_]);
        weightMatmul_.SetSingleShape(inChannels_, outChannels_, coreTaskCount_);

        weightMatmul_.template IterateAll<false>(outputWeightGradGM_[centerK_ * inChannels_ * outChannels_], 1);
    }

    __aicore__ inline void calSpraseWeightMatmul(int32_t k) {
        if (sparseNum_ == 0) {
            return ;
        }
        weightMatmul_.SetTensorA(tmpSparseFeaturesGM_, true);
        weightMatmul_.SetTensorB(tmpSparseGradOutFeaturesGM_);
        weightMatmul_.SetSingleShape(inChannels_, outChannels_, sparseNum_);

        weightMatmul_.template IterateAll<false>(outputWeightGradGM_[k * inChannels_ * outChannels_], 1, false, true);
    }

    __aicore__ inline void calCenterFeatureMatmul() {
        if (coreTaskCount_ == 0) {
            return ;
        }
        featureMatmul_.SetTensorA(inputGradOutFeaturesGM_[globalTaskStartIdx_ * outChannels_]);
        featureMatmul_.SetTensorB(inputWeightGM_[centerK_ * inChannels_ * outChannels_], true);
        featureMatmul_.SetSingleShape(coreTaskCount_, inChannels_, outChannels_);

        featureMatmul_.template IterateAll<false>(outputFeaturesGradGM_[globalTaskStartIdx_ * inChannels_], 1);
    }

    __aicore__ inline void calSparseFeatureMatmul(int32_t k) {
        if (sparseNum_ == 0) {
            return ;
        }
        featureMatmul_.SetTensorA(tmpSparseGradOutFeaturesGM_);
        featureMatmul_.SetTensorB(inputWeightGM_[k * inChannels_ * outChannels_], true);
        featureMatmul_.SetSingleShape(sparseNum_, inChannels_, outChannels_);

        featureMatmul_.template IterateAll<false>(tmpFeatureMatmulResGM_, 0, false, true);
    }

    __aicore__ inline void scatterAddSparseFeatures() {
        SetFlag<HardEvent::MTE3_MTE2>(1);

        for (int32_t idx = 0; idx < sparseNum_; idx += singleLoopTask_) {
            int32_t curTaskCount = min(singleLoopTask_, sparseNum_ - idx);

            WaitFlag<HardEvent::MTE3_MTE2>(1);

            DataCopyPad(tmpSparseIndicesLocalBak_, tmpSparseIndicesGM_[idx], 
                {1, static_cast<uint32_t>(curTaskCount * INT32_BYTE_SIZE), 0, 0, 0}, {false, 0, 0, 0});
            DataCopyPad(tmpFeaturesLocalBak_, tmpFeatureMatmulResGM_[idx * inChannels_],
                {static_cast<uint16_t>(curTaskCount), static_cast<uint32_t>(inChannels_) * byteSizePerElement_, 0, 0, 0}, {false, 0, 0, 0});

            SetFlag<HardEvent::MTE2_S>(0);
            WaitFlag<HardEvent::MTE2_S>(0);

            for (int32_t taskIdx = 0; taskIdx < curTaskCount; taskIdx++) {
                int32_t indiceVal = tmpSparseIndicesLocalBak_.GetValue(taskIdx);

                SetAtomicAdd<T>();
                DataCopyPad(outputFeaturesGradGM_[indiceVal * inChannels_], tmpFeaturesLocalBak_[taskIdx * inChannelsAligned_],
                    {1, static_cast<uint32_t>(inChannels_) * byteSizePerElement_, 0, 0, 0});
                SetAtomicNone();

            }
            
            SetFlag<HardEvent::MTE3_MTE2>(1);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(1);
    }


protected:

    int32_t aivNum_, 
            k0_, 
            k1_, 
            k2_, 
            k12_, 
            centerK_,
            kernelSize_, 
            inChannels_, 
            inChannelsAligned_, 
            outChannels_, 
            outChannelsAlinged_, 
            byteSizePerElement_, 
            coreTaskCount_, 
            bigCoreCount_, 
            totalTaskCount_, 
            totalTaskAligned_, 
            singleLoopTask_, 
            singleLoopTaskAligned_, 
            sparseNum_, 
            tmpSparseBufIdx_,
            globalTaskOffset_, 
            globalTaskStartIdx_, 
            globalTaskEndIdx_,
            tmpFeaturesBufIdx_, 
            blockIdx_;

    GlobalTensor<T> inputFeaturesGM_, 
                    inputWeightGM_, 
                    inputGradOutFeaturesGM_, 
                    tmpFeatureMatmulResGM_,
                    outputFeaturesGradGM_, 
                    tmpSparseGradOutFeaturesGM_, 
                    tmpSparseFeaturesGM_;

    GlobalTensor<int32_t> inputIndicesOffsetGM_, 
                          tmpSparseIndicesGM_;

    GlobalTensor<float> outputWeightGradGM_;

    LocalTensor<T> tmpFeaturesLocal_, 
                   tmpSparseGradOutFeaturesLocal_, 
                   tmpFeaturesLocalBak_;

    LocalTensor<float> tmpValFloatLocal_, 
                       tmpValResLocal_, 
                       reduceTmpBufuer_, 
                       tmpFeaturesLocalBakCast_;

    LocalTensor<int32_t> tmpSparseIndicesLocal_, 
                         tmpSparseIndicesLocalBak_, 
                         tmpSparseIndicesLocalContiguous_;

    TBuf<TPosition::VECCALC> tmpFeaturesBuf_, 
                             tmpSparseGradOutFeaturesBuf_, 
                             tmpSparseIndicesBuf_, 
                             tmpValFloatBuf_;
    
    TPipe* pipe_;
};

extern "C" __global__ __aicore__ void subm_sparse_conv3d_grad_v2(
    GM_ADDR features, GM_ADDR weight, GM_ADDR grad_out_features, GM_ADDR indices_offset,
    GM_ADDR features_grad, GM_ADDR weight_grad, GM_ADDR workspace, GM_ADDR tiling
)
{
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return ;
    }

    SubmSparseConv3dGradV2<DTYPE_FEATURES> op;
    TPipe pipe;

    // must register matmul object if using matmul ops
    REGIST_MATMUL_OBJ(
        &pipe, GetSysWorkSpacePtr(),
        op.featureMatmul_, &(tiling_data.featureMatmulTilingData),
        op.weightMatmul_, &(tiling_data.weightMatmulTilingData)
    );

    op.Init(&pipe, features, weight, grad_out_features, indices_offset, features_grad, weight_grad, usrWorkspace, &tiling_data);
    op.Process();
}