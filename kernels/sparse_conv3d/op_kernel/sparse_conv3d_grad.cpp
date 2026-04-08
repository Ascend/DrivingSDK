#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;

namespace {
constexpr int32_t BLOCK_BYTE = 32;
constexpr int32_t FLOAT32_BYTE = 4;
constexpr int32_t FLOAT_BLOCK_NUM = BLOCK_BYTE / FLOAT32_BYTE;
constexpr int32_t DATA_BLOCK_PER_REPEAT = 8;
constexpr int32_t SORT_RES_BYTE = 8;

constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t TREBLE_BUFFER = 3;
constexpr uint32_t QUAD_BUFFER = 4;

constexpr int32_t DATA_NUM_PER_CONCAT = 16;
constexpr int32_t DATA_NUM_PER_SORT = 32;
constexpr int32_t SORT_CONCAT_RATIO = DATA_NUM_PER_SORT / DATA_NUM_PER_CONCAT;

constexpr MatmulConfig SPARSE_CONV3D_CFG = GetNormalConfig(); // 替换config
} // namespace

template<typename T>
class SparseConv3dGrad {
public:
    using weightMatType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using imgToColMatType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using gradOutFeaturesMatType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using weightGradMatType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using featureGradMatType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;

    matmul::Matmul<gradOutFeaturesMatType, weightMatType, featureGradMatType, featureGradMatType, SPARSE_CONV3D_CFG>
        featureMatmul_;
    matmul::Matmul<imgToColMatType, gradOutFeaturesMatType, weightGradMatType, weightGradMatType, SPARSE_CONV3D_CFG>
        weightMatmul_;

    __aicore__ inline SparseConv3dGrad() {};

    __aicore__ inline void Init(TPipe* pipe, GM_ADDR features, GM_ADDR weight, GM_ADDR grad_out_features,
        GM_ADDR former_sorted_indices, GM_ADDR indices_offset, GM_ADDR features_grad, GM_ADDR weight_grad,
        GM_ADDR usrWorkspace, SparseConv3dGradTillingData* tilingData)
    {
        pipe_ = pipe;
        blockIdx_ = GetBlockIdx();
        InitTiling(tilingData);
        InitBuffer(features, weight, grad_out_features, former_sorted_indices, indices_offset, features_grad,
            weight_grad, usrWorkspace);

        eventMTE3ToMTE2_ = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
        copyOutEvent_ = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= usedVectorNum_) {
            return;
        }
        bool doingMatmul = false;
        for (int32_t k = 0; k < kernelSize_; k++) {
            prepareMatmulFeatures(k, doingMatmul);
            if (totalSparseM == 0) {
                continue;
            } else {
                doingMatmul = true;
            }
            calGradFeaturesMatmul(k);
            calGradWeightMatmul(k);

            featureMatmul_.WaitIterateAll();
            gradFeaturesScatterAdd(); // wait featurematmul before scatter add
        }
        if (doingMatmul) {
            weightMatmul_.WaitIterateAll();
        }
        weightMatmul_.End();
        featureMatmul_.End();
    }

private:
    __aicore__ inline void InitTiling(SparseConv3dGradTillingData* tilingData)
    {
        featureByteSize_ = sizeof(T);
        indicesByteSize_ = sizeof(DTYPE_INDICES_OFFSET);
        blockDataNum_ = BLOCK_BYTE / featureByteSize_;
        blockIndicesNum_ = BLOCK_BYTE / indicesByteSize_;
        usedVectorNum_ = tilingData->usedVectorNum;
        kernelSize_ = tilingData->kernelSize;
        inChannels_ = tilingData->inChannels;
        outChannels_ = tilingData->outChannels;
        totalTaskNum_ = tilingData->totalTaskNum;
        mainCoreTask_ = tilingData->mainCoreTask;
        lastCoreTask_ = tilingData->lastCoreTask;
        sparseRatio_ = tilingData->sparseRatio;
        ubMaxTaskNum_ = tilingData->ubMaxTaskNum;
        featuresGradSize_ = tilingData->featuresGradSize;
        weightGradSize_ = tilingData->weightGradSize;

        tmpSortSize_ = tilingData->tmpSortSize;
        kernelSizeAlign32_ = tilingData->kernelSizeAlign32;

        featuresWorkSpaceOffset_ = tilingData->featuresWorkSpaceOffset;
        tmpGradFeaturesWorkSpaceOffset_ = tilingData->tmpGradFeaturesWorkSpaceOffset;
        startIndicesWorkSpaceOffset_ = tilingData->startIndicesWorkSpaceOffset;
        endIndicesWorkSpaceOffset_ = tilingData->endIndicesWorkSpaceOffset;
        inputIndicesPtrWorkSpaceOffset_ = tilingData->inputIndicesPtrWorkSpaceOffset;
        inputIndicesWorkSpaceOffset_ = tilingData->inputIndicesWorkSpaceOffset;
        kernelIndicesWorkSpaceOffset_ = tilingData->kernelIndicesWorkSpaceOffset;

        sparseIndicesTaskNum_ = sparseRatio_ * ubMaxTaskNum_;
        globalTaskOffset_ = mainCoreTask_ * blockIdx_;
        coreTaskCount_ = (blockIdx_ == usedVectorNum_ - 1) ? lastCoreTask_ : mainCoreTask_;
    }

    __aicore__ inline void InitBuffer(GM_ADDR features, GM_ADDR weight, GM_ADDR grad_out_features,
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

        uint32_t loopEndIdxSize = AlignUp(DivCeil(coreTaskCount_, sparseIndicesTaskNum_), FLOAT_BLOCK_NUM);
        uint32_t taskNumAligned = AlignUp(ubMaxTaskNum_, blockIndicesNum_);
        uint32_t taskKernelNumAligned = AlignUp(ubMaxTaskNum_ * kernelSize_, blockIndicesNum_);
        uint32_t kernelSizeAligned = AlignUp(kernelSize_, blockIndicesNum_);

        pipe_->InitBuffer(featuresBuf_, (DOUBLE_BUFFER * inChannels_ + outChannels_) * ubMaxTaskNum_ * featureByteSize_);
        pipe_->InitBuffer(indicesOffsetBuf_, (QUAD_BUFFER * taskNumAligned) * indicesByteSize_);
        pipe_->InitBuffer(inputIndicesBuf_, (DOUBLE_BUFFER * taskKernelNumAligned + kernelSizeAligned) * indicesByteSize_);
        pipe_->InitBuffer(inputIndicesFloatBuf_, (kernelSizeAlign32_ + FLOAT_BLOCK_NUM) * FLOAT32_BYTE);
        pipe_->InitBuffer(forSortFloatBuf_, TREBLE_BUFFER * kernelSizeAlign32_ * FLOAT32_BYTE + tmpSortSize_);
        pipe_->InitBuffer(loopEndIdxBuf_, loopEndIdxSize);
        loopEndIdxLocal_ = loopEndIdxBuf_.Get<uint32_t>();

        tmpFeaturesLocal_ = featuresBuf_.Get<T>();                              // input features
        tmpGradFeaturesLocal_ = tmpFeaturesLocal_[ubMaxTaskNum_ * inChannels_]; // grad_features临时暂存
        tmpGradOutLocal_ = tmpGradFeaturesLocal_[ubMaxTaskNum_ * inChannels_];  // grad_out_features

        startIndicesLocal_ = indicesOffsetBuf_.Get<int32_t>(); // 记录起始和结束位置
        endIndicesLocal_ = startIndicesLocal_[taskNumAligned];
        tmpInputIndicesLocal_ = endIndicesLocal_[taskNumAligned]; // 记录对应有效输入的indices
        inputIndicesPtrLocal_ = tmpInputIndicesLocal_[taskNumAligned];

        formerInputIndicesLocal_ = inputIndicesBuf_.Get<int32_t>();                 // input points位置
        formerKernelIndicesLocal_ = formerInputIndicesLocal_[taskKernelNumAligned]; // kernel位置
        sortedIndicesLocal_ = formerKernelIndicesLocal_[taskKernelNumAligned];      // kernelSize for single point

        tmpIndicesFloatLocal_ = inputIndicesFloatBuf_.Get<float>();
        kernelSizeFloatLocal_ = tmpIndicesFloatLocal_[kernelSizeAlign32_];

        concatFloatLocal_ = forSortFloatBuf_.Get<float>();
        sortedResFloatLocal_ = concatFloatLocal_[kernelSizeAlign32_]; // need 2 * kernelSizeAlign32_
        sortTmpLocal_ = sortedResFloatLocal_[2 * kernelSizeAlign32_];

        // todo: set workspace
        uint64_t globalInChannels = globalTaskOffset_ * inChannels_;
        uint64_t globalOutChannels = globalTaskOffset_ * outChannels_;
        uint64_t coreTaskInChannels = coreTaskCount_ * inChannels_;
        uint64_t coreTaskOutChannels = coreTaskCount_ * outChannels_;

        uint64_t globalKernelSize = globalTaskOffset_ * kernelSize_;
        uint64_t coreTaskKernelSize = coreTaskCount_ * kernelSize_;

        gradOutWorkSpace.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(usrWorkspace) + globalOutChannels, coreTaskOutChannels);
        featuresWorkSpace.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(usrWorkspace) + featuresWorkSpaceOffset_ + globalInChannels,
            coreTaskInChannels);
        tmpGradFeaturesWorkSpace.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(usrWorkspace) + tmpGradFeaturesWorkSpaceOffset_ + globalInChannels,
            coreTaskInChannels);

        startIndicesWorkSpace.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(usrWorkspace) + startIndicesWorkSpaceOffset_ + globalTaskOffset_,
            coreTaskCount_);
        endIndicesWorkSpace.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(usrWorkspace) + endIndicesWorkSpaceOffset_ + globalTaskOffset_,
            coreTaskCount_);
        inputIndicesPtrWorkSpace.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(usrWorkspace) + inputIndicesPtrWorkSpaceOffset_ + globalTaskOffset_,
            coreTaskCount_);

        formerInputIndicesWorkSpace.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(usrWorkspace) + inputIndicesWorkSpaceOffset_ + globalKernelSize,
            coreTaskKernelSize);
        formerKernelIndicesWorkSpace.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(usrWorkspace) + kernelIndicesWorkSpaceOffset_ + globalKernelSize,
            coreTaskKernelSize);
    }

    __aicore__ inline void prepareMatmulFeatures(int32_t kIdx, const bool hasMat)
    {
        sparseM = 0;
        totalSparseM = 0;
        uint32_t inputIndicesNum = 0;
        SetFlag<HardEvent::MTE3_MTE2>(eventMTE3ToMTE2_);
        for (int32_t taskIdx = 0; taskIdx < coreTaskCount_; taskIdx += sparseIndicesTaskNum_) {
            uint32_t curTaskCount = min(sparseIndicesTaskNum_, coreTaskCount_ - taskIdx);
            uint32_t curTaskByteSize = curTaskCount * indicesByteSize_;
            int32_t loopStartSortedIdx = 0;
            bool gotSparsePoints = false;

            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3ToMTE2_);
            if (kIdx == 0) {
                // record startIdx and endIdx
                DataCopyPad(startIndicesLocal_, indicesOffsetGM_[globalTaskOffset_ + taskIdx],
                    {1, curTaskByteSize, 0, 0, 0}, {true, 0, 0, 0});
                DataCopyPad(endIndicesLocal_, indicesOffsetGM_[globalTaskOffset_ + taskIdx + 1],
                    {1, curTaskByteSize, 0, 0, 0}, {true, 0, 0, 0});

                loopStartSortedIdx = startIndicesLocal_.GetValue(0);
                Adds(startIndicesLocal_, startIndicesLocal_, (-1) * loopStartSortedIdx,
                    curTaskCount); // 变成了相对第一个有效点的Idx
                SetFlag<HardEvent::V_MTE3>(0);
                Adds(endIndicesLocal_, endIndicesLocal_, (-1) * loopStartSortedIdx, curTaskCount);

                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopyPad(startIndicesWorkSpace[taskIdx], startIndicesLocal_,
                    {1, curTaskByteSize, 0, 0, 0}); // 降序排序，起始点不变
            } else {
                // todo:只有一轮循环的时候，此搬运可以避免，小shape可优化
                DataCopyPad(
                    startIndicesLocal_, startIndicesWorkSpace[taskIdx], {1, curTaskByteSize, 0, 0, 0}, {true, 0, 0, 0});
                DataCopyPad(
                    endIndicesLocal_, endIndicesWorkSpace[taskIdx], {1, curTaskByteSize, 0, 0, 0}, {true, 0, 0, 0});

                uint32_t endIdx = loopEndIdxLocal_.GetValue(taskIdx / sparseIndicesTaskNum_);
                uint32_t sparseNum = endIdx - inputIndicesNum;
                uint32_t sparseNumByteSize = sparseNum * indicesByteSize_;

                DataCopyPad(formerInputIndicesLocal_, formerInputIndicesWorkSpace[inputIndicesNum],
                    {1, sparseNumByteSize, 0, 0, 0}, {true, 0, 0, 0});
                DataCopyPad(formerKernelIndicesLocal_, formerKernelIndicesWorkSpace[inputIndicesNum],
                    {1, sparseNumByteSize, 0, 0, 0}, {true, 0, 0, 0});
                inputIndicesNum = endIdx;
            }

            int32_t kernelIdx = -1;
            int32_t inputIdx = -1;
            for (int32_t idx = 0; idx < curTaskCount; idx++) {
                int32_t end_ = endIndicesLocal_.GetValue(idx);
                int32_t start_ = startIndicesLocal_.GetValue(idx);
                if (start_ == end_) { // already search all the sparse input points for this out points
                    continue;
                }

                if (kIdx == 0) {
                    // k==0时，每次只针对单个点的inputIdx
                    SetFlag<HardEvent::S_V>(0);
                    SetFlag<HardEvent::V_MTE2>(0);
                    SetFlag<HardEvent::MTE3_V>(0);
                    getSortedInputIndices(start_, end_, inputIndicesNum, loopStartSortedIdx, inputIdx, kernelIdx);
                    WaitFlag<HardEvent::S_V>(0);
                    WaitFlag<HardEvent::V_MTE2>(0);
                    WaitFlag<HardEvent::MTE3_V>(0);
                    if (idx == curTaskCount - 1) {
                        loopEndIdxLocal_.SetValue(taskIdx / sparseIndicesTaskNum_, inputIndicesNum);
                    }
                } else {
                    kernelIdx = formerKernelIndicesLocal_.GetValue(end_ - 1); // inputIdx和kernelIdx已降序排序
                    inputIdx = formerInputIndicesLocal_.GetValue(end_ - 1);
                }

                if (kernelIdx == kIdx) {
                    gotSparsePoints = true;
                    endIndicesLocal_.SetValue(idx, end_ - 1);

                    DataCopyPad(tmpGradOutLocal_[sparseM * outChannels_],
                        gradOutFeaturesGM_[(globalTaskOffset_ + taskIdx + idx) * outChannels_],
                        {1, outChannels_ * featureByteSize_, 0, 0, 0}, {true, 0, 0, 0});
                    DataCopyPad(tmpFeaturesLocal_[sparseM * inChannels_], featuresGM_[inputIdx * inChannels_],
                        {1, inChannels_ * featureByteSize_, 0, 0, 0}, {true, 0, 0, 0});
                    tmpInputIndicesLocal_.SetValue(sparseM, inputIdx);
                    sparseM++;

                    if (sparseM == ubMaxTaskNum_) {
                        copyOutFeaturesAndInputIndices(ubMaxTaskNum_, totalSparseM, hasMat);
                        totalSparseM += ubMaxTaskNum_;
                        sparseM = 0;
                    }
                }
            }
            if (gotSparsePoints || kIdx == 0) {
                // save the record of startIndices to workspace
                DataCopyPad(endIndicesWorkSpace[taskIdx], endIndicesLocal_, {1, curTaskByteSize, 0, 0, 0});
            }
            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3ToMTE2_);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3ToMTE2_);
        if (sparseM > 0) {
            copyOutFeaturesAndInputIndices(sparseM, totalSparseM, hasMat);
            totalSparseM += sparseM;
        }
    }

    __aicore__ inline void getSortedInputIndices(int32_t startIdx, int32_t endIdx, uint32_t& formerInputOffset,
        int32_t indicesStartOffset, int32_t& inpIdx, int32_t& knIdx)
    {
        uint32_t singleSparseNum = endIdx - startIdx;
        uint32_t singleSparseNumByteSize = singleSparseNum * indicesByteSize_;
        WaitFlag<HardEvent::V_MTE2>(0);
        DataCopyPad(sortedIndicesLocal_, sortedIndicesGM_[indicesStartOffset + startIdx],
            {1, singleSparseNumByteSize, 0, 0, 0}, {true, 0, 0, 0}); // 为了做sort只能单点操作
        SetFlag<HardEvent::MTE2_V>(0);

        // inputIdx
        WaitFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        Cast<float, int32_t>(tmpIndicesFloatLocal_, sortedIndicesLocal_, RoundMode::CAST_NONE, singleSparseNum);
        uint64_t mask = 64; // blockIndicesNum_ * DATA_BLOCK_PER_REPEAT
        uint8_t repeatTime = DivCeil(singleSparseNum, blockIndicesNum_ * DATA_BLOCK_PER_REPEAT);
        Duplicate<float>(kernelSizeFloatLocal_, (1.0f) * kernelSize_, FLOAT_BLOCK_NUM); // 一个block的27
        Div(tmpIndicesFloatLocal_, tmpIndicesFloatLocal_, kernelSizeFloatLocal_, mask, repeatTime, {1, 1, 0, 8, 8, 0});
        Cast<int32_t, float>(
            formerInputIndicesLocal_, tmpIndicesFloatLocal_, RoundMode::CAST_FLOOR, singleSparseNum); // 向下取整

        Muls(formerKernelIndicesLocal_, formerInputIndicesLocal_, static_cast<int32_t>(kernelSize_), singleSparseNum);
        Sub(formerKernelIndicesLocal_, sortedIndicesLocal_, formerKernelIndicesLocal_, singleSparseNum);

        if (singleSparseNum > 1) {
            // 对kernelIdx进行降序排序
            int32_t repeat = DivCeil(singleSparseNum, DATA_NUM_PER_SORT);
            Duplicate(tmpIndicesFloatLocal_, -1.0f, kernelSizeAlign32_);
            Cast<float, int32_t>(
                tmpIndicesFloatLocal_, formerKernelIndicesLocal_, RoundMode::CAST_NONE, singleSparseNum);
            Concat(concatFloatLocal_, tmpIndicesFloatLocal_, sortTmpLocal_,
                SORT_CONCAT_RATIO * repeat); // concat一次处理16个数
            Sort<float, true>(sortedResFloatLocal_, concatFloatLocal_,
                formerInputIndicesLocal_.ReinterpretCast<uint32_t>(), sortTmpLocal_,
                repeat); // formerInputIndicesLocal_需要uint32_t
            Extract<float>(tmpIndicesFloatLocal_, formerInputIndicesLocal_.ReinterpretCast<uint32_t>(),
                sortedResFloatLocal_, repeat);
            Cast<int32_t, float>(
                formerKernelIndicesLocal_, tmpIndicesFloatLocal_, RoundMode::CAST_FLOOR, singleSparseNum); // 向下取整
        }
        SetFlag<HardEvent::V_MTE2>(0);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        knIdx = formerKernelIndicesLocal_.GetValue(singleSparseNum - 1); // inputIdx和kernelIdx已降序排序
        inpIdx = formerInputIndicesLocal_.GetValue(singleSparseNum - 1);
        SetFlag<HardEvent::S_V>(0);

        // 搬出inputIdx和kernelIdx
        SetFlag<HardEvent::V_MTE3>(1);
        WaitFlag<HardEvent::V_MTE3>(1);
        DataCopyPad(formerInputIndicesWorkSpace[formerInputOffset], formerInputIndicesLocal_,
            {1, singleSparseNumByteSize, 0, 0, 0});
        DataCopyPad(formerKernelIndicesWorkSpace[formerInputOffset], formerKernelIndicesLocal_,
            {1, singleSparseNumByteSize, 0, 0, 0});
        formerInputOffset += singleSparseNum;
        SetFlag<HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void copyOutFeaturesAndInputIndices(uint32_t m, uint32_t baseM, const bool doMat)
    {
        // move features
        if (doMat && baseM == 0) {
            weightMatmul_.WaitIterateAll();
        }
        SetFlag<HardEvent::S_MTE3>(1);
        WaitFlag<HardEvent::S_MTE3>(1);
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);
        // save input indices
        DataCopyPad(inputIndicesPtrWorkSpace[baseM], tmpInputIndicesLocal_, {1, m * indicesByteSize_, 0, 0, 0});
        DataCopyPad(gradOutWorkSpace[baseM * outChannels_], tmpGradOutLocal_,
            {1, m * outChannels_ * featureByteSize_, 0, 0, 0});
        DataCopyPad(featuresWorkSpace[baseM * inChannels_], tmpFeaturesLocal_,
            {1, m * inChannels_ * featureByteSize_, 0, 0, 0});
        SetFlag<HardEvent::MTE3_S>(1);
        WaitFlag<HardEvent::MTE3_S>(1);

        SetFlag<HardEvent::MTE3_MTE2>(1);
        WaitFlag<HardEvent::MTE3_MTE2>(1);
    }

    __aicore__ inline void calGradFeaturesMatmul(int32_t k)
    {
        // gradOutWorkSpace[T, outChannels_] @ weightGM_[inChannels_, outChannels_].T = tmpGradFeaturesWorkSpace[T, inChannels_]
        featureMatmul_.SetTensorA(gradOutWorkSpace);
        featureMatmul_.SetTensorB(weightGM_[k * inChannels_ * outChannels_], true); // 测试是否随路转置性能更优
        featureMatmul_.SetSingleShape(totalSparseM, inChannels_, outChannels_);

        featureMatmul_.template IterateAll<false>(tmpGradFeaturesWorkSpace, 0, false, true);
    }

    __aicore__ inline void calGradWeightMatmul(int32_t k)
    {
        // @ featuresWorkSpace[T, inChannels_].T @ gradOutWorkSpace[T, outChannels_] = weightGradGM_[inChannels_, outChannel_]
        weightMatmul_.SetTensorA(featuresWorkSpace, true);
        weightMatmul_.SetTensorB(gradOutWorkSpace);
        weightMatmul_.SetSingleShape(inChannels_, outChannels_, totalSparseM);

        weightMatmul_.template IterateAll<false>(weightGradGM_[k * inChannels_ * outChannels_], 1, false, true);
    }

    __aicore__ inline void gradFeaturesScatterAdd()
    {
        SetFlag<HardEvent::MTE3_MTE2>(copyOutEvent_);
        for (int32_t idxM = 0; idxM < totalSparseM; idxM += ubMaxTaskNum_) {
            uint32_t loopTaskCount = min(ubMaxTaskNum_, totalSparseM - idxM);
            WaitFlag<HardEvent::MTE3_MTE2>(copyOutEvent_);
            DataCopyPad(inputIndicesPtrLocal_, inputIndicesPtrWorkSpace[idxM],
                {1, loopTaskCount * indicesByteSize_, 0, 0, 0}, {true, 0, 0, 0});
            DataCopyPad(tmpGradFeaturesLocal_, tmpGradFeaturesWorkSpace[idxM * inChannels_],
                {1, loopTaskCount * inChannels_ * featureByteSize_, 0, 0, 0}, {true, 0, 0, 0});

            SetFlag<HardEvent::MTE2_S>(1);
            WaitFlag<HardEvent::MTE2_S>(1);
            for (int32_t m = 0; m < loopTaskCount; m++) {
                int32_t indiceVal = inputIndicesPtrLocal_.GetValue(m);
                SetAtomicAdd<T>();
                DataCopyPad(featuresGradGM_[indiceVal * inChannels_], tmpGradFeaturesLocal_[m * inChannels_],
                    {1, inChannels_ * featureByteSize_, 0, 0, 0});
                SetAtomicNone();
            }
            SetFlag<HardEvent::MTE3_MTE2>(copyOutEvent_);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(copyOutEvent_);
    }

protected:
    TPipe* pipe_;
    GlobalTensor<T> featuresGM_, weightGM_, gradOutFeaturesGM_, featuresGradGM_, weightGradGM_;
    GlobalTensor<int32_t> sortedIndicesGM_, indicesOffsetGM_;

    GlobalTensor<T> gradOutWorkSpace, featuresWorkSpace, tmpGradFeaturesWorkSpace;
    GlobalTensor<int32_t> startIndicesWorkSpace, endIndicesWorkSpace, inputIndicesPtrWorkSpace;
    GlobalTensor<int32_t> formerInputIndicesWorkSpace, formerKernelIndicesWorkSpace;

    TBuf<TPosition::VECCALC> indicesOffsetBuf_, inputIndicesBuf_, featuresBuf_, inputIndicesFloatBuf_, forSortFloatBuf_,
        loopEndIdxBuf_;

    LocalTensor<uint32_t> loopEndIdxLocal_;
    LocalTensor<int32_t> startIndicesLocal_, endIndicesLocal_, tmpInputIndicesLocal_, inputIndicesPtrLocal_;
    LocalTensor<int32_t> sortedIndicesLocal_, formerInputIndicesLocal_, formerKernelIndicesLocal_;
    LocalTensor<float> tmpIndicesFloatLocal_, kernelSizeFloatLocal_;
    LocalTensor<float> concatFloatLocal_, sortedResFloatLocal_, sortTmpLocal_; // sort
    LocalTensor<T> tmpFeaturesLocal_, tmpGradFeaturesLocal_, tmpGradOutLocal_;

    int32_t eventMTE3ToMTE2_, copyOutEvent_;
    uint32_t kernelSize_, tmpSortSize_, kernelSizeAlign32_;
    uint32_t blockIdx_, featureByteSize_, indicesByteSize_, blockDataNum_, blockIndicesNum_;
    uint32_t sparseM {0};
    uint32_t totalSparseM {0};
    uint32_t usedVectorNum_, inChannels_, outChannels_, totalTaskNum_, mainCoreTask_, lastCoreTask_, sparseRatio_,
        ubMaxTaskNum_, sparseIndicesTaskNum_, globalTaskOffset_, coreTaskCount_;

    uint64_t featuresGradSize_, weightGradSize_;
    uint64_t featuresWorkSpaceOffset_, tmpGradFeaturesWorkSpaceOffset_, startIndicesWorkSpaceOffset_,
        endIndicesWorkSpaceOffset_, inputIndicesPtrWorkSpaceOffset_, inputIndicesWorkSpaceOffset_,
        kernelIndicesWorkSpaceOffset_;
};

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
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.featureMatmul_, &(tiling_data.featureMatmulTilingData),
        op.weightMatmul_, &(tiling_data.weightMatmulTilingData));
    op.Init(&pipe, features, weight, grad_out_features, former_sorted_indices, indices_offset, features_grad,
        weight_grad, usrWorkspace, &tiling_data);
    op.Process();
}