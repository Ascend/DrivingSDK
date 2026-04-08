#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "deformable_conv2d_utils.h"

using namespace AscendC;
using namespace MicroAPI;

namespace {
constexpr int32_t DOUBLE_BUF = 2;
constexpr int32_t OFFSET_DIVIDE_KERNEL_SIZE = 2;
constexpr int32_t FLOAT_BYTE_SIZE = 4;
constexpr MatmulConfig DEFORMABLE_CONV2D_CFG0 = GetMDLConfig(false, false, 0, true, false, false, true);
constexpr MatmulConfig DEFORMABLE_CONV2D_CFG1 = GetMDLConfig(false, false, 2, true, false, false, true);
}

template<typename T, bool modulated>
class DeformableConv2dGradV2Kernel {
public:
    using A0Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using A1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;

    matmul::MatmulImpl<A0Type, BType, CType, CType, DEFORMABLE_CONV2D_CFG0> mm0_;
    matmul::MatmulImpl<A1Type, BType, CType, CType, DEFORMABLE_CONV2D_CFG1> mm1_;

    __aicore__ inline DeformableConv2dGradV2Kernel() = default;

    __aicore__ inline void Init(GM_ADDR inputFeatures, GM_ADDR weight, GM_ADDR bias, GM_ADDR offset, GM_ADDR mask,
        GM_ADDR gradOutputFeatures, GM_ADDR gradInputFeatures, GM_ADDR gradWeight, GM_ADDR gradBias, GM_ADDR gradOffset,
        GM_ADDR gradMask, GM_ADDR workspace, const DeformableConv2dGradV2TilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;

        if ASCEND_IS_AIV {
            blkIdx_ = GetBlockIdx() / 2;
        } else {
            blkIdx_ = GetBlockIdx();
        }
        isOneAiv_ = GetSubBlockIdx();

        InitTiling(tilingData);
        InitGM(inputFeatures, weight, bias, offset, mask, gradOutputFeatures, gradInputFeatures,
            gradWeight, gradBias, gradOffset, gradMask, workspace);
        InitBuffer();
    }

    __aicore__ inline void Process();

protected:
    TPipe* pipe_;
    GlobalTensor<T> inputFeaturesGm_, offsetGm_, weightGm_, gradOutputFeaturesGm_;
    GlobalTensor<T> maskGm_, gradMaskGm_;
    GlobalTensor<T> gradinputFeaturesGm_, gradOffsetGm_, gradWeightGm_;
    GlobalTensor<T> img2colMatGradGm_, img2colMatGm_;
    
    // double buffer
    uint8_t ping1_ = 0, ping2_ = 0;

    // tiling
    int64_t blkIdx_;
    int32_t isOneAiv_, byteSizePerElements_, batchSize_, inChannels_, outChannels_, inHeightSize_, outHeightSize_, taskStartOffset_, repeatTimes_, repeatTimesAligned_;
    int32_t inWidthSize_, outWidthSize_, kernelSize_, heightKernelSize_, widthKernelSize_, hightPadding_, widthPadding_, heightStride_, widthStride_;
    int32_t singleLoopTask_, bigCoreCount_, coreTaskCount_, cube0TileTaskCount_, cube1TileTaskCount_, groups_;
    int32_t aivNum_, kernelSizeAligned_, twoTimesKernelSizeAligned_, inChannelsAligned_, vecElementsCount_;
    int32_t featureMapSize_, doubleGMBufferSize_, doubleUBBufferSize_;
    int32_t mm0TaskOffset_, doubleBuffer_;

    LocalTensor<T> inputOffsetLocal_;
    LocalTensor<int32_t> topLeftOffsetLocal_, topRightOffsetLocal_, bottomLeftOffsetLocal_, bottomRightOffsetLocal_,
        innerKernelWidthOffsetLocal_, innerKernelHeightOffsetLocal_;
    LocalTensor<T> pointWeightLocal_;
    LocalTensor<T> fracHeightLocal_, fracWidthLocal_;
    LocalTensor<T> topLeftFeaturesLocal_, topRightFeaturesLocal_, bottomLeftFeaturesLocal_, bottomRightFeaturesLocal_, img2colFeaturesLocal_;
    LocalTensor<T> img2colGradFeaturesLocal_;
    LocalTensor<T> gradXOffsetLocal_, gradYOffsetLocal_, gradMaskLocal_;
    
    // Buffers
    TBuf<TPosition::VECCALC> ubBuf_;

private:
    __aicore__ inline void InitTiling(const DeformableConv2dGradV2TilingData* tilingData)
    {
        byteSizePerElements_ = sizeof(T);
        batchSize_ = tilingData->n;
        inChannels_ = tilingData->cIn;
        outChannels_ = tilingData->cOut;
        inHeightSize_ = tilingData->hIn;
        outHeightSize_ = tilingData->hOut;
        inWidthSize_ = tilingData->wIn;
        outWidthSize_ = tilingData->wOut;
        heightKernelSize_ = tilingData->kH;
        widthKernelSize_ = tilingData->kW;
        hightPadding_ = tilingData->padH;
        widthPadding_ = tilingData->padW;
        heightStride_ = tilingData->strideH;
        widthStride_ = tilingData->strideW;
        groups_ = tilingData->groups;
        singleLoopTask_ = tilingData->singleLoopTask;
        bigCoreCount_ = tilingData->bigCoreCount;
        coreTaskCount_ = tilingData->coreTaskCount;
        cube0TileTaskCount_ = tilingData->cube0TileTaskCount;
        cube1TileTaskCount_ = tilingData->cube1TileTaskCount;
        aivNum_ = tilingData->coreCount;
        doubleBuffer_ = tilingData->doubleBuffer;
        kernelSize_ = heightKernelSize_ * widthKernelSize_;
        kernelSizeAligned_ = AlignUp(kernelSize_, UB_BLOCK_BYTE_SIZE / byteSizePerElements_);
        twoTimesKernelSizeAligned_ = AlignUp(kernelSize_ * OFFSET_DIVIDE_KERNEL_SIZE, UB_BLOCK_BYTE_SIZE / byteSizePerElements_);
        inChannelsAligned_ = AlignUp(inChannels_, UB_BLOCK_BYTE_SIZE / byteSizePerElements_);
        vecElementsCount_ = VEC_LENGTH / byteSizePerElements_;
        featureMapSize_ = outHeightSize_ * outWidthSize_;
        doubleUBBufferSize_ = kernelSize_ * inChannelsAligned_;
        repeatTimes_ = inChannelsAligned_ / vecElementsCount_;

        if (blkIdx_ < bigCoreCount_) {
            taskStartOffset_ = (tilingData->coreTaskCount + 1) * blkIdx_;
            coreTaskCount_ = tilingData->coreTaskCount + 1;
        } else {
            taskStartOffset_ = (tilingData->coreTaskCount + 1) * bigCoreCount_ +
                                tilingData->coreTaskCount * (blkIdx_ - bigCoreCount_);
            coreTaskCount_ = tilingData->coreTaskCount;
        }

    }

    __aicore__ inline void InitGM(GM_ADDR inputFeatures, GM_ADDR weight, GM_ADDR bias, GM_ADDR offset, GM_ADDR mask, GM_ADDR gradOutputFeatures,
        GM_ADDR gradinputFeatures, GM_ADDR gradWeight, GM_ADDR gradBias, GM_ADDR gradOffset, GM_ADDR gradMask, GM_ADDR workspace)
    {
        if ASCEND_IS_AIV {
            inputFeaturesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputFeatures));
            offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(offset));
            gradinputFeaturesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradinputFeatures));
            gradOffsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradOffset));
            maskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(mask));
            gradMaskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradMask));
        }

        img2colMatGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace) +
            batchSize_ * outHeightSize_ * outWidthSize_ * kernelSize_ * inChannels_);
        img2colMatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace));

        gradOutputFeaturesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradOutputFeatures));
        gradWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradWeight));
        weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight));
    }

    __aicore__ inline void InitBuffer()
    {
        if ASCEND_IS_AIV {
            pipe_->InitBuffer(ubBuf_, 256 * 1024);

            inputOffsetLocal_ = ubBuf_.Get<T>();
            innerKernelWidthOffsetLocal_ = inputOffsetLocal_[singleLoopTask_ * twoTimesKernelSizeAligned_].template ReinterpretCast<int32_t>();
            innerKernelHeightOffsetLocal_ = innerKernelWidthOffsetLocal_[vecElementsCount_];
            topLeftOffsetLocal_ = innerKernelHeightOffsetLocal_[vecElementsCount_];
            topRightOffsetLocal_ = topLeftOffsetLocal_[singleLoopTask_ * kernelSizeAligned_];
            bottomLeftOffsetLocal_ = topRightOffsetLocal_[singleLoopTask_ * kernelSizeAligned_];
            bottomRightOffsetLocal_ = bottomLeftOffsetLocal_[singleLoopTask_ * kernelSizeAligned_];

            fracHeightLocal_ = bottomRightOffsetLocal_[singleLoopTask_ * kernelSizeAligned_].template ReinterpretCast<T>();
            fracWidthLocal_ = fracHeightLocal_[singleLoopTask_ * kernelSizeAligned_];
            pointWeightLocal_ = fracWidthLocal_[singleLoopTask_ * kernelSizeAligned_];

            topLeftFeaturesLocal_ = pointWeightLocal_[singleLoopTask_ * kernelSizeAligned_];
            topRightFeaturesLocal_ = topLeftFeaturesLocal_[doubleBuffer_ * DOUBLE_BUF * kernelSize_ * inChannelsAligned_];
            bottomLeftFeaturesLocal_ = topRightFeaturesLocal_[doubleBuffer_ * DOUBLE_BUF * kernelSize_ * inChannelsAligned_];
            bottomRightFeaturesLocal_ = bottomLeftFeaturesLocal_[doubleBuffer_ * DOUBLE_BUF * kernelSize_ * inChannelsAligned_];
            img2colFeaturesLocal_ = bottomRightFeaturesLocal_[doubleBuffer_ * DOUBLE_BUF * kernelSize_ * inChannelsAligned_];
            img2colGradFeaturesLocal_ = img2colFeaturesLocal_[doubleBuffer_ * kernelSize_ * inChannelsAligned_];

            gradXOffsetLocal_ = img2colGradFeaturesLocal_[doubleBuffer_ * kernelSize_ * inChannelsAligned_];
            gradYOffsetLocal_ = gradXOffsetLocal_[kernelSize_ * inChannelsAligned_];
            gradMaskLocal_ = gradYOffsetLocal_[kernelSize_ * inChannelsAligned_];

            InitConstLocalVf(innerKernelHeightOffsetLocal_, innerKernelWidthOffsetLocal_, heightKernelSize_, widthKernelSize_, vecElementsCount_);
        }
    }
    __aicore__ inline void CopyInAndCopyOutAllFeature(const int32_t& innerOffset);
    __aicore__ inline void ComputeGradWeightInCube(const int32_t& taskOffset, const int32_t& taskCount);
    __aicore__ inline void ComputeImg2colMatGradInCube(const int32_t& taskOffset, const int32_t& taskCount);

    __aicore__ inline void ProcessVector(const int32_t& taskOffset, const int32_t& taskCount);
};

template<typename T, bool modulated>
__aicore__ inline void DeformableConv2dGradV2Kernel<T, modulated>::CopyInAndCopyOutAllFeature(const int32_t& innerOffset)
{
    CopyInAndCopyOutFeature(topLeftOffsetLocal_, topLeftFeaturesLocal_[ping1_ * doubleUBBufferSize_],
        topLeftFeaturesLocal_[(doubleBuffer_ + ping1_) * doubleUBBufferSize_], inputFeaturesGm_, gradinputFeaturesGm_, innerOffset, 0, inChannels_);
    CopyInAndCopyOutFeature(topRightOffsetLocal_, topRightFeaturesLocal_[ping1_ * doubleUBBufferSize_],
        topRightFeaturesLocal_[(doubleBuffer_ + ping1_) * doubleUBBufferSize_], inputFeaturesGm_, gradinputFeaturesGm_, innerOffset, 0, inChannels_);
    CopyInAndCopyOutFeature(bottomLeftOffsetLocal_, bottomLeftFeaturesLocal_[ping1_ * doubleUBBufferSize_],
        bottomLeftFeaturesLocal_[(doubleBuffer_ + ping1_) * doubleUBBufferSize_], inputFeaturesGm_, gradinputFeaturesGm_, innerOffset, 0, inChannels_);
    CopyInAndCopyOutFeature(bottomRightOffsetLocal_, bottomRightFeaturesLocal_[ping1_ * doubleUBBufferSize_],
        bottomRightFeaturesLocal_[(doubleBuffer_ + ping1_) * doubleUBBufferSize_], inputFeaturesGm_, gradinputFeaturesGm_, innerOffset, 0, inChannels_);
}

template<typename T, bool modulated>
__aicore__ inline void DeformableConv2dGradV2Kernel<T, modulated>::ProcessVector(const int32_t& taskOffset, const int32_t& taskCount)
{
    for (int32_t i = 0; i < taskCount; i++) {
        ComputeOffsetAndWeightVf<T>(topLeftOffsetLocal_[i * kernelSizeAligned_], topRightOffsetLocal_[i * kernelSizeAligned_],
            bottomLeftOffsetLocal_[i * kernelSizeAligned_], bottomRightOffsetLocal_[i * kernelSizeAligned_], fracHeightLocal_[i * kernelSizeAligned_],
            fracWidthLocal_[i * kernelSizeAligned_], inputOffsetLocal_[i * twoTimesKernelSizeAligned_], innerKernelHeightOffsetLocal_,
            innerKernelWidthOffsetLocal_, taskStartOffset_ + taskOffset + i, featureMapSize_, outHeightSize_, outWidthSize_, heightKernelSize_, widthKernelSize_);
        
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_MTE2>(ping1_);

        DataCopyPad(img2colGradFeaturesLocal_[ping1_ * doubleUBBufferSize_], img2colMatGradGm_[(taskStartOffset_ + taskOffset + i) * kernelSize_ * inChannels_],
            {static_cast<uint16_t>(kernelSize_), static_cast<uint32_t>(inChannels_ * byteSizePerElements_), 0, 0, 0}, {false, 0, 0, 0});
        
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE3_V>(ping1_);
        
        ComputeInputFeatureGrad(topLeftFeaturesLocal_[(doubleBuffer_ + ping1_) * doubleUBBufferSize_], topRightFeaturesLocal_[(doubleBuffer_ + ping1_) * doubleUBBufferSize_],
            bottomLeftFeaturesLocal_[(doubleBuffer_ + ping1_) * doubleUBBufferSize_], bottomRightFeaturesLocal_[(doubleBuffer_ + ping1_) * doubleUBBufferSize_],
            img2colGradFeaturesLocal_[ping1_ * doubleUBBufferSize_], fracHeightLocal_[i * kernelSizeAligned_], fracWidthLocal_[i * kernelSizeAligned_], pointWeightLocal_[i * kernelSizeAligned_],
            inChannelsAligned_, heightKernelSize_, widthKernelSize_, repeatTimes_, vecElementsCount_);

        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_S>(0);

        CopyInAndCopyOutAllFeature(i);

        SetFlag<HardEvent::MTE3_V>(ping1_);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        ComputeOffsetGradAndMaskGrad(gradXOffsetLocal_, gradMaskLocal_, img2colFeaturesLocal_, img2colGradFeaturesLocal_[ping1_ * doubleUBBufferSize_],
            topLeftFeaturesLocal_[ping1_ * doubleUBBufferSize_], topRightFeaturesLocal_[ping1_ * doubleUBBufferSize_],
            bottomLeftFeaturesLocal_[ping1_ * doubleUBBufferSize_], bottomRightFeaturesLocal_[ping1_ * doubleUBBufferSize_],
            fracHeightLocal_[i * kernelSizeAligned_], fracWidthLocal_[i * kernelSizeAligned_], pointWeightLocal_[i * kernelSizeAligned_],
            inChannelsAligned_, heightKernelSize_, widthKernelSize_, repeatTimes_, vecElementsCount_);

        SetFlag<HardEvent::V_MTE2>(ping1_);
        SetFlag<HardEvent::V_MTE3>(2);
        WaitFlag<HardEvent::V_MTE3>(2);

        DataCopyPad(img2colMatGm_[(taskStartOffset_ + taskOffset + i) * kernelSize_ * inChannels_], img2colFeaturesLocal_,
            {static_cast<uint16_t>(kernelSize_), static_cast<uint32_t>(inChannels_  * byteSizePerElements_), 0, 0, 0});
        
        int32_t repeatForPointWeightBlockReduce = kernelSize_ * inChannelsAligned_ / vecElementsCount_;
        int32_t maskForPointWeightWholeReduceSum = (inChannelsAligned_ * byteSizePerElements_) / UB_BLOCK_BYTE_SIZE;
        int32_t repeatForPointWeightWholeReduceSum = kernelSize_;
        BlockReduceSum<T>(gradMaskLocal_, gradMaskLocal_, repeatForPointWeightBlockReduce, vecElementsCount_, 1, 1, 8);
        WholeReduceSum<T>(gradMaskLocal_, gradMaskLocal_, maskForPointWeightWholeReduceSum, repeatForPointWeightWholeReduceSum, 1, 1,
            maskForPointWeightWholeReduceSum / (UB_BLOCK_BYTE_SIZE / byteSizePerElements_));

        int32_t repeatForOffsetBlockReduce = 2 * kernelSize_ * inChannelsAligned_ / vecElementsCount_;
        int32_t maskForOffsetWholeReduceSum = (inChannelsAligned_ * byteSizePerElements_) / UB_BLOCK_BYTE_SIZE;
        int32_t repeatForOffsetWholeReduceSum = 2 * kernelSize_;
        BlockReduceSum<T>(gradXOffsetLocal_, gradXOffsetLocal_, repeatForOffsetBlockReduce, vecElementsCount_, 1, 1, 8);
        WholeReduceSum<T>(gradXOffsetLocal_, gradXOffsetLocal_, maskForOffsetWholeReduceSum, repeatForOffsetWholeReduceSum, 1, 1,
            maskForOffsetWholeReduceSum / (UB_BLOCK_BYTE_SIZE / byteSizePerElements_));

        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        DataCopyPad(gradMaskGm_[(taskStartOffset_ + taskOffset + i) * kernelSize_], gradMaskLocal_,
            {static_cast<uint16_t>(1), static_cast<uint32_t>(kernelSize_  * byteSizePerElements_), 0, 0, 0});
        DataCopyPad(gradOffsetGm_[(taskStartOffset_ + taskOffset + i) * (kernelSize_ * 2)], gradXOffsetLocal_,
            {static_cast<uint16_t>(1), static_cast<uint32_t>(2 * kernelSize_  * byteSizePerElements_), 0, 0, 0});
        
        if (doubleBuffer_ == 2) {
            ping1_ = 1 - ping1_;
        }
    }
}

template<typename T, bool modulated>
__aicore__ inline void DeformableConv2dGradV2Kernel<T, modulated>::Process()
{
    int8_t cvPing1 = 0, cvPing2 = 1;

    if ASCEND_IS_AIV {
        int32_t innerAicAivOffset = isOneAiv_ * singleLoopTask_;
        SetFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::MTE3_V>(1);
        SetFlag<HardEvent::MTE3_V>(2);
        SetFlag<HardEvent::MTE3_V>(3);

        SetFlag<HardEvent::V_MTE2>(0);
        SetFlag<HardEvent::V_MTE2>(1);
        SetFlag<HardEvent::V_MTE2>(2);

        SetFlag<HardEvent::V_MTE3>(1);
        for (int32_t taskOffset = 0; taskOffset < coreTaskCount_; taskOffset += 2 * singleLoopTask_) {
            
            // cv sync
            bool matmulTaskEqualTwoTimescubeTileFlag1 = taskOffset % (2 * cube0TileTaskCount_) == 0;
            if (matmulTaskEqualTwoTimescubeTileFlag1) {
                CrossCoreWaitFlag<0x2>(cvPing1);
                cvPing1 = (cvPing1 + 1) % 8;
            }

            if (innerAicAivOffset + taskOffset < coreTaskCount_) {
                int32_t aivTaskOffset = innerAicAivOffset + taskOffset;
                int32_t taskCount = min(singleLoopTask_, coreTaskCount_ - aivTaskOffset);
                
                WaitFlag<HardEvent::V_MTE2>(2);

                DataCopyPad(inputOffsetLocal_, offsetGm_[(taskStartOffset_ + aivTaskOffset) * (2 * kernelSize_)],
                    {static_cast<uint16_t>(taskCount), static_cast<uint32_t>(2 * kernelSize_ * byteSizePerElements_), 0, 0, 0}, {false, 0, 0, 0});
                DataCopyPad(pointWeightLocal_, maskGm_[(taskStartOffset_ + aivTaskOffset) * kernelSize_],
                    {static_cast<uint16_t>(taskCount), static_cast<uint32_t>(kernelSize_ * byteSizePerElements_), 0, 0, 0}, {false, 0, 0, 0});
                
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                
                ProcessVector(aivTaskOffset, taskCount);

                SetFlag<HardEvent::V_MTE2>(2);
            }

            // cv sync
            bool matmulTaskEqualTwoTimescubeTileFlag2 = (taskOffset + 2 * singleLoopTask_) % (2 * cube1TileTaskCount_) == 0;
            bool finalMatmulFlag = taskOffset + 2 * singleLoopTask_ >= coreTaskCount_;
            if (matmulTaskEqualTwoTimescubeTileFlag2 || finalMatmulFlag) {
                CrossCoreSetFlag<0x2, PIPE_MTE3>(cvPing2);
                cvPing2 = (cvPing2 + 1) % 8;
            }
        }
        WaitFlag<HardEvent::V_MTE3>(1);

        WaitFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(1);
        WaitFlag<HardEvent::V_MTE2>(2);

        WaitFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(1);
        WaitFlag<HardEvent::MTE3_V>(2);
        WaitFlag<HardEvent::MTE3_V>(3);
    }
    
    if ASCEND_IS_AIC {
        for (int32_t taskOffset = 0; taskOffset < coreTaskCount_; taskOffset += 2 * cube0TileTaskCount_) {
            uint32_t taskCount = min(2 * cube0TileTaskCount_, coreTaskCount_ - taskOffset);
            ComputeImg2colMatGradInCube(taskStartOffset_ + taskOffset, taskCount);
            PipeBarrier<PIPE_ALL>();
            CrossCoreSetFlag<0x2, PIPE_FIX>(cvPing1);
            cvPing1 = (cvPing1 + 1) % 8;
        }

        for (int32_t taskOffset = 0; taskOffset < coreTaskCount_; taskOffset += 2 * cube1TileTaskCount_) {
            uint32_t taskCount = min(2 * cube1TileTaskCount_, coreTaskCount_ - taskOffset);
            CrossCoreWaitFlag<0x2>(cvPing2);
            ComputeGradWeightInCube(taskStartOffset_ + taskOffset, taskCount);
            cvPing2 = (cvPing2 + 1) % 8;
        }
    }
}

template<typename T, bool modulated>
__aicore__ inline void DeformableConv2dGradV2Kernel<T, modulated>::ComputeImg2colMatGradInCube(const int32_t& taskOffset, const int32_t& taskCount)
{
    if (taskCount <= 0) {
        return;
    }

    mm0_.SetTensorA(gradOutputFeaturesGm_[taskOffset * outChannels_]);
    mm0_.SetTensorB(weightGm_);
    if (byteSizePerElements_ == FLOAT_BYTE_SIZE) {
        mm0_.SetHF32(true, 1);
    }
    mm0_.SetOrgShape(taskCount, kernelSize_ * inChannels_, outChannels_);
    mm0_.SetSingleShape(taskCount, kernelSize_ * inChannels_, outChannels_);
    mm0_.template IterateAll<false> (img2colMatGradGm_[taskOffset * kernelSize_ * inChannels_]);
}

template<typename T, bool modulated>
__aicore__ inline void DeformableConv2dGradV2Kernel<T, modulated>::ComputeGradWeightInCube(const int32_t& taskOffset, const int32_t& taskCount)
{
    if (taskCount <= 0)
        return;

    mm1_.SetTensorA(gradOutputFeaturesGm_[taskOffset * outChannels_], true);
    mm1_.SetTensorB(img2colMatGm_[taskOffset * kernelSize_ * inChannels_]);
    if (byteSizePerElements_ == FLOAT_BYTE_SIZE) {
        mm1_.SetHF32(true, 1);
    }
    mm1_.SetOrgShape(outChannels_, kernelSize_ * inChannels_, taskCount);
    mm1_.SetSingleShape(outChannels_, kernelSize_ * inChannels_, taskCount);
    mm1_.template IterateAll<false> (gradWeightGm_, 1);
}

extern "C" __global__ __aicore__ void deformable_conv2d_grad_v2(GM_ADDR inputFeatures, GM_ADDR weight, GM_ADDR bias, GM_ADDR offset,
    GM_ADDR mask, GM_ADDR gradOutputFeatures, GM_ADDR gradinputFeatures, GM_ADDR gradWeight, GM_ADDR gradBias, GM_ADDR gradOffset, GM_ADDR gradMask,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }

    TPipe pipe;

    if (TILING_KEY_IS(0)) {
        DeformableConv2dGradV2Kernel<DTYPE_INPUTFEATURES, false> op;
        
        op.mm0_.SetSubBlockIdx(0);
        op.mm0_.Init(&tilingData.mm0TilingData, &pipe);
        op.mm1_.SetSubBlockIdx(0);
        op.mm1_.Init(&tilingData.mm1TilingData, &pipe);
        
        op.Init(inputFeatures, weight, bias, offset, mask, gradOutputFeatures, gradinputFeatures, gradWeight, gradBias, gradOffset, gradMask,
            usrWorkspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        DeformableConv2dGradV2Kernel<DTYPE_INPUTFEATURES, true> op;
        
        op.mm0_.SetSubBlockIdx(0);
        op.mm0_.Init(&tilingData.mm0TilingData, &pipe);
        op.mm1_.SetSubBlockIdx(0);
        op.mm1_.Init(&tilingData.mm1TilingData, &pipe);
        
        op.Init(inputFeatures, weight, bias, offset, mask, gradOutputFeatures, gradinputFeatures, gradWeight, gradBias, gradOffset, gradMask,
            usrWorkspace, &tilingData, &pipe);
        op.Process();
    }
}