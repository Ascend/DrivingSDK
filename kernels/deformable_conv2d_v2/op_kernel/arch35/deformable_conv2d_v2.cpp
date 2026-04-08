#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "deformable_conv2d_utils.h"

using namespace AscendC;
using namespace MicroAPI;

namespace {
constexpr int32_t DOUBLE_BUF = 2;
constexpr int32_t FLOAT_BYTE_SIZE = 4;
constexpr int32_t VECTOR_CUBE_RATIO = 2;
constexpr int32_t OFFSET_DIVIDE_KERNEL_SIZE = 2;
constexpr MatmulConfig DEFORMABLE_CONV2D_CFG = GetMDLConfig(false, false, 0, true, false, false, true);
} // namespace


template<typename T, bool modulated>
class DeformableConv2dV2Kernel {
public:
    using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    matmul::MatmulImpl<AType, BType, CType, CType, DEFORMABLE_CONV2D_CFG> mm_;

    __aicore__ inline DeformableConv2dV2Kernel() = default;

    __aicore__ inline void Init(GM_ADDR inputFeatures, GM_ADDR offset, GM_ADDR mask, GM_ADDR weight, GM_ADDR bias, GM_ADDR outputFeatures, GM_ADDR offsetOutput,
        GM_ADDR workspace, const DeformableConv2dV2TilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;

        if ASCEND_IS_AIV {
            blkIdx_ = GetBlockIdx() / 2;
        } else {
            blkIdx_ = GetBlockIdx();
        }
        isOneAiv_ = GetSubBlockIdx();

        InitTiling(tilingData);
        InitGM(inputFeatures, offset, mask, weight, bias, outputFeatures, offsetOutput, workspace);
        InitBuffer();
    }

    __aicore__ inline void Process();
    __aicore__ inline void ProcessVector(const int32_t& taskOffset, const int32_t& taskCount);
    __aicore__ inline void PrefechOffset(const int32_t& taskOffset, const int32_t& innerOffset);
protected:
    int64_t blkIdx_;
    TPipe* pipe_;

    int8_t ping1_ = 0;
    int8_t ping2_ = 0;

    // tiling
    int32_t isOneAiv_, aicNum_, aivNum_, batchSize_, inChannels_, inChannelsAligned_, outChannels_, inHeightSize_, outHeightSize_, inWidthSize_, outWidthSize_, totalTasks_;
    int32_t kernelSize_, kernelSizeAligned_, twoTimesKernelSizeAligned_, heightKernelSize_, widthKernelSize_, hightPadding_, widthPadding_, heightStride_, widthStride_, groups_;
    int32_t cubeTileTaskCount_, singleLoopTask_, bigCoreCount_, byteSizePerElements_, coreTaskCount_, vecElementsCount_, taskStartOffset_, featureMapSize_, doubleUBBufferSize_;
    int8_t cubePing_ = 0, domMatmulCounter_ = 0, doubleGMBufferSize_;

    GlobalTensor<T> inputFeaturesGm_, offsetGm_, weightGm_, maskGm_, biasGm_;
    GlobalTensor<T> outputFeaturesGm_, img2colMatGm_;

    LocalTensor<T> inputOffsetLocal_, pointWeightLocal_, topLeftFeaturesLocal_, topRightFeaturesLocal_, bottomLeftFeaturesLocal_, bottomRightFeaturesLocal_,
        topLeftWeightLocal_, topRightWeightLocal_, bottomLeftWeightLocal_, bottomRightWeightLocal_, fracHeightLocal_, fracWidthLocal_, outputFeaturesLocal_;
    
    LocalTensor<int32_t> topLeftOffsetLocal_, topRightOffsetLocal_, bottomLeftOffsetLocal_, bottomRightOffsetLocal_,
        innerKernelWidthOffsetLocal_, innerKernelHeightOffsetLocal_;

    TBuf<TPosition::VECCALC> ubBuf_;

private:
    __aicore__ inline void InitTiling(const DeformableConv2dV2TilingData* tilingData)
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
        cubeTileTaskCount_ = tilingData->cubeTileTaskCount;
        aivNum_ = tilingData->coreCount;
        aicNum_ = tilingData->coreCount / VECTOR_CUBE_RATIO;
        kernelSize_ = heightKernelSize_ * widthKernelSize_;
        kernelSizeAligned_ = AlignUp(kernelSize_, UB_BLOCK_BYTE_SIZE / byteSizePerElements_);
        twoTimesKernelSizeAligned_ = AlignUp(kernelSize_ * OFFSET_DIVIDE_KERNEL_SIZE, UB_BLOCK_BYTE_SIZE / byteSizePerElements_);
        inChannelsAligned_ = AlignUp(inChannels_, UB_BLOCK_BYTE_SIZE / byteSizePerElements_);
        vecElementsCount_ = VEC_LENGTH / byteSizePerElements_;
        featureMapSize_ = outHeightSize_ * outWidthSize_;
        doubleUBBufferSize_ = kernelSize_ * inChannelsAligned_;

        if (blkIdx_ < bigCoreCount_) {
            taskStartOffset_ = (tilingData->coreTaskCount + 1) * blkIdx_;
            coreTaskCount_ = tilingData->coreTaskCount + 1;
        } else {
            taskStartOffset_ = (tilingData->coreTaskCount + 1) * bigCoreCount_ +
                                tilingData->coreTaskCount * (blkIdx_ - bigCoreCount_);
            coreTaskCount_ = tilingData->coreTaskCount;
        }
    }

    __aicore__ inline void InitGM(GM_ADDR inputFeatures, GM_ADDR offset, GM_ADDR mask, GM_ADDR weight,
        GM_ADDR bias, GM_ADDR outputFeatures, GM_ADDR offsetOutput, GM_ADDR workspace)
    {
        if ASCEND_IS_AIV {
            inputFeaturesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputFeatures));
            biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));
            offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(offset));
            maskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(mask));
        }
        
        weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight));
        outputFeaturesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputFeatures));
        img2colMatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(offsetOutput));
    }

    __aicore__ inline void InitBuffer()
    {
        if ASCEND_IS_AIV {
            pipe_->InitBuffer(ubBuf_, 256 * 1024 * 0.9);

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
            topRightFeaturesLocal_ = topLeftFeaturesLocal_[DOUBLE_BUF * kernelSize_ * inChannelsAligned_];
            bottomLeftFeaturesLocal_ = topRightFeaturesLocal_[DOUBLE_BUF * kernelSize_ * inChannelsAligned_];
            bottomRightFeaturesLocal_ = bottomLeftFeaturesLocal_[DOUBLE_BUF * kernelSize_ * inChannelsAligned_];
            outputFeaturesLocal_ = bottomRightFeaturesLocal_[DOUBLE_BUF * kernelSize_ * inChannelsAligned_];
            
            InitConstLocalVf(innerKernelHeightOffsetLocal_, innerKernelWidthOffsetLocal_, heightKernelSize_, widthKernelSize_, vecElementsCount_);
        }
    }

    __aicore__ inline void ProcessCube(const int32_t &taskCount, const int32_t &taskOffset)
    {
        if (taskCount <= 0)
            return;
        
        mm_.SetTensorA(img2colMatGm_[taskOffset * kernelSize_ * inChannels_]);
        mm_.SetTensorB(weightGm_, true);
        if (byteSizePerElements_ == FLOAT_BYTE_SIZE) {
            mm_.SetHF32(true, 1);
        }
        mm_.SetOrgShape(taskCount, outChannels_, kernelSize_ * inChannels_);
        mm_.SetSingleShape(taskCount, outChannels_, kernelSize_ * inChannels_);
        mm_.template IterateAll<false> (outputFeaturesGm_[taskOffset * outChannels_]);
    }

    __aicore__ inline void CopyFeaturesFromGMToUB(const int32_t& innerOffset, const int32_t& ubOffset)
    {
        CopyInFeature<T>(topLeftOffsetLocal_, topLeftFeaturesLocal_, inputFeaturesGm_, innerOffset, ubOffset, inChannels_);
        CopyInFeature<T>(topRightOffsetLocal_, topRightFeaturesLocal_, inputFeaturesGm_, innerOffset, ubOffset, inChannels_);
        CopyInFeature<T>(bottomLeftOffsetLocal_, bottomLeftFeaturesLocal_, inputFeaturesGm_, innerOffset, ubOffset, inChannels_);
        CopyInFeature<T>(bottomRightOffsetLocal_, bottomRightFeaturesLocal_, inputFeaturesGm_, innerOffset, ubOffset, inChannels_);
    }
};

template<typename T, bool modulated>
__aicore__ inline void DeformableConv2dV2Kernel<T, modulated>::PrefechOffset(const int32_t& taskOffset, const int32_t& innerOffset)
{
    int32_t ubOffset = innerOffset * kernelSizeAligned_;
    ComputeOffsetAndWeightVf<T>(topLeftOffsetLocal_[ubOffset], topRightOffsetLocal_[ubOffset],
        bottomLeftOffsetLocal_[ubOffset], bottomRightOffsetLocal_[ubOffset], fracHeightLocal_[ubOffset],
        fracWidthLocal_[ubOffset], inputOffsetLocal_[innerOffset * twoTimesKernelSizeAligned_], innerKernelHeightOffsetLocal_,
        innerKernelWidthOffsetLocal_, taskStartOffset_ + taskOffset + innerOffset, featureMapSize_, outHeightSize_, outWidthSize_, heightKernelSize_, widthKernelSize_);
}

template<typename T, bool modulated>
__aicore__ inline void DeformableConv2dV2Kernel<T, modulated>::ProcessVector(const int32_t& taskOffset, const int32_t& taskCount)
{
    uint16_t repeatTimes = inChannelsAligned_ / vecElementsCount_;
    PrefechOffset(taskOffset, 0);

    SetFlag<HardEvent::V_S>(0);
    for (int32_t i = 0; i < taskCount; i++) {
        int32_t ubOffset1 = i * kernelSizeAligned_;
        int32_t ubOffset2 = doubleUBBufferSize_ * ping1_;

        WaitFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_MTE2>(ping1_);

        CopyFeaturesFromGMToUB(i, ubOffset2);
        if ((i + 1) < taskCount) {
            PrefechOffset(taskOffset, i + 1);
        }
        SetFlag<HardEvent::V_S>(0);

        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE3_V>(ping1_);

        interpolationVF<T>(outputFeaturesLocal_[ubOffset2], topLeftFeaturesLocal_[ubOffset2], topRightFeaturesLocal_[ubOffset2],
            bottomLeftFeaturesLocal_[ubOffset2], bottomRightFeaturesLocal_[ubOffset2], fracHeightLocal_[ubOffset1], fracWidthLocal_[ubOffset1],
            pointWeightLocal_[ubOffset1], inChannelsAligned_, heightKernelSize_, widthKernelSize_, repeatTimes, vecElementsCount_);

        SetFlag<HardEvent::V_MTE2>(ping1_);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        DataCopyPad(img2colMatGm_[(taskOffset + i + taskStartOffset_) * kernelSize_ * inChannels_], outputFeaturesLocal_[ubOffset2],
            {static_cast<uint16_t>(kernelSize_), static_cast<uint32_t>(inChannels_  * byteSizePerElements_), 0, 0, 0});
        SetFlag<HardEvent::MTE3_V>(ping1_);

        ping1_ = 1 - ping1_;
    }
    WaitFlag<HardEvent::V_S>(0);
}

template<typename T, bool modulated>
__aicore__ inline void DeformableConv2dV2Kernel<T, modulated>::Process()
{
    int8_t cvPing1 = 0;

    if ASCEND_IS_AIV {
        int32_t innerAicAivOffset = isOneAiv_ * singleLoopTask_;

        SetFlag<HardEvent::V_MTE2>(0);
        SetFlag<HardEvent::V_MTE2>(1);
        SetFlag<HardEvent::V_MTE2>(2);
        SetFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::MTE3_V>(1);

        for (int32_t taskOffset = 0; taskOffset < coreTaskCount_; taskOffset += 2 * singleLoopTask_) {
        
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

            bool matmulTaskEqualTwoTimescubeTileFlag = (taskOffset + 2 * singleLoopTask_) % (2 * cubeTileTaskCount_) == 0;
            bool finalMatmulFlag = taskOffset + 2 * singleLoopTask_ >= coreTaskCount_;
            if (matmulTaskEqualTwoTimescubeTileFlag || finalMatmulFlag) {
                CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
                cvPing1 = (cvPing1 + 1) % 8;
            }
        }

        WaitFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(1);
        WaitFlag<HardEvent::V_MTE2>(2);
        WaitFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(1);
    }

    if ASCEND_IS_AIC {
        for (int32_t taskOffset = 0; taskOffset < coreTaskCount_; taskOffset += 2 * cubeTileTaskCount_) {
            uint32_t taskCount = min(2 * cubeTileTaskCount_, coreTaskCount_ - taskOffset);

            CrossCoreWaitFlag<0x2>(0);
            ProcessCube(taskCount, taskOffset + taskStartOffset_);
            cvPing1 = (cvPing1 + 1) % 8;
        }
    }
}


extern "C" __global__ __aicore__ void deformable_conv2d_v2(GM_ADDR inputFeatures, GM_ADDR offset, GM_ADDR mask, GM_ADDR weight,
    GM_ADDR bias, GM_ADDR outputFeatures, GM_ADDR offsetOutput, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }

    TPipe pipe;

    if (TILING_KEY_IS(0)) {
        DeformableConv2dV2Kernel<DTYPE_INPUTFEATURES, false> op;
        op.mm_.SetSubBlockIdx(0);
        op.mm_.Init(&tilingData.mmTilingData, &pipe);

        op.Init(inputFeatures, offset, mask, weight, bias, outputFeatures, offsetOutput, usrWorkspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        DeformableConv2dV2Kernel<DTYPE_INPUTFEATURES, true> op;
        op.mm_.SetSubBlockIdx(0);
        op.mm_.Init(&tilingData.mmTilingData, &pipe);

        op.Init(inputFeatures, offset, mask, weight, bias, outputFeatures, offsetOutput, usrWorkspace, &tilingData, &pipe);
        op.Process();
    }
}
