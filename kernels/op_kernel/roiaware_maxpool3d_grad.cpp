#include "kernel_operator.h"
using namespace AscendC;
#define BUFFER_NUM 2
#define BUFFER_ALIGN_BYTE 32
#define COMPARE_ALGIN_BYTE 256


template<typename dataType, bool aicpu>
class KernelRoiawareMaxpool3dGrad {
public:
    __aicore__ inline KernelRoiawareMaxpool3dGrad() {}
    __aicore__ void Init(const GM_ADDR argmax, const GM_ADDR gradOut, GM_ADDR gradIn, TPipe* pipe,
                            const RoiawareMaxpool3dGradTilingData* tiling_data)
    
    {
        this->pipe_ = pipe;
        this->blkIdx_ = GetBlockIdx();
        this->InitTiling(tiling_data);
        this->InitGlobalBuffer(argmax, gradOut, gradIn, tiling_data);
        this->InitUBBuffer();
    }

    __aicore__ void Process()
    {
        uint32_t globalIdx = goutCoreStartIdx_;
        for (uint32_t curLoopTaskIdx = 0; curLoopTaskIdx < coreTask_; curLoopTaskIdx += singleLoopTask_, globalIdx += singleLoopTask_) {
            uint32_t curLoopTaskCount = min(singleLoopTask_, coreTask_ - curLoopTaskIdx);
            CopyInNTask(globalIdx, curLoopTaskCount);
            ProcessNTask(curLoopTaskCount);
        }
    }

protected:
    TPipe *pipe_;
    uint32_t blkIdx_;
    uint32_t channels_, nPoints_, channelAligned_, alignedArgmaxByteLength_;
    uint32_t coreTask_, singleLoopTask_, singleLoopOutput_;
    uint32_t argmaxCoreStartIdx_, goutCoreStartIdx_;

    TQue<QuePosition::VECIN, BUFFER_NUM> argmaxQue_, goutQue_;
    TBuf<TPosition::VECCALC> tmpMaskBuf_;
    TBuf<TPosition::VECCALC> tmpSelectedBuf_;
    TBuf<TPosition::VECCALC> ginBuf_;

    GlobalTensor<int32_t> argmaxGm_;
    GlobalTensor<dataType> goutGm_, ginGm_;

    DataCopyExtParams goutCopyParams_;
    DataCopyExtParams argmaxCopyParams_;
    DataCopyPadExtParams<dataType> goutPadParams_{true, 0, 0, 0};
    DataCopyPadExtParams<int32_t> argmaxPadParams_{true, 0, 0, 0};

private:
    __aicore__ inline void InitTiling(const RoiawareMaxpool3dGradTilingData* tiling_data)
    {
        this->channels_ = tiling_data->channels;
        this->nPoints_ = tiling_data->npoints;
        this->singleLoopTask_ = tiling_data->singleLoopTask;
        this->singleLoopOutput_ = tiling_data->singleLoopOutput;
        this->channelAligned_ = tiling_data->channelAligned;

        if (blkIdx_ >= tiling_data->firstSmallCoreIdx) {
            this->coreTask_ = tiling_data->coreTask - 1;
            argmaxCoreStartIdx_ = tiling_data->coreTask * blkIdx_ - (blkIdx_ - tiling_data->firstSmallCoreIdx);
            goutCoreStartIdx_ = argmaxCoreStartIdx_;
        } else {
            this->coreTask_ = tiling_data->coreTask;
            argmaxCoreStartIdx_ = tiling_data->coreTask * blkIdx_;
            goutCoreStartIdx_ = argmaxCoreStartIdx_;
        }
        this->alignedArgmaxByteLength_ = Ceil(singleLoopTask_ * channelAligned_ * sizeof(int32_t), COMPARE_ALGIN_BYTE) * COMPARE_ALGIN_BYTE;
    }

    __aicore__ inline void InitGlobalBuffer(const GM_ADDR argmax, const GM_ADDR gradOut, const GM_ADDR gradIn,
        const RoiawareMaxpool3dGradTilingData* tiling_data)
    {
        uint64_t argmaxLength = static_cast<uint64_t>(tiling_data->totalTask) * channels_ * sizeof(int32_t);
        uint64_t gradOutLength = static_cast<uint64_t>(tiling_data->totalTask) * channels_ * sizeof(dataType);
        uint64_t gradInLength = static_cast<uint64_t>(nPoints_) * channels_ * sizeof(dataType);

        this->argmaxGm_.SetGlobalBuffer((__gm__ int32_t*) argmax, argmaxLength);
        this->goutGm_.SetGlobalBuffer((__gm__ dataType*) gradOut, gradOutLength);
        this->ginGm_.SetGlobalBuffer((__gm__ dataType*) gradIn, gradInLength);
    }

    __aicore__ inline void InitUBBuffer()
    {
        this->pipe_->InitBuffer(argmaxQue_, BUFFER_NUM, alignedArgmaxByteLength_);

        this->pipe_->InitBuffer(goutQue_, BUFFER_NUM, singleLoopTask_ * channelAligned_ * sizeof(dataType));
        this->pipe_->InitBuffer(ginBuf_, singleLoopOutput_ * channelAligned_ * sizeof(dataType));

        uint32_t tmpMaskBufferByteLength = BUFFER_ALIGN_BYTE * Ceil(Ceil(channelAligned_ * singleLoopTask_, AscendCUtils::GetBitSize(sizeof(uint8_t))), BUFFER_ALIGN_BYTE);
        this->pipe_->InitBuffer(this->tmpMaskBuf_, tmpMaskBufferByteLength);
        this->pipe_->InitBuffer(this->tmpSelectedBuf_, singleLoopTask_ * channelAligned_ * sizeof(dataType));
    }
    
    __aicore__ inline void CopyInNTask(int64_t inputIdx, int32_t curLoopTaskCount)
    {
        LocalTensor<dataType> goutLocal = goutQue_.AllocTensor<dataType>();
        LocalTensor<int32_t> argmaxLocal = argmaxQue_.AllocTensor<int32_t>();

        int64_t goutLocalOffset = inputIdx * channels_;
        goutCopyParams_ = {static_cast<uint16_t>(curLoopTaskCount), static_cast<uint32_t>(channels_ * sizeof(dataType)), 0, 0, 0};
        DataCopyPad(goutLocal, goutGm_[goutLocalOffset], goutCopyParams_, goutPadParams_);
        
        int64_t argmaxLocalOffset = inputIdx * channels_;
        argmaxCopyParams_ = {static_cast<uint16_t>(curLoopTaskCount), static_cast<uint32_t>(channels_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(argmaxLocal, argmaxGm_[argmaxLocalOffset], argmaxCopyParams_, argmaxPadParams_);

        goutQue_.EnQue<dataType>(goutLocal);
        argmaxQue_.EnQue<int32_t>(argmaxLocal);
    }

    __aicore__ inline void ProcessNTask(const uint32_t &taskCount)
    {
        bool free_data = false;
        for (int32_t outputIdx = 0; outputIdx < nPoints_; outputIdx += singleLoopOutput_) {
            uint32_t curOutputTask = min(singleLoopOutput_, nPoints_ - outputIdx);
            if (outputIdx + singleLoopOutput_ >= nPoints_)
                free_data = true;
            Compute(taskCount, curOutputTask, outputIdx, free_data);
        }
    }

    __aicore__ inline void GinLocalReduceSum(const uint32_t &oi, const uint32_t &curLoopTaskCount, LocalTensor<dataType> &ginLocal, LocalTensor<dataType> &selectLocal)
    {
        uint32_t selectRepStride = (channelAligned_ * sizeof(dataType)) / (BUFFER_ALIGN_BYTE);
        uint64_t maxElementsCount = 256 / sizeof(dataType);
        uint32_t baseIdx = oi * channelAligned_;

        if (selectRepStride >= 64) {
            for (uint32_t taskIdx = 0; taskIdx < curLoopTaskCount; taskIdx++) {
                Add(ginLocal[oi * channelAligned_], selectLocal[taskIdx * channelAligned_], ginLocal[oi * channelAligned_], channels_);
            }
            return;
        }
        
        for (int32_t offsetIdx = 0; offsetIdx < channelAligned_; offsetIdx += maxElementsCount) {
            uint64_t mask = maxElementsCount < channelAligned_ - offsetIdx ? maxElementsCount : channelAligned_ - offsetIdx;
            Add(ginLocal[offsetIdx + baseIdx], selectLocal[offsetIdx], ginLocal[offsetIdx + baseIdx], mask, curLoopTaskCount, {1, 1, 1, 0, static_cast<uint8_t>(selectRepStride), 0});
        }
    }

    __aicore__ inline void Compute(const uint32_t &curLoopTaskCount, const uint32_t &curLoopOutputCount,
                                   const uint64_t &outputIdx, const bool &freeTensor)
    {
        LocalTensor<dataType> goutLocal = goutQue_.DeQue<dataType>();
        LocalTensor<int32_t> argmaxLocal = argmaxQue_.DeQue<int32_t>();
        LocalTensor<dataType> ginLocal = ginBuf_.AllocTensor<dataType>();
        LocalTensor<uint8_t> maskLocal = tmpMaskBuf_.AllocTensor<uint8_t>();
        LocalTensor<dataType> selectLocal = tmpSelectedBuf_.AllocTensor<dataType>();
        
        int32_t eventId1 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId1);
        WaitFlag<HardEvent::MTE3_V>(eventId1);
        Duplicate<dataType>(ginLocal, 0, singleLoopOutput_ * channelAligned_);
        PipeBarrier<PIPE_ALL>();

        for (int32_t oi = 0; oi < curLoopOutputCount; oi++) {
            int32_t curOutIdx = outputIdx + oi;
            CompareScalar(maskLocal, argmaxLocal, static_cast<int32_t>(curOutIdx), CMPMODE::EQ,
                alignedArgmaxByteLength_ / sizeof(int32_t));
            PipeBarrier<PIPE_V>();
            Select(selectLocal, maskLocal, goutLocal, static_cast<dataType>(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                curLoopTaskCount * channelAligned_);
            PipeBarrier<PIPE_V>();
            GinLocalReduceSum(oi, curLoopTaskCount, ginLocal, selectLocal);
        }

        int32_t eventId2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId2);
        WaitFlag<HardEvent::V_MTE3>(eventId2);
        
        SetAtomicAdd<dataType>();
        DataCopyPad(ginGm_[outputIdx * channels_], ginLocal, {static_cast<uint16_t>(curLoopOutputCount), static_cast<uint32_t>(channels_ * sizeof(dataType)), 0, 0, 0});
        SetAtomicNone();

        if (!freeTensor) {
            goutQue_.EnQue<dataType>(goutLocal);
            argmaxQue_.EnQue<int32_t>(argmaxLocal);
        } else {
            goutQue_.FreeTensor(goutLocal);
            argmaxQue_.FreeTensor(argmaxLocal);
        }
    }
};

extern "C" __global__ __aicore__ void roiaware_maxpool3d_grad(GM_ADDR argmax, GM_ADDR grad_out, GM_ADDR grad_in, GM_ADDR workspace,
                                                               GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    SetSysWorkspace(workspace);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }

    TPipe pipe;
    KernelRoiawareMaxpool3dGrad<float, false> op;
    op.Init(argmax, grad_out, grad_in, &pipe, &tiling_data);
    op.Process();
}
