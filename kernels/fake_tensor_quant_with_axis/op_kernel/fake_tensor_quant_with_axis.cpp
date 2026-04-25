/**
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/

#include "kernel_operator.h"

using namespace AscendC;

static constexpr uint32_t BUFFER_NUM = 2;
static constexpr uint32_t NORMALMODE = 0;
static constexpr uint32_t SPECIALMODE = 1;

template<typename dType>
class FakeTensorQuantWithAxisKernal {
public:
    __aicore__ inline FakeTensorQuantWithAxisKernal() {}

    __aicore__ inline void Init(GM_ADDR inputsIn, GM_ADDR amaxIn, GM_ADDR outPut,
        const FakeTensorQuantWithAxisTilingData* __restrict tilingData, TPipe* tPipe)
    {
        ParseTilingData(tilingData);
        InitParams();
        InitAndSetBuffer(inputsIn, amaxIn, outPut, tPipe);
    }
    __aicore__ inline void InitAndSetBuffer(GM_ADDR inputsIn, GM_ADDR amaxIn, GM_ADDR outPut, TPipe* tPipe)
    {
        inputsInGm_.SetGlobalBuffer((__gm__ dType*)(inputsIn + coreOffset_));
        outGm_.SetGlobalBuffer((__gm__ dType*)(outPut + coreOffset_));
        amaxGm_.SetGlobalBuffer((__gm__ dType*)(amaxIn));

        pipe_ = tPipe;

        pipe_->InitBuffer(inputsInQueue_, BUFFER_NUM, normalCopyElemNum_ * sizeof(dType));
        pipe_->InitBuffer(outQueue_, BUFFER_NUM, normalCopyElemNum_ * sizeof(dType));
        if (tilingMode_ == NORMALMODE) {
            pipe_->InitBuffer(amaxBuf_, normalCopyElemNum_ * sizeof(dType));
        } else {
            pipe_->InitBuffer(amaxInQueue_, BUFFER_NUM, normalCopyElemNum_ * sizeof(dType));
        }
    }
    __aicore__ inline void InitParams()
    {
        blockIdx_ = GetBlockIdx();
        normalCopyElemNum_ = tilingData_.normalCopyElemNumInOneTask;
        uint32_t headCoreNum = tilingData_.headCoreNum;
        countInOneTask_ = tilingData_.countInOneTask;
        lastCopyElemNumInOneTask_ = tilingData_.lastCopyElemNumInOneTask;
        totalElemNumPerTask_ = tilingData_.totalElemNumPerTask;
        if (blockIdx_ < headCoreNum) {
            copyNum_ = tilingData_.eachHeadCoreTaskNum;
            taskId_ = tilingData_.eachHeadCoreTaskNum * blockIdx_;
            coreOffset_ = taskId_ * totalElemNumPerTask_ * sizeof(dType);
        } else {
            copyNum_ = tilingData_.eachTailCoreTaskNum;
            taskId_ = tilingData_.eachHeadCoreTaskNum * headCoreNum +
                      (blockIdx_ - headCoreNum) * tilingData_.eachTailCoreTaskNum;
            coreOffset_ = taskId_ * totalElemNumPerTask_ * sizeof(dType);
        }

        amaxTotalNum_ = tilingData_.amaxTotalNum;
        tilingMode_ = tilingData_.tilingMode;
        maxBound_ = tilingData_.maxBound;
        minBound_ = tilingData_.minBound;
    }
    __aicore__ inline void ParseTilingData(const FakeTensorQuantWithAxisTilingData* tilingData)
    {
        tilingData_.headCoreNum = tilingData->headCoreNum;
        tilingData_.totalElemNumPerTask = tilingData->totalElemNumPerTask;
        tilingData_.copyNumPerTask = tilingData->copyNumPerTask;
        tilingData_.eachHeadCoreTaskNum = tilingData->eachHeadCoreTaskNum;
        tilingData_.eachTailCoreTaskNum = tilingData->eachTailCoreTaskNum;
        tilingData_.countInOneTask = tilingData->countInOneTask;
        tilingData_.normalCopyElemNumInOneTask = tilingData->normalCopyElemNumInOneTask;
        tilingData_.lastCopyElemNumInOneTask = tilingData->lastCopyElemNumInOneTask;
        tilingData_.amaxTotalNum = tilingData->amaxTotalNum;
        tilingData_.maxBound = tilingData->maxBound;
        tilingData_.minBound = tilingData->minBound;
        tilingData_.tilingMode = tilingData->tilingMode;
    }

    __aicore__ inline void ProcessNormal()
    {
        amaxTmpLocal_ = amaxBuf_.Get<dType>();
        for (uint32_t i = 0; i < copyNum_; ++i) {
            if (amaxIndex_ != (taskId_ % amaxTotalNum_)) {
                amaxIndex_ = taskId_ % amaxTotalNum_;
                amaxValue_ = amaxGm_.GetValue(amaxIndex_); // 更新amax
            }
            for (uint32_t j = 0; j < countInOneTask_; ++j) {
                CopyIn(i, j);
                Compute(j);
                CopyOut(i, j);
            }
            taskId_++;
        }
    }
    __aicore__ inline void ProcessSpecial()
    {
        // 在NormalMode模式下，amax与totalElemNumPerTask_数量是一样的。
        // 体现在做copyin的时候，需要与inputs一样搬运amax到local
        for (uint32_t i = 0; i < copyNum_; ++i) {
            for (uint32_t j = 0; j < countInOneTask_; ++j) {
                CopyInSpecial(i, j);
                amaxTmpLocal_ = amaxInQueue_.DeQue<dType>();
                Compute(j);
                amaxInQueue_.FreeTensor(amaxTmpLocal_);
                CopyOut(i, j);
            }
        }
    }
    __aicore__ inline void Process()
    {
        if (tilingMode_ == NORMALMODE) {
            ProcessNormal();
        } else if (tilingMode_ == SPECIALMODE) {
            ProcessSpecial();
        }
    }

    __aicore__ inline void CopyIn(uint32_t taskId, uint32_t progress)
    {
        LocalTensor<dType> inputsInLocal = inputsInQueue_.AllocTensor<dType>();
        DataCopyPadParams padParams {true, 0, 0, 0};
        if (likely(progress < countInOneTask_ - 1)) {
            DataCopyParams copyParamsIn {1, (uint16_t)(normalCopyElemNum_ * sizeof(dType)), 0, 0};
            DataCopyPad(inputsInLocal, inputsInGm_[taskId * totalElemNumPerTask_ + progress * normalCopyElemNum_],
                copyParamsIn, padParams);
        } else {
            DataCopyParams copyParamsIn {1, (uint16_t)(lastCopyElemNumInOneTask_ * sizeof(dType)), 0, 0};
            DataCopyPad(inputsInLocal,
                inputsInGm_[taskId * totalElemNumPerTask_ + (countInOneTask_ - 1) * normalCopyElemNum_], copyParamsIn,
                padParams);
        }
        inputsInQueue_.EnQue(inputsInLocal);
    }
    __aicore__ inline void CopyInSpecial(uint32_t taskId, uint32_t progress)
    {
        LocalTensor<dType> inputsInLocal = inputsInQueue_.AllocTensor<dType>();
        amaxTmpLocal_ = amaxInQueue_.AllocTensor<dType>();

        DataCopyPadParams padParams {true, 0, 0, 0};
        if (likely(progress < countInOneTask_ - 1)) {
            DataCopyParams copyParamsIn {1, (uint16_t)(normalCopyElemNum_ * sizeof(dType)), 0, 0};
            DataCopyPad(inputsInLocal, inputsInGm_[taskId * totalElemNumPerTask_ + progress * normalCopyElemNum_],
                copyParamsIn, padParams);
            DataCopyPad(amaxTmpLocal_, amaxGm_[progress * normalCopyElemNum_], copyParamsIn, padParams);
        } else {
            DataCopyParams copyParamsIn {1, (uint16_t)(lastCopyElemNumInOneTask_ * sizeof(dType)), 0, 0};
            DataCopyPad(inputsInLocal,
                inputsInGm_[taskId * totalElemNumPerTask_ + (countInOneTask_ - 1) * normalCopyElemNum_], copyParamsIn,
                padParams);
            DataCopyPad(amaxTmpLocal_, amaxGm_[(countInOneTask_ - 1) * normalCopyElemNum_], copyParamsIn, padParams);
        }
        inputsInQueue_.EnQue(inputsInLocal);
        amaxInQueue_.EnQue(amaxTmpLocal_);
    }

    __aicore__ inline void Compute(uint32_t progress)
    {
        LocalTensor<dType> inputsInLocal = inputsInQueue_.DeQue<dType>();
        LocalTensor<dType> outLocal = outQueue_.AllocTensor<dType>();
        uint32_t elemNumForCopy = 0;
        if (progress < countInOneTask_ - 1) {
            elemNumForCopy = normalCopyElemNum_;
        } else {
            elemNumForCopy = lastCopyElemNumInOneTask_; // 最后一次
        }
        if (tilingMode_ == NORMALMODE) {
            Duplicate(amaxTmpLocal_, static_cast<dType>(amaxValue_), elemNumForCopy);
        }
        Duplicate(outLocal, static_cast<dType>(maxBound_), elemNumForCopy);
        Div(amaxTmpLocal_, outLocal, amaxTmpLocal_, elemNumForCopy);
        Mul(inputsInLocal, inputsInLocal, amaxTmpLocal_, elemNumForCopy); // input * scale
        Round(outLocal, inputsInLocal, elemNumForCopy); // round(input * scale)
        // output > max_bound ? max_bound : output;
        AscendC::ClampMax<dType>(inputsInLocal, outLocal, static_cast<dType>(maxBound_), elemNumForCopy);
        // output < min_bound ? min_bound : output;
        AscendC::ClampMin<dType>(outLocal, inputsInLocal, static_cast<dType>(minBound_), elemNumForCopy);

        Div(outLocal, outLocal, amaxTmpLocal_, elemNumForCopy);
        inputsInQueue_.FreeTensor(inputsInLocal);
        outQueue_.EnQue(outLocal);
    }
    __aicore__ inline void CopyOut(uint32_t taskId, uint32_t progress)
    {
        LocalTensor<dType> outLocal = outQueue_.DeQue<dType>();
        if (progress < countInOneTask_ - 1) {
            DataCopyParams copyParamsOut {1, (uint16_t)(normalCopyElemNum_ * sizeof(dType)), 0, 0};
            DataCopyPad(outGm_[taskId * totalElemNumPerTask_ + progress * normalCopyElemNum_], outLocal, copyParamsOut);
        } else {
            DataCopyParams copyParamsOut {1, (uint16_t)(lastCopyElemNumInOneTask_ * sizeof(dType)), 0, 0};
            DataCopyPad(outGm_[taskId * totalElemNumPerTask_ + (countInOneTask_ - 1) * normalCopyElemNum_], outLocal,
                copyParamsOut);
        }
        outQueue_.FreeTensor(outLocal);
    }

private:
    FakeTensorQuantWithAxisTilingData tilingData_;
    GlobalTensor<dType> inputsInGm_;
    GlobalTensor<dType> amaxGm_;
    GlobalTensor<dType> outGm_;

    LocalTensor<dType> amaxTmpLocal_;

    TPipe* pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputsInQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> amaxInQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;

    TBuf<TPosition::VECCALC> amaxBuf_;

    uint32_t blockIdx_;
    uint32_t copyNum_;
    uint32_t countInOneTask_;
    uint32_t coreOffset_;
    uint32_t normalCopyElemNum_;
    uint32_t lastCopyElemNumInOneTask_;
    uint32_t amaxTotalNum_;
    uint32_t totalElemNumPerTask_;
    uint32_t tilingMode_;

    int32_t amaxIndex_ = -1; // 设为-1方便初始化逻辑
    uint32_t taskId_;
    dType amaxValue_;
    int32_t maxBound_;
    int32_t minBound_;
};


extern "C" __global__ __aicore__ void fake_tensor_quant_with_axis(
    GM_ADDR inputsIn, GM_ADDR amaxIn, GM_ADDR outPut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        FakeTensorQuantWithAxisKernal<float> op;
        op.Init(inputsIn, amaxIn, outPut, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        FakeTensorQuantWithAxisKernal<half> op;
        op.Init(inputsIn, amaxIn, outPut, &tiling_data, &pipe);
        op.Process();
    }
}