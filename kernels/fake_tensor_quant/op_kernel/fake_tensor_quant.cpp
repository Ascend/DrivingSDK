/**
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/

#include "kernel_operator.h"

using namespace AscendC;

static constexpr uint32_t BUFFER_NUM = 2;

template<typename dType>
class FakeTensorQuantKernal {
public:
    __aicore__ inline FakeTensorQuantKernal() {}

    __aicore__ inline void Init(GM_ADDR inputsIn, GM_ADDR amaxIn, GM_ADDR outPut,
        const FakeTensorQuantTilingData* __restrict tilingData, TPipe* tPipe)
    {
        ParseTilingData(tilingData);
        InitParams();
        InitAndSetBuffer(inputsIn, amaxIn, outPut, tPipe);
    }
    __aicore__ inline void InitAndSetBuffer(GM_ADDR inputsIn, GM_ADDR amaxIn, GM_ADDR outPut, TPipe* tPipe)
    {
        inputsInGm_.SetGlobalBuffer((__gm__ dType*)(inputsIn + coreOffset_));
        amaxGm_.SetGlobalBuffer((__gm__ dType*)(amaxIn));
        outGm_.SetGlobalBuffer((__gm__ dType*)(outPut + coreOffset_));

        pipe_ = tPipe;

        pipe_->InitBuffer(inputsInQueue_, BUFFER_NUM, normalCopyElemNum_ * sizeof(dType));
        pipe_->InitBuffer(outQueue_, BUFFER_NUM, normalCopyElemNum_ * sizeof(dType));

        pipe_->InitBuffer(amaxBuf_, normalCopyElemNum_ * sizeof(dType));

        amax_ = amaxGm_.GetValue(0);
    }
    __aicore__ inline void Process()
    {
        LocalTensor<dType> inputsInLocal = inputsInQueue_.AllocTensor<dType>();
        amaxInLocal_ = amaxBuf_.Get<dType>();
        Duplicate(inputsInLocal, static_cast<dType>(maxBound_), normalCopyElemNum_);
        Duplicate(amaxInLocal_, amax_, normalCopyElemNum_);
        Div(amaxInLocal_, inputsInLocal, amaxInLocal_, normalCopyElemNum_);
        inputsInQueue_.FreeTensor(inputsInLocal);
        for (uint32_t i = 0; i < copyNum_; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
    __aicore__ inline void InitParams()
    {
        blockIdx_ = GetBlockIdx();
        normalCopyElemNum_ = tilingData_.normalCopyElemNum;
        uint32_t headCoreNum = tilingData_.headCoreNum;
        if (blockIdx_ < headCoreNum) {
            copyNum_ = tilingData_.headCoreCopyNum;
            lastCopyElemNum_ = tilingData_.headCoreLastCopyElemNum;
            coreOffset_ = tilingData_.eachHeadCoreElemNum * blockIdx_ * sizeof(dType);
        } else {
            copyNum_ = tilingData_.tailCoreCopyNum;
            lastCopyElemNum_ = tilingData_.tailCoreLastCopyElemNum;
            coreOffset_ = tilingData_.eachHeadCoreElemNum * headCoreNum * sizeof(dType) +
                          (blockIdx_ - headCoreNum) * tilingData_.eachTailCoreElemNum * sizeof(dType);
        }

        maxBound_ = tilingData_.maxBound;
        minBound_ = tilingData_.minBound;
    }
    __aicore__ inline void ParseTilingData(const FakeTensorQuantTilingData* tilingData)
    {
        tilingData_.headCoreNum = tilingData->headCoreNum;
        tilingData_.normalCopyElemNum = tilingData->normalCopyElemNum;
        tilingData_.eachHeadCoreElemNum = tilingData->eachHeadCoreElemNum;
        tilingData_.eachTailCoreElemNum = tilingData->eachTailCoreElemNum;
        tilingData_.headCoreCopyNum = tilingData->headCoreCopyNum;
        tilingData_.tailCoreCopyNum = tilingData->tailCoreCopyNum;
        tilingData_.headCoreLastCopyElemNum = tilingData->headCoreLastCopyElemNum;
        tilingData_.tailCoreLastCopyElemNum = tilingData->tailCoreLastCopyElemNum;
        tilingData_.maxBound = tilingData->maxBound;
        tilingData_.minBound = tilingData->minBound;
    }
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        LocalTensor<dType> inputsInLocal = inputsInQueue_.AllocTensor<dType>();
        DataCopyPadParams padParams {true, 0, 0, 0};
        if (likely(progress < copyNum_ - 1)) {
            DataCopyParams copyParamsIn {1, (uint16_t)(normalCopyElemNum_ * sizeof(dType)), 0, 0};
            DataCopyPad(inputsInLocal, inputsInGm_[progress * normalCopyElemNum_], copyParamsIn, padParams);
        } else {
            DataCopyParams copyParamsIn {1, (uint16_t)(lastCopyElemNum_ * sizeof(dType)), 0, 0};
            DataCopyPad(inputsInLocal, inputsInGm_[(copyNum_ - 1) * normalCopyElemNum_], copyParamsIn, padParams);
        }
        inputsInQueue_.EnQue(inputsInLocal);
    }

    __aicore__ inline void Compute(uint32_t progress)
    {
        LocalTensor<dType> inputsInLocal = inputsInQueue_.DeQue<dType>();
        LocalTensor<dType> outLocal = outQueue_.AllocTensor<dType>();
        uint32_t elemNumForCopy = 0;
        if (progress < copyNum_ - 1) {
            elemNumForCopy = normalCopyElemNum_;
        } else {
            elemNumForCopy = lastCopyElemNum_; // 最后一次
        }
        Mul(inputsInLocal, inputsInLocal, amaxInLocal_, elemNumForCopy); // input * scale
        Round(outLocal, inputsInLocal, elemNumForCopy);                                     // round(input * scale)
        // output > max_bound ? max_bound : output;
        AscendC::ClampMax<dType>(inputsInLocal, outLocal, static_cast<dType>(maxBound_), elemNumForCopy);
        // output < min_bound ? min_bound : output;
        AscendC::ClampMin<dType>(outLocal, inputsInLocal, static_cast<dType>(minBound_), elemNumForCopy);

        Div(outLocal, outLocal, amaxInLocal_, elemNumForCopy);
        inputsInQueue_.FreeTensor(inputsInLocal);
        outQueue_.EnQue(outLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        LocalTensor<dType> outLocal = outQueue_.DeQue<dType>();
        if (progress < copyNum_ - 1) {
            DataCopyParams copyParamsOut {1, (uint16_t)(normalCopyElemNum_ * sizeof(dType)), 0, 0};
            DataCopyPad(outGm_[progress * normalCopyElemNum_], outLocal, copyParamsOut);
        } else {
            DataCopyParams copyParamsOut {1, (uint16_t)(lastCopyElemNum_ * sizeof(dType)), 0, 0};
            DataCopyPad(outGm_[(copyNum_ - 1) * normalCopyElemNum_], outLocal, copyParamsOut);
        }
        outQueue_.FreeTensor(outLocal);
    }

private:
    FakeTensorQuantTilingData tilingData_;
    GlobalTensor<dType> inputsInGm_;
    GlobalTensor<dType> amaxGm_;
    GlobalTensor<dType> outGm_;

    TPipe* pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputsInQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;

    TBuf<TPosition::VECCALC> amaxBuf_;
    LocalTensor<dType> amaxInLocal_;
    uint32_t blockIdx_;
    uint32_t copyNum_;
    uint32_t coreOffset_;
    uint32_t normalCopyElemNum_; // 每次搬运的个数
    uint32_t lastCopyElemNum_;   // 最后一次搬运的个数

    dType amax_;

    int32_t maxBound_;
    int32_t minBound_;
};


extern "C" __global__ __aicore__ void fake_tensor_quant(
    GM_ADDR inputsIn, GM_ADDR amaxIn, GM_ADDR outPut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(1)) {
        FakeTensorQuantKernal<float> op;
        op.Init(inputsIn, amaxIn, outPut, &tiling_data, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        FakeTensorQuantKernal<half> op;
        op.Init(inputsIn, amaxIn, outPut, &tiling_data, &pipe);
        op.Process();
    }
}