/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#include <limits>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
using namespace AscendC;

namespace {
    constexpr uint32_t NUM_FEATURE = 8;
    constexpr uint32_t FLOAT_BYTE_SIZE = 4;
    constexpr uint32_t INT_BYTE_SIZE = 4;
    constexpr float SAVEVALUE = 1e-16;
    constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
}

class KernelGraphSoftmax {
public:
    __aicore__ inline KernelGraphSoftmax() {}
    __aicore__ inline void Init(TPipe *pipe, GM_ADDR src, GM_ADDR index, GM_ADDR softmaxResult, GM_ADDR workspace,
    const GraphSoftmaxTilingData* tiling)
    {
        pipe_ = pipe;
        blkIdx_ = GetBlockIdx();
        InitTiling(tiling);
        InitUB();
        InitGM(src, index, softmaxResult, workspace);
        InitEvent();
    }
    __aicore__ inline void Process()
    {
        SyncAll();

        // scatter max
        uint32_t endTaskOffset = taskOffset_ + coreTask_;
        for (int32_t offset = taskOffset_; offset < endTaskOffset; offset += singleLoopTaskCount_) {
            uint32_t taskCount = min(singleLoopTaskCount_, endTaskOffset - offset);

            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2_);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2_);
            
            CopyIn(offset, taskCount);

            SetFlag<HardEvent::MTE2_MTE3>(eventMTE2MTE3_);
            WaitFlag<HardEvent::MTE2_MTE3>(eventMTE2MTE3_);

            SetAtomicMax<float>();
            for (uint32_t i = 0; i < taskCount; i++) {
                DataCopy(scatterMaxResGm_[static_cast<uint32_t>(indexLocal_.GetValue(i)) * NUM_FEATURE], srcLocal_[i * NUM_FEATURE], NUM_FEATURE);
            }
            SetAtomicNone();
        }
        SyncAll();

        // scatter sum
        for (int32_t offset = taskOffset_; offset < endTaskOffset; offset += singleLoopTaskCount_) {
            uint32_t taskCount = min(singleLoopTaskCount_, endTaskOffset - offset);

            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2_);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2_);
            
            CopyIn(offset, taskCount);
            for (uint32_t i = 0; i < taskCount; i++) {
                DataCopy(scatterMaxLocal_[i * NUM_FEATURE], scatterMaxResGm_[static_cast<uint32_t>(indexLocal_.GetValue(i)) * NUM_FEATURE], NUM_FEATURE);
            }

            SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
            
            ComputeSum(taskCount);

            SetFlag<HardEvent::V_MTE3>(eventVMTE3_);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3_);

            SetAtomicAdd<float>();
            for (uint32_t i = 0; i < taskCount; i++) {
                DataCopy(scatterSumResGm_[static_cast<uint32_t>(indexLocal_.GetValue(i)) * NUM_FEATURE], srcLocal_[i * NUM_FEATURE], NUM_FEATURE);
            }
            SetAtomicNone();

            DataCopy(softmaxResultGm_[static_cast<uint64_t>(offset) * NUM_FEATURE], srcLocal_, NUM_FEATURE * taskCount);
        }
        SyncAll();

        // calculate softmax
        for (int32_t offset = taskOffset_; offset < endTaskOffset; offset += singleLoopTaskCount_) {
            uint32_t taskCount = min(singleLoopTaskCount_, endTaskOffset - offset);

            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2_);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2_);
            
            DataCopyExtParams indexDataCopyParams{1, static_cast<uint16_t>(taskCount)*INT_BYTE_SIZE, 0, 0, 0};
            DataCopy(srcLocal_, softmaxResultGm_[static_cast<uint64_t>(offset) * NUM_FEATURE], NUM_FEATURE * taskCount);
            DataCopyPad(indexLocal_, indexGm_[static_cast<uint64_t>(offset)], indexDataCopyParams, {true, 0, 0, 0});
            for (uint32_t i = 0; i < taskCount; i++) {
                DataCopy(scatterMaxLocal_[i * NUM_FEATURE], scatterSumResGm_[static_cast<uint32_t>(indexLocal_.GetValue(i)) * NUM_FEATURE], NUM_FEATURE);
            }
            
            SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);

            ComputeSoftmax(taskCount);
            
            SetFlag<HardEvent::V_MTE3>(eventVMTE3_);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3_);

            CopyOut(offset, taskCount);
        }
    }

private:
    __aicore__ inline void InitTiling(const GraphSoftmaxTilingData* tiling)
    {
        this->coreTask_ = tiling->coreTask;
        this->coreWorkspace_ = tiling->coreWorkspace;
        this->totalTask_ = tiling->totalTask;
        this->totalWorkspace_ = tiling->totalWorkspace;
        if (blkIdx_ < tiling->bigCoreCount) {
            this->taskOffset_ = blkIdx_ * coreTask_;
        } else {
            this->taskOffset_ = tiling->bigCoreCount * coreTask_ +
                (blkIdx_ - tiling->bigCoreCount) * (coreTask_ - 1);
            this->coreTask_ = this->coreTask_ - 1;
        }
        this->singleLoopTaskCount_ = tiling->singleLoopTaskCount;
    }

    __aicore__ inline void InitGM(GM_ADDR src, GM_ADDR index, GM_ADDR softmaxResult, GM_ADDR workspace)
    {
        this->srcGm_.SetGlobalBuffer((__gm__ float*) src, totalTask_ * NUM_FEATURE);
        this->indexGm_.SetGlobalBuffer((__gm__ uint32_t*) index, totalTask_);
        this->softmaxResultGm_.SetGlobalBuffer((__gm__ float*) softmaxResult, totalTask_ * NUM_FEATURE);
        scatterMaxResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        scatterSumResGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + totalWorkspace_ * NUM_FEATURE);

        GlobalTensor<float> scatterMaxResTmpGlobal = scatterMaxResGm_[coreWorkspace_ * blkIdx_ * NUM_FEATURE];
        InitGlobalMemory(scatterMaxResTmpGlobal, coreWorkspace_ * NUM_FEATURE, NEG_INF);
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<float> scatterSumResTmpGlobal = scatterSumResGm_[coreWorkspace_ * blkIdx_ * NUM_FEATURE];
        InitGlobalMemory(scatterSumResTmpGlobal, coreWorkspace_ * NUM_FEATURE, 0.0f);
    }

    __aicore__ inline void InitUB()
    {
        pipe_->InitBuffer(srcBuf_, singleLoopTaskCount_ * NUM_FEATURE * FLOAT_BYTE_SIZE);
        pipe_->InitBuffer(scatterMaxBuf_, singleLoopTaskCount_ * NUM_FEATURE * FLOAT_BYTE_SIZE);
        pipe_->InitBuffer(indexBuf_, singleLoopTaskCount_ * INT_BYTE_SIZE);

        srcLocal_ = srcBuf_.Get<float>();
        scatterMaxLocal_ = scatterMaxBuf_.Get<float>();
        indexLocal_ = indexBuf_.Get<uint32_t>();
    }

    __aicore__ inline void InitEvent()
    {
        eventMTE3MTE2_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        eventMTE2V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        eventVMTE3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        eventMTE2MTE3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t taskCount);
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t taskCount);
    __aicore__ inline void ComputeSum(uint32_t taskCount);
    __aicore__ inline void ComputeSoftmax(uint32_t taskCount);

private:
    TPipe* pipe_;
    int32_t eventMTE3MTE2_, eventMTE2V_, eventVMTE3_, eventMTE2MTE3_;
    uint32_t blkIdx_;
    uint32_t coreTask_, coreWorkspace_, totalTask_, totalWorkspace_, taskOffset_, singleLoopTaskCount_;

    GlobalTensor<float> srcGm_;
    GlobalTensor<float> softmaxResultGm_;
    GlobalTensor<float> scatterMaxResGm_;
    GlobalTensor<float> scatterSumResGm_;
    GlobalTensor<uint32_t> indexGm_;

    TBuf<TPosition::VECCALC> srcBuf_, indexBuf_, scatterMaxBuf_;
    LocalTensor<float> srcLocal_;
    LocalTensor<float> scatterMaxLocal_;
    LocalTensor<uint32_t> indexLocal_;
};

__aicore__ inline void KernelGraphSoftmax::CopyIn(uint32_t offset, uint32_t taskCount)
{
    DataCopyExtParams indexDataCopyParams{1, static_cast<uint16_t>(taskCount)*INT_BYTE_SIZE, 0, 0, 0};

    DataCopy(srcLocal_, srcGm_[static_cast<uint64_t>(offset) * NUM_FEATURE], NUM_FEATURE * taskCount);
    DataCopyPad(indexLocal_, indexGm_[static_cast<uint64_t>(offset)], indexDataCopyParams, {true, 0, 0, 0});
}

__aicore__ inline void KernelGraphSoftmax::CopyOut(uint32_t offset, uint32_t taskCount)
{
    DataCopy(softmaxResultGm_[static_cast<uint64_t>(offset) * NUM_FEATURE], srcLocal_, NUM_FEATURE * taskCount);
}

__aicore__ inline void KernelGraphSoftmax::ComputeSum(uint32_t taskCount)
{
    Sub(srcLocal_, srcLocal_, scatterMaxLocal_, taskCount * NUM_FEATURE);
    Exp(srcLocal_, srcLocal_, taskCount * NUM_FEATURE);
}

__aicore__ inline void KernelGraphSoftmax::ComputeSoftmax(uint32_t taskCount)
{
    Adds(scatterMaxLocal_, scatterMaxLocal_, SAVEVALUE, taskCount * NUM_FEATURE);
    Div(srcLocal_, srcLocal_, scatterMaxLocal_, taskCount * NUM_FEATURE);
}

extern "C" __global__ __aicore__ void graph_softmax(GM_ADDR src, GM_ADDR index, GM_ADDR softmaxResult,
 GM_ADDR workspace, GM_ADDR tiling_data)
{
    GET_TILING_DATA(tiling, tiling_data);
    SetSysWorkspace(workspace);

    TPipe pipe;
    KernelGraphSoftmax op;
    op.Init(&pipe, src, index, softmaxResult, workspace, &tiling);
    op.Process();
}