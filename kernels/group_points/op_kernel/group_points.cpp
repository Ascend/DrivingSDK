/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 */
#include "kernel_operator.h"
using namespace AscendC;


class KernelGroupPoints {
public:
    __aicore__ inline KernelGroupPoints() {}
    __aicore__ inline void Init(GM_ADDR points, GM_ADDR indices, GM_ADDR out, const GroupPointsTilingData* tiling_data)
    {
        uint32_t batchSize_ = tiling_data->batchSize;
        cSize_ = tiling_data->cSize;
        nSize_ = tiling_data->nSize;
        npoints_ = tiling_data->npoints;
        nsample_ = tiling_data->nsample;
        cAligned = tiling_data->cAligned;
        maxUbTaskNum = tiling_data->maxUbTaskNum;
        coreTaskNum = tiling_data->coreTaskNum;
        mainCoreLoop = tiling_data->mainCoreLoop;
        mainCoreTail = tiling_data->mainCoreTail;
        lastCoreLoop = tiling_data->lastCoreLoop;
        lastCoreTail = tiling_data->lastCoreTail;
        lastCoreTailAligned = tiling_data->lastCoreTailAligned;
        useCoreNum = tiling_data->useCoreNum;

        uint64_t inputLength = static_cast<uint64_t>(batchSize_) * nSize_ * cSize_;
        uint64_t indicesLength = static_cast<uint64_t>(batchSize_) * npoints_ * nsample_;
        uint64_t outLength = indicesLength * cSize_;

        inputGm.SetGlobalBuffer((__gm__ DTYPE_POINTS*)points, inputLength);
        indicesGm.SetGlobalBuffer((__gm__ DTYPE_INDICES*)indices, indicesLength);
        outGm.SetGlobalBuffer((__gm__ DTYPE_OUT*)out, outLength);

        pipe.InitBuffer(inputBuffer, maxUbTaskNum * cAligned * sizeof(DTYPE_POINTS));
        pipe.InitBuffer(indicesBuffer, maxUbTaskNum * sizeof(DTYPE_INDICES));

        eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
        eventIDMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void Process()
    {
        uint32_t blockIdx = GetBlockIdx();
        if (blockIdx > useCoreNum) {
            return;
        }

        uint64_t taskOffset = blockIdx * coreTaskNum;
        uint32_t loopCount = mainCoreLoop;
        uint32_t tailTaskNum = mainCoreTail;
        uint32_t tailTaskAligned = mainCoreTail; // have been set 64-aligned in mainCore
        if (blockIdx == useCoreNum - 1) {
            loopCount = lastCoreLoop;
            tailTaskNum = lastCoreTail;
            tailTaskAligned = lastCoreTailAligned;
        }

        for (int32_t i = 0; i < loopCount; i++) {
            CopyInAndCopyOut(taskOffset, maxUbTaskNum, maxUbTaskNum);
            taskOffset += maxUbTaskNum;
        }
        if (tailTaskNum != 0) {
            CopyInAndCopyOut(taskOffset, tailTaskNum, tailTaskAligned);
        }
    }

private:
    __aicore__ inline void CopyInAndCopyOut(uint64_t taskOffset, uint32_t taskNum, uint32_t taskAligned)
    {
        LocalTensor<DTYPE_POINTS> input_local = inputBuffer.Get<DTYPE_POINTS>();
        LocalTensor<DTYPE_INDICES> indices_local = indicesBuffer.Get<DTYPE_INDICES>();

        DataCopy(indices_local, indicesGm[taskOffset], taskAligned);
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

        for (int32_t i = 0; i < taskNum; i++) {
            uint32_t idx = indices_local.GetValue(i);
            uint32_t b_idx = (taskOffset + i) / npoints_ / nsample_;
            uint64_t src_idx = b_idx * nSize_ * cSize_ + idx * cSize_;
            DataCopy(input_local[i * cAligned], inputGm[src_idx], cAligned);
        }
        SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
        WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);

        DataCopyExtParams outCopyParams {
            static_cast<uint16_t>(taskNum), static_cast<uint32_t>(cSize_ * sizeof(DTYPE_POINTS)), 0, 0, 0};
        DataCopyPad(outGm[taskOffset * cSize_], input_local, outCopyParams);
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> indicesBuffer, inputBuffer;
    GlobalTensor<DTYPE_POINTS> inputGm;
    GlobalTensor<DTYPE_INDICES> indicesGm;
    GlobalTensor<DTYPE_OUT> outGm;

    uint32_t cSize_;
    uint32_t nSize_;
    uint32_t npoints_;
    uint32_t nsample_;
    uint32_t cAligned;
    uint32_t maxUbTaskNum;
    uint32_t coreTaskNum;
    uint32_t mainCoreLoop;
    uint32_t mainCoreTail;
    uint32_t lastCoreLoop;
    uint32_t lastCoreTail;
    uint32_t lastCoreTailAligned;
    uint32_t useCoreNum;

    uint8_t eventIDMTE2ToMTE3;
    uint8_t eventIDMTE3ToMTE2;
};

extern "C" __global__ __aicore__ void group_points(
    GM_ADDR points, GM_ADDR indices, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGroupPoints op;
    op.Init(points, indices, out, &tiling_data);
    op.Process();
}