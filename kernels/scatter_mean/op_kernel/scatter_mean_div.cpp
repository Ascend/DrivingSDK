/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 */
#include "kernel_operator.h"

using namespace AscendC;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MAX_DEAL_NUM = 2048;
constexpr uint32_t MAX_MASK = 64;

class KernelScatterMeanDiv {
public:
    __aicore__ inline KernelScatterMeanDiv() {}
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR count, GM_ADDR out, ScatterMeanDivTilingData *tiling_data, TPipe* tmpPipe)
    {
        pipe = tmpPipe;
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        TilingDataInit(tiling_data);

        countGm.SetGlobalBuffer((__gm__ DTYPE_COUNT*)count, countNum);
        srcGm.SetGlobalBuffer((__gm__ DTYPE_SRC*)src, srcNum);
        outGm.SetGlobalBuffer((__gm__ DTYPE_OUT*)out, outNum);

        eventIdMte2ToV_0 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        pipe->InitBuffer(inQueueCount, AlignUp(ubCountNum, countEachBlock) * sizeof(DTYPE_COUNT));
        pipe->InitBuffer(inQueueSrc, AlignUp(ubTailNum, countEachBlock) * sizeof(DTYPE_SRC));
    }

    __aicore__ inline void TilingDataInit(ScatterMeanDivTilingData *tiling_data)
    {
        curBlockIdx = GetBlockIdx();
        usedCoreNum = tiling_data->usedCoreNum;
        tail = tiling_data->tail;
        taskNum = tiling_data->taskNum;
        taskEachLine = tiling_data->taskEachLine;
        taskLastLine = tiling_data->taskLastLine;
        bigCoreNum = tiling_data->bigCoreNum;
        srcNum = tiling_data->srcNum;
        countNum = tiling_data->countNum;
        outNum = tiling_data->outNum;
        ubCountNum = tiling_data->ubCountNum;
        ubTailNum = tiling_data->ubTailNum;

        uint64_t coreDataLine = tiling_data->coreSmallLine;
        if (curBlockIdx < bigCoreNum) {
            coreDataLine = coreDataLine + 1;
            countBaseOffset = curBlockIdx * coreDataLine;
        } else {
            taskNum = tiling_data->taskNumSmall;
            taskEachLine = tiling_data->taskEachLineSmall;
            taskLastLine = tiling_data->taskLastLineSmall;
            countBaseOffset = bigCoreNum * (coreDataLine + 1) + (curBlockIdx - bigCoreNum) * coreDataLine;
        }

        countEachBlock = BLOCK_SIZE / sizeof(DTYPE_COUNT);
        dataEachBlock = BLOCK_SIZE / sizeof(DTYPE_SRC);
    }
    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < taskNum - 1; i++) {
            ComputeEachTask(i, taskEachLine);
        }
        if (taskLastLine != 0) {
            ComputeEachTask(taskNum - 1, taskLastLine);
        }
    }

private:
    __aicore__ inline void CopyParamasInit(const uint32_t calCount)
    {
        copyParamsOut.blockCount = 1;
        copyParamsOut.blockLen = static_cast<uint32_t>(calCount * sizeof(float));
        copyParamsOut.srcStride = 0;
        copyParamsOut.dstStride = 0;
        copyParamsOut.rsv = 0;
    }

    __aicore__ inline void ComputeTailDiv(uint64_t tail, uint64_t baseOffset, float mulValue)
    {
        uint64_t tailLoop = tail / ubTailNum;
        uint64_t offset = 0;
        pipe_barrier(PIPE_ALL);
        for (uint64_t loop = 0; loop < tailLoop; loop++) {
            pipe_barrier(PIPE_ALL);
            offset = loop * ubTailNum;
            DataCopy(srcLocal, srcGm[baseOffset + offset], ubTailNum);
            pipe_barrier(PIPE_ALL);
            Muls(srcLocal, srcLocal, mulValue, ubTailNum);
            pipe_barrier(PIPE_ALL);
            DataCopy(outGm[baseOffset + offset], srcLocal, ubTailNum);
        }

        offset = tailLoop * ubTailNum;
        uint64_t tailLast = tail - offset;
        pipe_barrier(PIPE_ALL);
        if (tailLast != 0) {
            CopyParamasInit(tailLast);
            pipe_barrier(PIPE_ALL);
            DataCopy(srcLocal, srcGm[baseOffset + offset], AlignUp(tailLast, dataEachBlock));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);
            Muls(srcLocal, srcLocal, mulValue, AlignUp(tailLast, dataEachBlock));
            pipe_barrier(PIPE_ALL);
            DataCopyPad(outGm[baseOffset + offset], srcLocal, copyParamsOut);
        }
    }

    __aicore__ inline void ComputeEachTask(int32_t taskId, uint64_t taskLine)
    {
        LocalTensor<DTYPE_COUNT>countLocal = inQueueCount.Get<DTYPE_COUNT>();
        srcLocal = inQueueSrc.Get<DTYPE_SRC>();

        auto count_offset = countBaseOffset + taskEachLine * taskId;
        DataCopy(countLocal, countGm[count_offset], AlignUp(taskLine, countEachBlock));

        CopyParamasInit(tail);
        pipe_barrier(PIPE_ALL);
        for (uint64_t idx = 0; idx < taskLine; idx++) {
            DTYPE_COUNT countValue = countLocal.GetValue(idx);
            pipe_barrier(PIPE_ALL);
            if (countValue != 0) {
                float mulValue = 1 / countValue;
                ComputeTailDiv(tail, count_offset * tail + idx * tail, mulValue);
            }
        }
    }

private:
    TPipe* pipe;
    TBuf<TPosition::VECCALC> inQueueSrc, inQueueCount;
    GlobalTensor<DTYPE_COUNT> countGm;
    GlobalTensor<DTYPE_SRC> srcGm;
    GlobalTensor<DTYPE_OUT> outGm;
    LocalTensor<DTYPE_SRC> srcLocal;

    DataCopyExtParams copyParamsOut;
    uint64_t curBlockIdx;
    uint64_t usedCoreNum, bigCoreNum;
    uint64_t tail;
    uint64_t taskNum;
    uint64_t taskEachLine, taskLastLine;
    uint64_t countEachBlock, dataEachBlock;
    uint64_t srcNum, countNum, outNum;
    uint64_t ubCountNum;
    uint64_t countBaseOffset;
    uint64_t ubTailNum;

    event_t eventIdMte2ToV_0;
};

extern "C" __global__ __aicore__ void scatter_mean_div(GM_ADDR src, GM_ADDR count, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    KernelScatterMeanDiv op;
    op.Init(src, count, out, &tiling_data, &pipe);
    op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void scatter_mean_div_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* src, uint8_t* count, uint8_t* out,
    uint8_t* workspace, uint8_t* tiling)
{
    scatter_mean_div<<<blockDim, l2ctrl, stream>>>(var, indices, updates, out, argmax, workspace, tiling);
}
#endif