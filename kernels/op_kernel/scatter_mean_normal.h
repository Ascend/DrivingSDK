/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef _SCATTER_MEAN_NORAML_H_
#define _SCATTER_MEAN_NORAML_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MAX_MASK = 64;

class KernelScatterMeanFix {
public:
    __aicore__ inline KernelScatterMeanFix() {}
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR indices, GM_ADDR var, GM_ADDR out, GM_ADDR count, ScatterMeanTilingData *tiling_data, TPipe* tmpPipe)
    {
        pipe = tmpPipe;
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        TilingDataInit(tiling_data);

        varGm.SetGlobalBuffer((__gm__ DTYPE_VAR*)var, outNum);
        indicesGm.SetGlobalBuffer((__gm__ DTYPE_INDICES*)indices, indicesNum);
        srcGm.SetGlobalBuffer((__gm__ DTYPE_SRC*)src, srcNum);
        outGm.SetGlobalBuffer((__gm__ DTYPE_OUT*)out, outNum);
        countGm.SetGlobalBuffer((__gm__ DTYPE_OUT*)count, outNum / tail);

        eventIdMte3ToMte2_0 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_MTE2>());
        eventIdMte2ToMte3_0 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_MTE3>());
        pipe->InitBuffer(inQueueIndices, AlignUp(ubIndicesNum, indicesEachBlock) * sizeof(DTYPE_INDICES));
        pipe->InitBuffer(inQueueSrc, AlignUp(ubTailNum, indicesEachBlock) * sizeof(DTYPE_SRC));
        pipe->InitBuffer(onesTensorBuff, dataEachBlock * sizeof(DTYPE_COUNT));
    }

    __aicore__ inline void TilingDataInit(ScatterMeanTilingData *tiling_data)
    {
        curBlockIdx = GetBlockIdx();
        usedCoreNum = tiling_data->usedCoreNum;
        tail = tiling_data->tail;
        body = tiling_data->body;
        taskNum = tiling_data->taskNum;
        taskEachLine = tiling_data->taskEachLine;
        taskLastLine = tiling_data->taskLastLine;
        bigCoreNum = tiling_data->bigCoreNum;
        outDimSize = tiling_data->outDimSize;
        dimSize = tiling_data->dimSize;
        srcNum = tiling_data->srcNum;
        indicesNum = tiling_data->indicesNum;
        outNum = tiling_data->outNum;
        ubIndicesNum = tiling_data->ubIndicesNum;
        ubTailNum = tiling_data->ubTailNum;

        uint64_t coreDataLine = tiling_data->bacthSmallCore;
        if (curBlockIdx < bigCoreNum) {
            coreDataLine = coreDataLine + 1;
            indicesBaseOffset = curBlockIdx * coreDataLine;
        } else {
            taskNum = tiling_data->taskNumLast;
            taskEachLine = tiling_data->taskEachLineLast;
            taskLastLine = tiling_data->taskLastLineLast;
            indicesBaseOffset = bigCoreNum * (coreDataLine + 1) + (curBlockIdx - bigCoreNum) * coreDataLine;
        }

        indicesEachBlock = BLOCK_SIZE / sizeof(DTYPE_INDICES);
        dataEachBlock = BLOCK_SIZE / sizeof(DTYPE_SRC);

        tailLoop = tail / ubTailNum;
        tailLast = tail - tailLoop * ubTailNum;

        copyParamsCount.blockCount = 1;
        copyParamsCount.blockLen = static_cast<uint32_t>(1 * sizeof(float));
        copyParamsCount.srcStride = 0;
        copyParamsCount.dstStride = 0;
        copyParamsCount.rsv = 0;
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

    __aicore__ inline void ComputeTailAdd(uint64_t idxTure, uint64_t dataInIndices, uint64_t src_offset)
    {
        uint64_t offset = 0;

        uint64_t srcLineEachHead = dimSize * body;
        auto idx1 = idxTure / srcLineEachHead;
        auto idx2 = (idxTure - idx1 * srcLineEachHead) / body;
        auto idx3 = idxTure - idx1 * srcLineEachHead - idx2 * body;
        uint64_t outLineOffset = idx3 + dataInIndices * body + idx1 * (outDimSize * body);
        pipe_barrier(PIPE_ALL);
        for (uint64_t loop = 0; loop < tailLoop; loop++) {
            pipe_barrier(PIPE_ALL);
            offset = loop * ubTailNum;

            DataCopy(srcLocal, srcGm[src_offset + offset], ubTailNum);
            SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3_0);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3_0);
            DataCopy(outGm[outLineOffset * tail + offset], srcLocal, ubTailNum);
        }

        offset = tailLoop * ubTailNum;
        if (tailLast != 0) {
            pipe_barrier(PIPE_ALL);
            CopyParamasInit(tailLast);
            DataCopy(srcLocal, srcGm[src_offset + offset], AlignUp(tailLast, dataEachBlock));
            SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3_0);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3_0);
            DataCopyPad(outGm[outLineOffset * tail + offset], srcLocal, copyParamsOut);
        }
        DataCopyPad<DTYPE_SRC>(countGm[outLineOffset], onesTensor, copyParamsCount);
    }

    __aicore__ inline void ComputeEachTask(int32_t taskId, uint64_t taskLine)
    {
        LocalTensor<DTYPE_INDICES>indicesLocal = inQueueIndices.Get<DTYPE_INDICES>();
        onesTensor = onesTensorBuff.Get<DTYPE_COUNT>();
        srcLocal = inQueueSrc.Get<DTYPE_SRC>();
        Duplicate(onesTensor, (float)1, dataEachBlock); // one Block

        uint64_t indices_offset = indicesBaseOffset + taskEachLine * taskId;
        DataCopy(indicesLocal, indicesGm[indices_offset], AlignUp(taskLine, indicesEachBlock));
        pipe_barrier(PIPE_ALL);
        for (uint64_t idx = 0; idx < taskLine; idx++) {
            DTYPE_INDICES dataInIndices = indicesLocal.GetValue(idx);
            auto idxTure = indices_offset + idx;
            auto src_offset = indices_offset * tail + idx * tail;

            SetAtomicAdd<DTYPE_SRC>();
            ComputeTailAdd(idxTure, dataInIndices, src_offset);
            SetAtomicNone();
        }
    }

private:
    TPipe* pipe;
    TBuf<TPosition::VECCALC> inQueueIndices, inQueueSrc;
    TBuf<TPosition::VECCALC> onesTensorBuff;

    GlobalTensor<DTYPE_VAR> varGm;
    GlobalTensor<DTYPE_INDICES> indicesGm;
    GlobalTensor<DTYPE_SRC> srcGm;
    GlobalTensor<DTYPE_OUT> outGm;
    GlobalTensor<DTYPE_COUNT> countGm;

    LocalTensor<DTYPE_SRC> srcLocal;
    LocalTensor<float>onesTensor;

    DataCopyExtParams copyParamsOut;
    DataCopyExtParams copyParamsCount;
    uint64_t curBlockIdx;
    bool isOneDeal;
    uint64_t usedCoreNum, bigCoreNum;
    uint64_t tail, body;
    uint64_t taskNum;
    uint64_t taskEachLine, taskLastLine;
    uint64_t indicesEachBlock, dataEachBlock;
    uint64_t srcNum, indicesNum, outNum;
    uint64_t ubIndicesNum;
    uint64_t outDimSize, dimSize;
    uint64_t ubTailNum;
    uint64_t indicesBaseOffset;

    int64_t tailLoop;
    uint64_t tailLast;

    event_t eventIdMte3ToMte2_0, eventIdMte2ToMte3_0;
};
#endif