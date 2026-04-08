/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#ifndef _SCATTER_ADD_NORAML_H_
#define _SCATTER_ADD_NORAML_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t MAX_MASK = 64;

class KernelScatterAddLine {
public:
    __aicore__ inline KernelScatterAddLine() {}
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR indices, GM_ADDR var, GM_ADDR out, ScatterAddTilingData *tiling_data, TPipe* tmpPipe)
    {
        pipe = tmpPipe;
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        TilingDataInit(tiling_data);

        varGm.SetGlobalBuffer((__gm__ DTYPE_VAR*)var, outNum);
        indicesGm.SetGlobalBuffer((__gm__ DTYPE_INDICES*)indices, indicesNum);
        srcGm.SetGlobalBuffer((__gm__ DTYPE_SRC*)src, srcNum);
        outGm.SetGlobalBuffer((__gm__ DTYPE_OUT*)out, outNum);

        pipe->InitBuffer(inQueueIndices, AlignUp(indicesDealNum, indicesEachBlock) * sizeof(DTYPE_INDICES));
        pipe->InitBuffer(inQueueSrc, ubSrcNum * sizeof(DTYPE_SRC));
    }

    __aicore__ inline void TilingDataInit(ScatterAddTilingData *tiling_data)
    {
        curBlockIdx = GetBlockIdx();
        tail = tiling_data->tail;
        taskNum = tiling_data->taskNum;
        taskEachLine = tiling_data->taskEachLine;
        taskLastLine = tiling_data->taskLastLine;
        bigCoreNum = tiling_data->bigCoreNum;
        outDimSize = tiling_data->outDimSize;
        dimSize = tiling_data->dimSize;
        srcNum = tiling_data->srcNum;
        indicesNum = tiling_data->indicesNum;
        outNum = tiling_data->outNum;
        indicesDealNum = tiling_data->indicesDealNum;
        ubTailNum = tiling_data->ubTailNum;
        tilingMode = tiling_data->tilingMode;

        dbTimes = tiling_data->dbTimes;

        ubSrcNum = tiling_data->ubSrcNum;

        uint64_t coreDataLine = tiling_data->lineSmallCore;
        if (curBlockIdx < bigCoreNum) {
            coreDataLine = coreDataLine + 1;
            indicesBaseOffset = curBlockIdx * coreDataLine;
        } else {
            if (taskLastLine == 1) {
                taskNum = taskNum - 1;
                taskLastLine = taskEachLine;
            } else {
                taskLastLine = taskLastLine - 1;
            }
            indicesBaseOffset = bigCoreNum * (coreDataLine + 1) + (curBlockIdx - bigCoreNum) * coreDataLine;
        }

        indicesEachBlock = BLOCK_SIZE / sizeof(DTYPE_INDICES);
        dataEachBlock = BLOCK_SIZE / sizeof(DTYPE_SRC);

        tailLoop = tail / ubTailNum;
        tailLast = tail - tailLoop * ubTailNum;

        copyParamsOut.blockCount = 1;
        copyParamsOut.blockLen = static_cast<uint32_t>(tailLast * sizeof(float));
        copyParamsOut.srcStride = 0;
        copyParamsOut.dstStride = 0;
        copyParamsOut.rsv = 0;

        countDB = 0;
        eventId = EVENT_ID0;
    }
    __aicore__ inline void Process()
    {
        if (tilingMode == 0) {
            copyParamsOut.blockLen = static_cast<uint32_t>(tailLast * sizeof(float));
            for (int32_t i = 0; i < taskNum - 1; i++) {
                ComputeEachTask(i, taskEachLine);
            }
            if (taskLastLine != 0) {
                ComputeEachTask(taskNum - 1, taskLastLine);
            }
        } else {
            copyParamsOut.blockLen = static_cast<uint32_t>(tail * sizeof(float));
            for (int32_t i = 0; i < taskNum - 1; i++) {
                ComputeSmallTail(i, taskEachLine);
            }
            if (taskLastLine != 0) {
                ComputeSmallTail(taskNum - 1, taskLastLine);
            }
        }
    }

private:
    __aicore__ inline int64_t getEventIdforDoublebuffer()
    {
        uint64_t localOffset;
        eventId = countDB % dbTimes; // int64_t
        localOffset = ubTailNum * eventId;
        countDB = countDB + 1;
        return localOffset;
    }

    __aicore__ inline void CopyParamasInit(const uint32_t blockCount, const uint32_t tail)
    {
        copyParamsIn.blockCount = static_cast<uint16_t>(blockCount);
        copyParamsIn.blockLen = static_cast<uint32_t>(tail * sizeof(float));
        copyParamsIn.srcStride = 0;
        copyParamsIn.dstStride = 0;
        copyParamsIn.rsv = 0;
    }

    __aicore__ inline void ComputeTailAdd(uint64_t idxTure, uint64_t dataInIndices, uint64_t src_offset)
    {
        auto headId = idxTure / dimSize;
        auto outLineOffset = (dataInIndices + headId * outDimSize) * tail;

        uint64_t offset = 0;
        for (uint64_t loop = 0; loop < tailLoop; loop++) {
            offset = loop * ubTailNum;
            uint64_t local_offset = getEventIdforDoublebuffer();
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            DataCopy(srcLocal[local_offset], srcGm[src_offset + offset], ubTailNum);
            SetFlag<HardEvent::MTE2_MTE3>(eventId);
            WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            DataCopy(outGm[outLineOffset + offset], srcLocal[local_offset], ubTailNum);
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
        }

        offset = tailLoop * ubTailNum;
        if (tailLast != 0) {
            uint64_t local_offset = getEventIdforDoublebuffer();
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
            DataCopy(srcLocal[local_offset], srcGm[src_offset + offset], AlignUp(tailLast, dataEachBlock));
            SetFlag<HardEvent::MTE2_MTE3>(eventId);
            WaitFlag<HardEvent::MTE2_MTE3>(eventId);
            DataCopyPad(outGm[outLineOffset + offset], srcLocal[local_offset], copyParamsOut);
            SetFlag<HardEvent::MTE3_MTE2>(eventId);
        }
    }

    __aicore__ inline void SetFlagMET3_2()
    {
        for (uint32_t i = 0; i < dbTimes; i++) {
            SetFlag<HardEvent::MTE3_MTE2>(i);
        }
    }

    __aicore__ inline void WaitFlagMET3_2()
    {
        for (uint32_t i = 0; i < dbTimes; i++) {
            WaitFlag<HardEvent::MTE3_MTE2>(i);
        }
    }

    __aicore__ inline void ComputeEachTask(int32_t taskId, uint64_t taskLine)
    {
        LocalTensor<DTYPE_INDICES>indicesLocal = inQueueIndices.Get<DTYPE_INDICES>();
        srcLocal = inQueueSrc.Get<DTYPE_SRC>();

        auto indices_offset = indicesBaseOffset + taskEachLine * taskId;
        DataCopy(indicesLocal, indicesGm[indices_offset], AlignUp(taskLine, indicesEachBlock));
        SetFlagMET3_2();

        for (uint64_t idx = 0; idx < taskLine; idx++) {
            DTYPE_INDICES dataInIndices = indicesLocal.GetValue(idx);
            auto idxTure = indices_offset + idx;
            auto src_offset = idxTure * tail;

            SetAtomicAdd<DTYPE_SRC>();
            ComputeTailAdd(idxTure, dataInIndices, src_offset);
            SetAtomicNone();
        }
        WaitFlagMET3_2();
    }

    __aicore__ inline void ComputeSmallTail(int32_t taskId, uint64_t taskLine)
    {
        LocalTensor<DTYPE_INDICES>indicesLocal = inQueueIndices.Get<DTYPE_INDICES>();
        srcLocal = inQueueSrc.Get<DTYPE_SRC>();

        auto indicesOffset = indicesBaseOffset + taskEachLine * taskId;
        auto srcOffset = indicesOffset * tail;
        pipe_barrier(PIPE_ALL);
        DataCopy(indicesLocal, indicesGm[indicesOffset], AlignUp(taskLine, 8));
        CopyParamasInit(taskLine, tail);
        DataCopyPadExtParams<float> queryCopyInPadParams{false, 0, 0, 0};
        DataCopyPad(srcLocal, srcGm[srcOffset], copyParamsIn, queryCopyInPadParams);

        SetAtomicAdd<DTYPE_SRC>();
        for (uint64_t idx = 0; idx < taskLine; idx++) {
            DTYPE_INDICES dataInIndices = indicesLocal.GetValue(idx);
            auto idxTure = indicesOffset + idx;
            auto srcLocalOffset = idx * tail;

            auto headId = idxTure / dimSize;
            auto outLineOffset = (dataInIndices + headId * outDimSize) * tail;

            DataCopyPad(outGm[outLineOffset], srcLocal[idx * AlignUp(tail, 8)], copyParamsOut);
        }
        SetAtomicNone();
    }

private:
    TPipe* pipe;
    TBuf<TPosition::VECCALC> inQueueIndices;
    TBuf<TPosition::VECCALC> inQueueSrc;

    GlobalTensor<DTYPE_VAR> varGm;
    GlobalTensor<DTYPE_INDICES> indicesGm;
    GlobalTensor<DTYPE_SRC> srcGm;
    GlobalTensor<DTYPE_OUT> outGm;

    LocalTensor<DTYPE_SRC> srcLocal;

    DataCopyExtParams copyParamsOut;
    DataCopyExtParams copyParamsIn;

    uint32_t curBlockIdx;
    uint64_t bigCoreNum;
    uint64_t tail;
    uint32_t taskNum;
    uint64_t taskEachLine, taskLastLine;
    uint64_t indicesEachBlock, dataEachBlock;
    uint64_t srcNum, indicesNum, outNum;
    uint64_t indicesDealNum;
    uint64_t outDimSize, dimSize;
    uint64_t ubTailNum;
    uint64_t indicesBaseOffset;
    uint32_t tilingMode;

    uint32_t countDB;
    uint32_t dbTimes;
    uint32_t ubSrcNum;
    uint32_t tailLoop;
    uint32_t tailLast;
    uint64_t eventId;
};
#endif