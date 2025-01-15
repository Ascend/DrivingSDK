/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#ifndef _SCATTER_ADD_NOTAIL_BIGHEAD_H_
#define _SCATTER_ADD_NOTAIL_BIGHEAD_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"


using namespace AscendC;
class ScatterAddNoTailBigHead {
public:
    __aicore__ inline ScatterAddNoTailBigHead() {}
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
        pipe->InitBuffer(inQueueSrc, AlignUp(indicesDealNum, dataEachBlock) * sizeof(DTYPE_SRC));
        pipe->InitBuffer(outQueueOut, AlignUp(taskEachDataNum, dataEachBlock) * sizeof(DTYPE_OUT));
    }

    __aicore__ inline void TilingDataInit(ScatterAddTilingData *tiling_data)
    {
        uint64_t curBlockIdx = GetBlockIdx();
        uint64_t head = tiling_data->head;
        uint64_t lineSmallCore = tiling_data->lineSmallCore;
        uint64_t lineBigCore = tiling_data->lineBigCore;
        taskNum = tiling_data->taskNum;
        taskEachDataNum = tiling_data->taskEachLine;
        taskLastDataNum = tiling_data->taskLastLine;
        uint64_t bigCoreNum = tiling_data->bigCoreNum;
        srcNum = tiling_data->srcNum;
        indicesNum = tiling_data->indicesNum;
        outNum = tiling_data->outNum;
        indicesDealNum = tiling_data->indicesDealNum;
        headEachNum = tiling_data->headNumEachTask;

        if (curBlockIdx < bigCoreNum) {
            batchNum = lineBigCore;
        } else {
            batchNum = lineSmallCore;
        }

        if (curBlockIdx < bigCoreNum) {
            baseHeadId = lineBigCore * curBlockIdx;
            taskNum = tiling_data->taskNum;
            taskEachDataNum = tiling_data->taskEachLine;
            taskLastDataNum = tiling_data->taskLastLine;
            headLastNum = tiling_data->headNumBigLast;
        } else {
            baseHeadId = lineBigCore * bigCoreNum + (curBlockIdx - bigCoreNum) * lineSmallCore;
            taskNum = tiling_data->taskNumLast;
            taskEachDataNum = tiling_data->taskEachLineLast;
            taskLastDataNum = tiling_data->taskLastLineLast;
            headLastNum = tiling_data->headNumSmallLast;
        }

        indicesHeadNum = indicesNum / head;
        outHeadNum = outNum / head;

        indicesEachBlock = BLOCK_SIZE / sizeof(DTYPE_INDICES);
        dataEachBlock = BLOCK_SIZE / sizeof(DTYPE_SRC);

        indicesLoop = indicesHeadNum / indicesDealNum;
        indicesLastNum = indicesHeadNum % indicesDealNum;
    }

    __aicore__ inline void Process()
    {
        for (uint64_t i = 0; i < taskNum - 1; i++) {
            ComputeEachTaskNoTail(i, headEachNum, taskEachDataNum);
        }
        if (taskLastDataNum != 0) {
            ComputeEachTaskNoTail(taskNum - 1, headLastNum, taskLastDataNum);
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

    __aicore__ inline void ComputeEachTaskNoTail(uint64_t taskId, uint64_t headNum, uint64_t dataNum)
    {
        indicesLocal = inQueueIndices.Get<DTYPE_INDICES>();
        srcLocal = inQueueSrc.Get<DTYPE_SRC>();
        outLocalTemp = outQueueOut.Get<DTYPE_OUT>();

        uint64_t headThisId = baseHeadId + taskId * headEachNum;
        uint64_t outBaseOffset = headThisId * outHeadNum;
        pipe_barrier(PIPE_ALL);

        DataCopy(outLocalTemp, outGm[outBaseOffset], AlignUp(dataNum, dataEachBlock));

        uint64_t indicesBaseOffset = headThisId * indicesHeadNum;
        auto indicesNumAlign = AlignUp(headNum * indicesHeadNum, dataEachBlock);
        DataCopy(indicesLocal, indicesGm[indicesBaseOffset], indicesNumAlign);
        DataCopy(srcLocal, srcGm[indicesBaseOffset], indicesNumAlign);

        uint64_t indicesThisOffset = 0;
        uint64_t thisOutOffset = 0;

        for (uint64_t h = 0; h < headNum; h++) {
            for (uint64_t idx = 0; idx < indicesHeadNum; idx++) {
                auto srcOffset = indicesThisOffset + idx;
                DTYPE_INDICES dataInIndices = indicesLocal.GetValue(srcOffset);
                int64_t offsetInOut = thisOutOffset + dataInIndices;
                outLocalTemp.SetValue(offsetInOut, outLocalTemp.GetValue(offsetInOut) + srcLocal.GetValue(srcOffset));
            }
            thisOutOffset = thisOutOffset + outHeadNum;
            indicesThisOffset = indicesThisOffset + indicesHeadNum;
        }

        CopyParamasInit(dataNum);
        DataCopyPad(outGm[outBaseOffset], outLocalTemp, copyParamsOut);
    }

private:
    TPipe* pipe;
    TBuf<TPosition::VECCALC> inQueueIndices, inQueueSrc;
    TBuf<TPosition::VECCALC> outQueueOut;

    GlobalTensor<DTYPE_VAR> varGm;
    GlobalTensor<DTYPE_INDICES> indicesGm;
    GlobalTensor<DTYPE_SRC> srcGm;
    GlobalTensor<DTYPE_OUT> outGm;

    LocalTensor<DTYPE_SRC> srcLocal;
    LocalTensor<DTYPE_OUT> outLocalTemp;
    LocalTensor<DTYPE_INDICES> indicesLocal;

    DataCopyExtParams copyParamsOut;
    uint32_t taskNum;
    uint64_t taskEachDataNum;
    uint64_t taskLastDataNum;
    uint64_t indicesEachBlock;
    uint64_t dataEachBlock;
    uint32_t indicesLoop;
    uint32_t indicesLastNum;
    uint32_t batchNum;
    uint32_t baseHeadId;
    uint64_t srcNum, indicesNum, outNum;
    uint64_t indicesHeadNum, outHeadNum;
    uint64_t indicesDealNum;
    uint64_t headEachNum;
    uint64_t headLastNum;
};
// }
#endif
