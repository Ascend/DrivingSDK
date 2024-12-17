/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef _SCATTER_MEAN_NOTAIL_BIGHEAD_H_
#define _SCATTER_MEAN_NOTAIL_BIGHEAD_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"


using namespace AscendC;
class ScatterMeanNoTailBigHead {
public:
    __aicore__ inline ScatterMeanNoTailBigHead() {}
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR indices, GM_ADDR var, GM_ADDR out, GM_ADDR count, ScatterMeanTilingData *tiling_data, TPipe* tmpPipe)
    {
        pipe = tmpPipe;
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        TilingDataInit(tiling_data);

        varGm.SetGlobalBuffer((__gm__ DTYPE_VAR*)var, outNum);
        indicesGm.SetGlobalBuffer((__gm__ DTYPE_INDICES*)indices, indicesNum);
        srcGm.SetGlobalBuffer((__gm__ DTYPE_SRC*)src, srcNum);
        outGm.SetGlobalBuffer((__gm__ DTYPE_OUT*)out, outNum);
        countGm.SetGlobalBuffer((__gm__ DTYPE_OUT*)count, outNum);

        pipe->InitBuffer(inQueueIndices, AlignUp(ubIndicesNum, indicesEachBlock) * sizeof(DTYPE_INDICES));
        pipe->InitBuffer(inQueueSrc, AlignUp(ubIndicesNum, dataEachBlock) * sizeof(DTYPE_SRC));
        pipe->InitBuffer(countBuff, AlignUp(taskEachDataNum, dataEachBlock) * sizeof(DTYPE_SRC));
        pipe->InitBuffer(outQueueOut, AlignUp(taskEachDataNum, dataEachBlock) * sizeof(DTYPE_OUT));
    }

    __aicore__ inline void TilingDataInit(ScatterMeanTilingData *tiling_data)
    {
        curBlockIdx = GetBlockIdx();
        head = tiling_data->head;
        bacthSmallCore = tiling_data->bacthSmallCore;
        bacthBigCore = tiling_data->bacthBigCore;
        taskNum = tiling_data->taskNum;
        taskEachDataNum = tiling_data->taskEachLine;
        taskLastDataNum = tiling_data->taskLastLine;
        bigCoreNum = tiling_data->bigCoreNum;
        outLineEachBacth = tiling_data->outLineEachBacth;
        indicesLoop = tiling_data->indicesLoop;
        indicesLastNum = tiling_data->indicesLastNum;
        srcNum = tiling_data->srcNum;
        indicesNum = tiling_data->indicesNum;
        outNum = tiling_data->outNum;
        ubIndicesNum = tiling_data->ubIndicesNum;
        headEachNum = tiling_data->headNumEachTask;

        if (curBlockIdx < bigCoreNum) {
            batchNum = bacthBigCore;
        } else {
            batchNum = bacthSmallCore;
        }

        if (curBlockIdx < bigCoreNum) {
            baseHeadId = bacthBigCore * curBlockIdx;
            taskNum = tiling_data->taskNum;
            taskEachDataNum = tiling_data->taskEachLine;
            taskLastDataNum = tiling_data->taskLastLine;
            headLastNum = tiling_data->headNumBigLast;
        } else {
            baseHeadId = bacthBigCore * bigCoreNum + (curBlockIdx - bigCoreNum) * bacthSmallCore;
            taskNum = tiling_data->taskNumLast;
            taskEachDataNum = tiling_data->taskEachLineLast;
            taskLastDataNum = tiling_data->taskLastLineLast;
            headLastNum = tiling_data->headNumSmallLast;
        }

        indicesHeadNum = indicesNum / head;
        outHeadNum = outNum / head;

        indicesEachBlock = BLOCK_SIZE / sizeof(DTYPE_INDICES);
        dataEachBlock = BLOCK_SIZE / sizeof(DTYPE_SRC);

        ubIndicesNumAlign = AlignUp(ubIndicesNum, dataEachBlock);
        indicesLastNumAlign = AlignUp(indicesLastNum, dataEachBlock);
    }

    __aicore__ inline void Process()
    {
        if (headEachNum == 0) {
            for (uint64_t i = 0; i < batchNum; i++) {
                ComputeSingle(i);
            }
        } else {
            ComputeMulti();
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

    __aicore__ inline void ComputeEachHeadNoTail(uint64_t headId, uint64_t dataNum, uint64_t outBaseOffset, int64_t indicesStart)
    {
        uint64_t basdIndicesOffset = headId * indicesHeadNum;
        uint64_t thisOutOffset = headId * outHeadNum;
        for (uint64_t loop = 0; loop < indicesLoop; loop++) {
            uint64_t indicesOffset = basdIndicesOffset + loop * ubIndicesNum;
            pipe_barrier(PIPE_ALL);
            DataCopy(indicesLocal, indicesGm[indicesOffset], ubIndicesNumAlign);
            DataCopy(srcLocal, srcGm[indicesOffset], ubIndicesNumAlign);

            for (uint64_t idx = 0; idx < ubIndicesNum; idx++) {
                DTYPE_INDICES dataInIndices = indicesLocal.GetValue(idx);
                // if this indices should be processed in this task
                if (dataInIndices >= indicesStart && dataInIndices < indicesStart + dataNum) {
                    int64_t offsetInOut = thisOutOffset + dataInIndices - outBaseOffset;
                    outLocalTemp.SetValue(offsetInOut, outLocalTemp.GetValue(offsetInOut) + srcLocal.GetValue(idx));
                    countTemp.SetValue(offsetInOut, countTemp.GetValue(offsetInOut) + 1);
                }
            }
        }
        if (indicesLastNum != 0) {
            uint64_t indicesOffset = basdIndicesOffset + indicesLoop * ubIndicesNum;
            DataCopy(indicesLocal, indicesGm[indicesOffset], indicesLastNumAlign);
            DataCopy(srcLocal, srcGm[indicesOffset], indicesLastNumAlign);

            for (uint64_t idx = 0; idx < indicesLastNum; idx++) {
                DTYPE_INDICES dataInIndices = indicesLocal.GetValue(idx);
                if (dataInIndices >= indicesStart && dataInIndices < indicesStart + dataNum) {
                    int64_t offsetInOut = thisOutOffset + dataInIndices - outBaseOffset;
                    outLocalTemp.SetValue(offsetInOut, outLocalTemp.GetValue(offsetInOut) + srcLocal.GetValue(idx));
                    countTemp.SetValue(offsetInOut, countTemp.GetValue(offsetInOut) + 1);
                }
            }
        }
    }

    __aicore__ inline void ComputeEachTaskNoTailFix(uint64_t taskId, uint64_t headNum, uint64_t dataNum)
    {
        indicesLocal = inQueueIndices.Get<DTYPE_INDICES>();
        srcLocal = inQueueSrc.Get<DTYPE_SRC>();
        countTemp = countBuff.Get<DTYPE_OUT>();
        outLocalTemp = outQueueOut.Get<DTYPE_OUT>();

        uint64_t headThisId = baseHeadId + taskId * headEachNum;
        uint64_t outBaseOffset = headThisId * outHeadNum;
        pipe_barrier(PIPE_ALL);
        auto dataNumAlign = AlignUp(dataNum, dataEachBlock);
        DataCopy(outLocalTemp, outGm[outBaseOffset], dataNumAlign);
        Duplicate(countTemp, (float)0, dataNumAlign);

        uint64_t indicesBaseOffset = headThisId * indicesHeadNum;
        auto indicesNumAlign = AlignUp(headNum * indicesHeadNum, dataEachBlock);
        DataCopy(indicesLocal, indicesGm[indicesBaseOffset], indicesNumAlign);
        DataCopy(srcLocal, srcGm[indicesBaseOffset], indicesNumAlign);

        uint64_t indicesThisOffset = 0;
        uint64_t thisOutOffset = 0;

        for (uint64_t h = 0; h < headNum; h++) {
            for (uint64_t idx = 0; idx < indicesHeadNum; idx++) {
                auto indicesThisOffsetaaa = indicesThisOffset + idx;
                DTYPE_INDICES dataInIndices = indicesLocal.GetValue(indicesThisOffsetaaa);
                int64_t offsetInOut = thisOutOffset + dataInIndices;
                outLocalTemp.SetValue(offsetInOut, outLocalTemp.GetValue(offsetInOut) + srcLocal.GetValue(indicesThisOffsetaaa));
                countTemp.SetValue(offsetInOut, countTemp.GetValue(offsetInOut) + 1);
            }
            thisOutOffset = thisOutOffset + outHeadNum;
            indicesThisOffset = indicesThisOffset + indicesHeadNum;
        }

        CopyParamasInit(dataNum);
        DataCopyPad(countGm[outBaseOffset], countTemp, copyParamsOut);
        DataCopyPad(outGm[outBaseOffset], outLocalTemp, copyParamsOut);
    }

    __aicore__ inline void ComputeEachTaskNoTailOri(uint64_t taskId, uint64_t batchId, uint64_t taskLine)
    {
        indicesLocal = inQueueIndices.Get<DTYPE_INDICES>();
        srcLocal = inQueueSrc.Get<DTYPE_SRC>();
        outLocalTemp = outQueueOut.Get<DTYPE_OUT>();
        countTemp = countBuff.Get<DTYPE_SRC>();

        uint64_t headId = baseHeadId + batchId;
        uint64_t outBaseOffset = headId * outHeadNum + taskId * taskEachDataNum;
        pipe_barrier(PIPE_ALL);
        DataCopy(outLocalTemp, outGm[outBaseOffset], AlignUp(taskLine, indicesEachBlock));
        Duplicate(countTemp, (float)0, taskLine);

        int64_t indicesStart = taskId * taskEachDataNum;
        ComputeEachHeadNoTail(headId, taskLine, outBaseOffset, indicesStart);

        CopyParamasInit(taskLine);
        pipe_barrier(PIPE_ALL);
        DataCopyPad(countGm[outBaseOffset], countTemp, copyParamsOut);
        DataCopyPad(outGm[outBaseOffset], outLocalTemp, copyParamsOut);
    }

    __aicore__ inline void ComputeMulti()
    {
        for (uint64_t i = 0; i < taskNum - 1; i++) {
            ComputeEachTaskNoTailFix(i, headEachNum, taskEachDataNum);
        }
        if (taskLastDataNum != 0) {
            ComputeEachTaskNoTailFix(taskNum - 1, headLastNum, taskLastDataNum);
        }
    }

    __aicore__ inline void ComputeSingle(uint64_t batchId)
    {
        for (uint64_t i = 0; i < taskNum - 1; i++) {
            ComputeEachTaskNoTailOri(i, batchId, taskEachDataNum);
        }
        if (taskLastDataNum != 0) {
            ComputeEachTaskNoTailOri(taskNum - 1, batchId, taskLastDataNum);
        }
    }

private:
    TPipe* pipe;
    TBuf<TPosition::VECCALC> inQueueIndices, inQueueSrc;
    TBuf<TPosition::VECCALC> outQueueOut;
    TBuf<TPosition::VECCALC> countBuff;

    GlobalTensor<DTYPE_VAR> varGm;
    GlobalTensor<DTYPE_INDICES> indicesGm;
    GlobalTensor<DTYPE_SRC> srcGm;
    GlobalTensor<DTYPE_OUT> outGm;
    GlobalTensor<DTYPE_COUNT> countGm;

    LocalTensor<float> countTemp;
    LocalTensor<DTYPE_SRC> srcLocal;
    LocalTensor<DTYPE_OUT> outLocalTemp;
    LocalTensor<DTYPE_INDICES> indicesLocal;

    DataCopyExtParams copyParamsOut;
    uint64_t curBlockIdx;
    uint64_t bigCoreNum;
    uint64_t head;
    uint64_t bacthSmallCore, bacthBigCore, taskNum;
    uint64_t taskEachDataNum, taskLastDataNum, outLineEachBacth, taskEachLineLast;
    uint64_t indicesEachBlock, dataEachBlock;
    uint64_t indicesLoop, indicesLastNum;
    uint64_t batchNum;
    uint64_t baseHeadId;
    uint64_t srcNum, indicesNum, outNum;
    uint64_t indicesHeadNum, outHeadNum;
    uint64_t ubIndicesNum;
    uint64_t ubIndicesNumAlign, indicesLastNumAlign;
    uint64_t headEachNum, headLastNum;
};
// }
#endif