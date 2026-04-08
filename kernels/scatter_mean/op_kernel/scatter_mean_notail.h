/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef _SCATTER_MEAN_NOTAIL_H_
#define _SCATTER_MEAN_NOTAIL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;
class ScatterMeanNoTail {
public:
    __aicore__ inline ScatterMeanNoTail() {}
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
        pipe->InitBuffer(outSetNote, AlignUp(taskEachDataNum, MAX_MASK) * sizeof(DTYPE_SRC));
        pipe->InitBuffer(outQueueOut, AlignUp(taskEachDataNum, MAX_MASK) * sizeof(DTYPE_OUT));
    }

    __aicore__ inline void TilingDataInit(ScatterMeanTilingData *tiling_data)
    {
        curBlockIdx = GetBlockIdx();
        usedCoreNum = tiling_data->usedCoreNum;
        head = tiling_data->head;
        bacthSmallCore = tiling_data->bacthSmallCore;
        bacthBigCore = tiling_data->bacthBigCore;
        taskNum = tiling_data->taskNum;
        taskEachDataNum = tiling_data->taskEachLine;
        taskLastDataNum = tiling_data->taskLastLine;
        bigCoreNum = tiling_data->bigCoreNum;
        outLineEachBacth = tiling_data->outLineEachBacth;
        coreEachHead = tiling_data->coreEachHead;
        indicesLoop = tiling_data->indicesLoop;
        indicesLastNum = tiling_data->indicesLastNum;
        srcNum = tiling_data->srcNum;
        indicesNum = tiling_data->indicesNum;
        outNum = tiling_data->outNum;
        ubIndicesNum = tiling_data->ubIndicesNum;

        indicesHeadNum = indicesNum / head;
        outHeadNum = outNum / head;

        if (curBlockIdx < bigCoreNum) {
            batchNum = bacthBigCore;
        } else {
            batchNum = bacthSmallCore;
        }

        if (curBlockIdx % coreEachHead == coreEachHead - 1) {
            taskNum = tiling_data->taskNumLast;
            taskEachDataNum = tiling_data->taskEachLineLast;
            taskLastDataNum = tiling_data->taskLastLineLast;
        }

        // if head < usedCoreNum, src in one head can be  processed by multiple cores
        baseheadId = curBlockIdx / coreEachHead;

        // indicates which part of currently head the core is processing, the headPartId of first part is 0
        headPartId = curBlockIdx % coreEachHead;

        indicesEachBlock = BLOCK_SIZE / sizeof(DTYPE_INDICES);
        dataEachBlock = BLOCK_SIZE / sizeof(DTYPE_SRC);
    }
    __aicore__ inline void Process()
    {
        for (uint64_t i = 0; i < batchNum; i++) {
            Compute(i);
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

    __aicore__ inline void ComputeEachTaskNoTail(uint64_t taskId, uint64_t batchId, uint64_t dataNum)
    {
        LocalTensor<DTYPE_INDICES>indicesLocal = inQueueIndices.Get<DTYPE_INDICES>();
        outLocalTemp = outQueueOut.Get<DTYPE_OUT>();
        srcLocal = inQueueSrc.Get<DTYPE_SRC>();

        uint64_t headId = baseheadId + batchId;
        uint64_t outHeadOffset = headId * outHeadNum;
        uint64_t outBaseOffset = outHeadOffset + headPartId * outLineEachBacth + taskId * taskEachDataNum;
        pipe_barrier(PIPE_ALL);
        DataCopy(outLocalTemp, outGm[outBaseOffset], AlignUp(taskEachDataNum, indicesEachBlock));

        countTemp = outSetNote.Get<DTYPE_SRC>();
        Duplicate(countTemp, (float)0, dataNum);

        int64_t indicesStart = outBaseOffset - outHeadOffset;
        int addsForSubindicesStart = 0-(int)indicesStart;
        auto indicesOffset = headId * indicesHeadNum;
        auto ubIndicesNumAlign = AlignUp(ubIndicesNum, indicesEachBlock);
        for (uint64_t loop = 0; loop < indicesLoop; loop++) {
            DataCopy(indicesLocal, indicesGm[indicesOffset], ubIndicesNumAlign);
            DataCopy(srcLocal, srcGm[indicesOffset], ubIndicesNumAlign);
            pipe_barrier(PIPE_ALL);

            Adds(indicesLocal, indicesLocal, addsForSubindicesStart, ubIndicesNumAlign);

            for (uint64_t idx = 0; idx < ubIndicesNum; idx++) {
                DTYPE_INDICES offsetInOut = indicesLocal.GetValue(idx);
                if (offsetInOut >= 0 && offsetInOut < dataNum) {
                    outLocalTemp.SetValue(offsetInOut, outLocalTemp.GetValue(offsetInOut) + srcLocal.GetValue(idx));
                    countTemp.SetValue(offsetInOut, countTemp.GetValue(offsetInOut) + 1);
                }
            }
            indicesOffset = indicesOffset + ubIndicesNum;
        }

        if (indicesLastNum != 0) {
            DataCopy(indicesLocal, indicesGm[indicesOffset], AlignUp(indicesLastNum, indicesEachBlock));
            DataCopy(srcLocal, srcGm[indicesOffset], AlignUp(indicesLastNum, indicesEachBlock));

            pipe_barrier(PIPE_ALL);
            Adds(indicesLocal, indicesLocal, addsForSubindicesStart, AlignUp(indicesLastNum, indicesEachBlock));

            for (uint64_t idx = 0; idx < indicesLastNum; idx++) {
                DTYPE_INDICES offsetInOut = indicesLocal.GetValue(idx);
                if (offsetInOut >= 0 && offsetInOut < dataNum) {
                    outLocalTemp.SetValue(offsetInOut, outLocalTemp.GetValue(offsetInOut) + srcLocal.GetValue(idx));
                    countTemp.SetValue(offsetInOut, countTemp.GetValue(offsetInOut) + 1);
                }
            }
        }

        CopyParamasInit(dataNum);
        DataCopyPad(countGm[outBaseOffset], countTemp, copyParamsOut);
        DataCopyPad(outGm[outBaseOffset], outLocalTemp, copyParamsOut);
    }

    __aicore__ inline void Compute(uint64_t batchId)
    {
        for (uint64_t i = 0; i < taskNum - 1; i++) {
            ComputeEachTaskNoTail(i, batchId, taskEachDataNum);
        }
        if (taskLastDataNum != 0) {
            ComputeEachTaskNoTail(taskNum - 1, batchId, taskLastDataNum);
        }
    }

private:
    TPipe* pipe;
    TBuf<TPosition::VECCALC> inQueueIndices, inQueueSrc;
    TBuf<TPosition::VECCALC> outQueueOut;
    TBuf<TPosition::VECCALC> outSetNote;

    GlobalTensor<DTYPE_VAR> varGm;
    GlobalTensor<DTYPE_INDICES> indicesGm;
    GlobalTensor<DTYPE_SRC> srcGm;
    GlobalTensor<DTYPE_OUT> outGm;
    GlobalTensor<DTYPE_COUNT> countGm;

    LocalTensor<float> countTemp;
    LocalTensor<DTYPE_SRC> srcLocal;
    LocalTensor<DTYPE_OUT> outLocalTemp;

    DataCopyExtParams copyParamsOut;
    uint64_t curBlockIdx;
    uint64_t usedCoreNum, bigCoreNum;
    uint64_t head;
    uint64_t bacthSmallCore, bacthBigCore, taskNum;
    uint64_t taskEachDataNum, taskLastDataNum, outLineEachBacth, taskEachLineLast;
    uint64_t coreEachHead;
    uint64_t indicesEachBlock, dataEachBlock;
    uint64_t indicesLoop, indicesLastNum;
    uint64_t batchNum;
    uint64_t baseheadId, headPartId;
    uint64_t srcNum, indicesNum, outNum;
    uint64_t indicesHeadNum, outHeadNum;
    uint64_t ubIndicesNum;
};
// }
#endif