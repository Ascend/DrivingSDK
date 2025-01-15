/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#ifndef _SCATTER_ADD_NOTAIL_H_
#define _SCATTER_ADD_NOTAIL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;
class ScatterAddNoTail {
public:
    __aicore__ inline ScatterAddNoTail() {}
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
        pipe->InitBuffer(outQueueOut, AlignUp(taskEachDataNum, MAX_MASK) * sizeof(DTYPE_OUT));
    }

    __aicore__ inline void TilingDataInit(ScatterAddTilingData *tiling_data)
    {
        uint64_t curBlockIdx = GetBlockIdx();
        uint32_t tilingMode = tiling_data->tilingMode;
        uint64_t head = tiling_data->head;
        uint64_t bigCoreNum = tiling_data->bigCoreNum;
        uint64_t coreEachHead = tiling_data->coreEachHead;
        uint64_t lineSmallCore = tiling_data->lineSmallCore;
        uint64_t lineBigCore = tiling_data->lineBigCore;

        taskNum = tiling_data->taskNum;
        taskEachDataNum = tiling_data->taskEachLine;
        taskLastDataNum = tiling_data->taskLastLine;
        outLineEachCore = tiling_data->outLineEachCore;
        srcNum = tiling_data->srcNum;
        indicesNum = tiling_data->indicesNum;
        outNum = tiling_data->outNum;
        indicesDealNum = tiling_data->indicesDealNum;

        indicesHeadNum = indicesNum / head;
        outHeadNum = outNum / head;

        if (curBlockIdx < bigCoreNum) {
            batchNum = lineBigCore;
        } else {
            batchNum = lineSmallCore;
        }

        if (curBlockIdx % coreEachHead == coreEachHead - 1) {
            taskNum = tiling_data->taskNumLast;
            taskEachDataNum = tiling_data->taskEachLineLast;
            taskLastDataNum = tiling_data->taskLastLineLast;
        }

        headPartId = 0;
        if (tilingMode == 1) {
            if (curBlockIdx % coreEachHead == coreEachHead - 1) {
                taskNum = tiling_data->taskNumLast;
                taskEachDataNum = tiling_data->taskEachLineLast;
                taskLastDataNum = tiling_data->taskLastLineLast;
            }
            // if head < usedCoreNum, src in one head can be  processed by multiple cores
            baseHeadId = curBlockIdx / coreEachHead;
            // indicates which part of currently head the core is processing, the headPartId of first part is 0
            headPartId = curBlockIdx % coreEachHead;
        } else {
            if (curBlockIdx < bigCoreNum) {
                baseHeadId = lineBigCore * curBlockIdx;
                taskNum = tiling_data->taskNum;
                taskEachDataNum = tiling_data->taskEachLine;
                taskLastDataNum = tiling_data->taskLastLine;
            } else {
                baseHeadId = lineBigCore * bigCoreNum + (curBlockIdx - bigCoreNum) * lineSmallCore;
                taskNum = tiling_data->taskNumLast;
                taskEachDataNum = tiling_data->taskEachLineLast;
                taskLastDataNum = tiling_data->taskLastLineLast;
            }
        }

        indicesEachBlock = BLOCK_SIZE / sizeof(DTYPE_INDICES);
        dataEachBlock = BLOCK_SIZE / sizeof(DTYPE_SRC);

        indicesLoop = indicesHeadNum / indicesDealNum;
        indicesLastNum = indicesHeadNum % indicesDealNum;
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

    __aicore__ inline void ComputeEachHeadNoTail(uint64_t headId, uint64_t dataNum, uint64_t outBaseOffset, int64_t indicesStart)
    {
        auto basdIndicesOffset = headId * indicesHeadNum;
        int addsForSubindicesStart = 0 - (int)indicesStart;

        uint64_t ubIndicesNumAlign = AlignUp(indicesDealNum, dataEachBlock);
        for (uint64_t loop = 0; loop < indicesLoop; loop++) {
            auto indicesOffset = basdIndicesOffset + loop * indicesDealNum;
            DataCopy(indicesLocal, indicesGm[indicesOffset], ubIndicesNumAlign);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            DataCopy(srcLocal, srcGm[indicesOffset], ubIndicesNumAlign);

            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Adds(indicesLocal, indicesLocal, addsForSubindicesStart, ubIndicesNumAlign);

            for (uint64_t idx = 0; idx < indicesDealNum; idx++) {
                DTYPE_INDICES dataInIndices = indicesLocal.GetValue(idx);
                if (dataInIndices >= 0 && dataInIndices < dataNum) {
                    outLocalTemp.SetValue(dataInIndices, outLocalTemp.GetValue(dataInIndices) + srcLocal.GetValue(idx));
                }
            }
        }
        uint64_t indicesLastNumAlign = AlignUp(indicesLastNum, dataEachBlock);
        if (indicesLastNum != 0) {
            auto indicesOffset = basdIndicesOffset + indicesLoop * indicesDealNum;
            DataCopy(indicesLocal, indicesGm[indicesOffset], indicesLastNumAlign);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            DataCopy(srcLocal, srcGm[indicesOffset], indicesLastNumAlign);

            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Adds(indicesLocal, indicesLocal, addsForSubindicesStart, AlignUp(indicesLastNum, indicesEachBlock));

            for (uint64_t idx = 0; idx < indicesLastNum; idx++) {
                DTYPE_INDICES dataInIndices = indicesLocal.GetValue(idx);
                if (dataInIndices >= 0 && dataInIndices < dataNum) {
                    outLocalTemp.SetValue(dataInIndices, outLocalTemp.GetValue(dataInIndices) + srcLocal.GetValue(idx));
                }
            }
        }
    }

    __aicore__ inline void ComputeEachTaskNoTail(uint64_t taskId, uint64_t batchId, uint64_t dataNum)
    {
        indicesLocal = inQueueIndices.Get<DTYPE_INDICES>();
        outLocalTemp = outQueueOut.Get<DTYPE_OUT>();
        srcLocal = inQueueSrc.Get<DTYPE_SRC>();

        uint64_t headId = baseHeadId + batchId;  // 一个batch 一个head
        uint64_t outHeadOffset = headId * outHeadNum;
        // 一个head由多个task处理
        uint64_t outBaseOffset = outHeadOffset + headPartId * outLineEachCore + taskId * taskEachDataNum;
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        DataCopy(outLocalTemp, outGm[outBaseOffset], AlignUp(taskEachDataNum, indicesEachBlock));

        int64_t indicesStart = outBaseOffset - outHeadOffset;
        ComputeEachHeadNoTail(headId, dataNum, outBaseOffset, indicesStart);

        CopyParamasInit(dataNum);
        DataCopyPad(outGm[outBaseOffset], outLocalTemp, copyParamsOut);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void Compute(uint64_t batchId)
    {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (uint64_t i = 0; i < taskNum - 1; i++) {
            ComputeEachTaskNoTail(i, batchId, taskEachDataNum);
        }
        if (taskLastDataNum != 0) {
            ComputeEachTaskNoTail(taskNum - 1, batchId, taskLastDataNum);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

private:
    TPipe* pipe;
    TBuf<TPosition::VECCALC> inQueueIndices;
    TBuf<TPosition::VECCALC> inQueueSrc;
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
    uint64_t outLineEachCore;
    uint64_t indicesEachBlock;
    uint64_t dataEachBlock;
    uint32_t indicesLoop;
    uint32_t indicesLastNum;
    uint32_t batchNum;
    uint32_t baseHeadId;
    uint32_t headPartId;
    uint64_t srcNum, indicesNum, outNum;
    uint64_t indicesHeadNum, outHeadNum;
    uint64_t indicesDealNum;
};
// }
#endif