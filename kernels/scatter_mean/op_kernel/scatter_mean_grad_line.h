/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef _SCATTER_MEAN_GRAD_LINE_H_
#define _SCATTER_MEAN_GRAD_LINE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scatter_mean_grad_base.h"

namespace ScatterMeanGradNS {
using namespace AscendC;
template <typename T>
class ScatterMeanGradLine : public ScatterMeanGradBase<T> {
public:
    __aicore__ inline ScatterMeanGradLine() {}
    __aicore__ inline void InitLine(GM_ADDR gradOut, GM_ADDR index, GM_ADDR count, GM_ADDR gradIn, const ScatterMeanGradTilingData* tilingData)
    {
        this->InitTiling(tilingData);
        InitLocalTiling(tilingData);
        gradInGm.SetGlobalBuffer((__gm__ T *)gradIn, this->gradInNum);
        indexGm.SetGlobalBuffer((__gm__ int32_t *)index, this->indexNum);
        gradOutGm.SetGlobalBuffer((__gm__ T *)gradOut, this->gradOutNum);
        countGm.SetGlobalBuffer((__gm__ T *)count, this->countNum);

        pipe.InitBuffer(inCountUb, this->paramsEachBlock * sizeof(T));
        pipe.InitBuffer(inIndexUb, this->indexUbSize * sizeof(int32_t));
        pipe.InitBuffer(outGradInUb, gradInUbSize * sizeof(T));
        pipe.InitBuffer(inGradOutUb, gradOutUbSize * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (this->tilingMode == 1) {
            for (int32_t i = 0; i < taskNum - 1; i++) {
                ComputeSmallTail(i, taskEachLine);
            }
            if (taskLastLine != 0) {
                ComputeSmallTail(taskNum - 1, taskLastLine);
            }
        } else {
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
            for (int32_t i = 0; i < taskNum - 1; i++) {
                ComputeEachTask(i, taskEachLine);
            }
            if (taskLastLine != 0) {
                ComputeEachTask(taskNum - 1, taskLastLine);
            }
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> inGradOutUb, inIndexUb, inCountUb, outGradInUb;
    GlobalTensor<T> gradInGm, gradOutGm, countGm;
    GlobalTensor<int32_t> indexGm;

    __aicore__ inline void InitLocalTiling(const ScatterMeanGradTilingData *tiling_data);
    __aicore__ inline void ComputeTail(uint64_t idxTure, uint64_t dataInIndices, uint64_t gradInLocalOffset);
    __aicore__ inline void ComputeEachTask(int32_t taskId, uint64_t taskLine);
    __aicore__ inline void ComputeSmallTail(int32_t taskId, uint64_t taskLine);
    __aicore__ inline int64_t getEventIdforDoublebuffer();

    uint64_t ubTailNum;
    uint64_t tailLoop;
    uint64_t tailLast;
    uint64_t gradInUbSize;
    uint64_t gradOutUbSize;
    uint64_t indicesBaseOffset;
    uint64_t taskNum, taskEachLine, taskLastLine;
    uint64_t gradInEachHead;
    uint64_t outEachHead;
    uint64_t eventId;
    uint64_t countUb;
};

template <typename T>
__aicore__ inline void ScatterMeanGradLine<T>::InitLocalTiling(const ScatterMeanGradTilingData *tiling_data)
{
    taskNum = tiling_data->taskNum;
    taskEachLine = tiling_data->taskEachLine;
    taskLastLine = tiling_data->taskLastLine;
    ubTailNum = tiling_data->ubTailNum;
    gradInUbSize = tiling_data->gradInUbSize;
    gradOutUbSize = tiling_data->gradOutUbSize;

    uint64_t coreDataLine = tiling_data->bacthSmallCore;
    if (this->curBlockIdx < this->bigCoreNum) {
        coreDataLine = coreDataLine + 1;
        indicesBaseOffset = this->curBlockIdx * coreDataLine;
    } else {
        if (taskLastLine == 1) {
            taskNum = taskNum - 1;
        } else {
            taskLastLine = taskLastLine - 1;
        }
        indicesBaseOffset = this->bigCoreNum * (coreDataLine + 1) + (this->curBlockIdx - this->bigCoreNum) * coreDataLine;
    }
    gradInEachHead = this->dimRange * this->body;
    outEachHead = this->dimRangeOut * this->body;

    tailLoop = this->tail / ubTailNum;
    tailLast = this->tail - tailLoop * ubTailNum;

    countUb = 0;
    eventId = EVENT_ID0;

    this->copyParamsOut.blockLen = static_cast<uint32_t>(tailLast * sizeof(float));
}

template <typename T>
__aicore__ inline int64_t ScatterMeanGradLine<T>::getEventIdforDoublebuffer()
{
    uint64_t localOffset;
    if (countUb % BUFFER_NUM == 0) {
        localOffset = 0;
        eventId = EVENT_ID0;
    } else {
        localOffset = ubTailNum;
        eventId = EVENT_ID1;
    }
    countUb = countUb + 1;
    return localOffset;
}

template <typename T>
__aicore__ inline void ScatterMeanGradLine<T>::ComputeSmallTail(int32_t taskId, uint64_t taskLine)
{
    LocalTensor<int32_t> indicesLocal = inIndexUb.Get<int32_t>();
    LocalTensor<T> gradInLocal = outGradInUb.Get<T>();
    LocalTensor<T> gradOutLocal = inGradOutUb.Get<T>();
    LocalTensor<T> countLocal = inCountUb.Get<T>();

    uint64_t indicesOffset = indicesBaseOffset + taskEachLine * taskId;
    uint64_t gradInOffset = indicesOffset * this->tail;
    pipe_barrier(PIPE_ALL);
    DataCopy(indicesLocal, indexGm[indicesOffset], AlignUp(taskLine, this->indicesEachBlock));
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    Muls(indicesLocal, indicesLocal, (int32_t)this->body, AlignUp(taskLine, this->indicesEachBlock));

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
#pragma bisheng auto_sync parallel
    for (uint64_t idx = 0; idx < taskLine; idx++) {
        DTYPE_INDEX dataInIndices = indicesLocal.GetValue(idx);
        auto idxTure = indicesOffset + idx;
        auto gradInLocalOffset = idx * this->tail;

        auto idx1 = idxTure / gradInEachHead;
        auto idx2 = (idxTure - idx1 * gradInEachHead) / this->body;
        auto idx3 = idxTure - idx1 * gradInEachHead - idx2 * this->body;
        auto outLineOffset = idx3 + dataInIndices + idx1 * outEachHead;

        DataCopy(countLocal, countGm[outLineOffset], this->indicesEachBlock);
        auto mulValue = 1 / countLocal.GetValue(0);

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        DataCopy(gradOutLocal, gradOutGm[outLineOffset * this->tail], ubTailNum);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        Muls(gradInLocal[gradInLocalOffset], gradOutLocal, mulValue, ubTailNum);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
    DataCopy(gradInGm[gradInOffset], gradInLocal, taskLine * this->tail);
}

template <typename T>
__aicore__ inline void ScatterMeanGradLine<T>::ComputeTail(uint64_t idxTure, uint64_t dataInIndices, uint64_t gradInLocalOffset)
{
    LocalTensor<T> gradOutLocal = inGradOutUb.Get<T>();
    LocalTensor<T> countLocal = inCountUb.Get<T>();

    auto idx1 = idxTure / gradInEachHead;
    auto idx2 = (idxTure - idx1 * gradInEachHead) / this->body;
    auto idx3 = idxTure - idx1 * gradInEachHead - idx2 * this->body;
    auto outLineOffset = idx3 + dataInIndices * this->body + idx1 * outEachHead;
    DataCopy(countLocal, countGm[outLineOffset], this->indicesEachBlock);
    auto mulValue = 1 / countLocal.GetValue(0);
    uint64_t offset = 0;

#pragma bisheng auto_sync parallel
    for (uint64_t loop = 0; loop < tailLoop; loop++) {
        offset = loop * ubTailNum;
        auto localOffset = getEventIdforDoublebuffer();
        wait_flag(PIPE_MTE3, PIPE_MTE2, eventId);
        DataCopy(gradOutLocal[localOffset], gradOutGm[outLineOffset * this->tail + offset], ubTailNum);
        set_flag(PIPE_MTE2, PIPE_V, eventId);
        wait_flag(PIPE_MTE2, PIPE_V, eventId);
        Muls(gradOutLocal[localOffset], gradOutLocal[localOffset], mulValue, ubTailNum);
        set_flag(PIPE_V, PIPE_MTE3, eventId);
        wait_flag(PIPE_V, PIPE_MTE3, eventId);
        DataCopy(gradInGm[gradInLocalOffset + offset], gradOutLocal[localOffset], ubTailNum);
        set_flag(PIPE_MTE3, PIPE_MTE2, eventId);
    }

    offset = tailLoop * ubTailNum;
    if (tailLast != 0) {
        auto localOffset = getEventIdforDoublebuffer();
        wait_flag(PIPE_MTE3, PIPE_MTE2, eventId);
        DataCopy(gradOutLocal[localOffset], gradOutGm[outLineOffset * this->tail + offset], AlignUp(tailLast, this->paramsEachBlock));
        set_flag(PIPE_MTE2, PIPE_V, eventId);
        wait_flag(PIPE_MTE2, PIPE_V, eventId);
        Muls(gradOutLocal[localOffset], gradOutLocal[localOffset], mulValue, AlignUp(tailLast, this->paramsEachBlock));
        set_flag(PIPE_V, PIPE_MTE3, eventId);
        wait_flag(PIPE_V, PIPE_MTE3, eventId);
        DataCopyPad(gradInGm[gradInLocalOffset + offset], gradOutLocal[localOffset], this->copyParamsOut);
        set_flag(PIPE_MTE3, PIPE_MTE2, eventId);
    }
}

template <typename T>
__aicore__ inline void ScatterMeanGradLine<T>::ComputeEachTask(int32_t taskId, uint64_t taskLine)
{
    LocalTensor<int32_t> indicesLocal = inIndexUb.Get<int32_t>();
    auto indicesOffset = indicesBaseOffset + taskEachLine * taskId;
    DataCopy(indicesLocal, indexGm[indicesOffset], AlignUp(taskLine, this->indicesEachBlock));

#pragma bisheng auto_sync parallel
    for (uint64_t idx = 0; idx < taskLine; idx++) {
        DTYPE_INDEX dataInIndices = indicesLocal.GetValue(idx);
        auto idxTure = indicesOffset + idx;
        auto gradInLocalOffset = idxTure * this->tail;
        ComputeTail(idxTure, dataInIndices, gradInLocalOffset);
    }
}

}
#endif