/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef _SCATTER_ADD_GRAD_LINE_H_
#define _SCATTER_ADD_GRAD_LINE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scatter_add_grad_base.h"

namespace ScatterAddGradNS {
using namespace AscendC;
template <typename T>
class ScatterAddGradLine : public ScatterAddGradBase<T> {
public:
    __aicore__ inline ScatterAddGradLine() {}
    __aicore__ inline void InitLine(GM_ADDR gradOut, GM_ADDR index, GM_ADDR gradIn, const ScatterAddGradTilingData* tilingData)
    {
        this->InitTiling(tilingData);
        InitLocalTiling(tilingData);
        gradInGm.SetGlobalBuffer((__gm__ T *)gradIn, this->gradInNum);
        indexGm.SetGlobalBuffer((__gm__ int32_t *)index, this->indexNum);
        gradOutGm.SetGlobalBuffer((__gm__ T *)gradOut, this->gradOutNum);

        pipe.InitBuffer(inIndexUb, this->indexUbSize * sizeof(int32_t));
        pipe.InitBuffer(inGradOutUb, gradOutUbSize * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (this->tilingMode == 0) {
            SetFlagMET3_2();
            for (int32_t i = 0; i < taskNum - 1; i++) {
                ComputeEachTask(i, taskEachLine);
            }
            if (taskLastLine != 0) {
                ComputeEachTask(taskNum - 1, taskLastLine);
            }
            WaitFlagMET3_2();
        } else {
            for (int32_t i = 0; i < taskNum - 1; i++) {
                ComputeSmallTail(i, taskEachLine);
            }
            if (taskLastLine != 0) {
                ComputeSmallTail(taskNum - 1, taskLastLine);
            }
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> inGradOutUb, inIndexUb, outGradInUb;
    GlobalTensor<T> gradInGm, gradOutGm;
    GlobalTensor<int32_t> indexGm;

    __aicore__ inline void InitLocalTiling(const ScatterAddGradTilingData *tiling_data);
    __aicore__ inline void ComputeTail(uint64_t idxTure, uint64_t dataInIndices, uint64_t gradInLocalOffset);
    __aicore__ inline void ComputeEachTask(int32_t taskId, uint64_t taskLine);
    __aicore__ inline void ComputeSmallTail(int32_t taskId, uint64_t taskLine);
    __aicore__ inline int64_t getEventIdforDoublebuffer();
    __aicore__ inline void WaitFlagMET3_2();
    __aicore__ inline void SetFlagMET3_2();

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

    uint64_t headTask;
    uint64_t headLastTask;
    uint64_t headBaseId;
    uint64_t countUb;
    uint32_t dbTimes;
};

template <typename T>
__aicore__ inline void ScatterAddGradLine<T>::InitLocalTiling(const ScatterAddGradTilingData *tiling_data)
{
    taskNum = tiling_data->taskNum;
    taskEachLine = tiling_data->taskEachLine;
    taskLastLine = tiling_data->taskLastLine;
    ubTailNum = tiling_data->ubTailNum;
    gradInUbSize = tiling_data->gradInUbSize;
    gradOutUbSize = tiling_data->gradOutUbSize;

    dbTimes = tiling_data->dbTimes;

    uint64_t coreDataLine = tiling_data->bacthSmallCore;
    if (this->curBlockIdx < this->bigCoreNum) {
        coreDataLine = coreDataLine + 1;
        indicesBaseOffset = this->curBlockIdx * coreDataLine;
    } else {
        if (taskLastLine == 1) {
            taskNum = taskNum - 1;
            taskLastLine = taskEachLine;
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
__aicore__ inline int64_t ScatterAddGradLine<T>::getEventIdforDoublebuffer()
{
    uint64_t localOffset;
    eventId = countUb % dbTimes; // int64_t
    localOffset = AlignUp(ubTailNum, this->paramsEachBlock) * eventId;

    countUb = countUb + 1;
    return localOffset;
}

template <typename T>
__aicore__ inline void ScatterAddGradLine<T>::SetFlagMET3_2() {
    for (uint32_t i = 0; i < dbTimes; i++) {
        SetFlag<HardEvent::MTE3_MTE2>(i);
    }
}

template <typename T>
__aicore__ inline void ScatterAddGradLine<T>::WaitFlagMET3_2() {
    for (uint32_t i = 0; i < dbTimes; i++) {
        WaitFlag<HardEvent::MTE3_MTE2>(i);
    }
}

template <typename T>
__aicore__ inline void ScatterAddGradLine<T>::ComputeSmallTail(int32_t taskId, uint64_t taskLine)
{
    LocalTensor<int32_t> indicesLocal = inIndexUb.Get<int32_t>();
    LocalTensor<T> gradOutLocal = inGradOutUb.Get<T>();

    auto indicesOffset = indicesBaseOffset + taskEachLine * taskId;
    auto gradInOffset = indicesOffset * this->tail;
    pipe_barrier(PIPE_ALL);
    DataCopy(indicesLocal, indexGm[indicesOffset], AlignUp(taskLine, this->indicesEachBlock));

    for (uint64_t idx = 0; idx < taskLine; idx++) {
        DTYPE_INDEX dataInIndices = indicesLocal.GetValue(idx);
        auto idxTure = indicesOffset + idx;
        auto gradInLocalOffset = idx * AlignUp(this->tail, this->paramsEachBlock);

        auto headId = idxTure / gradInEachHead;
        auto outLineOffset = dataInIndices + headId * outEachHead;

        DataCopy(gradOutLocal[gradInLocalOffset], gradOutGm[outLineOffset * this->tail], ubTailNum);
    }

    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID2);
    this->copyParamsOut = {static_cast<uint16_t>(taskLine),
                           static_cast<uint32_t>(this->tail * sizeof(float)), 0, 0, 0};
    DataCopyPad(gradInGm[gradInOffset], gradOutLocal, this->copyParamsOut);
}

template <typename T>
__aicore__ inline void ScatterAddGradLine<T>::ComputeTail(uint64_t idxTure, uint64_t dataInIndices, uint64_t gradInLocalOffset)
{
    LocalTensor<T> gradOutLocal = inGradOutUb.Get<T>();

    auto headId = idxTure / gradInEachHead;
    auto outLineOffset = dataInIndices + headId * outEachHead;
    uint64_t offset = 0;

    for (uint64_t loop = 0; loop < tailLoop; loop++) {
        offset = loop * ubTailNum;
        auto localOffset = getEventIdforDoublebuffer();
        wait_flag(PIPE_MTE3, PIPE_MTE2, eventId);
        DataCopy(gradOutLocal[localOffset], gradOutGm[outLineOffset * this->tail + offset], ubTailNum);
        set_flag(PIPE_MTE2, PIPE_MTE3, eventId);
        wait_flag(PIPE_MTE2, PIPE_MTE3, eventId);
        DataCopy(gradInGm[gradInLocalOffset + offset], gradOutLocal[localOffset], ubTailNum);
        set_flag(PIPE_MTE3, PIPE_MTE2, eventId);
    }

    offset = tailLoop * ubTailNum;
    if (tailLast != 0) {
        auto localOffset = getEventIdforDoublebuffer();
        wait_flag(PIPE_MTE3, PIPE_MTE2, eventId);
        DataCopy(gradOutLocal[localOffset], gradOutGm[outLineOffset * this->tail + offset], AlignUp(tailLast, this->paramsEachBlock));
        set_flag(PIPE_MTE2, PIPE_MTE3, eventId);
        wait_flag(PIPE_MTE2, PIPE_MTE3, eventId);
        DataCopyPad(gradInGm[gradInLocalOffset + offset], gradOutLocal[localOffset], this->copyParamsOut);
        set_flag(PIPE_MTE3, PIPE_MTE2, eventId);
    }
}

template <typename T>
__aicore__ inline void ScatterAddGradLine<T>::ComputeEachTask(int32_t taskId, uint64_t taskLine)
{
    LocalTensor<int32_t> indicesLocal = inIndexUb.Get<int32_t>();
    auto indicesOffset = indicesBaseOffset + taskEachLine * taskId;
    DataCopy(indicesLocal, indexGm[indicesOffset], AlignUp(taskLine, this->indicesEachBlock));

    for (uint64_t idx = 0; idx < taskLine; idx++) {
        DTYPE_INDEX dataInIndices = indicesLocal.GetValue(idx);
        auto idxTure = indicesOffset + idx;
        auto gradInLocalOffset = idxTure * this->tail;
        ComputeTail(idxTure, dataInIndices, gradInLocalOffset);
    }
}

}
#endif