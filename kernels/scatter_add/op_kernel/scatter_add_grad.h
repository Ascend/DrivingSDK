/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef _SCATTER_ADD_GRAD_H_
#define _SCATTER_ADD_GRAD_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scatter_add_grad_base.h"
namespace ScatterAddGradNS {
using namespace AscendC;

template <typename T>
class ScatterAddGradV2 : public ScatterAddGradBase<T> {
public:
    __aicore__ inline ScatterAddGradV2() {}
    __aicore__ inline void Init(GM_ADDR gradOut, GM_ADDR index, GM_ADDR gradIn, const ScatterAddGradTilingData* tilingData)
    {
        this->InitTiling(tilingData);
        InitNoTailTiling(tilingData);

        gradInGm.SetGlobalBuffer((__gm__ T *)gradIn, this->gradInNum);
        indexGm.SetGlobalBuffer((__gm__ int32_t *)index, this->indexNum);
        gradOutGm.SetGlobalBuffer((__gm__ T *)gradOut, this->gradOutNum);

        pipe.InitBuffer(inGradOutUb, this->gradOutUbSize * sizeof(T));
        pipe.InitBuffer(inIndexUb, this->indexUbSize * sizeof(int32_t));
        pipe.InitBuffer(outGradInUb, this->indexUbSize * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (this->tilingMode == 0) {
            this->copyParamsOut.blockLen = static_cast<uint32_t>(headIndexSize * sizeof(float));
            this->copyParamsIn.blockLen = static_cast<uint32_t>(headIndexSize * sizeof(float));
            this->copyParamsInPad.isPad = true;
            this->copyParamsInPad.rightPadding = headIndexSizeAlign - headIndexSize;
            for (uint64_t taskId = 0; taskId < taskNum - 1; taskId++) {
                ComputeModeSmallData(taskId, headTask);
            }
            if (headLastTask != 0) {
                ComputeModeSmallData(taskNum - 1, headLastTask);
            }
        } else {
            indexLoop = headIndexSize / this->indexUbSize;
            indexLast = headIndexSize - indexLoop * this->indexUbSize;
            this->copyParamsOut.blockLen = static_cast<int32_t>(indexLast * sizeof(float));
            set_flag(PIPE_MTE3, PIPE_V, 0);
            set_flag(PIPE_V, PIPE_MTE2, 0);
            set_flag(PIPE_V, PIPE_MTE2, 1);
            for (uint64_t taskId = 0; taskId < taskNum - 1; taskId++) {
                ComputeModeLargeData(taskId, headTask);
            }
            if (headLastTask != 0) {
                ComputeModeLargeData(taskNum - 1, headLastTask);
            }
            wait_flag(PIPE_MTE3, PIPE_V, 0);
            wait_flag(PIPE_V, PIPE_MTE2, 0);
            wait_flag(PIPE_V, PIPE_MTE2, 1);
        }
    }

private:
    __aicore__ inline void InitNoTailTiling(const ScatterAddGradTilingData *tiling_data)
    {
        auto headTaskSmall = tiling_data->headTaskSmall;
        auto taskNumSmall = tiling_data->taskNumSmall;
        auto headLastTaskSmall = tiling_data->headLastTaskSmall;
        auto headTaskBig = tiling_data->headTaskBig;
        auto taskNumBig = tiling_data->taskNumBig;
        auto headLastTaskBig = tiling_data->headLastTaskBig;

        headOutSize = this->dimRangeOut * this->paramsPro;
        headIndexSize = this->dimRange * this->paramsPro;
        headIndexSizeAlign = AlignUp(headIndexSize, B32_DATA_NUM_PER_BLOCK);

        auto headBigCore = (taskNumBig - 1) * headTaskBig + headLastTaskBig;
        auto headSmallCore = headBigCore - 1;

        if (this->curBlockIdx < this->bigCoreNum) {
            taskNum = taskNumBig;
            headTask = headTaskBig;
            headLastTask = headLastTaskBig;
            headBaseId = this->curBlockIdx * headBigCore;
        } else {
            taskNum = taskNumSmall;
            headTask = headTaskSmall;
            headLastTask = headLastTaskSmall;
            headBaseId = this->bigCoreNum * headBigCore + (this->curBlockIdx - this->bigCoreNum) * headSmallCore;
        }
    }

    __aicore__ inline void ComputeModeSmallData(uint64_t taskId, uint64_t headNum)
    {
        LocalTensor<int32_t> indexLocal = inIndexUb.Get<int32_t>();
        LocalTensor<T> gradOutLocal = inGradOutUb.Get<T>();
        LocalTensor<T> gradInLocal = outGradInUb.Get<T>();

        uint64_t firstHeadId = headBaseId + headTask * taskId;
        uint64_t indexOffset = firstHeadId * headIndexSize;
        uint64_t outOffset = firstHeadId * headOutSize;

        this->copyParamsIn.blockCount = static_cast<uint32_t>(headNum);
        this->copyParamsOut.blockCount = static_cast<uint32_t>(headNum);
        uint64_t outAlign = AlignUp(headNum * headOutSize, this->paramsEachBlock);

        set_flag(PIPE_V, PIPE_MTE2, 0);
        wait_flag(PIPE_V, PIPE_MTE2, 0);
        DataCopyPad(indexLocal, indexGm[indexOffset], copyParamsIn, copyParamsInPad);
        DataCopy(gradOutLocal, gradOutGm[outOffset], outAlign);
        set_flag(PIPE_MTE2, PIPE_V, 0);
        wait_flag(PIPE_MTE2, PIPE_V, 0);
        for (uint64_t head = 0; head < headNum; head++) {
            int32_t indexLocalOffset = head * headIndexSizeAlign;
            int32_t outLocalOffset = head * headOutSize;
            Adds(indexLocal[indexLocalOffset], indexLocal[indexLocalOffset], outLocalOffset, headIndexSizeAlign);
        }
        Muls(indexLocal, indexLocal, (int32_t)sizeof(T), headNum * headIndexSizeAlign);
        set_flag(PIPE_MTE3, PIPE_V, 0);
        wait_flag(PIPE_MTE3, PIPE_V, 0);
        Gather(gradInLocal, gradOutLocal, indexLocal.ReinterpretCast<uint32_t>(), 0, headNum * headIndexSizeAlign);
        set_flag(PIPE_V, PIPE_MTE3, 0);
        wait_flag(PIPE_V, PIPE_MTE3, 0);
        DataCopyPad(gradInGm[indexOffset], gradInLocal, this->copyParamsOut);
    }

    __aicore__ inline void ComputeModeLargeData(uint64_t taskId, uint64_t headNum)
    {
        LocalTensor<int32_t> indexLocal = inIndexUb.Get<int32_t>();
        LocalTensor<T> gradOutLocal = inGradOutUb.Get<T>();
        LocalTensor<T> gradInLocal = outGradInUb.Get<T>();

        uint64_t firstHeadId = headBaseId + headTask * taskId;
        uint64_t indexOffset = firstHeadId * headIndexSize;
        uint64_t outOffset = firstHeadId * headOutSize;
        uint64_t outAlign = AlignUp(headNum * headOutSize, this->paramsEachBlock);

        wait_flag(PIPE_V, PIPE_MTE2, 1);
        DataCopy(gradOutLocal, gradOutGm[outOffset], outAlign);
        set_flag(PIPE_MTE2, PIPE_V, 1);
        wait_flag(PIPE_MTE2, PIPE_V, 1);

        for (uint64_t head = 0; head < headNum; head++) {
            uint64_t indicesAlign = AlignUp(headIndexSize, this->indicesEachBlock);
            auto headOutOffset = head * headOutSize;
            for (uint64_t loop = 0; loop < indexLoop; loop++) {
                uint64_t offset = this->indexUbSize * loop;
                wait_flag(PIPE_V, PIPE_MTE2, 0);
                DataCopy(indexLocal, indexGm[indexOffset + head * headIndexSize + offset], this->indexUbSize);
                set_flag(PIPE_MTE2, PIPE_V, 0);
                wait_flag(PIPE_MTE2, PIPE_V, 0);
                Adds(indexLocal, indexLocal, (int32_t)headOutOffset, this->indexUbSize);
                Muls(indexLocal, indexLocal, (int32_t)sizeof(T), this->indexUbSize);
                wait_flag(PIPE_MTE3, PIPE_V, 0);
                Gather(gradInLocal, gradOutLocal, indexLocal.ReinterpretCast<uint32_t>(), (uint32_t)0, this->indexUbSize);
                set_flag(PIPE_V, PIPE_MTE2, 0);
                set_flag(PIPE_V, PIPE_MTE3, 0);
                wait_flag(PIPE_V, PIPE_MTE3, 0);
                DataCopy(gradInGm[indexOffset + head * headIndexSize + offset], gradInLocal, this->indexUbSize);
                set_flag(PIPE_MTE3, PIPE_V, 0);
            }
            if (indexLast != 0) {
                uint64_t offset = this->indexUbSize * indexLoop;
                uint64_t indicesAlign = AlignUp(indexLast, this->indicesEachBlock);
                wait_flag(PIPE_V, PIPE_MTE2, 0);
                DataCopy(indexLocal, indexGm[indexOffset + head * headIndexSize + offset], indicesAlign);
                set_flag(PIPE_MTE2, PIPE_V, 0);
                wait_flag(PIPE_MTE2, PIPE_V, 0);
                Adds(indexLocal, indexLocal, (int32_t)headOutOffset, indicesAlign);
                Muls(indexLocal, indexLocal, (int32_t)sizeof(T), indicesAlign);
                wait_flag(PIPE_MTE3, PIPE_V, 0);
                Gather(gradInLocal, gradOutLocal, indexLocal.ReinterpretCast<uint32_t>(), (uint32_t)0, indexLast);
                set_flag(PIPE_V, PIPE_MTE2, 0);
                set_flag(PIPE_V, PIPE_MTE3, 0);
                wait_flag(PIPE_V, PIPE_MTE3, 0);
                DataCopyPad(gradInGm[indexOffset + head * headIndexSize + offset], gradInLocal, this->copyParamsOut);
                set_flag(PIPE_MTE3, PIPE_V, 0);
            }
        }
        set_flag(PIPE_V, PIPE_MTE2, 1);
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> inGradOutUb, inIndexUb, outGradInUb;

    GlobalTensor<T> gradInGm, gradOutGm;
    GlobalTensor<int32_t> indexGm;

    uint64_t headOutSize;
    uint64_t headIndexSize;
    uint64_t headIndexSizeAlign;
    uint64_t taskNum;
    uint64_t headTask;
    uint64_t headLastTask;
    uint64_t headBaseId;
    uint64_t indexLoop;
    uint64_t indexLast;

    DataCopyExtParams copyParamsIn = {1, 8, 0, 0, 0};
    DataCopyPadExtParams<int32_t> copyParamsInPad = {false, 0, 0, 0};
};
}
#endif