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
            for (uint64_t taskId = 0; taskId < taskNum - 1; taskId++) {
                ComputeModeLargeData(taskId, headTask);
            }
            if (headLastTask != 0) {
                ComputeModeLargeData(taskNum - 1, headLastTask);
            }
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

        this->copyParamsOut.blockLen = static_cast<int32_t>(headTask * headIndexSize * sizeof(float));
    }

    __aicore__ inline void ComputeModeSmallData(uint64_t taskId, uint64_t headNum)
    {
        LocalTensor<int32_t> indexLocal = inIndexUb.Get<int32_t>();
        LocalTensor<T> gradOutLocal = inGradOutUb.Get<T>();
        LocalTensor<T> gradInLocal = outGradInUb.Get<T>();

        uint64_t firstHeadId = headBaseId + headTask * taskId;
        uint64_t indexOffset = firstHeadId * headIndexSize;
        uint64_t outOffset = firstHeadId * headOutSize;

        uint64_t indicesAlign = AlignUp(headNum * headIndexSize, this->indicesEachBlock);
        uint64_t outAlign = AlignUp(headNum * headOutSize, this->paramsEachBlock);

        DataCopy(indexLocal, indexGm[indexOffset], indicesAlign);
        DataCopy(gradOutLocal, gradOutGm[outOffset], outAlign);

        pipe_barrier(PIPE_ALL);
        for (uint64_t head = 0; head < headNum; head++) {
            uint64_t indexLocalOffset = head * headIndexSize;
            uint64_t outLocalOffset = head * headOutSize;
            for (uint64_t idx = 0; idx < headIndexSize; idx++) {
                uint64_t indexTrueOffset = indexLocalOffset + idx;
                auto indexValue = indexLocal.GetValue(indexTrueOffset);
                auto offsetInOut = indexValue + outLocalOffset;
                auto gradInValue = gradOutLocal.GetValue(offsetInOut);
                gradInLocal.SetValue(indexTrueOffset, gradInValue);
            }
        }
        this->copyParamsOut.blockLen = static_cast<int32_t>(headNum * headIndexSize * sizeof(float));
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

        DataCopy(gradOutLocal, gradOutGm[outOffset], outAlign);
        pipe_barrier(PIPE_ALL);
        for (uint64_t head = 0; head < headNum; head++) {
            uint64_t indicesAlign = AlignUp(headIndexSize, this->indicesEachBlock);
            auto headOutOffset = head * headOutSize;

            uint64_t offset = 0;
            for (uint64_t loop = 0; loop < indexLoop; loop++) {
                offset = this->indexUbSize * loop;
                DataCopy(indexLocal, indexGm[indexOffset + head * headIndexSize + offset], this->indexUbSize);
                pipe_barrier(PIPE_ALL);
                Adds(indexLocal, indexLocal, (int32_t)headOutOffset, this->indexUbSize);
                Duplicate(gradInLocal, float(0), this->indexUbSize);
                for (uint64_t idx = 0; idx < this->indexUbSize; idx++) {
                    auto indexValue = indexLocal.GetValue(idx);
                    auto gradInValue = gradOutLocal.GetValue(indexValue);
                    gradInLocal.SetValue(idx, gradInValue);
                }
                SetAtomicAdd<T>();
                DataCopy(gradInGm[indexOffset + head * headIndexSize + offset], gradInLocal, this->indexUbSize);
                SetAtomicNone();
            }
            if (indexLast != 0) {
                offset = this->indexUbSize * indexLoop;
                uint64_t indicesAlign = AlignUp(indexLast, this->indicesEachBlock);
                DataCopy(indexLocal, indexGm[indexOffset + head * headIndexSize + offset], indicesAlign);
                pipe_barrier(PIPE_ALL);
                Adds(indexLocal, indexLocal, (int32_t)headOutOffset, indicesAlign);
                Duplicate(gradInLocal, float(0), indicesAlign);
                for (uint64_t idx = 0; idx < indexLast; idx++) {
                    auto indexValue = indexLocal.GetValue(idx);
                    auto gradInValue = gradOutLocal.GetValue(indexValue);
                    gradInLocal.SetValue(idx, gradInValue);
                }
                SetAtomicAdd<T>();
                DataCopyPad(gradInGm[indexOffset + head * headIndexSize + offset], gradInLocal, this->copyParamsOut);
                SetAtomicNone();
            }
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> inGradOutUb, inIndexUb, outGradInUb;

    GlobalTensor<T> gradInGm, gradOutGm;
    GlobalTensor<int32_t> indexGm;

    uint64_t headOutSize;
    uint64_t headIndexSize;
    uint64_t taskNum;
    uint64_t headTask;
    uint64_t headLastTask;
    uint64_t headBaseId;
    uint64_t indexLoop;
    uint64_t indexLast;
};
}
#endif