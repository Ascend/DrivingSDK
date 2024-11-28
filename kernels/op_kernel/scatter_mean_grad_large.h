/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef _SCATTER_MEAN_GRAD_LARGE_H_
#define _SCATTER_MEAN_GRAD_LARGE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scatter_mean_grad_base.h"
namespace ScatterMeanGradNS {
using namespace AscendC;

template <typename T>
class ScatterMeanGradLarge : public ScatterMeanGradBase<T> {
public:
    __aicore__ inline ScatterMeanGradLarge() {}
    __aicore__ inline void Init(GM_ADDR gradOut, GM_ADDR index, GM_ADDR count, GM_ADDR gradIn, const ScatterMeanGradTilingData* tilingData)
    {
        this->InitTiling(tilingData);
        InitNoTailTiling(tilingData);
        gradInGm.SetGlobalBuffer((__gm__ T *)gradIn, this->gradInNum);
        indexGm.SetGlobalBuffer((__gm__ int32_t *)index, this->indexNum);
        gradOutGm.SetGlobalBuffer((__gm__ T *)gradOut, this->gradOutNum);
        countGm.SetGlobalBuffer((__gm__ T *)count, this->countNum);

        pipe.InitBuffer(inGradOutUb, this->gradOutUbSize * sizeof(T));
        pipe.InitBuffer(inIndexUb, this->indexUbSize * sizeof(int32_t));
        pipe.InitBuffer(outGradInUb, this->indexUbSize * sizeof(T));
        pipe.InitBuffer(inCountUb, this->gradOutUbSize * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint64_t taskId = 0; taskId < taskNum; taskId++) {
            auto taskIdAll = taskId + baseTaskNum;
            uint64_t headPartId = taskIdAll % taskEachHead;
            uint64_t headBaseId = taskIdAll / taskEachHead;
            if (headPartId == taskEachHead - 1) {
                auto lastDealNum = headOutSize % this->gradOutUbSize;
                taskDealNum = lastDealNum == 0 ? this->gradOutUbSize : lastDealNum;
            } else {
                taskDealNum = this->gradOutUbSize;
            }
            ComputeModePart(taskId, taskDealNum, headBaseId, headPartId);
        }
    }

private:
    __aicore__ inline void InitNoTailTiling(const ScatterMeanGradTilingData *tiling_data)
    {
        auto taskNumSmall = tiling_data->taskNumSmall;
        auto taskNumBig = tiling_data->taskNumBig;
        taskEachHead = tiling_data->taskEachHead;
        headOutSize = this->dimRangeOut * this->paramsPro;
        headIndexSize = this->dimRange * this->paramsPro;
        taskDealNum = this->gradOutUbSize;

        if (this->curBlockIdx < this->bigCoreNum) {
            taskNum = taskNumBig;
            baseTaskNum = this->curBlockIdx * taskNum;
        } else {
            taskNum = taskNumSmall;
            baseTaskNum = this->bigCoreNum * taskNumBig + (this->curBlockIdx - this->bigCoreNum) * taskNum;
        }

        indexLoop = headIndexSize / this->indexUbSize;
        indexLast = headIndexSize - indexLoop * this->indexUbSize;

        this->copyParamsOut.blockLen = static_cast<uint32_t>(indexLast * sizeof(float));
    }

    __aicore__ inline void ComputeModePart(uint64_t taskId, uint64_t taskDealNum, uint64_t headBaseId, uint64_t headPartId)
    {
        LocalTensor<int32_t> indexLocal = inIndexUb.Get<int32_t>();
        LocalTensor<T> gradOutLocal = inGradOutUb.Get<T>();
        LocalTensor<T> countLocal = inCountUb.Get<T>();
        LocalTensor<T> gradInLocal = outGradInUb.Get<T>();

        uint64_t indexOffset = headBaseId * headIndexSize;
        uint64_t outOffset = headBaseId * headOutSize + headPartId * this->gradOutUbSize;
        uint64_t outAlign = AlignUp(taskDealNum, this->paramsEachBlock);
        auto baseOutOffset = headPartId * this->gradOutUbSize;

        DataCopy(gradOutLocal, gradOutGm[outOffset], outAlign);
        DataCopy(countLocal, countGm[outOffset], outAlign);

        pipe_barrier(PIPE_ALL);
        Div(gradOutLocal, gradOutLocal, countLocal, outAlign);

        uint64_t offset = 0;
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        for (uint64_t loop = 0; loop < indexLoop; loop++) {
            offset = loop * this->indexUbSize;
            DataCopy(indexLocal, indexGm[indexOffset + offset], this->indexUbSize);

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            Duplicate(gradInLocal, float(0), this->indexUbSize);
            for (uint64_t idx = 0; idx < this->indexUbSize; idx++) {
                auto indexValue = indexLocal.GetValue(idx);
                if (indexValue >= baseOutOffset && indexValue < baseOutOffset + taskDealNum) {
                    auto gradOutValue = gradOutLocal.GetValue(indexValue - baseOutOffset);
                    gradInLocal.SetValue(idx, gradOutValue);
                }
            }
            SetAtomicAdd<T>();
            DataCopy(gradInGm[indexOffset + offset], gradInLocal, this->indexUbSize);
            SetAtomicNone();
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
        if (indexLast != 0) {
            offset = indexLoop * this->indexUbSize;
            uint64_t indicesAlign = AlignUp(indexLast, this->indicesEachBlock);
            DataCopy(indexLocal, indexGm[indexOffset + offset], indicesAlign);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            Duplicate(gradInLocal, float(0), indicesAlign);
            for (uint64_t idx = 0; idx < indexLast; idx++) {
                auto indexValue = indexLocal.GetValue(idx);
                if (indexValue >= baseOutOffset && indexValue < baseOutOffset + taskDealNum) {
                    auto gradOutValue = gradOutLocal.GetValue(indexValue - baseOutOffset);
                    gradInLocal.SetValue(idx, gradOutValue);
                }
            }
            SetAtomicAdd<T>();
            DataCopyPad(gradInGm[indexOffset + offset], gradInLocal, this->copyParamsOut);
            SetAtomicNone();
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> inGradOutUb, inIndexUb, outGradInUb, inCountUb;

    GlobalTensor<T> gradInGm, gradOutGm, countGm;
    GlobalTensor<int32_t> indexGm;

    uint64_t headOutSize;
    uint64_t headIndexSize;
    uint64_t taskNum;
    uint64_t headTask;
    uint64_t headLastTask;
    uint64_t headBaseId;
    uint64_t baseTaskNum;
    uint64_t taskDealNum;
    uint64_t indexLoop;
    uint64_t indexLast;
    uint64_t taskEachHead;
};
}
#endif