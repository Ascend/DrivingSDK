/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef _SCATTER_ADD_GRAD_LARGE_H_
#define _SCATTER_ADD_GRAD_LARGE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scatter_add_grad_base.h"
namespace ScatterAddGradNS {
using namespace AscendC;

template <typename T>
class ScatterAddGradLarge : public ScatterAddGradBase<T> {
public:
    __aicore__ inline ScatterAddGradLarge() {}
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
        pipe.InitBuffer(validMaskUb, 2 * validMaskLen * sizeof(uint8_t)); // 0 <= index < taskNum
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
    __aicore__ inline void InitNoTailTiling(const ScatterAddGradTilingData *tiling_data)
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

    __aicore__ inline void ComputeSingleLoop(const LocalTensor<int32_t>& indexLocal, const LocalTensor<T>& gradOutLocal, 
        const LocalTensor<T>& gradInLocal, const LocalTensor<uint8_t>& validMask, uint64_t offset, uint32_t totalNum,
        uint64_t indexOffset, uint32_t baseOutOffset, uint64_t taskDealNum)
    {
        uint32_t outerLoops = DivCeil(totalNum, 64);
        uint32_t totalNumAligned = AlignUp(totalNum, 64);
        float taskDealNumFloat = static_cast<float>(int32_t(taskDealNum));
        WaitFlag<HardEvent::V_MTE2>(0);
        DataCopy(indexLocal, indexGm[indexOffset + offset], totalNumAligned);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        Adds(indexLocal, indexLocal, -1 * (int32_t)baseOutOffset, totalNumAligned);
        WaitFlag<HardEvent::MTE3_V>(0);
        Cast(gradInLocal, indexLocal, RoundMode::CAST_NONE, totalNumAligned);
        CompareScalar(validMask, gradInLocal, (float)0, CMPMODE::GE, totalNumAligned);
        CompareScalar(validMask[validMaskLen], gradInLocal, taskDealNumFloat, CMPMODE::LT, totalNumAligned);
        And(validMask.ReinterpretCast<uint16_t>(), validMask.ReinterpretCast<uint16_t>(), validMask[validMaskLen].ReinterpretCast<uint16_t>(), validMaskLen / 2); //uint8 -> uint16, length / 2
        Select(indexLocal.ReinterpretCast<float>(), validMask, indexLocal.ReinterpretCast<float>(), 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, B32_DATA_NUM_PER_REPEAT, outerLoops, {1, 1, 1, 8, 8, 8});
        Muls(indexLocal, indexLocal, (int32_t)sizeof(T), totalNumAligned);
        Gather(gradInLocal, gradOutLocal, indexLocal.ReinterpretCast<uint32_t>(), (uint32_t)0, totalNum);
        SetFlag<HardEvent::V_MTE2>(0);
        Select(gradInLocal, validMask, gradInLocal, (T)0, SELMODE::VSEL_TENSOR_SCALAR_MODE, B32_DATA_NUM_PER_REPEAT, outerLoops, {1, 1, 1, 8, 8, 8});
        SetFlag<HardEvent::V_MTE3>(0);
    }

    __aicore__ inline void ComputeModePart(uint64_t taskId, uint64_t taskDealNum, uint64_t headBaseId, uint64_t headPartId)
    {
        LocalTensor<int32_t> indexLocal = inIndexUb.Get<int32_t>();
        LocalTensor<T> gradOutLocal = inGradOutUb.Get<T>();
        LocalTensor<T> gradInLocal = outGradInUb.Get<T>();
        LocalTensor<uint8_t> validMask = validMaskUb.Get<uint8_t>();

        uint64_t indexOffset = headBaseId * headIndexSize;
        uint64_t outOffset = headBaseId * headOutSize + headPartId * this->gradOutUbSize;
        uint64_t outAlign = AlignUp(taskDealNum, this->paramsEachBlock);
        uint32_t baseOutOffset = headPartId * this->gradOutUbSize;

        pipe_barrier(PIPE_ALL);
        DataCopy(gradOutLocal, gradOutGm[outOffset], outAlign);
        pipe_barrier(PIPE_ALL);

        SetFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::V_MTE2>(0);
        for (uint64_t loop = 0; loop < indexLoop; loop++) {
            uint64_t offset = loop * this->indexUbSize;
            ComputeSingleLoop(indexLocal, gradOutLocal, gradInLocal, validMask, offset, this->indexUbSize, indexOffset, baseOutOffset, taskDealNum);
            WaitFlag<HardEvent::V_MTE3>(0);
            SetAtomicAdd<T>();
            DataCopy(gradInGm[indexOffset + offset], gradInLocal, this->indexUbSize);
            SetAtomicNone();
            SetFlag<HardEvent::MTE3_V>(0);
        }
        if (indexLast != 0) {
            uint64_t offset = indexLoop * this->indexUbSize;
            ComputeSingleLoop(indexLocal, gradOutLocal, gradInLocal, validMask, offset, indexLast, indexOffset, baseOutOffset, taskDealNum);
            WaitFlag<HardEvent::V_MTE3>(0);
            SetAtomicAdd<T>();
            DataCopyPad(gradInGm[indexOffset + offset], gradInLocal, this->copyParamsOut);
            SetAtomicNone();
            SetFlag<HardEvent::MTE3_V>(0);
        }
        WaitFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> inGradOutUb, inIndexUb, outGradInUb, validMaskUb;

    GlobalTensor<T> gradInGm, gradOutGm;
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
    uint64_t validMaskLen = 4096;
};
}
#endif