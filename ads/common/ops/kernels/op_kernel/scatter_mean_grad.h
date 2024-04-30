/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef _SCATTER_MEAN_GRAD_H_
#define _SCATTER_MEAN_GRAD_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "scatter_mean_grad_base.h"
namespace ScatterMeanGradNS {
using namespace AscendC;

template <typename T>
class ScatterMeanGrad : public ScatterMeanGradBase<T> {
public:
    __aicore__ inline ScatterMeanGrad() {}
    __aicore__ inline void ComputeModeSmallData(int32_t curBlockIdx)
    {
        LocalTensor<T> gradInLocal = outGradInUb.DeQue<T>();
        LocalTensor<int32_t> indexLocal = inIndexUb.DeQue<int32_t>();
        LocalTensor<T> gradOutLocal = inGradOutUb.DeQue<T>();
        LocalTensor<int32_t> indexSumLocal = indexSumUb.Get<int32_t>();
        LocalTensor<uint8_t> maskLocal = maskUb.Get<uint8_t>();
        LocalTensor<int32_t> compareLocal = compareUb.Get<int32_t>();
        LocalTensor<float> duplicateLocal = duplicateUb.Get<float>();
        LocalTensor<float> indexDivLocal = indexDivUb.Get<float>();
        
        uint32_t copyInNum = this->dimRange * this->paramsPro;
        uint32_t copyOutNum = this->dimRangeOut * this->paramsPro;
        uint32_t repeat = this->CeilValue(copyOutNum, MASK) / MASK;
        uint32_t indexInBlock = this->CeilValue(copyInNum, BLOCK_BYTES / sizeof(int32_t));
        uint32_t gradOutBlock = this->CeilValue(copyOutNum, BLOCK_BYTES / sizeof(T));
        uint32_t copyInBytes = (uint32_t)(copyInNum * sizeof(T));
        DataCopyExtParams copyParams{1, copyInBytes, 0, 0, 0};

        Duplicate(compareLocal, (int32_t)0, MASK);
        Duplicate(duplicateLocal, (float)1, MASK);
        pipe_barrier(PIPE_V);

        uint32_t curCoreTaskNum = curBlockIdx < this->taskTailCore ? this->taskPerCore + 1 : this->taskPerCore;
        uint32_t startTaskid = curBlockIdx < this->taskTailCore ? (this->taskPerCore + 1) * curBlockIdx : this->taskPerCore * curBlockIdx + this->taskTailCore;
        for (uint32_t loopIndex = 0; loopIndex < curCoreTaskNum; loopIndex++) {
            int32_t curTaskIdx = loopIndex + startTaskid;
            // Zero operation for indexSum
            Duplicate(indexSumLocal, (int32_t)0, this->gradOutUbSize);
            DataCopy(indexLocal, indexGm[curTaskIdx * copyInNum], indexInBlock);
            DataCopy(gradOutLocal, gradOutGm[curTaskIdx * copyOutNum], gradOutBlock);
            pipe_barrier(PIPE_ALL);
            // Count the occurrence of each index.
            for (uint32_t indexId = 0; indexId < copyInNum; indexId++) {
                uint32_t index1 = indexId % this->paramsPro;
                int32_t indexValue = indexLocal.GetValue(indexId);
                int32_t outOffset = indexValue * this->paramsPro + index1;
                int32_t preValue = indexSumLocal.GetValue(outOffset);
                indexSumLocal.SetValue(outOffset, preValue + 1);
                indexLocal.SetValue(indexId, outOffset);
            }
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            // Fill with 1 and then divide.
            Compare(maskLocal, indexSumLocal, compareLocal, CMPMODE::EQ, MASK, repeat, this->repeatParamsCompare);
            Cast(indexDivLocal, indexSumLocal, RoundMode::CAST_NONE, copyOutNum);
            pipe_barrier(PIPE_V);
            Select(indexDivLocal, maskLocal, duplicateLocal, indexDivLocal,
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, MASK, repeat, this->repeatParamsSelect);
            pipe_barrier(PIPE_V);
            Div(gradOutLocal, gradOutLocal, indexDivLocal, this->gradOutUbSize);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);

            for (uint32_t indexId = 0; indexId < copyInNum; indexId++) {
                int32_t outOffset = indexLocal.GetValue(indexId);
                T grad = gradOutLocal.GetValue(outOffset);
                gradInLocal.SetValue(indexId, grad);
            }
            set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
            DataCopyPad(gradInGm[curTaskIdx * copyInNum], gradInLocal, copyParams);
        }
        outGradInUb.FreeTensor(gradInLocal);
        inIndexUb.FreeTensor(indexLocal);
        inGradOutUb.FreeTensor(gradOutLocal);
    }

    __aicore__ inline void InitUb()
    {
        LocalTensor<T> gradInLocal = outGradInUb.AllocTensor<T>();
        LocalTensor<T> gradOutLocal = inGradOutUb.AllocTensor<T>();
        LocalTensor<int32_t> indexLocal = inIndexUb.AllocTensor<int32_t>();
        inGradOutUb.EnQue(gradOutLocal);
        inIndexUb.EnQue(indexLocal);
        outGradInUb.EnQue(gradInLocal);
    }

    __aicore__ inline void Init(GM_ADDR gradOut, GM_ADDR index, GM_ADDR gradIn, const ScatterMeanGradTilingData* tilingData)
    {
        this->InitTiling(tilingData);
        gradInGm.SetGlobalBuffer((__gm__ T *)gradIn, this->gradInNum);
        indexGm.SetGlobalBuffer((__gm__ int32_t *)index, this->indexNum);
        gradOutGm.SetGlobalBuffer((__gm__ T *)gradOut, this->gradOutNum);

        pipe.InitBuffer(inGradOutUb, 1, this->gradOutUbSize * sizeof(T));
        pipe.InitBuffer(inIndexUb, 1, this->indexUbSize * sizeof(int32_t));
        pipe.InitBuffer(outGradInUb, 1, this->gradInUbSize * sizeof(T));
        pipe.InitBuffer(indexSumUb, this->indexSumUbSize * sizeof(int32_t));
        pipe.InitBuffer(maskUb, this->CeilValue((this->gradOutUbSize - 1) / this->indexNumPerBlock + 1, BLOCK_BYTES));
        pipe.InitBuffer(duplicateUb, MASK_BYTES);
        pipe.InitBuffer(compareUb, MASK_BYTES);
        pipe.InitBuffer(indexDivUb, this->indexSumUbSize * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        InitUb();
        ComputeModeSmallData(this->curBlockIdx);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inGradOutUb, inIndexUb;
    TQue<QuePosition::VECOUT, 1> outGradInUb;
    TBuf<TPosition::VECCALC> indexSumUb, indexDivUb, maskUb, compareUb, duplicateUb;
    GlobalTensor<T> gradInGm, gradOutGm;
    GlobalTensor<int32_t> indexGm;
};
}
#endif