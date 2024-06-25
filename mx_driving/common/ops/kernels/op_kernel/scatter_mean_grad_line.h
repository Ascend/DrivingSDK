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
    __aicore__ inline void InitLine(GM_ADDR gradOut, GM_ADDR index, GM_ADDR gradIn, const ScatterMeanGradTilingData* tilingData)
    {
        this->InitTiling(tilingData);
        gradInGm.SetGlobalBuffer((__gm__ T *)gradIn, this->gradInNum);
        indexGm.SetGlobalBuffer((__gm__ int32_t *)index, this->indexNum);
        gradOutGm.SetGlobalBuffer((__gm__ T *)gradOut, this->gradOutNum);

        pipe.InitBuffer(inGradOutUb, 1, this->gradOutUbSize * sizeof(T));
        pipe.InitBuffer(inIndexUb, 1, this->indexUbSize * sizeof(int32_t));
        pipe.InitBuffer(outGradInUb, 1, this->gradInUbSize * sizeof(T));
        pipe.InitBuffer(indexSumUb, this->indexSumUbSize * sizeof(int32_t));
        pipe.InitBuffer(maskUb, this->CeilValue((this->indexSumUbSize - 1) / this->indexNumPerBlock + 1, BLOCK_BYTES));
        pipe.InitBuffer(duplicateUb, MASK_BYTES);
        pipe.InitBuffer(compareUb, MASK_BYTES);
        pipe.InitBuffer(indexDivUb, this->indexSumUbSize * sizeof(int32_t));
        pipe.InitBuffer(allOneUb, this->indexSumUbSize * sizeof(int32_t));
    }

    __aicore__ inline void ComputeModeLastDimLine(int32_t curBlockIdx)
    {
        LocalTensor<T> gradInLocal = outGradInUb.DeQue<T>();
        LocalTensor<int32_t> indexLocal = inIndexUb.DeQue<int32_t>();
        LocalTensor<T> gradOutLocal = inGradOutUb.DeQue<T>();
        LocalTensor<int32_t> indexSumLocal = indexSumUb.Get<int32_t>();
        LocalTensor<uint8_t> maskLocal = maskUb.Get<uint8_t>();
        LocalTensor<int32_t> compareLocal = compareUb.Get<int32_t>();
        LocalTensor<float> duplicateLocal = duplicateUb.Get<float>();
        LocalTensor<float> indexDivLocal = indexDivUb.Get<float>();
        LocalTensor<float> allOneLocal = allOneUb.Get<float>();

        uint32_t repeat = this->CeilValue(this->dimRangeOut, MASK) / MASK;
        uint32_t copyInBytes = (uint32_t)(this->paramsPro * sizeof(T));
        DataCopyExtParams copyParams{1, copyInBytes, 0, 0, 0};

        Duplicate(compareLocal, (int32_t)0, MASK);
        Duplicate(duplicateLocal, (float)1, MASK);
        Duplicate(allOneLocal, (float)1, this->indexSumUbSize);
        pipe_barrier(PIPE_V);

        uint32_t lowerLimit = curBlockIdx < this->taskTailCore ? curBlockIdx * (this->taskPerCore + 1) : curBlockIdx * this->taskPerCore + this->taskTailCore;
        uint32_t upperLimit = curBlockIdx < this->taskTailCore ? (curBlockIdx + 1) * (this->taskPerCore + 1) : (curBlockIdx + 1) * this->taskPerCore + this->taskTailCore;
        for (uint32_t preId = 0; preId < this->paramsPre; preId++) {
            // Zero operation for indexSum
            Duplicate(indexSumLocal, (int32_t)0, this->indexSumUbSize);
            DataCopy(indexLocal, indexGm[preId * this->dimRange],  this->CeilValue(this->dimRange, this->indexNumPerBlock));
            pipe_barrier(PIPE_ALL);
            // Count the occurrence of each index.
            for (uint32_t indexId = 0; indexId < this->dimRange; indexId++) {
                uint32_t indexValue = indexLocal.GetValue(indexId);
                int32_t preValue = indexSumLocal.GetValue(indexValue);
                indexSumLocal.SetValue(indexValue, preValue + 1);
            }
            set_flag(PIPE_S, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
            // Fill with 1 and then divide by line.
            Compare(maskLocal, indexSumLocal, compareLocal, CMPMODE::EQ, MASK, repeat, this->repeatParamsCompare);
            Cast(indexDivLocal, indexSumLocal, RoundMode::CAST_NONE, this->dimRangeOut);
            pipe_barrier(PIPE_V);
            Select(indexDivLocal, maskLocal, duplicateLocal, indexDivLocal,
                   SELMODE::VSEL_TENSOR_TENSOR_MODE, MASK, repeat, this->repeatParamsSelect);
            pipe_barrier(PIPE_V);
            Div(indexDivLocal, allOneLocal, indexDivLocal, this->indexSumUbSize);
            set_flag(PIPE_V, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
            for (uint32_t indexId = 0; indexId < this->dimRange; indexId++) {
                uint32_t indexValue = indexLocal.GetValue(indexId);
                if (indexValue >= lowerLimit && indexValue < upperLimit) {
                    T outIndexValue = (T)indexDivLocal.GetValue(indexValue);
                    DataCopy(gradOutLocal,
                             gradOutGm[preId * this->dimRangeOut * this->paramsPro + indexValue * this->paramsPro],
                             this->CeilValue(this->paramsPro, this->paramsNumPerBlock));
                    pipe_barrier(PIPE_ALL);
                    Muls(gradInLocal, gradOutLocal, outIndexValue, this->paramsPro);
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    DataCopyPad(gradInGm[preId * this->dimRange * this->paramsPro + indexId * this->paramsPro], gradInLocal, copyParams);
                }
            }
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

    __aicore__ inline void Process()
    {
        InitUb();
        ComputeModeLastDimLine(this->curBlockIdx);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inGradOutUb, inIndexUb;
    TQue<QuePosition::VECOUT, 1> outGradInUb;
    TBuf<TPosition::VECCALC> indexSumUb, indexDivUb, maskUb, compareUb, duplicateUb, allOneUb;
    GlobalTensor<T> gradInGm, gradOutGm;
    GlobalTensor<int32_t> indexGm;
};
}
#endif