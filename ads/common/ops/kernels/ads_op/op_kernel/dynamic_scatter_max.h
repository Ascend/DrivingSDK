/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef _DYNAMIC_SCATTER_MAX_H_
#define _DYNAMIC_SCATTER_MAX_H_

#include "dynamic_scatter_base.h"

namespace DynamicScatterN {
using namespace AscendC;

template <typename T>
class DynamicScatterMax : public DynamicScatterBase<T> {
public:
    __aicore__ inline DynamicScatterMax() {}
    __aicore__ inline void Init(GM_ADDR feats, GM_ADDR coorsMap, GM_ADDR reducedFeats,
                                DynamicScatterTilingData* tilingData)
    {
        this->BaseInit(tilingData);
        BufferInit();
        featsGm.SetGlobalBuffer((__gm__ T *)feats + this->featsOffset, this->blockLength);
        coorsMapGm.SetGlobalBuffer((__gm__ int32_t *)coorsMap + this->coorsMapOffset, this->inputNum);
        reducedFeatsGm.SetGlobalBuffer((__gm__ T *)reducedFeats, this->outLength);
        if (GetBlockIdx() == 0) {
            InitOutput<T>(this->reducedFeatsGm, this->outLength, static_cast<T>(-INFINITY));
        }
        SyncAll();
    }

    __aicore__ inline void Process()
    {
        // loop count need to be doubled, due to double buffer
        for (int32_t i = 0; i < this->loop; i++) {
            CopyIn(i * this->tilePointNum);
            pipe_barrier(PIPE_ALL);
            Compute(i * this->tileLength, this->tilePointNum);
            pipe_barrier(PIPE_ALL);
        }

        if (this->lastLength) {
            CopyInTail(this->mapLastStartIndex);
            pipe_barrier(PIPE_ALL);
            ComputeTail(this->lastPointNum);
        }
    }

private:
    __aicore__ inline void BufferInit()
    {
        pipe.InitBuffer(inQueueFeats, BUFFER_NUM, this->featsAligned * sizeof(T));
        pipe.InitBuffer(inQueueCoorsMap, BUFFER_NUM, this->tilePointNum * sizeof(int32_t));
    }

    __aicore__ inline void CopyIn(int32_t startIndex)
    {
        // alloc tensor from queue memory
        LocalTensor<int32_t> coorsMapLocal = inQueueCoorsMap.AllocTensor<int32_t>();
        // copy progress_th tile from global tensor to local tensor
        DataCopy(coorsMapLocal, this->coorsMapGm[startIndex], this->tilePointNum);
        // enque input tensors to VECIN queue
        inQueueCoorsMap.EnQue(coorsMapLocal);
    }

    __aicore__ inline void Compute(int32_t gmOffset, int32_t pointNum)
    {
        // deque input tensors from VECIN queue
        LocalTensor<T> featsLocal = inQueueFeats.AllocTensor<T>();
        LocalTensor<int32_t> coorsMapLocal = inQueueCoorsMap.DeQue<int32_t>();

        for (uint32_t idx = 0; idx < pointNum; idx++) {
            int32_t reduce_to = coorsMapLocal.GetValue(idx);
            if (reduce_to > -1) {
                DataCopy(featsLocal, this->featsGm[gmOffset + idx * this->featsNum], this->featsAligned);
                pipe_barrier(PIPE_ALL);
                this->CopyOutMax(reducedFeatsGm, reduce_to * this->featsNum, featsLocal);
            }
        }
        // free input tensors for reuse
        inQueueFeats.FreeTensor(featsLocal);
        inQueueCoorsMap.FreeTensor(coorsMapLocal);
    }

    __aicore__ inline void CopyInTail(int32_t mapLastStartIndex)
    {
        // alloc tensor from queue memory
        LocalTensor<int32_t> coorsMapLocal = inQueueCoorsMap.AllocTensor<int32_t>();
        // copy progress_th tile from global tensor to local tensor
        DataCopy(coorsMapLocal, this->coorsMapGm[mapLastStartIndex], this->tilePointNum);
        // enque input tensors to VECIN queue
        inQueueCoorsMap.EnQue(coorsMapLocal);
    }

    __aicore__ inline void ComputeTail(int32_t pointNum)
    {
        // deque input tensors from VECIN queue
        LocalTensor<T> featsLocal = inQueueFeats.AllocTensor<T>();
        LocalTensor<int32_t> coorsMapLocal = inQueueCoorsMap.DeQue<int32_t>();

        for (uint32_t idx = 0; idx < pointNum; idx++) {
            int32_t reduce_to = coorsMapLocal.GetValue(idx);
            if (reduce_to > -1) {
                DataCopy(featsLocal, this->featsGm[this->featsLastStartIndex + idx * this->featsNum],
                         this->featsAligned);
                pipe_barrier(PIPE_ALL);
                this->CopyOutMax(reducedFeatsGm, reduce_to * this->featsNum, featsLocal);
            }
        }
        // free input tensors for reuse
        inQueueFeats.FreeTensor(featsLocal);
        inQueueCoorsMap.FreeTensor(coorsMapLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueFeats, inQueueCoorsMap, inQueueReduceCount;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueReducedFeats;

    GlobalTensor<T> featsGm, reducedFeatsGm;
    GlobalTensor<int32_t> coorsMapGm, reduceCountGm;
};
} // DynamicScatterN
#endif  // _DYNAMIC_SCATTER_MAX_H_