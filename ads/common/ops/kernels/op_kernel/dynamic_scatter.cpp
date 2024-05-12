/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
 */
#include <cmath>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class DynamicScatter {
public:
    __aicore__ inline DynamicScatter() {}
    __aicore__ inline void Init(GM_ADDR feats, GM_ADDR coorsMap, GM_ADDR reducedFeats,
                                DynamicScatterTilingData* tilingData)
    {
        featsNum = tilingData->featsNum;
        pointNum = tilingData->pointNum;
        coreNum = tilingData->coreNum;
        outNum = tilingData->outNum;
        reduceMode = tilingData->reduceMode;

        taskPerCore = DivCeil(pointNum, coreNum);
        featsNumAlign = (featsNum + alignNum - 1) / alignNum * alignNum;
        pointNumAlign = (pointNum + alignNum - 1) / alignNum * alignNum;

        copyParamsOut.blockCount = 1;
        copyParamsOut.blockLen = static_cast<uint32_t>(featsNum * sizeof(T));
        copyParamsOut.srcStride = 0;
        copyParamsOut.dstStride = 0;
        copyParamsOut.rsv = 0;
 
        featsGm.SetGlobalBuffer((__gm__ T *)feats, pointNum * featsNum);
        coorsMapGm.SetGlobalBuffer((__gm__ int32_t *)coorsMap, pointNum);
        reducedFeatsGm.SetGlobalBuffer((__gm__ T *)reducedFeats, outNum * featsNum);

        pipe.InitBuffer(featsQueue, featsNumAlign * sizeof(T));
        pipe.InitBuffer(coorsQueue, pointNumAlign * sizeof(int32_t));
 
        curBlockIdx = GetBlockIdx();
        startOffset = curBlockIdx * taskPerCore;
        endOffset = (curBlockIdx + 1) * taskPerCore;

        if (endOffset > pointNum) {
            endOffset = pointNum;
        }
        if (curBlockIdx == 0 && reduceMode == 0) {
            InitOutput<T>(reducedFeatsGm, outNum * featsNum, static_cast<T>(-INFINITY));
        }
        SyncAll();
    }

    __aicore__ inline void Process()
    {
        // alloc tensor from queue memory
        coorsLocal = coorsQueue.Get<int32_t>();

        // loop count need to be doubled, due to double buffer
        DataCopy(coorsLocal, coorsMapGm, pointNumAlign);
        for (int32_t taskNum = startOffset; taskNum < endOffset; taskNum++) {
            Compute(taskNum);
        }
    }

private:
    __aicore__ inline void Compute(int32_t taskNum)
    {
        featsLocal = featsQueue.Get<T>();
        DataCopy(featsLocal, featsGm[taskNum * featsNum], featsNumAlign);

        outOffset = coorsLocal.GetValue(taskNum) * featsNum;
        if (reduceMode == 0) {
            SetAtomicMax<T>();
        } else {
            SetAtomicAdd<T>();
        }
        pipe_barrier(PIPE_ALL);
        DataCopyPad(reducedFeatsGm[outOffset], featsLocal, copyParamsOut);
        SetAtomicNone();
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> featsQueue, coorsQueue;
    LocalTensor<int32_t> coorsLocal;
    LocalTensor<T> featsLocal;

    GlobalTensor<T> featsGm, reducedFeatsGm;
    GlobalTensor<int32_t> coorsMapGm;

    uint32_t featsNum, pointNum, outNum, coreNum, reduceMode, taskPerCore, currentTaskNum;
    DataCopyExtParams copyParamsOut;
    uint32_t featsNumAlign, pointNumAlign, alignNum = 8;
    uint32_t startOffset, endOffset, curBlockIdx, outOffset;
};


extern "C" __global__ __aicore__ void dynamic_scatter(GM_ADDR feats, GM_ADDR coors_map, GM_ADDR reduced_feats,
                                                      GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    DynamicScatter<float> op;
    op.Init(feats, coors_map, reduced_feats, &tilingData);
    op.Process();
}
