/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#include "kernel_operator.h"
using namespace AscendC;

namespace {
constexpr static int32_t BUFFER_NUM = 1;
};

class KernelSparseConv3d {
public:
    __aicore__ inline KernelSparseConv3d() {}
    __aicore__ inline void Init(GM_ADDR indices, GM_ADDR indices_out, GM_ADDR indices_pair, GM_ADDR workspace, SparseConv3dTilingData *tiling_data, TPipe *tmpPipe)
    {
        pipe = tmpPipe;
        curBlockIdx = GetBlockIdx();
        // features dtype must be same with weight
        initTilingData(tiling_data);

        uint64_t beginOffset = curBlockIdx * coreTask;

        if (curBlockIdx < usedCoreNum - 1) {
            taskNum = coreTask;
            coreRepeatTimes = repeatTimes;
            coreMoveTail = moveTail;
        } else {
            taskNum = lastCoreTask;
            coreRepeatTimes = lastRepeatTimes;
            coreMoveTail = lastMoveTail;
        }

        indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES *>(indices) + beginOffset * 4);

        outputIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES *>(indices_out) + beginOffset * kernelSize);
        outputIndicesPairGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES *>(indices_pair) + beginOffset * kernelSize * 4);

        pipe->InitBuffer(indicesQueue, BUFFER_NUM, moveLen * 4 * sizeof(DTYPE_INDICES));
        pipe->InitBuffer(outIndicesUB, moveLen * kernelSize * sizeof(DTYPE_INDICES));
        pipe->InitBuffer(outIndicesPairUB, moveLen * kernelSize * 4 * sizeof(DTYPE_INDICES));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < coreRepeatTimes; i++) {
            Compute(i);
            pipe_barrier(PIPE_ALL);
        }
    }

private:

    __aicore__ inline void initTilingData(SparseConv3dTilingData *tiling_data)
    {
        usedCoreNum = tiling_data->usedCoreNum;
        coreTask = tiling_data->coreTask;
        lastCoreTask = tiling_data->lastCoreTask;

        moveLen = tiling_data->moveLen;

        repeatTimes = tiling_data->repeatTimes;
        moveTail = tiling_data->moveTail;
        lastRepeatTimes = tiling_data->lastRepeatTimes;
        lastMoveTail = tiling_data->lastMoveTail;

        kernelD = tiling_data-> kernelD;
        kernelH = tiling_data->kernelH;
        kernelW = tiling_data->kernelW;
        kernelSize = tiling_data->kernelSize;

        outfeatureB = tiling_data->outfeatureB;
        outputDepth = tiling_data->outputDepth;
        outputHeight = tiling_data->outputHeight;
        outputWidth = tiling_data->outputWidth;

        strideDepth = tiling_data->strideDepth;
        strideHeight = tiling_data->strideHeight;
        strideWidth = tiling_data->strideWidth;

        paddingDepth = tiling_data->paddingDepth;
        paddingHeight = tiling_data->paddingHeight;
        paddingWidth = tiling_data->paddingWidth;
    }

    __aicore__ inline void Compute(uint32_t query)
    {
        uint32_t taskOffset = query * moveLen;
        uint32_t forMoveLen = moveLen;
        if (query == coreRepeatTimes - 1) {
            forMoveLen = coreMoveTail;
        }

        DataCopyExtParams indicesCopyParams {1, (uint32_t)(forMoveLen * 4 * sizeof(DTYPE_INDICES)), 0, 0, 0};

        DataCopyExtParams outIndicesCopyParams {1, (uint32_t)(forMoveLen * kernelSize * sizeof(DTYPE_INDICES)), 0, 0, 0};
        DataCopyExtParams outPairCopyParams {1, (uint32_t)(forMoveLen * kernelSize * 4 * sizeof(DTYPE_INDICES)), 0, 0, 0};

        DataCopyPadExtParams<DTYPE_INDICES> indicesPadParams{true, 0, 0, 0};

        LocalTensor<DTYPE_INDICES> indicesLocal = indicesQueue.AllocTensor<DTYPE_INDICES>();
        LocalTensor<DTYPE_INDICES> outIndicesTemp = outIndicesUB.Get<DTYPE_INDICES>();
        LocalTensor<DTYPE_INDICES> outIndicesPairTemp = outIndicesPairUB.Get<DTYPE_INDICES>();

        DTYPE_INDICES onesVal = -1;
        Duplicate<DTYPE_INDICES>(outIndicesTemp, onesVal, moveLen * kernelSize);

        event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));

        SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
        WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
        DataCopyPad(indicesLocal, indicesGm[taskOffset * 4], indicesCopyParams, indicesPadParams);
        pipe_barrier(PIPE_MTE2);

        for (uint32_t i = 0; i < forMoveLen; i++) {
            // GetValue feature's locations
            int32_t idxOffset = i * 4;
            int32_t featureB = indicesLocal.GetValue(idxOffset);
            int32_t featureD = indicesLocal.GetValue(idxOffset + 1) + paddingDepth;
            int32_t featureH = indicesLocal.GetValue(idxOffset + 2) + paddingHeight;
            int32_t featureW = indicesLocal.GetValue(idxOffset + 3) + paddingWidth;
            int32_t bOffset = featureB * outputDepth * outputHeight * outputWidth;
            // Calculate the features of this position that affect the positions of the output
            int32_t startD = Max(featureD - kernelD + 1, 0);
            int32_t startH = Max(featureH - kernelH + 1, 0);
            int32_t startW = Max(featureW - kernelW + 1, 0);
            int32_t outBeginD = AlignUp(startD, strideDepth) / strideDepth;
            int32_t outBeginH = AlignUp(startH, strideHeight) / strideHeight;
            int32_t outBeginW = AlignUp(startW, strideWidth) / strideWidth;
            int32_t outEndD = Min(AlignUp(featureD + 1, strideDepth) / strideDepth, outputDepth);
            int32_t outEndH = Min(AlignUp(featureH + 1, strideHeight) / strideHeight, outputHeight);
            int32_t outEndW = Min(AlignUp(featureW + 1, strideWidth) / strideWidth, outputWidth);

            pipe_barrier(PIPE_ALL);
            for (int32_t ix = outBeginD; ix < outEndD; ix++) {
                uint32_t xOffset = (uint32_t)ix * outputHeight * outputWidth;
                for (int32_t iy = outBeginH; iy < outEndH; iy++) {
                    uint32_t yOffset = (uint32_t)iy * outputWidth;
                    for (int32_t iz = outBeginW; iz < outEndW; iz++) {
                        uint32_t zOffset = (uint32_t)iz;
                        uint32_t gmOutValueOffset = bOffset + xOffset + yOffset + zOffset;
                        uint32_t weightD = featureD - ix * strideDepth;
                        uint32_t weightH = featureH - iy * strideHeight;
                        uint32_t weightW = featureW - iz * strideWidth;
                        uint32_t convOffset = weightD * kernelH * kernelW + weightH * kernelW + weightW;
                        uint32_t weightOffset = convOffset;
                        int64_t outFeatureOffset = (taskOffset + i) * kernelSize + convOffset;
                        int64_t outInidcesOffset = i * kernelSize + convOffset;
                        int64_t outInidcesPairOffset = (i * kernelSize + convOffset) * 4;
                        outIndicesTemp.SetValue(outInidcesOffset, gmOutValueOffset);
                        outIndicesPairTemp.SetValue(outInidcesPairOffset, featureB);
                        outIndicesPairTemp.SetValue(outInidcesPairOffset + 1, ix);
                        outIndicesPairTemp.SetValue(outInidcesPairOffset + 2, iy);
                        outIndicesPairTemp.SetValue(outInidcesPairOffset + 3, iz);
                    }
                }
            }
            pipe_barrier(PIPE_ALL);
        }
        DataCopyPad(outputIndicesGm[taskOffset * kernelSize], outIndicesTemp, outIndicesCopyParams);
        DataCopyPad(outputIndicesPairGm[taskOffset * kernelSize * 4], outIndicesPairTemp, outPairCopyParams);
        indicesQueue.FreeTensor(indicesLocal);
    }

    __aicore__ inline uint32_t Max(int32_t a, int32_t b)
    {
        if (a > b)  return a;
        return b;
    }
    __aicore__ inline uint32_t Min(int32_t a, int32_t b)
    {
        if (a > b)  return b;
        return a;
    }

private:
// Private Member
    TPipe *pipe;
    GlobalTensor<DTYPE_INDICES> indicesGm, outputIndicesGm, outputIndicesPairGm;

    TQue<QuePosition::VECIN, 1> indicesQueue;
    TBuf<TPosition::VECCALC> outIndicesUB, outIndicesPairUB;

    uint32_t usedCoreNum;
    uint32_t coreTask;
    uint32_t lastCoreTask;

    uint32_t moveLen;

    uint32_t repeatTimes;
    uint32_t moveTail;
    uint32_t lastRepeatTimes;
    uint32_t lastMoveTail;

    uint32_t kernelD;
    uint32_t kernelH;
    uint32_t kernelW;
    uint32_t kernelSize;

    uint32_t outfeatureB;
    uint32_t outputDepth;
    uint32_t outputHeight;
    uint32_t outputWidth;

    uint32_t strideDepth;
    uint32_t strideHeight;
    uint32_t strideWidth;

    uint32_t paddingDepth;
    uint32_t paddingHeight;
    uint32_t paddingWidth;

    uint32_t curBlockIdx;

    uint32_t taskNum;
    uint32_t coreRepeatTimes;
    uint32_t coreMoveTail;
    uint32_t maskAlign;
    uint32_t mulmask;
    uint32_t mulRepeatTimes;
    uint32_t workSize;
};
extern "C" __global__ __aicore__ void sparse_conv3d(GM_ADDR indices, GM_ADDR indices_out, GM_ADDR indices_pair, GM_ADDR workspace, GM_ADDR tiling) {
    SetSysWorkspace(workspace);
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    KernelSparseConv3d op;
    op.Init(indices, indices_out, indices_pair, workspace, &tiling_data, &pipe);
    op.Process();
}