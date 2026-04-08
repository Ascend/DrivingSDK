/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file max_pool2d.cpp
 * \brief
 */

#include "kernel_operator.h"
using namespace AscendC;

class KernelMaxPool2d {
public:
    __aicore__ inline KernelMaxPool2d() {}
    __aicore__ inline void Init(
        GM_ADDR x_trans, GM_ADDR y_trans, const MaxPool2dTilingData* tiling_data, TPipe* tmpPipe)
    {
        pipe = tmpPipe;
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        dataAlign = blockNum / sizeof(DTYPE_X_TRANS);
        batchSize = tiling_data->batchSize;
        channel = tiling_data->channel;
        inHeight = tiling_data->inHeight;
        inWidth = tiling_data->inWidth;
        outHeight = tiling_data->outHeight;
        outWidth = tiling_data->outWidth;
        coreNum = tiling_data->coreNum;

        batchNum = channel * kernelSize;

        taskNum = batchSize * outHeight;
        taskNumPerCore = DivCeil(taskNum, coreNum);

        curBlockIdx = GetBlockIdx();
        startOffset = curBlockIdx * taskNumPerCore;
        endOffset = (curBlockIdx + 1) * taskNumPerCore;
        if (endOffset > taskNum) {
            endOffset = taskNum;
        }

        wBatch = (numAlign / channel - 1) / stride;
        validW = outWidth - 1;
        if (inWidth % 2 == 1) {
            validW = outWidth - 2;
        }
        wRounds = validW / wBatch;
        wTail = validW % wBatch;

        eventIdVToMte3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());

        copyParams = {(uint16_t)kernelSize, uint32_t(batchNum * sizeof(DTYPE_X_TRANS)),
            uint32_t((inWidth - kernelSize) * channel * sizeof(DTYPE_X_TRANS)), 0, 0};

        xTransGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_X_TRANS*>(x_trans), batchSize * inHeight * inWidth * channel);
        yTransGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_X_TRANS*>(y_trans), batchSize * outHeight * outWidth * channel);

        pipe->InitBuffer(xPart1Ub, batchNum * kernelSize * sizeof(DTYPE_X_TRANS));
        pipe->InitBuffer(xPart2Ub, batchNum * sizeof(DTYPE_X_TRANS));
        pipe->InitBuffer(xPart3Ub, batchNum * sizeof(DTYPE_X_TRANS));

        pipe->InitBuffer(xBatchUb1, numAlign * sizeof(DTYPE_X_TRANS));
        pipe->InitBuffer(xBatchUb2, numAlign * sizeof(DTYPE_X_TRANS));
        pipe->InitBuffer(xBatchUb3, numAlign * sizeof(DTYPE_X_TRANS));
        pipe->InitBuffer(xBatchUb4, numAlign * sizeof(DTYPE_X_TRANS));

        pipe->InitBuffer(maxPart1Ub, batchNum * sizeof(DTYPE_X_TRANS));
        pipe->InitBuffer(maxPart2Ub, batchNum * sizeof(DTYPE_X_TRANS));

        pipe->InitBuffer(resUb, channel * sizeof(DTYPE_X_TRANS));
    }

    __aicore__ inline void Process()
    {
        ComputeNH();
    }

private:
    __aicore__ inline void MovePart()
    {
        DataCopy(xPart2Local, xTransGm[baseOffset * channel], batchNum - channel);
        DataCopy(xPart3Local, xTransGm[(baseOffset + inWidth) * channel], batchNum - channel);
        pipe_barrier(PIPE_ALL);

        Max(maxPart2Local, xPart2Local, xPart3Local, batchNum - channel);
        Max(resLocal, maxPart2Local, maxPart2Local[channel], channel);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        DataCopy(yTransGm[outOffset], resLocal, channel);

        for (uint32_t idx1 = 0; idx1 < wRounds; idx1++) {
            inOffset = baseOffset + (idx1 * wBatch + 1) * stride - padding;
            DataCopy(xBatchLocal1, xTransGm[inOffset * channel], (wBatch * stride + 1) * channel);
            DataCopy(xBatchLocal2, xTransGm[(inOffset + inWidth) * channel], (wBatch * stride + 1) * channel);
            pipe_barrier(PIPE_ALL);
            Max(xBatchLocal3, xBatchLocal1, xBatchLocal2, (wBatch * stride + 1) * channel);

            for (uint32_t idx2 = 0; idx2 < wBatch; idx2++) {
                Max(xBatchLocal4[idx2 * channel], xBatchLocal3[idx2 * stride * channel],
                    xBatchLocal3[idx2 * stride * channel + channel], channel);
                Max(xBatchLocal4[idx2 * channel], xBatchLocal4[idx2 * channel],
                    xBatchLocal3[idx2 * stride * channel + 2 * channel], channel);
            }
            pipe_barrier(PIPE_ALL);

            DataCopy(yTransGm[outOffset + (idx1 * wBatch + 1) * channel], xBatchLocal4, wBatch * channel);
        }
        if (wTail > 0) {
            inOffset = baseOffset + (wBatch * wRounds + 1) * stride - padding;
            DataCopy(xBatchLocal1, xTransGm[inOffset * channel], (wTail * stride + 1) * channel);
            DataCopy(xBatchLocal2, xTransGm[(inOffset + inWidth) * channel], (wTail * stride + 1) * channel);
            pipe_barrier(PIPE_ALL);
            Max(xBatchLocal3, xBatchLocal1, xBatchLocal2, (wTail * stride + 1) * channel);

            for (uint32_t idx2 = 0; idx2 < wTail; idx2++) {
                Max(xBatchLocal4[idx2 * channel], xBatchLocal3[idx2 * stride * channel],
                    xBatchLocal3[idx2 * stride * channel + channel], channel);
                Max(xBatchLocal4[idx2 * channel], xBatchLocal4[idx2 * channel],
                    xBatchLocal3[idx2 * stride * channel + 2 * channel], channel);
            }
            pipe_barrier(PIPE_ALL);

            DataCopy(yTransGm[outOffset + (wBatch * wRounds + 1) * channel], xBatchLocal4, wTail * channel);
        }
        if (inWidth % 2 == 1) {
            inOffset = baseOffset + (outWidth - 1) * stride - padding;
            DataCopy(xPart2Local, xTransGm[inOffset * channel], batchNum - channel);
            DataCopy(xPart3Local, xTransGm[(inOffset + inWidth) * channel], batchNum - channel);
            pipe_barrier(PIPE_ALL);

            Max(maxPart2Local, xPart2Local, xPart3Local, batchNum - channel);
            Max(resLocal, maxPart2Local, maxPart2Local[channel], channel);

            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

            DataCopy(yTransGm[outOffset + (outWidth - 1) * channel], resLocal, channel);
        }
    }

    __aicore__ inline void MoveMain()
    {
        DataCopy(xPart1Local, xTransGm[baseOffset * channel], batchNum - channel);
        DataCopy(xPart2Local, xTransGm[(baseOffset + inWidth) * channel], batchNum - channel);
        DataCopy(xPart3Local, xTransGm[(baseOffset + inWidth * 2) * channel], batchNum - channel);
        pipe_barrier(PIPE_ALL);

        Max(maxPart1Local, xPart1Local, xPart2Local, batchNum - channel);
        Max(maxPart2Local, maxPart1Local, xPart3Local, batchNum - channel);
        Max(xBatchLocal4, maxPart2Local, maxPart2Local[channel], channel);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopy(yTransGm[outOffset], xBatchLocal4, channel);

        for (uint32_t idx1 = 0; idx1 < wRounds; idx1++) {
            inOffset = baseOffset + (idx1 * wBatch + 1) * stride - padding;
            DataCopy(xBatchLocal1, xTransGm[inOffset * channel], (wBatch * stride + 1) * channel);
            DataCopy(xBatchLocal2, xTransGm[(inOffset + inWidth) * channel], (wBatch * stride + 1) * channel);
            DataCopy(xBatchLocal3, xTransGm[(inOffset + inWidth * 2) * channel], (wBatch * stride + 1) * channel);

            pipe_barrier(PIPE_ALL);

            Max(xBatchLocal1, xBatchLocal1, xBatchLocal2, (wBatch * stride + 1) * channel);
            Max(xBatchLocal3, xBatchLocal1, xBatchLocal3, (wBatch * stride + 1) * channel);

            for (uint32_t idx2 = 0; idx2 < wBatch; idx2++) {
                Max(xBatchLocal4[idx2 * channel], xBatchLocal3[idx2 * stride * channel],
                    xBatchLocal3[idx2 * stride * channel + channel], channel);
                Max(xBatchLocal4[idx2 * channel], xBatchLocal4[idx2 * channel],
                    xBatchLocal3[idx2 * stride * channel + 2 * channel], channel);
            }
            pipe_barrier(PIPE_ALL);

            DataCopy(yTransGm[outOffset + (idx1 * wBatch + 1) * channel], xBatchLocal4, wBatch * channel);
        }
        if (wTail > 0) {
            inOffset = baseOffset + (wBatch * wRounds + 1) * stride - padding;
            DataCopy(xBatchLocal1, xTransGm[inOffset * channel], (wTail * stride + 1) * channel);
            DataCopy(xBatchLocal2, xTransGm[(inOffset + inWidth) * channel], (wTail * stride + 1) * channel);
            DataCopy(xBatchLocal3, xTransGm[(inOffset + inWidth * 2) * channel], (wTail * stride + 1) * channel);
            pipe_barrier(PIPE_ALL);

            Max(xBatchLocal1, xBatchLocal1, xBatchLocal2, (wTail * stride + 1) * channel);
            Max(xBatchLocal3, xBatchLocal1, xBatchLocal3, (wTail * stride + 1) * channel);

            for (uint32_t idx2 = 0; idx2 < wTail; idx2++) {
                Max(xBatchLocal4[idx2 * channel], xBatchLocal3[idx2 * stride * channel],
                    xBatchLocal3[idx2 * stride * channel + channel], channel);
                Max(xBatchLocal4[idx2 * channel], xBatchLocal4[idx2 * channel],
                    xBatchLocal3[idx2 * stride * channel + 2 * channel], channel);
            }
            pipe_barrier(PIPE_ALL);

            DataCopy(yTransGm[outOffset + (wBatch * wRounds + 1) * channel], xBatchLocal4, wTail * channel);
        }

        if (inWidth % 2 == 1) {
            inOffset = baseOffset + (outWidth - 1) * stride - padding;

            DataCopy(xPart1Local, xTransGm[inOffset * channel], batchNum - channel);
            DataCopy(xPart2Local, xTransGm[(inOffset + inWidth) * channel], batchNum - channel);
            DataCopy(xPart3Local, xTransGm[(inOffset + inWidth * 2) * channel], batchNum - channel);
            pipe_barrier(PIPE_ALL);

            Max(maxPart1Local, xPart1Local, xPart2Local, batchNum - channel);
            Max(maxPart2Local, maxPart1Local, xPart3Local, batchNum - channel);
            Max(resLocal, maxPart2Local, maxPart2Local[channel], channel);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            DataCopy(yTransGm[outOffset + (outWidth - 1) * channel], resLocal, channel);
        }
    }

    __aicore__ inline void ComputeNH()
    {
        xPart1Local = xPart1Ub.Get<DTYPE_X_TRANS>();
        xPart2Local = xPart2Ub.Get<DTYPE_X_TRANS>();
        xPart3Local = xPart3Ub.Get<DTYPE_X_TRANS>();

        maxPart1Local = maxPart1Ub.Get<DTYPE_X_TRANS>();
        maxPart2Local = maxPart2Ub.Get<DTYPE_X_TRANS>();

        xBatchLocal1 = xBatchUb1.Get<DTYPE_X_TRANS>();
        xBatchLocal2 = xBatchUb2.Get<DTYPE_X_TRANS>();
        xBatchLocal3 = xBatchUb3.Get<DTYPE_X_TRANS>();
        xBatchLocal4 = xBatchUb4.Get<DTYPE_X_TRANS>();

        resLocal = resUb.Get<DTYPE_X_TRANS>();

        for (uint32_t idx = startOffset; idx < endOffset; idx++) {
            high = idx % outHeight;
            batch = idx / outHeight;

            outOffset = idx * outWidth * channel;
            oriHeight = high * stride - padding;
            baseOffset = (batch * inHeight + oriHeight) * inWidth;

            if (oriHeight == -padding) {
                baseOffset = baseOffset + inWidth;
                MovePart();
            } else if (oriHeight + kernelSize > inHeight) {
                MovePart();
            } else {
                MoveMain();
            }
        }
    }

private:
    TPipe* pipe;
    GlobalTensor<DTYPE_X_TRANS> xTransGm, yTransGm;
    TBuf<TPosition::VECCALC> xPart1Ub, xPart2Ub, xPart3Ub, maxPart1Ub, maxPart2Ub, resUb, xBatchUb1, xBatchUb2,
        xBatchUb3, xBatchUb4;
    LocalTensor<DTYPE_X_TRANS> xPart1Local, xPart2Local, xPart3Local, maxPart1Local, maxPart2Local, xBatchLocal1,
        xBatchLocal2, xBatchLocal3, xBatchLocal4;
    LocalTensor<DTYPE_X_TRANS> resLocal;
    uint32_t batchSize;
    uint32_t channel;
    uint32_t inHeight;
    uint32_t inWidth;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t coreNum;
    uint32_t numAlign = 64 * 64;

    uint32_t wBatch;
    uint32_t validW;

    uint32_t wRounds;
    uint32_t wTail;

    uint32_t oriHeight;
    uint32_t oriWidth;
    uint32_t inOffset;
    uint32_t baseOffset;
    uint32_t outOffset;

    uint32_t batch;
    uint32_t high;
    uint32_t wide;

    uint32_t taskNum;
    uint32_t taskNumPerCore;
    uint32_t curBlockIdx;
    uint32_t startOffset;
    uint32_t endOffset;
    uint32_t dataAlign;
    uint32_t blockNum = 32;
    uint32_t padding = 1;
    uint32_t stride = 2;
    uint32_t kernelSize = 3;
    uint32_t batchNum;

    event_t eventIdVToMte3;

    DataCopyExtParams copyParams;
    DataCopyPadExtParams<DTYPE_X_TRANS> padParams {false, 0, 0, 0};
};

extern "C" __global__ __aicore__ void max_pool2d(GM_ADDR x_trans, GM_ADDR y_trans, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    KernelMaxPool2d op;
    op.Init(x_trans, y_trans, &tiling_data, &pipe);
    op.Process();
}
