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
        taskNum = outHeight;
        taskNumPerCore = DivCeil(taskNum, coreNum);

        curBlockIdx = GetBlockIdx();
        startOffset = curBlockIdx * taskNumPerCore;
        endOffset = (curBlockIdx + 1) * taskNumPerCore;
        if (endOffset > taskNum) {
            endOffset = taskNum;
        }

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

        pipe->InitBuffer(maxPart1Ub, batchNum * sizeof(DTYPE_X_TRANS));
        pipe->InitBuffer(maxPart2Ub, batchNum * sizeof(DTYPE_X_TRANS));

        pipe->InitBuffer(resUb, channel * sizeof(DTYPE_X_TRANS));
        pipe->InitBuffer(tmpUb, channel * sizeof(DTYPE_X_TRANS));
    }

    __aicore__ inline void Process()
    {
        Compute();
    }

private:
    __aicore__ inline void MovePart(uint32_t offset)
    {
        if (oriWidth == -1) {
            DataCopy(xPart2Local, xTransGm[(offset + 1) * channel], batchNum - channel);
            DataCopy(xPart3Local, xTransGm[(offset + inWidth + 1) * channel], batchNum - channel);
            pipe_barrier(PIPE_ALL);

            Max(maxPart2Local, xPart2Local, xPart3Local, batchNum - channel);
            Max(resLocal, maxPart2Local, maxPart2Local[channel], channel);
        } else if (oriWidth + kernelSize > inWidth) {
            DataCopy(xPart2Local, xTransGm[offset * channel], batchNum - channel);
            DataCopy(xPart3Local, xTransGm[(offset + inWidth) * channel], batchNum - channel);
            pipe_barrier(PIPE_ALL);

            Max(maxPart2Local, xPart2Local, xPart3Local, batchNum - channel);
            Max(resLocal, maxPart2Local, maxPart2Local[channel], channel);
        } else {
            DataCopy(xPart2Local, xTransGm[offset * channel], batchNum);
            DataCopy(xPart3Local, xTransGm[(offset + inWidth) * channel], batchNum);
            pipe_barrier(PIPE_ALL);

            Max(maxPart2Local, xPart2Local, xPart3Local, batchNum);
            Max(tmpLocal, maxPart2Local, maxPart2Local[channel], channel);
            Max(resLocal, tmpLocal, maxPart2Local[batchNum - channel], channel);
        }
    }

    __aicore__ inline void MoveMain()
    {
        if (oriWidth == -1) {
            DataCopy(xPart1Local, xTransGm[(inOffset + 1) * channel], batchNum - channel);
            DataCopy(xPart2Local, xTransGm[(inOffset + inWidth + 1) * channel], batchNum - channel);
            DataCopy(xPart3Local, xTransGm[(inOffset + inWidth * 2 + 1) * channel], batchNum - channel);
            pipe_barrier(PIPE_ALL);

            Max(maxPart1Local, xPart1Local, xPart2Local, batchNum - channel);
            Max(maxPart2Local, maxPart1Local, xPart3Local, batchNum - channel);
            Max(resLocal, maxPart2Local, maxPart2Local[channel], channel);
        } else if (oriWidth + kernelSize > inWidth) {
            DataCopy(xPart1Local, xTransGm[inOffset * channel], batchNum - channel);
            DataCopy(xPart2Local, xTransGm[(inOffset + inWidth) * channel], batchNum - channel);
            DataCopy(xPart3Local, xTransGm[(inOffset + inWidth * 2) * channel], batchNum - channel);
            pipe_barrier(PIPE_ALL);

            Max(maxPart1Local, xPart1Local, xPart2Local, batchNum - channel);
            Max(maxPart2Local, maxPart1Local, xPart3Local, batchNum - channel);
            Max(resLocal, maxPart2Local, maxPart2Local[channel], channel);
        } else {
            DataCopyPad(xPart1Local, xTransGm[inOffset * channel], copyParams, padParams);
            pipe_barrier(PIPE_ALL);

            Max(maxPart1Local, xPart1Local, xPart1Local[batchNum], batchNum);
            Max(maxPart2Local, maxPart1Local, xPart1Local[batchNum * 2], batchNum);
            Max(tmpLocal, maxPart2Local, maxPart2Local[channel], channel);
            Max(resLocal, tmpLocal, maxPart2Local[batchNum - channel], channel);
        }
    }

    __aicore__ inline void Compute()
    {
        xPart1Local = xPart1Ub.Get<DTYPE_X_TRANS>();
        xPart2Local = xPart2Ub.Get<DTYPE_X_TRANS>();
        xPart3Local = xPart3Ub.Get<DTYPE_X_TRANS>();

        maxPart1Local = maxPart1Ub.Get<DTYPE_X_TRANS>();
        maxPart2Local = maxPart2Ub.Get<DTYPE_X_TRANS>();

        resLocal = resUb.Get<DTYPE_X_TRANS>();
        tmpLocal = tmpUb.Get<DTYPE_X_TRANS>();

        for (uint32_t batch = 0; batch < batchSize; batch++) {
            outOffset = batch * outHeight * outWidth * channel;
            for (uint32_t high = startOffset; high < endOffset; high++) {
                oriHeight = high * stride - padding;
                baseOffset = (batch * inHeight + oriHeight) * inWidth;
                for (uint32_t wide = 0; wide < outWidth; wide++) {
                    oriWidth = wide * stride - padding;
                    inOffset = baseOffset + oriWidth;
                    if (oriHeight == -padding) {
                        MovePart(inOffset + inWidth);
                    } else if (oriHeight + kernelSize > inHeight) {
                        MovePart(inOffset);
                    } else {
                        MoveMain();
                    }
                    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

                    DataCopy(yTransGm[outOffset + (high * outWidth + wide) * channel], resLocal, channel);
                }
            }
        }
    }

private:
    TPipe* pipe;
    GlobalTensor<DTYPE_X_TRANS> xTransGm, yTransGm;
    TBuf<TPosition::VECCALC> xPart1Ub, xPart2Ub, xPart3Ub, maxPart1Ub, maxPart2Ub, resUb, tmpUb;
    LocalTensor<DTYPE_X_TRANS> xPart1Local, xPart2Local, xPart3Local, maxPart1Local, maxPart2Local;
    LocalTensor<DTYPE_X_TRANS> resLocal, tmpLocal;
    uint32_t batchSize;
    uint32_t channel;
    uint32_t inHeight;
    uint32_t inWidth;
    uint32_t outHeight;
    uint32_t outWidth;
    uint32_t coreNum;

    uint32_t oriHeight;
    uint32_t oriWidth;
    uint32_t inOffset;
    uint32_t baseOffset;
    uint32_t outOffset;

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
