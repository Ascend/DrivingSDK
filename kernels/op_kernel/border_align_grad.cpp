/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include <cmath>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t SIZE_OF_FP32 = 4;
constexpr int32_t BLOCK_ALIGN = 32 / SIZE_OF_FP32;
enum borderMode {top = 0, left = 1, bottom = 2, right = 3};

class KernelBorderAlignGrad {
public:
    __aicore__ inline KernelBorderAlignGrad() {}
    __aicore__ inline void Init(
        GM_ADDR gradOut, GM_ADDR boxes, GM_ADDR argmaxIdx, GM_ADDR gradInput, const BorderAlignGradTilingData* tilingData)
    {
        channels = tilingData->channels;
        boxSize = tilingData->boxSize;
        height = tilingData->height;
        width = tilingData->width;
        poolSize = tilingData->poolSize;
        batchSize = tilingData->batchSize;
        coreCompNum = tilingData->coreCompNum;
        taskLast = tilingData->taskLast;

        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        gradOutLength = batchSize * channels * boxSize * 4;
        argmaxIdxLength = batchSize * channels * boxSize * 4;
        boxesLength = batchSize * boxSize * 4;
        gradInputLength = batchSize * 4 * channels * boxSize;

        argmaxIdxGm.SetGlobalBuffer((__gm__ DTYPE_ARGMAXIDX *)argmaxIdx, argmaxIdxLength);
        gradOutGm.SetGlobalBuffer((__gm__ DTYPE_GRADOUT *)gradOut, gradOutLength);
        boxesGm.SetGlobalBuffer((__gm__ DTYPE_BOXES *)boxes, boxesLength);
        gradInputGm.SetGlobalBuffer((__gm__ DTYPE_GRADINPUT *)gradInput, gradInputLength);

        pipe.InitBuffer(inQueueGradOut, BUFFER_NUM, 4 * compNum * sizeof(float));
        pipe.InitBuffer(inQueueArgmaxIdx, BUFFER_NUM, 4 * compNum * sizeof(int32_t));
        pipe.InitBuffer(inQueueBoxes, BUFFER_NUM, BLOCK_ALIGN);
        pipe.InitBuffer(outQueueGradInput, BUFFER_NUM, BLOCK_ALIGN);
    }

    __aicore__ inline void Process()
    {
        int64_t offset = coreCompNum * GetBlockIdx() + taskLast;

        if (GetBlockIdx() < taskLast) {
            coreCompNum = coreCompNum + 1;
            offset = coreCompNum * GetBlockIdx();
        }

        int64_t lastNum = coreCompNum % compNum;
        int64_t loopTimes = coreCompNum / compNum;
        for (int64_t currentLoop = 0; currentLoop < loopTimes; currentLoop++) {
            ComputeAndCopyOut(currentLoop * compNum, offset, compNum);
        }
        if (lastNum != 0) {
            ComputeAndCopyOut(loopTimes * compNum, offset, lastNum);
        }
    }

    __aicore__ inline void ComputeAndCopyOut(int64_t index, int64_t offset, int64_t taskNum)
    {
        LocalTensor<float> gradOutLocal = inQueueGradOut.AllocTensor<float>();
        LocalTensor<int32_t> argmaxIdxLocal = inQueueArgmaxIdx.AllocTensor<int32_t>();
        LocalTensor<float> boxesLocal = inQueueBoxes.AllocTensor<float>();
        LocalTensor<float> gradInputLocal = outQueueGradInput.AllocTensor<float>();
        
        DataCopy(gradOutLocal, gradOutGm[(static_cast<int64_t>(offset) + index) * 4], 4 * compNum);
        DataCopy(argmaxIdxLocal, argmaxIdxGm[(static_cast<int64_t>(offset) + index) * 4], 4 * compNum);
        pipe_barrier(PIPE_ALL);

        for (int64_t currentTask = 0; currentTask < taskNum; currentTask++) {
            int64_t batchIdx = (offset + index + currentTask) / (channels * boxSize);
            int64_t boxIdx = (offset + index + currentTask) % boxSize + batchIdx * boxSize;
            DataCopy(boxesLocal, boxesGm[boxIdx * 4], 8);
            pipe_barrier(PIPE_ALL);

            int64_t channelsIdx = (offset + index + currentTask) / boxSize % channels;
            float boxWidth, boxHeight, stride, xStride, yStride, x, y;
            float w1, w2, w3, w4;
            int32_t xLow, xHigh, yLow, yHigh;

            boxWidth = boxesLocal.GetValue(2) - boxesLocal.GetValue(0);
            boxHeight = boxesLocal.GetValue(3) - boxesLocal.GetValue(1);

            for (int32_t i = 0; i < 4; i++) {
                float gradOutput = gradOutLocal.GetValue(4 * currentTask + i);
                int32_t offsetArgmaxIdx = argmaxIdxLocal.GetValue(4 * currentTask + i);

                switch (i) {
                    // top
                    case borderMode::top:
                        stride = boxWidth / poolSize;
                        xStride = stride;
                        yStride = 0;
                        break;
                    // left
                    case borderMode::left:
                        stride = boxHeight / poolSize;
                        xStride = 0;
                        yStride = stride;
                        break;
                    // bottom
                    case borderMode::bottom:
                        stride = boxWidth / poolSize;
                        xStride = -stride;
                        yStride = 0;
                        break;
                    // right
                    case borderMode::right:
                        stride = boxHeight / poolSize;
                        xStride = 0;
                        yStride = -stride;
                        break;
                    default:
                        break;
                }
        
                x = boxesLocal.GetValue((i / 2 * 2));
                y = boxesLocal.GetValue((i / 2 * 2 + 1));
                x += xStride * float(offsetArgmaxIdx);
                y += yStride * float(offsetArgmaxIdx);

                // bilinear_interpolate_gradient
                if (y < -1.0f || y > height || x < -1.0f || x > width) {
                    w1 = w2 = w3 = w4 = 0.0;
                    xLow = xHigh = yLow = yHigh = -1;
                    continue;
                }

                if (y <= 0.0f) {
                    y = 0;
                }
                if (x <= 0.0f) {
                    x = 0;
                }
                
                yLow = AscendC::ScalarCast<float, int32_t, AscendC::RoundMode::CAST_FLOOR>(y);
                xLow = AscendC::ScalarCast<float, int32_t, AscendC::RoundMode::CAST_FLOOR>(x);

                if (yLow >= height - 1) {
                    yHigh = yLow = height - 1;
                    y = static_cast<float>(yLow);
                } else {
                    yHigh = yLow + 1;
                }

                if (xLow >= width - 1) {
                    xHigh = xLow = width - 1;
                    x = static_cast<float>(xLow);
                } else {
                    xHigh = xLow + 1;
                }

                float ly = y - yLow;
                float lx = x - xLow;
                float hy = 1.0f - ly;
                float hx = 1.0f - lx;

                w1 = hy * hx;
                w2 = hy * lx;
                w3 = ly * hx;
                w4 = ly * lx;

                int64_t dstIdx1 = (batchIdx * channels * 4 + i * channels + channelsIdx) * height * width + yLow * width + xLow;
                float values1 = w1 * gradOutput;

                int64_t dstIdx2 = (batchIdx * channels * 4 + i * channels + channelsIdx) * height * width + yLow * width + xHigh;
                float values2 = w2 * gradOutput;

                int64_t dstIdx3 = (batchIdx * channels * 4 + i * channels + channelsIdx) * height * width + yHigh * width + xLow;
                float values3 = w3 * gradOutput;

                int64_t dstIdx4 = (batchIdx * channels * 4 + i * channels + channelsIdx) * height * width + yHigh * width + xHigh;
                float values4 = w4 * gradOutput;
                
                AscendC::SetAtomicAdd<float>();
                gradInputLocal.SetValue(0, values1);
                DataCopyExtParams outCopyParams {1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                DataCopyPad(gradInputGm[dstIdx1], gradInputLocal, outCopyParams);

                gradInputLocal.SetValue(0, values2);
                DataCopyPad(gradInputGm[dstIdx2], gradInputLocal, outCopyParams);

                gradInputLocal.SetValue(0, values3);
                DataCopyPad(gradInputGm[dstIdx3], gradInputLocal, outCopyParams);

                gradInputLocal.SetValue(0, values4);
                DataCopyPad(gradInputGm[dstIdx4], gradInputLocal, outCopyParams);

                AscendC::SetAtomicNone();
                pipe_barrier(PIPE_ALL);
            }
        }
        
        inQueueGradOut.FreeTensor(gradOutLocal);
        inQueueArgmaxIdx.FreeTensor(argmaxIdxLocal);
        inQueueBoxes.FreeTensor(boxesLocal);
        outQueueGradInput.FreeTensor(gradInputLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGradOut;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueArgmaxIdx;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueBoxes;

    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueGradInput;

    GlobalTensor<int32_t> argmaxIdxGm;
    GlobalTensor<float> gradOutGm;
    GlobalTensor<float> boxesGm;
    GlobalTensor<float> gradInputGm;

    int64_t gradOutLength;
    int64_t argmaxIdxLength;
    int64_t boxesLength;
    int64_t gradInputLength;

    int32_t channels;
    int32_t boxSize;
    int32_t height;
    int32_t width;

    int64_t compNum = 128;
    int32_t poolSize;
    int64_t batchSize;
    int64_t coreCompNum;
    int64_t taskLast;

    DataCopyExtParams outCopyParams;
};

extern "C" __global__ __aicore__ void border_align_grad(
    GM_ADDR gradOut, GM_ADDR boxes, GM_ADDR argmaxIdx, GM_ADDR gradInput, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelBorderAlignGrad op;
    op.Init(gradOut, boxes, argmaxIdx, gradInput, &tilingData);
    op.Process();
}