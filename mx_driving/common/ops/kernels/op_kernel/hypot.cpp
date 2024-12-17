/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;

class KernelHypot {
public:
    __aicore__ inline KernelHypot() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, HypotTilingData * tiling_data)
    {
        uint32_t formerNum = tiling_data->formerNum;
        uint64_t formerLength = tiling_data->formerLength;
        uint32_t tailLength = tiling_data->tailLength;

        if (GetBlockIdx() < formerNum) {
            tileLength = tiling_data->formerTileLength;
            tileNum = tiling_data->formerTileNum;
            remainTileLength = tiling_data->formerRemainTileLength;

            xGm.SetGlobalBuffer((__gm__ float *)x + formerLength * GetBlockIdx(), formerLength);
            yGm.SetGlobalBuffer((__gm__ float *)y + formerLength * GetBlockIdx(), formerLength);
            zGm.SetGlobalBuffer((__gm__ float *)z + formerLength * GetBlockIdx(), formerLength);
        } else {
            tileLength = tiling_data->tailTileLength;
            tileNum = tiling_data->tailTileNum;
            remainTileLength = tiling_data->tailRemainTileLength;

            xGm.SetGlobalBuffer(
                (__gm__ float *)x + formerLength * formerNum + tailLength * (GetBlockIdx() - formerNum), tailLength);
            yGm.SetGlobalBuffer(
                (__gm__ float *)y + formerLength * formerNum + tailLength * (GetBlockIdx() - formerNum), tailLength);
            zGm.SetGlobalBuffer(
                (__gm__ float *)z + formerLength * formerNum + tailLength * (GetBlockIdx() - formerNum), tailLength);
        }

        if (tileLength == 0) {
            return;
        }

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (tileNum == 0) {
            return;
        }

        for (int32_t i = 0; i < tileNum - 1; ++i) {
            CopyIn(i, tileLength);
            Compute(tileLength);
            CopyOut(i, tileLength);
        }

        CopyIn(tileNum - 1, remainTileLength);
        Compute(remainTileLength);
        CopyOut(tileNum - 1, remainTileLength);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * tileLength], length);
        DataCopy(yLocal, yGm[progress * tileLength], length);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(uint32_t length)
    {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        Mul(xLocal, xLocal, xLocal, length);
        Mul(yLocal, yLocal, yLocal, length);
        Add(xLocal, xLocal, yLocal, length);
        Sqrt(zLocal, xLocal, length);
        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        DataCopy(zGm[progress * tileLength], zLocal, length);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    GlobalTensor<float> zGm;
    uint32_t tileLength;
    uint32_t tileNum;
    uint32_t remainTileLength;
};

extern "C" __global__ __aicore__ void hypot(GM_ADDR input, GM_ADDR other, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelHypot op;
    op.Init(input, other, out, &tiling_data);
    op.Process();
}
