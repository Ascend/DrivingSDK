/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;

class KernelHypotGrad {
public:
    __aicore__ inline KernelHypotGrad() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR z_grad, GM_ADDR x_grad, GM_ADDR y_grad, HypotTilingData * tiling_data)
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
            zGradGm.SetGlobalBuffer((__gm__ float *)z_grad + formerLength * GetBlockIdx(), formerLength);
            xGradGm.SetGlobalBuffer((__gm__ float *)x_grad + formerLength * GetBlockIdx(), formerLength);
            yGradGm.SetGlobalBuffer((__gm__ float *)y_grad + formerLength * GetBlockIdx(), formerLength);
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
            zGradGm.SetGlobalBuffer(
                (__gm__ float *)z_grad + formerLength * formerNum + tailLength * (GetBlockIdx() - formerNum), tailLength);
            xGradGm.SetGlobalBuffer(
                (__gm__ float *)x_grad + formerLength * formerNum + tailLength * (GetBlockIdx() - formerNum), tailLength);
            yGradGm.SetGlobalBuffer(
                (__gm__ float *)y_grad + formerLength * formerNum + tailLength * (GetBlockIdx() - formerNum), tailLength);
        }

        if (tileLength == 0) {
            return;
        }

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(inQueueZ, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(inQueueZGrad, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueXGrad, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(outQueueYGrad, BUFFER_NUM, tileLength * sizeof(float));
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
        LocalTensor<float> zLocal = inQueueZ.AllocTensor<float>();
        LocalTensor<float> zGradLocal = inQueueZGrad.AllocTensor<float>();
        DataCopy(xLocal, xGm[progress * tileLength], length);
        DataCopy(yLocal, yGm[progress * tileLength], length);
        DataCopy(zLocal, zGm[progress * tileLength], length);
        DataCopy(zGradLocal, zGradGm[progress * tileLength], length);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
        inQueueZ.EnQue(zLocal);
        inQueueZGrad.EnQue(zGradLocal);
    }
    __aicore__ inline void Compute(uint32_t length)
    {
        LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        LocalTensor<float> zLocal = inQueueZ.DeQue<float>();
        LocalTensor<float> zGradLocal = inQueueZGrad.DeQue<float>();
        LocalTensor<float> xGradLocal = outQueueXGrad.AllocTensor<float>();
        LocalTensor<float> yGradLocal = outQueueYGrad.AllocTensor<float>();

        // according to chain rule, dL/dx = (x/z)*dL/dz, dL/dy = (y/z)*dL/dz
        Mul(xGradLocal, zGradLocal, xLocal, length);
        Div(xGradLocal, xGradLocal, zLocal, length);
        Mul(yGradLocal, zGradLocal, yLocal, length);
        Div(yGradLocal, yGradLocal, zLocal, length);

        outQueueXGrad.EnQue<float>(xGradLocal);
        outQueueYGrad.EnQue<float>(yGradLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        inQueueZ.FreeTensor(zLocal);
        inQueueZGrad.FreeTensor(zGradLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        LocalTensor<float> xGradLocal = outQueueXGrad.DeQue<float>();
        LocalTensor<float> yGradLocal = outQueueYGrad.DeQue<float>();
        DataCopy(xGradGm[progress * tileLength], xGradLocal, length);
        DataCopy(yGradGm[progress * tileLength], yGradLocal, length);
        outQueueXGrad.FreeTensor(xGradLocal);
        outQueueYGrad.FreeTensor(yGradLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY, inQueueZ, inQueueZGrad;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueXGrad, outQueueYGrad;
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    GlobalTensor<float> zGm;
    GlobalTensor<float> zGradGm;
    GlobalTensor<float> xGradGm;
    GlobalTensor<float> yGradGm;
    uint32_t tileLength;
    uint32_t tileNum;
    uint32_t remainTileLength;
};

extern "C" __global__ __aicore__ void hypot_grad(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR z_grad, GM_ADDR x_grad, GM_ADDR y_grad, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelHypotGrad op;
    op.Init(x, y, z, z_grad, x_grad, y_grad, &tiling_data);
    op.Process();
}
