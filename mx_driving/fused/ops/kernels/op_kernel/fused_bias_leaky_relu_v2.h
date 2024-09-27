/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "kernel_operator.h"


using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t SINGLE_BLOCK = 8;

class KernelFusedBiasLeakyReluV2 {
public:
    __aicore__ inline KernelFusedBiasLeakyReluV2(GM_ADDR x, GM_ADDR bias, GM_ADDR output, FusedBiasLeakyReluV2TilingData* tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zeronumber!");

        this->negative_slope = tiling_data->negative_slope;
        this->scale = tiling_data->scale;
    
        this->usedCoreNum = tiling_data->usedCoreNum;
        this->average = tiling_data->average;
        this->remainder = tiling_data->remainder;
        this->totalDataLength = tiling_data->totalDataLength;
        this->singleBlock = tiling_data->singleBlock;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, this->totalDataLength);
        biasGm.SetGlobalBuffer((__gm__ DTYPE_X*)bias, this->totalDataLength);
        outputGm.SetGlobalBuffer((__gm__ DTYPE_X*)output, this->totalDataLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->singleBlock * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueBIAS, BUFFER_NUM, this->singleBlock * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueOUTPUT, BUFFER_NUM, this->singleBlock * sizeof(DTYPE_X));
    }

    __aicore__ inline void Process()
    {
        int32_t tmp = average;
        if (GetBlockIdx() < remainder) {
            tmp += 1;
        }
        for (int32_t i = 0; i < tmp; ++i) {
            int32_t offset = average * GetBlockIdx() + remainder;
            if (GetBlockIdx() < remainder) {
                offset = (average + 1) * GetBlockIdx();
            }
            
            CopyIn(i, offset);
            Compute(i, offset);
            Copyout(i, offset);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, int32_t offset)
    {
        LocalTensor<DTYPE_X> inputX = inQueueX.AllocTensor<DTYPE_X>();
        LocalTensor<DTYPE_X> inputBias = inQueueBIAS.AllocTensor<DTYPE_X>();

        DataCopy(inputX, xGm[(offset + progress) * this->singleBlock], this->singleBlock);
        DataCopy(inputBias, biasGm[(offset + progress) * this->singleBlock], this->singleBlock);

        inQueueX.EnQue<DTYPE_X>(inputX);
        inQueueBIAS.EnQue<DTYPE_X>(inputBias);
    }

    __aicore__ inline void Compute(int32_t progress, int32_t offset)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_X> biasLoacl = inQueueBIAS.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_X> outputLocal = outQueueOUTPUT.AllocTensor<DTYPE_X>();

        Add(outputLocal, xLocal, biasLoacl, this->singleBlock);
        LeakyRelu(outputLocal, outputLocal, (DTYPE_X)(this->negative_slope), this->singleBlock);
        Muls(outputLocal, outputLocal, (DTYPE_X)(this->scale), this->singleBlock);

        outQueueOUTPUT.EnQue<DTYPE_X>(outputLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueBIAS.FreeTensor(biasLoacl);
    }

    __aicore__ inline void Copyout(int32_t progress, int32_t offset)
    {
        LocalTensor<DTYPE_X> outputLocal = outQueueOUTPUT.DeQue<DTYPE_X>();

        DataCopy(outputGm[(offset + progress) * this->singleBlock], outputLocal, this->singleBlock);

        outQueueOUTPUT.FreeTensor(outputLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueBIAS;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOUTPUT;

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_X> biasGm;
    GlobalTensor<DTYPE_X> outputGm;

    float negative_slope;
    float scale;

    uint32_t usedCoreNum;
    uint32_t average;
    uint32_t remainder;
    uint32_t totalDataLength;
    int32_t singleBlock;
};