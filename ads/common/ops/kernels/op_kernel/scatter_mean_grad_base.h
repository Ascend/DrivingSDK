/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef _SCATTER_MEAN_GRAD_BASE_H_
#define _SCATTER_MEAN_GRAD_BASE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
namespace ScatterMeanGradNS {

using namespace AscendC;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t MASK_BYTES = 256;
constexpr uint32_t MASK = 256 / sizeof(int32_t);
template <typename T>
class ScatterMeanGradBase {
public:
    __aicore__ inline ScatterMeanGradBase() {}
    __aicore__ inline void InitTiling(const ScatterMeanGradTilingData* tilingData)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->coreNum = GetBlockNum();
        this->curBlockIdx = GetBlockIdx();
        this->paramsPre = tilingData->paramsPre;
        this->dimRange = tilingData->dimRange;
        this->dimRangeOut = tilingData->dimRangeOut;
        this->paramsPro = tilingData->paramsPro;
        this->dim = tilingData->dim;

        this->paramsNumPerMask = MASK_BYTES / sizeof(T);
        this->paramsNumPerBlock = BLOCK_BYTES / sizeof(T);
        this->indexNumPerBlock = BLOCK_BYTES / sizeof(int32_t);
        this->inLastTwoDims = this->dimRange * this->paramsPro;
        this->outLastTwoDims = this->dimRangeOut * this->paramsPro;

        this->taskPerCore = tilingData->taskPerCore;
        this->taskTailCore = tilingData->taskTailCore;

        this->gradInUbSize = tilingData->gradInUbSize;
        this->indexUbSize = tilingData->indexUbSize;
        this->gradOutUbSize = tilingData->gradOutUbSize;
        this->indexSumUbSize = tilingData->indexSumUbSize;

        this->gradInNum = tilingData->gradInNum;
        this->indexNum = tilingData->indexNum;
        this->gradOutNum = tilingData->gradOutNum;
    }

    __aicore__ inline uint32_t CeilValue(uint32_t a, uint32_t b)
    {
        if (b == 0) {
            return 0;
        }
        return ((a - 1) / b + 1) * b;
    }

protected:
    uint32_t coreNum;
    uint32_t curBlockIdx;
    uint32_t gradInUbSize;
    uint32_t indexUbSize;
    uint32_t gradOutUbSize;
    uint32_t indexSumUbSize;
    uint32_t paramsPre;
    uint32_t dimRange;
    uint32_t dimRangeOut;
    uint32_t paramsPro;
    uint32_t gradInNum;
    uint32_t indexNum;
    uint32_t gradOutNum;
    int32_t dim;

    uint32_t taskPerCore;
    uint32_t taskTailCore;

    uint32_t paramsNumPerMask;
    uint32_t paramsNumPerBlock;
    uint32_t indexNumPerBlock;
    uint32_t inLastTwoDims;
    uint32_t outLastTwoDims;
    BinaryRepeatParams repeatParamsCompare = {1, 1, 0, 8, 8, 0};
    BinaryRepeatParams repeatParamsSelect = {1, 0, 1, 8, 0, 8};
};
}
#endif