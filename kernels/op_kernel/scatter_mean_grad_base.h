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
constexpr uint32_t BUFFER_NUM = 2;
template <typename T>
class ScatterMeanGradBase {
public:
    __aicore__ inline ScatterMeanGradBase() {}
    __aicore__ inline void InitTiling(const ScatterMeanGradTilingData* tilingData)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->curBlockIdx = GetBlockIdx();
        this->tilingMode = tilingData->tilingMode;
        this->dimRange = tilingData->dimRange;
        this->dimRangeOut = tilingData->dimRangeOut;
        this->paramsPro = tilingData->paramsPro;
        this->tail = tilingData->tail;
        this->body = this->paramsPro / this->tail;
        this->bigCoreNum = tilingData->bigCoreNum;
        this->indexUbSize = tilingData->indexUbSize;
        this->gradOutUbSize = tilingData->gradOutUbSize;
        this->gradInNum = tilingData->gradInNum;
        this->indexNum = tilingData->indexNum;
        this->gradOutNum = tilingData->gradOutNum;
        this->countNum = tilingData->countNum;

        this->indicesEachBlock = BLOCK_BYTES / sizeof(DTYPE_INDEX);
        this->paramsEachBlock = BLOCK_BYTES / sizeof(DTYPE_COUNT);
    }

protected:
    uint32_t curBlockIdx;
    uint64_t indexUbSize;
    uint64_t gradOutUbSize;
    uint64_t dimRange;
    uint64_t dimRangeOut;
    uint64_t paramsPro;
    uint64_t gradInNum;
    uint64_t indexNum;
    uint64_t gradOutNum;
    uint64_t countNum;
    int32_t dim;

    uint64_t tilingMode;
    uint64_t tail;
    uint64_t body;
    uint64_t bigCoreNum;

    uint32_t indicesEachBlock;
    uint32_t paramsEachBlock;

    DataCopyExtParams copyParamsOut = {1, 8, 0, 0, 0};
};
}
#endif