/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef _DYNAMIC_SCATTER_BASE_H_
#define _DYNAMIC_SCATTER_BASE_H_

#include <cmath>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace DynamicScatterN {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class DynamicScatterBase {
public:
    __aicore__ inline DynamicScatterBase() {}
    __aicore__ inline void BaseInit(DynamicScatterTilingData* tilingData)
    {
        TilingDataInit(tilingData);
        MemberDataInit();
        uint32_t inputNumAligned = (inputNum + alignNum - 1) / alignNum * alignNum;
        tileLength = inputNumAligned * featsNum > tileLength ? tileLength : inputNumAligned * featsNum;
        tilePointNum = tileLength / featsNum;
        loop = blockLength / tileLength;
        lastLength = blockLength % tileLength;
        lastPointNum = lastLength / featsNum;
        featsLastStartIndex = blockLength - lastLength;
        featsLastStartIndex = featsLastStartIndex > 0 ? featsLastStartIndex : 0;
        mapLastStartIndex = featsLastStartIndex / featsNum;
        outLength = outPointNumAligned * featsNum;
        CopyParamasInit();
    }

    __aicore__ inline void TilingDataInit(DynamicScatterTilingData* tilingData)
    {
        totalLength = tilingData->totalLength;
        formerNum = tilingData->formerNum;
        tailNum = tilingData->tailNum;
        formerLength = tilingData->formerLength;
        tailLength = tilingData->tailLength;
        alignNum = tilingData->alignNum;
        totalLengthAligned = tilingData->totalLengthAligned;
        formerInputNum = tilingData->formerInputNum;
        tailInputNum = tilingData->tailInputNum;
        featsNum = tilingData->featsNum;
        outPointNum = tilingData->outPointNum;
        outPointNumAligned = tilingData->outPointNumAligned;
        featsAligned = tilingData->featsAligned;
        tileLength = tilingData->tileLength;
    }

    __aicore__ inline void MemberDataInit()
    {
        if (GetBlockIdx() < formerNum) {
            blockLength = formerLength;
            inputNum = formerInputNum;
            featsOffset = blockLength * GetBlockIdx();
            coorsMapOffset = formerInputNum * GetBlockIdx();
        } else {
            blockLength = tailLength;
            inputNum = tailInputNum;
            featsOffset = formerLength * formerNum + tailLength * (GetBlockIdx() - formerNum);
            coorsMapOffset = formerInputNum * formerNum + tailInputNum * (GetBlockIdx() - formerNum);
        }
    }

    __aicore__ inline void CopyParamasInit()
    {
        copyParamsOut.blockCount = 1;
        copyParamsOut.blockLen = static_cast<uint32_t>(featsNum * sizeof(T));
        copyParamsOut.srcStride = 0;
        copyParamsOut.dstStride = 0;
        copyParamsOut.rsv = 0;
    }

    __aicore__ inline void CopyOutMax(GlobalTensor<T> reducedFeatsGm, uint32_t index, LocalTensor<T> featsLocal)
    {
        SetAtomicMax<T>();
        DataCopyPad(reducedFeatsGm[index], featsLocal, copyParamsOut);
        SetAtomicNone();
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void CopyOutAdd(GlobalTensor<T> reducedFeatsGm, uint32_t index, LocalTensor<T> featsLocal)
    {
        SetAtomicAdd<T>();
        DataCopyPad(reducedFeatsGm[index], featsLocal, copyParamsOut);
        SetAtomicNone();
        pipe_barrier(PIPE_ALL);
    }
protected:
    uint32_t totalLength, formerNum, tailNum, formerLength, tailLength, alignNum, totalLengthAligned, outLength;
    uint32_t formerInputNum, tailInputNum, featsNum;
    uint32_t outPointNum, outPointNumAligned, featsAligned;
    uint32_t blockLength, tilePointNum, inputNum, featsOffset, coorsMapOffset;
    DataCopyExtParams copyParamsOut;
    uint32_t blockLengthAligned;
    int32_t tileLength = 8;
    int32_t loop, lastLength, featsLastStartIndex, mapLastStartIndex, lastPointNum;
};
} // DynamicScatterN
#endif  // _DYNAMIC_SCATTER_BASE_H_