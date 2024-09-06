/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */
#ifndef FURTHEST_POINT_SAMPLING_H
#define FURTHEST_POINT_SAMPLING_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace AscendC {
constexpr uint32_t BUFFER_NUM = 1u;
constexpr uint32_t OP_MAX_REPEAT_NUM = 255u;
constexpr uint32_t ALLIGNED_BYTES = 256u;

enum PointAxis {
    pointAxis_x,
    pointAxis_y,
    pointAxis_z
};

template<typename dataType, typename idxType>
struct UbBlocks_tag {
    __aicore__ UbBlocks_tag() = default;

    LocalTensor<dataType> pointXLocal;
    LocalTensor<dataType> pointYLocal;
    LocalTensor<dataType> pointZLocal;
    LocalTensor<dataType> pointTempXLocal;
    LocalTensor<dataType> pointTempYLocal;
    LocalTensor<dataType> pointTempZLocal;
    LocalTensor<dataType> nearestDistLocal;
    LocalTensor<dataType> distLocal;
    LocalTensor<idxType>  idxLocal;
    LocalTensor<dataType> idxTempLocal;
    LocalTensor<dataType> pointSampledLocal;
    LocalTensor<dataType> workLocal;
};
template<typename dataType, typename idxType>
using UbBlocks = UbBlocks_tag<dataType, idxType>;

class tilingArgs {
public:
    __aicore__ inline tilingArgs() = default;
public:
    uint32_t N;
    uint32_t batch;
    uint32_t numPoints;
    uint32_t pieces;
    uint32_t formerNum;
    uint32_t tailNum;
    uint32_t workSize;
    uint32_t idxTempSize;
    uint32_t bigCoreBatch;
    uint32_t smallCoreBatch;
    uint32_t bigCoreNum;
    uint32_t repeats;
};

template<typename dataType, typename idxType>
class furthestPointSamplingKernel {
public:
    __aicore__ inline furthestPointSamplingKernel(GM_ADDR point_xyz, GM_ADDR temp, GM_ADDR index, GM_ADDR workspace,
        tilingArgs *tiling);
    __aicore__ inline ~furthestPointSamplingKernel();
    __aicore__ inline void Process();

private:
    __aicore__ inline void Process_first_sampling(uint32_t loopSplit = 0);
    __aicore__ inline void Process_split_data();
    __aicore__ inline void Process_complete_data();

private:
    __aicore__ inline void CopyInPointAxis(PointAxis pointAxis, uint32_t loopSplit = 0);
    __aicore__ inline void CopyInNearestDist(uint32_t loopSplit = 0);
    __aicore__ inline void CopyInNearestDistTemp(uint32_t loopSplit = 0);
    __aicore__ inline void CopyInIdx(uint32_t loopNum);
    __aicore__ inline void CopyOut(uint32_t loopNum);
    __aicore__ inline void CopyOutNearestDistTemp(uint32_t loopSplit = 0);

private:
    __aicore__ inline void ComputePointsSquare();
    __aicore__ inline void ComputePointDeltaSquare(LocalTensor<dataType> &pointLocal,
        LocalTensor<dataType> &pointTempLocal, dataType pointSampled);
    __aicore__ inline void ComputeDist();
    __aicore__ inline void ComputeSamplePoints(uint32_t loopSplit, uint32_t ComBlock);
    __aicore__ inline void updateDist();

private:
    __aicore__ inline void InitGm(GM_ADDR point_xyz, GM_ADDR temp, GM_ADDR index, GM_ADDR workspace);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> pointXQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> pointYQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> pointZQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> pointTempXUb;
    TQue<QuePosition::VECIN, BUFFER_NUM> pointTempYUb;
    TQue<QuePosition::VECIN, BUFFER_NUM> pointTempZUb;
    TQue<QuePosition::VECIN, BUFFER_NUM> nearestDistQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> distUb;
    TQue<QuePosition::VECOUT, BUFFER_NUM> workUb;

    TQue<QuePosition::VECOUT, BUFFER_NUM> idxQue;

    TQue<QuePosition::VECOUT, BUFFER_NUM> idxTempUb;
    TQue<QuePosition::VECOUT, BUFFER_NUM> pointSampled;

private:
    GlobalTensor<dataType> pointGm;
    GlobalTensor<dataType> nearestDistGm;
    GlobalTensor<idxType> idxGm;
    GlobalTensor<dataType> nearestDistTempGm;
    UbBlocks<dataType, idxType> ubBlocks;

private:
    dataType pointXSampled {0};
    dataType pointYSampled {0};
    dataType pointZSampled {0};
    dataType maxDist {0};
    idxType maxDistIdx {0};
    uint32_t core_batch;

private:
    // tiling value
    tilingArgs *TA;

private:
    uint32_t sizeofFormer;
    uint32_t sizeofTail;
    uint32_t dataNumIn32Bytes;
    uint32_t dataNumIn64Bytes;
    uint32_t dataNumIn256Bytes;
    uint32_t dataNumIn1024Bytes;
    uint32_t batchOffsetPoint;
    uint32_t batchOffsetNearest;
};
}

#endif // FURTHEST_POINT_SAMPLING_H