/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace matmul;

namespace {
    constexpr uint32_t BUFFER_NUM = 1;
    constexpr uint32_t COORDINATE_DIM = 2;
    constexpr uint32_t BLOCK_BYTES = 32;
    constexpr uint32_t ALIGN_NUM_8 = 8;
    constexpr uint32_t ALIGN_NUM_64 = 64; // CompareScalar function requires 256B alignment.
    constexpr float MAX_FLOAT = 3.40282347E+38;
}

class KernelRadius {
public:
    __aicore__ inline KernelRadius() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR ptrX, GM_ADDR ptrY, GM_ADDR outTemp, GM_ADDR outFinal, GM_ADDR numNeighbors, GM_ADDR usrWorkspace, RadiusTilingData * tiling_data)
    {
        batchSize = tiling_data->batchSize;
        numPointsX = tiling_data->numPointsX;
        numPointsY = tiling_data->numPointsY;
        maxNumNeighbors = tiling_data->maxNumNeighbors;
        headCoreNum = tiling_data->headCoreNum;
        batchPerCore = tiling_data->batchPerCore;
        batchPerCoreTail = tiling_data->batchPerCoreTail;
        bufferSizePtr = tiling_data->bufferSizePtr;
        bufferSizePoints = tiling_data->bufferSizePoints;
        numLocalPtr = tiling_data->numLocalPtr;
        numLocalPoints = tiling_data->numLocalPoints;
        usedCoreNum = tiling_data->usedCoreNum;
        r = tiling_data->r;
        blockIdx = GetBlockIdx();
        numOutputPoints = 0;
        maxNumPointsPerIter = numLocalPoints;

        if (blockIdx < headCoreNum) {
            batchThisCore = batchPerCore;
            ptrAddrOffset = blockIdx * batchPerCore;
        } else {
            batchThisCore = batchPerCoreTail;
            ptrAddrOffset = blockIdx * batchPerCore - (blockIdx - headCoreNum);
        }

        xGm.SetGlobalBuffer((__gm__ float *)x, COORDINATE_DIM * numPointsX); // [2, num_points_x]
        yGm.SetGlobalBuffer((__gm__ float *)y, COORDINATE_DIM * numPointsY); // [2, num_points_y]
        ptrXGm.SetGlobalBuffer((__gm__ int32_t *)ptrX, batchSize + 1); // [batch_size + 1]
        ptrYGm.SetGlobalBuffer((__gm__ int32_t *)ptrY, batchSize + 1); // [batch_size + 1]
        outTempGm.SetGlobalBuffer((__gm__ int32_t *)outTemp, COORDINATE_DIM * numPointsY * maxNumNeighbors); // [2, num_points_y * max_num_neighbors]
        outFinalGm.SetGlobalBuffer((__gm__ int32_t *)outFinal, COORDINATE_DIM * numPointsY * maxNumNeighbors); // [2, num_points_y * max_num_neighbors]
        numNeighborsGm.SetGlobalBuffer((__gm__ int32_t *)numNeighbors, ALIGN_NUM_8); // [8]
        numNeighborsCoreGm.SetGlobalBuffer((__gm__ int32_t *)usrWorkspace, (usedCoreNum + 1) * ALIGN_NUM_8); // [usedCoreNum + 1, 8]

        pipe.InitBuffer(ptrXBuf, bufferSizePtr);
        pipe.InitBuffer(ptrYBuf, bufferSizePtr);
        pipe.InitBuffer(xBuf, bufferSizePoints); // 32KB
        pipe.InitBuffer(yBuf, bufferSizePoints); // 32KB
        pipe.InitBuffer(distBuf, bufferSizePoints / COORDINATE_DIM); // 16KB
        pipe.InitBuffer(tempBuf, bufferSizePoints / COORDINATE_DIM); // 16KB
        pipe.InitBuffer(maskBuf, bufferSizePoints / COORDINATE_DIM  / sizeof(int32_t)); // 4KB
        pipe.InitBuffer(indexXBuf, bufferSizePoints / COORDINATE_DIM); // 16KB
        pipe.InitBuffer(indexYBuf, bufferSizePoints / COORDINATE_DIM); // 16KB
        pipe.InitBuffer(numNeighborsBuf, ALIGN_NUM_8 * sizeof(int32_t)); // 8B
        pipe.InitBuffer(numNeighborsCoreBuf, (usedCoreNum + 1) * ALIGN_NUM_8 * sizeof(int32_t)); // (usedCoreNum + 1) * 8B, store the number of each core and the total number of neighbors

        // numNeighborsCoreGm needs to be initialized to 0
        if (blockIdx == 0) {
            InitOutput<int32_t>(numNeighborsCoreGm, (usedCoreNum + 1) * ALIGN_NUM_8, 0);
        }
        SyncAll();
    }

    __aicore__ inline void Process()
    {
        // Input batch address pointer
        CopyInPtr();
        for (int32_t i = 0; i < batchThisCore; i++) {
            // According to the batch address pointer, points within a single batch are moved in
            CopyInPoints(i);
            Compute(i);
        }
        CopyOut();
        SyncAll();
        // Rearrange non-contiguous data to ensure continuous distribution in memory
        DataMoveFinal();
    }

private:
    __aicore__ inline void CopyInPtr()
    {
        ptrXLocal = ptrXBuf.Get<int32_t>();
        ptrYLocal = ptrYBuf.Get<int32_t>();
        DataCopyParams copyParams {1, static_cast<uint16_t>(numLocalPtr * sizeof(int32_t)), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};
        DataCopyPad(ptrXLocal, ptrXGm[ptrAddrOffset], copyParams, padParams);
        DataCopyPad(ptrYLocal, ptrYGm[ptrAddrOffset], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyInPoints(uint32_t batchIdx)
    {
        ptrXLeft = ptrXLocal.GetValue(batchIdx);
        ptrYLeft = ptrYLocal.GetValue(batchIdx);
        ptrXRight = ptrXLocal.GetValue(batchIdx + 1);
        ptrYRight = ptrYLocal.GetValue(batchIdx + 1);
        if (batchIdx == 0) {
            outputGmOffsetY = ptrYLeft * maxNumNeighbors;
            outputGmOffsetX = outputGmOffsetY + numPointsY * maxNumNeighbors;
        }
        numBatchPointsX = ptrXRight - ptrXLeft;
        numBatchPointsY = ptrYRight - ptrYLeft;
        numBatchPointsAlignedX = (numBatchPointsX + ALIGN_NUM_64) / ALIGN_NUM_64 * ALIGN_NUM_64;
        numBatchPointsAlignedY = (numBatchPointsY + ALIGN_NUM_64) / ALIGN_NUM_64 * ALIGN_NUM_64;
        pointsXLocal = xBuf.Get<float>();
        pointsYLocal = yBuf.Get<float>();
        PipeBarrier<PIPE_ALL>();
        DataCopyParams copyParamsX {1, static_cast<uint16_t>(numBatchPointsX * sizeof(float)), 0, 0};
        DataCopyParams copyParamsY {1, static_cast<uint16_t>(numBatchPointsY * sizeof(float)), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};
        DataCopyPad(pointsXLocal, xGm[ptrXLeft], copyParamsX, padParams);
        DataCopyPad(pointsYLocal, yGm[ptrYLeft], copyParamsY, padParams);
        DataCopyPad(pointsXLocal[numLocalPoints], xGm[numPointsX + ptrXLeft], copyParamsX, padParams);
        DataCopyPad(pointsYLocal[numLocalPoints], yGm[numPointsY + ptrYLeft], copyParamsY, padParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Compute(uint32_t batchIdx)
    {
        distLocal = distBuf.Get<float>();
        tempLocal = tempBuf.Get<float>();
        maskUint8 = maskBuf.Get<uint8_t>();
        indexXTensor = indexXBuf.Get<int32_t>();
        indexYTensor = indexYBuf.Get<int32_t>();

        int32_t pointIdxAbs = 0;
        uint64_t selectCnt = 0;
        for (int32_t pointIdx = 0; pointIdx < numBatchPointsY; pointIdx++) {
            y1 = -1 * pointsYLocal.GetValue(pointIdx);
            y2 = -1 * pointsYLocal.GetValue(numLocalPoints + pointIdx);
            Adds(distLocal, pointsXLocal, y1, numBatchPointsAlignedX);
            Adds(tempLocal, pointsXLocal[numLocalPoints], y2, numBatchPointsAlignedX);
            PipeBarrier<PIPE_V>();
            Mul(distLocal, distLocal, distLocal, numBatchPointsAlignedX);
            Mul(tempLocal, tempLocal, tempLocal, numBatchPointsAlignedX);
            PipeBarrier<PIPE_V>();
            Add(distLocal, distLocal, tempLocal, numBatchPointsAlignedX);
            PipeBarrier<PIPE_V>();
            CompareScalar(maskUint8, distLocal, r, CMPMODE::LT, numBatchPointsAlignedX);
            maskUint32 = maskUint8.ReinterpretCast<uint32_t>();
            pointIdxAbs = pointIdx + ptrYLeft;
            Duplicate(indexYTensor, pointIdxAbs, numBatchPointsAlignedX);
            CreateVecIndex(indexXTensor, ptrXLeft, numBatchPointsAlignedX);
            GatherMask(indexXTensor, indexXTensor, maskUint32, true, numBatchPointsX, {1, 1, 0, 0}, selectCnt);

            PipeBarrier<PIPE_ALL>();
            if (selectCnt > maxNumNeighbors) {
                selectCnt = maxNumNeighbors;
            }
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams outCopyParams {1, static_cast<uint32_t>(selectCnt * sizeof(float)), 0, 0, 0};
            DataCopyPad(outTempGm[outputGmOffsetX + numOutputPoints], indexXTensor, outCopyParams);
            DataCopyPad(outTempGm[outputGmOffsetY + numOutputPoints], indexYTensor, outCopyParams);
            numOutputPoints = numOutputPoints + selectCnt;
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void CopyOut()
    {
        numNeighborsCoreTensor = numNeighborsCoreBuf.Get<int32_t>();
        Duplicate(numNeighborsCoreTensor, numOutputPoints, usedCoreNum * ALIGN_NUM_8);
        PipeBarrier<PIPE_ALL>();
        SetAtomicAdd<int32_t>();
        DataCopyParams outCopyParams {1, static_cast<uint16_t>((usedCoreNum - blockIdx) * ALIGN_NUM_8 * sizeof(int32_t)), 0, 0};
        DataCopyPad(numNeighborsCoreGm, numNeighborsCoreTensor, outCopyParams);
        SetAtomicNone();
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void DataMoveFinal()
    {
        numNeighborsTensor = numNeighborsBuf.Get<int32_t>();
        DataCopyParams copyParamsNumNeighbors {1, static_cast<uint16_t>(ALIGN_NUM_8 * sizeof(int32_t)), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};
        DataCopyPad(numNeighborsTensor, numNeighborsCoreGm[(usedCoreNum - blockIdx) * ALIGN_NUM_8], copyParamsNumNeighbors, padParams);
        PipeBarrier<PIPE_ALL>();
        uint32_t addrsPtrY = numNeighborsTensor.GetValue(0);
        uint32_t addrsPtrX = addrsPtrY + numPointsY * maxNumNeighbors;
        uint32_t numIters = numOutputPoints / maxNumPointsPerIter + 1;
        uint32_t numPointsTail = numOutputPoints % maxNumPointsPerIter;
        numOutputPoints = numOutputPoints + addrsPtrY;
        DataCopyParams copyParams {1, static_cast<uint16_t>(maxNumPointsPerIter * sizeof(int32_t)), 0, 0};
        DataCopyParams outCopyParams {1, static_cast<uint16_t>(maxNumPointsPerIter * sizeof(int32_t)), 0, 0};
        DataCopyParams copyParamsTail {1, static_cast<uint16_t>(numPointsTail * sizeof(int32_t)), 0, 0};
        DataCopyParams outCopyParamsTail {1, static_cast<uint16_t>(numPointsTail * sizeof(int32_t)), 0, 0};

        for (int32_t i = 0; i < numIters; i++) {
            PipeBarrier<PIPE_ALL>();
            if (i < numIters - 1) {
                DataCopyPad(indexXTensor, outTempGm[outputGmOffsetX], copyParams, padParams);
                DataCopyPad(indexYTensor, outTempGm[outputGmOffsetY], copyParams, padParams);
            } else {
                DataCopyPad(indexXTensor, outTempGm[outputGmOffsetX], copyParamsTail, padParams);
                DataCopyPad(indexYTensor, outTempGm[outputGmOffsetY], copyParamsTail, padParams);
            }
            PipeBarrier<PIPE_ALL>();
            if (i < numIters - 1) {
                DataCopyPad(outFinalGm[addrsPtrX], indexXTensor, outCopyParams);
                DataCopyPad(outFinalGm[addrsPtrY], indexYTensor, outCopyParams);
            } else {
                DataCopyPad(outFinalGm[addrsPtrX], indexXTensor, outCopyParamsTail);
                DataCopyPad(outFinalGm[addrsPtrY], indexYTensor, outCopyParamsTail);
            }
            outputGmOffsetX = outputGmOffsetX + maxNumPointsPerIter;
            outputGmOffsetY = outputGmOffsetY + maxNumPointsPerIter;
            addrsPtrX = addrsPtrX + maxNumPointsPerIter;
            addrsPtrY = addrsPtrY + maxNumPointsPerIter;
            PipeBarrier<PIPE_ALL>();
        }
        if (blockIdx == usedCoreNum - 1) {
            PipeBarrier<PIPE_ALL>();
            Duplicate(numNeighborsTensor, numOutputPoints, ALIGN_NUM_8);
            DataCopyPad(numNeighborsGm, numNeighborsTensor, copyParamsNumNeighbors);
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> xBuf, yBuf;
    TBuf<TPosition::VECCALC> ptrXBuf, ptrYBuf;
    TBuf<TPosition::VECCALC> distBuf, tempBuf, maskBuf, indexXBuf, indexYBuf;
    TBuf<TPosition::VECCALC> numNeighborsBuf, numNeighborsCoreBuf;
    LocalTensor<int32_t> ptrXLocal, ptrYLocal;
    LocalTensor<int32_t>  indexXTensor, indexYTensor, numNeighborsTensor, numNeighborsCoreTensor;
    LocalTensor<float> pointsXLocal, pointsYLocal, distLocal, tempLocal;
    LocalTensor<uint8_t> maskUint8;
    LocalTensor<uint32_t> maskUint32;
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    GlobalTensor<int32_t> ptrXGm;
    GlobalTensor<int32_t> ptrYGm;
    GlobalTensor<int32_t> outTempGm;
    GlobalTensor<int32_t> outFinalGm;
    GlobalTensor<int32_t> numNeighborsGm;
    GlobalTensor<int32_t> numNeighborsCoreGm;
    uint32_t blockIdx, headCoreNum, batchPerCore, batchPerCoreTail, batchThisCore;
    uint32_t ptrAddrOffset, bufferSizePtr, bufferSizePoints;
    uint32_t batchSize, maxNumNeighbors, usedCoreNum, maxNumPointsPerIter;
    uint32_t numBatchPointsX, numBatchPointsY, numBatchPointsAlignedX, numBatchPointsAlignedY;
    uint32_t outputGmOffsetX,  outputGmOffsetY;
    int32_t ptrXLeft, ptrXRight, ptrYLeft, ptrYRight;
    int32_t numPointsX, numPointsY, numLocalPoints, numLocalPtr, numOutputPoints;
    float r, y1, y2;
};

extern "C" __global__ __aicore__ void radius(GM_ADDR x, GM_ADDR y, GM_ADDR ptrX, GM_ADDR ptrY, GM_ADDR outTemp, GM_ADDR outFinal, GM_ADDR numTotalNeighbors, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    KernelRadius op;
    op.Init(x, y, ptrX, ptrY, outTemp, outFinal, numTotalNeighbors, usrWorkspace, &tiling_data);
    op.Process();
}
