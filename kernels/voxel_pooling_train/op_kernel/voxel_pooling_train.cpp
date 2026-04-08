/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
*/
#include <cmath>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t INDICES_NUM = 3;

template<typename DTYPE_G, typename DTYPE_I>
class KernelVoxelPoolingTrain {
public:
    __aicore__ inline KernelVoxelPoolingTrain() {}
    __aicore__ inline void Init(GM_ADDR geom, GM_ADDR featuresIn, GM_ADDR featuresOut, GM_ADDR posMemo,
        const VoxelPoolingTilingData* tiling_data)
    {
        batchSize = tiling_data->batchSize;
        numPoints = tiling_data->numPoints;
        numChannels = tiling_data->numChannels;
        numVoxelX = tiling_data->numVoxelX;
        numVoxelY = tiling_data->numVoxelY;
        numVoxelZ = tiling_data->numVoxelZ;
        numChannelsAligned = tiling_data->cAligned;
        indicesAligned = tiling_data->indicesAligned;
        average = tiling_data->average;
        taskLast = tiling_data->taskLast;
        usedCoreNum = tiling_data->usedCoreNum;

        CopyParamasInit();

        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        geomLength = static_cast<uint64_t>(batchSize) * numPoints * INDICES_NUM;
        featuresLength = static_cast<uint64_t>(batchSize) * numPoints * numChannels;
        outLength = static_cast<uint64_t>(batchSize) * numVoxelY * numVoxelX * numChannels;
        posLength = static_cast<uint64_t>(batchSize) * numPoints * INDICES_NUM;

        geomGm.SetGlobalBuffer((__gm__ DTYPE_I*)geom, geomLength);
        featuresGm.SetGlobalBuffer((__gm__ DTYPE_G*)featuresIn, featuresLength);
        outGm.SetGlobalBuffer((__gm__ DTYPE_G*)featuresOut, outLength);
        posGm.SetGlobalBuffer((__gm__ DTYPE_I*)posMemo, posLength);

        pipe.InitBuffer(inQueueFeat, BUFFER_NUM, numChannelsAligned * sizeof(DTYPE_G));
        pipe.InitBuffer(inQueueGeom, BUFFER_NUM, indicesAligned * sizeof(DTYPE_I));
        pipe.InitBuffer(inQueuePos, BUFFER_NUM, indicesAligned * sizeof(DTYPE_I));
    }

    __aicore__ inline void CopyParamasInit()
    {
        copyParamsOut.blockCount = 1;
        copyParamsOut.blockLen = static_cast<uint32_t>(numChannels * sizeof(DTYPE_G));
        copyParamsOut.srcStride = 0;
        copyParamsOut.dstStride = 0;
        copyParamsOut.rsv = 0;

        copyParamsPos.blockCount = 1;
        copyParamsPos.blockLen = static_cast<uint32_t>(INDICES_NUM * sizeof(DTYPE_I));
        copyParamsPos.srcStride = 0;
        copyParamsPos.dstStride = 0;
        copyParamsPos.rsv = 0;
    }

    __aicore__ inline void Process()
    {
        int32_t tmp = average;
        if (GetBlockIdx() < taskLast) {
            tmp = tmp + 1;
        }
        for (int32_t i = 0; i < tmp; i++) {
            ComputeAndCopyOut(i);
        }
    }

    __aicore__ inline void ComputeAndCopyOut(int32_t i)
    {
        LocalTensor<DTYPE_G> outLocal = inQueueFeat.AllocTensor<DTYPE_G>();
        LocalTensor<DTYPE_I> geomLocal = inQueueGeom.AllocTensor<DTYPE_I>();
        LocalTensor<DTYPE_I> posLocal = inQueuePos.AllocTensor<DTYPE_I>();

        int64_t offset = static_cast<int64_t>(average) * GetBlockIdx() + taskLast;
        if (GetBlockIdx() < taskLast) {
            offset = (static_cast<int64_t>(average) + 1) * GetBlockIdx();
        }

        DataCopy(geomLocal, geomGm[(offset + i) * INDICES_NUM], indicesAligned);

        int32_t sampleX = geomLocal.GetValue(0);
        int32_t sampleY = geomLocal.GetValue(1);
        int32_t sampleZ = geomLocal.GetValue(2);
        if ((sampleX >= 0 && sampleX < numVoxelX) && (sampleY >= 0 && sampleY < numVoxelY) && (sampleZ >= 0
            && sampleZ < numVoxelZ)) {
            int32_t b_idx = (offset + i) / numPoints;
            posLocal.SetValue(0, b_idx);
            posLocal.SetValue(1, sampleY);
            posLocal.SetValue(2, sampleX);

            DataCopy(outLocal, featuresGm[(offset + i) * numChannels], numChannelsAligned);
            pipe_barrier(PIPE_ALL);
            DataCopyPad(posGm[(offset + i) * INDICES_NUM], posLocal, copyParamsPos);

            int64_t offset_out =
                static_cast<int64_t>(b_idx) * numVoxelY * numVoxelX * numChannels + sampleY * numVoxelX * numChannels + sampleX * numChannels;
            SetAtomicAdd<DTYPE_G>();
            DataCopyPad(outGm[offset_out], outLocal, copyParamsOut);
            pipe_barrier(PIPE_ALL);
            SetAtomicNone();
        }

        inQueueFeat.FreeTensor(outLocal);
        inQueueGeom.FreeTensor(geomLocal);
        inQueuePos.FreeTensor(posLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueFeat;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGeom;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuePos;

    GlobalTensor<DTYPE_I> geomGm;
    GlobalTensor<DTYPE_G> featuresGm;
    GlobalTensor<DTYPE_G> outGm;
    GlobalTensor<DTYPE_I> posGm;

    uint64_t geomLength;
    uint64_t featuresLength;
    uint64_t outLength;
    uint64_t posLength;
    uint32_t batchSize;
    uint32_t numPoints;
    uint32_t numChannels;
    uint32_t numVoxelX;
    uint32_t numVoxelY;
    uint32_t numVoxelZ;
    uint32_t numChannelsAligned;
    uint32_t indicesAligned;
    uint32_t average;
    uint32_t taskLast;
    uint32_t usedCoreNum;
    DataCopyExtParams copyParamsOut;
    DataCopyExtParams copyParamsPos;
};

extern "C" __global__ __aicore__ void voxel_pooling_train(
    GM_ADDR geom, GM_ADDR featuresIn, GM_ADDR featuresOut, GM_ADDR posMemo, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelVoxelPoolingTrain<float, int32_t> op;
    op.Init(geom, featuresIn, featuresOut, posMemo, &tiling_data);
    op.Process();
}