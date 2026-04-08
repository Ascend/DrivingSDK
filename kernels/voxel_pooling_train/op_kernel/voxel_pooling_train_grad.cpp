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
class KernelVoxelPoolingTrainGrad {
public:
    __aicore__ inline KernelVoxelPoolingTrainGrad() {}
    __aicore__ inline void Init(
        GM_ADDR gradOut, GM_ADDR posMemo, GM_ADDR gradFeatures, const VoxelPoolingTrainGradTilingData* tiling_data)
    {
        batch_size = tiling_data->batchSize;
        num_points = tiling_data->numPoints;
        num_channels = tiling_data->numChannels;
        h = tiling_data->h;
        w = tiling_data->w;
        numChannelsAligned = tiling_data->numChannelsAligned;
        indicesAligned = tiling_data->indicesAligned;
        average = tiling_data->average;
        taskLast = tiling_data->taskLast;
        usedCoreNum = tiling_data->usedCoreNum;

        CopyParamasInit();

        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        posMemoLength = static_cast<uint64_t>(batch_size) * num_points * INDICES_NUM;
        gradOutLength = static_cast<uint64_t>(batch_size) * h * w * num_channels;
        gradFeaturesLength = static_cast<uint64_t>(batch_size) * num_points * num_channels;

        posMemoGm.SetGlobalBuffer((__gm__ DTYPE_I*)posMemo, posMemoLength);
        gradOutGm.SetGlobalBuffer((__gm__ DTYPE_G*)gradOut, gradOutLength);
        gradFeaturesGm.SetGlobalBuffer((__gm__ DTYPE_G*)gradFeatures, gradFeaturesLength);

        pipe.InitBuffer(inQueueGradOut, BUFFER_NUM, numChannelsAligned * sizeof(DTYPE_G));
        pipe.InitBuffer(inQueuePosMemo, BUFFER_NUM, indicesAligned * sizeof(DTYPE_I));
    }

    __aicore__ inline void CopyParamasInit()
    {
        copyParamsOut.blockCount = 1;
        copyParamsOut.blockLen = static_cast<uint32_t>(num_channels * sizeof(DTYPE_G));
        copyParamsOut.srcStride = 0;
        copyParamsOut.dstStride = 0;
        copyParamsOut.rsv = 0;
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
        LocalTensor<DTYPE_G> gradOutLocal = inQueueGradOut.AllocTensor<DTYPE_G>();
        LocalTensor<DTYPE_I> posMemoLocal = inQueuePosMemo.AllocTensor<DTYPE_I>();

        int64_t offset = static_cast<int64_t>(average) * GetBlockIdx() + taskLast;
        if (GetBlockIdx() < taskLast) {
            offset = (static_cast<int64_t>(average) + 1) * GetBlockIdx();
        }

        DataCopy(posMemoLocal, posMemoGm[(offset + i) * INDICES_NUM], indicesAligned);

        int32_t idx0 = posMemoLocal.GetValue(0);
        if (idx0 != -1) {
            int32_t idx1 = posMemoLocal.GetValue(1);
            int32_t idx2 = posMemoLocal.GetValue(2);
            int64_t offset_grad_out = static_cast<int64_t>(idx0) * h * w * num_channels + idx1 * w * num_channels + idx2 * num_channels;
            DataCopy(gradOutLocal, gradOutGm[offset_grad_out], numChannelsAligned);

            pipe_barrier(PIPE_ALL);
            DataCopyPad(gradFeaturesGm[(offset + i) * num_channels], gradOutLocal, copyParamsOut);
        }

        inQueueGradOut.FreeTensor(gradOutLocal);
        inQueuePosMemo.FreeTensor(posMemoLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGradOut;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueuePosMemo;
    GlobalTensor<DTYPE_G> gradOutGm;
    GlobalTensor<DTYPE_I> posMemoGm;
    GlobalTensor<DTYPE_G> gradFeaturesGm;

    uint64_t gradOutLength;
    uint64_t posMemoLength;
    uint64_t gradFeaturesLength;
    uint32_t batch_size;
    uint32_t num_points;
    uint32_t num_channels;
    uint32_t h;
    uint32_t w;
    uint32_t numChannelsAligned;
    uint32_t indicesAligned;
    uint32_t average;
    uint32_t taskLast;
    uint32_t usedCoreNum;
    DataCopyExtParams copyParamsOut;
};

extern "C" __global__ __aicore__ void voxel_pooling_train_grad(
    GM_ADDR gradOut, GM_ADDR posMemo, GM_ADDR gradFeatures, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelVoxelPoolingTrainGrad<float, int32_t> op;
    op.Init(gradOut, posMemo, gradFeatures, &tiling_data);
    op.Process();
}