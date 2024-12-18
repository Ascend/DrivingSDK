/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 */
#include <cmath>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template<typename DTYPE_G, typename DTYPE_I>
class KernelGroupPointsGrad {
public:
    __aicore__ inline KernelGroupPointsGrad() {}
    __aicore__ inline void Init(
        GM_ADDR gradOut, GM_ADDR indices, GM_ADDR gradPoints, const GroupPointsGradTilingData* tiling_data)
    {
        b = tiling_data->b;
        c = tiling_data->c;
        n = tiling_data->n;
        npoints = tiling_data->npoints;
        nsample = tiling_data->nsample;
        cAligned = tiling_data->cAligned;
        indicesAligned = tiling_data->indicesAligned;
        average = tiling_data->average;
        taskLast = tiling_data->taskLast;
        usedCoreNum = tiling_data->usedCoreNum;

        CopyParamasInit();

        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        indicesLength = static_cast<uint64_t>(b)* npoints * nsample;
        gradOutLength = static_cast<uint64_t>(b) * npoints * nsample * c;
        gradPointsLength = static_cast<uint64_t>(b) * n * c;

        indicesGm.SetGlobalBuffer((__gm__ DTYPE_I*)indices, indicesLength);
        gradOutGm.SetGlobalBuffer((__gm__ DTYPE_G*)gradOut, gradOutLength);
        gradPointsGm.SetGlobalBuffer((__gm__ DTYPE_G*)gradPoints, gradPointsLength);

        pipe.InitBuffer(inQueueGradOut, BUFFER_NUM, cAligned * sizeof(DTYPE_G));
        pipe.InitBuffer(inQueueIndices, BUFFER_NUM, indicesAligned * sizeof(DTYPE_I));
    }

    __aicore__ inline void CopyParamasInit()
    {
        copyParamsOut.blockCount = 1;
        copyParamsOut.blockLen = static_cast<uint32_t>(c * sizeof(DTYPE_G));
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
        LocalTensor<DTYPE_I> indicesLocal = inQueueIndices.AllocTensor<DTYPE_I>();

        DataCopyExtParams copyGradOutParams {1, static_cast<uint32_t>(c * sizeof(DTYPE_G)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_G> gradOutPadParams {false, 0, 0, 0};

        int64_t offset = average * GetBlockIdx() + taskLast;
        if (GetBlockIdx() < taskLast) {
            offset = (average + 1) * GetBlockIdx();
        }

        DataCopy(gradOutLocal, gradOutGm[(offset + i) * c], cAligned);
        DataCopy(indicesLocal, indicesGm[offset + i], indicesAligned);
        pipe_barrier(PIPE_ALL);

        int32_t b_idx = (offset + i) / (npoints * nsample);
        int32_t idx = indicesLocal.GetValue(0);
        int64_t gradPointOffset = static_cast<int64_t>(b_idx) * n * c + idx * c;

        SetAtomicAdd<DTYPE_G>();
        DataCopyPad(gradPointsGm[gradPointOffset], gradOutLocal, copyParamsOut);
        SetAtomicNone();

        inQueueGradOut.FreeTensor(gradOutLocal);
        inQueueIndices.FreeTensor(indicesLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGradOut;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueIndices;
    GlobalTensor<DTYPE_G> gradOutGm;
    GlobalTensor<DTYPE_I> indicesGm;
    GlobalTensor<DTYPE_G> gradPointsGm;

    uint64_t gradOutLength;
    uint64_t indicesLength;
    uint64_t gradPointsLength;
    uint32_t b;
    uint32_t c;
    uint32_t n;
    uint32_t npoints;
    uint32_t nsample;
    uint32_t cAligned;
    uint32_t indicesAligned;
    uint32_t average;
    uint32_t taskLast;
    uint32_t usedCoreNum;
    DataCopyExtParams copyParamsOut;
};

extern "C" __global__ __aicore__ void group_points_grad(
    GM_ADDR gradOut, GM_ADDR indices, GM_ADDR gradPoints, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        KernelGroupPointsGrad<float, int32_t> op;
        op.Init(gradOut, indices, gradPoints, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelGroupPointsGrad<half, int32_t> op;
        op.Init(gradOut, indices, gradPoints, &tiling_data);
        op.Process();
    }
}