/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 */
#include "kernel_operator.h"
using namespace AscendC;


constexpr uint32_t BUFFER_NUM = 2;

class KernelGroupPoints {
public:
    __aicore__ inline KernelGroupPoints() {}
    __aicore__ inline void Init(GM_ADDR points, GM_ADDR indices, GM_ADDR out, const GroupPointsTilingData* tiling_data)
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

        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        pointsLength = static_cast<uint64_t>(b) * n * c;
        indicesLength = static_cast<uint64_t>(b) * npoints * nsample;
        outLength = static_cast<uint64_t>(b) * npoints * nsample * c;

        pointsGm.SetGlobalBuffer((__gm__ DTYPE_POINTS*)points, pointsLength);
        indicesGm.SetGlobalBuffer((__gm__ DTYPE_INDICES*)indices, indicesLength);
        outGm.SetGlobalBuffer((__gm__ DTYPE_OUT*)out, outLength);

        pipe.InitBuffer(pointsBuffer, cAligned * sizeof(DTYPE_POINTS));
        pipe.InitBuffer(indicesBuffer, indicesAligned * sizeof(DTYPE_INDICES));
    }

    __aicore__ inline void Process()
    {
        uint64_t tmp = average;
        uint64_t offset = tmp * GetBlockIdx() + taskLast;
        if (GetBlockIdx() < taskLast) {
            tmp = tmp + 1;
            offset = tmp * GetBlockIdx();
        }
        for (uint64_t i = 0; i < tmp; i++) {
            CopyInAndCopyOut(i, offset);
        }
    }

private:
    __aicore__ inline void CopyInAndCopyOut(uint64_t progress, uint64_t offset)
    {
        LocalTensor<DTYPE_POINTS> points_local = pointsBuffer.Get<DTYPE_POINTS>();
        LocalTensor<DTYPE_INDICES> indices_local = indicesBuffer.Get<DTYPE_INDICES>();
        pipe_barrier(PIPE_ALL);
        DataCopy(indices_local, indicesGm[offset + progress], indicesAligned);
        
        uint32_t b_idx = (offset + progress) / (npoints * nsample);
        uint32_t idx = indices_local.GetValue(0);
        uint64_t src_idx = b_idx * n * c + idx * c;
        uint64_t dst_idx = (offset + progress) * c;
        pipe_barrier(PIPE_ALL);
        DataCopy(points_local, pointsGm[src_idx], cAligned);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

        DataCopyExtParams outCopyParams {1, static_cast<uint32_t>(c * sizeof(DTYPE_POINTS)), 0, 0, 0};
        DataCopyPad(outGm[dst_idx], points_local, outCopyParams);
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> indicesBuffer, pointsBuffer;
    GlobalTensor<DTYPE_POINTS> pointsGm;
    GlobalTensor<DTYPE_INDICES> indicesGm;
    GlobalTensor<DTYPE_OUT> outGm;

    uint64_t pointsLength;
    uint64_t indicesLength;
    uint64_t outLength;
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
    DataCopyExtParams outCopyParams;
};

extern "C" __global__ __aicore__ void group_points(GM_ADDR points, GM_ADDR indices, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGroupPoints op;
    op.Init(points, indices, out, &tiling_data);
    op.Process();
}