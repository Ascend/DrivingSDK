/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

namespace {
    constexpr uint32_t BUFFER_NUM = 1;
}

class CalcPolyStartEndSl {
public:
    __aicore__ inline CalcPolyStartEndSl() {}
    __aicore__ inline void Init(GM_ADDR min_idx, GM_ADDR poly_line, GM_ADDR points, GM_ADDR s_cum, GM_ADDR poly_start, GM_ADDR poly_end, GM_ADDR sl, CalcPolyStartEndSlTilingData* tiling)
    {
        batchSize = tiling->batchSize;
        npoint = tiling->npoints;
        numIdx = tiling->numIdx;
        numTaskPerCore = tiling->numTaskPerCore;
        numTaskRemained = tiling->numTaskRemained;
        uint32_t numIdxRound64 = (numIdx + 63) / 64 * 64;

        uint32_t coreId = GetBlockIdx();
        // 此处是计算当前核心需要处理的任务总数和该任务的开始ID
        if (coreId < numTaskRemained) {
            numTaskInCore = numTaskPerCore + 1;
            startTaskId = numTaskInCore * coreId;
        } else {
            numTaskInCore = numTaskPerCore;
            startTaskId = numTaskInCore * coreId + numTaskRemained;
        }

        // 计算方式：所涉及的开始的batchid,和涉及的batch个数
        startBatchId = startTaskId / numIdx;
        numBatchInCore = (startTaskId + numTaskInCore + numIdx - 1) / numIdx;
        numBatchInCore = numBatchInCore - startBatchId;

        // 计算在一个核心上处理的每个batch上应当处理的任务个数
        if (numBatchInCore == 1) {
            numTaskPerBatch[0] = numTaskInCore;
        } else {
            numTaskPerBatch[0] = (startBatchId + 1) * numIdx - startTaskId;
            uint32_t tempTaskId = (startBatchId + 1) * numIdx;
            uint32_t i = 1;
            for (uint32_t batchId = startBatchId + 1; batchId < startBatchId + numBatchInCore - 1; batchId++, i++) {
                numTaskPerBatch[i] = numIdx;
                tempTaskId += numIdx;
            }
            numTaskPerBatch[i] = startTaskId + numTaskInCore - tempTaskId;
        }
        numIdxAligned = (numIdx + 8 - 1) / 8 * 8;
        polyLineGm.SetGlobalBuffer((__gm__ float*)poly_line + startBatchId * npoint * 2, numBatchInCore * npoint * 2);
        minIdxGm.SetGlobalBuffer((__gm__ int32_t*)min_idx + startTaskId, numTaskInCore);
        pointsGm.SetGlobalBuffer((__gm__ float*)points + startTaskId * 2, numTaskInCore * 2);
        sCumGm.SetGlobalBuffer((__gm__ float*)s_cum + startBatchId * npoint, numBatchInCore * npoint);
        polyStartGm.SetGlobalBuffer((__gm__ float*)poly_start + startTaskId * 2, numTaskInCore * 2);
        polyEndGm.SetGlobalBuffer((__gm__ float*)poly_end + startTaskId * 2, numTaskInCore * 2);
        slGm.SetGlobalBuffer((__gm__ float*)sl + startTaskId * 2, numTaskInCore * 2);

        pipe.InitBuffer(inQueuePolyLine, BUFFER_NUM, npoint * 2 * sizeof(float) + 32);
        pipe.InitBuffer(inQueueMinIdx, BUFFER_NUM, numIdx * sizeof(int32_t));
        pipe.InitBuffer(inQueuePoints, BUFFER_NUM, numIdxRound64 * 2 * sizeof(float));
        pipe.InitBuffer(inQueueSCum, BUFFER_NUM, npoint * sizeof(float));
        pipe.InitBuffer(outQueuePolyStart, BUFFER_NUM, numIdxAligned * 2 * sizeof(float));
        pipe.InitBuffer(outQueuePolyEnd, BUFFER_NUM, numIdxAligned * 2 * sizeof(float));
        pipe.InitBuffer(outQueueSl, BUFFER_NUM, numIdxAligned * 2 * sizeof(float));

        // 用于给中间变量申请UBuf内存空间使用
        pipe.InitBuffer(minIdxLocalsCumBuf, numIdx * sizeof(int32_t));

        pipe.InitBuffer(polyStartBufX, numIdxAligned * 2 * sizeof(float));
        pipe.InitBuffer(polyEndBufX, numIdxAligned * 2 * sizeof(float));

        pipe.InitBuffer(subPolyEndBufX, numIdxAligned * 2 * sizeof(float));
        pipe.InitBuffer(pointBufX, numIdxRound64 * 2 * sizeof(float));
        pipe.InitBuffer(dotValueBufX, numIdxAligned * 2 * sizeof(float));

        pipe.InitBuffer(offsetBufX, numIdxAligned * 2 * sizeof(int32_t));
        pipe.InitBuffer(offsetBufY, numIdxAligned * 2 * sizeof(int32_t));
        pipe.InitBuffer(offsetFloatBuf, numIdxAligned * 2 * sizeof(float));

        offset = offsetBufX.Get<int32_t>();
        LocalTensor<int32_t> offsetY = offsetBufY.Get<int32_t>();
        LocalTensor<float> offsetFloat = offsetFloatBuf.Get<float>();

        // 创建[0 1 2 3 4 5 6 7 8 .. 2N]
        CreateVecIndex(offset, (int32_t)0, numIdxAligned * 2);
        // 转成float
        Cast(offsetFloat, offset, RoundMode::CAST_CEIL, numIdxAligned * 2);
        // [0.0 1.0 2.0 3.0 4.0 5.0] * 0.5 -> [0 0.5 1.0 1.5 2.0 2.5]
        Muls(offsetFloat, offsetFloat, (float)(5e-1), numIdxAligned * 2);
        // 转成int32_t [0 0 1 1 2 2]
        Cast(offsetY, offsetFloat, RoundMode::CAST_FLOOR, numIdxAligned * 2);
        // [0 0 1 1 2 2] * 2 -> [0 0 2 2 4 4]
        Muls(offsetY, offsetY, (int32_t)2, numIdxAligned * 2);
        // [0 1 2 3 4 5] - [0 0 2 2 4 4] -> [0 1 0 1 0 1]
        Sub(offset, offset, offsetY, numIdxAligned * 2);
        // 得到[0 4N 0 4N 0 4N ... 0 4N]
        Muls(offset, offset, (int32_t)(numIdxAligned * 4), numIdxAligned * 2);
        // [0 0 4 4 8 8 ... 4n-4 4n-4]
        Muls(offsetY, offsetY, (int32_t)2, numIdxAligned * 2);
        // [0 4N 4 4N+4   ... 4n+4n]
        Add(offset, offset, offsetY, numIdxAligned * 2);
    }

    __aicore__ inline void Process()
    {
        // 一个计算核心的任务总量一般部分片段涉及一个batch，另一个部分片段涉及另一个batch，再一个片段涉及再一个batch，因此，需要将任务片段与batch绑定送入compute模块，进行一次AICore 计算
        uint32_t taskOffset = 0;
        for (uint32_t i = 0 ; i < numBatchInCore; i++) {
            if (i > 0) {
                taskOffset += numTaskPerBatch[i-1];
            }
            CopyIn(i, taskOffset, numTaskPerBatch[i]);
            Compute(startBatchId + i, numTaskPerBatch[i], numIdxAligned);
            CopyOut(taskOffset, numTaskPerBatch[i]);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t batchId, uint32_t taskOffset, uint32_t numTask)
    {
        LocalTensor<float> polyLineLocal = inQueuePolyLine.AllocTensor<float>();
        LocalTensor<int32_t> minIdxLocal = inQueueMinIdx.AllocTensor<int32_t>();
        LocalTensor<float> pointsLocal = inQueuePoints.AllocTensor<float>();
        LocalTensor<float> sCumLocal = inQueueSCum.AllocTensor<float>();
        DataCopyPad(polyLineLocal, polyLineGm[batchId * npoint * 2], {1, static_cast<uint32_t>(npoint*2 * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});

        DataCopyPad(minIdxLocal, minIdxGm[taskOffset], {1, static_cast<uint32_t>(numTask * sizeof(int32_t)), 0, 0, 0}, {false, 0, 0, 0});
        DataCopyPad(pointsLocal, pointsGm[taskOffset * 2], {1, static_cast<uint32_t>(numTask * 2 * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
        DataCopyPad(sCumLocal, sCumGm[batchId * npoint], {1, static_cast<uint32_t>(npoint * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});

        inQueuePolyLine.EnQue(polyLineLocal);
        inQueueMinIdx.EnQue(minIdxLocal);
        inQueuePoints.EnQue(pointsLocal);
        inQueueSCum.EnQue(sCumLocal);
    }

    __aicore__ inline void Compute(uint32_t batchId, uint32_t numTask, uint32_t numIdxAligned)
    {
        LocalTensor<float> polyLineLocal = inQueuePolyLine.DeQue<float>();
        LocalTensor<int32_t> minIdxLocal = inQueueMinIdx.DeQue<int32_t>();
        LocalTensor<float> pointsLocal = inQueuePoints.DeQue<float>();
        LocalTensor<float> sCumLocal = inQueueSCum.DeQue<float>();

        LocalTensor<float> polyStart = outQueuePolyStart.AllocTensor<float>();
        LocalTensor<float> polyEnd = outQueuePolyEnd.AllocTensor<float>();
        LocalTensor<float> Sl = outQueueSl.AllocTensor<float>();

        LocalTensor<int32_t> minIdxLocalsCum = minIdxLocalsCumBuf.Get<int32_t>();

        LocalTensor<float> polyStartX = polyStartBufX.Get<float>();
        LocalTensor<float> polyEndX = polyEndBufX.Get<float>();
        LocalTensor<float> subPolyEndX = subPolyEndBufX.Get<float>();
        LocalTensor<float> ptLocalx = pointBufX.Get<float>();
        LocalTensor<float> dotLocalx = dotValueBufX.Get<float>();

        Muls(minIdxLocalsCum, minIdxLocal, (int32_t)4, numTask);
        Muls(minIdxLocal, minIdxLocal, (int32_t)8, numTask);

        Gather(polyStartX, polyLineLocal, minIdxLocal.ReinterpretCast<uint32_t>(), (uint32_t)0, numTask);
        Gather(polyStartX[numIdxAligned], polyLineLocal, minIdxLocal.ReinterpretCast<uint32_t>(), (uint32_t)(0 + 4), numTask);
        Gather(polyEndX, polyLineLocal, minIdxLocal.ReinterpretCast<uint32_t>(), (uint32_t)(0 + 8), numTask);
        Gather(polyEndX[numIdxAligned], polyLineLocal, minIdxLocal.ReinterpretCast<uint32_t>(), (uint32_t)(0 + 12), numTask);

        uint64_t rsvdCnt = 0;
        uint32_t mask = 0;
        uint16_t repeatTimes = (2 * numTask + 63) / 64;
        GatherMask(ptLocalx, pointsLocal, 1, false, mask, { 1, repeatTimes, 8, 8 }, rsvdCnt);
        GatherMask(ptLocalx[numIdxAligned], pointsLocal, 2, false, mask, { 1, repeatTimes, 8, 8 }, rsvdCnt);

        // pt-polyStart_point
        Sub(ptLocalx, ptLocalx, polyStartX, numIdxAligned * 2);

        // polyEnd_point-polyStart_point
        Sub(subPolyEndX, polyEndX, polyStartX, numIdxAligned * 2);

        // x1*x2+y1*y2--->dot
        Mul(dotLocalx, ptLocalx, subPolyEndX, numIdxAligned * 2);
        Add(dotLocalx, dotLocalx, dotLocalx[numIdxAligned], numTask);

        // x1*y2-y1*x2--->cross
        Mul(ptLocalx, ptLocalx, subPolyEndX[numIdxAligned], numTask);
        Mul(ptLocalx[numIdxAligned], ptLocalx[numIdxAligned], subPolyEndX, numTask);
        Sub(dotLocalx[numIdxAligned], ptLocalx, ptLocalx[numIdxAligned], numTask);
        // norm
        Mul(subPolyEndX, subPolyEndX, subPolyEndX, numIdxAligned * 2);
        Add(subPolyEndX, subPolyEndX, subPolyEndX[numIdxAligned], numTask);
        Sqrt(subPolyEndX, subPolyEndX, numTask);

        // clip,mins maxs
        Mins(subPolyEndX, subPolyEndX, static_cast<float>(1e10), numTask);
        Maxs(subPolyEndX, subPolyEndX, static_cast<float>(1e-5), numTask);

        // 计算s, l
        Div(dotLocalx, dotLocalx, subPolyEndX, numTask);
        Div(dotLocalx[numIdxAligned], dotLocalx[numIdxAligned], subPolyEndX, numTask);
        Muls(dotLocalx[numIdxAligned], dotLocalx[numIdxAligned], static_cast<float>(-1), numTask);

        Gather(subPolyEndX, sCumLocal, minIdxLocalsCum.ReinterpretCast<uint32_t>(), (uint32_t)0, numTask);
        Add(dotLocalx, dotLocalx, subPolyEndX, numTask);

        Gather(polyStart, polyStartX, offset.ReinterpretCast<uint32_t>(), (uint32_t)0, numIdxAligned * 2);
        Gather(polyEnd, polyEndX, offset.ReinterpretCast<uint32_t>(), (uint32_t)0, numIdxAligned * 2);
        Gather(Sl, dotLocalx, offset.ReinterpretCast<uint32_t>(), (uint32_t)0, numIdxAligned * 2);

        outQueuePolyStart.EnQue<float>(polyStart);
        outQueuePolyEnd.EnQue<float>(polyEnd);
        outQueueSl.EnQue<float>(Sl);

        inQueuePolyLine.FreeTensor(polyLineLocal);
        inQueueMinIdx.FreeTensor(minIdxLocal);
        inQueuePoints.FreeTensor(pointsLocal);
        inQueueSCum.FreeTensor(sCumLocal);
    }
    __aicore__ inline void CopyOut(uint32_t taskOffset, uint32_t numTask)
    {
        LocalTensor<float> polyStart = outQueuePolyStart.DeQue<float>();
        LocalTensor<float> polyEnd = outQueuePolyEnd.DeQue<float>();
        LocalTensor<float> Sl = outQueueSl.DeQue<float>();

        DataCopyPad(polyStartGm[taskOffset * 2], polyStart, {1, static_cast<uint32_t>(numTask *2 * sizeof(float)), 0, 0, 0});
        DataCopyPad(polyEndGm[taskOffset * 2], polyEnd, {1, static_cast<uint32_t>(numTask *2 * sizeof(float)), 0, 0, 0});
        DataCopyPad(slGm[taskOffset * 2], Sl, {1, static_cast<uint32_t>(numTask *2 * sizeof(float)), 0, 0, 0});

        outQueuePolyStart.FreeTensor(polyStart);
        outQueuePolyEnd.FreeTensor(polyEnd);
        outQueueSl.FreeTensor(Sl);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueuePolyLine, inQueueMinIdx, inQueuePoints, inQueueSCum;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueuePolyStart, outQueuePolyEnd, outQueueSl;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> minIdxLocalsCumBuf, polyStartBufX, polyEndBufX, subPolyEndBufX, dotValueBufX, \
                                                 pointBufX, offsetBufX, offsetBufY, offsetFloatBuf;
    AscendC::GlobalTensor<float> polyLineGm;
    AscendC::GlobalTensor<int32_t> minIdxGm;
    AscendC::GlobalTensor<float> pointsGm;
    AscendC::GlobalTensor<float> sCumGm;
    AscendC::GlobalTensor<float> polyStartGm;
    AscendC::GlobalTensor<float> polyEndGm;
    AscendC::GlobalTensor<float> slGm;
    AscendC::LocalTensor<int32_t> offset;
    uint32_t batchSize, npoint, numIdx, numIdxAligned, numTaskPerCore, numTaskRemained, numTaskInCore, startTaskId, startBatchId, numBatchInCore;
    uint32_t numTaskPerBatch[1000] = {0};
};

extern "C" __global__ __aicore__ void calc_poly_start_end_sl(GM_ADDR min_idx, GM_ADDR poly_line, GM_ADDR pt, GM_ADDR s_cum, GM_ADDR poly_start, GM_ADDR poly_end, GM_ADDR sl, GM_ADDR workspace, GM_ADDR tiling_data)
{
    GET_TILING_DATA(tiling, tiling_data);
    SetSysWorkspace(workspace);
    CalcPolyStartEndSl op;
    op.Init(min_idx, poly_line, pt, s_cum, poly_start, poly_end, sl, &tiling);
    op.Process();
}
