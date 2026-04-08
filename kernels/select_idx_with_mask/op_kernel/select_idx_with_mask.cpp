/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

class KernelSelectIdxWithMask {
public:
    __aicore__ inline KernelSelectIdxWithMask() {}
    __aicore__ inline void Init(GM_ADDR poly_line, GM_ADDR min_idx, GM_ADDR pt, GM_ADDR back_idx, GM_ADDR out_min_idx, SelectIdxWithMaskTilingData* tiling)
    {
        batchSize = tiling->batchSize;
        numPoint = tiling->numPoint;
        numIdx = tiling->numIdx;
        compBatchNum = tiling->compBatchNum;
        numTaskPerCore = tiling->numTaskPerCore;
        numTaskTail = tiling->numTaskTail;

        uint32_t blockSize = 32;
        uint32_t dataSize = blockSize / sizeof(float);

        numPointAligned = (numPoint + dataSize - 1) / dataSize * dataSize;
        numIdxAligned = (numIdx + dataSize - 1) / dataSize * dataSize;
        numIdxRound64 = (numIdx + 63) / 64 * 64;

        uint32_t coreId = GetBlockIdx();
        if (coreId < numTaskTail) {
            numTaskPerCore = numTaskPerCore + 1;
            startTaskId = numTaskPerCore * coreId;
        } else {
            startTaskId = numTaskPerCore * coreId + numTaskTail;
        }
        compBatchNum = min(numTaskPerCore, compBatchNum);

        ubLengthBatchPointSize = compBatchNum * numPointAligned;
        ubLengthBatchIdxSize = compBatchNum * numIdxAligned;
        ubLengthBatchIdxSizeRound64 = compBatchNum * numIdxRound64;

        polyLineGm.SetGlobalBuffer((__gm__ float*)poly_line, batchSize * numPoint * 2);
        minIdxGm.SetGlobalBuffer((__gm__ int32_t*)min_idx, batchSize * numIdx);
        ptGm.SetGlobalBuffer((__gm__ float*)pt, batchSize * numIdx * 2);
        backIdxGm.SetGlobalBuffer((__gm__ int32_t*)back_idx, batchSize * numIdx);
        outMinIdxGm.SetGlobalBuffer((__gm__ int32_t*)out_min_idx, batchSize * numIdx);

        InitBuffer();
        InitLocalTensor();
    }

    __aicore__ inline void Process()
    {
        uint64_t rsvdCnt = 0;
        for (uint32_t batchIdx = compBatchNum; batchIdx > 0; batchIdx--) {
            Duplicate(batchIdxOffset, (int32_t)((batchIdx - 1) * numPoint), batchIdx * numIdx);
        }
        for (uint32_t loopTimes = 0 ; loopTimes < numTaskPerCore; loopTimes += compBatchNum) {
            uint32_t batchIdx = loopTimes + startTaskId;
            uint32_t compBatchNumActual = loopTimes + compBatchNum <= numTaskPerCore ? compBatchNum : numTaskPerCore - loopTimes;

            uint32_t compBatchPointSize = compBatchNumActual * numPoint;
            uint32_t compBatchIdxSize = compBatchNumActual * numIdx;
            uint32_t compBatchIdxRound64Size = compBatchNumActual * numIdxRound64;
            uint16_t gatherRepeatTimes = (2 * compBatchNumActual * numIdx + 63) / 64;

            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

            DataCopyPad(polyLineLocal, polyLineGm[batchIdx * numPoint * 2], {1, static_cast<uint32_t>(compBatchPointSize * 2 * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            DataCopyPad(minIdxLocal, minIdxGm[batchIdx * numIdx], {1, static_cast<uint32_t>(compBatchIdxSize * sizeof(int32_t)), 0, 0, 0}, {false, 0, 0, 0});
            DataCopyPad(ptLocal, ptGm[batchIdx * numIdx * 2], {1, static_cast<uint32_t>(compBatchIdxSize * 2 * sizeof(float)), 0, 0, 0}, {false, 0, 0, 0});
            DataCopyPad(backidxLocal, backIdxGm[batchIdx * numIdx], {1, static_cast<uint32_t>(compBatchIdxSize * sizeof(int32_t)), 0, 0, 0}, {false, 0, 0, 0});

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            BatchIdxSelect(frontPoint, minIdxLocal, polyLineLocal, ubLengthBatchIdxSize, compBatchIdxSize);
            BatchIdxSelect(backPoint, backidxLocal, polyLineLocal, ubLengthBatchIdxSize, compBatchIdxSize);
            GatherMask(ptGather, ptLocal, 1, false, 0, {1, gatherRepeatTimes, 8, 8}, rsvdCnt);
            GatherMask(ptGather[ubLengthBatchIdxSize], ptLocal, 2, false, 0, {1, gatherRepeatTimes, 8, 8}, rsvdCnt);

            // x1 * x2 + y1 * y2 ---> dot;
            Sub(ptGather, ptGather, frontPoint, 2 * ubLengthBatchIdxSize);
            Sub(backPoint, backPoint, frontPoint, 2 * ubLengthBatchIdxSize);
            Mul(dotValue, ptGather, backPoint, 2 * ubLengthBatchIdxSize);
            Add(dotValue, dotValue, dotValue[ubLengthBatchIdxSize], compBatchIdxSize);

            Cast(minIdxLocalToFloat, minIdxLocal, AscendC::RoundMode::CAST_NONE, compBatchIdxRound64Size);
            CompareScalar(minIdxCompareWithZero, minIdxLocalToFloat, static_cast<float>(0), AscendC::CMPMODE::GT, compBatchIdxRound64Size);
            CompareScalar(dotValueCompareWithZero, dotValue, static_cast<float>(0), AscendC::CMPMODE::GT, compBatchIdxRound64Size);
            And(andResult, dotValueCompareWithZero.ReinterpretCast<uint16_t>(), minIdxCompareWithZero.ReinterpretCast<uint16_t>(), compBatchIdxRound64Size);
            CompareScalar(eqResult, minIdxLocal, static_cast<int32_t>(numPoint - 1), AscendC::CMPMODE::EQ, compBatchIdxRound64Size);
            Or(orResult, andResult, eqResult, compBatchIdxRound64Size);

            Adds(minIdxSubOne, minIdxLocal, static_cast<int32_t>(-1), compBatchIdxRound64Size);

            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

            Select(outMinIdxLocal.ReinterpretCast<float>(), orResult, minIdxSubOne.ReinterpretCast<float>(), minIdxLocal.ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE, compBatchIdxRound64Size);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            DataCopyPad(outMinIdxGm[batchIdx * numIdx], outMinIdxLocal, {1, static_cast<uint32_t>(compBatchIdxSize * sizeof(int32_t)), 0, 0, 0});
        }
    }

private:
    __aicore__ inline void InitBuffer()
    {
        pipe.InitBuffer(polyLineBuf, ubLengthBatchPointSize * 2 * sizeof(float));
        pipe.InitBuffer(minIdxBuf, ubLengthBatchIdxSize * sizeof(int32_t));
        pipe.InitBuffer(ptBuf, ubLengthBatchIdxSize * 2 * sizeof(float));
        pipe.InitBuffer(backIdxBuf, ubLengthBatchIdxSize * sizeof(int32_t));

        pipe.InitBuffer(selectIdxBuf, ubLengthBatchIdxSize * sizeof(int32_t));
        pipe.InitBuffer(batchIdxOffsetBuf, ubLengthBatchIdxSize * sizeof(int32_t));

        pipe.InitBuffer(frontPointBuf, 2 * ubLengthBatchIdxSize * sizeof(float));
        pipe.InitBuffer(backPointBuf, 2 * ubLengthBatchIdxSize * sizeof(float));
        pipe.InitBuffer(ptGatherBuf, 2 * ubLengthBatchIdxSizeRound64 * sizeof(float));
        pipe.InitBuffer(dotValueBuf, 2 * ubLengthBatchIdxSize * sizeof(float));

        pipe.InitBuffer(minIdxLocalToFloatBuf, ubLengthBatchIdxSizeRound64 * sizeof(float));
        pipe.InitBuffer(dotValueCompareWithZeroBuf, ubLengthBatchIdxSizeRound64 * sizeof(uint8_t));
        pipe.InitBuffer(minIdxCompareWithZeroBuf, ubLengthBatchIdxSizeRound64 * sizeof(uint8_t));

        pipe.InitBuffer(andResultBuf, ubLengthBatchIdxSizeRound64 * sizeof(uint16_t));
        pipe.InitBuffer(eqResultBuf, ubLengthBatchIdxSizeRound64 * sizeof(uint16_t));
        pipe.InitBuffer(orResultBuf, ubLengthBatchIdxSizeRound64 * sizeof(uint16_t));

        pipe.InitBuffer(minIdxSubOneBuf, ubLengthBatchIdxSizeRound64 * sizeof(int32_t));
        pipe.InitBuffer(outMinIdxBuf, ubLengthBatchIdxSizeRound64 * sizeof(int32_t));
    }

    __aicore__ inline void InitLocalTensor()
    {
        polyLineLocal = polyLineBuf.Get<float>();
        minIdxLocal = minIdxBuf.Get<int32_t>();
        ptLocal = ptBuf.Get<float>();
        backidxLocal = backIdxBuf.Get<int32_t>();

        selectIdx = selectIdxBuf.Get<int32_t>();
        batchIdxOffset = batchIdxOffsetBuf.Get<int32_t>();

        frontPoint = frontPointBuf.Get<float>();
        backPoint = backPointBuf.Get<float>();
        ptGather = ptGatherBuf.Get<float>();
        dotValue = dotValueBuf.Get<float>();

        minIdxLocalToFloat = minIdxLocalToFloatBuf.Get<float>();
        dotValueCompareWithZero = dotValueCompareWithZeroBuf.Get<uint8_t>();
        minIdxCompareWithZero = minIdxCompareWithZeroBuf.Get<uint8_t>();

        andResult = andResultBuf.Get<uint16_t>();
        eqResult = eqResultBuf.Get<uint16_t>();
        orResult = orResultBuf.Get<uint16_t>();

        minIdxSubOne = minIdxSubOneBuf.Get<int32_t>();
        outMinIdxLocal = outMinIdxBuf.Get<int32_t>();
    }

    __aicore__ inline void BatchIdxSelect(LocalTensor<float> dstLocal, LocalTensor<int32_t> idxLocal, LocalTensor<float> srcLocal, int32_t dstOffset, int32_t compNum)
    {
        Add(selectIdx, idxLocal, batchIdxOffset, compNum);
        Muls(selectIdx, selectIdx, (int32_t)(2 * sizeof(float)), compNum);
        Gather(dstLocal, srcLocal, selectIdx.ReinterpretCast<uint32_t>(), (uint32_t)0, compNum);
        Adds(selectIdx, selectIdx, (int32_t)(sizeof(float)), compNum);
        Gather(dstLocal[dstOffset], srcLocal, selectIdx.ReinterpretCast<uint32_t>(), (uint32_t)0, compNum);
    }
    
private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> polyLineBuf, minIdxBuf, ptBuf, backIdxBuf, outMinIdxBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> frontPointBuf, backPointBuf, ptGatherBuf, dotValueBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> dotValueCompareWithZeroBuf, minIdxCompareWithZeroBuf, minIdxLocalToFloatBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> andResultBuf, eqResultBuf, orResultBuf, minIdxSubOneBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> selectIdxBuf, batchIdxOffsetBuf;
    AscendC::GlobalTensor<float> polyLineGm;
    AscendC::GlobalTensor<int32_t> minIdxGm;
    AscendC::GlobalTensor<float> ptGm;
    AscendC::GlobalTensor<int32_t> backIdxGm;
    AscendC::GlobalTensor<int32_t> outMinIdxGm;

    LocalTensor<float> polyLineLocal, ptLocal;
    LocalTensor<int32_t> minIdxLocal, backidxLocal, outMinIdxLocal;
    LocalTensor<float> frontPoint, backPoint, ptGather, dotValue;
    LocalTensor<float> minIdxLocalToFloat;
    LocalTensor<uint8_t> dotValueCompareWithZero, minIdxCompareWithZero;
    LocalTensor<uint16_t> andResult, eqResult, orResult;
    LocalTensor<int32_t> minIdxSubOne;
    LocalTensor<int32_t> selectIdx, batchIdxOffset;

    uint32_t batchSize, numPoint, numIdx;
    uint32_t numTaskPerCore, numTaskTail, startTaskId;
    uint32_t numPointAligned, numIdxAligned, numIdxRound64;
    uint32_t ubLengthBatchPointSize, ubLengthBatchIdxSize, ubLengthBatchIdxSizeRound64;
    uint32_t compBatchNum;
};

extern "C" __global__ __aicore__ void select_idx_with_mask(GM_ADDR poly_line, GM_ADDR min_idx, GM_ADDR pt, GM_ADDR back_idx, GM_ADDR out_min_idx, GM_ADDR workspace, GM_ADDR tiling_data)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling, tiling_data);
    SetSysWorkspace(workspace);
    KernelSelectIdxWithMask op;
    op.Init(poly_line, min_idx, pt, back_idx, out_min_idx, &tiling);
    op.Process();
}