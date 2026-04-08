/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 *
 * Function : z = x + y
 * This sample is a very basic sample that implements vector add on Ascend plaform.
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;
constexpr uint32_t PT_BYTE_SIZE = sizeof(float);
constexpr uint32_t IDX_BYTE_SIZE = sizeof(int32_t);
constexpr uint32_t DATA_BLOCK_SIZE = 32;
constexpr uint32_t REDUCE_LENGTH = 64;
constexpr uint32_t MAX_REPEATS = 255;
constexpr uint32_t IDX_ALIGN_NUM = DATA_BLOCK_SIZE / IDX_BYTE_SIZE;
constexpr uint32_t PT_ALIGN_NUM = DATA_BLOCK_SIZE / PT_BYTE_SIZE;
constexpr uint32_t CMP_ALIGN_NUM = 256 / IDX_BYTE_SIZE;
constexpr uint32_t MAX_LOOP = 12;
class CartesianToFrenet1 {
public:
    __aicore__ inline CartesianToFrenet1() {}
    __aicore__ inline void Init(const GM_ADDR dist_vec, const GM_ADDR min_idx, const GM_ADDR back_idx, const CartesianToFrenet1TilingData *tiling)
    {
        InitTiling(tiling);
        InitGM(dist_vec, min_idx, back_idx);
        InitUB();
        InitEvent();
    }
    __aicore__ inline void Process()
    {
        if (coreId < usedCoreNum) {
            tileNum = tileRemainder > 0 ? tileNum + 1 : tileNum;
            for (int32_t i = 0; i < tileNum; i ++) {
                if (i == tileNum - 1 && tileRemainder > 0) {
                    tileLength = tileRemainder * taskSizeAligned;
                    tileTaskNum = tileRemainder;
                }
                SetFlag<HardEvent::V_MTE2>(eventVMTE2_);
                WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);

                CopyIn(i, tileLength, tileTaskNum, taskSize);

                SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
                WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);

                SetFlag<HardEvent::MTE3_V>(eventMTE3V_);
                WaitFlag<HardEvent::MTE3_V>(eventMTE3V_);

                Compute(tileTaskNum, taskSizeAligned, numPolyLinePoints);

                SetFlag<HardEvent::V_MTE3>(eventVMTE3_);
                WaitFlag<HardEvent::V_MTE3>(eventVMTE3_);

                CopyOut(i, tileTaskNum);
            }
        }
    }
private:
    __aicore__ inline void InitTiling(const CartesianToFrenet1TilingData *tiling)
    {
        this->copyInAlignNum = tiling->copyInAlignNum;
        this->dstStride = tiling->dstStride;
        this->rightPadding = tiling->rightPadding;
        this->numPolyLinePoints = tiling->numPolyLinePoints;
        this->pointDim = tiling->pointDim;
        this->taskSize = tiling->taskSize;
        this->taskSizeElem = tiling->taskSizeElem;
        this->taskSizeAligned = tiling->taskSizeAligned;
        this->bigCoreCount = tiling->bigCoreCount;
        this->usedCoreNum = tiling->usedCoreNum;
        this->avgTaskNum = tiling->avgTaskNum;
        this->coreId = GetBlockIdx();

        this->tileNum = coreId < bigCoreCount ? tiling->formerTileNum : tiling->tailTileNum;
        this->tileRemainder = coreId < bigCoreCount ? tiling->formerTileRemainder : tiling->tailTileRemainder;
        this->taskResultSizeAligned = coreId < bigCoreCount ? tiling->taskResultSizeAligned_b : tiling->taskResultSizeAligned_s;
        this->axisSizeAligned = coreId < bigCoreCount ? tiling->axisSizeAligned_b : tiling->axisSizeAligned_s;
        this->numTaskCurCore = coreId < bigCoreCount ? tiling->numTaskCurCore_b : tiling->avgTaskNum;
        this->TaskLengthCurCore = coreId < bigCoreCount ? tiling->TaskLengthCurCore_b : tiling->TaskLengthCurCore_s;
        this->tileTaskNum = coreId < bigCoreCount ? tiling->tileTaskNum_b : tiling->tileTaskNum_s;

        this->formerTileTaskNum = this->tileTaskNum;
        this->tileLength = tileTaskNum * taskSizeAligned;

        if (coreId < bigCoreCount) {
            startTaskId = TaskLengthCurCore * coreId;
            startResId = numTaskCurCore * coreId;
        } else {
            startTaskId = (avgTaskNum + 1) * taskSizeElem * bigCoreCount + numTaskCurCore * taskSizeElem * (coreId - bigCoreCount);
            startResId = (avgTaskNum + 1) * bigCoreCount + avgTaskNum * (coreId - bigCoreCount);
        }
    }
    __aicore__ inline void InitEvent()
    {
        eventMTE2V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        eventVMTE3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        eventMTE3V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        eventVMTE2_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    }
    __aicore__ inline void InitGM(const GM_ADDR distVec, const GM_ADDR minIdx, const GM_ADDR backIdx)
    {
        this->distVecGm.SetGlobalBuffer((__gm__ float*) distVec + startTaskId,
            TaskLengthCurCore);
        this->minIdxGm.SetGlobalBuffer((__gm__ int32_t*) minIdx + startResId,
            numTaskCurCore);
        this->backIdxGm.SetGlobalBuffer((__gm__ int32_t*) backIdx + startResId,
            numTaskCurCore);
    }
    __aicore__ inline void InitUB()
    {
        pipe.InitBuffer(inQueueDistVec, tileLength);
        pipe.InitBuffer(outQueueMinIdx, taskResultSizeAligned);
        pipe.InitBuffer(outQueueBackIdx, taskResultSizeAligned);
        pipe.InitBuffer(xBuffer, axisSizeAligned);
        pipe.InitBuffer(yBuffer, axisSizeAligned + 255);
        pipe.InitBuffer(minRecBuffer, tileTaskNum * IDX_BYTE_SIZE);
        pipe.InitBuffer(minTempBuffer, axisSizeAligned);
        pipe.InitBuffer(workerBuffer, axisSizeAligned);
        pipe.InitBuffer(minTempVal, axisSizeAligned);

        minIdxLocal = outQueueMinIdx.Get<int32_t>();
        backIdxLocal = outQueueBackIdx.Get<int32_t>();
        distVecLocal = inQueueDistVec.Get<float>();
        ones = minRecBuffer.Get<int32_t>();
        xLocal = xBuffer.Get<float>();
        yLocal = yBuffer.Get<float>();
        minTempLocal = minTempBuffer.Get<float>();
        workerLocal = workerBuffer.Get<float>();
        xLocalAligned = minTempVal.Get<float>();
    }
    __aicore__ inline void CopyIn(int32_t curTile, uint32_t tileLength, uint32_t tileTaskNum, uint32_t taskSize)
    {
        DataCopyPad(distVecLocal, distVecGm[curTile * formerTileTaskNum * taskSizeElem], {(uint16_t)tileTaskNum, (uint32_t)taskSize, 0, dstStride, 0}, {true, 0, static_cast<uint8_t>(rightPadding), 0});
    }
    __aicore__ inline void Compute(uint32_t tileTaskNum, uint32_t taskSizeAligned, uint32_t numPolyLinePoints)
    {
        uint32_t numPolyLinePointsAligned = tileTaskNum * ((numPolyLinePoints + PT_ALIGN_NUM - 1) / PT_ALIGN_NUM * PT_ALIGN_NUM);
        uint32_t numPolyLinePointsDimAligned = tileTaskNum * copyInAlignNum / PT_BYTE_SIZE;
        uint32_t numPolyLinePointsAlignedEach = (numPolyLinePoints + PT_ALIGN_NUM - 1) / PT_ALIGN_NUM * PT_ALIGN_NUM;

        uint64_t rsvdCnt = 0;
        uint8_t xPattern = 2;  // 10101010
        uint8_t yPattern = 1;  // 01010101

        // torch.norm()
        Mul(distVecLocal, distVecLocal, distVecLocal, numPolyLinePointsDimAligned);

        GatherMask(xLocal, distVecLocal, xPattern, true, numPolyLinePointsDimAligned, {1, 1, 0, 0}, rsvdCnt);
        GatherMask(yLocal, distVecLocal, yPattern, true, numPolyLinePointsDimAligned, {1, 1, 0, 0}, rsvdCnt);

        Add(xLocal, xLocal, yLocal, numPolyLinePointsAligned);

        Sqrt(xLocal, xLocal, numPolyLinePointsAligned);

        // finding argmin for each point
        for (uint32_t i = 0; i < tileTaskNum; i ++) {
            ReduceMin(minTempLocal[i * 2], xLocal[i * numPolyLinePointsAlignedEach], workerLocal, numPolyLinePoints, true);
        }

        GatherMask(yLocal, minTempLocal, xPattern, true, tileTaskNum * 2, {1, 1, 0, 0}, rsvdCnt);

        minIdxLocal = yLocal.ReinterpretCast<int32_t>();

        // calculating back_idx = min_idx - 1 < 0 ? 0 : min_idx - 1
        Duplicate(ones, 1, numPolyLinePointsAligned);

        Sub(backIdxLocal, minIdxLocal, ones, tileTaskNum);

        Relu(backIdxLocal, backIdxLocal, tileTaskNum);
    }
    __aicore__ inline void CopyOut(uint32_t curTile, uint32_t tileTaskNum)
    {
        DataCopyPad(minIdxGm[curTile * formerTileTaskNum], minIdxLocal, {1, static_cast<uint16_t>(tileTaskNum * IDX_BYTE_SIZE), 0, 0});
        DataCopyPad(backIdxGm[curTile * formerTileTaskNum], backIdxLocal, {1, static_cast<uint16_t>(tileTaskNum * IDX_BYTE_SIZE), 0, 0});
    }
private:
    TPipe pipe;
    GlobalTensor<float> distVecGm;
    GlobalTensor<int32_t> minIdxGm;
    GlobalTensor<int32_t> backIdxGm;
    LocalTensor<int32_t> minIdxLocal, backIdxLocal, ones;
    LocalTensor<float> distVecLocal, xLocal, yLocal, minTempLocal, workerLocal, xLocalAligned;
    int32_t eventMTE2V_, eventVMTE3_, eventMTE3V_, eventVMTE2_;
    uint32_t startTaskId, startResId;
    uint32_t copyInAlignNum, dstStride, rightPadding, numPolyLinePoints, pointDim, taskSize, taskSizeElem, taskSizeAligned, bigCoreCount,
    usedCoreNum, avgTaskNum, coreId, tileNum, tileRemainder, taskResultSizeAligned, axisSizeAligned,
    numTaskCurCore, TaskLengthCurCore, tileTaskNum, formerTileTaskNum, tileLength;
    TBuf<TPosition::VECCALC> xBuffer, yBuffer, minRecBuffer, outQueueMinIdx, outQueueBackIdx, minTempBuffer, workerBuffer, minTempVal, inQueueDistVec;
};
extern "C" __global__ __aicore__ void cartesian_to_frenet1(GM_ADDR distVec, GM_ADDR minIdx,
    GM_ADDR backIdx, GM_ADDR workspace, GM_ADDR tiling_data)
{
    GET_TILING_DATA(tiling, tiling_data);
    SetSysWorkspace(workspace);
    CartesianToFrenet1 op;
    op.Init(distVec, minIdx, backIdx, &tiling);
    op.Process();
}