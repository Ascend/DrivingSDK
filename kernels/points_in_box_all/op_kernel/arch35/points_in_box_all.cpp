/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelPointsInBoxAll {
  public:
    __aicore__ inline KernelPointsInBoxAll() {}

    __aicore__ inline void Init(
        GM_ADDR boxes, GM_ADDR pts, GM_ADDR boxes_idx_of_points, PointsInBoxAllTilingData *tiling_data) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zeronumber!");
        usedCoreNum = tiling_data->usedCoreNum;
        coreData = tiling_data->coreData;
        copyLoop = tiling_data->copyLoop;
        copyTail = tiling_data->copyTail;
        lastCopyLoop = tiling_data->lastCopyLoop;
        lastCopyTail = tiling_data->lastCopyTail;
        npoints = tiling_data->npoints;
        boxNumber = tiling_data->boxNumber;
        availableUbSize = tiling_data->availableUbSize;
        batchSize = tiling_data->batchSize;
        boxNumLoop = availableUbSize;

        ptsGm.SetGlobalBuffer((__gm__ DTYPE_PTS *)pts + GetBlockIdx() * coreData * 3, coreData * 3);
        boxesGm.SetGlobalBuffer((__gm__ DTYPE_PTS *)boxes, boxNumber * 7 * batchSize);
        outputGm.SetGlobalBuffer(
            (__gm__ DTYPE_BOXES_IDX_OF_POINTS *)boxes_idx_of_points + GetBlockIdx() * coreData * boxNumber,
            coreData * boxNumber);
        pipe.InitBuffer(inQueuePTS, BUFFER_NUM, availableUbSize * 3 * 8 * sizeof(DTYPE_PTS));
        pipe.InitBuffer(inQueueBOXES, BUFFER_NUM, availableUbSize * 7 * sizeof(DTYPE_PTS));
        pipe.InitBuffer(outQueueOUTPUT, 1, availableUbSize * boxNumLoop * sizeof(DTYPE_BOXES_IDX_OF_POINTS));
        pipe.InitBuffer(shiftxque, availableUbSize * boxNumLoop * sizeof(DTYPE_PTS));
        pipe.InitBuffer(shiftyque, availableUbSize * boxNumLoop * sizeof(DTYPE_PTS));
        pipe.InitBuffer(cosaque, availableUbSize * boxNumLoop * sizeof(DTYPE_PTS));
        pipe.InitBuffer(sinaque, availableUbSize * boxNumLoop * sizeof(DTYPE_PTS));
        pipe.InitBuffer(xLocalque, availableUbSize * boxNumLoop * sizeof(DTYPE_PTS));
        pipe.InitBuffer(yLocalque, availableUbSize * boxNumLoop * sizeof(DTYPE_PTS));
        pipe.InitBuffer(tempque, availableUbSize * boxNumLoop * sizeof(DTYPE_PTS));
        pipe.InitBuffer(uint8que, availableUbSize * boxNumLoop * sizeof(DTYPE_PTS));
    }

    __aicore__ inline void Process() {
        uint32_t coreIdx = GetBlockIdx();
        if (coreIdx > usedCoreNum - 1) {
            return;
        }
        if (coreIdx != (usedCoreNum - 1)) {
            for (int32_t i = 0; i < copyLoop; i++) {
                ComputeDiffBatch(i, availableUbSize, coreIdx, i + 1, 0);
            }
            if (copyTail != 0) {
                ComputeDiffBatch(copyLoop, copyTail, coreIdx, copyLoop, copyTail);
            }
        } else {
            for (int32_t i = 0; i < lastCopyLoop; i++) {
                ComputeDiffBatch(i, availableUbSize, coreIdx, i + 1, 0);
            }
            if (lastCopyTail != 0) {
                ComputeDiffBatch(lastCopyLoop, lastCopyTail, coreIdx, lastCopyLoop, lastCopyTail);
            }
        }
    }

  private:
    __aicore__ inline void ComputeDiffBatch(
        int32_t progress, int32_t dataNum, uint32_t coreIdx, int32_t copyLoopOffset, uint32_t copyTailOffset) {
        uint64_t addressPoints = progress * availableUbSize;
        uint64_t addressOutput = progress * availableUbSize * boxNumber;
        int32_t coreBatchIdx = (coreIdx * coreData + progress * availableUbSize) / npoints;
        int32_t tail_num =
            coreIdx * coreData + copyLoopOffset * availableUbSize + copyTailOffset - (coreBatchIdx + 1) * npoints;

        if (tail_num < 0) {
            ComputeBox(progress, dataNum, coreBatchIdx, addressOutput, addressPoints);
        } else {
            int32_t head_num = dataNum - tail_num;
            ComputeBox(progress, head_num, coreBatchIdx, addressOutput, addressPoints);
            coreBatchIdx++;
            addressPoints += head_num;
            addressOutput += head_num * boxNumber;

            while (tail_num > npoints) {
                ComputeBox(progress, npoints, coreBatchIdx, addressOutput, addressPoints);
                tail_num -= npoints;
                addressPoints += npoints;
                addressOutput += npoints * boxNumber;
                coreBatchIdx++;
            }
            ComputeBox(progress, tail_num, coreBatchIdx, addressOutput, addressPoints);
        }
    }

    __aicore__ inline void ComputeBox(
        int32_t progress, int32_t dataNum, int32_t coreBatchIdx, uint64_t addressOutput, uint64_t addressPoints) {
        int32_t computeBoxNum = boxNumber;
        uint32_t boxCopyAddress = 0;
        uint32_t copyOutStride = (computeBoxNum > boxNumLoop) ? (computeBoxNum - boxNumLoop) : 0;
        uint32_t copyOutStrideTail = AlignUp(computeBoxNum, boxNumLoop) - boxNumLoop;
        uint32_t outAddressOffset = 0;

        while (computeBoxNum > boxNumLoop) {
            CopyBox(coreBatchIdx, boxNumLoop, boxCopyAddress);
            PipeBarrier<PIPE_ALL>();
            Compute(progress, dataNum, addressOutput, addressPoints, boxNumLoop, copyOutStride, outAddressOffset);
            boxCopyAddress += boxNumLoop;
            computeBoxNum -= boxNumLoop;
            outAddressOffset += boxNumLoop;
        }
        CopyBox(coreBatchIdx, computeBoxNum, boxCopyAddress);
        PipeBarrier<PIPE_ALL>();
        Compute(progress, dataNum, addressOutput, addressPoints, computeBoxNum, copyOutStrideTail, outAddressOffset);
    }

    __aicore__ inline void CopyBox(uint32_t boxCopyBatch, uint32_t boxCopyNum, uint32_t boxCopyAddress) {
        boxesLocalCx = inQueueBOXES.AllocTensor<DTYPE_BOXES>();
        boxesLocalCy = boxesLocalCx[availableUbSize];
        boxesLocalCz = boxesLocalCx[availableUbSize * 2];
        boxesLocalDx = boxesLocalCx[availableUbSize * 3];
        boxesLocalDy = boxesLocalCx[availableUbSize * 4];
        boxesLocalDz = boxesLocalCx[availableUbSize * 5];
        boxesLocalRz = boxesLocalCx[availableUbSize * 6];

        boxCopyNum = static_cast<int32_t>((boxCopyNum * sizeof(DTYPE_BOXES) + 32 - 1) / 32) * 32 / sizeof(DTYPE_BOXES);
        DataCopyParams copyParams_box{1, (uint16_t)(boxCopyNum * sizeof(DTYPE_BOXES)), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};

        DataCopyPad(boxesLocalCx, boxesGm[boxNumber * boxCopyBatch * 7 + boxCopyAddress], copyParams_box, padParams);
        DataCopyPad(
            boxesLocalCy, boxesGm[boxNumber * (boxCopyBatch * 7 + 1) + boxCopyAddress], copyParams_box, padParams);
        DataCopyPad(
            boxesLocalCz, boxesGm[boxNumber * (boxCopyBatch * 7 + 2) + boxCopyAddress], copyParams_box, padParams);
        DataCopyPad(
            boxesLocalDx, boxesGm[boxNumber * (boxCopyBatch * 7 + 3) + boxCopyAddress], copyParams_box, padParams);
        DataCopyPad(
            boxesLocalDy, boxesGm[boxNumber * (boxCopyBatch * 7 + 4) + boxCopyAddress], copyParams_box, padParams);
        DataCopyPad(
            boxesLocalDz, boxesGm[boxNumber * (boxCopyBatch * 7 + 5) + boxCopyAddress], copyParams_box, padParams);
        DataCopyPad(
            boxesLocalRz, boxesGm[boxNumber * (boxCopyBatch * 7 + 6) + boxCopyAddress], copyParams_box, padParams);
    }

    __aicore__ inline void Compute(int32_t progress, uint32_t tensorSize, uint64_t addressOutput,
        uint64_t addressPoints, uint32_t computeBoxNumOri, uint32_t copyOutStride, uint32_t outAddressOffset) {
        pointLocalx = inQueuePTS.AllocTensor<DTYPE_PTS>();
        pointLocaly = pointLocalx[availableUbSize * 8];
        pointLocalz = pointLocalx[availableUbSize * 8 * 2];
        zLocal = outQueueOUTPUT.AllocTensor<DTYPE_BOXES_IDX_OF_POINTS>();
        shiftx = shiftxque.Get<DTYPE_BOXES>();
        shifty = shiftyque.Get<DTYPE_BOXES>();
        cosa = cosaque.Get<DTYPE_BOXES>();
        sina = sinaque.Get<DTYPE_BOXES>();
        xLocal = xLocalque.Get<DTYPE_BOXES>();
        yLocal = yLocalque.Get<DTYPE_BOXES>();
        temp = tempque.Get<DTYPE_BOXES>();
        uint8temp = uint8que.Get<uint8_t>();
        DataCopyExtParams copyParams_out{static_cast<uint16_t>(tensorSize),
            (uint32_t)(computeBoxNumOri * sizeof(DTYPE_BOXES_IDX_OF_POINTS)), 0,
            (uint32_t)(copyOutStride * sizeof(DTYPE_BOXES_IDX_OF_POINTS)), 0};
        uint32_t computeBoxNum =
            static_cast<int32_t>((computeBoxNumOri * sizeof(DTYPE_BOXES_IDX_OF_POINTS) + 32 - 1) / 32) * 32 /
            sizeof(DTYPE_BOXES_IDX_OF_POINTS);
        DataCopyPadParams padParams{false, 0, 0, 0};

        // move points to localtensor
        DataCopyParams copyParams_in{static_cast<uint16_t>(tensorSize), (uint16_t)(1 * sizeof(DTYPE_BOXES)),
            (uint16_t)(2 * sizeof(DTYPE_BOXES)), 0};
        DataCopyPad(pointLocalx, ptsGm[addressPoints * 3], copyParams_in, padParams);
        DataCopyPad(pointLocaly, ptsGm[addressPoints * 3 + 1], copyParams_in, padParams);
        DataCopyPad(pointLocalz, ptsGm[addressPoints * 3 + 2], copyParams_in, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        uint32_t dstShape[2] = {tensorSize, computeBoxNum};
        uint32_t srcShape[2] = {1, computeBoxNum};
        // broadcast Rz to xLocal
        BroadCast<DTYPE_BOXES, 2, 0>(xLocal, boxesLocalRz, dstShape, srcShape);
        // cosa = Cos(-boxes_ub[ :, 6]) sina = Sin(-boxes_ub[ :, 6])
        Muls(temp, xLocal, -1, computeBoxNum * tensorSize);
        Cos<DTYPE_BOXES, false>(cosa, temp, uint8temp, computeBoxNum * tensorSize);
        Sin<DTYPE_BOXES, false>(sina, temp, uint8temp, computeBoxNum * tensorSize);
        PipeBarrier<PIPE_V>();

        ComputePointsInBox(tensorSize, computeBoxNum);

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(outputGm[addressOutput + outAddressOffset], zLocal, copyParams_out);
        inQueuePTS.FreeTensor(pointLocalx);
        inQueueBOXES.FreeTensor(boxesLocalCx);
        outQueueOUTPUT.FreeTensor(zLocal);
    }

    __aicore__ inline void ComputePointsInBox(uint16_t tensorSize, uint32_t computeBoxNum) {
        __local_mem__ float *pointXPtr = (__local_mem__ float *)pointLocalx.GetPhyAddr();
        __local_mem__ float *pointYPtr = (__local_mem__ float *)pointLocaly.GetPhyAddr();
        __local_mem__ float *pointZPtr = (__local_mem__ float *)pointLocalz.GetPhyAddr();

        __local_mem__ float *boxesCxPtr = (__local_mem__ float *)boxesLocalCx.GetPhyAddr();
        __local_mem__ float *boxesCyPtr = (__local_mem__ float *)boxesLocalCy.GetPhyAddr();
        __local_mem__ float *boxesCzPtr = (__local_mem__ float *)boxesLocalCz.GetPhyAddr();
        __local_mem__ float *boxesDxPtr = (__local_mem__ float *)boxesLocalDx.GetPhyAddr();
        __local_mem__ float *boxesDyPtr = (__local_mem__ float *)boxesLocalDy.GetPhyAddr();
        __local_mem__ float *boxesDzPtr = (__local_mem__ float *)boxesLocalDz.GetPhyAddr();
        __local_mem__ float *boxesRzPtr = (__local_mem__ float *)boxesLocalRz.GetPhyAddr();

        __local_mem__ float *cosaPtr = (__local_mem__ float *)cosa.GetPhyAddr();
        __local_mem__ float *sinaPtr = (__local_mem__ float *)sina.GetPhyAddr();

        __local_mem__ int32_t *zLocalPtr = (__local_mem__ int32_t *)zLocal.GetPhyAddr();

        __VEC_SCOPE__ {
            MicroAPI::RegTensor<float> ptXReg, ptYReg, ptZReg;
            MicroAPI::RegTensor<float> cxReg, cyReg, czReg, dxReg, dyReg, dzReg, rzReg;
            MicroAPI::RegTensor<float> shiftxReg, shiftyReg, cosaReg, sinaReg;
            MicroAPI::RegTensor<float> localXReg, localYReg, tempReg;
            MicroAPI::RegTensor<int32_t> resultReg;
            MicroAPI::MaskReg mask = MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg cmpMask1, cmpMask2, cmpMask3, finalMask;

            uint16_t loops = DivCeil(computeBoxNum, B32_DATA_NUM_PER_REPEAT);

            for (uint16_t ptIdx = 0; ptIdx < tensorSize; ++ptIdx) {
                MicroAPI::DataCopy<DTYPE_BOXES, MicroAPI::LoadDist::DIST_BRC_B32>(ptXReg, pointXPtr + ptIdx * 8);
                MicroAPI::DataCopy<DTYPE_BOXES, MicroAPI::LoadDist::DIST_BRC_B32>(ptYReg, pointYPtr + ptIdx * 8);
                MicroAPI::DataCopy<DTYPE_BOXES, MicroAPI::LoadDist::DIST_BRC_B32>(ptZReg, pointZPtr + ptIdx * 8);

                uint32_t count = computeBoxNum;
                for (uint16_t boxLoops = 0; boxLoops < loops; ++boxLoops) {
                    uint32_t boxIdx = boxLoops * B32_DATA_NUM_PER_REPEAT;
                    uint32_t outIdx = ptIdx * computeBoxNum + boxIdx;

                    MicroAPI::MaskReg validMask = MicroAPI::UpdateMask<int32_t>(count);

                    MicroAPI::DataCopy(cxReg, boxesCxPtr + boxIdx);
                    MicroAPI::DataCopy(cyReg, boxesCyPtr + boxIdx);
                    MicroAPI::DataCopy(czReg, boxesCzPtr + boxIdx);
                    MicroAPI::DataCopy(dxReg, boxesDxPtr + boxIdx);
                    MicroAPI::DataCopy(dyReg, boxesDyPtr + boxIdx);
                    MicroAPI::DataCopy(dzReg, boxesDzPtr + boxIdx);

                    MicroAPI::DataCopy(cosaReg, cosaPtr + outIdx);
                    MicroAPI::DataCopy(sinaReg, sinaPtr + outIdx);

                    MicroAPI::Sub(shiftxReg, ptXReg, cxReg, mask);
                    MicroAPI::Sub(shiftyReg, ptYReg, cyReg, mask);

                    MicroAPI::Mul(tempReg, shiftxReg, cosaReg, mask);
                    MicroAPI::Mul(localYReg, shiftyReg, sinaReg, mask);
                    MicroAPI::Sub(localXReg, tempReg, localYReg, mask);

                    MicroAPI::Mul(tempReg, shiftxReg, sinaReg, mask);
                    MicroAPI::Mul(localYReg, shiftyReg, cosaReg, mask);
                    MicroAPI::Add(localYReg, localYReg, tempReg, mask);

                    MicroAPI::Abs(localXReg, localXReg, mask);
                    MicroAPI::Abs(localYReg, localYReg, mask);

                    MicroAPI::Muls(dxReg, dxReg, 0.5f, mask);
                    MicroAPI::Compare<float, CMPMODE::LT>(cmpMask1, localXReg, dxReg, mask);

                    MicroAPI::Muls(dyReg, dyReg, 0.5f, mask);
                    MicroAPI::Compare<float, CMPMODE::LT>(cmpMask2, localYReg, dyReg, mask);

                    MicroAPI::Muls(shiftyReg, dzReg, 0.5f, mask);
                    MicroAPI::Add(czReg, czReg, shiftyReg, mask);
                    MicroAPI::Sub(ptZReg, ptZReg, czReg, mask);
                    MicroAPI::Abs(ptZReg, ptZReg, mask);
                    MicroAPI::Compare<float, CMPMODE::LE>(cmpMask3, ptZReg, shiftyReg, mask);

                    MicroAPI::And(finalMask, cmpMask1, cmpMask2, mask);
                    MicroAPI::And(finalMask, finalMask, cmpMask3, mask);

                    MicroAPI::Duplicate(resultReg, 1, finalMask);
                    MicroAPI::DataCopy(zLocalPtr + outIdx, resultReg, validMask);
                }
            }
        }
    }

  private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueuePTS, inQueueBOXES;
    TBuf<TPosition::VECCALC> shiftxque, shiftyque, cosaque, sinaque, xLocalque, yLocalque, tempque, uint8que;
    TQue<QuePosition::VECOUT, 1> outQueueOUTPUT;
    GlobalTensor<DTYPE_BOXES> boxesGm;
    GlobalTensor<DTYPE_PTS> ptsGm;
    GlobalTensor<DTYPE_BOXES_IDX_OF_POINTS> outputGm;
    uint32_t usedCoreNum;
    uint32_t coreData;
    uint32_t copyLoop;
    uint32_t copyTail;
    uint32_t lastCopyLoop;
    uint32_t lastCopyTail;
    uint32_t npoints;
    uint32_t boxNumber;
    uint32_t availableUbSize;
    uint32_t batchSize;
    uint32_t boxNumLoop;
    LocalTensor<DTYPE_BOXES> boxesLocalCx;
    LocalTensor<DTYPE_BOXES> boxesLocalCy;
    LocalTensor<DTYPE_BOXES> boxesLocalCz;
    LocalTensor<DTYPE_BOXES> boxesLocalDx;
    LocalTensor<DTYPE_BOXES> boxesLocalDy;
    LocalTensor<DTYPE_BOXES> boxesLocalDz;
    LocalTensor<DTYPE_BOXES> boxesLocalRz;
    LocalTensor<DTYPE_PTS> pointLocalx;
    LocalTensor<DTYPE_PTS> pointLocaly;
    LocalTensor<DTYPE_PTS> pointLocalz;
    LocalTensor<DTYPE_BOXES_IDX_OF_POINTS> zLocal;
    LocalTensor<DTYPE_BOXES> shiftx;
    LocalTensor<DTYPE_BOXES> shifty;
    LocalTensor<DTYPE_BOXES> cosa;
    LocalTensor<DTYPE_BOXES> sina;
    LocalTensor<DTYPE_BOXES> xLocal;
    LocalTensor<DTYPE_BOXES> yLocal;
    LocalTensor<DTYPE_BOXES> temp;
    LocalTensor<uint8_t> uint8temp;
};

extern "C" __global__ __aicore__ void points_in_box_all(
    GM_ADDR boxes, GM_ADDR pts, GM_ADDR boxes_idx_of_points, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelPointsInBoxAll op;
    op.Init(boxes, pts, boxes_idx_of_points, &tiling_data);
    op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void points_in_box_all_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *boxes, uint8_t *pts,
    uint8_t *boxes_idx_of_points, uint8_t *workspace, uint8_t *tiling) {
    points_in_box_all<<<blockDim, l2ctrl, stream>>>(boxes, pts, boxes_idx_of_points, workspace, tiling);
}
#endif
