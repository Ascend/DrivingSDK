/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelPointsInBoxAll {
public:
    __aicore__ inline KernelPointsInBoxAll() {}

    __aicore__ inline void Init(GM_ADDR boxes, GM_ADDR pts, GM_ADDR boxes_idx_of_points,
                                PointsInBoxAllTilingData *tiling_data)
    {
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

        ptsGm.SetGlobalBuffer((__gm__ DTYPE_PTS*)pts + GetBlockIdx() * coreData * 3, coreData * 3);
        boxesGm.SetGlobalBuffer((__gm__ DTYPE_PTS*)boxes, boxNumber * 7 * batchSize);
        outputGm.SetGlobalBuffer(
            (__gm__ DTYPE_BOXES_IDX_OF_POINTS*)boxes_idx_of_points + GetBlockIdx() * coreData * boxNumber, coreData * boxNumber);
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

    __aicore__ inline void Process()
    {
        uint32_t coreIdx = GetBlockIdx();
        if (coreIdx > usedCoreNum -1) {
            return;
        }
        if (coreIdx != (usedCoreNum -1)) {
            for (int32_t i = 0; i < copyLoop; i++) {
                ComputeDiffBatch(i, availableUbSize, coreIdx, i+1, 0);
            }
            if (copyTail != 0) {
                ComputeDiffBatch(copyLoop, copyTail, coreIdx, copyLoop, copyTail);
            }
        } else {
            for (int32_t i = 0; i < lastCopyLoop; i++) {
                ComputeDiffBatch(i, availableUbSize, coreIdx, i+1, 0);
            }
            if (lastCopyTail != 0) {
                ComputeDiffBatch(lastCopyLoop, lastCopyTail, coreIdx, lastCopyLoop, lastCopyTail);
            }
        }
    }

private:
    __aicore__ inline void ComputeDiffBatch(int32_t progress, int32_t dataNum, uint32_t coreIdx, int32_t copyLoopOffset, uint32_t copyTailOffset)
    {
        uint64_t addressPoints = progress * availableUbSize;
        uint64_t addressOutput = progress * availableUbSize * boxNumber;
        int32_t coreBatchIdx = (coreIdx * coreData + progress * availableUbSize) / npoints;
        int32_t tail_num = coreIdx * coreData + copyLoopOffset * availableUbSize + copyTailOffset - (coreBatchIdx+1) * npoints;

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

    __aicore__ inline void ComputeBox(int32_t progress, int32_t dataNum, int32_t coreBatchIdx, uint64_t addressOutput, uint64_t addressPoints)
    {
        int32_t computeBoxNum = boxNumber;
        uint32_t boxCopyAddress = 0;
        uint32_t copyOutStride = (computeBoxNum > boxNumLoop) ? (computeBoxNum - boxNumLoop) : 0;
        uint32_t copyOutStrideTail = AlignUp(computeBoxNum, boxNumLoop) - boxNumLoop;
        uint32_t outAddressOffset = 0;

        while (computeBoxNum > boxNumLoop) {
            CopyBox(coreBatchIdx, boxNumLoop, boxCopyAddress);
            Compute(progress, dataNum, addressOutput, addressPoints, boxNumLoop, copyOutStride, outAddressOffset);
            boxCopyAddress += boxNumLoop;
            computeBoxNum -= boxNumLoop;
            outAddressOffset += boxNumLoop;
        }
        CopyBox(coreBatchIdx, computeBoxNum, boxCopyAddress);
        Compute(progress, dataNum, addressOutput, addressPoints, computeBoxNum, copyOutStrideTail, outAddressOffset);
    }

    __aicore__ inline void CopyBox(uint32_t boxCopyBatch, uint32_t boxCopyNum, uint32_t boxCopyAddress)
    {
        boxesLocalCx = inQueueBOXES.AllocTensor<DTYPE_BOXES>();
        boxesLocalCy = boxesLocalCx[availableUbSize];
        boxesLocalCz = boxesLocalCx[availableUbSize * 2];
        boxesLocalDx = boxesLocalCx[availableUbSize * 3];
        boxesLocalDy = boxesLocalCx[availableUbSize * 4];
        boxesLocalDz = boxesLocalCx[availableUbSize * 5];
        boxesLocalRz = boxesLocalCx[availableUbSize * 6];

        boxCopyNum = static_cast<int32_t>((boxCopyNum * sizeof(DTYPE_BOXES) + 32 - 1) /32) * 32 / sizeof(DTYPE_BOXES);
        DataCopyParams copyParams_box{1, (uint16_t)(boxCopyNum * sizeof(DTYPE_BOXES)), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};

        DataCopyPad(boxesLocalCx, boxesGm[boxNumber * boxCopyBatch * 7 + boxCopyAddress], copyParams_box, padParams);
        DataCopyPad(boxesLocalCy, boxesGm[boxNumber * (boxCopyBatch * 7 + 1) + boxCopyAddress], copyParams_box, padParams);
        DataCopyPad(boxesLocalCz, boxesGm[boxNumber * (boxCopyBatch * 7 + 2) + boxCopyAddress], copyParams_box, padParams);
        DataCopyPad(boxesLocalDx, boxesGm[boxNumber * (boxCopyBatch * 7 + 3) + boxCopyAddress], copyParams_box, padParams);
        DataCopyPad(boxesLocalDy, boxesGm[boxNumber * (boxCopyBatch * 7 + 4) + boxCopyAddress], copyParams_box, padParams);
        DataCopyPad(boxesLocalDz, boxesGm[boxNumber * (boxCopyBatch * 7 + 5) + boxCopyAddress], copyParams_box, padParams);
        DataCopyPad(boxesLocalRz, boxesGm[boxNumber * (boxCopyBatch * 7 + 6) + boxCopyAddress], copyParams_box, padParams);
    }

    __aicore__ inline void Compute(int32_t progress, int32_t tensorSize, uint64_t addressOutput, uint64_t addressPoints, uint32_t computeBoxNumOri, uint32_t copyOutStride, uint32_t outAddressOffset)
    {
        float oneminsnumber = -1;
        float halfnumber =  0.5;
        float zeronumber =  0;
        float onenumber =  1;
        float threenumber =  3;

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
        DataCopyExtParams copyParams_out{static_cast<uint16_t>(tensorSize), (uint32_t)(computeBoxNumOri * sizeof(DTYPE_BOXES_IDX_OF_POINTS)), 0, (uint32_t)(copyOutStride * sizeof(DTYPE_BOXES_IDX_OF_POINTS)), 0};
        uint32_t computeBoxNum = static_cast<int32_t>((computeBoxNumOri * sizeof(DTYPE_BOXES_IDX_OF_POINTS) + 32 - 1) / 32) *32 / sizeof(DTYPE_BOXES_IDX_OF_POINTS);
        DataCopyPadParams padParams{true, 0, 0, 0};

        // move points to localtensor
        DataCopyParams copyParams_in{static_cast<uint16_t>(tensorSize), (uint16_t)(1 * sizeof(DTYPE_BOXES)),  (uint16_t)(2 * sizeof(DTYPE_BOXES)), 0};
        DataCopyPad(pointLocalx, ptsGm[addressPoints * 3], copyParams_in, padParams);
        DataCopyPad(pointLocaly, ptsGm[addressPoints * 3 + 1], copyParams_in, padParams);
        DataCopyPad(pointLocalz, ptsGm[addressPoints * 3 + 2], copyParams_in, padParams);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        Duplicate<DTYPE_BOXES_IDX_OF_POINTS>(zLocal, zeronumber, tensorSize * computeBoxNum);

        // broadcast param
        uint8_t dim = 2;
        uint32_t dstShape[2];
        uint32_t srcShapePoint[2];
        srcShapePoint[0] = tensorSize;
        srcShapePoint[1] = 1;
        dstShape[0] = tensorSize;
        dstShape[1] = computeBoxNum;

        uint32_t srcShapeBoxes[2];
        srcShapeBoxes[0] = 1;
        srcShapeBoxes[1] = computeBoxNum;

        uint64_t mask = 64;
        int32_t repeat = (tensorSize + 7) / 8 ;
        BlockReduceMax<DTYPE_BOXES>(pointLocalx, pointLocalx, repeat, mask, 1, 1, 8);
        BlockReduceMax<DTYPE_BOXES>(pointLocaly, pointLocaly, repeat, mask, 1, 1, 8);
        BlockReduceMax<DTYPE_BOXES>(pointLocalz, pointLocalz, repeat, mask, 1, 1, 8);
        BroadCast<DTYPE_BOXES, 2, 1>(shiftx, pointLocalx, dstShape, srcShapePoint);

        // broadcast Cx to xLocal
        BroadCast<DTYPE_BOXES, 2, 0>(xLocal, boxesLocalCx, dstShape, srcShapeBoxes);
        repeat = (computeBoxNum * tensorSize + mask - 1) / mask;
        BinaryRepeatParams repeatParams = { 1, 1, 1, 8, 8, 8 };

        // shift_x = x - boxes_ub[ :, 0]
        Muls(temp, xLocal, oneminsnumber, mask, repeat, { 1, 1, 8, 8 });
        Add(shiftx, shiftx, temp, mask, repeat, {1, 1, 1, 8, 8, 8 });

        // broadcast Cy to yLocal
        BroadCast<DTYPE_BOXES, 2, 0>(yLocal, boxesLocalCy, dstShape, srcShapeBoxes);
        BroadCast<DTYPE_BOXES, 2, 1>(shifty, pointLocaly, dstShape, srcShapePoint);

        // shift_y = y - boxes_ub[ :, 1]
        Muls(temp, yLocal, oneminsnumber, mask, repeat, { 1, 1, 8, 8 });
        Add(shifty, shifty, temp, mask, repeat, {1, 1, 1, 8, 8, 8 });

        // broadcast Rz to xLocal
        BroadCast<DTYPE_BOXES, 2, 0>(xLocal, boxesLocalRz, dstShape, srcShapeBoxes);

        // cosa = Cos(-boxes_ub[ :, 6])
        Muls(temp, xLocal, oneminsnumber, mask, repeat, { 1, 1, 8, 8 });
        Cos<DTYPE_BOXES, false>(cosa, temp, uint8temp, computeBoxNum * tensorSize);

        // sina = Sin(-boxes_ub[ :, 6])
        Sin<DTYPE_BOXES, false>(sina, temp, uint8temp, computeBoxNum * tensorSize);

        // local_x = shift_x * cosa + shift_y * (-sina)
        Mul(temp, shiftx, cosa, mask, repeat, {1, 1, 1, 8, 8, 8 });
        Duplicate<DTYPE_BOXES>(xLocal, zeronumber, computeBoxNum * tensorSize);
        Add(xLocal, xLocal, temp, mask, repeat, {1, 1, 1, 8, 8, 8 });
        Muls(temp, sina, oneminsnumber, mask, repeat, { 1, 1, 8, 8 });
        Mul(temp, shifty, temp, mask, repeat, {1, 1, 1, 8, 8, 8 });
        Add(xLocal, xLocal, temp, mask, repeat, {1, 1, 1, 8, 8, 8 });

        // local_y = shift_x * sina + shift_y * cosa
        Mul(temp, shiftx, sina, mask, repeat, {1, 1, 1, 8, 8, 8 });
        Mul(sina, shifty, cosa, mask, repeat, {1, 1, 1, 8, 8, 8 });
        Add(yLocal, sina, temp,  mask, repeat, {1, 1, 1, 8, 8, 8 });
        Abs(xLocal, xLocal, mask, repeat, { 1, 1, 8, 8 });
        pipe_barrier(PIPE_V);
        Abs(yLocal, yLocal, mask, repeat, { 1, 1, 8, 8 });

        // dup full zeronumber tensor
        Duplicate<DTYPE_BOXES>(sina, zeronumber, mask, repeat, 1, 8);
        // dup full onenumber tensor
        Duplicate<DTYPE_BOXES>(temp, onenumber, mask, repeat, 1, 8);

        // broadcast Dx to cosa
        BroadCast<DTYPE_BOXES, 2, 0>(cosa, boxesLocalDx, dstShape, srcShapeBoxes);
        // shiftx = 0.5 dx
        Muls(shiftx, cosa, halfnumber, mask, repeat, { 1, 1, 8, 8 });

        // cmp_1 = Abs(local_x) < x_size
        pipe_barrier(PIPE_V);
        uint8temp = xLocal < shiftx;
        Duplicate<DTYPE_BOXES>(xLocal, zeronumber, mask, repeat, 1, 8);
        Select(xLocal, uint8temp, temp, sina,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, mask, repeat, repeatParams);

        // shifty = 0.5 dy
        BroadCast<DTYPE_BOXES, 2, 0>(cosa, boxesLocalDy, dstShape, srcShapeBoxes);
        Muls(shifty, cosa, halfnumber, mask, repeat, { 1, 1, 8, 8 });

        // cmp_2 = Abs(local_y) < y_size
        pipe_barrier(PIPE_V);
        uint8temp = yLocal < shifty;
        Duplicate<DTYPE_BOXES>(yLocal, zeronumber, mask, repeat, 1, 8);
        Select(yLocal, uint8temp, temp, sina,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, mask, repeat, repeatParams);

        // broadcast Dz to shiftx
        BroadCast<DTYPE_BOXES, 2, 0>(shiftx, boxesLocalDz, dstShape, srcShapeBoxes);
        // broadcast Cz to shifty
        BroadCast<DTYPE_BOXES, 2, 0>(shifty, boxesLocalCz, dstShape, srcShapeBoxes);

        // cz += zsize / 2
        Muls(cosa, shiftx, halfnumber, mask, repeat, { 1, 1, 8, 8 });
        Add(shifty, shifty, cosa, mask, repeat, {1, 1, 1, 8, 8, 8 });

        // zlocal = z-cz
        Muls(sina, shifty, oneminsnumber, mask, repeat, { 1, 1, 8, 8 });
        BroadCast<DTYPE_BOXES, 2, 1>(shifty, pointLocalz, dstShape, srcShapePoint);
        Add(sina, sina, shifty, mask, repeat, {1, 1, 1, 8, 8, 8 });
        Abs(sina, sina, mask, repeat, { 1, 1, 8, 8 });

        // dup full zeronumber tensor
        Duplicate<DTYPE_BOXES>(shifty, zeronumber, mask, repeat, 1, 8);
        // dup full onenumber tensor
        Duplicate<DTYPE_BOXES>(temp, onenumber, mask, repeat, 1, 8);

        // cmp_3 = Abs(zlocal) < z_size
        pipe_barrier(PIPE_V);
        uint8temp = sina <= cosa;
        Duplicate<DTYPE_BOXES>(sina, zeronumber, mask, repeat, 1, 8);
        Select(sina, uint8temp, temp, shifty,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, mask, repeat, repeatParams);
        
        // select which point is in box
        Add(yLocal, yLocal, sina, computeBoxNum * tensorSize);
        Add(yLocal, yLocal, xLocal, computeBoxNum * tensorSize);

        Duplicate<DTYPE_BOXES>(shiftx, threenumber, mask, repeat, 1, 8);
        pipe_barrier(PIPE_V);
        uint8temp = yLocal == shiftx;
        Select(yLocal, uint8temp, temp, shifty,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, computeBoxNum * tensorSize);
        Cast(zLocal, yLocal, RoundMode::CAST_RINT, computeBoxNum * tensorSize);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        DataCopyPad(outputGm[addressOutput + outAddressOffset], zLocal, copyParams_out);
        inQueuePTS.FreeTensor(pointLocalx);
        inQueueBOXES.FreeTensor(boxesLocalCx);
        outQueueOUTPUT.FreeTensor(zLocal);
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

extern "C" __global__ __aicore__ void points_in_box_all(GM_ADDR boxes, GM_ADDR pts,
                                                    GM_ADDR boxes_idx_of_points,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelPointsInBoxAll op;
    op.Init(boxes, pts, boxes_idx_of_points, &tiling_data);
    op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void points_in_box_all_do(uint32_t blockDim, void* l2ctrl,
                          void* stream, uint8_t* boxes, uint8_t* pts, uint8_t* boxes_idx_of_points,
                          uint8_t* workspace, uint8_t* tiling)
{
    points_in_box_all<<<blockDim, l2ctrl, stream>>>(boxes, pts, boxes_idx_of_points, workspace, tiling);
}
#endif