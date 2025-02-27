/*
* Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
*/
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t INPUT_NUM = 3;
constexpr int32_t OUTPUT_NUM = 2;
constexpr int32_t GRAD_INPUT_INDEX = 0;
constexpr int32_t X_INPUT_INDEX = 1;
constexpr int32_t GRID_INPUT_INDEX = 2;
constexpr int32_t DX_INPUT_INDEX = 3;
constexpr int32_t DGRID_INPUT_INDEX = 4;
constexpr int32_t GRID_GRAD_OUTPUT_INDEX = 1;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t FP32_BLOCK_NUM = 8;
constexpr int32_t BYTE_BLOCK = 32;
constexpr uint32_t ELE_NUM_PER_REPEAT = 64;
constexpr uint32_t UINT8_BITS = 8;

constexpr uint32_t FLOAT_BYTES = 4;
constexpr uint32_t ALIGN_256_BYTES = 256;

class GridSampler2dV2GradKernel {
public:
    __aicore__ inline GridSampler2dV2GradKernel() {}

    __aicore__ inline void Init(GridSampler2dV2GradTilingData* tilingData, GM_ADDR inputTensors[INPUT_NUM + OUTPUT_NUM])
    {
        batch = tilingData->batch;
        pNumPerCore = tilingData->pNumPerCore;
        tailPNum = tilingData->tailPNum;
        channel = tilingData->channel;
        alignedChannel = tilingData->alignedChannel;
        height = tilingData->height;
        width = tilingData->width;
        blockNum = tilingData->blockNum;
        calcCountPerLoop = tilingData->calcCountPerLoop;
        interpolation = tilingData->interpolation;
        padding = tilingData->padding;
        alignCorners = tilingData->alignCorners;
        groupSize = tilingData->groupSize;
        coordPosition = tilingData->coordPosition;
        gridH = tilingData->gridH;
        gridW = tilingData->gridW;
        maskSize = AlignUp(Ceil(calcCountPerLoop, UINT8_BITS), BYTE_BLOCK);
        maskNum = maskSize / sizeof(uint8_t);
        dxStrideN = channel * width * height;
        dxStrideH = width;
        blockIdx = GetBlockIdx();

        inputGm[GRAD_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputTensors[GRAD_INPUT_INDEX]));
        inputGm[X_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputTensors[X_INPUT_INDEX]));
        inputGm[GRID_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputTensors[GRID_INPUT_INDEX]));
        inputGm[DX_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputTensors[DX_INPUT_INDEX]));
        inputGm[DGRID_INPUT_INDEX].SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputTensors[DGRID_INPUT_INDEX]));
    }

    __aicore__ inline void InitTask()
    {
        nwCpInOffsetStart = 0;
        neCpInOffsetStart = groupSize * alignedChannel;
        swCpInOffsetStart = groupSize * alignedChannel * 2;
        seCpInOffsetStart = groupSize * alignedChannel * 3;
    }

    __aicore__ inline void InitBuffer(TPipe* pipe)
    {
        pipe->InitBuffer(dataInQueue[0], BUFFER_NUM, groupSize * alignedChannel * sizeof(float));
        pipe->InitBuffer(dataInQueue[1], BUFFER_NUM, coordPosition * groupSize * alignedChannel * sizeof(float));
        pipe->InitBuffer(dataInQueue[GRID_INPUT_INDEX], BUFFER_NUM, BUFFER_NUM * calcCountPerLoop * sizeof(float));
        pipe->InitBuffer(dataOutQueue[0], BUFFER_NUM, alignedChannel * sizeof(float));
        pipe->InitBuffer(dataOutQueue[1], BUFFER_NUM, BUFFER_NUM * calcCountPerLoop * sizeof(float));

        pipe->InitBuffer(xCoordinateBuf, (calcCountPerLoop + ELE_NUM_PER_REPEAT) * sizeof(float));
        pipe->InitBuffer(yCoordinateBuf, (calcCountPerLoop + ELE_NUM_PER_REPEAT) * sizeof(float));
        pipe->InitBuffer(xGradInBuf, calcCountPerLoop * sizeof(float));
        pipe->InitBuffer(yGradInBuf, calcCountPerLoop * sizeof(float));

        pipe->InitBuffer(ixBuf, coordPosition * calcCountPerLoop * sizeof(float));
        pipe->InitBuffer(iyBuf, coordPosition * calcCountPerLoop * sizeof(float));

        pipe->InitBuffer(ixIntBuf, coordPosition * calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(iyIntBuf, coordPosition * calcCountPerLoop * sizeof(int32_t));

        pipe->InitBuffer(nwBuf, calcCountPerLoop * sizeof(float));
        pipe->InitBuffer(neBuf, calcCountPerLoop * sizeof(float));
        pipe->InitBuffer(swBuf, calcCountPerLoop * sizeof(float));
        pipe->InitBuffer(seBuf, calcCountPerLoop * sizeof(float));

        pipe->InitBuffer(tmp1Buf, calcCountPerLoop * sizeof(float));
        pipe->InitBuffer(tmp2Buf, calcCountPerLoop * sizeof(float));

        pipe->InitBuffer(mask1Buf, maskSize);
        pipe->InitBuffer(mask2Buf, maskSize);

        pipe->InitBuffer(dupOneBuf, calcCountPerLoop * sizeof(float));
        pipe->InitBuffer(selBuf1, calcCountPerLoop * sizeof(float));
        pipe->InitBuffer(selBuf2, calcCountPerLoop * sizeof(float));
        pipe->InitBuffer(selBuf3, calcCountPerLoop * sizeof(float));
        pipe->InitBuffer(selBuf4, calcCountPerLoop * sizeof(float));

        pipe->InitBuffer(computeIndexBuf2, calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf3, calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf4, calcCountPerLoop * sizeof(int32_t));
        pipe->InitBuffer(computeIndexBuf5, calcCountPerLoop * sizeof(int32_t));

        pipe->InitBuffer(giCoorBuf, 2 * alignedChannel * sizeof(float));
        pipe->InitBuffer(sumBuf,  2 * alignedChannel * sizeof(float));
    }

    __aicore__ inline void InitLocalTensor()

    {
        mask1Tensor = mask1Buf.Get<uint8_t>(maskNum);
        mask2Tensor = mask2Buf.Get<uint8_t>(maskNum);
        dupOneTensor = dupOneBuf.Get<float>(calcCountPerLoop);
        selTensor1 = selBuf1.Get<float>(calcCountPerLoop);
        tmp1Tensor = tmp1Buf.Get<float>(calcCountPerLoop);
        tmp2Tensor = tmp2Buf.Get<float>(calcCountPerLoop);
        selTensor2 = selBuf2.Get<float>(calcCountPerLoop);
        selTensor3 = selBuf3.Get<float>(calcCountPerLoop);
        selTensor4 = selBuf4.Get<float>(calcCountPerLoop);
        sum = sumBuf.Get<float>(2 * alignedChannel);
    }

    __aicore__ inline void Process()
    {
        uint32_t computePNum;
        int64_t gridGmOffset;
        int32_t gridOffset;
        int32_t cycleOffset;
        int64_t curGridPointIndex;
        if (blockIdx < tailPNum) {
            computePNum = pNumPerCore + 1;
            gridOffset = blockIdx * computePNum;
        } else {
            computePNum = pNumPerCore;
            gridOffset = blockIdx * pNumPerCore + tailPNum;
        }
        int32_t copyCountPerTime = 2 * calcCountPerLoop;
        int32_t actualComputeNum = copyCountPerTime;
        int32_t copyTimes = Ceil(computePNum, calcCountPerLoop);
        for (int j = 0; j < copyTimes; j++) {
            if (j == copyTimes - 1) {
                actualComputeNum = computePNum * 2 - (copyTimes - 1) * copyCountPerTime;
            }
            cycleOffset = j * copyCountPerTime;
            gridGmOffset = cycleOffset + gridOffset * 2;
            curGridPointIndex = gridOffset + j * copyCountPerTime / 2;
            CopyIn(gridGmOffset, actualComputeNum, GRID_INPUT_INDEX);
            Compute(actualComputeNum / 2, curGridPointIndex);
            CopyOut(gridGmOffset, actualComputeNum);
        }
    }

private:
    __aicore__ inline void CopyIn(const int64_t offset, const int32_t calCount, const int32_t inputIndex)
    {
        LocalTensor<float> dataLocal = dataInQueue[inputIndex].AllocTensor<float>();
        DataCopyParams copyParams = {1, 0, 0, 0};
        DataCopyPadParams padParams = {true, 0, 0, 0};
        int32_t alignCalCount = AlignUp(calCount, FP32_BLOCK_NUM);
        copyParams.blockLen = calCount * sizeof(float);
        padParams.rightPadding = alignCalCount - calCount;

        DataCopyPad(dataLocal, inputGm[inputIndex][offset], copyParams, padParams);
        dataInQueue[inputIndex].EnQue(dataLocal);
    }

    __aicore__ inline void Compute(const int32_t computeCount, const int64_t curGridPointIndex)
    {
        int32_t groupOffset = 0;
        int32_t groupNum = computeCount / groupSize;
        int32_t tailNums = computeCount % groupSize;
        uint64_t cnt = 0;
        int64_t gridPointIndex = 0;
        int32_t gradStrideN = channel * gridH * gridW;
        int32_t gradStrideH = gridW;
        int32_t gradStrideW = 1;
        int64_t w = 0;
        int64_t h = 0;
        int64_t n = 0;
        int64_t ncBaseOffset = 0;
        DataCopyParams copyParams = {1, 0, 0, 0};
        DataCopyPadParams padParams = {true, 0, 0, 0};
        copyParams.blockLen = channel * sizeof(float);
        padParams.rightPadding = alignedChannel - channel;
        uint16_t repeatTimes = Ceil(computeCount * 2, ELE_NUM_PER_REPEAT);

        int32_t alignedComputeCount = AlignUp(computeCount, BYTE_BLOCK);
        int32_t nwOffset = 0;
        int32_t neOffset = alignedComputeCount;
        int32_t swOffset = 2 * alignedComputeCount;
        int32_t seOffset = 3 * alignedComputeCount;

        LocalTensor<float> xTensor = xCoordinateBuf.Get<float>(calcCountPerLoop + ELE_NUM_PER_REPEAT);
        LocalTensor<float> yTensor = yCoordinateBuf.Get<float>(calcCountPerLoop + ELE_NUM_PER_REPEAT);
        LocalTensor<float> xGradIn = xGradInBuf.Get<float>(calcCountPerLoop);
        LocalTensor<float> yGradIn = yGradInBuf.Get<float>(calcCountPerLoop);
        LocalTensor<float> inputCoordinate = dataInQueue[GRID_INPUT_INDEX].DeQue<float>();
        LocalTensor<float> dstLocal = dataOutQueue[1].AllocTensor<float>();

        DupValue();
        GatherMask(xTensor, inputCoordinate, 1, false, 0, {1, repeatTimes, 8, 8}, cnt);
        GatherMask(yTensor, inputCoordinate, 2, false, 0, {1, repeatTimes, 8, 8}, cnt);

        ComputeSourceIndex(xTensor, xGradIn, width, computeCount);
        ComputeSourceIndex(yTensor, yGradIn, height, computeCount);

        LocalTensor<float> nw = nwBuf.Get<float>(calcCountPerLoop);
        LocalTensor<float> ne = neBuf.Get<float>(calcCountPerLoop);
        LocalTensor<float> sw = swBuf.Get<float>(calcCountPerLoop);
        LocalTensor<float> se = seBuf.Get<float>(calcCountPerLoop);

        LocalTensor<int32_t> nwIndex = computeIndexBuf2.Get<int32_t>(calcCountPerLoop);
        LocalTensor<int32_t> neIndex = computeIndexBuf3.Get<int32_t>(calcCountPerLoop);
        LocalTensor<int32_t> swIndex = computeIndexBuf4.Get<int32_t>(calcCountPerLoop);
        LocalTensor<int32_t> seIndex = computeIndexBuf5.Get<int32_t>(calcCountPerLoop);

        LocalTensor<float> ix = ixBuf.Get<float>(coordPosition * calcCountPerLoop);
        LocalTensor<float> iy = iyBuf.Get<float>(coordPosition * calcCountPerLoop);

        LocalTensor<int32_t> ixInt = ixIntBuf.Get<int32_t>(coordPosition * calcCountPerLoop);
        LocalTensor<int32_t> iyInt = iyIntBuf.Get<int32_t>(coordPosition * calcCountPerLoop);

        giCoorLocalTensor = giCoorBuf.Get<float>(2 * alignedChannel);

        Cast(ixInt[nwOffset], xTensor, RoundMode::CAST_FLOOR, computeCount);
        Cast(iyInt[nwOffset], yTensor, RoundMode::CAST_FLOOR, computeCount);
        Cast(iyInt[neOffset], yTensor, RoundMode::CAST_FLOOR, computeCount);
        Cast(ixInt[swOffset], xTensor, RoundMode::CAST_FLOOR, computeCount);

        Adds(ixInt[neOffset], ixInt[nwOffset], static_cast<int32_t>(1), computeCount);
        Adds(iyInt[swOffset], iyInt[nwOffset], static_cast<int32_t>(1), computeCount);
        Adds(ixInt[seOffset], ixInt[nwOffset], static_cast<int32_t>(1), computeCount);
        Adds(iyInt[seOffset], iyInt[nwOffset], static_cast<int32_t>(1), computeCount);

        // convert to float32
        Cast(ix, ixInt, RoundMode::CAST_NONE, coordPosition * alignedComputeCount);
        Cast(iy, iyInt, RoundMode::CAST_NONE, coordPosition * alignedComputeCount);

        // compute nw weight
        ComputeWeight(nw, xTensor, ix[seOffset], yTensor, iy[seOffset], computeCount);
        // compute ne weight
        ComputeWeight(ne, ix[swOffset], xTensor, yTensor, iy[swOffset], computeCount);
        // compute sw weight
        ComputeWeight(sw, xTensor, ix[neOffset], iy[neOffset], yTensor, computeCount);
        // compute se weight
        ComputeWeight(se, ix[nwOffset], xTensor, iy[nwOffset], yTensor, computeCount);

        WithinBounds2d(selTensor1, iy[nwOffset], ix[nwOffset], nw, computeCount);
        WithinBounds2d(selTensor2, iy[neOffset], ix[neOffset], ne, computeCount);
        WithinBounds2d(selTensor3, iy[swOffset], ix[swOffset], sw, computeCount);
        WithinBounds2d(selTensor4, iy[seOffset], ix[seOffset], se, computeCount);

        ComputeIndex(nwIndex, iyInt[nwOffset], ixInt[nwOffset], computeCount);
        ComputeIndex(neIndex, iyInt[neOffset], ixInt[neOffset], computeCount);
        ComputeIndex(swIndex, iyInt[swOffset], ixInt[swOffset], computeCount);
        ComputeIndex(seIndex, iyInt[seOffset], ixInt[seOffset], computeCount);

        for (int32_t i = 0; i < groupNum; i++) {
            LocalTensor<float> gOutLocalTensor = dataInQueue[0].AllocTensor<float>();
            LocalTensor<float> inputXLocalTensor = dataInQueue[1].AllocTensor<float>();
            groupOffset = groupSize * i;
            for (int32_t j = 0; j < groupSize; j++) {
                gridPointIndex = curGridPointIndex + groupOffset + j;
                w = gridPointIndex % gridW;
                h = (gridPointIndex / gridW) % gridH;
                n = gridPointIndex / (gridH * gridW);

                ncBaseOffset = n * dxStrideN;
                gradGmOffset = n * gradStrideN + (h * gradStrideH + w * gradStrideW) * channel;
                int32_t cpInOffset = alignedChannel * j;
                DataCopyPad(gOutLocalTensor[cpInOffset], inputGm[0][gradGmOffset], copyParams, padParams);

                int32_t nwPointIndex = nwIndex.GetValue(groupOffset + j);
                int32_t nePointIndex = neIndex.GetValue(groupOffset + j);
                int32_t swPointIndex = swIndex.GetValue(groupOffset + j);
                int32_t sePointIndex = seIndex.GetValue(groupOffset + j);

                DataCopyPad(inputXLocalTensor[nwCpInOffsetStart + cpInOffset], inputGm[1][ncBaseOffset + nwPointIndex], copyParams, padParams);
                DataCopyPad(inputXLocalTensor[neCpInOffsetStart + cpInOffset], inputGm[1][ncBaseOffset + nePointIndex], copyParams, padParams);
                DataCopyPad(inputXLocalTensor[swCpInOffsetStart + cpInOffset], inputGm[1][ncBaseOffset + swPointIndex], copyParams, padParams);
                DataCopyPad(inputXLocalTensor[seCpInOffsetStart + cpInOffset], inputGm[1][ncBaseOffset + sePointIndex], copyParams, padParams);
            }
            event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            set_flag(PIPE_MTE2, PIPE_V, eventID1);
            wait_flag(PIPE_MTE2, PIPE_V, eventID1);
            for (int32_t k = 0; k < groupSize; k++) {
                int32_t cpInOffset = alignedChannel * k;
                int32_t coorIndex = groupOffset + k;
                gridPointIndex = curGridPointIndex + groupOffset + k;
                n = gridPointIndex / (gridH * gridW);
                ncBaseOffset = n * dxStrideN;

                ComputeGridGrad(iy[seOffset], yTensor, ix[seOffset], xTensor, gOutLocalTensor[cpInOffset], inputXLocalTensor[nwCpInOffsetStart + cpInOffset], selTensor1, coorIndex);
                ComputeXGrad(nwIndex, nw, coorIndex, ncBaseOffset, gOutLocalTensor[cpInOffset]);
                ComputeGridGrad(yTensor, iy[swOffset], xTensor, ix[swOffset], gOutLocalTensor[cpInOffset], inputXLocalTensor[neCpInOffsetStart + cpInOffset], selTensor2, coorIndex);
                ComputeXGrad(neIndex, ne, coorIndex, ncBaseOffset, gOutLocalTensor[cpInOffset]);
                ComputeGridGrad(yTensor, iy[neOffset], xTensor, ix[neOffset], gOutLocalTensor[cpInOffset], inputXLocalTensor[swCpInOffsetStart + cpInOffset], selTensor3, coorIndex);
                ComputeXGrad(swIndex, sw, coorIndex, ncBaseOffset, gOutLocalTensor[cpInOffset]);
                ComputeGridGrad(iy[nwOffset], yTensor, ix[nwOffset], xTensor, gOutLocalTensor[cpInOffset], inputXLocalTensor[seCpInOffsetStart + cpInOffset], selTensor4, coorIndex);
                ComputeXGrad(seIndex, se, coorIndex, ncBaseOffset, gOutLocalTensor[cpInOffset]);

                ReduceSum<float>(sum, sum, sum, alignedChannel);
                ReduceSum<float>(sum[alignedChannel], sum[alignedChannel], sum[alignedChannel], alignedChannel);

                gix -= sum.GetValue(0);
                giy -= sum[alignedChannel].GetValue(0);

                dstLocal.SetValue(2 * coorIndex, gix * xGradIn.GetValue(coorIndex));
                dstLocal.SetValue(2 * coorIndex + 1, giy * yGradIn.GetValue(coorIndex));
                Duplicate<float>(sum, 0, 2 * alignedChannel);
                gix = 0.f;
                giy = 0.f;
            }
            dataInQueue[0].FreeTensor(gOutLocalTensor);
            dataInQueue[1].FreeTensor(inputXLocalTensor);
        }
        if (tailNums != 0) {
            LocalTensor<float> gOutLocalTensor = dataInQueue[0].AllocTensor<float>();
            LocalTensor<float> inputXLocalTensor = dataInQueue[1].AllocTensor<float>();
            groupOffset = groupSize * groupNum;
            for (int32_t j = 0; j < tailNums; j++) {
                gridPointIndex = curGridPointIndex + groupOffset + j;
                w = gridPointIndex % gridW;
                h = (gridPointIndex / gridW) % gridH;
                n = gridPointIndex / (gridH * gridW);

                ncBaseOffset = n * dxStrideN;
                gradGmOffset = n * gradStrideN + (h * gradStrideH + w * gradStrideW) * channel;
                int32_t cpInOffset = alignedChannel * j;
                DataCopyPad(gOutLocalTensor[cpInOffset], inputGm[0][gradGmOffset], copyParams, padParams);

                int32_t nwPointIndex = nwIndex.GetValue(groupOffset + j);
                int32_t nePointIndex = neIndex.GetValue(groupOffset + j);
                int32_t swPointIndex = swIndex.GetValue(groupOffset + j);
                int32_t sePointIndex = seIndex.GetValue(groupOffset + j);

                DataCopyPad(inputXLocalTensor[nwCpInOffsetStart + cpInOffset], inputGm[1][ncBaseOffset + nwPointIndex], copyParams, padParams);
                DataCopyPad(inputXLocalTensor[neCpInOffsetStart + cpInOffset], inputGm[1][ncBaseOffset + nePointIndex], copyParams, padParams);
                DataCopyPad(inputXLocalTensor[swCpInOffsetStart + cpInOffset], inputGm[1][ncBaseOffset + swPointIndex], copyParams, padParams);
                DataCopyPad(inputXLocalTensor[seCpInOffsetStart + cpInOffset], inputGm[1][ncBaseOffset + sePointIndex], copyParams, padParams);
            }
            event_t eventID1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            set_flag(PIPE_MTE2, PIPE_V, eventID1);
            wait_flag(PIPE_MTE2, PIPE_V, eventID1);
            for (int32_t k = 0; k < tailNums; k++) {
                int32_t cpInOffset = alignedChannel * k;
                int32_t coorIndex = groupOffset + k;
                gridPointIndex = curGridPointIndex + groupOffset + k;
                n = gridPointIndex / (gridH * gridW);
                ncBaseOffset = n * dxStrideN;

                ComputeGridGrad(iy[seOffset], yTensor, ix[seOffset], xTensor, gOutLocalTensor[cpInOffset], inputXLocalTensor[nwCpInOffsetStart + cpInOffset], selTensor1, coorIndex);
                ComputeXGrad(nwIndex, nw, coorIndex, ncBaseOffset, gOutLocalTensor[cpInOffset]);
                ComputeGridGrad(yTensor, iy[swOffset], xTensor, ix[swOffset], gOutLocalTensor[cpInOffset], inputXLocalTensor[neCpInOffsetStart + cpInOffset], selTensor2, coorIndex);
                ComputeXGrad(neIndex, ne, coorIndex, ncBaseOffset, gOutLocalTensor[cpInOffset]);
                ComputeGridGrad(yTensor, iy[neOffset], xTensor, ix[neOffset], gOutLocalTensor[cpInOffset], inputXLocalTensor[swCpInOffsetStart + cpInOffset], selTensor3, coorIndex);
                ComputeXGrad(swIndex, sw, coorIndex, ncBaseOffset, gOutLocalTensor[cpInOffset]);
                ComputeGridGrad(iy[nwOffset], yTensor, ix[nwOffset], xTensor, gOutLocalTensor[cpInOffset], inputXLocalTensor[seCpInOffsetStart + cpInOffset], selTensor4, coorIndex);
                ComputeXGrad(seIndex, se, coorIndex, ncBaseOffset, gOutLocalTensor[cpInOffset]);

                ReduceSum<float>(sum, sum, sum, alignedChannel);
                ReduceSum<float>(sum[alignedChannel], sum[alignedChannel], sum[alignedChannel], alignedChannel);

                gix -= sum.GetValue(0);
                giy -= sum[alignedChannel].GetValue(0);

                dstLocal.SetValue(2 * coorIndex, gix * xGradIn.GetValue(coorIndex));
                dstLocal.SetValue(2 * coorIndex + 1, giy * yGradIn.GetValue(coorIndex));
                Duplicate<float>(sum, 0, 2 * alignedChannel);
                gix = 0.f;
                giy = 0.f;
            }
            dataInQueue[0].FreeTensor(gOutLocalTensor);
            dataInQueue[1].FreeTensor(inputXLocalTensor);
        }
        dataOutQueue[GRID_GRAD_OUTPUT_INDEX].EnQue(dstLocal);
        dataInQueue[GRID_INPUT_INDEX].FreeTensor(inputCoordinate);
    }

    __aicore__ inline void ComputeGridGrad(LocalTensor<float> yCoor1, LocalTensor<float> yCoor2,
        LocalTensor<float> xCoor1, LocalTensor<float> xCoor2, LocalTensor<float> gOutLocalTensor,
        LocalTensor<float> inputXLocalTensor, LocalTensor<float> selTensor, const int32_t coorIndex)
    {
        float xVal = yCoor1.GetValue(coorIndex) - yCoor2.GetValue(coorIndex);
        float yVal = xCoor1.GetValue(coorIndex) - xCoor2.GetValue(coorIndex);
        float flag = selTensor.GetValue(coorIndex);
        Mul(tmp1Tensor, inputXLocalTensor, gOutLocalTensor, alignedChannel);
        Muls(giCoorLocalTensor, tmp1Tensor, xVal, alignedChannel);
        Muls(giCoorLocalTensor[alignedChannel], tmp1Tensor, yVal, alignedChannel);
        Axpy(sum, giCoorLocalTensor, flag, 2 * alignedChannel);
    }

    __aicore__ inline void ComputeXGrad(LocalTensor<int32_t> srcIndex, LocalTensor<float> weight,
        const int32_t coorIndex, const int64_t ncOffset, LocalTensor<float> gOutLocalTensor)
    {
        float weightVal = weight.GetValue(coorIndex);
        int64_t offset = ncOffset + srcIndex.GetValue(coorIndex);

        LocalTensor<float> localTensor = dataOutQueue[0].AllocTensor<float>();
        Muls(localTensor, gOutLocalTensor, weightVal, alignedChannel);
        dataOutQueue[0].EnQue(localTensor);
        localTensor = dataOutQueue[0].DeQue<float>();

        DataCopyParams copyParams {1, 0, 0, 0};
        copyParams.blockLen = channel * sizeof(float);
        SetAtomicAdd<float>();
        DataCopyPad(inputGm[DX_INPUT_INDEX][offset], localTensor, copyParams);
        SetAtomicNone();
        dataOutQueue[0].FreeTensor(localTensor);
    }

    __aicore__ inline void DupValue()
    {
        Duplicate<float>(dupOneTensor, 1, calcCountPerLoop);
        Duplicate<float>(sum, 0, 2 * alignedChannel);
    }

    __aicore__ inline void ComputeSourceIndex(
        LocalTensor<float> dataTensor, LocalTensor<float> dupTensor, const int32_t size, const int32_t calCount)
    {
        if (alignCorners) {
            float val = static_cast<float>(size - 1) / 2;
            Duplicate<float>(dupTensor, val, calCount);
            Adds(dataTensor, dataTensor, 1.f, calCount);
            Muls(dataTensor, dataTensor, 0.5f, calCount);
            Muls(dataTensor, dataTensor, static_cast<float>(size - 1), calCount);
        } else {
            float val = static_cast<float>(size) / 2;
            Duplicate<float>(dupTensor, val, calCount);
            Adds(dataTensor, dataTensor, 1.f, calCount);
            Muls(dataTensor, dataTensor, static_cast<float>(size), calCount);
            Adds(dataTensor, dataTensor, -1.f, calCount);
            Muls(dataTensor, dataTensor, 0.5f, calCount);
        }
        int32_t newCalCount =
            ((calCount * FLOAT_BYTES + ALIGN_256_BYTES - 1) / ALIGN_256_BYTES * ALIGN_256_BYTES) / FLOAT_BYTES;
        if (padding == 1) {
            CompareScalar(mask1Tensor, dataTensor, 0.f, CMPMODE::GT, newCalCount);
            Select(dataTensor, mask1Tensor, dataTensor, 0.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
            Select(dupTensor, mask1Tensor, dupTensor, 0.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
            CompareScalar(mask1Tensor, dataTensor, static_cast<float>(size - 1), CMPMODE::LT, newCalCount);
            Select(dataTensor, mask1Tensor, dataTensor, static_cast<float>(size - 1), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                newCalCount);
            Select(dupTensor, mask1Tensor, dupTensor, 0.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
        }
    }

    __aicore__ inline void ComputeWeight(LocalTensor<float> dst, LocalTensor<float> xCoorTensor1,
        LocalTensor<float> xCoorTensor2, LocalTensor<float> yCoorTensor1, LocalTensor<float> yCoorTensor2,
        const int32_t calCount)
    {
        Muls(tmp1Tensor, xCoorTensor1, -1.f, calCount);
        Add(tmp1Tensor, xCoorTensor2, tmp1Tensor, calCount);
        Muls(tmp2Tensor, yCoorTensor1, -1.f, calCount);
        Add(tmp2Tensor, yCoorTensor2, tmp2Tensor, calCount);
        Mul(dst, tmp1Tensor, tmp2Tensor, calCount);
    }

    __aicore__ inline void WithinBounds2d(LocalTensor<float> dst, LocalTensor<float> iyT, LocalTensor<float> ixT,
        LocalTensor<float> weight, const int32_t calCount)
    {
        int32_t newCalCount =
            ((calCount * FLOAT_BYTES + ALIGN_256_BYTES - 1) / ALIGN_256_BYTES * ALIGN_256_BYTES) / FLOAT_BYTES;
        CompareScalar(mask1Tensor, iyT, 0.f, CMPMODE::GE, newCalCount);
        CompareScalar(mask2Tensor, iyT, static_cast<float>(height), CMPMODE::LT, newCalCount);
        int8ToInt16Mask1 = mask1Tensor.ReinterpretCast<uint16_t>();
        int8ToInt16Mask2 = mask2Tensor.ReinterpretCast<uint16_t>();

        // Read data according to int16
        And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);
        CompareScalar(mask1Tensor, ixT, 0.f, CMPMODE::GE, newCalCount);
        And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);
        CompareScalar(mask1Tensor, ixT, static_cast<float>(width), CMPMODE::LT, newCalCount);
        And(int8ToInt16Mask2, int8ToInt16Mask1, int8ToInt16Mask2, maskNum / 2);
        Select(dst, int8ToInt16Mask2, dupOneTensor, 0.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
        Select(weight, int8ToInt16Mask2, weight, 0.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, newCalCount);
    }

    __aicore__ inline void ComputeIndex(
        LocalTensor<int32_t> dstIndex, LocalTensor<int32_t> yCoor, LocalTensor<int32_t> xCoor, const int32_t calCount)
    {
        Mins(yCoor, yCoor, height - 1, calCount);
        Maxs(yCoor, yCoor, 0, calCount);
        Mins(xCoor, xCoor, width - 1, calCount);
        Maxs(xCoor, xCoor, 0, calCount);

        Muls(yCoor, yCoor, dxStrideH, calCount);
        Add(dstIndex, yCoor, xCoor, calCount);
        Muls(dstIndex, dstIndex, channel, calCount);
    }

    __aicore__ inline void CopyOut(const int32_t offset, const int32_t calCount)
    {
        LocalTensor<float> dstLocal = dataOutQueue[1].DeQue<float>();
        DataCopyParams copyParams {1, 0, 0, 0};
        copyParams.blockLen = calCount * sizeof(float);
        DataCopyPad(inputGm[DGRID_INPUT_INDEX][offset], dstLocal, copyParams);
        dataOutQueue[1].FreeTensor(dstLocal);
    }

private:
    TPipe* pipe;
    GlobalTensor<float> inputGm[INPUT_NUM + OUTPUT_NUM];

    TQue<QuePosition::VECIN, BUFFER_NUM> dataInQueue[INPUT_NUM];
    TQue<QuePosition::VECOUT, BUFFER_NUM> dataOutQueue[OUTPUT_NUM];

    TBuf<TPosition::VECCALC> xCoordinateBuf, yCoordinateBuf, xGradInBuf, yGradInBuf;
    TBuf<TPosition::VECCALC> nwBuf, neBuf, swBuf, seBuf;
    TBuf<TPosition::VECCALC> ixBuf, iyBuf, ixIntBuf, iyIntBuf;
    TBuf<TPosition::VECCALC> tmp1Buf, tmp2Buf;
    TBuf<TPosition::VECCALC> mask1Buf, mask2Buf;
    TBuf<TPosition::VECCALC> dupOneBuf, selBuf1, selBuf2, selBuf3, selBuf4;
    TBuf<TPosition::VECCALC> computeIndexBuf2, computeIndexBuf3, computeIndexBuf4, computeIndexBuf5;
    TBuf<TPosition::VECCALC> giCoorBuf, sumBuf;

    uint32_t batch, pNumPerCore, tailPNum;
    int32_t channel, alignedChannel, height, width;
    uint32_t interpolation, padding;
    bool alignCorners;
    uint32_t gridH, gridW;
    uint32_t blockNum, calcCountPerLoop, blockIdx, baseOffset;
    uint32_t maskSize, maskNum;
    uint32_t dxStrideN;
    int32_t dxStrideH;
    int64_t pointIndex;
    int64_t gradGmOffset, xGmOffset;
    int32_t ncOffset;

    int32_t groupSize, coordPosition;
    int32_t nwCpInOffsetStart, neCpInOffsetStart, swCpInOffsetStart, seCpInOffsetStart;
    int32_t nwOffset, neOffset, swOffset, seOffset;

    float gix, giy;

    LocalTensor<uint8_t> mask1Tensor, mask2Tensor;
    LocalTensor<uint16_t> int8ToInt16Mask1, int8ToInt16Mask2;
    LocalTensor<float> dupOneTensor, selTensor1, selTensor2, selTensor3, selTensor4;
    LocalTensor<float> tmp1Tensor, tmp2Tensor;
    LocalTensor<float> giCoorLocalTensor;
    LocalTensor<float> sum;
};

extern "C" __global__ __aicore__ void grid_sampler2d_v2_grad(
    GM_ADDR grad, GM_ADDR x, GM_ADDR grid, GM_ADDR dx, GM_ADDR dgrid, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    GridSampler2dV2GradKernel op;
    GM_ADDR gmTensor[5] = {grad, x, grid, dx, dgrid};
    op.Init(&tilingData, gmTensor);
    op.InitTask();
    op.InitBuffer(&pipe);
    op.InitLocalTensor();
    op.Process();
}