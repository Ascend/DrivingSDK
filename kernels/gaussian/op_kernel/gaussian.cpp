/*
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
*/
#include "kernel_operator.h"
using namespace AscendC;

class KernelGaussian {
public:
    __aicore__ inline KernelGaussian() = delete;

    __aicore__ inline KernelGaussian(
        GM_ADDR gt_boxes,
        GM_ADDR center_int,
        GM_ADDR radius,
        GM_ADDR mask,
        GM_ADDR ind,
        GM_ADDR ret_boxes,
        const GaussianTilingData& tiling_data,
        TPipe* pipe)
        : pipe_(pipe)
    {
        InitTask(tiling_data);
        InitGM(gt_boxes, center_int, radius, mask, ind, ret_boxes);
        InitBuffer();
    }

    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTask(const GaussianTilingData& tiling)
    {
        coreId = GetBlockIdx();
        usedCoreNum = tiling.usedCoreNum;
        numObjs = tiling.numObjs;
        totalCoreTaskNum = tiling.totalCoreTaskNum;
        coreProcessTaskNum = tiling.coreProcessTaskNum;
        lastCoreProcessTaskNum = tiling.lastCoreProcessTaskNum;
        singleProcessTaskNum = tiling.singleProcessTaskNum;
        featureMapSizeX = tiling.featureMapSizeX;
        featureMapSizeY = tiling.featureMapSizeY;
        voxelXSize = tiling.voxelXSize;
        voxelYSize = tiling.voxelYSize;
        prcX = tiling.prcX;
        prcY = tiling.prcY;
        featureMapStride = tiling.featureMapStride;
        numMaxObjs = tiling.numMaxObjs;
        minRadius = tiling.minRadius;
        minOverLap = tiling.minOverLap;
        dimSize = tiling.dimSize;
        normBbox = tiling.normBbox;
        flipAngle = tiling.flipAngle;
        curCoreTaskNum = coreProcessTaskNum;
        if (unlikely(coreId == usedCoreNum - 1)) {
            curCoreTaskNum = lastCoreProcessTaskNum;
        }
        coreRepeatTimes = (curCoreTaskNum - 1) / singleProcessTaskNum + 1;
        a1 = 1;
        a2 = 4;
        a3 = 4 * minOverLap;
    }

    __aicore__ inline void InitGM(GM_ADDR gt_boxes,
                                  GM_ADDR center_int,
                                  GM_ADDR radius,
                                  GM_ADDR mask,
                                  GM_ADDR ind,
                                  GM_ADDR ret_boxes)
    {
        gtBoxesGm.SetGlobalBuffer((__gm__ float*)(gt_boxes));
        centerIntGm.SetGlobalBuffer((__gm__ int32_t*)(center_int));
        radiusGm.SetGlobalBuffer((__gm__ int32_t*)(radius));
        maskGm.SetGlobalBuffer((__gm__ uint8_t*)(mask));
        indGm.SetGlobalBuffer((__gm__ int32_t*)(ind));
        retBoxesGm.SetGlobalBuffer((__gm__ float*)(ret_boxes));
    }

     __aicore__ inline void InitBuffer()
    {
        pipe_->InitBuffer(gtBoxesQue_, singleProcessTaskNum * dimSize * sizeof(float));
        pipe_->InitBuffer(pcrUB, singleProcessTaskNum * 2 * sizeof(float));
        pipe_->InitBuffer(voxelSizeUB, singleProcessTaskNum * 2 * sizeof(float));
        pipe_->InitBuffer(featureMapStrideUB, singleProcessTaskNum * sizeof(float));
        pipe_->InitBuffer(coordUB, singleProcessTaskNum * 2 * sizeof(float));
        pipe_->InitBuffer(centerIntUB, singleProcessTaskNum * 2 * sizeof(int32_t));
        pipe_->InitBuffer(centerFloatUB, singleProcessTaskNum * 2 * sizeof(float));
        pipe_->InitBuffer(dxUb, singleProcessTaskNum * sizeof(float));
        pipe_->InitBuffer(dyUb, singleProcessTaskNum * sizeof(float));
        pipe_->InitBuffer(sumDxDyUB, singleProcessTaskNum * sizeof(float));
        pipe_->InitBuffer(mulDxDyUB, singleProcessTaskNum * sizeof(float));
        pipe_->InitBuffer(bUB, singleProcessTaskNum * 3 * sizeof(int32_t));
        pipe_->InitBuffer(cUB, singleProcessTaskNum * 3 * sizeof(float));
        pipe_->InitBuffer(sqrtUB, singleProcessTaskNum * 3 * sizeof(float));
        pipe_->InitBuffer(rUB, singleProcessTaskNum * sizeof(float));
        pipe_->InitBuffer(radiusUB, singleProcessTaskNum * sizeof(int32_t));
        pipe_->InitBuffer(cmpUB, singleProcessTaskNum * sizeof(int32_t));
        pipe_->InitBuffer(maskHalfUB, singleProcessTaskNum * sizeof(float));
        pipe_->InitBuffer(maskUB, singleProcessTaskNum * sizeof(int32_t));
        pipe_->InitBuffer(indUB, singleProcessTaskNum * sizeof(int32_t));
        pipe_->InitBuffer(indFloatUB, singleProcessTaskNum * sizeof(float));
        pipe_->InitBuffer(retBoxesUB, singleProcessTaskNum * (dimSize + 1) * sizeof(float));
    }

    __aicore__ inline void ProcessSingle(uint64_t taskIdx, uint32_t actualTaskNum)
    {
        uint64_t singleBaseGmOffset =  coreId * coreProcessTaskNum + taskIdx * singleProcessTaskNum;
        uint32_t copyLen = AlignUp(actualTaskNum, 8);
        uint32_t halfCopyLen = AlignUp(actualTaskNum, 16);
        uint32_t uintCopyLen = AlignUp(actualTaskNum, 32);
        LocalTensor<float> gtBoxes = gtBoxesQue_.Get<float>();
        LocalTensor<float> pcr = pcrUB.Get<float>();
        LocalTensor<float> voxelSize = voxelSizeUB.Get<float>();
        LocalTensor<float> featureMapStride_ = featureMapStrideUB.Get<float>();
        LocalTensor<float> coord = coordUB.Get<float>();
        LocalTensor<int32_t> centerInt = centerIntUB.Get<int32_t>();
        LocalTensor<float> centerFloat = centerFloatUB.Get<float>();
        LocalTensor<float> dx = dxUb.Get<float>();
        LocalTensor<float> dy = dyUb.Get<float>();
        LocalTensor<float> sumDxDy = sumDxDyUB.Get<float>();
        LocalTensor<float> mulDxDy = mulDxDyUB.Get<float>();
        LocalTensor<float> bLocal = bUB.Get<float>();
        LocalTensor<float> cLocal = cUB.Get<float>();
        LocalTensor<float> sqrLocal = sqrtUB.Get<float>();
        LocalTensor<float> rLocal = rUB.Get<float>();
        LocalTensor<int32_t> radiusLocal = radiusUB.Get<int32_t>();
        LocalTensor<uint8_t> cmpLocal = cmpUB.Get<uint8_t>();
        LocalTensor<half> maskHalf = maskHalfUB.Get<half>();
        LocalTensor<uint8_t> mask = maskUB.Get<uint8_t>();
        LocalTensor<int32_t> indLocal = indUB.Get<int32_t>();
        LocalTensor<float> indFloatLocal = indFloatUB.Get<float>();
        LocalTensor<float> retBoxes = retBoxesUB.Get<float>();

        Duplicate(pcr, prcX, copyLen);
        Duplicate(pcr[copyLen], prcY, copyLen);
        Duplicate(voxelSize, voxelXSize, copyLen);
        Duplicate(voxelSize[copyLen], voxelYSize, copyLen);
        Duplicate(featureMapStride_, static_cast<float>(featureMapStride), copyLen);
        Duplicate(maskHalf, static_cast<half>(1.0), halfCopyLen);
        for (uint32_t i = 0; i < dimSize; i++) {
            DataCopy(gtBoxes[copyLen * i], gtBoxesGm[singleBaseGmOffset + numObjs * i], copyLen);
        }
        pipe_barrier(PIPE_ALL);
        Sub(coord, gtBoxes, pcr, copyLen);
        Sub(coord[copyLen], gtBoxes[copyLen], pcr[copyLen], copyLen);
        Div(coord, coord, voxelSize, copyLen);
        Div(coord[copyLen], coord[copyLen], voxelSize[copyLen], copyLen);
        Div(coord, coord, featureMapStride_, copyLen);
        Div(coord[copyLen], coord[copyLen], featureMapStride_, copyLen);
        Cast(centerInt,  coord, RoundMode::CAST_TRUNC, copyLen * 2);
        Cast(centerFloat, coord, RoundMode::CAST_TRUNC, copyLen * 2);
        Div(dx, gtBoxes[copyLen * 3], voxelSize, copyLen);
        Div(dy, gtBoxes[copyLen * 4], voxelSize[copyLen], copyLen);
        Div(dx, dx, featureMapStride_, copyLen);
        Div(dy, dy, featureMapStride_, copyLen);
        Add(sumDxDy, dx, dy, copyLen);
        Mul(mulDxDy, dx, dy, copyLen);
        Muls(bLocal, sumDxDy, 1.0f, copyLen);
        Muls(bLocal[copyLen], sumDxDy, 2.0f, copyLen);
        Muls(bLocal[copyLen * 2], sumDxDy, (-2.0f * minOverLap), copyLen);
        Muls(cLocal, mulDxDy, (1.0f - minOverLap) / (1.0f + minOverLap), copyLen);
        Muls(cLocal[copyLen], mulDxDy, (1.0f - minOverLap), copyLen);
        Muls(cLocal[copyLen * 2], mulDxDy, (minOverLap - 1.0f), copyLen);
        Muls(cLocal, cLocal, 4.0f, copyLen * 3);
        Muls(cLocal, cLocal, a1, copyLen);
        Muls(cLocal[copyLen], cLocal[copyLen], a2, copyLen);
        Muls(cLocal[copyLen * 2], cLocal[copyLen * 2], a3, copyLen);
        Mul(sqrLocal, bLocal, bLocal, copyLen * 3);
        Sub(sqrLocal, sqrLocal, cLocal, copyLen * 3);
        Sqrt(sqrLocal, sqrLocal, copyLen * 3);
        Add(sqrLocal, sqrLocal, bLocal, copyLen * 3);
        Muls(sqrLocal, sqrLocal, 0.5f, copyLen * 3);
        Min(rLocal, sqrLocal, sqrLocal[copyLen], copyLen);
        Min(rLocal, rLocal, sqrLocal[copyLen * 2], copyLen);
        Cast(radiusLocal, rLocal, RoundMode::CAST_TRUNC, copyLen);
        Maxs(radiusLocal, radiusLocal, minRadius, copyLen);
        // mask
        CompareScalar(cmpLocal, dx, 0.0f, CMPMODE::GT, AlignUp(copyLen, 64));
        Select(maskHalf, cmpLocal, maskHalf, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, copyLen);
        CompareScalar(cmpLocal, dy, 0.0f, CMPMODE::GT, AlignUp(copyLen, 64));
        Select(maskHalf, cmpLocal, maskHalf, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, copyLen);
        CompareScalar(cmpLocal, centerFloat, 0.0f, CMPMODE::GE, AlignUp(copyLen, 64));
        Select(maskHalf, cmpLocal, maskHalf, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, copyLen);
        CompareScalar(cmpLocal, centerFloat[copyLen], 0.0f, CMPMODE::GE, AlignUp(copyLen, 64));
        Select(maskHalf, cmpLocal, maskHalf, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, copyLen);
        CompareScalar(cmpLocal, centerFloat, static_cast<float>(featureMapSizeX), CMPMODE::LT, AlignUp(copyLen, 64));
        Select(maskHalf, cmpLocal, maskHalf, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, copyLen);
        CompareScalar(cmpLocal, centerFloat[copyLen], static_cast<float>(featureMapSizeY), CMPMODE::LT, AlignUp(copyLen, 64));
        Select(maskHalf, cmpLocal, maskHalf, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, copyLen);
        Cast(mask, maskHalf, RoundMode::CAST_NONE, copyLen);
        CompareScalar(cmpLocal, maskHalf, (half)1.0, CMPMODE::EQ, AlignUp(copyLen, 128));
        // ind
        Muls(indLocal, centerInt[copyLen], featureMapSizeX, copyLen);
        Add(indLocal, indLocal, centerInt, copyLen);
        Cast(indFloatLocal, indLocal, RoundMode::CAST_TRUNC, copyLen);
        Select(indFloatLocal, cmpLocal, indFloatLocal, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, copyLen);
        Cast(indLocal, indFloatLocal, RoundMode::CAST_TRUNC, copyLen);
        // ret
        for (uint32_t i = 0; i < 2; i++) {
            Sub(retBoxes[copyLen * i], coord[copyLen * i], centerFloat[copyLen * i], copyLen);
        }
        Muls(retBoxes[copyLen * 2], gtBoxes[copyLen * 2], 1.0f, copyLen * 4);
        if (normBbox == true) {
            Log(retBoxes[copyLen * 3], retBoxes[copyLen * 3], copyLen * 3);
        }
        Sin(retBoxes[copyLen * 6], gtBoxes[copyLen * 6], copyLen);
        Cos(retBoxes[copyLen * 7], gtBoxes[copyLen * 6], copyLen);
        if (flipAngle == true) {
            Cos(retBoxes[copyLen * 6], gtBoxes[copyLen * 6], copyLen);
            Sin(retBoxes[copyLen * 7], gtBoxes[copyLen * 6], copyLen);
        }
        for (uint32_t i = 7; i < dimSize ; i++) {
            Muls(retBoxes[copyLen * (i + 1)], gtBoxes[copyLen * i], 1.0f, copyLen);
        }
        for (uint32_t i = 0; i < dimSize + 1; i++) {
            Select(retBoxes[copyLen * i], cmpLocal, retBoxes[copyLen * i], 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, copyLen);
        }
        pipe_barrier(PIPE_ALL);
        DataCopyExtParams centerIntCopyParams {1, (uint16_t)(actualTaskNum * sizeof(int32_t)), 0, 0, 0};
        DataCopyExtParams radiusCopyParams {1, (uint16_t)(actualTaskNum * sizeof(int32_t)), 0, 0, 0};
        DataCopyExtParams maskCopyParams {1, (uint16_t)(actualTaskNum * sizeof(uint8_t)), 0, 0, 0};
        DataCopyExtParams indCopyParams {1, (uint16_t)(actualTaskNum * sizeof(int32_t)), 0, 0, 0};
        DataCopyExtParams retCopyParams {1, (uint16_t)(actualTaskNum * sizeof(float)), 0, 0, 0};

        DataCopyPad(centerIntGm[singleBaseGmOffset], centerInt, centerIntCopyParams);
        DataCopyPad(centerIntGm[singleBaseGmOffset + totalCoreTaskNum], centerInt[copyLen], centerIntCopyParams);
        DataCopyPad(radiusGm[singleBaseGmOffset], radiusLocal, radiusCopyParams);
        DataCopyPad(maskGm[singleBaseGmOffset], mask, maskCopyParams);
        DataCopyPad(indGm[singleBaseGmOffset], indLocal, indCopyParams);
        for (uint32_t i = 0; i < dimSize + 1; i++) {
            DataCopyPad(retBoxesGm[singleBaseGmOffset + numMaxObjs * i], retBoxes[copyLen * i], retCopyParams);
        }
        pipe_barrier(PIPE_ALL);
    }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> gtBoxesQue_, pcrUB, voxelSizeUB, featureMapStrideUB, coordUB;
    TBuf<TPosition::VECCALC> centerIntUB, centerFloatUB, dxUb, dyUb, sumDxDyUB, mulDxDyUB;
    TBuf<TPosition::VECCALC> bUB, cUB, sqrtUB, rUB, radiusUB, cmpUB;
    TBuf<TPosition::VECCALC> maskHalfUB, maskUB, indUB, indFloatUB, retBoxesUB;
    GlobalTensor<float> gtBoxesGm, retBoxesGm;
    GlobalTensor<int32_t> centerIntGm, radiusGm, indGm;
    GlobalTensor<uint8_t> maskGm;
    float a1, a2, a3;
    float prcX, prcY, voxelXSize, voxelYSize, minOverLap;
    int32_t numMaxObjs, numObjs, featureMapStride, minRadius, featureMapSizeX, featureMapSizeY, dimSize;
    bool normBbox, flipAngle;
    int32_t coreId, usedCoreNum, totalCoreTaskNum, coreProcessTaskNum;
    int32_t lastCoreProcessTaskNum, singleProcessTaskNum, curCoreTaskNum, coreRepeatTimes;
};

__aicore__ inline void KernelGaussian::Process()
{
    for (uint32_t i = 0; i < coreRepeatTimes; ++i) {
        uint32_t actualTaskNum = singleProcessTaskNum;
        if (unlikely(i == coreRepeatTimes - 1)) {
            actualTaskNum = (curCoreTaskNum - 1) % singleProcessTaskNum + 1;
        }
        ProcessSingle(i, actualTaskNum);
        pipe_barrier(PIPE_ALL);
    }
}

extern "C" __global__ __aicore__ void gaussian(GM_ADDR gt_boxes, GM_ADDR center_int,
                                               GM_ADDR radius, GM_ADDR mask,
                                               GM_ADDR ind, GM_ADDR ret_boxes,
                                               GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    KernelGaussian op(
        gt_boxes,
        center_int,
        radius,
        mask,
        ind,
        ret_boxes,
        tiling_data,
        &pipe
    );
    op.Process();
}