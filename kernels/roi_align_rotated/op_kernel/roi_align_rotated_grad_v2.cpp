/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include "kernel_operator.h"

using namespace AscendC;

class KernelRoiAlignRotatedGradV2 {
public:
    __aicore__ inline KernelRoiAlignRotatedGradV2() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR rois, GM_ADDR grad_output, GM_ADDR grad_input,
                                const RoiAlignRotatedGradV2TilingData *__restrict tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        coreRoisNums = tiling_data->coreRoisNums;
        coreRoisTail = tiling_data->coreRoisTail;
        boxSize = tiling_data->boxSize;
        pooledWidth = tiling_data->pooledWidth;
        pooledHeight = tiling_data->pooledHeight;
        batchSize = tiling_data->batchSize;
        channelNum = tiling_data->channelNum;
        width = tiling_data->width;
        height = tiling_data->height;
        aligned = tiling_data->aligned;
        clockwise = tiling_data->clockwise;
        samplingRatio = tiling_data->samplingRatio;
        spatialScale = tiling_data->spatialScale;

        dataSize = 32 / sizeof(DTYPE_INPUT);
        alignChannelNum = (channelNum + dataSize - 1) / dataSize;
        alignChannelNum = alignChannelNum * dataSize;

        uint32_t coreId = GetBlockIdx();
        if (coreId < coreRoisTail) {
            coreRoisNums += 1;
            startOffset = coreRoisNums * coreId;
        } else {
            startOffset = coreRoisNums * coreId + coreRoisTail;
        }

        eventIdMte2ToV = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());
        eventIdMte3ToV = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_V>());
        eventIdVToMte2 = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMte3 = static_cast<event_t>(pipe.AllocEventID<HardEvent::V_MTE3>());
        eventIdMte3ToMte2 = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE3_MTE2>());

        copyParams = {2, static_cast<uint16_t>(channelNum * 2 / dataSize), 0,
                      static_cast<uint16_t>((width - 2) * channelNum / dataSize)};

        inputGM.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INPUT*>(input),
                                static_cast<uint64_t>(batchSize) * channelNum * height * width);
        roisGM.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INPUT*>(rois),
                               static_cast<uint64_t>(boxLength) * boxSize);
        gradOutputsGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INPUT*>(grad_output),
                                      static_cast<uint64_t>(boxLength) * channelNum * pooledHeight * pooledWidth);
        gradInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INPUT*>(grad_input),
                                    static_cast<uint64_t>(batchSize) * channelNum * height * width);
        InitBuffer();
    }

    __aicore__ inline void Process()
    {
        GetLocalTensor();
        uint32_t computeBatchSize = constComputeBatchSize;
        uint32_t computeBatchNum = (coreRoisNums + constComputeBatchSize - 1) / constComputeBatchSize;
        for (uint32_t taskBatchIdx = 0; taskBatchIdx < computeBatchNum; taskBatchIdx++) {
            uint32_t offset = startOffset + taskBatchIdx * constComputeBatchSize;
            if (taskBatchIdx == computeBatchNum - 1) {
                computeBatchSize = coreRoisNums - taskBatchIdx * computeBatchSize;
            }
            uint32_t alignComputeBatchNum = (computeBatchSize + dataSize - 1) / dataSize;
            alignComputeBatchNum = alignComputeBatchNum * dataSize;
            CopyIn(offset, alignComputeBatchNum);
            for (uint32_t taskIdx = 0; taskIdx < computeBatchSize; taskIdx++) {
                Compute(taskIdx, offset + taskIdx);
            }
        }
    }

    __aicore__ inline void InitBuffer()
    {
        pipe.InitBuffer(idxUb, constComputeBatchSize * sizeof(int32_t));
        pipe.InitBuffer(xUb, constComputeBatchSize * sizeof(DTYPE_INPUT));
        pipe.InitBuffer(yUb, constComputeBatchSize * sizeof(DTYPE_INPUT));
        pipe.InitBuffer(hUb, constComputeBatchSize * sizeof(DTYPE_INPUT));
        pipe.InitBuffer(wUb, constComputeBatchSize * sizeof(DTYPE_INPUT));
        pipe.InitBuffer(angleUb, constComputeBatchSize * sizeof(DTYPE_INPUT));

        pipe.InitBuffer(cosUb, constComputeBatchSize * sizeof(DTYPE_INPUT));
        pipe.InitBuffer(sinUb, constComputeBatchSize * sizeof(DTYPE_INPUT));

        pipe.InitBuffer(binSizeHUb, constComputeBatchSize * sizeof(DTYPE_INPUT));
        pipe.InitBuffer(binSizeWUb, constComputeBatchSize * sizeof(DTYPE_INPUT));

        pipe.InitBuffer(binGridWUb, constComputeBatchSize * sizeof(int32_t));
        pipe.InitBuffer(binGridHUb, constComputeBatchSize * sizeof(int32_t));

        pipe.InitBuffer(binGridSizeWUb, constComputeBatchSize * sizeof(DTYPE_INPUT));
        pipe.InitBuffer(binGridSizeHUb, constComputeBatchSize * sizeof(DTYPE_INPUT));

        pipe.InitBuffer(deltaStartWUb, constComputeBatchSize * sizeof(DTYPE_INPUT));
        pipe.InitBuffer(deltaStartHUb, constComputeBatchSize * sizeof(DTYPE_INPUT));

        pipe.InitBuffer(countTmpUb, constComputeBatchSize * sizeof(int32_t));
        pipe.InitBuffer(countUb, constComputeBatchSize * sizeof(DTYPE_INPUT));
        pipe.InitBuffer(tmpUb, constComputeBatchSize * sizeof(DTYPE_INPUT));

        pipe.InitBuffer(gradBinUb, alignChannelNum * sizeof(DTYPE_INPUT));
        pipe.InitBuffer(gradW1Ub, alignChannelNum * sizeof(DTYPE_INPUT));
        pipe.InitBuffer(gradW2Ub, alignChannelNum * sizeof(DTYPE_INPUT));
        pipe.InitBuffer(gradW3Ub, alignChannelNum * sizeof(DTYPE_INPUT));
        pipe.InitBuffer(gradW4Ub, alignChannelNum * sizeof(DTYPE_INPUT));

        pipe.InitBuffer(gradOutUb, 4 * alignChannelNum * sizeof(DTYPE_INPUT));

        pipe.InitBuffer(tmpChannelUb, alignChannelNum * sizeof(DTYPE_INPUT));
    }

    __aicore__ inline void GetLocalTensor()
    {
        idxLocal = idxUb.Get<int32_t>();
        xLocal = xUb.Get<DTYPE_INPUT>();
        yLocal = yUb.Get<DTYPE_INPUT>();
        hLocal = hUb.Get<DTYPE_INPUT>();
        wLocal = wUb.Get<DTYPE_INPUT>();
        angleLocal = angleUb.Get<DTYPE_INPUT>();

        cosLocal = cosUb.Get<DTYPE_INPUT>();
        sinLocal = sinUb.Get<DTYPE_INPUT>();

        binSizeHLocal = binSizeHUb.Get<DTYPE_INPUT>();
        binSizeWLocal = binSizeWUb.Get<DTYPE_INPUT>();

        binGridWLocal = binGridWUb.Get<int32_t>();
        binGridHLocal = binGridHUb.Get<int32_t>();

        binGridSizeWLocal = binGridSizeWUb.Get<DTYPE_INPUT>();
        binGridSizeHLocal = binGridSizeHUb.Get<DTYPE_INPUT>();

        deltaStartWLocal = deltaStartWUb.Get<DTYPE_INPUT>();
        deltaStartHLocal = deltaStartHUb.Get<DTYPE_INPUT>();

        countTmpLocal = countTmpUb.Get<int32_t>();
        countLocal = countUb.Get<DTYPE_INPUT>();
        tmpLocal = tmpUb.Get<DTYPE_INPUT>();

        gradBinLocal = gradBinUb.Get<DTYPE_INPUT>();
        gradW1Local = gradW1Ub.Get<DTYPE_INPUT>();
        gradW2Local = gradW2Ub.Get<DTYPE_INPUT>();
        gradW3Local = gradW3Ub.Get<DTYPE_INPUT>();
        gradW4Local = gradW4Ub.Get<DTYPE_INPUT>();

        gradOutLocal = gradOutUb.Get<DTYPE_INPUT>();

        tmpChannelLocal = tmpChannelUb.Get<DTYPE_INPUT>();
        Duplicate(tmpChannelLocal, (DTYPE_INPUT)0.0, alignChannelNum);
        Duplicate(tmpChannelLocal, (DTYPE_INPUT)1.0, channelNum);
    }
private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t computeBatchSize)
    {
        DataCopy(tmpLocal, roisGM[offset + static_cast<uint64_t>(boxSize) * 0], computeBatchSize);
        DataCopy(xLocal, roisGM[offset + static_cast<uint64_t>(boxSize) * 1], computeBatchSize);
        DataCopy(yLocal, roisGM[offset + static_cast<uint64_t>(boxSize) * 2], computeBatchSize);
        DataCopy(wLocal, roisGM[offset + static_cast<uint64_t>(boxSize) * 3], computeBatchSize);
        DataCopy(hLocal, roisGM[offset + static_cast<uint64_t>(boxSize) * 4], computeBatchSize);
        DataCopy(angleLocal, roisGM[offset + static_cast<uint64_t>(boxSize) * 5], computeBatchSize);

        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        if (clockwise) {
            Muls(angleLocal, angleLocal, (DTYPE_INPUT)-1.0, computeBatchSize);
        }

        Cast(idxLocal, tmpLocal, AscendC::RoundMode::CAST_RINT, computeBatchSize);
        Muls(xLocal, xLocal, (DTYPE_INPUT)spatialScale, computeBatchSize);
        Muls(yLocal, yLocal, (DTYPE_INPUT)spatialScale, computeBatchSize);
        Muls(hLocal, hLocal, (DTYPE_INPUT)spatialScale, computeBatchSize);
        Muls(wLocal, wLocal, (DTYPE_INPUT)spatialScale, computeBatchSize);

        if (aligned) {
            Adds(xLocal, xLocal, (DTYPE_INPUT)-0.5, computeBatchSize);
            Adds(yLocal, yLocal, (DTYPE_INPUT)-0.5, computeBatchSize);
        } else {
            Maxs(hLocal, hLocal, (DTYPE_INPUT)1.0, computeBatchSize);
            Maxs(wLocal, wLocal, (DTYPE_INPUT)1.0, computeBatchSize);
        }

        Cos(cosLocal, angleLocal, computeBatchSize);
        Sin(sinLocal, angleLocal, computeBatchSize);

        Duplicate(tmpLocal, (DTYPE_INPUT)pooledHeight, computeBatchSize);
        Div(binSizeHLocal, hLocal, tmpLocal, computeBatchSize);
        Duplicate(tmpLocal, (DTYPE_INPUT)pooledWidth, computeBatchSize);
        Div(binSizeWLocal, wLocal, tmpLocal, computeBatchSize);

        if (samplingRatio > 0) {
            Duplicate(binGridHLocal, samplingRatio, computeBatchSize);
            Duplicate(binGridWLocal, samplingRatio, computeBatchSize);
        } else {
            Cast(binGridHLocal, binSizeHLocal, AscendC::RoundMode::CAST_CEIL, computeBatchSize);
            Cast(binGridWLocal, binSizeWLocal, AscendC::RoundMode::CAST_CEIL, computeBatchSize);
        }

        Cast(tmpLocal, binGridHLocal, AscendC::RoundMode::CAST_NONE, computeBatchSize);
        Div(binGridSizeHLocal, binSizeHLocal, tmpLocal, computeBatchSize);

        Cast(tmpLocal, binGridWLocal, AscendC::RoundMode::CAST_NONE, computeBatchSize);
        Div(binGridSizeWLocal, binSizeWLocal, tmpLocal, computeBatchSize);

        Muls(deltaStartWLocal, wLocal, (DTYPE_INPUT)-0.5, computeBatchSize);
        Muls(deltaStartHLocal, hLocal, (DTYPE_INPUT)-0.5, computeBatchSize);

        Mul(countTmpLocal, binGridWLocal, binGridHLocal, computeBatchSize);
        Cast(countLocal, countTmpLocal, AscendC::RoundMode::CAST_NONE, computeBatchSize);
        Maxs(countLocal, countLocal, (DTYPE_INPUT)1.0, computeBatchSize);
    }

    __aicore__ inline void Compute(uint32_t taskIdx, uint64_t offset)
    {
        pIdx = idxLocal.GetValue(taskIdx);
        pX = xLocal.GetValue(taskIdx);
        pY = yLocal.GetValue(taskIdx);
        pCos = cosLocal.GetValue(taskIdx);
        pSin = sinLocal.GetValue(taskIdx);

        pBinSizeW = binSizeWLocal.GetValue(taskIdx);
        pBinSizeH = binSizeHLocal.GetValue(taskIdx);
        pBinGridW = binGridWLocal.GetValue(taskIdx);
        pBinGridH = binGridHLocal.GetValue(taskIdx);

        pDeltaStartW = deltaStartWLocal.GetValue(taskIdx);
        pDeltaStartH = deltaStartHLocal.GetValue(taskIdx);

        pBinGridSizeH = binGridSizeHLocal.GetValue(taskIdx);
        pBinGridSizeW = binGridSizeWLocal.GetValue(taskIdx);

        pCount = countLocal.GetValue(taskIdx);

        for (index = 0; index < pooledHeight * pooledWidth; index++) {
            pH = index / pooledWidth;
            pW = index - pH * pooledWidth;
            baseOffset = ((offset * pooledHeight + pH) * pooledWidth + pW) * channelNum;

            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            DataCopy(gradBinLocal, gradOutputsGm[baseOffset], alignChannelNum);

            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            if (alignChannelNum != channelNum) {
                Mul(gradBinLocal, gradBinLocal, tmpChannelLocal, alignChannelNum);
            }
            Muls(gradBinLocal, gradBinLocal, (DTYPE_INPUT)1.0 / (DTYPE_INPUT)pCount, alignChannelNum);

            for (iy = 0; iy < pBinGridH; iy++) {
                yy = pDeltaStartH + pH * pBinSizeH + (iy + DTYPE_INPUT(0.5)) * pBinGridSizeH;
                for (ix = 0; ix < pBinGridW; ix++) {
                    xx = pDeltaStartW + pW * pBinSizeW + (ix + DTYPE_INPUT(0.5)) * pBinGridSizeW;

                    x = yy * pSin + xx * pCos + pX;
                    y = yy * pCos - xx * pSin + pY;

                    bilinearInterpolate();

                    if (xl >= 0 && xh >= 0 && yl >= 0 && yh >= 0) {
                        if (channelNum == alignChannelNum && xh > xl && yh > yl) {
                            CopyOutTogether();
                        } else {
                            CopyOut();
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void bilinearInterpolate()
    {
        if (y < -1 || y > height || x < -1 || x > width) {
            xl = -1;
            return ;
        }
        if (y <= 0) y = 0;
        if (x <= 0) x = 0;

        yl = static_cast<int32_t>(y);
        xl = static_cast<int32_t>(x);

        if (yl >= height - 1) {
            yl = yh = height - 1;
            y = DTYPE_INPUT(yl);
        } else {
            yh = yl + 1;
        }

        if (xl >= width - 1) {
            xl = xh = width - 1;
            x = DTYPE_INPUT(xl);
        } else {
            xh = xl + 1;
        }

        ly = y - yl;
        lx = x - xl;

        w4 = ly * lx;
        w1 = w4 + 1 - ly - lx;
        w2 = lx - w4;
        w3 = ly - w4;
    }

    __aicore__ inline void CopyOut()
    {
        w1Offset = ((static_cast<uint64_t>(pIdx) * height+ yl)* width + xl) * channelNum;
        w2Offset = ((static_cast<uint64_t>(pIdx) * height+ yl)* width + xh) * channelNum;
        w3Offset = ((static_cast<uint64_t>(pIdx) * height+ yh)* width + xl) * channelNum;
        w4Offset = ((static_cast<uint64_t>(pIdx) * height+ yh)* width + xh) * channelNum;

        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        Muls(gradW1Local, gradBinLocal, w1, alignChannelNum);
        Muls(gradW2Local, gradBinLocal, w2, alignChannelNum);
        Muls(gradW3Local, gradBinLocal, w3, alignChannelNum);
        Muls(gradW4Local, gradBinLocal, w4, alignChannelNum);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        SetAtomicAdd<DTYPE_INPUT>();

        DataCopy(gradInputGm[w1Offset], gradW1Local, alignChannelNum);
        DataCopy(gradInputGm[w2Offset], gradW2Local, alignChannelNum);
        DataCopy(gradInputGm[w3Offset], gradW3Local, alignChannelNum);
        DataCopy(gradInputGm[w4Offset], gradW4Local, alignChannelNum);

        SetAtomicNone();
    }

    __aicore__ inline void CopyOutTogether()
    {
        w1Offset = ((pIdx * height+ yl)* width + xl) * channelNum;

        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        Muls(gradOutLocal, gradBinLocal, w1, channelNum);
        Muls(gradOutLocal[channelNum], gradBinLocal, w2, channelNum);
        Muls(gradOutLocal[channelNum * 2], gradBinLocal, w3, channelNum);
        Muls(gradOutLocal[channelNum * 3], gradBinLocal, w4, channelNum);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        SetAtomicAdd<DTYPE_INPUT>();

        DataCopy(gradInputGm[w1Offset], gradOutLocal, copyParams);

        SetAtomicNone();
    }
private:
    TPipe pipe;
    GlobalTensor<DTYPE_INPUT> inputGM, roisGM, gradOutputsGm, gradInputGm;

    TBuf <TPosition::VECCALC> idxUb, xUb, yUb, hUb, wUb, angleUb;
    TBuf <TPosition::VECCALC> cosUb, sinUb;
    TBuf <TPosition::VECCALC> binSizeHUb, binSizeWUb;
    TBuf <TPosition::VECCALC> binGridWUb, binGridHUb;
    TBuf <TPosition::VECCALC> binGridSizeWUb, binGridSizeHUb;
    TBuf <TPosition::VECCALC> deltaStartWUb, deltaStartHUb;
    TBuf <TPosition::VECCALC> countTmpUb, countUb;
    TBuf <TPosition::VECCALC> tmpUb;

    TBuf <TPosition::VECCALC> gradBinUb;
    TBuf <TPosition::VECCALC> gradW1Ub, gradW2Ub, gradW3Ub, gradW4Ub;
    TBuf <TPosition::VECCALC> tmpChannelUb;
    TBuf <TPosition::VECCALC> gradOutUb;

    LocalTensor<int32_t> idxLocal;
    LocalTensor<DTYPE_INPUT> xLocal, yLocal, hLocal, wLocal, angleLocal;
    LocalTensor<DTYPE_INPUT> cosLocal, sinLocal;
    LocalTensor<DTYPE_INPUT> binSizeHLocal, binSizeWLocal;
    LocalTensor<int32_t> binGridWLocal, binGridHLocal;
    LocalTensor<DTYPE_INPUT> binGridSizeWLocal, binGridSizeHLocal;
    LocalTensor<DTYPE_INPUT> deltaStartWLocal, deltaStartHLocal;
    LocalTensor<int32_t> countTmpLocal;
    LocalTensor<DTYPE_INPUT> countLocal;
    LocalTensor<DTYPE_INPUT> tmpLocal;

    LocalTensor<DTYPE_INPUT> gradBinLocal;
    LocalTensor<DTYPE_INPUT> gradW1Local, gradW2Local, gradW3Local, gradW4Local;
    LocalTensor<DTYPE_INPUT> tmpChannelLocal;
    LocalTensor<DTYPE_INPUT> gradOutLocal;

    uint32_t coreRoisNums;
    uint32_t coreRoisTail;
    uint32_t boxSize;
    uint32_t boxLength = 6;
    uint32_t batchSize;
    uint32_t channelNum;
    uint32_t width, height;
    int32_t pooledWidth;
    int32_t pooledHeight;
    bool aligned;
    bool clockwise;
    int32_t samplingRatio;
    float spatialScale;

    uint32_t dataSize;
    uint32_t alignChannelNum;

    uint32_t startOffset;
    uint64_t baseOffset, w1Offset, w2Offset, w3Offset, w4Offset;
    uint32_t constComputeBatchSize = 256;

    int32_t pIdx;
    int32_t index;
    DTYPE_INPUT pX, pY;
    int32_t pH, pW;
    DTYPE_INPUT pCos, pSin;

    DTYPE_INPUT pBinSizeW, pBinSizeH;
    int32_t pBinGridW, pBinGridH;
    DTYPE_INPUT pDeltaStartW, pDeltaStartH;
    DTYPE_INPUT pBinGridSizeH, pBinGridSizeW;

    DTYPE_INPUT pCount;

    DTYPE_INPUT tmpH, tmpW;
    DTYPE_INPUT tmpPH, tmpPW;

    int32_t ix, iy;
    DTYPE_INPUT xx, yy;
    DTYPE_INPUT x, y;
    int32_t xl, xh;
    int32_t yl, yh;

    DTYPE_INPUT lx, hx;
    DTYPE_INPUT ly, hy;

    DTYPE_INPUT w1, w2, w3, w4;

    event_t eventIdVToMte2, eventIdVToMte3, eventIdMte2ToV, eventIdMte3ToV, eventIdMte3ToMte2;
    AscendC::DataCopyParams copyParams;
};

extern "C" __global__ __aicore__ void roi_align_rotated_grad_v2(GM_ADDR input, GM_ADDR rois, GM_ADDR grad_output, GM_ADDR grad_input, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SetSysWorkspace(workspace);
    GET_TILING_DATA(tilingData, tiling);
    const RoiAlignRotatedGradV2TilingData *__restrict tilingDevice = &tilingData;
    KernelRoiAlignRotatedGradV2 op;
    op.Init(input, rois, grad_output, grad_input, tilingDevice);
    op.Process();
}