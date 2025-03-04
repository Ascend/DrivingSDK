/*
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#include "kernel_operator.h"
using namespace AscendC;

class KernelDrawGaussianToHeatmap {
public:
    __aicore__ inline KernelDrawGaussianToHeatmap() = delete;

    __aicore__ inline KernelDrawGaussianToHeatmap(
        GM_ADDR mask,
        GM_ADDR cur_class_id,
        GM_ADDR center_int,
        GM_ADDR radius,
        GM_ADDR heatmap,
        const DrawGaussianToHeatmapTilingData& tiling_data,
        TPipe* pipe)
        : pipe_(pipe)
    {
        InitTask(tiling_data);
        InitGM(mask, cur_class_id, center_int, radius, heatmap);
        InitBuffer();
        InitEvent();
    }

    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTask(const DrawGaussianToHeatmapTilingData& tiling)
    {
        coreId = GetBlockIdx();
        numClasses = tiling.numClasses;
        coreTaskLen = tiling.coreTaskLen;
        taskObj = tiling.taskObj;
        taskRepeatTimes = tiling.taskRepeatTimes;
        singlePorcessCopyLen = tiling.singlePorcessCopyLen;
        featureMapSizeX = tiling.featureMapSizeX;
        featureMapSizeY = tiling.featureMapSizeY;
        beginId = coreId * coreTaskLen;
        endId = Min((coreId + 1) * coreTaskLen, numClasses);
    }

    __aicore__ inline void InitGM(GM_ADDR mask,
                                  GM_ADDR cur_class_id,
                                  GM_ADDR center_int,
                                  GM_ADDR radius,
                                  GM_ADDR heatmap)
    {
        maskGm.SetGlobalBuffer((__gm__ uint8_t*)(mask));
        curClassIdGm.SetGlobalBuffer((__gm__ int32_t*)(cur_class_id));
        centerIntGm.SetGlobalBuffer((__gm__ int32_t*)(center_int));
        radiusGm.SetGlobalBuffer((__gm__ int32_t*)(radius));
        heatmapGm.SetGlobalBuffer((__gm__ float*)(heatmap));
    }

     __aicore__ inline void InitBuffer()
    {
        pipe_->InitBuffer(maskUB, singlePorcessCopyLen * sizeof(uint8_t));
        pipe_->InitBuffer(centerIntUB, singlePorcessCopyLen * 2 * sizeof(int32_t));
        pipe_->InitBuffer(radiusUB, singlePorcessCopyLen * sizeof(int32_t));
        pipe_->InitBuffer(curIdUB, singlePorcessCopyLen * sizeof(int32_t));
        pipe_->InitBuffer(tmpUB, 8 * sizeof(int32_t));
        pipe_->InitBuffer(heatmapUB, 1024 * sizeof(int32_t));
        pipe_->InitBuffer(xUB, 1024 * sizeof(int32_t));
        pipe_->InitBuffer(yUB, 1024 * sizeof(int32_t));
        pipe_->InitBuffer(gaussian2DUB, 1024 * sizeof(int32_t));
        pipe_->InitBuffer(cmpUB, 1024 * sizeof(int32_t));
    }

    __aicore__ inline void ProcessSingle(int32_t taskIdx)
    {
        LocalTensor<uint8_t> maskLocal = maskUB.Get<uint8_t>();
        LocalTensor<int32_t> centerIntLocal = centerIntUB.Get<int32_t>();
        LocalTensor<int32_t> xLocal = centerIntLocal;
        LocalTensor<int32_t> yLocal = centerIntLocal[singlePorcessCopyLen];
        LocalTensor<int32_t> radiusLocal = radiusUB.Get<int32_t>();
        LocalTensor<int32_t> curIdLocal = curIdUB.Get<int32_t>();

        LocalTensor<float> tmpTensor = tmpUB.Get<float>();
        LocalTensor<float> heatmap = heatmapUB.Get<float>();
        LocalTensor<float> xTensor = xUB.Get<float>();
        LocalTensor<float> yTensor = yUB.Get<float>();
        LocalTensor<float> gaussian2DLocal = gaussian2DUB.Get<float>();
        LocalTensor<uint8_t> cmpLocal = cmpUB.Get<uint8_t>();

        uint32_t heatmapOffset = taskIdx * featureMapSizeY * featureMapSizeX;
        uint32_t copyLen = singlePorcessCopyLen;
        for (uint32_t i = 0; i < taskRepeatTimes; i++) {
            if (i == taskRepeatTimes - 1) {
                copyLen = (taskObj - 1) % singlePorcessCopyLen + 1;
            }
            uint32_t maskcopyLen = AlignUp(copyLen, 32);
            uint32_t floatcopyLen = AlignUp(copyLen, 8);
            DataCopy(maskLocal, maskGm[singlePorcessCopyLen * i], maskcopyLen);
            DataCopy(xLocal, centerIntGm[singlePorcessCopyLen * i], floatcopyLen);
            DataCopy(yLocal, centerIntGm[singlePorcessCopyLen * i + taskObj], floatcopyLen);
            DataCopy(radiusLocal, radiusGm[singlePorcessCopyLen * i], floatcopyLen);
            DataCopy(curIdLocal, curClassIdGm[singlePorcessCopyLen * i], floatcopyLen);
            pipe_barrier(PIPE_ALL);
            for (uint32_t j = 0; j < copyLen; j ++) {
                uint8_t mask = maskLocal.GetValue(j);
                int32_t curid = curIdLocal.GetValue(j);
                int32_t radius = radiusLocal.GetValue(j);
                int32_t x = xLocal.GetValue(j);
                int32_t y = yLocal.GetValue(j);
                if (mask == (uint8_t)0) {
                    continue;
                }
                if (curid - 1 != taskIdx) {
                    continue;
                }
                float sigma = (float)(2 * radius + 1) / 6;
                Duplicate(tmpTensor, (float)radius, 8);
                Mul(tmpTensor, tmpTensor, tmpTensor, 8);
                Muls(tmpTensor, tmpTensor, -2.0f, 8);
                Muls(tmpTensor, tmpTensor, (float)18 / (sigma * sigma), 8);
                Exp(tmpTensor, tmpTensor, 8);
                // np.finfo(np.float32).eps * max
                Muls(tmpTensor, tmpTensor, (float)(2.220446049250313e-16), 8);
                float min_exp =  tmpTensor.GetValue(0);

                int32_t left = Min(x, radius);
                int32_t right = Min((int32_t)featureMapSizeX - x, radius + 1);
                int32_t top = Min(y, radius);
                int32_t bottom = Min((int32_t)featureMapSizeY - y, radius + 1);
                for (int32_t height_id = -top; height_id < bottom; height_id++) {
                    uint32_t yGmOffset = (y + height_id) * featureMapSizeX;
                    uint32_t xGmOffset = (x - left);
                    uint32_t copyHeatmapLen = AlignUp(right + left, 8);
                    DataCopy(heatmap, heatmapGm[heatmapOffset + yGmOffset + xGmOffset], copyHeatmapLen);
                    Duplicate(yTensor, (float)height_id, copyHeatmapLen);
                    ArithProgression<float>(xTensor, (float)(-left), 1.0f, copyHeatmapLen);
                    Mul(xTensor, xTensor, xTensor, copyHeatmapLen);
                    Mul(yTensor, yTensor, yTensor, copyHeatmapLen);
                    Add(xTensor, xTensor, yTensor, copyHeatmapLen);
                    Muls(xTensor, xTensor, -1.0f, copyHeatmapLen);
                    Muls(xTensor, xTensor, (float)1 / (2 * sigma * sigma), copyHeatmapLen);
                    Exp(gaussian2DLocal, xTensor, copyHeatmapLen);
                    CompareScalar(cmpLocal, gaussian2DLocal, min_exp, CMPMODE::GT, AlignUp(copyHeatmapLen, 64));
                    Select(gaussian2DLocal, cmpLocal, gaussian2DLocal, (float)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, copyHeatmapLen);
                    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                    Max(heatmap, heatmap, gaussian2DLocal, copyHeatmapLen);
                    DataCopyExtParams outCopyParams {1, (uint16_t)((right + left) * sizeof(int32_t)), 0, 0, 0};
                    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    DataCopyPad(heatmapGm[heatmapOffset + yGmOffset + xGmOffset], heatmap, outCopyParams);
                    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                }
            }
        }
    }

    __aicore__ inline int32_t Min(int32_t x, int32_t y)
    {
        if (x > y) {
            return y;
        }
        return x;
    }

    __aicore__ inline void InitEvent()
    {
        eventIDMTE2ToV = pipe_->FetchEventID(HardEvent::MTE2_V);
        eventIDVToMTE3 = pipe_->FetchEventID(HardEvent::V_MTE3);
        eventIDMTE3ToMTE2 = pipe_->FetchEventID(HardEvent::MTE3_MTE2);
    }

private:
    TPipe* pipe_;
    TBuf<TPosition::VECCALC> maskUB, centerIntUB, radiusUB, curIdUB;
    TBuf<TPosition::VECCALC> tmpUB, heatmapUB, xUB, yUB, gaussian2DUB, cmpUB;
    GlobalTensor<uint8_t> maskGm;
    GlobalTensor<int32_t> curClassIdGm, centerIntGm, radiusGm;
    GlobalTensor<float> heatmapGm;
    uint32_t coreId, numClasses, coreTaskLen, taskObj, taskRepeatTimes, singlePorcessCopyLen;
    uint32_t featureMapSizeX, featureMapSizeY;
    uint32_t beginId, endId;
    TEventID eventIDMTE2ToV, eventIDVToMTE3, eventIDMTE3ToMTE2;
};

__aicore__ inline void KernelDrawGaussianToHeatmap::Process()
{
    for (int32_t i = beginId; i < endId; i++) {
        ProcessSingle(i);
        pipe_barrier(PIPE_ALL);
    }
}

extern "C" __global__ __aicore__ void draw_gaussian_to_heatmap(GM_ADDR mask, GM_ADDR cur_class_id, GM_ADDR center_int,
                                                               GM_ADDR radius, GM_ADDR heatmap,
                                                               GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    KernelDrawGaussianToHeatmap op(
        mask,
        cur_class_id,
        center_int,
        radius,
        heatmap,
        tiling_data,
        &pipe
    );
    op.Process();
}