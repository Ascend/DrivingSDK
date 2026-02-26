/*
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
*/
#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t MAX_FEATURE_SIZE = 1024;
constexpr uint32_t COORD_DIM_NUM = 2;
constexpr uint32_t TMP_BUFFER_SIZE = 8;
constexpr uint32_t MASK_ALIGN_SIZE = 32;
constexpr uint32_t FLOAT_ALIGN_SIZE = 8;
constexpr int32_t GAUSSIAN_SIGMA_DIVISOR = 6;
constexpr float GAUSSIAN_EXP_FACTOR = -2.0f;
constexpr int32_t GAUSSIAN_EXP_COEFFICIENT = 18;
constexpr float MIN_EPSILON = 2.220446049250313e-16f;
constexpr uint32_t CMP_ALIGN_SIZE = 64;


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
        singleProcessCopyLen = tiling.singleProcessCopyLen;
        featureMapSizeX = tiling.featureMapSizeX;
        featureMapSizeY = tiling.featureMapSizeY;
        beginId = coreId * coreTaskLen;
        endId = min((coreId + 1) * coreTaskLen, numClasses);
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
        pipe_->InitBuffer(maskUB, singleProcessCopyLen * sizeof(uint8_t));
        pipe_->InitBuffer(centerIntUB, singleProcessCopyLen * COORD_DIM_NUM * sizeof(int32_t));
        pipe_->InitBuffer(radiusUB, singleProcessCopyLen * sizeof(int32_t));
        pipe_->InitBuffer(curIdUB, singleProcessCopyLen * sizeof(int32_t));
        pipe_->InitBuffer(tmpUB, TMP_BUFFER_SIZE * sizeof(int32_t));
        pipe_->InitBuffer(heatmapUB, MAX_FEATURE_SIZE * sizeof(int32_t));
        pipe_->InitBuffer(xUB, MAX_FEATURE_SIZE * sizeof(int32_t));
        pipe_->InitBuffer(yUB, MAX_FEATURE_SIZE * sizeof(int32_t));
        pipe_->InitBuffer(gaussian2DUB, MAX_FEATURE_SIZE * sizeof(int32_t));
        pipe_->InitBuffer(cmpUB, MAX_FEATURE_SIZE * sizeof(int32_t));
    }

    __aicore__ inline void ProcessSingle(int32_t taskIdx)
    {
        LocalTensor<uint8_t> maskLocal = maskUB.Get<uint8_t>();
        LocalTensor<int32_t> centerIntLocal = centerIntUB.Get<int32_t>();
        LocalTensor<int32_t> xLocal = centerIntLocal;
        LocalTensor<int32_t> yLocal = centerIntLocal[singleProcessCopyLen];
        LocalTensor<int32_t> radiusLocal = radiusUB.Get<int32_t>();
        LocalTensor<int32_t> curIdLocal = curIdUB.Get<int32_t>();

        LocalTensor<float> tmpTensor = tmpUB.Get<float>();
        LocalTensor<float> heatmap = heatmapUB.Get<float>();
        LocalTensor<float> xTensor = xUB.Get<float>();
        LocalTensor<float> yTensor = yUB.Get<float>();
        LocalTensor<float> gaussian2DLocal = gaussian2DUB.Get<float>();
        LocalTensor<uint8_t> cmpLocal = cmpUB.Get<uint8_t>();

        uint32_t heatmapOffset = taskIdx * featureMapSizeY * featureMapSizeX;
        uint32_t copyLen = singleProcessCopyLen;
        for (uint32_t i = 0; i < taskRepeatTimes; i++) {
            if (i == taskRepeatTimes - 1) {
                copyLen = (taskObj - 1) % singleProcessCopyLen + 1;
            }
            uint32_t maskcopyLen = AlignUp(copyLen, MASK_ALIGN_SIZE);
            uint32_t floatcopyLen = AlignUp(copyLen, FLOAT_ALIGN_SIZE);
            DataCopy(maskLocal, maskGm[singleProcessCopyLen * i], maskcopyLen);
            DataCopy(xLocal, centerIntGm[singleProcessCopyLen * i], floatcopyLen);
            DataCopy(yLocal, centerIntGm[singleProcessCopyLen * i + taskObj], floatcopyLen);
            DataCopy(radiusLocal, radiusGm[singleProcessCopyLen * i], floatcopyLen);
            DataCopy(curIdLocal, curClassIdGm[singleProcessCopyLen * i], floatcopyLen);
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
                float sigma = (float)(2 * radius + 1) / GAUSSIAN_SIGMA_DIVISOR;
                Duplicate(tmpTensor, (float)radius, TMP_BUFFER_SIZE);
                Mul(tmpTensor, tmpTensor, tmpTensor, TMP_BUFFER_SIZE);
                Muls(tmpTensor, tmpTensor, GAUSSIAN_EXP_FACTOR, TMP_BUFFER_SIZE);
                Muls(tmpTensor, tmpTensor, (float)GAUSSIAN_EXP_COEFFICIENT / (sigma * sigma), TMP_BUFFER_SIZE);
                Exp(tmpTensor, tmpTensor, TMP_BUFFER_SIZE);
                // np.finfo(np.float32).eps * max
                Muls(tmpTensor, tmpTensor, MIN_EPSILON, TMP_BUFFER_SIZE);
                float min_exp =  tmpTensor.GetValue(0);

                int32_t left = min(x, radius);
                int32_t right = min((int32_t)featureMapSizeX - x, radius + 1);
                int32_t top = min(y, radius);
                int32_t bottom = min((int32_t)featureMapSizeY - y, radius + 1);
                for (int32_t height_id = -top; height_id < bottom; height_id++) {
                    uint32_t yGmOffset = (y + height_id) * featureMapSizeX;
                    uint32_t xGmOffset = (x - left);
                    uint32_t copyHeatmapLen = AlignUp(right + left, FLOAT_ALIGN_SIZE);
                    DataCopy(heatmap, heatmapGm[heatmapOffset + yGmOffset + xGmOffset], copyHeatmapLen);
                    Duplicate(yTensor, (float)height_id, copyHeatmapLen);
                    ArithProgression<float>(xTensor, (float)(-left), 1.0f, copyHeatmapLen);
                    Mul(xTensor, xTensor, xTensor, copyHeatmapLen);
                    Mul(yTensor, yTensor, yTensor, copyHeatmapLen);
                    Add(xTensor, xTensor, yTensor, copyHeatmapLen);
                    Muls(xTensor, xTensor, -1.0f, copyHeatmapLen);
                    Muls(xTensor, xTensor, (float)1 / (2 * sigma * sigma), copyHeatmapLen);
                    Exp(gaussian2DLocal, xTensor, copyHeatmapLen);
                    CompareScalar(cmpLocal, gaussian2DLocal, min_exp, CMPMODE::GT, AlignUp(copyHeatmapLen, CMP_ALIGN_SIZE));
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
    uint32_t coreId, numClasses, coreTaskLen, taskObj, taskRepeatTimes, singleProcessCopyLen;
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
