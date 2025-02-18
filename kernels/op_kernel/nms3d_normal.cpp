/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr float EPS = 1e-8;

template <typename T>
class KernelNms3dNormal {
public:
    __aicore__ inline KernelNms3dNormal() {}
    __aicore__ inline void Init(GM_ADDR boxes, GM_ADDR mask, const Nms3dNormalTilingData* __restrict tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        usedCoreNum = tiling_data->usedCoreNum;
        eachSum = tiling_data->eachSum;
        boxNum = tiling_data->boxNum;
        tailSum = tiling_data->tailSum;
        tailNum = tiling_data->tailNum;
        maskNum = tiling_data->maskNum;
        loopTime = tiling_data->loopTime;
        overlapThresh = tiling_data->overlapThresh;

        uint32_t core_id = GetBlockIdx();
        isLastCore = (core_id == (tiling_data->usedCoreNum - 1));

        boxGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(boxes), static_cast<uint64_t>(boxNum) * 7);
        maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t*>(mask), static_cast<uint64_t>(maskNum) * boxNum);
        
        pipe.InitBuffer(inQueueCur, BUFFER_NUM, dataAlign * sizeof(T));
        pipe.InitBuffer(inQueueBox, BUFFER_NUM, dataAlign * 7 * sizeof(T));
        pipe.InitBuffer(outQueueMask, BUFFER_NUM, dataAlign * sizeof(int16_t));
        pipe.InitBuffer(oneMask, BUFFER_NUM, dataAlign * sizeof(int16_t));
        if constexpr (sizeof(T) == sizeof(half)) {
            pipe.InitBuffer(calcBuf, dataAlign * 2 * 7 * sizeof(float));
            curTemp = calcBuf.Get<float>(dataAlign * 2 * 7);
            boxTemp = curTemp[8];
        }
    }
    __aicore__ inline void Process()
    {
        uint32_t core_id = GetBlockIdx();
        LocalTensor<int16_t> oneLocal = oneMask.AllocTensor<int16_t>();
        Duplicate(oneLocal, static_cast<int16_t>(1), dataAlign);
        for (size_t i = 0; i < boxNum; ++i) {
            for (size_t j = 0; j < loopTime; ++j) {
                uint32_t start = core_id * eachSum + dataAlign * j;
                if (i >= start + dataAlign) {
                    DataCopy(maskGm[i * maskNum + start], oneLocal, dataAlign);
                    continue;
                }
                CopyIn(i, start);
                Compute(i, start);
                CopyOut(i, start);
            }
        }
        oneMask.FreeTensor(oneLocal);
    }

private:
    __aicore__ inline void CopyIn(int32_t cur_box, int32_t com_box)
    {
        LocalTensor<T> curLocal = inQueueCur.AllocTensor<T>();
        LocalTensor<T> boxLocal = inQueueBox.AllocTensor<T>();
        DataCopy(curLocal, boxGm[static_cast<uint64_t>(cur_box) * 7], dataAlign);
        DataCopy(boxLocal, boxGm[static_cast<uint64_t>(com_box) * 7], dataAlign * 7);
        inQueueCur.EnQue(curLocal);
        inQueueBox.EnQue(boxLocal);
    }
    __aicore__ inline void Compute(int32_t cur_box, int32_t com_box)
    {
        uint32_t cmpNum = dataAlign;
        if constexpr (sizeof(T) == sizeof(half)) {
            LocalTensor<T> curLocal = inQueueCur.DeQue<T>();
            LocalTensor<T> boxLocal = inQueueBox.DeQue<T>();
            Cast(curTemp, curLocal, RoundMode::CAST_NONE, dataAlign);
            Cast(boxTemp, boxLocal, RoundMode::CAST_NONE, 7 * dataAlign);
            inQueueCur.FreeTensor(curLocal);
            inQueueBox.FreeTensor(boxLocal);
        } else {
            curTemp = inQueueCur.DeQue<T>();
            boxTemp = inQueueBox.DeQue<T>();
        }
        PipeBarrier<PIPE_ALL>();
        LocalTensor<int16_t> outLocal = outQueueMask.AllocTensor<int16_t>();
        float Sa = curTemp.GetValue(3) * curTemp.GetValue(4);
        for (size_t i = 0; i < cmpNum; i++) {
            if (cur_box >= com_box + i) {
                outLocal.SetValue(i, 1);
                continue;
            }
            float left = max(curTemp.GetValue(0) - curTemp.GetValue(3) / 2.0f, boxTemp.GetValue(i * 7) - boxTemp.GetValue(i * 7 + 3) / 2.0f);
            float right = min(curTemp.GetValue(0) + curTemp.GetValue(3) / 2.0f, boxTemp.GetValue(i * 7) + boxTemp.GetValue(i * 7 + 3) / 2.0f);
            float top = max(curTemp.GetValue(1) - curTemp.GetValue(4) / 2.0f, boxTemp.GetValue(i * 7 + 1) - boxTemp.GetValue(i * 7 + 4) / 2.0f);
            float bottom = min(curTemp.GetValue(1) + curTemp.GetValue(4) / 2.0f, boxTemp.GetValue(i * 7 + 1) + boxTemp.GetValue(i * 7 + 4) / 2.0f);
            float width = max(right - left, 0.f);
            float height = max(bottom - top, 0.f);
            float interS = width * height;
            float Sb = boxTemp.GetValue(i * 7 + 3) * boxTemp.GetValue(i * 7 + 4);
            if (interS / max(Sa + Sb - interS, EPS) >= overlapThresh) {
                outLocal.SetValue(i, 0);
            } else {
                outLocal.SetValue(i, 1);
            }
        }
        PipeBarrier<PIPE_ALL>();
        outQueueMask.EnQue<int16_t>(outLocal);
        if constexpr (sizeof(T) != sizeof(half)) {
            inQueueCur.FreeTensor(curTemp);
            inQueueBox.FreeTensor(boxTemp);
        }
    }
    __aicore__ inline void CopyOut(int32_t cur_box, int32_t com_box)
    {
        LocalTensor<int16_t> outLocal = outQueueMask.DeQue<int16_t>();
        DataCopy(maskGm[static_cast<uint64_t>(cur_box) * maskNum + static_cast<uint64_t>(com_box)], outLocal, dataAlign);
        outQueueMask.FreeTensor(outLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueCur, inQueueBox;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueMask, oneMask;
    TBuf<QuePosition::VECCALC> calcBuf;
    GlobalTensor<T> boxGm;
    GlobalTensor<int16_t> maskGm;
    LocalTensor<float> curTemp, boxTemp;
    uint32_t usedCoreNum;
    uint32_t loopTime;
    uint32_t eachSum;
    uint32_t boxNum;
    uint32_t tailSum;
    uint32_t tailNum;
    uint32_t maskNum;
    uint32_t dataAlign = 16;
    float overlapThresh;
    bool isLastCore;
};

extern "C" __global__ __aicore__ void nms3d_normal(GM_ADDR boxes, GM_ADDR mask, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    const Nms3dNormalTilingData* __restrict tilingDevice = &tilingData;
    if (TILING_KEY_IS(1)) {
        KernelNms3dNormal<float> op;
        op.Init(boxes, mask, tilingDevice);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelNms3dNormal<half> op;
        op.Init(boxes, mask, tilingDevice);
        op.Process();
    }
}
