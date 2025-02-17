/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t BUFFER_NUM_INPUT = 1;
constexpr uint32_t BLOCK = 256;
template<typename T>
class KernelNms3dOnSight {
public:
    __aicore__ inline KernelNms3dOnSight() {}

    __aicore__ inline void Init(GM_ADDR boxes, GM_ADDR mask, const Nms3dOnSightTilingData* __restrict tiling_data)
    {
        // 所有的计算在kernel侧以alignedN进行，这样在搬入搬出时都能保证对齐
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        usedCoreNum = tiling_data->usedCoreNum;
        boxNum = tiling_data->boxNum;
        alignedN = tiling_data->alignedN;
        loopTime = tiling_data->loopTime;
        threshold = tiling_data->threshold;
        assignBox = tiling_data->assignBox;

        uint32_t core_id = GetBlockIdx();

        boxGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(boxes), static_cast<uint64_t>(alignedN) * 7);
        maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t*>(mask), static_cast<uint64_t>(alignedN) * boxNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM_INPUT, assignBox * sizeof(T));
        pipe.InitBuffer(inQueueY, BUFFER_NUM_INPUT, assignBox * sizeof(T));
        pipe.InitBuffer(inQueueR, BUFFER_NUM_INPUT, assignBox * sizeof(T));
        pipe.InitBuffer(outQueueMask, BUFFER_NUM, assignBox * sizeof(int16_t));

        pipe.InitBuffer(xBuf, assignBox * sizeof(T));
        pipe.InitBuffer(yBuf, assignBox * sizeof(T));
        pipe.InitBuffer(rBuf, assignBox * sizeof(T));

        pipe.InitBuffer(distBuf, assignBox * sizeof(T));
        pipe.InitBuffer(thresholdBuf, assignBox * sizeof(T));
        pipe.InitBuffer(maskBuf, alignedN * sizeof(uint8_t));
        
        pipe.InitBuffer(upBuf, alignedN * sizeof(T));
        pipe.InitBuffer(downBuf, alignedN * sizeof(T));
        pipe.InitBuffer(distBuf1, alignedN * sizeof(T));
        pipe.InitBuffer(distBuf2, alignedN * sizeof(T));
        pipe.InitBuffer(fovBuf, alignedN * sizeof(T));
        pipe.InitBuffer(selBuf, alignedN * sizeof(T));
        
        // 计算过程中用到的缓存tensor
        pipe.InitBuffer(comBuf, alignedN * sizeof(T));
        pipe.InitBuffer(comXBuf, alignedN * sizeof(T));
        pipe.InitBuffer(comYBuf, alignedN * sizeof(T));
        pipe.InitBuffer(comRBuf, alignedN * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        uint32_t core_id = GetBlockIdx();
        uint32_t coreStart = core_id * loopTime; // 每个核处理LoopTime个boxes
        if (coreStart >= boxNum) {
            return ; // 空核
        }

        CopyIn(); // 把对应的boxes的xyr搬到对应的tensor中
        
        for (uint32_t index = coreStart; index < coreStart + loopTime; index++) {
            if (index >= boxNum) {
                return ; // 达到任务数，或者直接跳出
            }
            Compute(index); // 计算boxes[i] 和所有boxes的dist，跟thresholdold比较并返回mask矩阵
            CopyOut(index); // 搬出boxes[i]对应的mask矩阵；[1, N]的大小
        }
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) 
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        LocalTensor<T> rLocal = inQueueR.AllocTensor<T>();

        DataCopy(xLocal, boxGm[static_cast<uint64_t>(0)], alignedN); // 这里搬的大小需要N向上对齐32B
        DataCopy(yLocal, boxGm[static_cast<uint64_t>(1) * boxNum], alignedN); // 这里的N是boxNum
        DataCopy(rLocal, boxGm[static_cast<uint64_t>(6) * boxNum], alignedN); // 这里的N是boxNum

        inX = xLocal;
        inY = yLocal;
        inR = rLocal;
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Compute(uint32_t curBox)
    {
        LocalTensor<T> xTemp = inX;
        LocalTensor<T> yTemp = inY;
        LocalTensor<T> rTemp = inR;
        
        // 声明输出mask,先赋值成全1
        LocalTensor<int16_t> outLocal = outQueueMask.AllocTensor<int16_t>();
        LocalTensor<half> selLocal = selBuf.Get<half>();
        Duplicate(selLocal, static_cast<half>(1), alignedN); 
        LocalTensor<uint8_t> maskdstLocal = maskBuf.Get<uint8_t>();
        
        // 将curBox需要的计算量curX, curY, curR得到; [1, alignedN]
        LocalTensor<T> curX = xBuf.Get<T>();
        LocalTensor<T> curY = yBuf.Get<T>();
        LocalTensor<T> curR = rBuf.Get<T>();

        T xCurBox = xTemp.GetValue(curBox);
        T yCurBox = yTemp.GetValue(curBox);
        T rCurBox = rTemp.GetValue(curBox);

        Duplicate(curX, xCurBox, alignedN);
        Duplicate(curY, yCurBox, alignedN);
        Duplicate(curR, rCurBox, alignedN);

        // 计算curBox(curX, curY, curR) 和所有boxes（xTemp，yTemp和rTemp）的dist_bev, 需要和threshold进行比较得到结果，并赋值给mask (outLocal)
        // 声明一个distLocal的tensor[1, N]和threshold的tensor[1, N], 数据类型为T，计算得到的值和threshold（float）进行比较
        LocalTensor<T> distLocal = distBuf.Get<T>();
        LocalTensor<T> thresholdLocal = thresholdBuf.Get<T>();
        Duplicate(thresholdLocal, static_cast<T>(threshold), alignedN);

        DistBev(curX, curY, curR, xTemp, yTemp, rTemp, distLocal);
        
        // distLocal和thresholdLocal进行compare，结果放在outLocal上
        CMPMODE compareMode = CMPMODE::LE;
        Compare(maskdstLocal, distLocal, thresholdLocal, compareMode, assignBox); // API要求必须对齐256B
        Select(selLocal, maskdstLocal, selLocal, static_cast<half>(0), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedN);
        selLocal.SetValue(curBox, static_cast<half>(1));
        Cast(outLocal, selLocal, RoundMode::CAST_ROUND, alignedN);

        outQueueMask.EnQue<int16_t>(outLocal);

        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyOut(int32_t curBox)
    {
        // 搬出curBox对应的alignedN个数据
        LocalTensor<int16_t> outLocal = outQueueMask.DeQue<int16_t>();
        DataCopy(maskGm[static_cast<uint64_t>(curBox) * alignedN], outLocal, alignedN);
        outQueueMask.FreeTensor(outLocal);
    }

private:
    __aicore__ inline void InFront120FOVTensor(const LocalTensor<T>& xTemp, const LocalTensor<T>& yTemp, const LocalTensor<uint8_t>& maskdstLocal)
    {
        LocalTensor<T> yFov = fovBuf.Get<T>();
        T tan30 = static_cast<T>(1.73205 / 3.0);

        Abs(yFov, yTemp, alignedN);
        Muls(yFov, yFov, tan30, alignedN);

        Compare(maskdstLocal, xTemp, yFov, CMPMODE::LE, assignBox);
    }

    __aicore__ inline bool InFront120FOV(float x, float y)
    {
        float tan30 = static_cast<float>(1.73205 / 3.0);
        y = abs(y);
        y = y * tan30;
        return x > y;
    }
    
    __aicore__ inline void DistBev(const LocalTensor<T>& curX, const LocalTensor<T>& curY, const LocalTensor<T>& curR, const LocalTensor<T>& xTemp,
                                    const LocalTensor<T>& yTemp, const LocalTensor<T>& rTemp, const LocalTensor<T>& distLocal)
    {
        // 计算cur_box和other_box之间的distLocal，cur_box的shape：[1, alignedN], other_box的shape：[1, alignedN]
        // 5.0 meter
        const float maxMergeDist = 5.0;
        // 30° degree
        const float maxRyDiff = 0.523598;
        // flag value, means 10m ** 2
        const float veryFar = -100.0;

        if (InFront120FOV(curX(0), curY(0))) {
            Duplicate(distLocal, static_cast<T>(veryFar), alignedN);
            return ;
        }

        // 临时变量comTempX, Y, R
        LocalTensor<T> comTempX = comXBuf.Get<T>();
        LocalTensor<T> comTempY = comYBuf.Get<T>();
        LocalTensor<T> comTempR = comRBuf.Get<T>();
        LocalTensor<T> distTemp = comBuf.Get<T>();
        LocalTensor<uint8_t> maskdstLocal = maskBuf.Get<uint8_t>(); // 表示每次的compare的输出和select的mask矩阵
        LocalTensor<T> upTensor = upBuf.Get<T>();
        LocalTensor<T> downTensor = downBuf.Get<T>();
        LocalTensor<T> distA = distBuf1.Get<T>();
        LocalTensor<T> distB = distBuf2.Get<T>();
        T srcScalar = static_cast<T>(0);
        
        Duplicate(distTemp, srcScalar, alignedN);

        comTempX = curX * xTemp;
        comTempY = curY * yTemp;
        comTempX = comTempX + comTempY;

        CMPMODE compareMode = CMPMODE::GT;
        CompareScalar(maskdstLocal, comTempX, srcScalar, compareMode, assignBox); // 该接口只支持256B对齐

        // 使用select进行对dist矩阵的赋值
        Select(distLocal, maskdstLocal, distTemp, static_cast<T>(veryFar), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedN); // mask中，0选veryFar,1选Tensor
        DataCopy(distTemp, distLocal, alignedN);

        // 第二步计算--> IN_FRONT_120FOV(box_b[0], box_b[1])
        InFront120FOVTensor(xTemp, yTemp, maskdstLocal);
        
        Select(distLocal, maskdstLocal, distTemp, static_cast<T>(veryFar), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedN);
        DataCopy(distTemp, distLocal, alignedN);

        // 第三步计算 --> 边界框距离超过maxMergeDist
        comTempX = (curX - xTemp);
        comTempX = comTempX * comTempX;

        comTempY = (curY - yTemp);
        comTempY = comTempY * comTempY;

        comTempX = comTempX + comTempY;

        srcScalar = static_cast<T>(maxMergeDist * maxMergeDist);
        CompareScalar(maskdstLocal, comTempX, srcScalar, CMPMODE::LT, assignBox);
        Select(distLocal, maskdstLocal, distTemp, static_cast<T>(veryFar), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedN);
        DataCopy(distTemp, distLocal, alignedN);

        // 第四步计算：角度差值
        comTempR = curR - rTemp;
        Abs(comTempR, comTempR, alignedN);

        srcScalar = static_cast<T>(maxRyDiff);
        CompareScalar(maskdstLocal, comTempR, srcScalar, CMPMODE::LT, assignBox);
        Select(distLocal, maskdstLocal, distTemp, static_cast<T>(veryFar), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedN);
        DataCopy(distTemp, distLocal, alignedN);

        // 第五步计算：投影距离
        comTempX = curX * yTemp;
        comTempY = curY * xTemp;
        upTensor = comTempX - comTempY;
        upTensor = upTensor * upTensor;
        Muls(upTensor, upTensor, static_cast<T>(-1), alignedN);

        comTempX = curX * curX;
        comTempY = curY * curY;
        distA = comTempX + comTempY;

        comTempX = xTemp * xTemp;
        comTempY = yTemp * yTemp;
        distB = comTempX + comTempY;
        
        Max(downTensor,  distA, distB, alignedN);
        Adds(downTensor, downTensor, static_cast<T>(0.0001), alignedN);

        comTempX = upTensor / downTensor;
        srcScalar = static_cast<T>(0);
        CompareScalar(maskdstLocal, distTemp, srcScalar, CMPMODE::GE, assignBox);
        Select(distLocal, maskdstLocal, comTempX, static_cast<T>(veryFar), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedN);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM_INPUT> inQueueX, inQueueY, inQueueR;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueMask;
    TBuf<TPosition::VECCALC> xBuf, yBuf, rBuf;
    TBuf<TPosition::VECCALC> distBuf, thresholdBuf;

    TBuf<TPosition::VECCALC> comBuf, comXBuf, comYBuf, comRBuf;
    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECCALC> upBuf, downBuf, distBuf1, distBuf2, fovBuf, selBuf;

    GlobalTensor<T> boxGm;
    GlobalTensor<int16_t> maskGm;
    LocalTensor<T> xTemp, yTemp, rTemp;
    LocalTensor<T> inX, inY, inR;
    uint32_t usedCoreNum;
    uint32_t loopTime;
    uint32_t eachSum;
    uint32_t boxNum;
    uint32_t tailSum;
    uint32_t tailNum;
    uint32_t alignedN;
    uint32_t assignBox;
    float threshold;
    bool isLastCore;
};

extern "C" __global__ __aicore__ void nms3d_on_sight(GM_ADDR boxes, GM_ADDR mask, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    const Nms3dOnSightTilingData* __restrict tilingDevice = &tilingData;
    if (TILING_KEY_IS(1)) {
        KernelNms3dOnSight<float> op;
        op.Init(boxes, mask, tilingDevice);
        op.Process();
    }
}