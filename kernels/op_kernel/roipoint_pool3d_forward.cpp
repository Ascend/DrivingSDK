/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Encode the geometry-specific features of each 3D proposal.
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
#define CEIL32(num) (((num) + 32 - 1) / 32 * 32)
#define CEIL_BASE(num, base) (((num) + (base) - 1) / (base) * (base))


template <typename U>
__aicore__ inline void gather(LocalTensor<U> dstLocal,
                              LocalTensor<U> srcLocal,
                              LocalTensor<uint32_t>& srcOffsetLocal,
                              uint32_t srcBaseAddr,
                              uint32_t count)
{
    for (uint32_t i = 0; i < count; i++) {
        dstLocal(i) = srcLocal(srcOffsetLocal(i));
    }
}

template <typename T, typename U>
__aicore__ inline void gathermask(LocalTensor<T>& dstLocal,
                                  LocalTensor<T>& src0Local,
                                  LocalTensor<U>& src1Pattern,
                                  uint32_t calCount,
                                  uint64_t& rsvdCnt)
{
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    for (uint32_t i = 0, j = 0; i < calCount; i++) {
        if (src1Pattern(i)) {
            dstLocal(j++) = src0Local(i);
            rsvdCnt++;
        }
    }
}

template <typename T>
class KernelRoipointPool3dForward {
public:
    __aicore__ inline KernelRoipointPool3dForward() {}
    __aicore__ inline void Init(GM_ADDR points,
                                GM_ADDR point_features,
                                GM_ADDR boxes3d,
                                GM_ADDR pooled_features,
                                GM_ADDR pooled_empty_flag,
                                RoipointPool3dForwardTilingData *tiling_data,
                                TPipe &pipe)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->numSampledPoints = tiling_data->numSampledPoints;
        this->batchSize = tiling_data->batchSize;
        this->pointNum = tiling_data->pointNum;
        this->pointNumT = CEIL_BASE(this->pointNum, 32 / sizeof(T));
        this->featureLen = tiling_data->featureLen;
        this->boxesNum = tiling_data->boxesNum;
        uint32_t eachCoreBoxes = tiling_data->eachCoreBoxes;
        uint32_t boxesNumActual = eachCoreBoxes < this->boxesNum ? eachCoreBoxes : this->boxesNum;
        uint32_t ubSize = tiling_data->ubSize;

        this->boxesStart = eachCoreBoxes * GetBlockIdx() % this->boxesNum;
        this->boxesEnd = eachCoreBoxes * (GetBlockIdx() + 1) % this->boxesNum;
        this->batchStart = eachCoreBoxes * GetBlockIdx() / this->boxesNum;
        this->batchEnd = this->batchStart + (this->boxesStart + eachCoreBoxes) / this->boxesNum;
        if (this->boxesEnd == 0) {
            this->boxesEnd = this->boxesNum;
            this->batchEnd -= 1;
        }
        if (GetBlockNum() - GetBlockIdx() == 1) {
            this->boxesEnd = this->boxesNum;
            this->batchEnd = this->batchSize - 1;
        }

        // gm，32字节对齐
        uint64_t pointsOffset = static_cast<uint64_t>(this->batchStart) * this->pointNum * 3;
        uint64_t pointsBufferSize = static_cast<uint64_t>(this->batchEnd - this->batchStart + 1) * this->pointNum * 3;
        pointsGm.SetGlobalBuffer((__gm__ T*)points + pointsOffset, CEIL_BASE(pointsBufferSize, 32 / sizeof(T)));
        uint64_t pointFeaturesOffset = static_cast<uint64_t>(this->batchStart) * this->pointNum * this->featureLen;
        uint64_t pointFeaturesBufferSize = static_cast<uint64_t>(this->batchEnd - this->batchStart + 1) * this->pointNum * this->featureLen;
        pointFeatureGm.SetGlobalBuffer((__gm__ T*)point_features + pointFeaturesOffset,
            CEIL_BASE(pointFeaturesBufferSize, 32 / sizeof(T)));
        uint64_t boxes3dBufferSize = static_cast<uint64_t>(eachCoreBoxes) * 7;
        boxes3dGm.SetGlobalBuffer(
            (__gm__ DTYPE_BOXES3D*)boxes3d + boxes3dBufferSize * GetBlockIdx(),
            CEIL_BASE(boxes3dBufferSize, 32 / sizeof(DTYPE_BOXES3D)));
        uint64_t pooledEachCore = static_cast<uint64_t>(eachCoreBoxes) * this->numSampledPoints * (this->featureLen + 3);
        pooledFeaturesGm.SetGlobalBuffer(
            (__gm__ T*)pooled_features + pooledEachCore * GetBlockIdx(),
            CEIL_BASE(pooledEachCore, 32 / sizeof(T)));
        pooledEmptyFlagGm.SetGlobalBuffer(
            (__gm__ DTYPE_POOLED_EMPTY_FLAG*)pooled_empty_flag + eachCoreBoxes * GetBlockIdx(),
            CEIL_BASE(eachCoreBoxes, 32 / sizeof(DTYPE_POOLED_EMPTY_FLAG)));

        // TQue，32字节对齐
        pipe.InitBuffer(inQueuePoints, 1, this->pointNumT * sizeof(T) * 3);
        pipe.InitBuffer(inQueuePointFeature, 1, this->pointNumT * sizeof(T) * this->featureLen);
        pipe.InitBuffer(inQueueBoxes3d, 1, CEIL32(boxesNumActual * 7 * sizeof(DTYPE_BOXES3D)));
        pipe.InitBuffer(outQueuePooledFeatures, 1,
            CEIL32(boxesNumActual * this->numSampledPoints * (3 + this->featureLen) * sizeof(T)));
        pipe.InitBuffer(outQueuePooledEmptyFlag, 1, CEIL32(boxesNumActual * sizeof(DTYPE_POOLED_EMPTY_FLAG)));

        // TBuf，32字节对齐
        if constexpr (sizeof(T) < sizeof(float)) {
            pipe.InitBuffer(calcPointsFloat, this->pointNumT * 3 * sizeof(float));
        }
        pipe.InitBuffer(calcPointsFlag, this->pointNumT * sizeof(T)); // gathermask与points相同sizeof大小
        pipe.InitBuffer(calcPointsIdx, this->pointNumT * sizeof(uint32_t));
        pipe.InitBuffer(calcUint8Flag, CEIL_BASE(this->pointNum, 256) / 8); // uint8_t
        pipe.InitBuffer(calcZeros, this->pointNumT * sizeof(T));
        pipe.InitBuffer(calcTemp, this->pointNumT * sizeof(float));
        pipe.InitBuffer(calcXYZ, CEIL_BASE(this->pointNum * sizeof(float), 256)); // CompareScalar
        pipe.InitBuffer(calcRz, 32);
        pipe.InitBuffer(calcSin, 32);
        pipe.InitBuffer(calcCos, 32);
        // totle_size(half): N*(3+C)*2+M*7*2+M*[num*(3+C)*2+4]+N*(12+2+4)+N*(2+4+4)=40N+(12*num+18)*M, max=(60,3488,60)
        // totle_size(float): N*(3+C)*4+M*7*4+M*[num*(3+C)*4+4]+N*(4+4)+N*(4+4+4)=44N+(24*num+32)*M, max=(52,2624,52)
    }

    __aicore__ inline void Process()
    {
        uint32_t boxesSum = 0;
        uint32_t boxesLen = 0;
        uint32_t batchCount = this->batchEnd - this->batchStart;
        for (uint32_t batchIdx = 0; batchIdx <= batchCount; batchIdx++) {
            boxesSum += boxesLen;
            boxesLen = this->boxesNum;
            if (batchIdx == 0) {
                boxesLen -= this->boxesStart;
            }
            if (batchIdx == batchCount) {
                boxesLen -= (this->boxesNum - this->boxesEnd);
            }
            Compute(batchIdx, boxesSum, boxesLen);
        }
    }

private:
    __aicore__ inline void CheckPointInBox3d(LocalTensor<float> &pointsFloat, LocalTensor<DTYPE_BOXES3D> box3d)
    {
        // pointsFloat(3, N) box3d(7) pointsFlag(N)
        LocalTensor<T> pointsFlag = calcPointsFlag.Get<T>();
        uint32_t pointNum = this->pointNum;
        uint32_t pointNumT = this->pointNumT;
        LocalTensor<uint8_t> uint8Flag = calcUint8Flag.Get<uint8_t>();
        LocalTensor<T> zeros = calcZeros.Get<T>();
        LocalTensor<float> temp = calcTemp.Get<float>();
        LocalTensor<float> local_x = calcXYZ.Get<float>();
        LocalTensor<float> local_y = calcXYZ.Get<float>();
        LocalTensor<float> local_z = calcXYZ.Get<float>();
        LocalTensor<float> x = pointsFloat[0];
        LocalTensor<float> y = pointsFloat[pointNumT];
        LocalTensor<float> z = pointsFloat[2 * pointNumT];
        float cx = -static_cast<float>(box3d(0));
        float cy = -static_cast<float>(box3d(1));
        float cz = -static_cast<float>(box3d(2));
        float x_size = static_cast<float>(box3d(3)) / static_cast<float>(2);
        float y_size = static_cast<float>(box3d(4)) / static_cast<float>(2);
        float z_size = static_cast<float>(box3d(5));
        LocalTensor<float> rz = calcRz.Get<float>();
        rz(0) = -static_cast<float>(box3d(6));

        // in_flag = (z_size < 0)
        if (z_size < 0) {
            Duplicate<T>(pointsFlag, 0, pointNum);
            return;
        }

        Duplicate<T>(zeros, 0, pointNum);
        Duplicate<T>(pointsFlag, 1, pointNum);

        // local_z = z - cz
        Adds(local_z, z, cz, pointNum);
        // in_flag = (local_z >= 0)
        CompareScalar(uint8Flag, local_z, static_cast<float>(0), CMPMODE::GE, CEIL_BASE(pointNum, 256 / sizeof(float)));
        Select(pointsFlag, uint8Flag, pointsFlag, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, pointNum);

        // in_flag = in_flag && (local_z <= z_size)
        CompareScalar(uint8Flag, local_z, z_size, CMPMODE::LE, CEIL_BASE(pointNum, 256 / sizeof(float)));
        Select(pointsFlag, uint8Flag, pointsFlag, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, pointNum);

        // cosa = torch.cos(-rz)
        // sina = torch.sin(-rz)
        LocalTensor<float> sin = calcSin.Get<float>();
        LocalTensor<float> cos = calcCos.Get<float>();
        Cos(cos[0], rz[0], 1);
        float cosa = cos(0);
        Sin(sin[0], rz[0], 1);
        float sina = sin(0);

        // local_x = (x - cx) * cosa + (y - cy) * (-sina)
        Adds(local_x, x, cx, pointNum);
        Muls(local_x, local_x, cosa, pointNum);
        Adds(temp, y, cy, pointNum);
        Muls(temp, temp, sina, pointNum);
        Sub(local_x, local_x, temp, pointNum);
        // local_x = abs(local_x)
        Abs(local_x, local_x, pointNum);

        // in_flag = in_flag && (local_x < x_size / 2)
        CompareScalar(uint8Flag, local_x, x_size, CMPMODE::LT, CEIL_BASE(pointNum, 256 / sizeof(float)));
        Select(pointsFlag, uint8Flag, pointsFlag, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, pointNum);

        // local_y = (x - cx) * sina + (y - cy) * cosa
        Adds(local_y, x, cx, pointNum);
        Muls(local_y, local_y, sina, pointNum);
        Adds(temp, y, cy, pointNum);
        Muls(temp, temp, cosa, pointNum);
        Add(local_y, local_y, temp, pointNum);
        // local_y = abs(local_y)
        Abs(local_y, local_y, pointNum);
        
        // in_flag = in_flag && (local_y < y_size / 2)
        CompareScalar(uint8Flag, local_y, y_size, CMPMODE::LT, CEIL_BASE(pointNum, 256 / sizeof(float)));
        Select(pointsFlag, uint8Flag, pointsFlag, zeros, SELMODE::VSEL_TENSOR_TENSOR_MODE, pointNum);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void Compute(uint32_t batchIdx, uint32_t boxesSum, uint32_t boxesLen)
    {
        LocalTensor<T> pointsLocal = inQueuePoints.AllocTensor<T>();
        LocalTensor<T> pointFeaturesLocal = inQueuePointFeature.AllocTensor<T>();
        LocalTensor<DTYPE_BOXES3D> boxes3dLocal = inQueueBoxes3d.AllocTensor<DTYPE_BOXES3D>();
        LocalTensor<T> pooledFeaturesLocal = outQueuePooledFeatures.AllocTensor<T>();
        LocalTensor<DTYPE_POOLED_EMPTY_FLAG> pooledEmptyFlagLocal =
            outQueuePooledEmptyFlag.AllocTensor<DTYPE_POOLED_EMPTY_FLAG>();

        DataCopy(pointsLocal, pointsGm[batchIdx * this->pointNum * 3], this->pointNumT);
        DataCopy(pointsLocal[this->pointNumT], pointsGm[batchIdx * this->pointNum * 3 + this->pointNum],
            this->pointNumT);
        DataCopy(pointsLocal[this->pointNumT * 2], pointsGm[batchIdx * this->pointNum * 3 + this->pointNum * 2],
            this->pointNumT);
        LocalTensor<float> pointsFloat;
        if constexpr (sizeof(T) == sizeof(float)) {
            pointsFloat = pointsLocal;
        } else {
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            pointsFloat = calcPointsFloat.Get<float>();
            Cast(pointsFloat, pointsLocal, RoundMode::CAST_NONE, this->pointNumT * 3);
        }
        for (uint32_t i = 0; i < this->featureLen; i++) {
            DataCopy(pointFeaturesLocal[this->pointNumT * i],
                pointFeatureGm[batchIdx * this->pointNum * this->featureLen + this->pointNum * i], this->pointNumT);
        }
        DataCopy(boxes3dLocal, boxes3dGm[boxesSum * 7], CEIL_BASE(boxesLen * 7, 32 / sizeof(DTYPE_BOXES3D)));
        Duplicate<T>(pooledFeaturesLocal, 0,
            CEIL_BASE(boxesLen * this->numSampledPoints * (3 + this->featureLen), 32 / sizeof(T)));
        Duplicate<DTYPE_POOLED_EMPTY_FLAG>(pooledEmptyFlagLocal, 0, pooledEmptyFlagLocal.GetSize());
        pipe_barrier(PIPE_ALL);

        for (int32_t boxesIdx = 0; boxesIdx < boxesLen; boxesIdx++) {
            CheckPointInBox3d(pointsFloat, boxes3dLocal[boxesIdx * 7]);

            uint64_t cnt = 0;
            LocalTensor<int32_t> pointsIdx = calcPointsIdx.Get<int32_t>();
            ArithProgression<int32_t>(pointsIdx, 0, 1, this->pointNum);
            if constexpr (sizeof(T) == sizeof(float)) {
                LocalTensor<T> tPointsFlag = calcPointsFlag.Get<T>();
                LocalTensor<int32_t> iPointsFlag = calcPointsFlag.Get<int32_t>();
                LocalTensor<uint32_t> uPointsFlag = iPointsFlag.ReinterpretCast<uint32_t>();
                Cast(iPointsFlag, tPointsFlag, RoundMode::CAST_RINT, this->pointNum);
                gathermask(pointsIdx, pointsIdx, uPointsFlag, this->pointNum, cnt);
            } else {
                LocalTensor<T> tPointsFlag = calcPointsFlag.Get<T>();
                LocalTensor<int16_t> iPointsFlag = calcPointsFlag.Get<int16_t>();
                LocalTensor<uint16_t> uPointsFlag = iPointsFlag.ReinterpretCast<uint16_t>();
                Cast(iPointsFlag, tPointsFlag, RoundMode::CAST_RINT, this->pointNum);
                gathermask(pointsIdx, pointsIdx, uPointsFlag, this->pointNum, cnt);
            }
            pipe_barrier(PIPE_V);

            if (cnt == 0) {
                pooledEmptyFlagLocal(boxesIdx) = 1;
                continue;
            }
            if (cnt < this->numSampledPoints) {
                if (cnt == 1) {
                    int32_t pointsIdxTemp = pointsIdx(0);
                    Duplicate<int32_t>(pointsIdx, pointsIdxTemp, this->numSampledPoints);
                    pipe_barrier(PIPE_V);
                } else {
                    for (uint32_t i = cnt; i < this->numSampledPoints; i++) {
                        pointsIdx(i) = pointsIdx(i % cnt);
                    }
                }
            }

            LocalTensor<uint32_t> uPointsIdx = pointsIdx.ReinterpretCast<uint32_t>();
            uint32_t baseOffset = boxesIdx * this->numSampledPoints * (3 + this->featureLen);
            gather(pooledFeaturesLocal[baseOffset], pointsLocal[0], uPointsIdx, 0, this->numSampledPoints);
            gather(pooledFeaturesLocal[baseOffset + this->numSampledPoints],
                   pointsLocal[this->pointNumT],
                   uPointsIdx, 0, this->numSampledPoints);
            gather(pooledFeaturesLocal[baseOffset + this->numSampledPoints * 2],
                   pointsLocal[this->pointNumT * 2],
                   uPointsIdx, 0, this->numSampledPoints);
            for (int32_t idx = 0; idx < this->featureLen; idx++) {
                gather(pooledFeaturesLocal[baseOffset + this->numSampledPoints * (3 + idx)],
                       pointFeaturesLocal[this->pointNumT * idx],
                       uPointsIdx, 0, this->numSampledPoints);
            }
            pipe_barrier(PIPE_ALL);
        }

        DataCopyExtParams dataCopyParams = {1, 0, 0, 0, 0};
        dataCopyParams.blockLen = sizeof(T) * boxesLen * this->numSampledPoints * (3 + this->featureLen);
        DataCopyPad(pooledFeaturesGm[boxesSum * this->numSampledPoints * (3 + this->featureLen)], pooledFeaturesLocal,
            dataCopyParams);
        dataCopyParams.blockLen = sizeof(DTYPE_POOLED_EMPTY_FLAG) * boxesLen;
        DataCopyPad(pooledEmptyFlagGm[boxesSum], pooledEmptyFlagLocal, dataCopyParams);
        pipe_barrier(PIPE_MTE3);

        inQueuePoints.FreeTensor(pointsLocal);
        inQueuePointFeature.FreeTensor(pointFeaturesLocal);
        inQueueBoxes3d.FreeTensor(boxes3dLocal);
        outQueuePooledFeatures.FreeTensor(pooledFeaturesLocal);
        outQueuePooledEmptyFlag.FreeTensor(pooledEmptyFlagLocal);
    }

private:
    TQue<QuePosition::VECIN, 1> inQueuePoints, inQueuePointFeature, inQueueBoxes3d;
    TQue<QuePosition::VECOUT, 1> outQueuePooledFeatures, outQueuePooledEmptyFlag;
    TBuf<TPosition::VECCALC> calcPointsFloat;
    TBuf<TPosition::VECCALC> calcPointsFlag, calcPointsIdx;
    TBuf<TPosition::VECCALC> calcUint8Flag, calcZeros, calcTemp, calcXYZ, calcRz, calcSin, calcCos;
    GlobalTensor<T> pointsGm, pointFeatureGm;
    GlobalTensor<DTYPE_BOXES3D> boxes3dGm;
    GlobalTensor<T> pooledFeaturesGm;
    GlobalTensor<DTYPE_POOLED_EMPTY_FLAG> pooledEmptyFlagGm;
    uint32_t numSampledPoints, batchSize, pointNum, featureLen, boxesNum;
    uint32_t pointNumT, boxesStart, boxesEnd, batchStart, batchEnd;
};

extern "C" __global__ __aicore__
void roipoint_pool3d_forward(GM_ADDR points,
                             GM_ADDR point_features,
                             GM_ADDR boxes3d,
                             GM_ADDR pooled_features,
                             GM_ADDR pooled_empty_flag,
                             GM_ADDR workspace,
                             GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) { // TILING_KEY_FLOAT
        KernelRoipointPool3dForward<float> op;
        op.Init(points, point_features, boxes3d, pooled_features, pooled_empty_flag, &tiling_data, pipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) { // TILING_KEY_HALF
        KernelRoipointPool3dForward<half> op;
        op.Init(points, point_features, boxes3d, pooled_features, pooled_empty_flag, &tiling_data, pipe);
        op.Process();
    }
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void roipoint_pool3d_forward_do(uint32_t blockDim,
                                void* l2ctrl,
                                void* stream,
                                uint8_t* points,
                                uint8_t* point_features,
                                uint8_t* boxes3d,
                                uint8_t* pooled_features,
                                uint8_t* pooled_empty_flag,
                                uint8_t* workspace,
                                uint8_t* tiling)
{
    roipoint_pool3d_forward<<<blockDim, l2ctrl, stream>>>(
        points, point_features, boxes3d, pooled_features, pooled_empty_flag, workspace, tiling);
}
#endif