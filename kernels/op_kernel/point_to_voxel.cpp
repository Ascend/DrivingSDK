// Copyright (c) 2024 Huawei Technologies Co., Ltd


#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t ONE_REPEAT_B64_SIZE = 32;
constexpr uint64_t SELECT_MASK = 64;
constexpr int32_t ENC_BITS = 11;
constexpr int32_t ENC_BITS_Z = 8;

template<typename T>
class PointToVoxelKernel {
public:
    __aicore__ inline PointToVoxelKernel() = delete;
    __aicore__ inline ~PointToVoxelKernel() = default;
    __aicore__ inline PointToVoxelKernel(GM_ADDR points, GM_ADDR voxels, const PointToVoxelTilingData& tiling)
        : blkIdx_(GetBlockIdx())
    {
        GetTiling(tiling);
        // init task
        curTaskIdx_ = blkIdx_ < tailTasks_ ? blkIdx_ * (avgTasks_ + 1) : blkIdx_ * avgTasks_ + tailTasks_;
        coreTasks_ = blkIdx_ < tailTasks_ ? avgTasks_ + 1 : avgTasks_;
        curPtsIdx_ = curTaskIdx_ * avgPts_;

        coorXOffset_ = 0;
        coorYOffset_ = avgPts_;
        coorZOffset_ = avgPts_ * 2;
        avgCpParam_.blockLen = avgPts_ * sizeof(T) / ONE_BLK_SIZE;
        tailCpParam_.blockLen = Ceil(tailPts_ * sizeof(T), ONE_BLK_SIZE);
        voxelScaleX_ = 1.0f / voxelSizeX_;
        voxelScaleY_ = 1.0f / voxelSizeY_;
        voxelScaleZ_ = 1.0f / voxelSizeZ_;

        rptTimes_ = avgPts_ / ONE_REPEAT_FLOAT_SIZE;
        maskRptTimes_ = Ceil(rptTimes_, ONE_REPEAT_B64_SIZE);

        ptsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(points));
        voxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(voxels));

        pipe_.InitBuffer(ptsQue_, BUFFER_NUM, avgPts_ * sizeof(T) * 3);
        pipe_.InitBuffer(voxQue_, BUFFER_NUM, avgPts_ * sizeof(float));
        pipe_.InitBuffer(maskXBuf_, maskRptTimes_ * ONE_REPEAT_BYTE_SIZE);
        pipe_.InitBuffer(maskYBuf_, maskRptTimes_ * ONE_REPEAT_BYTE_SIZE);
        pipe_.InitBuffer(maskZBuf_, maskRptTimes_ * ONE_REPEAT_BYTE_SIZE);
        pipe_.InitBuffer(maskX1Buf_, maskRptTimes_ * ONE_REPEAT_BYTE_SIZE);
        pipe_.InitBuffer(maskY1Buf_, maskRptTimes_ * ONE_REPEAT_BYTE_SIZE);
        pipe_.InitBuffer(maskZ1Buf_, maskRptTimes_ * ONE_REPEAT_BYTE_SIZE);

        SetVectorMask<int32_t>(FULL_MASK, FULL_MASK);
    }

    template<bool is_raw_point, bool is_xyz>
    __aicore__ inline void Process();

private:
    int32_t blkIdx_, usedBlkNum_;
    TPipe pipe_;

    GlobalTensor<T> ptsGm_;
    GlobalTensor<float> voxGm_;

    TQue<QuePosition::VECIN, BUFFER_NUM> ptsQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> voxQue_;
    TBuf<TPosition::VECCALC> maskXBuf_, maskYBuf_, maskZBuf_;
    TBuf<TPosition::VECCALC> maskX1Buf_, maskY1Buf_, maskZ1Buf_;

    float gridX_, gridY_, gridZ_;
    float voxelSizeX_, voxelSizeY_, voxelSizeZ_;
    float voxelScaleX_, voxelScaleY_, voxelScaleZ_;
    float coorXMin_, coorYMin_, coorZMin_;
    int32_t coorXOffset_, coorYOffset_, coorZOffset_;

    // for task iteration, totalPts = avgPts * (avgTasks + tailTasks - 1)  + tailPts
    int32_t curTaskIdx_, curPtsIdx_;
    int32_t avgPts_, tailPts_, totalPts_; // here, avgPts_must be multiple of 64
    int32_t avgTasks_, tailTasks_, totalTasks_, coreTasks_;

    DataCopyParams avgCpParam_, tailCpParam_;
    UnaryRepeatParams unRptParam_ {1, 1, 8, 8};
    BinaryRepeatParams binRptParam_ {1, 1, 1, 8, 8, 8};
    uint8_t rptTimes_, maskRptTimes_;

private:
    __aicore__ inline void GetTiling(const PointToVoxelTilingData& tiling)
    {
        usedBlkNum_ = tiling.usedBlkNum;
        avgTasks_ = tiling.avgTasks;
        tailTasks_ = tiling.tailTasks;
        totalTasks_ = tiling.totalTasks;
        avgPts_ = tiling.avgPts;
        tailPts_ = tiling.tailPts;
        totalPts_ = tiling.totalPts;
        gridX_ = *reinterpret_cast<const float*>(&tiling.gridX);
        gridY_ = *reinterpret_cast<const float*>(&tiling.gridY);
        gridZ_ = *reinterpret_cast<const float*>(&tiling.gridZ);
        voxelSizeX_ = tiling.voxelSizeX;
        voxelSizeY_ = tiling.voxelSizeY;
        voxelSizeZ_ = tiling.voxelSizeZ;
        coorXMin_ = tiling.coorXMin;
        coorYMin_ = tiling.coorYMin;
        coorZMin_ = tiling.coorZMin;
    }

    __aicore__ inline bool IsLastTask() const
    {
        return curTaskIdx_ == totalTasks_ - 1;
    }

    template<bool is_raw_point, bool is_xyz, bool is_tail>
    __aicore__ inline void DoProcess();

    template<bool is_raw_point, bool is_xyz>
    __aicore__ inline void Compute();

    template<bool is_tail>
    __aicore__ inline void CopyIn();

    template<bool is_tail>
    __aicore__ inline void CopyOut();

    __aicore__ inline void ConvertRawPointToVoxel(
        const LocalTensor<float>& coorX, const LocalTensor<float>& coorY, const LocalTensor<float>& coorZ);

    template<bool is_xyz>
    __aicore__ inline void EncVoxel(
        const LocalTensor<int32_t>& coorX, const LocalTensor<int32_t>& coorY, const LocalTensor<int32_t>& coorZ);
};

template<typename T>
template<bool is_raw_point, bool is_xyz>
__aicore__ inline void PointToVoxelKernel<T>::Process()
{
    for (int32_t i = 0; i < coreTasks_ - 1; ++i) {
        DoProcess<is_raw_point, is_xyz, false>();
        ++curTaskIdx_;
        curPtsIdx_ += avgPts_;
    }
    if (IsLastTask()) {
        DoProcess<is_raw_point, is_xyz, true>();
    } else {
        DoProcess<is_raw_point, is_xyz, false>();
    }
}

template<typename T>
template<bool is_raw_point, bool is_xyz, bool is_tail>
__aicore__ inline void PointToVoxelKernel<T>::DoProcess()
{
    CopyIn<is_tail>();
    Compute<is_raw_point, is_xyz>();
    CopyOut<is_tail>();
}

template<typename T>
template<bool is_raw_point, bool is_xyz>
__aicore__ inline void PointToVoxelKernel<T>::Compute()
{
    LocalTensor<T> ptsT = ptsQue_.DeQue<T>();
    LocalTensor<T> coorX = ptsT[coorXOffset_];
    LocalTensor<T> coorY = ptsT[coorYOffset_];
    LocalTensor<T> coorZ = ptsT[coorZOffset_];
    if (is_raw_point) {
        ConvertRawPointToVoxel(coorX.template ReinterpretCast<float>(), coorY.template ReinterpretCast<float>(),
            coorZ.template ReinterpretCast<float>());
    }
    EncVoxel<is_xyz>(coorX.template ReinterpretCast<int32_t>(), coorY.template ReinterpretCast<int32_t>(),
        coorZ.template ReinterpretCast<int32_t>());
    ptsQue_.FreeTensor(ptsT);
}

template<typename T>
template<bool is_tail>
__aicore__ inline void PointToVoxelKernel<T>::CopyIn()
{
    auto cpParam = is_tail ? tailCpParam_ : avgCpParam_;
    LocalTensor<T> ptsT = ptsQue_.AllocTensor<T>();
    // [coor_x, coor_y, coor_z]
    DataCopy(ptsT[coorXOffset_], ptsGm_[curPtsIdx_], cpParam);
    DataCopy(ptsT[coorYOffset_], ptsGm_[totalPts_ + curPtsIdx_], cpParam);
    DataCopy(ptsT[coorZOffset_], ptsGm_[totalPts_ * 2 + curPtsIdx_], cpParam);
    ptsQue_.EnQue(ptsT);
}


template<typename T>
template<bool is_tail>
__aicore__ inline void PointToVoxelKernel<T>::CopyOut()
{
    auto cpParam = is_tail ? tailCpParam_ : avgCpParam_;
    LocalTensor<float> voxT = voxQue_.DeQue<float>();
    DataCopy(voxGm_[curPtsIdx_], voxT, cpParam);
    voxQue_.FreeTensor(voxT);
}

template<typename T>
__aicore__ inline void PointToVoxelKernel<T>::ConvertRawPointToVoxel(
    const LocalTensor<float>& coorX, const LocalTensor<float>& coorY, const LocalTensor<float>& coorZ)
{
    Adds<float, false>(coorX, coorX, -coorXMin_, MASK_PLACEHOLDER, rptTimes_, unRptParam_);
    Adds<float, false>(coorY, coorY, -coorYMin_, MASK_PLACEHOLDER, rptTimes_, unRptParam_);
    Adds<float, false>(coorZ, coorZ, -coorZMin_, MASK_PLACEHOLDER, rptTimes_, unRptParam_);
    Muls<float, false>(coorX, coorX, voxelScaleX_, MASK_PLACEHOLDER, rptTimes_, unRptParam_);
    Muls<float, false>(coorY, coorY, voxelScaleY_, MASK_PLACEHOLDER, rptTimes_, unRptParam_);
    Muls<float, false>(coorZ, coorZ, voxelScaleZ_, MASK_PLACEHOLDER, rptTimes_, unRptParam_);
    Cast<int32_t, float, false>(
        coorX.ReinterpretCast<int32_t>(), coorX, RoundMode::CAST_FLOOR, MASK_PLACEHOLDER, rptTimes_, unRptParam_);
    Cast<int32_t, float, false>(
        coorY.ReinterpretCast<int32_t>(), coorY, RoundMode::CAST_FLOOR, MASK_PLACEHOLDER, rptTimes_, unRptParam_);
    Cast<int32_t, float, false>(
        coorZ.ReinterpretCast<int32_t>(), coorZ, RoundMode::CAST_FLOOR, MASK_PLACEHOLDER, rptTimes_, unRptParam_);
}

template<typename T>
template<bool is_xyz>
__aicore__ inline void PointToVoxelKernel<T>::EncVoxel(
    const LocalTensor<int32_t>& coorX, const LocalTensor<int32_t>& coorY, const LocalTensor<int32_t>& coorZ)
{
    LocalTensor<uint8_t> maskX = maskXBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> maskY = maskYBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> maskZ = maskZBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> maskX1 = maskX1Buf_.Get<uint8_t>();
    LocalTensor<uint8_t> maskY1 = maskY1Buf_.Get<uint8_t>();
    LocalTensor<uint8_t> maskZ1 = maskZ1Buf_.Get<uint8_t>();
    // 1. find the valid part(> -1)
    CompareScalar<float, uint8_t, false>(
        maskX, coorX.ReinterpretCast<float>(), 0.f, CMPMODE::GE, MASK_PLACEHOLDER, rptTimes_, unRptParam_);
    CompareScalar<float, uint8_t, false>(
        maskY, coorY.ReinterpretCast<float>(), 0.f, CMPMODE::GE, MASK_PLACEHOLDER, rptTimes_, unRptParam_);
    CompareScalar<float, uint8_t, false>(
        maskZ, coorZ.ReinterpretCast<float>(), 0.f, CMPMODE::GE, MASK_PLACEHOLDER, rptTimes_, unRptParam_);
    CompareScalar<float, uint8_t, false>(
        maskX1, coorX.ReinterpretCast<float>(), gridX_, CMPMODE::LT, MASK_PLACEHOLDER, rptTimes_, unRptParam_);
    CompareScalar<float, uint8_t, false>(
        maskY1, coorY.ReinterpretCast<float>(), gridY_, CMPMODE::LT, MASK_PLACEHOLDER, rptTimes_, unRptParam_);
    CompareScalar<float, uint8_t, false>(
        maskZ1, coorZ.ReinterpretCast<float>(), gridZ_, CMPMODE::LT, MASK_PLACEHOLDER, rptTimes_, unRptParam_);
    PipeBarrier<PIPE_V>();
    And<uint16_t, false>(maskX.ReinterpretCast<uint16_t>(), maskX.ReinterpretCast<uint16_t>(),
        maskY.ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, maskRptTimes_, binRptParam_);
    And<uint16_t, false>(maskX.ReinterpretCast<uint16_t>(), maskX.ReinterpretCast<uint16_t>(),
        maskZ.ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, maskRptTimes_, binRptParam_);
    And<uint16_t, false>(maskX.ReinterpretCast<uint16_t>(), maskX.ReinterpretCast<uint16_t>(),
        maskX1.ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, maskRptTimes_, binRptParam_);
    And<uint16_t, false>(maskX.ReinterpretCast<uint16_t>(), maskX.ReinterpretCast<uint16_t>(),
        maskY1.ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, maskRptTimes_, binRptParam_);
    And<uint16_t, false>(maskX.ReinterpretCast<uint16_t>(), maskX.ReinterpretCast<uint16_t>(),
        maskZ1.ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER, maskRptTimes_, binRptParam_);
    // 2. encode voxel
    if (is_xyz) { // xyz
        ShiftLeft<int32_t, false>(coorY, coorY, ENC_BITS_Z, MASK_PLACEHOLDER, rptTimes_, unRptParam_);
        ShiftLeft<int32_t, false>(coorX, coorX, ENC_BITS + ENC_BITS_Z, MASK_PLACEHOLDER, rptTimes_, unRptParam_);
    } else { // zyx
        ShiftLeft<int32_t, false>(coorY, coorY, ENC_BITS, MASK_PLACEHOLDER, rptTimes_, unRptParam_);
        ShiftLeft<int32_t, false>(coorZ, coorZ, ENC_BITS + ENC_BITS, MASK_PLACEHOLDER, rptTimes_, unRptParam_);
    }
    LocalTensor<float> voxT = voxQue_.AllocTensor<float>();
    Add<int32_t, false>(coorX, coorX, coorY, MASK_PLACEHOLDER, rptTimes_, binRptParam_);
    Add<int32_t, false>(voxT.ReinterpretCast<int32_t>(), coorX, coorZ, MASK_PLACEHOLDER, rptTimes_, binRptParam_);
    // 3. filter the invalid with -1, select only support half & float
    Select<float, uint8_t, false>(
        voxT, maskX, voxT, -1.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, SELECT_MASK, rptTimes_, binRptParam_);
    SetVectorMask<int32_t>(FULL_MASK, FULL_MASK);
    voxQue_.EnQue(voxT);
}

extern "C" __global__ __aicore__ void point_to_voxel(GM_ADDR points, GM_ADDR voxels, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(0)) {
        PointToVoxelKernel<float> op(points, voxels, tilingData);
        op.template Process<true, true>();
    } else if (TILING_KEY_IS(1)) {
        PointToVoxelKernel<float> op(points, voxels, tilingData);
        op.template Process<true, false>();
    } else if (TILING_KEY_IS(2)) {
        PointToVoxelKernel<int32_t> op(points, voxels, tilingData);
        op.template Process<false, true>();
    } else if (TILING_KEY_IS(3)) {
        PointToVoxelKernel<int32_t> op(points, voxels, tilingData);
        op.template Process<false, false>();
    }
}
