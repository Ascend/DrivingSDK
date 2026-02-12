// Copyright (c) 2024 Huawei Technologies Co., Ltd


#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t ONE_REPEAT_B64_SIZE = 32;
constexpr uint64_t SELECT_MASK = 64;
constexpr int16_t ENC_BITS = 11;
constexpr int16_t ENC_BITS_Z = 8;
constexpr int32_t invalidNumber = -1082130432;
constexpr int32_t XYZ_COUNT = 3;


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
        coorYOffset_ = coorXOffset_ + avgPts_;
        coorZOffset_ = coorYOffset_ + avgPts_;
        avgCpParam_.blockLen = avgPts_ * sizeof(T) / ONE_BLK_SIZE;
        tailCpParam_.blockLen = Ceil(tailPts_ * sizeof(T), ONE_BLK_SIZE);
        voxelScaleX_ = 1.0f / voxelSizeX_;
        voxelScaleY_ = 1.0f / voxelSizeY_;
        voxelScaleZ_ = 1.0f / voxelSizeZ_;

        rptTimes_ = avgPts_ / ONE_REPEAT_FLOAT_SIZE;
        maskRptTimes_ = Ceil(rptTimes_, ONE_REPEAT_B64_SIZE);

        ptsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(points));
        voxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(voxels));

        pipe_.InitBuffer(ptsQue_, BUFFER_NUM, avgPts_ * sizeof(T) * XYZ_COUNT);
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

    int32_t gridX_, gridY_, gridZ_;
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
    uint16_t rptTimes_, maskRptTimes_;

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
        gridX_ = tiling.gridX;
        gridY_ = tiling.gridY;
        gridZ_ = tiling.gridZ;
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
    
    template<bool is_raw_point, bool is_xyz>
    __aicore__ inline void EncodeVoxelVF(const LocalTensor<T>& coorX, const LocalTensor<T>& coorY,
        const LocalTensor<T>& coorZ, const LocalTensor<int32_t>& voxT);
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
    LocalTensor<float> voxT = voxQue_.AllocTensor<float>();
    EncodeVoxelVF<is_raw_point, is_xyz>(coorX, coorY, coorZ, voxT.ReinterpretCast<int32_t>());
    voxQue_.EnQue(voxT);
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
template<bool is_raw_point, bool is_xyz>
__aicore__ inline void PointToVoxelKernel<T>::EncodeVoxelVF(const LocalTensor<T>& coorX, const LocalTensor<T>& coorY,
    const LocalTensor<T>& coorZ, const LocalTensor<int32_t>& voxT)
{
    __local_mem__ float* coorXFloatPtr = (__local_mem__ float*) coorX.GetPhyAddr();
    __local_mem__ float* coorYFloatPtr = (__local_mem__ float*) coorY.GetPhyAddr();
    __local_mem__ float* coorZFloatPtr = (__local_mem__ float*) coorZ.GetPhyAddr();

    __local_mem__ int32_t* coorXIntPtr = (__local_mem__ int32_t*) coorX.GetPhyAddr();
    __local_mem__ int32_t* coorYIntPtr = (__local_mem__ int32_t*) coorY.GetPhyAddr();
    __local_mem__ int32_t* coorZIntPtr = (__local_mem__ int32_t*) coorZ.GetPhyAddr();

    __local_mem__ int32_t* voxTPtr = (__local_mem__ int32_t*) voxT.GetPhyAddr();

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<float> xFloatReg, yFloatReg, zFloatReg;
        MicroAPI::RegTensor<int32_t> xIntReg, yIntReg, zIntReg;
        MicroAPI::RegTensor<int32_t> constValueReg, voxTReg;

        MicroAPI::MaskReg mask = MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg validMask, tmp1Mask, tmp2Mask;

        static constexpr AscendC::MicroAPI::CastTrait castF2ITrait = 
            {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};

        MicroAPI::Duplicate(constValueReg, invalidNumber, mask);

        for (uint16_t taskIdx = 0; taskIdx < rptTimes_; ++taskIdx) {
            uint32_t localOffset = taskIdx * B32_DATA_NUM_PER_REPEAT;

            if (is_raw_point) {
                MicroAPI::DataCopy(xFloatReg, coorXFloatPtr + localOffset);
                MicroAPI::DataCopy(yFloatReg, coorYFloatPtr + localOffset);
                MicroAPI::DataCopy(zFloatReg, coorZFloatPtr + localOffset);
                MicroAPI::Adds(xFloatReg, xFloatReg, -coorXMin_, mask);
                MicroAPI::Adds(yFloatReg, yFloatReg, -coorYMin_, mask);
                MicroAPI::Adds(zFloatReg, zFloatReg, -coorZMin_, mask);
                MicroAPI::Muls(xFloatReg, xFloatReg, voxelScaleX_, mask);
                MicroAPI::Muls(yFloatReg, yFloatReg, voxelScaleY_, mask);
                MicroAPI::Muls(zFloatReg, zFloatReg, voxelScaleZ_, mask);
                MicroAPI::Cast<int32_t, float, castF2ITrait>(xIntReg, xFloatReg, mask);
                MicroAPI::Cast<int32_t, float, castF2ITrait>(yIntReg, yFloatReg, mask);
                MicroAPI::Cast<int32_t, float, castF2ITrait>(zIntReg, zFloatReg, mask);
            } else {
                MicroAPI::DataCopy(xIntReg, coorXIntPtr + localOffset);
                MicroAPI::DataCopy(yIntReg, coorYIntPtr + localOffset);
                MicroAPI::DataCopy(zIntReg, coorZIntPtr + localOffset);
            }

            MicroAPI::Compares<int32_t, CMPMODE::GE>(tmp1Mask, xIntReg, 0, mask);
            MicroAPI::Compares<int32_t, CMPMODE::LT>(tmp2Mask, xIntReg, gridX_, mask);
            MicroAPI::And(validMask, tmp1Mask, tmp2Mask, mask);
            MicroAPI::Compares<int32_t, CMPMODE::GE>(tmp1Mask, yIntReg, 0, mask);
            MicroAPI::Compares<int32_t, CMPMODE::LT>(tmp2Mask, yIntReg, gridY_, mask);
            MicroAPI::And(validMask, validMask, tmp1Mask, mask);
            MicroAPI::And(validMask, validMask, tmp2Mask, mask);
            MicroAPI::Compares<int32_t, CMPMODE::GE>(tmp1Mask, zIntReg, 0, mask);
            MicroAPI::Compares<int32_t, CMPMODE::LT>(tmp2Mask, zIntReg, gridZ_, mask);
            MicroAPI::And(validMask, validMask, tmp1Mask, mask);
            MicroAPI::And(validMask, validMask, tmp2Mask, mask);

            if (is_xyz) {
                MicroAPI::ShiftLefts(yIntReg, yIntReg, ENC_BITS_Z, mask);
                MicroAPI::ShiftLefts(xIntReg, xIntReg, static_cast<int16_t>(ENC_BITS + ENC_BITS_Z), mask);
            } else {
                MicroAPI::ShiftLefts(yIntReg, yIntReg, ENC_BITS, mask);
                MicroAPI::ShiftLefts(zIntReg, zIntReg, static_cast<int16_t>(ENC_BITS + ENC_BITS), mask);
            }

            MicroAPI::Add(xIntReg, xIntReg, yIntReg, mask);
            MicroAPI::Add(voxTReg, xIntReg, zIntReg, mask);
            MicroAPI::Select(voxTReg, voxTReg, constValueReg, validMask);

            MicroAPI::DataCopy(voxTPtr + localOffset, voxTReg, mask);
        }
    }
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
