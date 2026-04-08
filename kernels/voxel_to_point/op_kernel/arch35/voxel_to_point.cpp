// Copyright (c) 2024 Huawei Technologies Co., Ltd


#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t ONE_REPEAT_B64_SIZE = 32;
constexpr uint64_t SELECT_MASK = 64;
constexpr int16_t ENC_BITS = 11;
constexpr int16_t ENC_BITS_Z = 8;
constexpr int16_t COORD_DIM = 3;

class VoxelToPointKernel {
public:
    __aicore__ inline VoxelToPointKernel() = delete;
    __aicore__ inline ~VoxelToPointKernel() = default;
    __aicore__ inline VoxelToPointKernel(GM_ADDR voxels, GM_ADDR points, const PointToVoxelTilingData& tiling, TPipe* pipe)
        : blkIdx_(GetBlockIdx()), usedBlkNum_(tiling.usedBlkNum), avgTasks_(tiling.avgTasks),
          tailTasks_(tiling.tailTasks), totalTasks_(tiling.totalTasks), avgPts_(tiling.avgPts),
          tailPts_(tiling.tailPts), totalPts_(tiling.totalPts), gridX_(tiling.gridX), gridY_(tiling.gridY),
          gridZ_(tiling.gridZ), voxelSizeX_(tiling.voxelSizeX), voxelSizeY_(tiling.voxelSizeY),
          voxelSizeZ_(tiling.voxelSizeZ), coorXMin_(tiling.coorXMin), coorYMin_(tiling.coorYMin),
          coorZMin_(tiling.coorZMin), pipe_(pipe)
    {
        // init task
        curTaskIdx_ = blkIdx_ < tailTasks_ ? blkIdx_ * (avgTasks_ + 1) : blkIdx_ * avgTasks_ + tailTasks_;
        coreTasks_ = blkIdx_ < tailTasks_ ? avgTasks_ + 1 : avgTasks_;
        curPtsIdx_ = curTaskIdx_ * avgPts_;

        coorXOffset_ = 0;
        coorYOffset_ = avgPts_;
        coorZOffset_ = avgPts_ * (COORD_DIM - 1);
        voxelScaleX_ = 1.0f / voxelSizeX_;
        voxelScaleY_ = 1.0f / voxelSizeY_;
        voxelScaleZ_ = 1.0f / voxelSizeZ_;

        avgCpParam_.blockLen = avgPts_ * sizeof(int32_t) / ONE_BLK_SIZE;
        tailCpParam_.blockLen = Ceil(tailPts_ * sizeof(int32_t), ONE_BLK_SIZE);
        tailPadParam_.blockLen = tailPts_ * sizeof(int32_t);
        rptTimes_ = avgPts_ / ONE_REPEAT_FLOAT_SIZE;
        maskRptTimes_ = Ceil(rptTimes_, ONE_REPEAT_B64_SIZE);

        ptsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(points));
        voxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(voxels));

        pipe_->InitBuffer(ptsQue_, BUFFER_NUM, avgPts_ * sizeof(int32_t) * COORD_DIM);
        pipe_->InitBuffer(voxQue_, BUFFER_NUM, avgPts_ * sizeof(int32_t));

        SetVectorMask<int32_t>(FULL_MASK, FULL_MASK);
    }

    template<bool is_xyz>
    __aicore__ inline void Process();

private:
    int32_t blkIdx_, usedBlkNum_;
    TPipe* pipe_;

    GlobalTensor<int32_t> ptsGm_;
    GlobalTensor<int32_t> voxGm_;

    TQue<QuePosition::VECIN, BUFFER_NUM> voxQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> ptsQue_;

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
    DataCopyExtParams tailPadParam_;
    UnaryRepeatParams unRptParam_ {1, 1, 8, 8};
    BinaryRepeatParams binRptParam_ {1, 1, 1, 8, 8, 8};
    uint8_t rptTimes_, maskRptTimes_;

private:
    __aicore__ inline bool IsLastTask() const
    {
        return curTaskIdx_ == totalTasks_ - 1;
    }

    template<bool is_xyz, bool is_tail>
    __aicore__ inline void DoProcess();

    template<bool is_xyz>
    __aicore__ inline void Compute();

    template<bool is_tail>
    __aicore__ inline void CopyIn();

    template<bool is_tail>
    __aicore__ inline void CopyOut();

    template<bool is_tail>
    __simd_vf__ inline void DecVoxel(__ubuf__ int32_t* srcPtr, __ubuf__ int32_t* dst0Ptr, __ubuf__ int32_t* dst1Ptr,
        __ubuf__ int32_t* dst2Ptr, uint32_t count, uint16_t oneRepeatSize, uint16_t repeatTimes);
};

template<bool is_xyz>
__aicore__ inline void VoxelToPointKernel::Process()
{
    for (int32_t i = 0; i < coreTasks_ - 1; ++i) {
        DoProcess<is_xyz, false>();
        ++curTaskIdx_;
        curPtsIdx_ += avgPts_;
    }
    if (IsLastTask()) {
        DoProcess<is_xyz, true>();
    } else {
        DoProcess<is_xyz, false>();
    }
}

template<bool is_xyz, bool is_tail>
__aicore__ inline void VoxelToPointKernel::DoProcess()
{
    CopyIn<is_tail>();
    Compute<is_xyz>();
    CopyOut<is_tail>();
}

template<bool is_xyz>
__aicore__ inline void VoxelToPointKernel::Compute()
{
    LocalTensor<int32_t> voxT = voxQue_.DeQue<int32_t>();
    LocalTensor<int32_t> ptsT = ptsQue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> coorX = ptsT[coorXOffset_];
    LocalTensor<int32_t> coorY = ptsT[coorYOffset_];
    LocalTensor<int32_t> coorZ = ptsT[coorZOffset_];
    constexpr uint16_t oneRepeatSize = GetVecLen()/sizeof(int32_t);
    uint16_t repeatTimes = CeilDivision(avgPts_, oneRepeatSize);
    __ubuf__ int32_t* srcPtr = (__ubuf__ int32_t*)voxT.GetPhyAddr();
    __ubuf__ int32_t* dst0Ptr = (__ubuf__ int32_t*)coorX.GetPhyAddr();
    __ubuf__ int32_t* dst1Ptr = (__ubuf__ int32_t*)coorY.GetPhyAddr();
    __ubuf__ int32_t* dst2Ptr = (__ubuf__ int32_t*)coorZ.GetPhyAddr();
    DecVoxel<is_xyz>(srcPtr, dst0Ptr, dst1Ptr, dst2Ptr, avgPts_, oneRepeatSize, repeatTimes);
    voxQue_.FreeTensor(voxT);
    ptsQue_.EnQue(ptsT);
}

template<bool is_tail>
__aicore__ inline void VoxelToPointKernel::CopyIn()
{
    auto cpParam = is_tail ? tailCpParam_ : avgCpParam_;
    LocalTensor<int32_t> voxT = voxQue_.AllocTensor<int32_t>();
    DataCopy(voxT, voxGm_[curPtsIdx_], cpParam);
    voxQue_.EnQue(voxT);
}


template<bool is_tail>
__aicore__ inline void VoxelToPointKernel::CopyOut()
{
    LocalTensor<int32_t> ptsT = ptsQue_.DeQue<int32_t>();
    // [coor_x, coor_y, coor_z]
    if (is_tail) {
        DataCopyPad(ptsGm_[curPtsIdx_], ptsT[coorXOffset_], tailPadParam_);
        DataCopyPad(ptsGm_[totalPts_ + curPtsIdx_], ptsT[coorYOffset_], tailPadParam_);
        DataCopyPad(ptsGm_[totalPts_ * 2 + curPtsIdx_], ptsT[coorZOffset_], tailPadParam_);
    } else {
        DataCopy(ptsGm_[curPtsIdx_], ptsT[coorXOffset_], avgCpParam_);
        DataCopy(ptsGm_[totalPts_ + curPtsIdx_], ptsT[coorYOffset_], avgCpParam_);
        DataCopy(ptsGm_[totalPts_ * 2 + curPtsIdx_], ptsT[coorZOffset_], avgCpParam_);
    }
    ptsQue_.FreeTensor(ptsT);
}

template<bool is_xyz>
__simd_vf__ inline void VoxelToPointKernel::DecVoxel(__ubuf__ int32_t* srcPtr, __ubuf__ int32_t* dst0Ptr, __ubuf__ int32_t* dst1Ptr,
    __ubuf__ int32_t* dst2Ptr, uint32_t count, uint16_t oneRepeatSize, uint16_t repeatTimes)
{
    MicroAPI::RegTensor<int32_t> voxTReg;
    MicroAPI::RegTensor<int32_t> coorXReg;
    MicroAPI::RegTensor<int32_t> coorYReg;
    MicroAPI::RegTensor<int32_t> coorZReg;
    MicroAPI::MaskReg maskReg;

    for (uint16_t i = 0; i < repeatTimes; i++) {
        maskReg = MicroAPI::UpdateMask<int32_t>(count);
        MicroAPI::DataCopy(voxTReg, srcPtr + i * oneRepeatSize);
        MicroAPI::ShiftRights(coorZReg, voxTReg, is_xyz ? ENC_BITS_Z : ENC_BITS, maskReg);
        MicroAPI::ShiftLefts(coorZReg, coorZReg, is_xyz ? ENC_BITS_Z : ENC_BITS, maskReg);
        MicroAPI::Sub(coorZReg, voxTReg, coorZReg, maskReg);
        MicroAPI::ShiftRights(voxTReg, voxTReg, is_xyz ? ENC_BITS_Z : ENC_BITS, maskReg);
        MicroAPI::ShiftRights(coorXReg, voxTReg, ENC_BITS, maskReg);
        MicroAPI::ShiftLefts(coorYReg, coorXReg, ENC_BITS, maskReg);
        MicroAPI::Sub(coorYReg, voxTReg, coorYReg, maskReg);
        MicroAPI::DataCopy(dst0Ptr + i * oneRepeatSize, coorXReg, maskReg);
        MicroAPI::DataCopy(dst1Ptr + i * oneRepeatSize, coorYReg, maskReg);
        MicroAPI::DataCopy(dst2Ptr + i * oneRepeatSize, coorZReg, maskReg);
    }
}

extern "C" __global__ __aicore__ void voxel_to_point(GM_ADDR voxels, GM_ADDR points, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;

    if (TILING_KEY_IS(0)) {
        VoxelToPointKernel op(voxels, points, tilingData, &pipe);
        op.Process<true>();
    } else if (TILING_KEY_IS(1)) {
        VoxelToPointKernel op(voxels, points, tilingData, &pipe);
        op.Process<false>();
    }
}
