// Copyright (c) 2024 Huawei Technologies Co., Ltd


#include <kernel_common.h>

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t FREE_NUM = 1024;

class HardVoxelizeDiffKernel {
public:
    __aicore__ inline HardVoxelizeDiffKernel() = delete;
    __aicore__ inline ~HardVoxelizeDiffKernel() = default;
    __aicore__ inline HardVoxelizeDiffKernel(GM_ADDR uniIdxs, GM_ADDR uniLens, const HardVoxelizeTilingData& tiling)
        : blkIdx_(GetBlockIdx()), usedBlkNum_(tiling.usedDiffBlkNum), avgTasks_(tiling.avgDiffTasks),
          tailTasks_(tiling.tailDiffTasks), totalTasks_(tiling.totalDiffTasks), avgPts_(tiling.avgPts),
          tailPts_(tiling.tailPts), totalPts_(tiling.totalPts), numPts_(tiling.numPts)
    {
        // init task
        curTaskIdx_ = blkIdx_ < tailTasks_ ? blkIdx_ * (avgTasks_ + 1) : blkIdx_ * avgTasks_ + tailTasks_;
        coreTasks_ = blkIdx_ < tailTasks_ ? avgTasks_ + 1 : avgTasks_;
        curPtsIdx_ = curTaskIdx_ * avgPts_;

        rptTimes_ = avgPts_ / ONE_REPEAT_FLOAT_SIZE;
        adjOffset_ = avgPts_;

        cpParam_.blockLen = static_cast<uint16_t>(avgPts_ / B32_DATA_NUM_PER_BLOCK);
        cpTailParam_.blockLen = static_cast<uint16_t>(Ceil(tailPts_, B32_DATA_NUM_PER_BLOCK));
        cpExtParam_.blockLen = static_cast<uint32_t>(tailPts_ * B32_BYTE_SIZE);

        uniIdxsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(uniIdxs));
        uniLensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(uniLens));

        pipe_.InitBuffer(inQue_, BUFFER_NUM, avgPts_ * 2 * B32_BYTE_SIZE);
        pipe_.InitBuffer(outQue_, BUFFER_NUM, avgPts_ * B32_BYTE_SIZE);
    }

    __aicore__ inline void Process();

    __aicore__ inline void Done();

private:
    TPipe pipe_;
    GlobalTensor<int32_t> uniIdxsGm_, uniLensGm_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQue_;

    int32_t blkIdx_, usedBlkNum_;
    int32_t curTaskIdx_, curPtsIdx_, curOutputIdx_, startOutputIdx_;
    int32_t avgTasks_, tailTasks_, totalTasks_, coreTasks_;
    int32_t avgPts_, tailPts_, totalPts_, numPts_; // here, avgPts_must be multiple of 64
    int32_t adjOffset_;
    DataCopyParams cpParam_, cpTailParam_;
    DataCopyExtParams cpExtParam_;
    BinaryRepeatParams binRptParam_ {1, 1, 1, 8, 8, 8};
    uint8_t rptTimes_;

private:
    __aicore__ inline bool IsLastTask() const
    {
        return curTaskIdx_ == totalTasks_ - 1;
    }

    template<bool is_tail>
    __aicore__ inline void DoProcess();

    template<bool is_tail>
    __aicore__ inline void Compute();

    template<bool is_tail>
    __aicore__ inline void CopyIn();

    template<bool is_tail>
    __aicore__ inline void CopyOut();
};

__aicore__ inline void HardVoxelizeDiffKernel::Process()
{
    if (blkIdx_ >= usedBlkNum_) {
        return;
    }

    for (int32_t i = 0; i < coreTasks_ - 1; ++i) {
        DoProcess<false>();
        ++curTaskIdx_;
        curPtsIdx_ += avgPts_;
    }

    if (IsLastTask()) {
        DoProcess<true>();
    } else {
        DoProcess<false>();
    }
}

template<bool is_tail>
__aicore__ inline void HardVoxelizeDiffKernel::DoProcess()
{
    CopyIn<is_tail>();
    Compute<is_tail>();
    CopyOut<is_tail>();
}

template<bool is_tail>
__aicore__ inline void HardVoxelizeDiffKernel::CopyIn()
{
    LocalTensor<int32_t> inT = inQue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> idxT0 = inT[0];
    LocalTensor<int32_t> idxT1 = inT[adjOffset_];
    if (is_tail) {
        DataCopy(idxT0, uniIdxsGm_[curPtsIdx_], cpTailParam_);
        DataCopy(idxT1, uniIdxsGm_[curPtsIdx_ + 1], cpTailParam_);
    } else {
        DataCopy(idxT0, uniIdxsGm_[curPtsIdx_], cpParam_);
        DataCopy(idxT1, uniIdxsGm_[curPtsIdx_ + 1], cpParam_);
    }
    inQue_.EnQue(inT);
}

template<bool is_tail>
__aicore__ inline void HardVoxelizeDiffKernel::Compute()
{
    LocalTensor<int32_t> idxT = inQue_.DeQue<int32_t>();
    LocalTensor<int32_t> idxT0 = idxT[0];
    LocalTensor<int32_t> idxT1 = idxT[adjOffset_];
    LocalTensor<int32_t> outT = outQue_.AllocTensor<int32_t>();
    if (is_tail) {
        idxT1.SetValue(tailPts_ - 1, numPts_);
    }
    Sub<int32_t, false>(outT, idxT1, idxT0, MASK_PLACEHOLDER, rptTimes_, binRptParam_);
    outQue_.EnQue(outT);
    inQue_.FreeTensor(idxT);
}

template<bool is_tail>
__aicore__ inline void HardVoxelizeDiffKernel::CopyOut()
{
    LocalTensor<int32_t> outT = outQue_.DeQue<int32_t>();
    if (is_tail) {
        DataCopyPad(uniLensGm_[curPtsIdx_], outT, cpExtParam_);
    } else {
        DataCopy(uniLensGm_[curPtsIdx_], outT, cpParam_);
    }
    outQue_.FreeTensor(outT);
}

__aicore__ inline void HardVoxelizeDiffKernel::Done()
{
    pipe_.Destroy();
    SyncAll();
}

template<bool is_aligned>
class HardVoxelizeCopyKernel {
public:
    __aicore__ inline HardVoxelizeCopyKernel() = delete;
    __aicore__ inline ~HardVoxelizeCopyKernel() = default;
    __aicore__ inline HardVoxelizeCopyKernel(GM_ADDR points, GM_ADDR uniVoxels, GM_ADDR argsortVoxelIdxs,
        GM_ADDR uniArgsortIdxs, GM_ADDR uniIdxs, GM_ADDR uniLens, GM_ADDR voxels, GM_ADDR numPointsPerVoxel,
        GM_ADDR sortedUniVoxels, const HardVoxelizeTilingData& tiling)
        : blkIdx_(GetBlockIdx()), usedBlkNum_(tiling.usedCopyBlkNum), avgTasks_(tiling.avgCopyTasks),
          tailTasks_(tiling.tailCopyTasks), totalTasks_(tiling.totalCopyTasks), avgVoxs_(tiling.avgVoxs),
          tailVoxs_(tiling.tailVoxs), totalVoxs_(tiling.totalVoxs), featNum_(tiling.featNum),
          maxPoints_(tiling.maxPoints)
    {
        // init task
        curTaskIdx_ = blkIdx_ < tailTasks_ ? blkIdx_ * (avgTasks_ + 1) : blkIdx_ * avgTasks_ + tailTasks_;
        coreTasks_ = blkIdx_ < tailTasks_ ? avgTasks_ + 1 : avgTasks_;
        curVoxIdx_ = curTaskIdx_ * avgVoxs_;
        ptsStride_ = maxPoints_ * featNum_;

        cpVoxParam_.blockLen = static_cast<uint16_t>(avgVoxs_ / B32_DATA_NUM_PER_BLOCK);
        cpVoxTailParam_.blockLen = static_cast<uint16_t>(Ceil(tailVoxs_, B32_DATA_NUM_PER_BLOCK));
        cpExtParam_.blockLen = static_cast<uint32_t>(featNum_ * B32_BYTE_SIZE); // not aligned
        featBlk_ = Ceil(static_cast<uint16_t>(featNum_), B32_DATA_NUM_PER_BLOCK);
        cpPtParam_.blockLen = featBlk_;
        alignedFeatNum_ = AlignUp(featNum_, B32_DATA_NUM_PER_BLOCK);
        alignedMaxPointsNum_ = AlignUp(maxPoints_, B32_DATA_NUM_PER_BLOCK);
        cpArgsortIdxParam_.blockLen = static_cast<uint16_t>(alignedMaxPointsNum_ / B32_DATA_NUM_PER_BLOCK);

        // init global memory
        ptsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(points));
        uniVoxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(uniVoxels));
        argsortVoxIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(argsortVoxelIdxs));
        uniArgsortIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(uniArgsortIdxs));
        uniIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(uniIdxs));
        uniLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(uniLens));
        voxelsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(voxels));
        numPointsPerVoxelGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(numPointsPerVoxel));
        sortedUniVoxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sortedUniVoxels));

        // init buffer
        pipe_.InitBuffer(uniArgsortIdxQue_, BUFFER_NUM, avgVoxs_ * B32_BYTE_SIZE);
        pipe_.InitBuffer(uniIdxBuf_, ONE_BLK_SIZE);
        pipe_.InitBuffer(uniLenBuf_, ONE_BLK_SIZE);
        pipe_.InitBuffer(uniVoxBuf_, ONE_BLK_SIZE);
        pipe_.InitBuffer(ptsBuf_, maxPoints_ * alignedFeatNum_ * B32_BYTE_SIZE);
        pipe_.InitBuffer(argsortVoxBuf_, alignedMaxPointsNum_ * B32_BYTE_SIZE);

        cpInId_ = pipe_.AllocEventID<HardEvent::MTE2_MTE3>();
        cpOutId_ = pipe_.AllocEventID<HardEvent::MTE3_MTE2>();
        calcId_ = pipe_.AllocEventID<HardEvent::MTE2_V>();
    }

    __aicore__ inline void Process();

private:
    TPipe pipe_;
    GlobalTensor<float> ptsGm_;
    GlobalTensor<int32_t> uniVoxGm_, argsortVoxIdxGm_, uniArgsortIdxGm_, uniIdxGm_, uniLenGm_;
    GlobalTensor<int32_t> numPointsPerVoxelGm_, sortedUniVoxGm_;
    GlobalTensor<float> voxelsGm_;
    TBuf<TPosition::LCM> uniIdxBuf_, uniLenBuf_, uniVoxBuf_, ptsBuf_, argsortVoxBuf_;
    TQue<TPosition::VECIN, BUFFER_NUM> uniArgsortIdxQue_;

    int32_t blkIdx_, usedBlkNum_;
    int32_t curTaskIdx_, curVoxIdx_;
    int32_t avgTasks_, tailTasks_, totalTasks_, coreTasks_;
    int32_t avgVoxs_, tailVoxs_, totalVoxs_;
    int32_t featNum_, alignedMaxPointsNum_, alignedFeatNum_;
    int32_t lenOffset_ {8};
    int32_t maxPoints_, ptsStride_;
    uint16_t featBlk_;

    DataCopyParams cpVoxParam_, cpVoxTailParam_, cpOneParam_ {1, 1, 0, 0}, cpArgsortIdxParam_, cpPtParam_,
        cpOutParam_ {1, 0, 0, 0};
    DataCopyExtParams cpExtParam_, cp4BytesParam_ {1, 4, 0, 0, 0};

    TEventID cpInId_, cpOutId_, calcId_;

private:
    __aicore__ inline bool IsLastTask() const
    {
        return curTaskIdx_ == totalTasks_ - 1;
    }

    template<bool is_tail>
    __aicore__ inline void DoProcess();

    template<bool is_tail>
    __aicore__ inline void Compute();

    template<bool is_tail>
    __aicore__ inline void CopyIn();

    template<bool is_tail>
    __aicore__ inline void CopyOut();
};

template<bool is_aligned>
__aicore__ inline void HardVoxelizeCopyKernel<is_aligned>::Process()
{
    if (blkIdx_ >= usedBlkNum_) {
        return;
    }

    for (int32_t i = 0; i < coreTasks_ - 1; ++i) {
        DoProcess<false>();
        ++curTaskIdx_;
        curVoxIdx_ += avgVoxs_;
    }

    if (IsLastTask()) {
        DoProcess<true>();
    } else {
        DoProcess<false>();
    }
}

template<bool is_aligned>
template<bool is_tail>
__aicore__ inline void HardVoxelizeCopyKernel<is_aligned>::DoProcess()
{
    CopyIn<is_tail>();
    CopyOut<is_tail>();
}

template<bool is_aligned>
template<bool is_tail>
__aicore__ inline void HardVoxelizeCopyKernel<is_aligned>::CopyIn()
{
    LocalTensor<int32_t> uniArgsortIdxT = uniArgsortIdxQue_.AllocTensor<int32_t>();
    if (is_tail) {
        DataCopy(uniArgsortIdxT, uniArgsortIdxGm_[curVoxIdx_], this->cpVoxTailParam_);
    } else {
        DataCopy(uniArgsortIdxT, uniArgsortIdxGm_[curVoxIdx_], cpVoxParam_);
    }
    uniArgsortIdxQue_.EnQue(uniArgsortIdxT);
}

template<bool is_aligned>
template<bool is_tail>
__aicore__ inline void HardVoxelizeCopyKernel<is_aligned>::CopyOut()
{
    auto loops = is_tail ? tailVoxs_ : avgVoxs_;
    LocalTensor<int32_t> uniArgsortIdxT = uniArgsortIdxQue_.DeQue<int32_t>();
    LocalTensor<int32_t> uniIdxT = uniIdxBuf_.Get<int32_t>();
    LocalTensor<int32_t> uniLenT = uniLenBuf_.Get<int32_t>();
    LocalTensor<int32_t> uniVoxT = uniVoxBuf_.Get<int32_t>();
    LocalTensor<float> ptsT = ptsBuf_.Get<float>();
    LocalTensor<int32_t> argsortVoxT = argsortVoxBuf_.Get<int32_t>();

    SetFlag<HardEvent::MTE3_MTE2>(cpOutId_);
    for (int32_t i = 0; i < loops; ++i) {
        int32_t idx = uniArgsortIdxT.GetValue(i);
        WaitFlag<HardEvent::MTE3_MTE2>(cpOutId_);
        DataCopy(uniIdxT, uniIdxGm_[idx], cpOneParam_);
        DataCopy(uniLenT, uniLenGm_[idx], cpOneParam_);
        DataCopy(uniVoxT, uniVoxGm_[idx], cpOneParam_);
        auto uniIdx = uniIdxT.GetValue(0);
        auto uniLen = uniLenT.GetValue(0);
        auto uniVox = uniVoxT.GetValue(0);
        if (uniLen > maxPoints_) {
            uniLen = maxPoints_;
            uniLenT.SetValue(0, maxPoints_);
        }

        DataCopy(argsortVoxT, argsortVoxIdxGm_[uniIdx], cpArgsortIdxParam_);
        SetFlag<HardEvent::MTE2_V>(calcId_);
        WaitFlag<HardEvent::MTE2_V>(calcId_);
        Muls(argsortVoxT, argsortVoxT, featNum_, alignedMaxPointsNum_);
        for (int32_t j = 0; j < uniLen; ++j) {
            DataCopy(ptsT[j * alignedFeatNum_], ptsGm_[argsortVoxT.GetValue(j)], cpPtParam_);
        }
        SetFlag<HardEvent::MTE2_MTE3>(cpInId_);
        
        WaitFlag<HardEvent::MTE2_MTE3>(cpInId_);
        if (is_aligned) {
            cpOutParam_.blockLen = uniLen * featBlk_;
            DataCopy(voxelsGm_[curVoxIdx_ * ptsStride_], ptsT, cpOutParam_);
        } else {
            cpExtParam_.blockCount = static_cast<uint16_t>(uniLen);
            DataCopyPad(voxelsGm_[curVoxIdx_ * ptsStride_], ptsT, cpExtParam_);
        }
        DataCopyPad(numPointsPerVoxelGm_[curVoxIdx_], uniLenT, cp4BytesParam_);
        DataCopyPad(sortedUniVoxGm_[curVoxIdx_], uniVoxT, cp4BytesParam_);
        SetFlag<HardEvent::MTE3_MTE2>(cpOutId_);

        curVoxIdx_++;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(cpOutId_);
    uniArgsortIdxQue_.FreeTensor(uniArgsortIdxT);
}

extern "C" __global__ __aicore__ void hard_voxelize(GM_ADDR points, GM_ADDR uniVoxels, GM_ADDR argsortVoxelIdxs,
    GM_ADDR uniArgsortIdxs, GM_ADDR uniIdxs, GM_ADDR voxels, GM_ADDR numPointsPerVoxel, GM_ADDR sortedUniVoxels,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(0)) {
        // phase 1: calculate the length of voxels, i.e. num_per_voxel
        HardVoxelizeDiffKernel diffOp(uniIdxs, workspace, tilingData);
        diffOp.Process();
        diffOp.Done();
        // phase 2: group the points by the voxel index, sort by the point order.
        HardVoxelizeCopyKernel<false> copyOp(points, uniVoxels, argsortVoxelIdxs, uniArgsortIdxs, uniIdxs, workspace,
            voxels, numPointsPerVoxel, sortedUniVoxels, tilingData);
        copyOp.Process();

    } else if (TILING_KEY_IS(1)) {
        // phase 1: calculate the length of voxels, i.e. num_per_voxel
        HardVoxelizeDiffKernel diffOp(uniIdxs, workspace, tilingData);
        diffOp.Process();
        diffOp.Done();
        // phase 2: group the points by the voxel index, sort by the point order.
        HardVoxelizeCopyKernel<true> copyOp(points, uniVoxels, argsortVoxelIdxs, uniArgsortIdxs, uniIdxs, workspace,
            voxels, numPointsPerVoxel, sortedUniVoxels, tilingData);
        copyOp.Process();
    }
}
