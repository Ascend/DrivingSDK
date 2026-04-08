// Copyright (c) 2024 Huawei Technologies Co., Ltd


#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t MOVE_BYTE = 160 * 1024;
constexpr int32_t MOVE_NUM = MOVE_BYTE / B32_BYTE_SIZE;

class UniqueVoxelKernel {
public:
    __aicore__ inline UniqueVoxelKernel() = delete;
    __aicore__ inline ~UniqueVoxelKernel() = default;
    __aicore__ inline UniqueVoxelKernel(GM_ADDR voxels, GM_ADDR idxs, GM_ADDR argsortIdxs, GM_ADDR uniVoxs,
        GM_ADDR uniIdxs, GM_ADDR uniArgsortIdxs, GM_ADDR voxNum, GM_ADDR workspace, const UniqueVoxelTilingData& tiling)
        : blkIdx_(GetBlockIdx()), usedBlkNum_(tiling.usedBlkNum), avgTasks_(tiling.avgTasks),
          tailTasks_(tiling.tailTasks), totalTasks_(tiling.totalTasks), avgPts_(tiling.avgPts), tailPts_(tiling.tailPts)
    {
        // init task
        curTaskIdx_ = blkIdx_ < tailTasks_ ? blkIdx_ * (avgTasks_ + 1) : blkIdx_ * avgTasks_ + tailTasks_;
        coreTasks_ = blkIdx_ < tailTasks_ ? avgTasks_ + 1 : avgTasks_;
        curPtsIdx_ = curTaskIdx_ * avgPts_;
        curOutputIdx_ = curTaskIdx_ * avgPts_ + 1;
        startOutputIdx_ = curOutputIdx_;

        rptTimes_ = avgPts_ / ONE_REPEAT_FLOAT_SIZE;
        adjOffset_ = avgPts_;
        idxOffset_ = 2 * avgPts_;
        argOffset_ = 3 * avgPts_;
        rsvCntOffset_ = 3 * avgPts_;

        cpParam_.blockLen = static_cast<uint16_t>(avgPts_ / B32_DATA_NUM_PER_BLOCK);
        cpExtParam_.blockLen = static_cast<uint32_t>(tailPts_ * B32_BYTE_SIZE);
        padParam_.rightPadding = static_cast<uint8_t>(AlignUp(tailPts_, B32_DATA_NUM_PER_BLOCK) - tailPts_);

        // init global memory
        voxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(voxels));
        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idxs));
        argsortIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(argsortIdxs));
        uniVoxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(uniVoxs));
        uniIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(uniIdxs));
        uniArgsortIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(uniArgsortIdxs));
        voxNumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(voxNum));
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace));

        // init buffer
        pipe_.InitBuffer(inQue_, BUFFER_NUM, avgPts_ * 3 * B32_BYTE_SIZE);
        pipe_.InitBuffer(uniQue_, BUFFER_NUM, avgPts_ * 3 * B32_BYTE_SIZE + 3 * ONE_BLK_SIZE);
        pipe_.InitBuffer(maskBuf_, avgPts_ / 8);

        vecEvent_ = pipe_.AllocEventID<HardEvent::V_MTE2>();
        SetVectorMask<float>(FULL_MASK, FULL_MASK);
    }

    __aicore__ inline void Process();

private:
    TPipe pipe_;
    GlobalTensor<int32_t> voxGm_, idxGm_, argsortIdxGm_, uniVoxGm_, uniIdxGm_, uniArgsortIdxGm_, voxNumGm_;
    GlobalTensor<int32_t> workspaceGm_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> uniQue_;
    TBuf<TPosition::VECCALC> maskBuf_;
    int32_t blkIdx_, usedBlkNum_;
    int32_t curTaskIdx_, curPtsIdx_, curOutputIdx_, startOutputIdx_;
    int32_t avgTasks_, tailTasks_, totalTasks_, coreTasks_;
    int32_t avgPts_, tailPts_; // here, avgPts_must be multiple of 64
    int32_t adjOffset_, idxOffset_, argOffset_, rsvCntOffset_;
    DataCopyParams cpParam_;
    DataCopyExtParams cpExtParam_, cpOneIntParam_ {1, 4, 0, 0, 0}, cpDoubleIntsParam_ {1, 8, 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParam_ {true, 0, 0, -1};
    UnaryRepeatParams unRptParam_ {1, 1, 8, 8};
    BinaryRepeatParams binRptParam_ {1, 1, 1, 8, 8, 8};
    GatherMaskParams gatherMaskParam_ {1, 1, 8, 1};
    uint8_t rptTimes_;
    uint64_t voxCnt_ {0};
    int32_t headVox_, headArgsortIdx_;
    bool hasHeadVox_;

    TEventID vecEvent_;

private:
    __aicore__ inline bool IsFirstTask() const
    {
        return curTaskIdx_ == 0;
    }

    __aicore__ inline bool IsLastTask() const
    {
        return curTaskIdx_ == totalTasks_ - 1;
    }

    template<bool is_head, bool is_tail>
    __aicore__ inline void DoProcess();

    template<bool is_head, bool is_tail>
    __aicore__ inline void Compute();

    template<bool is_tail>
    __aicore__ inline void CopyIn();

    template<bool is_head>
    __aicore__ inline void CopyOut();

    __aicore__ inline void CopyVoxel();

    __aicore__ inline void CompactOutput();
};

__aicore__ inline void UniqueVoxelKernel::Process()
{
    int32_t i = 0;
    if (IsFirstTask()) {
        if (IsLastTask()) {
            DoProcess<true, true>();
        } else {
            DoProcess<true, false>();
            ++curTaskIdx_;
            curPtsIdx_ += avgPts_;
        }
        ++i;
    }

    for (; i < coreTasks_ - 1; ++i) {
        DoProcess<false, false>();
        ++curTaskIdx_;
        curPtsIdx_ += avgPts_;
    }

    if (i < coreTasks_) {
        if (IsLastTask()) {
            DoProcess<false, true>();
        } else {
            DoProcess<false, false>();
        }
    }

    CopyVoxel();

    pipe_.Destroy();
    SyncAll();

    CompactOutput();
}

template<bool is_head, bool is_tail>
__aicore__ inline void UniqueVoxelKernel::DoProcess()
{
    CopyIn<is_tail>();
    Compute<is_head, is_tail>();
    CopyOut<is_head>();
}

template<bool is_tail>
__aicore__ inline void UniqueVoxelKernel::CopyIn()
{
    // we need to pad -1 for tail
    LocalTensor<int32_t> inT = inQue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> voxA = inT[0];
    LocalTensor<int32_t> voxB = inT[adjOffset_];
    LocalTensor<int32_t> idxT = inT[idxOffset_];
    LocalTensor<int32_t> argT = inT[argOffset_];
    if (is_tail) {
        Duplicate<int32_t, false>(voxA, -1, MASK_PLACEHOLDER, rptTimes_, 1, 8);
        Duplicate<int32_t, false>(voxB, -1, MASK_PLACEHOLDER, rptTimes_, 1, 8);
        SetFlag<HardEvent::V_MTE2>(vecEvent_);
        WaitFlag<HardEvent::V_MTE2>(vecEvent_);
        DataCopyPad(voxB, voxGm_[curPtsIdx_], cpExtParam_, padParam_);
        DataCopyPad(voxA, voxGm_[curPtsIdx_ + 1], cpExtParam_, padParam_);
        DataCopyPad(idxT, idxGm_[curPtsIdx_], cpExtParam_, padParam_);
        DataCopyPad(argT, argsortIdxGm_[curPtsIdx_ + 1], cpExtParam_, padParam_);
    } else {
        DataCopy(voxB, voxGm_[curPtsIdx_], cpParam_);
        DataCopy(voxA, voxGm_[curPtsIdx_ + 1], cpParam_);
        DataCopy(idxT, idxGm_[curPtsIdx_], cpParam_);
        DataCopy(argT, argsortIdxGm_[curPtsIdx_ + 1], cpParam_);
    }
    inQue_.EnQue(inT);
}

template<bool is_head, bool is_tail>
__aicore__ inline void UniqueVoxelKernel::Compute()
{
    LocalTensor<int32_t> voxT = inQue_.DeQue<int32_t>();
    LocalTensor<int32_t> voxA = voxT[0];
    LocalTensor<int32_t> voxB = voxT[adjOffset_];
    LocalTensor<int32_t> idxT = voxT[idxOffset_];
    LocalTensor<int32_t> argT = voxT[argOffset_];
    LocalTensor<uint8_t> mask = maskBuf_.AllocTensor<uint8_t>();
    LocalTensor<int32_t> uniT = uniQue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> uniVox = uniT[0];
    LocalTensor<int32_t> uniIdx = uniT[adjOffset_];
    LocalTensor<int32_t> uniArg = uniT[idxOffset_];
    LocalTensor<int32_t> rsvCntT = uniT[rsvCntOffset_];
    uint64_t rsvCnt;
    // we need to look at the first element of the voxel
    if (is_head) {
        headVox_ = voxB.GetValue(0);
        headArgsortIdx_ = argsortIdxGm_.GetValue(0);
        hasHeadVox_ = headVox_ > -1;
        if (hasHeadVox_) {
            ++voxCnt_;
        }
    }

    if (is_tail) {
        voxA.SetValue(tailPts_ - 1, -1);
    }

    Sub<int32_t, false>(uniVox, voxA, voxB, MASK_PLACEHOLDER, rptTimes_, binRptParam_);
    PipeBarrier<PIPE_V>();
    CompareScalar<float, uint8_t, false>(
        mask, uniVox.ReinterpretCast<float>(), 0, CMPMODE::GT, MASK_PLACEHOLDER, rptTimes_, unRptParam_);
    GatherMask<int32_t, uint32_t>(
        uniVox, voxA, mask.ReinterpretCast<uint32_t>(), true, avgPts_, gatherMaskParam_, rsvCnt);
    GatherMask<int32_t, uint32_t>(
        uniIdx, idxT, mask.ReinterpretCast<uint32_t>(), true, avgPts_, gatherMaskParam_, rsvCnt);
    GatherMask<int32_t, uint32_t>(
        uniArg, argT, mask.ReinterpretCast<uint32_t>(), true, avgPts_, gatherMaskParam_, rsvCnt);
    inQue_.FreeTensor(voxT);
    SetVectorMask<float>(FULL_MASK, FULL_MASK);
    voxCnt_ += rsvCnt;
    rsvCntT.SetValue(0, static_cast<int32_t>(rsvCnt));
    uniQue_.EnQue(uniT);
}

template<bool is_head>
__aicore__ inline void UniqueVoxelKernel::CopyOut()
{
    LocalTensor<int32_t> uniT = uniQue_.DeQue<int32_t>();
    LocalTensor<int32_t> uniVoxT = uniT[0];
    LocalTensor<int32_t> uniIdxT = uniT[adjOffset_];
    LocalTensor<int32_t> uniArgT = uniT[idxOffset_];
    LocalTensor<int32_t> rsvCntT = uniT[rsvCntOffset_];
    int32_t rsvCnt = rsvCntT.GetValue(0);

    // since we do the adjcent difference(the first element is not counted in), we need to check the first element
    if (is_head) {
        if (hasHeadVox_) {
            rsvCntT.SetValue(0, headVox_);
            rsvCntT.SetValue(8, 0);
            rsvCntT.SetValue(16, headArgsortIdx_);
            DataCopyPad(uniVoxGm_, rsvCntT, cpOneIntParam_);
            DataCopyPad(uniIdxGm_, rsvCntT[8], cpOneIntParam_);
            DataCopyPad(uniArgsortIdxGm_, rsvCntT[16], cpOneIntParam_);
        } else {
            curOutputIdx_ = 0;
        }
    }

    if (rsvCnt > 0) {
        DataCopyParams mvParam(1,
            static_cast<uint16_t>(
                AlignUp(static_cast<int32_t>(rsvCnt), B32_DATA_NUM_PER_BLOCK) / B32_DATA_NUM_PER_BLOCK),
            0, 0);
        DataCopy(uniVoxGm_[curOutputIdx_], uniVoxT, mvParam);
        DataCopy(uniIdxGm_[curOutputIdx_], uniIdxT, mvParam);
        DataCopy(uniArgsortIdxGm_[curOutputIdx_], uniArgT, mvParam);
        PipeBarrier<PIPE_ALL>();
        
        curOutputIdx_ += rsvCnt;
    }
    uniQue_.FreeTensor(uniT);
}

__aicore__ inline void UniqueVoxelKernel::CopyVoxel()
{
    // copy voxel count to workspace
    LocalTensor<int32_t> cntT = uniQue_.AllocTensor<int32_t>();
    cntT.SetValue(0, startOutputIdx_);
    cntT.SetValue(1, static_cast<int32_t>(voxCnt_));
    DataCopyPad(workspaceGm_[blkIdx_ * 2], cntT, cpDoubleIntsParam_);
    uniQue_.FreeTensor(cntT);
}

__aicore__ inline void UniqueVoxelKernel::CompactOutput()
{
    if (blkIdx_ == 0) {
        TPipe pipe;
        TBuf<TPosition::VECCALC> mvBuf;
        pipe.InitBuffer(mvBuf, MOVE_BYTE);
        TEventID mte2Event = pipe.AllocEventID<HardEvent::MTE2_MTE3>();
        TEventID mte3Event = pipe.AllocEventID<HardEvent::MTE3_MTE2>();

        int32_t totalVoxelCnt = workspaceGm_.GetValue(1);
        LocalTensor<int32_t> inT = mvBuf.Get<int32_t>();

        for (int32_t i = 1; i < usedBlkNum_; ++i) {
            int32_t startIdx = workspaceGm_.GetValue(i * 2);
            int32_t voxelCnt = workspaceGm_.GetValue(i * 2 + 1);
            while (voxelCnt > 0) {
                int32_t moveCnt = voxelCnt > MOVE_NUM ? MOVE_NUM : voxelCnt;
                voxelCnt -= moveCnt;

                DataCopyParams mvParam(1,
                    static_cast<uint16_t>(
                        AlignUp(static_cast<uint32_t>(moveCnt), B32_DATA_NUM_PER_BLOCK) / B32_DATA_NUM_PER_BLOCK),
                    0, 0);
                DataCopy(inT, uniVoxGm_[startIdx], mvParam);
                SetFlag<HardEvent::MTE2_MTE3>(mte2Event);
                WaitFlag<HardEvent::MTE2_MTE3>(mte2Event);
                DataCopy(uniVoxGm_[totalVoxelCnt], inT, mvParam);
                SetFlag<HardEvent::MTE3_MTE2>(mte3Event);
                WaitFlag<HardEvent::MTE3_MTE2>(mte3Event);
                DataCopy(inT, uniIdxGm_[startIdx], mvParam);
                SetFlag<HardEvent::MTE2_MTE3>(mte2Event);
                WaitFlag<HardEvent::MTE2_MTE3>(mte2Event);
                DataCopy(uniIdxGm_[totalVoxelCnt], inT, mvParam);
                SetFlag<HardEvent::MTE3_MTE2>(mte3Event);
                WaitFlag<HardEvent::MTE3_MTE2>(mte3Event);
                DataCopy(inT, uniArgsortIdxGm_[startIdx], mvParam);
                SetFlag<HardEvent::MTE2_MTE3>(mte2Event);
                WaitFlag<HardEvent::MTE2_MTE3>(mte2Event);
                DataCopy(uniArgsortIdxGm_[totalVoxelCnt], inT, mvParam);
                SetFlag<HardEvent::MTE3_MTE2>(mte3Event);
                WaitFlag<HardEvent::MTE3_MTE2>(mte3Event);

                totalVoxelCnt += moveCnt;
                startIdx += moveCnt;
            }
        }
        inT.SetValue(0, totalVoxelCnt);
        DataCopyPad(voxNumGm_, inT, cpOneIntParam_);
    }
}

extern "C" __global__ __aicore__ void unique_voxel(GM_ADDR voxels, GM_ADDR idxs, GM_ADDR argsortIdxs, GM_ADDR uniVoxs,
    GM_ADDR uniIdxs, GM_ADDR uniArgsortIdxs, GM_ADDR voxNum, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    UniqueVoxelKernel op(voxels, idxs, argsortIdxs, uniVoxs, uniIdxs, uniArgsortIdxs, voxNum, workspace, tilingData);
    op.Process();
}
