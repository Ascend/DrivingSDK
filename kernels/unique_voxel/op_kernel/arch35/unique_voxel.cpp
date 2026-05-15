// Copyright (c) 2024 Huawei Technologies Co., Ltd

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t MOVE_BYTE = 16 * 1024;
constexpr int32_t MOVE_NUM = MOVE_BYTE / B32_BYTE_SIZE;

__aicore__ inline void GatherUniqueVoxel(LocalTensor<int32_t> voxA, LocalTensor<int32_t> voxB,
    LocalTensor<int32_t> idxT, LocalTensor<int32_t> argT, LocalTensor<uint32_t> gatherMask, LocalTensor<int32_t> uniVox,
    LocalTensor<int32_t> uniIdx, LocalTensor<int32_t> uniArg, uint32_t count, uint16_t oneRepeatSize,
    uint16_t repeatTimes) {
    __local_mem__ int32_t *voxAPtr = (__local_mem__ int32_t *)voxA.GetPhyAddr();
    __local_mem__ int32_t *voxBPtr = (__local_mem__ int32_t *)voxB.GetPhyAddr();
    __local_mem__ int32_t *idxTPtr = (__local_mem__ int32_t *)idxT.GetPhyAddr();
    __local_mem__ int32_t *argTPtr = (__local_mem__ int32_t *)argT.GetPhyAddr();
    __local_mem__ int32_t *uniVoxPtr = (__local_mem__ int32_t *)uniVox.GetPhyAddr();
    __local_mem__ int32_t *uniIdxPtr = (__local_mem__ int32_t *)uniIdx.GetPhyAddr();
    __local_mem__ int32_t *uniArgPtr = (__local_mem__ int32_t *)uniArg.GetPhyAddr();
    __local_mem__ uint32_t *gatherMaskPtr = (__local_mem__ uint32_t *)gatherMask.GetPhyAddr();

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<int32_t> rVoxA;
        MicroAPI::RegTensor<int32_t> rVoxB;
        MicroAPI::RegTensor<int32_t> rDiff;
        MicroAPI::RegTensor<int32_t> rIdxT;
        MicroAPI::RegTensor<int32_t> rArgT;
        MicroAPI::RegTensor<int32_t> rUniVox;
        MicroAPI::RegTensor<int32_t> rUniIdx;
        MicroAPI::RegTensor<int32_t> rUniArg;
        MicroAPI::MaskReg maskReg;
        MicroAPI::MaskReg cmpMaskReg;
        MicroAPI::UnalignReg uUniVoxReg;
        MicroAPI::UnalignReg uUniIdxReg;
        MicroAPI::UnalignReg uUniArgReg;

        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskReg = MicroAPI::UpdateMask<uint32_t>(count);
            MicroAPI::AddrReg aReg = MicroAPI::CreateAddrReg<uint32_t>(i, ONE_BLK_SIZE);
            MicroAPI::DataCopy(rVoxA, voxAPtr + i * oneRepeatSize);
            MicroAPI::DataCopy(rVoxB, voxBPtr + i * oneRepeatSize);

            MicroAPI::Sub<int32_t>(rDiff, rVoxA, rVoxB, maskReg);
            MicroAPI::CompareScalar<int32_t, CMPMODE::GT>(cmpMaskReg, rDiff, 0, maskReg);
            MicroAPI::DataCopy(gatherMaskPtr, cmpMaskReg, aReg);
        }

        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        MicroAPI::ClearSpr<SpecialPurposeReg::AR>();

        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::AddrReg aReg = MicroAPI::CreateAddrReg<uint32_t>(i, ONE_BLK_SIZE);
            MicroAPI::DataCopy(cmpMaskReg, gatherMaskPtr, aReg);
            MicroAPI::DataCopy(rVoxA, voxAPtr + i * oneRepeatSize);
            MicroAPI::GatherMask<int32_t, MicroAPI::GatherMaskMode::STORE_REG>(rUniVox, rVoxA, cmpMaskReg);
            MicroAPI::DataCopyUnAlign(uniVoxPtr, rUniVox, uUniVoxReg);
        }

        MicroAPI::DataCopyUnAlignPost(uniVoxPtr, uUniVoxReg);
        MicroAPI::ClearSpr<SpecialPurposeReg::AR>();

        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::AddrReg aReg = MicroAPI::CreateAddrReg<uint32_t>(i, ONE_BLK_SIZE);
            MicroAPI::DataCopy(cmpMaskReg, gatherMaskPtr, aReg);
            MicroAPI::DataCopy(rIdxT, idxTPtr + i * oneRepeatSize);
            MicroAPI::GatherMask<int32_t, MicroAPI::GatherMaskMode::STORE_REG>(rUniIdx, rIdxT, cmpMaskReg);
            MicroAPI::DataCopyUnAlign(uniIdxPtr, rUniIdx, uUniIdxReg);
        }

        MicroAPI::DataCopyUnAlignPost(uniIdxPtr, uUniIdxReg);
        MicroAPI::ClearSpr<SpecialPurposeReg::AR>();

        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::AddrReg aReg = MicroAPI::CreateAddrReg<uint32_t>(i, ONE_BLK_SIZE);
            MicroAPI::DataCopy(cmpMaskReg, gatherMaskPtr, aReg);
            MicroAPI::DataCopy(rArgT, argTPtr + i * oneRepeatSize);
            MicroAPI::GatherMask<int32_t, MicroAPI::GatherMaskMode::STORE_REG>(rUniArg, rArgT, cmpMaskReg);
            MicroAPI::DataCopyUnAlign(uniArgPtr, rUniArg, uUniArgReg);
        }

        MicroAPI::DataCopyUnAlignPost(uniArgPtr, uUniArgReg);
    }
}

class UniqueVoxelKernel {
  public:
    __aicore__ inline UniqueVoxelKernel() = delete;
    __aicore__ inline ~UniqueVoxelKernel() = default;
    __aicore__ inline UniqueVoxelKernel(GM_ADDR voxels, GM_ADDR idxs, GM_ADDR argsortIdxs, GM_ADDR uniVoxs,
        GM_ADDR uniIdxs, GM_ADDR uniArgsortIdxs, GM_ADDR voxNum, GM_ADDR workspace, const UniqueVoxelTilingData &tiling)
        : blkIdx_(GetBlockIdx()), usedBlkNum_(tiling.usedBlkNum), avgTasks_(tiling.avgTasks),
          tailTasks_(tiling.tailTasks), totalTasks_(tiling.totalTasks), avgPts_(tiling.avgPts),
          tailPts_(tiling.tailPts) {
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

        cpParam_.blockLen = static_cast<uint16_t>(avgPts_ / B32_DATA_NUM_PER_BLOCK);
        cpExtParam_.blockLen = static_cast<uint32_t>(tailPts_ * B32_BYTE_SIZE);
        padParam_.rightPadding = static_cast<uint8_t>(AlignUp(tailPts_, B32_DATA_NUM_PER_BLOCK) - tailPts_);

        // init global memory
        voxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(voxels));
        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(idxs));
        argsortIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(argsortIdxs));
        uniVoxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(uniVoxs));
        uniIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(uniIdxs));
        uniArgsortIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(uniArgsortIdxs));
        voxNumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(voxNum));

        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace));
        uint64_t gmOffset = AlignUp(usedBlkNum_ * 2, ONE_REPEAT_FLOAT_SIZE);
        tmpUniVoxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace) + gmOffset);
        gmOffset += AlignUp(tiling.totalPts + 1, ONE_REPEAT_FLOAT_SIZE);
        tmpUniIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace) + gmOffset);
        gmOffset += AlignUp(tiling.totalPts + 1, ONE_REPEAT_FLOAT_SIZE);
        tmpUniArgsortIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace) + gmOffset);

        // init buffer
        int32_t buffer_used = 4; // vox1, vox2, idx, arg
        pipe_.InitBuffer(inQue_, BUFFER_NUM, avgPts_ * buffer_used * B32_BYTE_SIZE);
        pipe_.InitBuffer(cntQue_, BUFFER_NUM, ONE_BLK_SIZE);
        pipe_.InitBuffer(maskBuf_, avgPts_);

        vecEvent_ = pipe_.AllocEventID<HardEvent::V_MTE2>();
        SetVectorMask<float>(FULL_MASK, FULL_MASK);
    }

    __aicore__ inline void Process();

  private:
    TPipe pipe_;
    GlobalTensor<int32_t> voxGm_, idxGm_, argsortIdxGm_, uniVoxGm_, uniIdxGm_, uniArgsortIdxGm_, voxNumGm_;
    GlobalTensor<int32_t> workspaceGm_, tmpUniVoxGm_, tmpUniIdxGm_, tmpUniArgsortIdxGm_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> cntQue_;
    TBuf<TPosition::VECCALC> maskBuf_;
    int32_t blkIdx_, usedBlkNum_;
    int32_t curTaskIdx_, curPtsIdx_, curOutputIdx_, startOutputIdx_;
    int32_t avgTasks_, tailTasks_, totalTasks_, coreTasks_;
    int32_t avgPts_, tailPts_; // here, avgPts_ must be multiple of 64
    int32_t adjOffset_, idxOffset_, argOffset_;
    DataCopyParams cpParam_;
    DataCopyExtParams cpExtParam_, cpOneIntParam_{1, 4, 0, 0, 0}, cpDoubleIntsParam_{1, 8, 0, 0, 0};
    DataCopyPadExtParams<int32_t> padParam_{true, 0, 0, -1};
    UnaryRepeatParams unRptParam_{1, 1, 8, 8};
    BinaryRepeatParams binRptParam_{1, 1, 1, 8, 8, 8};
    GatherMaskParams gatherMaskParam_{1, 1, 8, 1};
    uint8_t rptTimes_;
    uint64_t voxCnt_{0};
    int32_t headVox_, headArgsortIdx_;
    bool hasHeadVox_;

    TEventID vecEvent_;

  private:
    __aicore__ inline bool IsFirstTask() const { return curTaskIdx_ == 0; }

    __aicore__ inline bool IsLastTask() const { return curTaskIdx_ == totalTasks_ - 1; }

    template <bool is_head, bool is_tail> __aicore__ inline void DoProcess();

    template <bool is_head, bool is_tail> __aicore__ inline void Compute();

    template <bool is_tail> __aicore__ inline void CopyIn();

    template <bool is_head> __aicore__ inline void CopyOut();

    __aicore__ inline void CopyVoxel();

    __aicore__ inline void CompactOutput();
};

__aicore__ inline void UniqueVoxelKernel::Process() {
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

template <bool is_head, bool is_tail> __aicore__ inline void UniqueVoxelKernel::DoProcess() {
    CopyIn<is_tail>();
    Compute<is_head, is_tail>();
    CopyOut<is_head>();
}

template <bool is_tail> __aicore__ inline void UniqueVoxelKernel::CopyIn() {
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

template <bool is_head, bool is_tail> __aicore__ inline void UniqueVoxelKernel::Compute() {
    LocalTensor<int32_t> voxT = inQue_.DeQue<int32_t>();
    LocalTensor<int32_t> voxA = voxT[0];
    LocalTensor<int32_t> voxB = voxT[adjOffset_];
    LocalTensor<int32_t> idxT = voxT[idxOffset_];
    LocalTensor<int32_t> argT = voxT[argOffset_];

    LocalTensor<int32_t> uniVox = voxT[adjOffset_];
    LocalTensor<int32_t> uniIdx = voxT[idxOffset_];
    LocalTensor<int32_t> uniArg = voxT[argOffset_];

    LocalTensor<int32_t> rsvCntT = cntQue_.AllocTensor<int32_t>();
    LocalTensor<uint32_t> mask = maskBuf_.AllocTensor<uint32_t>();

    // we need to look at the first element of the voxel
    if (is_head) {
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
        headVox_ = voxB.GetValue(0);
        headArgsortIdx_ = argsortIdxGm_.GetValue(0);
        hasHeadVox_ = headVox_ > -1;
        if (hasHeadVox_) {
            ++voxCnt_;
        }
    }

    if (is_tail) {
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
        voxA.SetValue(tailPts_ - 1, -1);
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
    }

    uint32_t count = is_tail ? tailPts_ : avgPts_;
    uint16_t oneRepeatSize = GetVecLen() / sizeof(int32_t);
    uint16_t repeatTimes = CeilDivision(count, oneRepeatSize);

    GatherUniqueVoxel(voxA, voxB, idxT, argT, mask, uniVox, uniIdx, uniArg, count, oneRepeatSize, repeatTimes);
    uint64_t rsvCnt = GetSpr<SpecialPurposeReg::AR>() / B32_BYTE_SIZE;

    SetVectorMask<float>(FULL_MASK, FULL_MASK);
    voxCnt_ += rsvCnt;
    rsvCntT.SetValue(0, static_cast<int32_t>(rsvCnt));
    cntQue_.EnQue(rsvCntT);
    inQue_.EnQue(voxT);
}

template <bool is_head> __aicore__ inline void UniqueVoxelKernel::CopyOut() {
    LocalTensor<int32_t> voxT = inQue_.DeQue<int32_t>();
    LocalTensor<int32_t> uniVox = voxT[adjOffset_];
    LocalTensor<int32_t> uniIdx = voxT[idxOffset_];
    LocalTensor<int32_t> uniArg = voxT[argOffset_];

    LocalTensor<int32_t> rsvCntT = cntQue_.DeQue<int32_t>();
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
            startOutputIdx_ = 0;
        }
    }

    if (rsvCnt > 0) {
        DataCopyParams mvParam(1,
            static_cast<uint16_t>(
                AlignUp(static_cast<int32_t>(rsvCnt), B32_DATA_NUM_PER_BLOCK) / B32_DATA_NUM_PER_BLOCK),
            0, 0);
        DataCopy(tmpUniVoxGm_[curOutputIdx_], uniVox, mvParam);
        DataCopy(tmpUniIdxGm_[curOutputIdx_], uniIdx, mvParam);
        DataCopy(tmpUniArgsortIdxGm_[curOutputIdx_], uniArg, mvParam);
        PipeBarrier<PIPE_ALL>();

        curOutputIdx_ += rsvCnt;
    }
    cntQue_.FreeTensor(rsvCntT);
    inQue_.FreeTensor(voxT);
}

__aicore__ inline void UniqueVoxelKernel::CopyVoxel() {
    // copy voxel count to workspace
    LocalTensor<int32_t> cntT = cntQue_.AllocTensor<int32_t>();
    cntT.SetValue(0, startOutputIdx_);
    cntT.SetValue(1, static_cast<int32_t>(voxCnt_));
    DataCopyPad(workspaceGm_[blkIdx_ * 2], cntT, cpDoubleIntsParam_);
    cntQue_.FreeTensor(cntT);
}

__aicore__ inline void UniqueVoxelKernel::CompactOutput() {
    TPipe pipe;
    TBuf<TPosition::VECCALC> mvBuf;
    int32_t xyz_buffer_num = 3;
    int32_t double_buffer = 2;
    pipe.InitBuffer(mvBuf, double_buffer * xyz_buffer_num * MOVE_BYTE);

    LocalTensor<int32_t> inT = mvBuf.Get<int32_t>();

    int32_t startIdx = workspaceGm_.GetValue(blkIdx_ * 2);
    int32_t voxelCnt = workspaceGm_.GetValue(blkIdx_ * 2 + 1);
    int32_t totalVoxelCnt = 0;
    for (int32_t i = 0; i < blkIdx_; ++i) {
        int32_t cntIdx = 2 * i + 1;
        totalVoxelCnt += workspaceGm_.GetValue(cntIdx);
    }

    if (hasHeadVox_ && (blkIdx_ == 0)) {
        totalVoxelCnt += 1;
        voxelCnt -= 1;
    }

    SetFlag<HardEvent::MTE3_MTE2>(0);
    SetFlag<HardEvent::MTE3_MTE2>(1);

    int32_t bufferIdx = 0;
    int32_t voxOffset = 0;
    int32_t idxOffset = 1;
    int32_t argOffset = 2;

    while (voxelCnt > 0) {
        LocalTensor<int32_t> voxT = inT[bufferIdx * xyz_buffer_num * MOVE_NUM];
        LocalTensor<int32_t> uniVox = voxT[voxOffset * MOVE_NUM];
        LocalTensor<int32_t> uniIdx = voxT[idxOffset * MOVE_NUM];
        LocalTensor<int32_t> uniArg = voxT[argOffset * MOVE_NUM];

        int32_t moveCnt = voxelCnt > MOVE_NUM ? MOVE_NUM : voxelCnt;
        DataCopyExtParams mvParam(1, moveCnt * sizeof(int32_t), 0, 0, 0);

        WaitFlag<HardEvent::MTE3_MTE2>(bufferIdx);
        DataCopyPad(uniVox, tmpUniVoxGm_[startIdx], mvParam, {});
        DataCopyPad(uniIdx, tmpUniIdxGm_[startIdx], mvParam, {});
        DataCopyPad(uniArg, tmpUniArgsortIdxGm_[startIdx], mvParam, {});
        SetFlag<HardEvent::MTE2_MTE3>(bufferIdx);
        WaitFlag<HardEvent::MTE2_MTE3>(bufferIdx);
        DataCopyPad(uniVoxGm_[totalVoxelCnt], uniVox, mvParam);
        DataCopyPad(uniIdxGm_[totalVoxelCnt], uniIdx, mvParam);
        DataCopyPad(uniArgsortIdxGm_[totalVoxelCnt], uniArg, mvParam);
        SetFlag<HardEvent::MTE3_MTE2>(bufferIdx);

        voxelCnt -= moveCnt;
        totalVoxelCnt += moveCnt;
        startIdx += moveCnt;
        bufferIdx = 1 - bufferIdx;
    }

    WaitFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(1);

    PipeBarrier<PIPE_ALL>();

    if (blkIdx_ == usedBlkNum_ - 1) {
        inT.SetValue(0, totalVoxelCnt);
        DataCopyPad(voxNumGm_, inT, cpOneIntParam_);
    }
}

extern "C" __global__ __aicore__ void unique_voxel(GM_ADDR voxels, GM_ADDR idxs, GM_ADDR argsortIdxs, GM_ADDR uniVoxs,
    GM_ADDR uniIdxs, GM_ADDR uniArgsortIdxs, GM_ADDR voxNum, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    UniqueVoxelKernel op(voxels, idxs, argsortIdxs, uniVoxs, uniIdxs, uniArgsortIdxs, voxNum, workspace, tilingData);
    op.Process();
}
