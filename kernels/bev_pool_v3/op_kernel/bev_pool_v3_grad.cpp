#include "kernel_operator.h"
using namespace AscendC;

// 00000001 00000001 00000001 00000001
static constexpr uint32_t PATTERN8_0 = 16843009;
// feature\depth\bev
static constexpr uint32_t RANK_KIND = 3;
static constexpr uint32_t DOUBLE_BUFFER = 2;

template <bool with_depth> class BEVPoolV3GradKernel {
  public:
    __aicore__ inline BEVPoolV3GradKernel() = delete;

    __aicore__ inline ~BEVPoolV3GradKernel() = default;

    __aicore__ inline BEVPoolV3GradKernel(TPipe *pipe, GM_ADDR gradOut, GM_ADDR depth, GM_ADDR feat, GM_ADDR ranksDepth,
        GM_ADDR ranksFeat, GM_ADDR ranksBev, GM_ADDR gradDepth, GM_ADDR gradFeat, const BEVPoolGradV3TilingData &tiling)
        : pipe_(pipe), blkIdx_(GetBlockIdx()), channel_(tiling.channel) {
        InitTask(tiling);
        InitOffset();
        InitGM(gradOut, depth, feat, ranksDepth, ranksFeat, ranksBev, gradDepth, gradFeat);
        InitBuffer();
    }

    __aicore__ inline void Process();
    __aicore__ inline void ProcessWithoutDepth();

  private:
    __aicore__ inline void InitTask(const BEVPoolGradV3TilingData &tiling) {
        int32_t avgTaskNum = tiling.avgTaskNum;
        int32_t tailTaskNum = tiling.tailTaskNum;
        totalTaskNum_ = tiling.totalTaskNum;
        avgRankNum_ = tiling.avgRankNum;
        tailRankNum_ = tiling.tailRankNum;
        if (blkIdx_ < tailTaskNum) {
            taskStartIdx_ = blkIdx_ * (avgTaskNum + 1);
            taskEndIdx_ = taskStartIdx_ + avgTaskNum + 1;
        } else {
            taskStartIdx_ = blkIdx_ * avgTaskNum + tailTaskNum;
            taskEndIdx_ = taskStartIdx_ + avgTaskNum;
        }

        RANK_STEP = tiling.rankStep;
    }

    __aicore__ inline void InitOffset() {
        rankSize_ = AlignUp(avgRankNum_, B32_DATA_NUM_PER_BLOCK);
        rankBevOffset_ = 0;

        if (with_depth) {
            rankFeatOffset_ = rankBevOffset_ + rankSize_;
            rankDepthOffset_ = rankFeatOffset_ + rankSize_;
            rankBatchSize_ = rankSize_ * RANK_KIND;

            inFeatOffset_ = B32_DATA_NUM_PER_BLOCK;
            inBevOffset_ = inFeatOffset_ + channel_;

            srcS[0] = RANK_STEP;
            srcS[1] = 1;
            dstS[0] = RANK_STEP;
            dstS[1] = channel_;

            srcSDepth[0] = RANK_STEP;
            srcSDepth[1] = 1;
            dstSDepth[0] = RANK_STEP;
            dstSDepth[1] = B32_DATA_NUM_PER_BLOCK;

            batchChannel_ = RANK_STEP * channel_;

            sumParams = {RANK_STEP, static_cast<uint32_t>(channel_), static_cast<uint32_t>(channel_)};
        }
    }

    __aicore__ inline void InitGM(GM_ADDR gradOut, GM_ADDR depth, GM_ADDR feat, GM_ADDR ranksDepth, GM_ADDR ranksFeat,
        GM_ADDR ranksBev, GM_ADDR gradDepth, GM_ADDR gradFeat) {
        gradOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gradOut));
        featGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(feat));
        ranksBevGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ranksBev));
        gradFeatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gradFeat));
        if (with_depth) {
            depthGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(depth));
            ranksDepthGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ranksDepth));
            ranksFeatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ranksFeat));
            gradDepthGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(gradDepth));
        }
    }

    __aicore__ inline void InitBuffer() {
        if (with_depth) {
            pipe_->InitBuffer(ranksBuf_, DOUBLE_BUFFER * rankBatchSize_ * sizeof(int32_t));
            pipe_->InitBuffer(featBuf_, DOUBLE_BUFFER * batchChannel_ * sizeof(float));
            pipe_->InitBuffer(gradOutBuf_, DOUBLE_BUFFER * batchChannel_ * sizeof(float));
            pipe_->InitBuffer(gradFeatBuf_, DOUBLE_BUFFER * batchChannel_ * sizeof(float));
            pipe_->InitBuffer(depthCopyTmpBuf_, DOUBLE_BUFFER * RANK_STEP * B32_DATA_NUM_PER_BLOCK * sizeof(float));

            pipe_->InitBuffer(depthBuf_, batchChannel_ * sizeof(float));
            pipe_->InitBuffer(depthGatherTmpBuf_, RANK_STEP * B32_DATA_NUM_PER_BLOCK * sizeof(float));
            pipe_->InitBuffer(patternBuf_, RANK_STEP * B32_DATA_NUM_PER_BLOCK * sizeof(uint32_t));

            pipe_->InitBuffer(gradTempDepthBuf_, batchChannel_ * sizeof(float));
            pipe_->InitBuffer(
                gradDepthBuf_, (RANK_STEP + DOUBLE_BUFFER * RANK_STEP * B32_DATA_NUM_PER_BLOCK) * sizeof(float));

            ranksBev_ = ranksBuf_.Get<int32_t>();
            ranksFeat_ = ranksBev_[rankSize_];
            ranksDepth_ = ranksFeat_[rankSize_];

            featLocal_ = featBuf_.Get<float>();
            depthLocal_ = depthBuf_.Get<float>();
            depthTmp_ = depthCopyTmpBuf_.Get<float>();
            depthGather_ = depthGatherTmpBuf_.Get<float>();

            gradOutLocal_ = gradOutBuf_.Get<float>();

            gradFeatLocal_ = gradFeatBuf_.Get<float>();
            gradDepthLocal_ = gradDepthBuf_.Get<float>();
            gradDepthBroad_ = gradDepthLocal_[RANK_STEP];
            gradTempDepthLocal_ = gradTempDepthBuf_.Get<float>();

            patternLocal_ = patternBuf_.Get<uint32_t>();
            Duplicate(patternLocal_, PATTERN8_0, RANK_STEP * B32_DATA_NUM_PER_BLOCK);
        } else {
            pipe_->InitBuffer(ranksQue_, DOUBLE_BUFFER, rankSize_ * sizeof(int32_t));
            pipe_->InitBuffer(inQue_, DOUBLE_BUFFER, rankSize_ * channel_ * sizeof(int32_t));

            cpInEvtID_ = pipe_->FetchEventID(HardEvent::MTE2_MTE3);
            cpOutEvtID_ = pipe_->FetchEventID(HardEvent::MTE3_MTE2);
        }
    }

    // handel gradDepth
    __aicore__ inline void CopyInStage0(uint8_t ping, uint32_t step, uint64_t off, uint64_t featOff);
    __aicore__ inline void ComputeStage0(uint8_t ping, uint32_t step, uint64_t featOff);
    __aicore__ inline void CopyOutStage0(uint8_t ping, uint32_t step, uint64_t off);
    // handel gradFeat
    __aicore__ inline void CopyInStage1(uint8_t ping, uint32_t step, uint64_t off, uint64_t depOff);
    __aicore__ inline void ComputeStage1(uint8_t ping, uint32_t step, uint64_t featOff, uint64_t depOff);
    __aicore__ inline void CopyOutStage1(uint8_t ping, uint32_t step, uint64_t off, uint64_t featOff);
    __aicore__ inline void ProcessDualRank(
        int32_t rankOffset, uint32_t step0, uint32_t step1, uint64_t featOff1, uint64_t depOff1, int32_t idx);
    __aicore__ inline void ProcessSingle(uint8_t ping, uint64_t taskIdx, uint32_t rankNum);

    __aicore__ inline void ProcessSingleWithoutDepth(uint64_t taskIdx, uint32_t rankNum);

  private:
    TPipe *pipe_;
    int32_t blkIdx_;
    GlobalTensor<float> gradOutGm_, depthGm_, featGm_, gradDepthGm_, gradFeatGm_;
    GlobalTensor<int32_t> ranksDepthGm_, ranksFeatGm_, ranksBevGm_;
    TQue<TPosition::VECIN, DOUBLE_BUFFER> ranksQue_;
    TQue<TPosition::VECIN, DOUBLE_BUFFER> inQue_;
    TQue<TPosition::VECOUT, DOUBLE_BUFFER> outQue_;

    LocalTensor<int32_t> ranksBev_, ranksFeat_, ranksDepth_;
    LocalTensor<float> featLocal_, depthLocal_, depthTmp_, depthGather_, out_, gradOutLocal_;
    LocalTensor<float> gradFeatLocal_, gradTempDepthLocal_, gradDepthLocal_, gradDepthBroad_;
    LocalTensor<uint32_t> patternLocal_;

    uint32_t taskStartIdx_, taskEndIdx_, totalTaskNum_;
    int32_t channel_, batchChannel_;
    uint32_t rankSize_, avgRankNum_, tailRankNum_, rankBatchSize_;
    uint64_t rankDepthOffset_, rankFeatOffset_, rankBevOffset_, inFeatOffset_, inBevOffset_;

    uint32_t RANK_STEP;
    uint32_t srcS[2], dstS[2];
    uint32_t srcSDepth[2], dstSDepth[2];
    SumParams sumParams;
    DataCopyParams cpSingleParams_{1, B32_BYTE_SIZE, 0, 0};
    TEventID cpInEvtID_, cpOutEvtID_;

    TBuf<TPosition::VECCALC> ranksBuf_, featBuf_, depthBuf_, depthGatherTmpBuf_, depthCopyTmpBuf_, patternBuf_, outBuf_,
        gradOutBuf_, gradFeatBuf_, gradTempDepthBuf_, gradDepthBuf_;
};

template <bool with_depth>
__aicore__ inline void BEVPoolV3GradKernel<with_depth>::CopyInStage0(
    uint8_t ping, uint32_t step, uint64_t off, uint64_t featOff) {
    WaitFlag<HardEvent::V_MTE2>(ping + 4);
    for (int32_t j = 0; j < step; j++) {
        uint64_t rf = ranksFeat_.GetValue(off + j);
        uint64_t rb = ranksBev_.GetValue(off + j);
        DataCopy(featLocal_[featOff + j * channel_], featGm_[rf], channel_);
        DataCopy(gradOutLocal_[featOff + j * channel_], gradOutGm_[rb], channel_);
    }
    SetFlag<HardEvent::MTE2_V>(ping + 4);
}

template <bool with_depth>
__aicore__ inline void BEVPoolV3GradKernel<with_depth>::ComputeStage0(uint8_t ping, uint32_t step, uint64_t featOff) {
    LocalTensor<float> broadTmp = gradDepthBroad_[ping * RANK_STEP * B32_DATA_NUM_PER_BLOCK];
    WaitFlag<HardEvent::MTE2_V>(ping + 4);
    Mul(gradTempDepthLocal_, gradOutLocal_[featOff], featLocal_[featOff], channel_ * step);
    Sum(gradDepthLocal_, gradTempDepthLocal_, {step, static_cast<uint32_t>(channel_), static_cast<uint32_t>(channel_)});

    WaitFlag<HardEvent::MTE3_V>(ping);
    BroadCast<float, 2, 1>(broadTmp, gradDepthLocal_, dstSDepth, srcSDepth);
    SetFlag<HardEvent::V_MTE3>(ping);
}

template <bool with_depth>
__aicore__ inline void BEVPoolV3GradKernel<with_depth>::CopyOutStage0(uint8_t ping, uint32_t step, uint64_t off) {
    LocalTensor<float> broadTmp = gradDepthBroad_[ping * RANK_STEP * B32_DATA_NUM_PER_BLOCK];
    WaitFlag<HardEvent::V_MTE3>(ping);
    SetAtomicAdd<float>();
    for (int32_t j = 0; j < step; j++) {
        uint64_t rd = ranksDepth_.GetValue(off + j);
        DataCopyPad(gradDepthGm_[rd], broadTmp[j * B32_DATA_NUM_PER_BLOCK], cpSingleParams_);
    }
    SetAtomicNone();
    SetFlag<HardEvent::MTE3_V>(ping);
}

template <bool with_depth>
__aicore__ inline void BEVPoolV3GradKernel<with_depth>::CopyInStage1(
    uint8_t ping, uint32_t step, uint64_t off, uint64_t depOff) {
    WaitFlag<HardEvent::V_MTE2>(ping + 6);
    for (int32_t j = 0; j < step; j++) {
        uint64_t rd = ranksDepth_.GetValue(off + j);
        DataCopy(depthTmp_[depOff + j * B32_DATA_NUM_PER_BLOCK], depthGm_[rd], B32_DATA_NUM_PER_BLOCK);
    }
    SetFlag<HardEvent::MTE2_V>(ping + 6);
}

template <bool with_depth>
__aicore__ inline void BEVPoolV3GradKernel<with_depth>::ComputeStage1(
    uint8_t ping, uint32_t step, uint64_t featOff, uint64_t depOff) {
    WaitFlag<HardEvent::MTE2_V>(ping + 6);

    uint64_t rsvdCnt = 0;
    GatherMask(depthGather_, depthTmp_[depOff], patternLocal_, false, 0,
        {1, static_cast<uint16_t>(RANK_STEP / 8), 8, 0}, rsvdCnt);
    BroadCast<float, 2, 1>(depthLocal_, depthGather_, dstS, srcS);

    WaitFlag<HardEvent::MTE3_V>(ping + 2);
    Mul(gradFeatLocal_[featOff], gradOutLocal_[featOff], depthLocal_, channel_ * step);
    SetFlag<HardEvent::V_MTE3>(ping + 2);

    SetFlag<HardEvent::V_MTE2>(ping + 4);
    SetFlag<HardEvent::V_MTE2>(ping + 6);
}

template <bool with_depth>
__aicore__ inline void BEVPoolV3GradKernel<with_depth>::CopyOutStage1(
    uint8_t ping, uint32_t step, uint64_t off, uint64_t featOff) {
    WaitFlag<HardEvent::V_MTE3>(ping + 2);
    SetAtomicAdd<float>();
    for (int32_t j = 0; j < step; j++) {
        uint64_t rf = ranksFeat_.GetValue(off + j);
        DataCopy(gradFeatGm_[rf], gradFeatLocal_[featOff + j * channel_], channel_);
    }
    SetAtomicNone();
    SetFlag<HardEvent::MTE3_V>(ping + 2);
}

template <bool with_depth>
__aicore__ inline void BEVPoolV3GradKernel<with_depth>::ProcessDualRank(
    int32_t rankOffset, uint32_t step0, uint32_t step1, uint64_t featOff1, uint64_t depOff1, int32_t idx) {
    uint64_t off0 = rankOffset + idx * RANK_STEP;
    uint64_t off1 = off0 + RANK_STEP;
    CopyInStage0(0, step0, off0, 0);
    CopyInStage0(1, step1, off1, featOff1);

    ComputeStage0(0, step0, 0);
    CopyInStage1(0, step0, off0, 0);
    CopyOutStage0(0, step0, off0);
    ComputeStage1(0, step0, 0, 0);
    CopyInStage1(1, step1, off1, depOff1);
    CopyOutStage1(0, step0, off0, 0);

    ComputeStage0(1, step1, featOff1);
    CopyOutStage0(1, step1, off1);
    ComputeStage1(1, step1, featOff1, depOff1);
    CopyOutStage1(1, step1, off1, featOff1);
}

template <bool with_depth>
__aicore__ inline void BEVPoolV3GradKernel<with_depth>::ProcessSingle(
    uint8_t ping, uint64_t taskIdx, uint32_t actualRankNum) {
    int32_t rankNum = AlignUp(actualRankNum, B32_DATA_NUM_PER_BLOCK);
    int32_t rankOffset = ping * rankBatchSize_;
    WaitFlag<HardEvent::V_MTE2>(ping);
    DataCopy(ranksBev_[rankOffset], ranksBevGm_[taskIdx * avgRankNum_], rankNum);
    DataCopy(ranksFeat_[rankOffset], ranksFeatGm_[taskIdx * avgRankNum_], rankNum);
    SetFlag<HardEvent::MTE2_V>(ping);

    WaitFlag<HardEvent::MTE2_V>(ping);
    // 2 * rankSize_ -> ranksBev_、ranksFeat_
    Muls(ranksBev_[rankOffset], ranksBev_[rankOffset], channel_, 2 * rankSize_);

    WaitFlag<HardEvent::V_MTE2>(ping + 2);
    DataCopy(ranksDepth_[rankOffset], ranksDepthGm_[taskIdx * avgRankNum_], rankNum);

    uint32_t cnt = (actualRankNum + RANK_STEP - 1) / RANK_STEP;
    uint32_t tail = actualRankNum - (cnt - 1) * RANK_STEP;
    uint32_t step0 = RANK_STEP;
    uint32_t step1 = RANK_STEP;
    uint64_t featOff1 = batchChannel_;
    uint64_t depOff1 = RANK_STEP * B32_DATA_NUM_PER_BLOCK;
    if ((cnt % 2) == 0) {
        for (int32_t i = 0; i < cnt; i += 2) {
            if (unlikely(i == cnt - 2)) {
                step1 = tail;
            }
            ProcessDualRank(rankOffset, step0, step1, featOff1, depOff1, i);
        }
    } else {
        for (int32_t i = 0; i < cnt - 1; i += 2) {
            ProcessDualRank(rankOffset, step0, step1, featOff1, depOff1, i);
        }
        {
            step0 = tail;
            int32_t i = cnt - 1;
            uint64_t off0 = rankOffset + i * RANK_STEP;
            CopyInStage0(0, step0, off0, 0);
            ComputeStage0(0, step0, 0);
            CopyInStage1(0, step0, off0, 0);
            CopyOutStage0(0, step0, off0);
            ComputeStage1(0, step0, 0, 0);
            CopyOutStage1(0, step0, off0, 0);
        }
    }

    SetFlag<HardEvent::V_MTE2>(ping);
    SetFlag<HardEvent::V_MTE2>(ping + 2);
}

template <bool with_depth> __aicore__ inline void BEVPoolV3GradKernel<with_depth>::Process() {
    uint8_t ping = 0;
    SetFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE2>(1);
    SetFlag<HardEvent::V_MTE2>(2);
    SetFlag<HardEvent::V_MTE2>(3);
    SetFlag<HardEvent::V_MTE2>(4);
    SetFlag<HardEvent::V_MTE2>(5);
    SetFlag<HardEvent::V_MTE2>(6);
    SetFlag<HardEvent::V_MTE2>(7);

    SetFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::MTE3_V>(1);
    SetFlag<HardEvent::MTE3_V>(2);
    SetFlag<HardEvent::MTE3_V>(3);

    for (uint32_t i = taskStartIdx_; i < taskEndIdx_; ++i) {
        uint32_t actualRankNum = avgRankNum_;
        if (unlikely(i == totalTaskNum_ - 1)) {
            actualRankNum = tailRankNum_;
        }
        ProcessSingle(ping, i, actualRankNum);
        ping = 1 - ping;
    }

    WaitFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(1);
    WaitFlag<HardEvent::V_MTE2>(2);
    WaitFlag<HardEvent::V_MTE2>(3);
    WaitFlag<HardEvent::V_MTE2>(4);
    WaitFlag<HardEvent::V_MTE2>(5);
    WaitFlag<HardEvent::V_MTE2>(6);
    WaitFlag<HardEvent::V_MTE2>(7);

    WaitFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(1);
    WaitFlag<HardEvent::MTE3_V>(2);
    WaitFlag<HardEvent::MTE3_V>(3);
}

template <bool with_depth>
__aicore__ inline void BEVPoolV3GradKernel<with_depth>::ProcessSingleWithoutDepth(
    uint64_t taskIdx, uint32_t actualRankNum) {
    int32_t rankNum = AlignUp(actualRankNum, B32_DATA_NUM_PER_BLOCK);
    LocalTensor<int32_t> ranks = ranksQue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> rankBev = ranks[rankBevOffset_];
    DataCopy(rankBev, ranksBevGm_[taskIdx * avgRankNum_], rankNum);

    ranksQue_.EnQue(ranks);
    ranksQue_.DeQue<int32_t>();
    Muls(rankBev, rankBev, channel_, rankNum);
    LocalTensor<float> in = inQue_.AllocTensor<float>();

    for (int32_t i = 0; i < actualRankNum; ++i) {
        DataCopy(in[i * channel_], gradOutGm_[rankBev.GetValue(i)], channel_);
    }
    SetFlag<HardEvent::MTE2_MTE3>(cpInEvtID_);
    WaitFlag<HardEvent::MTE2_MTE3>(cpInEvtID_);
    DataCopy(gradFeatGm_[taskIdx * avgRankNum_ * channel_], in, actualRankNum * channel_);
    SetFlag<HardEvent::MTE3_MTE2>(cpOutEvtID_);
    WaitFlag<HardEvent::MTE3_MTE2>(cpOutEvtID_);
    inQue_.FreeTensor(in);
    ranksQue_.FreeTensor(ranks);
}

template <bool with_depth> __aicore__ inline void BEVPoolV3GradKernel<with_depth>::ProcessWithoutDepth() {
    for (uint32_t i = taskStartIdx_; i < taskEndIdx_; ++i) {
        uint32_t actualRankNum = avgRankNum_;
        if (unlikely(i == totalTaskNum_ - 1)) {
            actualRankNum = tailRankNum_;
        }
        ProcessSingleWithoutDepth(i, actualRankNum);
    }
}

extern "C" __global__ __aicore__ void bev_pool_v3_grad(GM_ADDR gradOut, GM_ADDR depth, GM_ADDR feat, GM_ADDR ranksDepth,
    GM_ADDR ranksFeat, GM_ADDR ranksBev, GM_ADDR gradDepth, GM_ADDR gradFeat, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(bevPoolTiling, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        BEVPoolV3GradKernel<false> kernel(
            &pipe, gradOut, depth, feat, ranksDepth, ranksFeat, ranksBev, gradDepth, gradFeat, bevPoolTiling);
        kernel.ProcessWithoutDepth();
    } else if (TILING_KEY_IS(1)) {
        BEVPoolV3GradKernel<true> kernel(
            &pipe, gradOut, depth, feat, ranksDepth, ranksFeat, ranksBev, gradDepth, gradFeat, bevPoolTiling);
        kernel.Process();
    }
}
