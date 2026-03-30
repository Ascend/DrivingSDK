#include "kernel_operator.h"
using namespace AscendC;

static constexpr uint32_t RANK_STEP = 16;
static constexpr uint32_t PATTERN8_0 = 16843009; // 00000001 00000001 00000001 00000001
static constexpr uint32_t RANK_KIND = 3;
static constexpr uint32_t DOUBLE_BUFFER = 2;

template<bool with_depth>
class BEVPoolV3Kernel {
public:
    __aicore__ inline BEVPoolV3Kernel() = delete;

    __aicore__ inline ~BEVPoolV3Kernel() = default;

    __aicore__ inline BEVPoolV3Kernel(TPipe* pipe, GM_ADDR depth, GM_ADDR feat, GM_ADDR ranksDepth, GM_ADDR ranksFeat,
        GM_ADDR ranksBev, GM_ADDR out, const BEVPoolV3TilingData& tiling)
        : pipe_(pipe), blkIdx_(GetBlockIdx()), channel_(tiling.channel)
    {
        InitTask(tiling);
        InitOffset();
        InitGM(depth, feat, ranksDepth, ranksFeat, ranksBev, out);
        InitBuffer();
    }

    __aicore__ inline void Process();
    __aicore__ inline void ProcessSingleWithoutDepth();

private:
    __aicore__ inline void InitTask(const BEVPoolV3TilingData& tiling)
    {
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
    }

    __aicore__ inline void InitOffset()
    {
        rankSize_ = AlignUp(avgRankNum_, B32_DATA_NUM_PER_BLOCK);
        rankBevOffset_ = 0;
        if (with_depth) {
            rankFeatOffset_ = rankBevOffset_ + rankSize_;
            rankDepthOffset_ = rankFeatOffset_ + rankSize_;
            rankBatchSize_ = rankSize_ * RANK_KIND;
            srcS[0] = RANK_STEP;
            srcS[1] = 1;
            dstS[0] = RANK_STEP;
            dstS[1] = channel_;
        }
    }

    __aicore__ inline void InitGM(
        GM_ADDR depth, GM_ADDR feat, GM_ADDR ranksDepth, GM_ADDR ranksFeat, GM_ADDR ranksBev, GM_ADDR out)
    {
        if (with_depth) {
            depthGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(depth));
            ranksDepthGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ranksDepth));
            ranksFeatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ranksFeat));
        }
        featGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(feat));
        ranksBevGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ranksBev));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
    }

    __aicore__ inline void InitBuffer()
    {
        if (with_depth) {
            pipe_->InitBuffer(ranksBuf_, DOUBLE_BUFFER * rankBatchSize_ * sizeof(int32_t));
            pipe_->InitBuffer(featBuf_, DOUBLE_BUFFER * RANK_STEP * channel_ * sizeof(float));
            pipe_->InitBuffer(depthCopyTmpBuf_, DOUBLE_BUFFER * RANK_STEP * B32_DATA_NUM_PER_BLOCK * sizeof(float));
            pipe_->InitBuffer(outBuf_, DOUBLE_BUFFER * RANK_STEP * channel_ * sizeof(float));
            pipe_->InitBuffer(depthBuf_, RANK_STEP * channel_ * sizeof(float));
            pipe_->InitBuffer(depthGatherTmpBuf_, RANK_STEP * B32_DATA_NUM_PER_BLOCK * sizeof(float));
            pipe_->InitBuffer(patternBuf_,  RANK_STEP * B32_DATA_NUM_PER_BLOCK * sizeof(uint32_t));

            ranksBev_ = ranksBuf_.Get<int32_t>();
            ranksFeat_ = ranksBev_[rankSize_];
            ranksDepth_ = ranksFeat_[rankSize_];

            featLocal_ = featBuf_.Get<float>();
            depthLocal_ = depthBuf_.Get<float>();
            depthTmp_ = depthCopyTmpBuf_.Get<float>();
            depthGather_ = depthGatherTmpBuf_.Get<float>();

            out_ = outBuf_.Get<float>();

            patternLocal_ = patternBuf_.Get<uint32_t>();
            Duplicate(patternLocal_, PATTERN8_0, RANK_STEP * B32_DATA_NUM_PER_BLOCK);
        } else {
            pipe_->InitBuffer(ranksQue_, DOUBLE_BUFFER, rankSize_ * sizeof(int32_t));
            pipe_->InitBuffer(inQue_, DOUBLE_BUFFER, rankSize_ * channel_ * sizeof(int32_t));
            cpInEvtID_ = pipe_->FetchEventID(HardEvent::MTE2_MTE3);
            cpOutEvtID_ = pipe_->FetchEventID(HardEvent::MTE3_MTE2);
        }
    }

    __aicore__ inline void CopyIn(uint8_t ping, uint32_t step, uint64_t off, uint64_t featOff, uint64_t depOff);
    __aicore__ inline void Compute(uint8_t ping, uint32_t step, uint64_t featOff, uint64_t depOff);
    __aicore__ inline void CopyOut(uint8_t ping, uint32_t step, uint64_t off, uint64_t featOff);
    __aicore__ inline void ProcessDualRank(int32_t rankOffset, uint32_t step0, uint32_t step1, uint64_t featOffset1, uint64_t depOffset1, int32_t idx);
    __aicore__ inline void ProcessSingle(uint8_t ping, uint64_t taskIdx, uint32_t rankNum);

    __aicore__ inline void ProcessSingleWithoutDepth(uint64_t taskIdx, uint32_t rankNum);

private:
    TPipe* pipe_;
    int32_t blkIdx_;
    GlobalTensor<float> depthGm_, featGm_, outGm_;
    GlobalTensor<int32_t> ranksDepthGm_, ranksFeatGm_, ranksBevGm_;
    TQue<TPosition::VECIN, 1> ranksQue_;
    TQue<TPosition::VECIN, 2> inQue_;
    TQue<TPosition::VECOUT, 2> outQue_;

    TBuf<TPosition::VECCALC> ranksBuf_, featBuf_, depthBuf_, depthGatherTmpBuf_, depthCopyTmpBuf_, patternBuf_, outBuf_;

    LocalTensor<int32_t> ranksBev_, ranksFeat_, ranksDepth_;
    LocalTensor<float> featLocal_, depthLocal_, depthTmp_, depthGather_, out_;
    LocalTensor<uint32_t> patternLocal_;

    uint32_t taskStartIdx_, taskEndIdx_, totalTaskNum_;
    int32_t channel_;
    uint32_t rankSize_, avgRankNum_, tailRankNum_, rankBatchSize_;
    uint64_t rankDepthOffset_, rankFeatOffset_, rankBevOffset_;
    
    uint32_t srcS[2], dstS[2];
    TEventID cpInEvtID_, cpOutEvtID_;
};


template<bool with_depth>
__aicore__ inline void BEVPoolV3Kernel<with_depth>::CopyIn(uint8_t ping, uint32_t step, uint64_t off, uint64_t featOff, uint64_t depOff)
{
    WaitFlag<HardEvent::V_MTE2>(ping + 4);
    for (int32_t j = 0; j < step; j++) {
        uint64_t rf = ranksFeat_.GetValue(off + j);
        uint64_t rd = ranksDepth_.GetValue(off + j);
        DataCopy(featLocal_[featOff + j * channel_], featGm_[rf], channel_);
        DataCopy(depthTmp_[depOff + j * B32_DATA_NUM_PER_BLOCK], depthGm_[rd], B32_DATA_NUM_PER_BLOCK);
    }
    SetFlag<HardEvent::MTE2_V>(ping + 4);
}

template<bool with_depth>
__aicore__ inline void BEVPoolV3Kernel<with_depth>::Compute(uint8_t ping, uint32_t step, uint64_t featOff, uint64_t depOff)
{
    uint64_t rsvdCnt = 0;
    WaitFlag<HardEvent::MTE2_V>(ping + 4);
    GatherMask(depthGather_, depthTmp_[depOff], patternLocal_, false, 0, { 1, RANK_STEP / 8, 8, 0 }, rsvdCnt);
    BroadCast<float, 2, 1>(depthLocal_, depthGather_, dstS, srcS);

    WaitFlag<HardEvent::MTE3_V>(ping);
    Mul(out_[featOff], featLocal_[featOff], depthLocal_, channel_ * step);
    SetFlag<HardEvent::V_MTE2>(ping + 4);
    SetFlag<HardEvent::V_MTE3>(ping);
}


template<bool with_depth>
__aicore__ inline void BEVPoolV3Kernel<with_depth>::CopyOut(uint8_t ping, uint32_t step, uint64_t off, uint64_t featOff)
{
    WaitFlag<HardEvent::V_MTE3>(ping);
    SetAtomicAdd<float>();
    for (int32_t j = 0; j < step; j++) {
        uint64_t rb = ranksBev_.GetValue(off + j);
        DataCopy(outGm_[rb], out_[featOff + j * channel_], channel_);
    }
    SetAtomicNone();
    SetFlag<HardEvent::MTE3_V>(ping);
}

template<bool with_depth>
__aicore__ inline void BEVPoolV3Kernel<with_depth>::ProcessDualRank(int32_t rankOffset, uint32_t step0, uint32_t step1, uint64_t featOffset1, uint64_t depOffset1, int32_t idx)
{
    uint64_t offset0 = rankOffset + idx * RANK_STEP;
    uint64_t offset1 = offset0 + RANK_STEP;
    CopyIn(0, step0, offset0, 0, 0);
    CopyIn(1, step1, offset1, featOffset1, depOffset1);
    Compute(0, step0, 0, 0);
    Compute(1, step1, featOffset1, depOffset1);
    CopyOut(0, step0, offset0, 0);
    CopyOut(1, step1, offset1, featOffset1);
}

template<bool with_depth>
__aicore__ inline void BEVPoolV3Kernel<with_depth>::ProcessSingle(uint8_t ping, uint64_t taskIdx, uint32_t actualRankNum)
{
    int32_t rankNum = AlignUp(actualRankNum, B32_DATA_NUM_PER_BLOCK);
    int32_t rankOffset = ping * rankBatchSize_;
    int32_t rankOffset0 = 0;
    int32_t rankOffset1 = rankBatchSize_;
    if (ping == 0) {
        {
            WaitFlag<HardEvent::V_MTE2>(0);
            DataCopy(ranksBev_[rankOffset0], ranksBevGm_[taskIdx * avgRankNum_], rankNum);
            DataCopy(ranksFeat_[rankOffset0], ranksFeatGm_[taskIdx * avgRankNum_], rankNum);
            SetFlag<HardEvent::MTE2_V>(0);
            
            WaitFlag<HardEvent::MTE2_V>(0);
            Muls(ranksBev_[rankOffset0], ranksBev_[rankOffset0], channel_, 2 * rankSize_);
            
            WaitFlag<HardEvent::V_MTE2>(2);
            DataCopy(ranksDepth_[rankOffset0], ranksDepthGm_[taskIdx * avgRankNum_], rankNum);
        }
        if (unlikely(taskIdx != taskEndIdx_ - 1)) {
            uint64_t taskIdx1 = taskIdx + 1;
            WaitFlag<HardEvent::V_MTE2>(1);
            DataCopy(ranksBev_[rankOffset1], ranksBevGm_[taskIdx1 * avgRankNum_], rankNum);
            DataCopy(ranksFeat_[rankOffset1], ranksFeatGm_[taskIdx1 * avgRankNum_], rankNum);
            SetFlag<HardEvent::MTE2_V>(1);
            
            WaitFlag<HardEvent::MTE2_V>(1);
            Muls(ranksBev_[rankOffset1], ranksBev_[rankOffset1], channel_, 2 * rankSize_);
            
            WaitFlag<HardEvent::V_MTE2>(3);
            DataCopy(ranksDepth_[rankOffset1], ranksDepthGm_[taskIdx1 * avgRankNum_], rankNum);
        }
    }
    PipeBarrier<PIPE_ALL>();
    uint32_t cnt = (actualRankNum + RANK_STEP - 1) / RANK_STEP;
    uint32_t tail = actualRankNum - (cnt - 1) * RANK_STEP;
    uint32_t step0 = RANK_STEP;
    uint32_t step1 = RANK_STEP;
    uint64_t featOffset1 = RANK_STEP * channel_;
    uint64_t depOffset1 = RANK_STEP * B32_DATA_NUM_PER_BLOCK;
    if ((cnt % 2)  == 0) {
        for (int32_t i = 0; i < cnt; i += 2) {
            if (unlikely(i == cnt - 2)) {
                step1 = tail;
            }
            ProcessDualRank(rankOffset, step0, step1, featOffset1, depOffset1, i);
        }
    } else {
        for (int32_t i = 0; i < cnt - 1; i += 2) {
            ProcessDualRank(rankOffset, step0, step1, featOffset1, depOffset1, i);
        }
        {
            step0 = tail;
            int32_t i =  cnt - 1;
            uint64_t offset0 = rankOffset + i * RANK_STEP;
            CopyIn(0, step0, offset0, 0, 0);
            Compute(0, step0, 0, 0);
            CopyOut(0, step0, offset0, 0);
        }
    }
    SetFlag<HardEvent::V_MTE2>(ping);
    SetFlag<HardEvent::V_MTE2>(ping + 2);
}

template<bool with_depth>
__aicore__ inline void BEVPoolV3Kernel<with_depth>::Process()
{
    uint8_t ping = 0;
    SetFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE2>(1);
    SetFlag<HardEvent::V_MTE2>(2);
    SetFlag<HardEvent::V_MTE2>(3);
    SetFlag<HardEvent::V_MTE2>(4);
    SetFlag<HardEvent::V_MTE2>(5);

    SetFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::MTE3_V>(1);

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

    WaitFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(1);
}

template<bool with_depth>
__aicore__ inline void BEVPoolV3Kernel<with_depth>::ProcessSingleWithoutDepth(uint64_t taskIdx, uint32_t actualRankNum)
{
    int32_t rankNum = AlignUp(actualRankNum, B32_DATA_NUM_PER_BLOCK);

    LocalTensor<int32_t> ranks = ranksQue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> rankBev = ranks[rankBevOffset_];
    DataCopy(rankBev, ranksBevGm_[taskIdx * avgRankNum_], rankNum);
    ranksQue_.EnQue(ranks);
    ranksQue_.DeQue<int32_t>();
    Muls(rankBev, rankBev, channel_, rankNum);
    LocalTensor<float> in = inQue_.AllocTensor<float>();
    DataCopy(in, featGm_[taskIdx * avgRankNum_ * channel_], actualRankNum * channel_);
    SetFlag<HardEvent::MTE2_MTE3>(cpInEvtID_);
    WaitFlag<HardEvent::MTE2_MTE3>(cpInEvtID_);
    for (int32_t i = 0; i < actualRankNum; ++i) {
        SetAtomicAdd<float>();
        DataCopy(outGm_[rankBev.GetValue(i)], in[i * channel_], channel_);
        SetAtomicNone();
    }
    SetFlag<HardEvent::MTE3_MTE2>(cpOutEvtID_);
    WaitFlag<HardEvent::MTE3_MTE2>(cpOutEvtID_);
    inQue_.FreeTensor(in);
    ranksQue_.FreeTensor(ranks);
}

template<bool with_depth>
__aicore__ inline void BEVPoolV3Kernel<with_depth>::ProcessSingleWithoutDepth()
{
    for (uint32_t i = taskStartIdx_; i < taskEndIdx_; ++i) {
        uint32_t actualRankNum = avgRankNum_;
        if (unlikely(i == totalTaskNum_ - 1)) {
            actualRankNum = tailRankNum_;
        }
        ProcessSingleWithoutDepth(i, actualRankNum);
    }
}

extern "C" __global__ __aicore__ void bev_pool_v3(GM_ADDR depth, GM_ADDR feat, GM_ADDR ranksDepth, GM_ADDR ranksFeat,
    GM_ADDR ranksBev, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(bevPoolTiling, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        BEVPoolV3Kernel<false> kernel(&pipe, depth, feat, ranksDepth, ranksFeat, ranksBev, out, bevPoolTiling);
        kernel.ProcessSingleWithoutDepth();
    } else if (TILING_KEY_IS(1)) {
        BEVPoolV3Kernel<true> kernel(&pipe, depth, feat, ranksDepth, ranksFeat, ranksBev, out, bevPoolTiling);
        kernel.Process();
    }
}