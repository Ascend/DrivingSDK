#include "kernel_operator.h"
using namespace AscendC;

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
        InitEvent();
    }

    __aicore__ inline void Process();

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
            pipe_->InitBuffer(ranksQue_, 1, 3 * rankSize_ * sizeof(int32_t));
            pipe_->InitBuffer(inQue_, 2, (B32_DATA_NUM_PER_BLOCK + channel_) * sizeof(int32_t));
            pipe_->InitBuffer(outQue_, 2, channel_ * sizeof(float));
        } else {
            pipe_->InitBuffer(ranksQue_, 2, rankSize_ * sizeof(int32_t));
            pipe_->InitBuffer(inQue_, 2, rankSize_ * channel_ * sizeof(int32_t));
        }
    }

    __aicore__ inline void InitEvent()
    {
        cpInEvtID_ = pipe_->FetchEventID(HardEvent::MTE2_MTE3);
        cpOutEvtID_ = pipe_->FetchEventID(HardEvent::MTE3_MTE2);
    }

    __aicore__ inline void CopyIn(uint64_t rd, uint64_t rf);

    __aicore__ inline void Compute();

    __aicore__ inline void CopyOut(uint64_t rb);

    __aicore__ inline void ProcessSingle(uint64_t taskIdx, uint32_t rankNum);

private:
    TPipe* pipe_;
    int32_t blkIdx_;
    GlobalTensor<float> depthGm_, featGm_, outGm_;
    GlobalTensor<int32_t> ranksDepthGm_, ranksFeatGm_, ranksBevGm_;
    TQue<TPosition::VECIN, 1> ranksQue_;
    TQue<TPosition::VECIN, 2> inQue_;
    TQue<TPosition::VECOUT, 2> outQue_;

    uint32_t taskStartIdx_, taskEndIdx_, totalTaskNum_;
    int32_t channel_;
    uint32_t rankSize_, avgRankNum_, tailRankNum_;
    uint64_t rankDepthOffset_, rankFeatOffset_, rankBevOffset_;

    TEventID cpInEvtID_, cpOutEvtID_;
};

template<bool with_depth>
__aicore__ inline void BEVPoolV3Kernel<with_depth>::CopyIn(uint64_t rd, uint64_t rf)
{
    LocalTensor<float> in = inQue_.AllocTensor<float>();
    DataCopy(in, depthGm_[rd], B32_DATA_NUM_PER_BLOCK);
    DataCopy(in[8], featGm_[rf], channel_);
    inQue_.EnQue(in);
}

template<bool with_depth>
__aicore__ inline void BEVPoolV3Kernel<with_depth>::Compute()
{
    LocalTensor<float> in = inQue_.DeQue<float>();
    LocalTensor<float> out = outQue_.AllocTensor<float>();
    Muls(out, in[8], in.GetValue(0), channel_);
    inQue_.FreeTensor(in);
    outQue_.EnQue(out);
}

template<bool with_depth>
__aicore__ inline void BEVPoolV3Kernel<with_depth>::CopyOut(uint64_t rb)
{
    LocalTensor<float> out = outQue_.DeQue<float>();
    SetAtomicAdd<float>();
    DataCopy(outGm_[rb], out, channel_);
    SetAtomicNone();
    outQue_.FreeTensor(out);
}

template<bool with_depth>
__aicore__ inline void BEVPoolV3Kernel<with_depth>::ProcessSingle(uint64_t taskIdx, uint32_t actualRankNum)
{
    int32_t rankNum = AlignUp(actualRankNum, B32_DATA_NUM_PER_BLOCK);

    LocalTensor<int32_t> ranks = ranksQue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> rankBev = ranks[rankBevOffset_];
    DataCopy(rankBev, ranksBevGm_[taskIdx * avgRankNum_], rankNum);
    if (with_depth) {
        LocalTensor<int32_t> rankDepth = ranks[rankDepthOffset_];
        LocalTensor<int32_t> rankFeat = ranks[rankFeatOffset_];
        DataCopy(rankDepth, ranksDepthGm_[taskIdx * avgRankNum_], rankNum);
        DataCopy(rankFeat, ranksFeatGm_[taskIdx * avgRankNum_], rankNum);
        ranksQue_.EnQue(ranks);
        ranksQue_.DeQue<int32_t>();
        Muls(rankBev, rankBev, channel_, rankNum);
        Muls(rankFeat, rankFeat, channel_, rankNum);
        for (int32_t i = 0; i < actualRankNum; ++i) {
            uint64_t rd = rankDepth.GetValue(i);
            uint64_t rf = rankFeat.GetValue(i);
            uint64_t rb = rankBev.GetValue(i);
            CopyIn(rd, rf);
            Compute();
            CopyOut(rb);
        }
    } else {
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
    }
    ranksQue_.FreeTensor(ranks);
}

template<bool with_depth>
__aicore__ inline void BEVPoolV3Kernel<with_depth>::Process()
{
    for (uint32_t i = taskStartIdx_; i < taskEndIdx_; ++i) {
        uint32_t actualRankNum = avgRankNum_;
        if (unlikely(i == totalTaskNum_ - 1)) {
            actualRankNum = tailRankNum_;
        }
        ProcessSingle(i, actualRankNum);
    }
}

extern "C" __global__ __aicore__ void bev_pool_v3(GM_ADDR depth, GM_ADDR feat, GM_ADDR ranksDepth, GM_ADDR ranksFeat,
    GM_ADDR ranksBev, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(bevPoolTiling, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        BEVPoolV3Kernel<false> kernel(&pipe, depth, feat, ranksDepth, ranksFeat, ranksBev, out, bevPoolTiling);
        kernel.Process();
    } else if (TILING_KEY_IS(1)) {
        BEVPoolV3Kernel<true> kernel(&pipe, depth, feat, ranksDepth, ranksFeat, ranksBev, out, bevPoolTiling);
        kernel.Process();
    }
}
