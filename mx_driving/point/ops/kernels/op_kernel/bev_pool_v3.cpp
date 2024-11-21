#include "kernel_operator.h"
using namespace AscendC;

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
        rankDepthOffset_ = 0;
        rankFeatOffset_ = rankDepthOffset_ + rankSize_;
        rankBevOffset_ = rankFeatOffset_ + rankSize_;
    }

    __aicore__ inline void InitGM(
        GM_ADDR depth, GM_ADDR feat, GM_ADDR ranksDepth, GM_ADDR ranksFeat, GM_ADDR ranksBev, GM_ADDR out)
    {
        depthGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(depth));
        featGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(feat));
        ranksDepthGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ranksDepth));
        ranksFeatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ranksFeat));
        ranksBevGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ranksBev));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
    }

    __aicore__ inline void InitBuffer()
    {
        pipe_->InitBuffer(ranksQue_, 1, 3 * rankSize_ * sizeof(int32_t));
        pipe_->InitBuffer(inQue_, 2, (B32_DATA_NUM_PER_BLOCK + channel_) * sizeof(int32_t));
        pipe_->InitBuffer(outQue_, 2, channel_ * sizeof(float));
    }

    __aicore__ inline void CopyIn(int32_t rd, int32_t rf);

    __aicore__ inline void Compute();

    __aicore__ inline void CopyOut(int32_t rb);

    __aicore__ inline void ProcessSingle(int32_t taskIdx, int32_t rankNum);

private:
    TPipe* pipe_;
    int32_t blkIdx_;
    GlobalTensor<float> depthGm_, featGm_, outGm_;
    GlobalTensor<int32_t> ranksDepthGm_, ranksFeatGm_, ranksBevGm_;
    TQue<TPosition::VECIN, 1> ranksQue_;
    TQue<TPosition::VECIN, 2> inQue_;
    TQue<TPosition::VECOUT, 2> outQue_;

    int32_t taskStartIdx_, taskEndIdx_, totalTaskNum_, avgRankNum_, tailRankNum_;
    int32_t channel_;
    int32_t rankSize_;
    int32_t rankDepthOffset_, rankFeatOffset_, rankBevOffset_;
};

__aicore__ inline void BEVPoolV3Kernel::CopyIn(int32_t rd, int32_t rf)
{
    LocalTensor<float> in = inQue_.AllocTensor<float>();
    DataCopy(in, depthGm_[rd], B32_DATA_NUM_PER_BLOCK);
    DataCopy(in[8], featGm_[rf], channel_);
    inQue_.EnQue(in);
}

__aicore__ inline void BEVPoolV3Kernel::Compute()
{
    LocalTensor<float> in = inQue_.DeQue<float>();
    LocalTensor<float> out = outQue_.AllocTensor<float>();
    Muls(out, in[8], in.GetValue(0), channel_);
    inQue_.FreeTensor(in);
    outQue_.EnQue(out);
}

__aicore__ inline void BEVPoolV3Kernel::CopyOut(int32_t rb)
{
    LocalTensor<float> out = outQue_.DeQue<float>();
    SetAtomicAdd<float>();
    DataCopy(outGm_[rb], out, channel_);
    SetAtomicNone();
    outQue_.FreeTensor(out);
}

__aicore__ inline void BEVPoolV3Kernel::ProcessSingle(int32_t taskIdx, int32_t actualRankNum)
{
    int32_t rankNum = AlignUp(actualRankNum, B32_DATA_NUM_PER_BLOCK);

    LocalTensor<int32_t> ranks = ranksQue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> rankDepth = ranks[rankDepthOffset_];
    LocalTensor<int32_t> rankFeat = ranks[rankFeatOffset_];
    LocalTensor<int32_t> rankBev = ranks[rankBevOffset_];
    DataCopy(rankDepth, ranksDepthGm_[taskIdx * avgRankNum_], rankNum);
    DataCopy(rankFeat, ranksFeatGm_[taskIdx * avgRankNum_], rankNum);
    DataCopy(rankBev, ranksBevGm_[taskIdx * avgRankNum_], rankNum);
    ranksQue_.EnQue(ranks);
    
    ranksQue_.DeQue<int32_t>();
    Muls(rankFeat, rankFeat, channel_, rankNum);
    Muls(rankBev, rankBev, channel_, rankNum);

    for (int32_t i = 0; i < actualRankNum; ++i) {
        int32_t rd = rankDepth.GetValue(i);
        int32_t rf = rankFeat.GetValue(i);
        int32_t rb = rankBev.GetValue(i);
        CopyIn(rd, rf);
        Compute();
        CopyOut(rb);
    }
    ranksQue_.FreeTensor(ranks);
}

__aicore__ inline void BEVPoolV3Kernel::Process()
{
    for (int32_t i = taskStartIdx_; i < taskEndIdx_; ++i) {
        int32_t actualRankNum = avgRankNum_;
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
    BEVPoolV3Kernel kernel(&pipe, depth, feat, ranksDepth, ranksFeat, ranksBev, out, bevPoolTiling);
    kernel.Process();
}
