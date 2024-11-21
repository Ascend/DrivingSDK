#include "kernel_operator.h"
using namespace AscendC;

class BEVPoolV3GradKernel {
public:
    __aicore__ inline BEVPoolV3GradKernel() = delete;

    __aicore__ inline ~BEVPoolV3GradKernel() = default;

    __aicore__ inline BEVPoolV3GradKernel(TPipe* pipe, GM_ADDR gradOut, GM_ADDR depth, GM_ADDR feat, GM_ADDR ranksDepth,
        GM_ADDR ranksFeat, GM_ADDR ranksBev, GM_ADDR gradDepth, GM_ADDR gradFeat, const BEVPoolV3TilingData& tiling)
        : pipe_(pipe), blkIdx_(GetBlockIdx()), channel_(tiling.channel)
    {
        InitTask(tiling);
        InitOffset();
        InitGM(gradOut, depth, feat, ranksDepth, ranksFeat, ranksBev, gradDepth, gradFeat);
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
        inFeatOffset_ = B32_DATA_NUM_PER_BLOCK;
        inBevOffset_ = inFeatOffset_ + channel_;
    }

    __aicore__ inline void InitGM(GM_ADDR gradOut, GM_ADDR depth, GM_ADDR feat, GM_ADDR ranksDepth, GM_ADDR ranksFeat,
        GM_ADDR ranksBev, GM_ADDR gradDepth, GM_ADDR gradFeat)
    {
        gradOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradOut));
        depthGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(depth));
        featGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(feat));
        ranksDepthGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ranksDepth));
        ranksFeatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ranksFeat));
        ranksBevGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ranksBev));
        gradDepthGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradDepth));
        gradFeatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradFeat));
    }

    __aicore__ inline void InitBuffer()
    {
        pipe_->InitBuffer(ranksQue_, 1, 3 * rankSize_ * sizeof(int32_t));
        pipe_->InitBuffer(inQue_, 2, (B32_DATA_NUM_PER_BLOCK + channel_ * 2) * sizeof(int32_t));
        pipe_->InitBuffer(outQue_, 2, channel_ * 3 * sizeof(float));
    }

    __aicore__ inline void CopyIn(int32_t rd, int32_t rf, int32_t rb);

    __aicore__ inline void Compute();

    __aicore__ inline void CopyOut(int32_t rd, int32_t rf);

private:
    TPipe* pipe_;
    int32_t blkIdx_;
    GlobalTensor<float> gradOutGm_, depthGm_, featGm_, gradDepthGm_, gradFeatGm_;
    GlobalTensor<int32_t> ranksDepthGm_, ranksFeatGm_, ranksBevGm_;
    TQue<TPosition::VECIN, 1> ranksQue_;
    TQue<TPosition::VECIN, 2> inQue_;
    TQue<TPosition::VECOUT, 2> outQue_;

    int32_t taskStartIdx_, taskEndIdx_, totalTaskNum_, avgRankNum_, tailRankNum_;
    int32_t channel_;
    int32_t rankSize_;
    int32_t rankDepthOffset_, rankFeatOffset_, rankBevOffset_, inFeatOffset_, inBevOffset_;

    DataCopyParams cpSingleParams_ {1, B32_BYTE_SIZE, 0, 0};
};

__aicore__ inline void BEVPoolV3GradKernel::CopyIn(int32_t rd, int32_t rf, int32_t rb)
{
    LocalTensor<float> in = inQue_.AllocTensor<float>();
    DataCopy(in, depthGm_[rd], B32_DATA_NUM_PER_BLOCK);
    DataCopy(in[inFeatOffset_], featGm_[rf], channel_);
    DataCopy(in[inBevOffset_], gradOutGm_[rb], channel_);
    inQue_.EnQue(in);
}
__aicore__ inline void BEVPoolV3GradKernel::Compute()
{
    LocalTensor<float> in = inQue_.DeQue<float>();
    LocalTensor<float> out = outQue_.AllocTensor<float>();
    Muls(out, in[inBevOffset_], in.GetValue(0), channel_);             // gradFeat = gradOut * depth
    Mul(out[channel_], in[inBevOffset_], in[inFeatOffset_], channel_); // gradDepth = \sum(gradOut * feat)
    ReduceSum(out[channel_], out[channel_], out[2 * channel_], channel_);
    inQue_.FreeTensor(in);
    outQue_.EnQue(out);
}

__aicore__ inline void BEVPoolV3GradKernel::CopyOut(int32_t rd, int32_t rf)
{
    LocalTensor<float> out = outQue_.DeQue<float>();
    SetAtomicAdd<float>();
    DataCopy(gradFeatGm_[rf], out, channel_);
    DataCopyPad(gradDepthGm_[rd], out[channel_], cpSingleParams_);
    SetAtomicNone();
    outQue_.FreeTensor(out);
}


__aicore__ inline void BEVPoolV3GradKernel::Process()
{
    for (int32_t i = taskStartIdx_; i < taskEndIdx_; ++i) {
        int32_t actualRankNum = avgRankNum_;
        if (unlikely(i == totalTaskNum_ - 1)) {
            actualRankNum = tailRankNum_;
        }
        int32_t rankNum = AlignUp(actualRankNum, B32_DATA_NUM_PER_BLOCK);

        LocalTensor<int32_t> ranks = ranksQue_.AllocTensor<int32_t>();
        LocalTensor<int32_t> rankDepth = ranks[rankDepthOffset_];
        LocalTensor<int32_t> rankFeat = ranks[rankFeatOffset_];
        LocalTensor<int32_t> rankBev = ranks[rankBevOffset_];
        DataCopy(rankDepth, ranksDepthGm_[i * avgRankNum_], rankNum);
        DataCopy(rankFeat, ranksFeatGm_[i * avgRankNum_], rankNum);
        DataCopy(rankBev, ranksBevGm_[i * avgRankNum_], rankNum);
        ranksQue_.EnQue(ranks);
        ranksQue_.DeQue<int32_t>();
        Muls(rankFeat, rankFeat, channel_, rankNum);
        Muls(rankBev, rankBev, channel_, rankNum);
        for (int32_t j = 0; j < actualRankNum; ++j) {
            int32_t rd = rankDepth.GetValue(j);
            int32_t rf = rankFeat.GetValue(j);
            int32_t rb = rankBev.GetValue(j);
            CopyIn(rd, rf, rb);
            Compute();
            CopyOut(rd, rf);
        }
        ranksQue_.FreeTensor(ranks);
    }
}

extern "C" __global__ __aicore__ void bev_pool_v3_grad(GM_ADDR gradOut, GM_ADDR depth, GM_ADDR feat, GM_ADDR ranksDepth,
    GM_ADDR ranksFeat, GM_ADDR ranksBev, GM_ADDR gradDepth, GM_ADDR gradFeat, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(bevPoolTiling, tiling);
    TPipe pipe;
    BEVPoolV3GradKernel kernel(
        &pipe, gradOut, depth, feat, ranksDepth, ranksFeat, ranksBev, gradDepth, gradFeat, bevPoolTiling);
    kernel.Process();
}
