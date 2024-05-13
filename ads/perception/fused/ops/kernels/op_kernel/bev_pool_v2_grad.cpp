#include "bev_pool_v2.h"
using namespace AscendC;

namespace BEVPoolV2 {
template<typename T, bool Align32B>
__aicore__ inline void BEVPoolV2GradKernel<T, Align32B>::DoProcess()
{
    LocalTensor<T> gradFeatT = gradFeatQue_.AllocTensor<T>(); // wait_flag(met3, v)
    Duplicate(gradFeatT, T(0.f), this->alignUpCCount_);       // pipe_v
    for (int32_t i = 0; i < this->length_; ++i) {
        this->depthOffset_ = this->rDGm_.GetValue(this->start_ + i);
        this->outOffset_ = this->rBGm_.GetValue(this->start_ + i) * this->stride0_;
        LocalTensor<T> gradOutT = gradOutQue_.AllocTensor<T>();                 // wait_flag(v, mte2)
        DataCopy(gradOutT, this->gOGm_[this->outOffset_], this->cpFeatParams_); // met2
        gradOutQue_.EnQue(gradOutT);                                            // set_flag(mte2, v)
        gradOutT = gradOutQue_.DeQue<T>();                                      // wait_flag(mte2, v)
        // actually, we just need to calculate gradDepth for the last time
        if (i == this->length_ - 1) {
            this->featOffset_ = this->rFGm_.GetValue(this->start_ + i) * this->stride0_;
            LocalTensor<T> featT = this->featQue_.template AllocTensor<T>();     // wait_flag(v, mte2)
            DataCopy(featT, this->fGm_[this->featOffset_], this->cpFeatParams_); // met2
            this->featQue_.EnQue(featT);                                         // set_flag(mte2, v)
            featT = this->featQue_.template DeQue<T>();                          // wait_flag(mte2, v)
            // calculate gradDepth, sum of feat * gradOut
            Mul(featT, gradOutT, featT, this->alignUpCCount_);                        // pipe_v
            LocalTensor<T> gradDepthT = gradDepthQue_.AllocTensor<T>();               // wait_flag(met3, v)
            ReduceSum(gradDepthT, featT, workT_, this->stride0_);                     // pipe_v
            this->featQue_.FreeTensor(featT);                                         // set_flag(v, mte2)
            gradDepthQue_.EnQue(gradDepthT);                                          // set_flag(v, mte3)
            gradDepthT = gradDepthQue_.DeQue<T>();                                    // wait_flag(v, mte3)
            DataCopyPad(gDGm_[this->depthOffset_], gradDepthT, this->cpDepthParams_); // mte3
            gradDepthQue_.FreeTensor(gradDepthT);                                     // set_flag(mte3, v)
        }

        // calculate gradFeat, sum of depth * gradOut
        T depth = this->dGm_.GetValue(this->depthOffset_);
        Muls(gradOutT, gradOutT, depth, this->alignUpCCount_);     // pipe_v
        Add(gradFeatT, gradFeatT, gradOutT, this->alignUpCCount_); // pipe_v
        this->featQue_.FreeTensor(gradOutT);                       // set_flag(v, mte2)
    }
    gradFeatQue_.EnQue(gradFeatT); // set_flag(v, mte3)

    gradFeatT = gradFeatQue_.DeQue<T>(); // wait_flag(v, mte3)
    int32_t featOffset = this->rFGm_.GetValue(this->start_) * this->stride0_;
    if (Align32B) {
        DataCopy(this->gFGm_[featOffset], gradFeatT, this->cpFeatParams_); // mte3
    } else {
        DataCopyPad(this->gFGm_[featOffset], gradFeatT, this->cpPadParams_); // mte3
    }
    gradFeatQue_.FreeTensor(gradFeatT); // set_flag(mte3, v)
}
} // namespace BEVPoolV2

extern "C" __global__ __aicore__ void bev_pool_v2_grad(GM_ADDR gradOut, GM_ADDR depth, GM_ADDR feat, GM_ADDR ranksDepth,
    GM_ADDR ranksFeat, GM_ADDR ranksBev, GM_ADDR intervalLengths, GM_ADDR intervalStarts, GM_ADDR gradDepth,
    GM_ADDR gradFeat, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(bevPoolTiling, tiling);
    int32_t blkIdx = GetBlockIdx();
    int32_t c = bevPoolTiling.stride0; // channel
// the tiling key represented as below:
// +----+----+----+-----+
// |bf16|fp16|fp32|align|
// +----+----+----+-----+
#if __CCE_AICORE__ == 220
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
    if (TILING_KEY_IS(3)) { // 1 << BEVPool::TILING_FP32_BIT | BEVPool::TILING_ALIGN32B_FLAG
        const int32_t cBytes = c * sizeof(float);
        const int32_t divCeilC = DivCeil(cBytes, ONE_BLK_SIZE);
        const int32_t alignUpCBytes = divCeilC * ONE_BLK_SIZE;
        BEVPoolV2::BEVPoolV2GradKernel<float, true> op(blkIdx, cBytes, divCeilC, alignUpCBytes, gradOut, depth, feat,
            ranksDepth, ranksFeat, ranksBev, intervalLengths, intervalStarts, gradDepth, gradFeat, bevPoolTiling);
        op.Process();
    } else if (TILING_KEY_IS(2)) { // 1 << BEVPool::TILING_FP32_BIT
        const int32_t cBytes = c * sizeof(float);
        const int32_t divCeilC = DivCeil(cBytes, ONE_BLK_SIZE);
        const int32_t alignUpCBytes = divCeilC * ONE_BLK_SIZE;
        BEVPoolV2::BEVPoolV2GradKernel<float, false> op(blkIdx, cBytes, divCeilC, alignUpCBytes, gradOut, depth, feat,
            ranksDepth, ranksFeat, ranksBev, intervalLengths, intervalStarts, gradDepth, gradFeat, bevPoolTiling);
        op.Process();
    } // we just support fp32 at present
    PipeBarrier<PIPE_ALL>();
}
