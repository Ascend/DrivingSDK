#include "bev_pool_v2.h"
using namespace AscendC;

namespace BEVPoolV2 {
template<typename T, bool Align32B>
__aicore__ inline void BEVPoolV2Kernel<T, Align32B>::DoProcess()
{
    LocalTensor<T> outT = outQue_.AllocTensor<T>(); // wait_flag(met3, v)
    Duplicate(outT, T(0.f), this->alignUpCCount_);  // pipe_v
    this->outOffset_ = this->rBGm_.GetValue(this->start_) * this->stride0_;
    for (int32_t i = 0; i < this->length_; ++i) {
        this->depthOffset_ = this->rDGm_.GetValue(this->start_ + i);
        this->featOffset_ = this->rFGm_.GetValue(this->start_ + i) * this->stride0_;
        T depth = this->dGm_.GetValue(this->depthOffset_);
        LocalTensor<T> featT = this->featQue_.template AllocTensor<T>();              // wait_flag(v, mte2)
        DataCopy(featT, this->fGm_[this->featOffset_], this->cpFeatParams_); // met2
        this->featQue_.EnQue(featT);                                         // set_flag(mte2, v)
        featT = this->featQue_.template DeQue<T>();                                   // wait_flag(mte2, v)
        Muls(featT, featT, depth, this->alignUpCCount_);                     // pipe_v
        Add(outT, featT, outT, this->alignUpCCount_);                        // pipe_v
        this->featQue_.FreeTensor(featT);                                    // set_flag(v, mte2)
    }
    outQue_.EnQue(outT); // set_flag(v, mte3)

    outT = outQue_.DeQue<T>(); // wait_flag(v, mte3)
    if (Align32B) {
        DataCopy(this->oGm_[this->outOffset_], outT, this->cpFeatParams_); // mte3
    } else {
        DataCopyPad(this->oGm_[this->outOffset_], outT, this->cpPadParams_); // mte3
    }
    outQue_.FreeTensor(outT); // set_flag(mte3, v)
}
} // namespace BEVPoolV2

extern "C" __global__ __aicore__ void bev_pool_v2(GM_ADDR depth, GM_ADDR feat, GM_ADDR ranksDepth, GM_ADDR ranksFeat,
    GM_ADDR ranksBev, GM_ADDR intervalLengths, GM_ADDR intervalStarts, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
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
        BEVPoolV2::BEVPoolV2Kernel<float, true> op(blkIdx, cBytes, divCeilC, alignUpCBytes, depth, feat, ranksDepth,
            ranksFeat, ranksBev, intervalLengths, intervalStarts, out, bevPoolTiling);
        op.Process();
    } else if (TILING_KEY_IS(2)) { // 1 << BEVPool::TILING_FP32_BIT
        const int32_t cBytes = c * sizeof(float);
        const int32_t divCeilC = DivCeil(cBytes, ONE_BLK_SIZE);
        const int32_t alignUpCBytes = divCeilC * ONE_BLK_SIZE;
        BEVPoolV2::BEVPoolV2Kernel<float, false> op(blkIdx, cBytes, divCeilC, alignUpCBytes, depth, feat, ranksDepth,
            ranksFeat, ranksBev, intervalLengths, intervalStarts, out, bevPoolTiling);
        op.Process();
    } // we just support fp32 at present
    PipeBarrier<PIPE_ALL>();
}
