#include "bev_pool.h"
using namespace AscendC;

namespace BEVPool {
template<typename T, bool Align32B>
__aicore__ inline void BEVPoolGradKernel<T, Align32B>::DoProcess()
{
    LocalTensor<T> gradOutT = que_.AllocTensor<T>();
    DataCopy(gradOutT, this->oGm_[this->outOffset_], this->cpFeatParams_);
    que_.EnQue(gradOutT); // set_flag(mte2, mte3)

    gradOutT = que_.DeQue<T>(); // wait_flag(mte2, mte3)
    for (int32_t i = 0; i < this->length_; ++i) {
        if (Align32B) {
            DataCopy(this->fGm_[this->featOffset_], gradOutT, this->cpFeatParams_);
        } else {
            DataCopyPad(this->fGm_[this->featOffset_], gradOutT, this->cpPadParams_);
        }
        this->featOffset_ += this->stride0_;
    }
    que_.FreeTensor(gradOutT); // set_flag(mte3, mte2)
}
} // namespace BEVPool

extern "C" __global__ __aicore__ void bev_pool_grad(GM_ADDR gradOut, GM_ADDR geomFeat, GM_ADDR intervalLengths,
    GM_ADDR intervalStarts, GM_ADDR gradFeat, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(bevPoolTiling, tiling);
    int32_t blkIdx = GetBlockIdx();
    int32_t c = bevPoolTiling.stride0; // channel

#if __CCE_AICORE__ == 220
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
    // the tiling key represented as below:
    // +----+----+----+-----+
    // |bf16|fp16|fp32|align|
    // +----+----+----+-----+
    if (TILING_KEY_IS(3)) { // 1 << BEVPool::TILING_FP32_BIT | BEVPool::TILING_ALIGN32B_FLAG
        const int32_t cBytes = c * sizeof(float);
        const int32_t divCeilC = DivCeil(cBytes, ONE_BLK_SIZE);
        const int32_t alignUpCBytes = divCeilC * ONE_BLK_SIZE;
        BEVPool::BEVPoolGradKernel<float, true> op(blkIdx, cBytes, divCeilC, alignUpCBytes, gradOut, geomFeat,
            intervalLengths, intervalStarts, gradFeat, bevPoolTiling);
        op.Process();
    } else if (TILING_KEY_IS(2)) { // 1 << BEVPool::TILING_FP32_BIT
        const int32_t cBytes = c * sizeof(float);
        const int32_t divCeilC = DivCeil(cBytes, ONE_BLK_SIZE);
        const int32_t alignUpCBytes = divCeilC * ONE_BLK_SIZE;
        BEVPool::BEVPoolGradKernel<float, false> op(blkIdx, cBytes, divCeilC, alignUpCBytes, gradOut, geomFeat,
            intervalLengths, intervalStarts, gradFeat, bevPoolTiling);
        op.Process();
    } // we just support fp32 at present
    PipeBarrier<PIPE_ALL>();
}
