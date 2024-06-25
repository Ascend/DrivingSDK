#include "bev_pool.h"
using namespace AscendC;

namespace BEVPool {
template<typename T, bool Align32B>
__aicore__ inline void BEVPoolKernel<T, Align32B>::DoProcess()
{
    LocalTensor<T> outT = outQue_.AllocTensor<T>(); // wait_flag(met3, v)
    Duplicate(outT, T(0.f), this->alignUpCCount_);  // pipe_v
    for (int32_t i = 0; i < this->length_; ++i) {
        LocalTensor<T> featT = featQue_.AllocTensor<T>();                    // wait_flag(v, mte2)
        DataCopy(featT, this->fGm_[this->featOffset_], this->cpFeatParams_); // met2
        featQue_.EnQue(featT);                                               // set_flag(mte2, v)
        featT = featQue_.DeQue<T>();                                         // wait_flag(mte2, v)
        Add(outT, featT, outT, this->alignUpCCount_);                        // pipe_v
        featQue_.FreeTensor(featT);                                          // set_flag(v, mte2)
        this->featOffset_ += this->stride0_;
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
} // namespace BEVPool

extern "C" __global__ __aicore__ void bev_pool(GM_ADDR feat, GM_ADDR geomFeat, GM_ADDR intervalLengths,
    GM_ADDR intervalStarts, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
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
        BEVPool::BEVPoolKernel<float, true> op(blkIdx, cBytes, divCeilC, alignUpCBytes, feat, geomFeat, intervalLengths,
            intervalStarts, out, bevPoolTiling);
        op.Process();
    } else if (TILING_KEY_IS(2)) { // 1 << BEVPool::TILING_FP32_BIT
        const int32_t cBytes = c * sizeof(float);
        const int32_t divCeilC = DivCeil(cBytes, ONE_BLK_SIZE);
        const int32_t alignUpCBytes = divCeilC * ONE_BLK_SIZE;
        BEVPool::BEVPoolKernel<float, false> op(blkIdx, cBytes, divCeilC, alignUpCBytes, feat, geomFeat,
            intervalLengths, intervalStarts, out, bevPoolTiling);
        op.Process();
    } // we just support fp32 at present
    PipeBarrier<PIPE_ALL>();
}
