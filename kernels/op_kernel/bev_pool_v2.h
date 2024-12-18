#ifndef BEV_POOL_V2_H
#define BEV_POOL_V2_H
#include "common.h"

namespace BEVPoolV2 {
constexpr int32_t BUFFER_NUM = 2; // double buffer
template<typename T, bool Align32B>
class BEVPoolV2BaseKernel {
public:
    __aicore__ inline BEVPoolV2BaseKernel() = delete;

    __aicore__ inline BEVPoolV2BaseKernel(int32_t blkIdx, int32_t cBytes, int32_t divCeilC, int32_t alignUpCBytes,
        GM_ADDR depth, GM_ADDR feat, GM_ADDR ranksDepth, GM_ADDR ranksFeat, GM_ADDR ranksBev, GM_ADDR intervalLengths,
        GM_ADDR intervalStarts, const BEVPoolTilingData& bevPoolTiling)
        : blkIdx_(blkIdx), it_(blkIdx, bevPoolTiling.usedCoreNum, bevPoolTiling.avgTaskNum, bevPoolTiling.tailTaskNum,
                               bevPoolTiling.totalTaskNum),
          cpFeatParams_(1, divCeilC, 0, 0), cpOneParams_(1, 1, 0, 0), alignUpCCount_(alignUpCBytes / sizeof(T)),
          cpPadParams_(1, cBytes, 0, 0, 0)
    {
        stride0_ = bevPoolTiling.stride0;
        stride1_ = bevPoolTiling.stride1;
        stride2_ = bevPoolTiling.stride2;
        stride3_ = bevPoolTiling.stride3;

        dGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(depth));
        fGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(feat));
        rDGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ranksDepth));
        rFGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ranksFeat));
        rBGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ranksBev));
        sGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(intervalStarts));
        lGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(intervalLengths));

        pipe_.InitBuffer(featQue_, BUFFER_NUM, alignUpCBytes);
    }

protected:
    int32_t blkIdx_;

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> featQue_;
    AscendC::GlobalTensor<int32_t> rDGm_, rFGm_, rBGm_, sGm_, lGm_;
    AscendC::GlobalTensor<T> dGm_, fGm_;

    uint32_t stride0_, stride1_, stride2_, stride3_;
    int32_t alignUpCCount_;
    TaskIterator it_;

    AscendC::DataCopyParams cpFeatParams_, cpOneParams_;
    AscendC::DataCopyExtParams cpPadParams_;
    int32_t start_, length_;
    uint64_t outOffset_, featOffset_, depthOffset_;

    __aicore__ inline void PreProcess(int32_t idx)
    {
        length_ = lGm_.GetValue(idx);
        start_ = sGm_.GetValue(idx);
    }
};

template<typename T, bool Align32B>
class BEVPoolV2Kernel : public BEVPoolV2BaseKernel<T, Align32B> {
public:
    __aicore__ inline BEVPoolV2Kernel() = delete;

    __aicore__ inline BEVPoolV2Kernel(int32_t blkIdx, int32_t cBytes, int32_t divCeilC, int32_t alignUpCBytes,
        GM_ADDR depth, GM_ADDR feat, GM_ADDR ranksDepth, GM_ADDR ranksFeat, GM_ADDR ranksBev, GM_ADDR intervalLengths,
        GM_ADDR intervalStarts, GM_ADDR out, const BEVPoolTilingData& bevPoolTiling)
        : BEVPoolV2BaseKernel<T, Align32B>(blkIdx, cBytes, divCeilC, alignUpCBytes, depth, feat, ranksDepth, ranksFeat,
              ranksBev, intervalLengths, intervalStarts, bevPoolTiling)

    {
        oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
        this->pipe_.InitBuffer(outQue_, BUFFER_NUM, alignUpCBytes);
    }

    __aicore__ inline void Process()
    {
        while (this->it_.HasNext()) {
            const int32_t idx = this->it_.Next();
            this->PreProcess(idx);
            DoProcess();
        }
    }

private:
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQue_;
    AscendC::GlobalTensor<T> oGm_;

    __aicore__ inline void DoProcess();
};

template<typename T, bool Align32B>
class BEVPoolV2GradKernel : public BEVPoolV2BaseKernel<T, Align32B> {
public:
    __aicore__ inline BEVPoolV2GradKernel() = delete;

    __aicore__ inline BEVPoolV2GradKernel(int32_t blkIdx, int32_t cBytes, int32_t divCeilC, int32_t alignUpCBytes,
        GM_ADDR gradOut, GM_ADDR depth, GM_ADDR feat, GM_ADDR ranksDepth, GM_ADDR ranksFeat, GM_ADDR ranksBev,
        GM_ADDR intervalLengths, GM_ADDR intervalStarts, GM_ADDR gradDepth, GM_ADDR gradFeat,
        const BEVPoolTilingData& bevPoolTiling)
        : BEVPoolV2BaseKernel<T, Align32B>(blkIdx, cBytes, divCeilC, alignUpCBytes, depth, feat, ranksDepth, ranksFeat,
              ranksBev, intervalLengths, intervalStarts, bevPoolTiling),
          cpDepthParams_(1, sizeof(T), 0, 0, 0)
    {
        gOGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradOut));
        gDGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradDepth));
        gFGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradFeat));

        this->pipe_.InitBuffer(gradOutQue_, BUFFER_NUM, alignUpCBytes);
        this->pipe_.InitBuffer(gradDepthQue_, BUFFER_NUM, AscendC::ONE_BLK_SIZE);
        this->pipe_.InitBuffer(gradFeatQue_, BUFFER_NUM, alignUpCBytes);
        this->pipe_.InitBuffer(workBuf_, alignUpCBytes);
        workT_ = workBuf_.Get<T>();
    }

    __aicore__ inline void Process()
    {
        while (this->it_.HasNext()) {
            const int32_t idx = this->it_.Next();
            this->PreProcess(idx);
            DoProcess();
        }
    }

private:
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> gradOutQue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> gradDepthQue_, gradFeatQue_;
    AscendC::TBuf<AscendC::QuePosition::LCM> workBuf_;
    AscendC::GlobalTensor<T> gOGm_, gDGm_, gFGm_;
    AscendC::LocalTensor<T> workT_;
    AscendC::DataCopyExtParams cpDepthParams_;

    __aicore__ inline void DoProcess();
};
} // namespace BEVPoolV2

#endif // BEV_POOL_V2_H