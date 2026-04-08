/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 *
 */
#ifndef BEV_POOL_H_
#define BEV_POOL_H_

#include "common.h"

namespace BEVPool {
constexpr int32_t BUFFER_NUM = 2; // double buffer

template<int32_t depth>
class TConjungateQue : public AscendC::TQueBind<AscendC::TPosition::LCM, AscendC::TPosition::LCM, depth, 0> {
public:
    __aicore__ inline TConjungateQue() = default;
};

template<typename T, bool Align32B>
class BEVPoolBaseKernel {
public:
    __aicore__ inline BEVPoolBaseKernel() = delete;

    __aicore__ inline BEVPoolBaseKernel(int32_t blkIdx, int32_t cBytes, int32_t divCeilC, int32_t alignUpCBytes,
        GM_ADDR geomFeat, GM_ADDR intervalLengths, GM_ADDR intervalStarts, const BEVPoolTilingData& bevPoolTiling)
        : blkIdx_(blkIdx), it_(blkIdx, bevPoolTiling.usedCoreNum, bevPoolTiling.avgTaskNum, bevPoolTiling.tailTaskNum,
                               bevPoolTiling.totalTaskNum),
          cpFeatParams_(1, divCeilC, 0, 0), cpOneParams_(1, 1, 0, 0), alignUpCCount_(alignUpCBytes / sizeof(T)),
          cpPadParams_(1, cBytes, 0, 0, 0)
    {
        stride0_ = bevPoolTiling.stride0;
        stride1_ = bevPoolTiling.stride1;
        stride2_ = bevPoolTiling.stride2;
        stride3_ = bevPoolTiling.stride3;

        sGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(intervalStarts));
        lGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(intervalLengths));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(geomFeat));

        pipe_.InitBuffer(geomQue_, BUFFER_NUM, AscendC::ONE_BLK_SIZE); // geom
    }

protected:
    int32_t blkIdx_;

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> geomQue_;

    AscendC::GlobalTensor<T> fGm_, oGm_;
    AscendC::GlobalTensor<int32_t> sGm_, lGm_, gGm_;

    uint64_t stride0_, stride1_, stride2_, stride3_;
    int32_t alignUpCCount_;
    TaskIterator it_;

    AscendC::DataCopyParams cpFeatParams_, cpOneParams_;
    AscendC::DataCopyExtParams cpPadParams_;
    int32_t length_;
    uint64_t featOffset_, outOffset_;

    __aicore__ inline void PreProcess(int32_t idx)
    {
        uint64_t start = sGm_.GetValue(idx);
        featOffset_ = start * stride0_;
        length_ = lGm_.GetValue(idx);

        AscendC::LocalTensor<int32_t> geomT = geomQue_.AllocTensor<int32_t>(); // wait_flag(v, mte2)
        DataCopy(geomT, gGm_[4 * start], cpOneParams_);               // pipe_v
        geomQue_.EnQue(geomT);                                        // set_flag(mte2, v)
        geomT = geomQue_.DeQue<int32_t>();                            // wait_flag(mte2, v)
        outOffset_ = geomT.GetValue(1) * stride0_ + geomT.GetValue(0) * stride1_ + geomT.GetValue(2) * stride2_ +
                     geomT.GetValue(3) * stride3_;
        geomQue_.FreeTensor(geomT); // set_flag(v, mte2)
    }
};

template<typename T, bool Align32B>
class BEVPoolKernel : public BEVPoolBaseKernel<T, Align32B> {
public:
    __aicore__ inline BEVPoolKernel() = delete;

    __aicore__ inline BEVPoolKernel(int32_t blkIdx, int32_t cBytes, int32_t divCeilC, int32_t alignUpCBytes,
        GM_ADDR feat, GM_ADDR geomFeat, GM_ADDR intervalLengths, GM_ADDR intervalStarts, GM_ADDR out,
        const BEVPoolTilingData& bevPoolTiling)
        : BEVPoolBaseKernel<T, Align32B>(
              blkIdx, cBytes, divCeilC, alignUpCBytes, geomFeat, intervalLengths, intervalStarts, bevPoolTiling)

    {
        this->oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
        this->fGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(feat));

        this->pipe_.InitBuffer(featQue_, BUFFER_NUM, alignUpCBytes);
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
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> featQue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQue_;

    __aicore__ inline void DoProcess();
};

template<typename T, bool Align32B>
class BEVPoolGradKernel : public BEVPoolBaseKernel<T, Align32B> {
public:
    __aicore__ inline BEVPoolGradKernel() = delete;

    __aicore__ inline BEVPoolGradKernel(int32_t blkIdx, int32_t cBytes, int32_t divCeilC, int32_t alignUpCBytes,
        GM_ADDR gradOut, GM_ADDR geomFeat, GM_ADDR intervalLengths, GM_ADDR intervalStarts, GM_ADDR gradFeat,
        const BEVPoolTilingData& bevPoolTiling)
        : BEVPoolBaseKernel<T, Align32B>(
              blkIdx, cBytes, divCeilC, alignUpCBytes, geomFeat, intervalLengths, intervalStarts, bevPoolTiling)
    {
        this->oGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradOut));
        this->fGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradFeat));

        this->pipe_.InitBuffer(que_, BUFFER_NUM, alignUpCBytes);
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
    TConjungateQue<BUFFER_NUM> que_;

    __aicore__ inline void DoProcess();
};
} // namespace BEVPool
#endif // BEV_POOL_H_
