#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

constexpr MatmulConfig DEFORMABLE_CONV2D_CFG = GetNormalConfig();

template<bool modulated>
class DeformableConv2dKernel {
public:
    using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, true, LayoutMode::NONE, true>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;

    matmul::Matmul<AType, BType, CType, CType, DEFORMABLE_CONV2D_CFG> mm_;

    __aicore__ inline DeformableConv2dKernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR offset, GM_ADDR mask, GM_ADDR y,
        GM_ADDR offsetOutput, GM_ADDR workspace, const DeformableConv2dTilingData* tilingData, TPipe* pipe)

    {
        pipe_ = pipe;
        blkIdx_ = GetBlockIdx();
        InitTiling(tilingData);
        InitTask();
        InitGM(x, weight, bias, offset, mask, y, offsetOutput, workspace);
        InitBuffer();
        InitEvent();

        SetVectorMask<float>(FULL_MASK, FULL_MASK);
        SetAtomicNone();
    }

    __aicore__ inline void Process();

protected:
    TPipe* pipe_;
    GlobalTensor<float> xGm_, weightGm_, offsetGm_, biasGm_, maskGm_;
    GlobalTensor<float> yGm_, offsetOutputGm_;
    GlobalTensor<float> auxHGm_, auxWGm_;
    TBuf<TPosition::VECCALC> auxHBuf_, auxWBuf_;
    TBuf<TPosition::VECCALC> offsetBuf_, offsetIntBuf_, weightBuf_, maskBuf_, featureBuf_, offsetOutputBuf_;

    // from tiling
    uint64_t n_, cIn_, hIn_, wIn_, cOut_, hOut_, wOut_, kH_, kW_;
    uint32_t usedBlkNum_;
    int64_t padH_, padW_, strideH_, strideW_, dilationH_, dilationW_, groups_;
    uint64_t rowOut_, rowOutPerGroup_, kwIn_, kwInPerGroup_, rowIn_, kernelSize_, kernelPerGroup_, cInPerGroup_,
        rowInPerGroup_;
    uint64_t rowOffset_, alignedRowOffset_, featureOffset_;
    uint16_t rowOffsetBlk_, doubleRowOffsetBlk_, cInBlk_;
    uint16_t rptTimes_, valRptTimes_;
    uint32_t blkIdx_;
    uint32_t auxStart_, auxEnd_, start_, end_;
    uint64_t srcOffset_, dstOffset_;

    TEventID calEvt_, copyEvt_;

    DataCopyParams cpOneValParams_, cpRowDoubleValParams_, cpColDoubleValParams_ {2, 0, 0, 0},
        cpQuadValParams_ {2, 0, 0, 0}, cpOffsetOutParams_;
    GatherMaskParams gatherParams_;

private:
    __aicore__ inline void PreProcess();

    __aicore__ inline void ProcessCube(uint32_t taskIdx);

    __aicore__ inline void ProcessVector(uint32_t taskIdx);

    __aicore__ inline void CopyInOffset(
        uint32_t taskIdx, const LocalTensor<float>& offset, const LocalTensor<float>& mask);

    __aicore__ inline void ComputeWeight(uint32_t taskIdx, const LocalTensor<float>& auxW,
        const LocalTensor<float>& auxH, const LocalTensor<float>& offset, const LocalTensor<int32_t>& offsetInt,
        const LocalTensor<float>& weight, const LocalTensor<float>& mask);

    __aicore__ inline void ComputeBilinearInterpolation(uint32_t w, const LocalTensor<float>& offset,
        const LocalTensor<int32_t>& offsetInt, const LocalTensor<float>& feature, const LocalTensor<float>& weight,
        const LocalTensor<float>& offsetOutput);

    __aicore__ inline void InitTiling(const DeformableConv2dTilingData* tilingData)
    {
        n_ = tilingData->n;
        cIn_ = tilingData->cIn;
        hIn_ = tilingData->hIn;
        wIn_ = tilingData->wIn;
        cOut_ = tilingData->cOut;
        hOut_ = tilingData->hOut;
        wOut_ = tilingData->wOut;
        kH_ = tilingData->kH;
        kW_ = tilingData->kW;
        kernelSize_ = kH_ * kW_;
        padH_ = tilingData->padH;
        padW_ = tilingData->padW;
        strideH_ = tilingData->strideH;
        strideW_ = tilingData->strideW;
        dilationH_ = tilingData->dilationH;
        dilationW_ = tilingData->dilationW;
        groups_ = tilingData->groups;
        usedBlkNum_ = tilingData->usedBlkNum;
        featureOffset_ = 4 * cIn_;
        rowOut_ = wOut_ * cOut_;
        rowOutPerGroup_ = rowOut_ / groups_;
        kwIn_ = kernelSize_ * cIn_;
        rowIn_ = wOut_ * kwIn_;
        kwInPerGroup_ = kwIn_ / groups_;
        rowInPerGroup_ = rowIn_ / groups_;
        cInPerGroup_ = cIn_ / groups_;
        kernelPerGroup_ = cOut_ / groups_ * kwInPerGroup_;
        rowOffset_ = wOut_ * kernelSize_;
        alignedRowOffset_ = AlignUp(rowOffset_, B32_DATA_NUM_PER_REPEAT);
        rowOffsetBlk_ = Ceil(rowOffset_, B32_DATA_NUM_PER_BLOCK);
        doubleRowOffsetBlk_ = Ceil(2 * rowOffset_, B32_DATA_NUM_PER_BLOCK);
        cInBlk_ = Ceil(cIn_, B32_DATA_NUM_PER_BLOCK);

        cpOneValParams_.blockLen = cInBlk_;
        cpRowDoubleValParams_.blockLen = 2 * cInBlk_;
        cpColDoubleValParams_.blockLen = cInBlk_;
        cpColDoubleValParams_.srcStride = (wIn_ - 1) * cInBlk_;
        cpColDoubleValParams_.dstStride = cInBlk_;
        cpQuadValParams_.blockLen = 2 * cInBlk_;
        cpQuadValParams_.srcStride = (wIn_ - 2) * cInBlk_;
        cpOffsetOutParams_.blockCount = kernelSize_;
        cpOffsetOutParams_.blockLen = cInBlk_ / groups_;
        cpOffsetOutParams_.srcStride = cInBlk_ - cInBlk_ / groups_;
        rptTimes_ = alignedRowOffset_ / B32_DATA_NUM_PER_REPEAT;
        valRptTimes_ = cIn_ / B32_DATA_NUM_PER_REPEAT;
        gatherParams_.repeatTimes = rptTimes_ * 2;
    }

    __aicore__ inline void InitEvent()
    {
        calEvt_ = pipe_->AllocEventID<HardEvent::V_MTE2>();
        copyEvt_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
    }

    __aicore__ inline void InitTask()
    {
        uint64_t auxAvgTasks = wOut_ / usedBlkNum_;
        uint64_t auxRemainTasks = wOut_ % usedBlkNum_;
        auxStart_ = auxAvgTasks * blkIdx_ + (blkIdx_ < auxRemainTasks ? blkIdx_ : auxRemainTasks);
        auxEnd_ = auxStart_ + auxAvgTasks + (blkIdx_ < auxRemainTasks ? 1 : 0);

        uint64_t totalTasks = n_ * hOut_;
        uint64_t avgTasks = totalTasks / usedBlkNum_;
        uint64_t remainTasks = totalTasks % usedBlkNum_;
        start_ = avgTasks * blkIdx_ + (blkIdx_ < remainTasks ? blkIdx_ : remainTasks);
        end_ = start_ + avgTasks + (blkIdx_ < remainTasks ? 1 : 0);
    }

    __aicore__ inline void InitBuffer()
    {
        pipe_->InitBuffer(auxHBuf_, alignedRowOffset_ * B32_BYTE_SIZE); // 9 * 100
        pipe_->InitBuffer(auxWBuf_, alignedRowOffset_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(offsetBuf_, 4 * alignedRowOffset_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(offsetIntBuf_, 2 * alignedRowOffset_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(weightBuf_, 4 * alignedRowOffset_ * B32_BYTE_SIZE);
        if (modulated) {
            pipe_->InitBuffer(maskBuf_, alignedRowOffset_ * B32_BYTE_SIZE);
        }
        pipe_->InitBuffer(offsetOutputBuf_, 2 * kwIn_ * B32_BYTE_SIZE); // 2 for double buffer
        pipe_->InitBuffer(featureBuf_, 2 * 4 * cIn_ * B32_BYTE_SIZE);
    }
    __aicore__ inline void InitGM(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR offset, GM_ADDR mask, GM_ADDR y,
        GM_ADDR offsetOutput, GM_ADDR workspace)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight));
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
        offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(offset));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        offsetOutputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(offsetOutput));

        auxHGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        auxWGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + rowOffset_);
        if (modulated) {
            maskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(mask));
        }
    }
};

template<bool modulated>
__aicore__ inline void DeformableConv2dKernel<modulated>::PreProcess()
{
    LocalTensor<float> auxH = auxHBuf_.Get<float>();
    LocalTensor<float> auxW = auxWBuf_.Get<float>();

    uint32_t idx = 0;
    for (int32_t w = auxStart_; w < auxEnd_; ++w) {
        for (int32_t i = 0; i < kH_; ++i) {
            for (int32_t j = 0; j < kW_; ++j) {
                auxW.SetValue(idx, static_cast<float>(w * strideW_ - padW_ + j * dilationW_));
                auxH.SetValue(idx, static_cast<float>(-padH_ + i * dilationH_));
                ++idx;
            }
        }
    }
    DataCopyPad(auxWGm_[auxStart_ * kernelSize_], auxW,
        {1, static_cast<uint16_t>(B32_BYTE_SIZE * (auxEnd_ - auxStart_) * kernelSize_), 0, 0});
    DataCopyPad(auxHGm_[auxStart_ * kernelSize_], auxH,
        {1, static_cast<uint16_t>(B32_BYTE_SIZE * (auxEnd_ - auxStart_) * kernelSize_), 0, 0});
    SyncAll();
    DataCopy(auxW, auxWGm_, {1, rowOffsetBlk_, 0, 0});
    DataCopy(auxH, auxHGm_, {1, rowOffsetBlk_, 0, 0});

    LocalTensor<float> feature = featureBuf_.Get<float>();
    Duplicate<float, false>(feature, 0.f, MASK_PLACEHOLDER, 4 * valRptTimes_, 1, 8);
}

template<bool modulated>
__aicore__ inline void DeformableConv2dKernel<modulated>::ProcessCube(uint32_t taskIdx)
{
    uint64_t aOffset = 0;
    uint64_t bOffset = taskIdx * rowIn_;
    uint64_t cOffset = taskIdx * rowOut_;
    for (uint32_t i = 0; i < groups_; ++i) {
        mm_.SetTensorA(weightGm_[aOffset]);
        mm_.SetTensorB(offsetOutputGm_[bOffset], true);
        mm_.template IterateAll<false>(yGm_[cOffset]);
        aOffset += kernelPerGroup_;
        bOffset += rowInPerGroup_;
        cOffset += rowOutPerGroup_;
    }
}
template<bool modulated>
__aicore__ inline void DeformableConv2dKernel<modulated>::ProcessVector(uint32_t taskIdx)
{
    uint32_t batch = taskIdx / hOut_;
    srcOffset_ = batch * hIn_ * wIn_ * cIn_;
    dstOffset_ = taskIdx * rowIn_;
    LocalTensor<float> offset = offsetBuf_.Get<float>();
    LocalTensor<float> auxW = auxWBuf_.Get<float>();
    LocalTensor<float> auxH = auxHBuf_.Get<float>();
    LocalTensor<int32_t> offsetInt = offsetIntBuf_.Get<int32_t>();
    LocalTensor<float> weight = weightBuf_.Get<float>();
    LocalTensor<float> feature = featureBuf_.Get<float>();
    LocalTensor<float> mask;
    if (modulated) {
        mask = maskBuf_.Get<float>();
    }
    LocalTensor<float> offsetOutput = offsetOutputBuf_.Get<float>();

    CopyInOffset(taskIdx, offset, mask);
    ComputeWeight(taskIdx, auxW, auxH, offset, offsetInt, weight, mask);

    SetFlag<HardEvent::V_MTE2>(calEvt_);
    WaitFlag<HardEvent::V_MTE2>(calEvt_);
    SetFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::MTE3_V>(1);

    uint8_t ping = 0;
    for (uint32_t w = 0; w < wOut_; ++w) {
        WaitFlag<HardEvent::MTE3_V>(ping);
        ComputeBilinearInterpolation(w, offset, offsetInt, feature, weight, offsetOutput[ping * kwIn_]);
        SetFlag<HardEvent::MTE3_V>(ping);
        ping = 1 - ping;
    }
    WaitFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(1);
}

template<bool modulated>
__aicore__ inline void DeformableConv2dKernel<modulated>::CopyInOffset(
    uint32_t taskIdx, const LocalTensor<float>& offset, const LocalTensor<float>& mask)
{
    uint32_t offsetIdx = taskIdx * rowOffset_ * 2;
    DataCopy(offset, offsetGm_[offsetIdx], {1, doubleRowOffsetBlk_, 0, 0});
    if (modulated) {
        DataCopy(mask, maskGm_[taskIdx * rowOffset_], {1, rowOffsetBlk_, 0, 0});
    }
    SetFlag<HardEvent::MTE2_V>(copyEvt_);
    WaitFlag<HardEvent::MTE2_V>(copyEvt_);
    uint64_t cnt;
    GatherMask(offset[2 * alignedRowOffset_], offset, 2, false, MASK_PLACEHOLDER, gatherParams_, cnt);
    GatherMask(offset[3 * alignedRowOffset_], offset, 1, false, MASK_PLACEHOLDER, gatherParams_, cnt);
    SetVectorMask<float>(FULL_MASK, FULL_MASK);
}

template<bool modulated>
__aicore__ inline void DeformableConv2dKernel<modulated>::ComputeWeight(uint32_t taskIdx,
    const LocalTensor<float>& auxW, const LocalTensor<float>& auxH, const LocalTensor<float>& offset,
    const LocalTensor<int32_t>& offsetInt, const LocalTensor<float>& weight, const LocalTensor<float>& mask)
{
    int32_t h = taskIdx % hOut_;
    Copy<float, false>(offset, auxW, MASK_PLACEHOLDER, rptTimes_, {1, 1, 8, 8});
    Adds<float, false>(offset[alignedRowOffset_], auxH, float(h * strideH_), MASK_PLACEHOLDER, rptTimes_, {1, 1, 8, 8});
    Add<float, false>(
        offset, offset, offset[2 * alignedRowOffset_], MASK_PLACEHOLDER, 2 * rptTimes_, {1, 1, 1, 8, 8, 8});

    Cast<int32_t, float, false>(
        offsetInt, offset, RoundMode::CAST_FLOOR, MASK_PLACEHOLDER, 2 * rptTimes_, {1, 1, 8, 8});
    Cast<float, int32_t, false>(
        offset[2 * alignedRowOffset_], offsetInt, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 2 * rptTimes_, {1, 1, 8, 8});
    Sub<float, false>(
        offset, offset, offset[2 * alignedRowOffset_], MASK_PLACEHOLDER, 2 * rptTimes_, {1, 1, 1, 8, 8, 8}); // lw, lh
    Duplicate<float, false>(weight, 1.f, MASK_PLACEHOLDER, 2 * rptTimes_, 1, 8);
    Sub<float, false>(
        offset[2 * alignedRowOffset_], weight, offset, MASK_PLACEHOLDER, 2 * rptTimes_, {1, 1, 1, 8, 8, 8}); // hw, hh

    Mul<float, false>(weight, offset[2 * alignedRowOffset_], offset[3 * alignedRowOffset_], MASK_PLACEHOLDER, rptTimes_,
        {1, 1, 1, 8, 8, 8}); // hw * hh
    Mul<float, false>(weight[alignedRowOffset_], offset, offset[3 * alignedRowOffset_], MASK_PLACEHOLDER, rptTimes_,
        {1, 1, 1, 8, 8, 8}); // lw * hh
    Mul<float, false>(weight[2 * alignedRowOffset_], offset[alignedRowOffset_], offset[2 * alignedRowOffset_],
        MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8}); // hw * lh
    Mul<float, false>(weight[3 * alignedRowOffset_], offset, offset[alignedRowOffset_], MASK_PLACEHOLDER, rptTimes_,
        {1, 1, 1, 8, 8, 8}); // lh * lw
    if (modulated) {
        Mul<float, false>(weight, weight, mask, MASK_PLACEHOLDER, rptTimes_, {1, 1, 1, 8, 8, 8});
        Mul<float, false>(weight[alignedRowOffset_], weight[alignedRowOffset_], mask, MASK_PLACEHOLDER, rptTimes_,
            {1, 1, 1, 8, 8, 8}); // lw * hh
        Mul<float, false>(weight[2 * alignedRowOffset_], weight[2 * alignedRowOffset_], mask, MASK_PLACEHOLDER,
            rptTimes_, {1, 1, 1, 8, 8, 8}); // hw * lh
        Mul<float, false>(weight[3 * alignedRowOffset_], weight[3 * alignedRowOffset_], mask, MASK_PLACEHOLDER,
            rptTimes_, {1, 1, 1, 8, 8, 8}); // lh * lw
    }
}

template<bool modulated>
__aicore__ inline void DeformableConv2dKernel<modulated>::ComputeBilinearInterpolation(uint32_t w,
    const LocalTensor<float>& offset, const LocalTensor<int32_t>& offsetInt, const LocalTensor<float>& feature,
    const LocalTensor<float>& weight, const LocalTensor<float>& offsetOutput)
{
    Duplicate<float, false>(offsetOutput, 0.f, MASK_PLACEHOLDER, kernelSize_ * valRptTimes_, 1, 8);
    uint8_t ping = 0;
    uint32_t kernelOffset = w * kernelSize_;
    SetFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE2>(1);
#pragma bisheng auto_sync parallel
    for (uint32_t kIdx = 0; kIdx < kernelSize_; ++kIdx) {
        uint32_t pw = kIdx + kernelOffset;
        uint32_t ph = pw + alignedRowOffset_;
        int32_t w0 = offsetInt.GetValue(pw);
        int32_t h0 = offsetInt.GetValue(ph);
        int32_t w1 = w0 + 1;
        int32_t h1 = h0 + 1;
        uint32_t outOffset = kIdx * cIn_;
        uint32_t ftOffset = ping * featureOffset_;
        WaitFlag<HardEvent::V_MTE2>(ping);

        if (0 < h1 && h1 < hIn_) {
            if (0 < w1 && w1 < wIn_) {
                uint64_t gmOffset = srcOffset_ + (h0 * wIn_ + w0) * cIn_;
                DataCopy(feature[ftOffset], xGm_[gmOffset], cpQuadValParams_);
                SetFlag<HardEvent::MTE2_V>(copyEvt_);
                WaitFlag<HardEvent::MTE2_V>(copyEvt_);
                PipeBarrier<PIPE_V>();
                Axpy<float, float, false>(offsetOutput[outOffset], feature[ftOffset], weight.GetValue(pw),
                    MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
                Axpy<float, float, false>(offsetOutput[outOffset], feature[ftOffset + cIn_], weight.GetValue(ph),
                    MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
                Axpy<float, float, false>(offsetOutput[outOffset], feature[ftOffset + 2 * cIn_],
                    weight.GetValue(pw + 2 * alignedRowOffset_), MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
                Axpy<float, float, false>(offsetOutput[outOffset], feature[ftOffset + 3 * cIn_],
                    weight.GetValue(ph + 2 * alignedRowOffset_), MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
            } else if (w1 == 0) {
                uint64_t gmOffset = srcOffset_ + (h0 * wIn_) * cIn_;
                DataCopy(feature[ftOffset + cIn_], xGm_[gmOffset], cpColDoubleValParams_);
                SetFlag<HardEvent::MTE2_V>(copyEvt_);
                WaitFlag<HardEvent::MTE2_V>(copyEvt_);
                PipeBarrier<PIPE_V>();
                Axpy<float, float, false>(offsetOutput[outOffset], feature[ftOffset + cIn_], weight.GetValue(ph),
                    MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
                Axpy<float, float, false>(offsetOutput[outOffset], feature[ftOffset + 3 * cIn_],
                    weight.GetValue(ph + 2 * alignedRowOffset_), MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
            } else if (w1 == wIn_) {
                uint64_t gmOffset = srcOffset_ + (h0 * wIn_ + w0) * cIn_;
                DataCopy(feature[ftOffset], xGm_[gmOffset], cpColDoubleValParams_);
                SetFlag<HardEvent::MTE2_V>(copyEvt_);
                WaitFlag<HardEvent::MTE2_V>(copyEvt_);
                PipeBarrier<PIPE_V>();
                Axpy<float, float, false>(offsetOutput[outOffset], feature[ftOffset], weight.GetValue(pw),
                    MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
                Axpy<float, float, false>(offsetOutput[outOffset], feature[ftOffset + 2 * cIn_],
                    weight.GetValue(pw + 2 * alignedRowOffset_), MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
            }
        } else if (h1 == 0) {
            if (0 < w1 && w1 < wIn_) {
                uint64_t gmOffset = srcOffset_ + w0 * cIn_;
                DataCopy(feature[ftOffset + 2 * cIn_], xGm_[gmOffset], cpRowDoubleValParams_);
                SetFlag<HardEvent::MTE2_V>(copyEvt_);
                WaitFlag<HardEvent::MTE2_V>(copyEvt_);
                PipeBarrier<PIPE_V>();
                Axpy<float, float, false>(offsetOutput[outOffset], feature[ftOffset + 2 * cIn_],
                    weight.GetValue(pw + 2 * alignedRowOffset_), MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
                Axpy<float, float, false>(offsetOutput[outOffset], feature[ftOffset + 3 * cIn_],
                    weight.GetValue(ph + 2 * alignedRowOffset_), MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
            } else if (w1 == 0) {
                uint64_t gmOffset = srcOffset_;
                DataCopy(feature[ftOffset + 3 * cIn_], xGm_[gmOffset], cpOneValParams_);
                SetFlag<HardEvent::MTE2_V>(copyEvt_);
                WaitFlag<HardEvent::MTE2_V>(copyEvt_);
                PipeBarrier<PIPE_V>();
                Axpy<float, float, false>(offsetOutput[outOffset], feature[ftOffset + 3 * cIn_],
                    weight.GetValue(ph + 2 * alignedRowOffset_), MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
            } else if (w1 == wIn_) {
                uint64_t gmOffset = srcOffset_ + w0 * cIn_;
                DataCopy(feature[ftOffset + 2 * cIn_], xGm_[gmOffset], cpOneValParams_);
                SetFlag<HardEvent::MTE2_V>(copyEvt_);
                WaitFlag<HardEvent::MTE2_V>(copyEvt_);
                PipeBarrier<PIPE_V>();
                Axpy<float, float, false>(offsetOutput[outOffset], feature[ftOffset + 2 * cIn_],
                    weight.GetValue(pw + 2 * alignedRowOffset_), MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
            }
        } else if (h1 == hIn_) {
            if (0 < w1 && w1 < wIn_) {
                uint64_t gmOffset = srcOffset_ + (h0 * wIn_ + w0) * cIn_;
                DataCopy(feature[ftOffset], xGm_[gmOffset], cpRowDoubleValParams_);
                SetFlag<HardEvent::MTE2_V>(copyEvt_);
                WaitFlag<HardEvent::MTE2_V>(copyEvt_);
                PipeBarrier<PIPE_V>();
                Axpy<float, float, false>(offsetOutput[outOffset], feature[ftOffset], weight.GetValue(pw),
                    MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
                PipeBarrier<PIPE_V>();
                Axpy<float, float, false>(offsetOutput[outOffset], feature[ftOffset + cIn_], weight.GetValue(ph),
                    MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
            } else if (w1 == 0) {
                uint64_t gmOffset = srcOffset_ + (h0 * wIn_) * cIn_;
                DataCopy(feature[ftOffset + cIn_], xGm_[gmOffset], cpOneValParams_);
                SetFlag<HardEvent::MTE2_V>(copyEvt_);
                WaitFlag<HardEvent::MTE2_V>(copyEvt_);
                PipeBarrier<PIPE_V>();
                Axpy<float, float, false>(offsetOutput[outOffset], feature[ftOffset + cIn_], weight.GetValue(ph),
                    MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
            } else if (w1 == wIn_) {
                uint64_t gmOffset = srcOffset_ + (h0 * wIn_ + w0) * cIn_;
                DataCopy(feature[ftOffset], xGm_[gmOffset], cpOneValParams_);
                SetFlag<HardEvent::MTE2_V>(copyEvt_);
                WaitFlag<HardEvent::MTE2_V>(copyEvt_);
                PipeBarrier<PIPE_V>();
                Axpy<float, float, false>(offsetOutput[outOffset], feature[ftOffset], weight.GetValue(pw),
                    MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
            }
        }
        SetFlag<HardEvent::V_MTE2>(ping);
        ping = 1 - ping;
    }
    SetFlag<HardEvent::V_MTE3>(calEvt_);
    WaitFlag<HardEvent::V_MTE3>(calEvt_);
    for (uint32_t i = 0; i < groups_; ++i) {
        DataCopy(offsetOutputGm_[dstOffset_ + rowInPerGroup_ * i], offsetOutput[i * cInPerGroup_], cpOffsetOutParams_);
    }
    dstOffset_ += kwInPerGroup_;
    WaitFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(1);
}

template<bool modulated>
__aicore__ inline void DeformableConv2dKernel<modulated>::Process()
{
    PreProcess();
    for (uint32_t taskIdx = start_; taskIdx < end_; ++taskIdx) {
        ProcessVector(taskIdx);
        ProcessCube(taskIdx);
    }
    mm_.End();
}

extern "C" __global__ __aicore__ void deformable_conv2d(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR offset,
    GM_ADDR mask, GM_ADDR y, GM_ADDR offsetOutput, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }

    TPipe pipe;

    if (TILING_KEY_IS(0)) {
        DeformableConv2dKernel<false> op;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm_, &(tilingData.mmTilingData));
        op.Init(x, weight, bias, offset, mask, y, offsetOutput, usrWorkspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        DeformableConv2dKernel<true> op;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm_, &(tilingData.mmTilingData));
        op.Init(x, weight, bias, offset, mask, y, offsetOutput, usrWorkspace, &tilingData, &pipe);
        op.Process();
    }
}
