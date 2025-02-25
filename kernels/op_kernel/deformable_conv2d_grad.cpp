#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

constexpr MatmulConfig DEFORMABLE_CONV2D_CFG = GetNormalConfig();

template<bool modulated>
class DeformableConv2dGradKernel {
public:
    using A0Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
    using A1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;

    matmul::Matmul<A0Type, BType, CType, CType, DEFORMABLE_CONV2D_CFG> mm0_;
    matmul::Matmul<A1Type, BType, CType, CType, DEFORMABLE_CONV2D_CFG> mm1_;

    __aicore__ inline DeformableConv2dGradKernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR offset, GM_ADDR mask,
        GM_ADDR offsetOutput, GM_ADDR gradY, GM_ADDR gradX, GM_ADDR gradWeight, GM_ADDR gradBias, GM_ADDR gradOffset,
        GM_ADDR gradMask, GM_ADDR workspace, const DeformableConv2dGradTilingData* tilingData, TPipe* pipe)

    {
        pipe_ = pipe;
        blkIdx_ = GetBlockIdx();
        InitTiling(tilingData);
        InitTask();
        InitGM(x, weight, bias, offset, mask, offsetOutput, gradY, gradX, gradWeight, gradBias, gradOffset, gradMask,
            workspace);
        InitBuffer();
        InitEvent();

        SetVectorMask<float>(FULL_MASK, FULL_MASK);
        SetAtomicNone();
    }

    __aicore__ inline void Process();

protected:
    TPipe* pipe_;
    GlobalTensor<float> xGm_, offsetGm_, weightGm_, offsetOutputGm_, gradYGm_;
    GlobalTensor<float> maskGm_, gradMaskGm_;
    GlobalTensor<float> gradXGm_, gradOffsetGm_, gradWeightGm_;
    GlobalTensor<float> auxHGm_, auxWGm_, gradOffsetOutputGm_;
    TBuf<TPosition::VECCALC> auxHBuf_, auxWBuf_, gradOffsetOutputBuf_, reducedValueBuf_;
    TBuf<TPosition::VECCALC> offsetBuf_, offsetIntBuf_, weightBuf_, maskBuf_, featureBuf_;
    TBuf<TPosition::VECCALC> gradXBuf_, gradOffsetBuf_;

    // from tiling
    uint64_t n_, cIn_, hIn_, wIn_, cOut_, hOut_, wOut_, kH_, kW_;
    uint32_t usedBlkNum_;
    int64_t padH_, padW_, strideH_, strideW_, dilationH_, dilationW_, groups_;
    uint64_t rowOut_, rowOutPerGroup_, kwIn_, kwInPerGroup_, rowIn_, rowOffset_, alignedRowOffset_, kernelSize_,
        kernelPerGroup_, cInPerGroup_, rowInPerGroup_;
    uint16_t kwInBlk_, rowOffsetBlk_, doubleRowOffsetBlk_, cInBlk_;
    uint16_t rptTimes_, valRptTimes_;
    uint32_t blkIdx_;
    uint32_t auxStart_, auxEnd_, start_, end_;
    uint64_t srcOffset_, dstOffset_;

    uint64_t reduceMask_;

    TEventID calEvt_, copyEvt_, cpFeatureEvt_;

    DataCopyParams cpOneValParams_, cpDoubleValParams_, cpOffsetOutParams_;
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
        const LocalTensor<int32_t>& offsetInt, const LocalTensor<float>& mask, const LocalTensor<float>& feature,
        const LocalTensor<float>& weight, const LocalTensor<float>& gradOffsetOutput, const LocalTensor<float>& gradX,
        const LocalTensor<float>& gradOffset, const LocalTensor<float>& reducedValue);

    __aicore__ inline void InitTiling(const DeformableConv2dGradTilingData* tilingData)
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
        rowOut_ = wOut_ * cOut_;
        rowOutPerGroup_ = rowOut_ / groups_;
        kwIn_ = kH_ * kW_ * cIn_;
        kwInBlk_ = Ceil(kwIn_, B32_DATA_NUM_PER_BLOCK);
        rowIn_ = wOut_ * kwIn_;
        rowOffset_ = wOut_ * kH_ * kW_;
        kwInPerGroup_ = kwIn_ / groups_;
        rowInPerGroup_ = rowIn_ / groups_;
        cInPerGroup_ = cIn_ / groups_;
        kernelPerGroup_ = cOut_ / groups_ * kwInPerGroup_;
        alignedRowOffset_ = AlignUp(rowOffset_, B32_DATA_NUM_PER_REPEAT);
        rowOffsetBlk_ = Ceil(rowOffset_, B32_DATA_NUM_PER_BLOCK);
        doubleRowOffsetBlk_ = Ceil(2 * rowOffset_, B32_DATA_NUM_PER_BLOCK);
        cInBlk_ = Ceil(cIn_, B32_DATA_NUM_PER_BLOCK);

        cpOneValParams_.blockLen = cInBlk_;
        cpDoubleValParams_.blockLen = 2 * cInBlk_;
        cpOffsetOutParams_.blockCount = kernelSize_;
        cpOffsetOutParams_.blockLen = cInBlk_ / groups_;
        cpOffsetOutParams_.dstStride = cInBlk_ - cInBlk_ / groups_;
        rptTimes_ = alignedRowOffset_ / B32_DATA_NUM_PER_REPEAT;
        valRptTimes_ = cIn_ / B32_DATA_NUM_PER_REPEAT;
        gatherParams_.repeatTimes = rptTimes_ * 2;

        reduceMask_ = (1 << (cIn_ / B32_DATA_NUM_PER_REPEAT)) - 1;
    }

    __aicore__ inline void InitEvent()
    {
        calEvt_ = pipe_->AllocEventID<HardEvent::V_MTE3>();
        copyEvt_ = pipe_->AllocEventID<HardEvent::MTE2_V>();
        cpFeatureEvt_ = pipe_->AllocEventID<HardEvent::V_MTE2>();
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
        pipe_->InitBuffer(gradOffsetOutputBuf_, kwIn_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(gradXBuf_, 4 * cIn_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(featureBuf_, 4 * cIn_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(gradOffsetBuf_, 4 * cIn_ * B32_BYTE_SIZE);
        pipe_->InitBuffer(reducedValueBuf_, 4 * cIn_ * B32_BYTE_SIZE);
    }
    __aicore__ inline void InitGM(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR offset, GM_ADDR mask,
        GM_ADDR offsetOutput, GM_ADDR gradY, GM_ADDR gradX, GM_ADDR gradWeight, GM_ADDR gradBias, GM_ADDR gradOffset,
        GM_ADDR gradMask, GM_ADDR workspace)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight));
        offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(offset));
        offsetOutputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(offsetOutput));
        gradYGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradY));
        gradXGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradX));
        gradOffsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradOffset));
        gradWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradWeight));

        auxHGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        auxWGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + rowOffset_);
        gradOffsetOutputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + 2 * rowOffset_);
        if (modulated) {
            maskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(mask));
            gradMaskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradMask));
        }
    }
};

template<bool modulated>
__aicore__ inline void DeformableConv2dGradKernel<modulated>::PreProcess()
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
    DataCopyPad(auxWGm_[auxStart_ * kH_ * kW_], auxW,
        {1, static_cast<uint16_t>(B32_BYTE_SIZE * (auxEnd_ - auxStart_) * kH_ * kW_), 0, 0});
    DataCopyPad(auxHGm_[auxStart_ * kH_ * kW_], auxH,
        {1, static_cast<uint16_t>(B32_BYTE_SIZE * (auxEnd_ - auxStart_) * kH_ * kW_), 0, 0});
    SyncAll();
    DataCopy(auxW, auxWGm_, {1, rowOffsetBlk_, 0, 0});
    DataCopy(auxH, auxHGm_, {1, rowOffsetBlk_, 0, 0});

    LocalTensor<float> feature = featureBuf_.Get<float>();
    Duplicate<float, false>(feature, 0.f, MASK_PLACEHOLDER, 4 * valRptTimes_, 1, 8);
}

template<bool modulated>
__aicore__ inline void DeformableConv2dGradKernel<modulated>::ProcessCube(uint32_t taskIdx)
{
    uint64_t aOffset = 0;
    uint64_t bOffset = taskIdx * rowIn_;
    uint64_t cOffset = taskIdx * rowOut_;
    for (uint32_t i = 0; i < groups_ - 1; ++i) {
        mm0_.SetTensorA(gradYGm_[cOffset], true);
        mm0_.SetTensorB(weightGm_[aOffset]);
        mm0_.template IterateAll<false>(gradOffsetOutputGm_[bOffset]);
        aOffset += kernelPerGroup_;
        bOffset += rowInPerGroup_;
        cOffset += rowOutPerGroup_;
    }
    mm0_.SetTensorA(gradYGm_[cOffset], true);
    mm0_.SetTensorB(weightGm_[aOffset]);
    mm0_.template IterateAll<false>(gradOffsetOutputGm_[bOffset], 0, false, true);

    for (uint32_t i = 0; i < groups_; ++i) {
        mm1_.SetTensorA(gradYGm_[cOffset]);
        mm1_.SetTensorB(offsetOutputGm_[bOffset]);
        mm1_.template IterateAll<false>(gradWeightGm_[aOffset], true);
        aOffset -= kernelPerGroup_;
        bOffset -= rowInPerGroup_;
        cOffset -= rowOutPerGroup_;
    }
}
template<bool modulated>
__aicore__ inline void DeformableConv2dGradKernel<modulated>::ProcessVector(uint32_t taskIdx)
{
    uint32_t batch = taskIdx / hOut_;
    srcOffset_ = batch * hIn_ * wIn_ * cIn_;
    dstOffset_ = taskIdx * wOut_ * kH_ * kW_;
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

    CopyInOffset(taskIdx, offset, mask);
    ComputeWeight(taskIdx, auxW, auxH, offset, offsetInt, weight, mask);

    mm0_.WaitIterateAll();

    uint32_t gradOffsetIdx = taskIdx * rowIn_;
    LocalTensor<float> gradOffsetOutput = gradOffsetOutputBuf_.Get<float>();
    LocalTensor<float> gradX = gradXBuf_.Get<float>();
    LocalTensor<float> gradOffset = gradOffsetBuf_.Get<float>();
    LocalTensor<float> reducedValue = reducedValueBuf_.Get<float>();

    SetFlag<HardEvent::V_MTE2>(cpFeatureEvt_);
    for (uint32_t w = 0; w < wOut_; ++w) {
        for (uint32_t i = 0; i < groups_; ++i) {
            DataCopy(gradOffsetOutput[i * cInPerGroup_], gradOffsetOutputGm_[gradOffsetIdx + rowInPerGroup_ * i],
                cpOffsetOutParams_);
        }

        SetFlag<HardEvent::MTE2_V>(copyEvt_);
        WaitFlag<HardEvent::MTE2_V>(copyEvt_);
        ComputeBilinearInterpolation(
            w, offset, offsetInt, mask, feature, weight, gradOffsetOutput, gradX, gradOffset, reducedValue);
        gradOffsetIdx += kwInPerGroup_;
    }
    WaitFlag<HardEvent::V_MTE2>(cpFeatureEvt_);
}

template<bool modulated>
__aicore__ inline void DeformableConv2dGradKernel<modulated>::CopyInOffset(
    uint32_t taskIdx, const LocalTensor<float>& offset, const LocalTensor<float>& mask)
{
    uint64_t offsetIdx = taskIdx * rowOffset_ * 2;
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
__aicore__ inline void DeformableConv2dGradKernel<modulated>::ComputeWeight(uint32_t taskIdx,
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
__aicore__ inline void DeformableConv2dGradKernel<modulated>::ComputeBilinearInterpolation(uint32_t w,
    const LocalTensor<float>& offset, const LocalTensor<int32_t>& offsetInt, const LocalTensor<float>& mask,
    const LocalTensor<float>& feature, const LocalTensor<float>& weight, const LocalTensor<float>& gradOffsetOutput,
    const LocalTensor<float>& gradX, const LocalTensor<float>& gradOffset, const LocalTensor<float>& reducedValue)
{
    uint32_t kernelOffset = w * kernelSize_;
    for (uint32_t kIdx = 0; kIdx < kernelSize_; ++kIdx) {
        uint32_t pw = kIdx + kernelOffset;
        uint32_t ph = pw + alignedRowOffset_;
        int32_t w0 = offsetInt.GetValue(pw);
        int32_t h0 = offsetInt.GetValue(ph);
        int32_t w1 = w0 + 1;
        int32_t h1 = h0 + 1;
        uint32_t outOffset = kIdx * cIn_;

        WaitFlag<HardEvent::V_MTE2>(cpFeatureEvt_);
        if (0 <= h0 && h0 < hIn_) {
            if (0 < w1 && w1 < wIn_) {
                uint32_t ubOffset = 0;
                uint64_t gmOffset = srcOffset_ + (h0 * wIn_ + w0) * cIn_;
                DataCopy(feature[ubOffset], xGm_[gmOffset], cpDoubleValParams_);
                Muls<float, false>(gradX[ubOffset], gradOffsetOutput[outOffset], weight.GetValue(pw), MASK_PLACEHOLDER,
                    valRptTimes_, {1, 1, 8, 8});
                Muls<float, false>(gradX[ubOffset + cIn_], gradOffsetOutput[outOffset], weight.GetValue(ph),
                    MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
                SetFlag<HardEvent::V_MTE3>(calEvt_);
                WaitFlag<HardEvent::V_MTE3>(calEvt_);
                SetAtomicAdd<float>();
                DataCopy(gradXGm_[gmOffset], gradX[ubOffset], cpDoubleValParams_);
                SetAtomicNone();
            } else if (0 <= w0 && w0 < wIn_) {
                uint32_t ubOffset = 0;
                uint64_t gmOffset = srcOffset_ + (h0 * wIn_ + w0) * cIn_;
                DataCopy(feature[ubOffset], xGm_[gmOffset], cpOneValParams_);
                Muls<float, false>(gradX[ubOffset], gradOffsetOutput[outOffset], weight.GetValue(pw), MASK_PLACEHOLDER,
                    valRptTimes_, {1, 1, 8, 8});
                SetFlag<HardEvent::V_MTE3>(calEvt_);
                WaitFlag<HardEvent::V_MTE3>(calEvt_);
                SetAtomicAdd<float>();
                DataCopy(gradXGm_[gmOffset], gradX[ubOffset], cpOneValParams_);
                SetAtomicNone();
            } else if (0 <= w1 && w1 < wIn_) {
                uint32_t ubOffset = cIn_;
                uint64_t gmOffset = srcOffset_ + (h0 * wIn_ + w1) * cIn_;
                DataCopy(feature[ubOffset], xGm_[gmOffset], cpOneValParams_);
                Muls<float, false>(gradX[ubOffset], gradOffsetOutput[outOffset], weight.GetValue(ph), MASK_PLACEHOLDER,
                    valRptTimes_, {1, 1, 8, 8});
                SetFlag<HardEvent::V_MTE3>(calEvt_);
                WaitFlag<HardEvent::V_MTE3>(calEvt_);
                SetAtomicAdd<float>();
                DataCopy(gradXGm_[gmOffset], gradX[ubOffset], cpOneValParams_);
                SetAtomicNone();
            }
        }
        if (0 <= h1 && h1 < hIn_) {
            if (0 < w1 && w1 < wIn_) {
                uint32_t ubOffset = 2 * cIn_;
                uint64_t gmOffset = srcOffset_ + (h1 * wIn_ + w0) * cIn_;
                DataCopy(feature[ubOffset], xGm_[gmOffset], cpDoubleValParams_);
                Muls<float, false>(gradX[ubOffset], gradOffsetOutput[outOffset],
                    weight.GetValue(pw + 2 * alignedRowOffset_), MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
                Muls<float, false>(gradX[ubOffset + cIn_], gradOffsetOutput[outOffset],
                    weight.GetValue(ph + 2 * alignedRowOffset_), MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
                SetFlag<HardEvent::V_MTE3>(calEvt_);
                WaitFlag<HardEvent::V_MTE3>(calEvt_);
                SetAtomicAdd<float>();
                DataCopy(gradXGm_[gmOffset], gradX[ubOffset], cpDoubleValParams_);
                SetAtomicNone();
            } else if (0 <= w0 && w0 < wIn_) {
                uint32_t ubOffset = 2 * cIn_;
                uint64_t gmOffset = srcOffset_ + (h1 * wIn_ + w0) * cIn_;
                DataCopy(feature[ubOffset], xGm_[gmOffset], cpOneValParams_);
                Muls<float, false>(gradX[ubOffset], gradOffsetOutput[outOffset],
                    weight.GetValue(pw + 2 * alignedRowOffset_), MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
                SetFlag<HardEvent::V_MTE3>(calEvt_);
                WaitFlag<HardEvent::V_MTE3>(calEvt_);
                SetAtomicAdd<float>();
                DataCopy(gradXGm_[gmOffset], gradX[ubOffset], cpOneValParams_);
                SetAtomicNone();
            } else if (0 <= w1 && w1 < wIn_) {
                uint32_t ubOffset = 3 * cIn_;
                uint64_t gmOffset = srcOffset_ + (h1 * wIn_ + w1) * cIn_;
                DataCopy(feature[ubOffset], xGm_[gmOffset], cpOneValParams_);
                Muls<float, false>(gradX[ubOffset], gradOffsetOutput[outOffset],
                    weight.GetValue(ph + 2 * alignedRowOffset_), MASK_PLACEHOLDER, valRptTimes_, {1, 1, 8, 8});
                SetFlag<HardEvent::V_MTE3>(calEvt_);
                WaitFlag<HardEvent::V_MTE3>(calEvt_);
                SetAtomicAdd<float>();
                DataCopy(gradXGm_[gmOffset], gradX[ubOffset], cpOneValParams_);
                SetAtomicNone();
            }
        }
        SetFlag<HardEvent::MTE2_V>(copyEvt_);
        WaitFlag<HardEvent::MTE2_V>(copyEvt_);
        for (uint32_t i = 0; i < 4; ++i) {
            Mul<float, false>(gradOffset[i * cIn_], feature[i * cIn_], gradOffsetOutput[outOffset], MASK_PLACEHOLDER,
                valRptTimes_, {1, 1, 1, 8, 8, 8});
        }

        Duplicate<float, false>(feature, 0.f, MASK_PLACEHOLDER, 4 * valRptTimes_, 1, 8);
        SetFlag<HardEvent::V_MTE2>(cpFeatureEvt_);
        for (uint32_t i = 0; i < 4; ++i) {
            WholeReduceSum<float, false>(
                reducedValue[i * 8], gradOffset[i * cIn_], MASK_PLACEHOLDER, valRptTimes_, 1, 1, 8);
        }
        SetVectorMask<float>(0, reduceMask_);
        WholeReduceSum<float, false>(gradOffset[32], reducedValue, MASK_PLACEHOLDER, 4, 1, 1, 1);
        float lw = offset.GetValue(pw);
        float lh = offset.GetValue(ph);
        float hw = offset.GetValue(pw + 2 * alignedRowOffset_);
        float hh = offset.GetValue(ph + 2 * alignedRowOffset_);
        float a = gradOffset.GetValue(32);
        float b = gradOffset.GetValue(33);
        float c = gradOffset.GetValue(34);
        float d = gradOffset.GetValue(35);

        if (modulated) {
            float scale = mask.GetValue(pw);
            gradOffset.SetValue(0, (-a * hw - b * lw + c * hw + d * lw) * scale);
            gradOffset.SetValue(1, (-a * hh + b * hh - c * lh + d * lh) * scale);
            gradOffset.SetValue(8, (a * hh * hw + b * lw * hh + c * hw * lh + d * lw * lh));
            DataCopyPad(gradOffsetGm_[dstOffset_ * 2], gradOffset, {1, 8, 0, 0});
            DataCopyPad(gradMaskGm_[dstOffset_], gradOffset[8], {1, 4, 0, 0});
            PipeBarrier<PIPE_ALL>();
        } else {
            gradOffset.SetValue(0, -a * hw - b * lw + c * hw + d * lw);
            gradOffset.SetValue(1, -a * hh + b * hh - c * lh + d * lh);
            DataCopyPad(gradOffsetGm_[dstOffset_ * 2], gradOffset, {1, 8, 0, 0});
            PipeBarrier<PIPE_ALL>();
        }
        dstOffset_ += 1;

        SetVectorMask<float>(FULL_MASK, FULL_MASK);
    }
}

template<bool modulated>
__aicore__ inline void DeformableConv2dGradKernel<modulated>::Process()
{
    PreProcess();
    for (uint32_t taskIdx = start_; taskIdx < end_; ++taskIdx) {
        ProcessCube(taskIdx);
        ProcessVector(taskIdx);
    }
    mm0_.End();
    mm1_.End();
}

extern "C" __global__ __aicore__ void deformable_conv2d_grad(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR offset,
    GM_ADDR mask, GM_ADDR offsetOutput, GM_ADDR gradY, GM_ADDR gradX, GM_ADDR gradWeight, GM_ADDR gradBias,
    GM_ADDR gradOffset, GM_ADDR gradMask, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }

    TPipe pipe;

    if (TILING_KEY_IS(0)) {
        DeformableConv2dGradKernel<false> op;
        REGIST_MATMUL_OBJ(
            &pipe, GetSysWorkSpacePtr(), op.mm0_, &(tilingData.mm0TilingData), op.mm1_, &(tilingData.mm1TilingData));
        op.Init(x, weight, bias, offset, mask, offsetOutput, gradY, gradX, gradWeight, gradBias, gradOffset, gradMask,
            usrWorkspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        DeformableConv2dGradKernel<true> op;
        REGIST_MATMUL_OBJ(
            &pipe, GetSysWorkSpacePtr(), op.mm0_, &(tilingData.mm0TilingData), op.mm1_, &(tilingData.mm1TilingData));
        op.Init(x, weight, bias, offset, mask, offsetOutput, gradY, gradX, gradWeight, gradBias, gradOffset, gradMask,
            usrWorkspace, &tilingData, &pipe);
        op.Process();
    }
}
