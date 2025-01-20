/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#include <climits>

#include "kernel_operator.h"
using namespace AscendC;

namespace {
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t GRID_CHANNEL = 2;
constexpr int32_t COMPARE_ALIGN_BYTE = 256;

enum PaddingMode {
    ZEROS,
    BORDER,
    REFLECTION
};
} // namespace

class GridSampler2dV2Kernel {
public:
    __aicore__ inline GridSampler2dV2Kernel() = delete;

    __aicore__ inline ~GridSampler2dV2Kernel() = default;

    __aicore__ inline GridSampler2dV2Kernel(
        GM_ADDR xTrans, GM_ADDR grid, GM_ADDR yTrans, const GridSampler2dV2TilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        InitTiling(tilingData);
        InitTask();
        InitGm(xTrans, grid, yTrans);
        InitBuffer();
        InitLocalTensor();
        InitEvent();
    }

    __aicore__ inline void Process();

private:
    __aicore__ inline void InitTiling(const GridSampler2dV2TilingData* tilingData)
    {
        interpolationMode_ = tilingData->interpolationMode;
        paddingMode_ = tilingData->paddingMode;
        alignCorners_ = tilingData->alignCorners;
        batchSize_ = tilingData->batchSize;
        channel_ = tilingData->channel;
        inHeight_ = tilingData->inHeight;
        inWidth_ = tilingData->inWidth;
        outHeight_ = tilingData->outHeight;
        outWidth_ = tilingData->outWidth;
        taskNumPerCore_ = tilingData->taskNumPerCore;
        usedCoreNum_ = tilingData->usedCoreNum;
        alignedChannel_ = tilingData->alignedChannel;
        alignedTaskNumPerLoop_ = tilingData->alignedTaskNumPerLoop;
        copyLoop_ = tilingData->copyLoop;
        copyTail_ = tilingData->copyTail;
        lastCopyLoop_ = tilingData->lastCopyLoop;
        lastCopyTail_ = tilingData->lastCopyTail;
        coordPosition_ = tilingData->coordPosition;
        groupSize_ = tilingData->groupSize;

        gridStrideN_ = outHeight_ * outWidth_ * GRID_CHANNEL;
        gridStrideH_ = outWidth_ * GRID_CHANNEL;
        gridStrideW_ = GRID_CHANNEL;
        inStrideN_ = inHeight_ * inWidth_ * channel_;
        inStrideH_ = inWidth_ * channel_;
        inStrideW_ = channel_;
        nOffsetStride_ = outHeight_ * outWidth_;
        alignedGridNumPerCore_ = alignedTaskNumPerLoop_ * GRID_CHANNEL;
        alignedCompareNumPerCore_ = AlignUp(alignedTaskNumPerLoop_, COMPARE_ALIGN_BYTE / sizeof(float));
        alignedMaskNum_ = AlignUp(alignedTaskNumPerLoop_, ONE_BLK_SIZE / sizeof(uint8_t));
        alignedChannelBlk_ = alignedChannel_ / B32_DATA_NUM_PER_BLOCK;

        gatherParams_.repeatTimes = AlignUp(alignedGridNumPerCore_, B32_DATA_NUM_PER_REPEAT) / B32_DATA_NUM_PER_REPEAT;
        inputCpParams_.blockLen = alignedChannelBlk_;
        outputCpOutBlockLen_ = alignedChannelBlk_;

        brcbRptTimes_ = groupSize_ / BRCB_BROADCAST_NUMBER;
        brcbRptParams_.dstBlkStride = alignedChannelBlk_;
        brcbRptParams_.dstRepStride = alignedChannelBlk_ * BRCB_BROADCAST_NUMBER;

        weightAddsRptCount_ = DivCeil(groupSize_ * coordPosition_ * B32_DATA_NUM_PER_BLOCK, B32_DATA_NUM_PER_REPEAT);
        weightAddsRptParams_.dstBlkStride = alignedChannelBlk_;
        weightAddsRptParams_.srcBlkStride = alignedChannelBlk_;
        weightAddsRptParams_.dstRepStride = alignedChannelBlk_ * B32_DATA_NUM_PER_BLOCK;
        weightAddsRptParams_.srcRepStride = alignedChannelBlk_ * B32_DATA_NUM_PER_BLOCK;
    }

    __aicore__ inline void InitTask()
    {
        coreId_ = GetBlockIdx();
        if (coreId_ == (usedCoreNum_ - 1)) {
            copyLoop_ = lastCopyLoop_;
            copyTail_ = lastCopyTail_;
        }
        startTaskIdx_ = coreId_ * taskNumPerCore_;
        computeCount_ = alignedTaskNumPerLoop_;
        nwWeightStart_ = 0;
        neWeightStart_ = alignedTaskNumPerLoop_;
        swWeightStart_ = alignedTaskNumPerLoop_ * 2;
        seWeightStart_ = alignedTaskNumPerLoop_ * 3;
        nwCpInOffsetStart_ = 0;
        neCpInOffsetStart_ = groupSize_ * alignedChannel_;
        swCpInOffsetStart_ = groupSize_ * alignedChannel_ * 2;
        seCpInOffsetStart_ = groupSize_ * alignedChannel_ * 3;
    }

    __aicore__ inline void InitGm(GM_ADDR xTrans, GM_ADDR grid, GM_ADDR yTrans)
    {
        xTransGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xTrans),
            static_cast<uint64_t>(batchSize_) * inHeight_ * inWidth_ * channel_);
        gridGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(grid),
            static_cast<uint64_t>(batchSize_) * outHeight_ * outWidth_ * GRID_CHANNEL);
        yTransGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(yTrans),
            static_cast<uint64_t>(batchSize_) * outHeight_ * outWidth_ * alignedChannel_);
    }

    __aicore__ inline void InitBuffer()
    {
        pipe_->InitBuffer(gridBuf_, alignedGridNumPerCore_ * sizeof(float));
        pipe_->InitBuffer(inputXYFpBuf_, alignedGridNumPerCore_ * sizeof(float));
        pipe_->InitBuffer(inputXFpBuf_, alignedTaskNumPerLoop_ * sizeof(float));
        pipe_->InitBuffer(inputYFpBuf_, alignedTaskNumPerLoop_ * sizeof(float));
        pipe_->InitBuffer(extraFpPointBuf_, alignedTaskNumPerLoop_ * sizeof(float));
        pipe_->InitBuffer(flipFloatBuf_, alignedTaskNumPerLoop_ * sizeof(float));
        pipe_->InitBuffer(modFloatBuf_, alignedTaskNumPerLoop_ * sizeof(float));
        pipe_->InitBuffer(mask1Buf_, alignedMaskNum_ * sizeof(uint8_t));
        pipe_->InitBuffer(mask2Buf_, alignedMaskNum_ * sizeof(uint8_t));
        pipe_->InitBuffer(selTensor1Buf_, alignedTaskNumPerLoop_ * sizeof(float));
        pipe_->InitBuffer(selTensor2Buf_, alignedTaskNumPerLoop_ * sizeof(float));
        pipe_->InitBuffer(selTensor3Buf_, alignedTaskNumPerLoop_ * sizeof(float));
        pipe_->InitBuffer(selTensor4Buf_, alignedTaskNumPerLoop_ * sizeof(float));
        pipe_->InitBuffer(tmpFloatBuf_, alignedTaskNumPerLoop_ * sizeof(float));
        pipe_->InitBuffer(tmpIntBuf_, alignedTaskNumPerLoop_ * sizeof(int32_t));
        pipe_->InitBuffer(dupOneBuf_, alignedTaskNumPerLoop_ * sizeof(float));
        pipe_->InitBuffer(inputXWFpBuf_, alignedTaskNumPerLoop_ * sizeof(float));
        pipe_->InitBuffer(inputXEFpBuf_, alignedTaskNumPerLoop_ * sizeof(float));
        pipe_->InitBuffer(inputYNFpBuf_, alignedTaskNumPerLoop_ * sizeof(float));
        pipe_->InitBuffer(inputYSFpBuf_, alignedTaskNumPerLoop_ * sizeof(float));
        pipe_->InitBuffer(inputXWIntBuf_, alignedTaskNumPerLoop_ * sizeof(int32_t));
        pipe_->InitBuffer(inputXEIntBuf_, alignedTaskNumPerLoop_ * sizeof(int32_t));
        pipe_->InitBuffer(inputYNIntBuf_, alignedTaskNumPerLoop_ * sizeof(int32_t));
        pipe_->InitBuffer(inputYSIntBuf_, alignedTaskNumPerLoop_ * sizeof(int32_t));
        pipe_->InitBuffer(nwOffsetIntBuf_, alignedTaskNumPerLoop_ * sizeof(int32_t));
        pipe_->InitBuffer(neOffsetIntBuf_, alignedTaskNumPerLoop_ * sizeof(int32_t));
        pipe_->InitBuffer(swOffsetIntBuf_, alignedTaskNumPerLoop_ * sizeof(int32_t));
        pipe_->InitBuffer(seOffsetIntBuf_, alignedTaskNumPerLoop_ * sizeof(int32_t));
        pipe_->InitBuffer(inputXCpInBuf_, alignedChannel_ * coordPosition_ * groupSize_ * BUFFER_NUM * sizeof(float));
        pipe_->InitBuffer(outputCpOutBuf_, alignedChannel_ * coordPosition_ * groupSize_ * BUFFER_NUM * sizeof(float));
        pipe_->InitBuffer(weightBuf_, alignedTaskNumPerLoop_ * coordPosition_ * sizeof(float));
        pipe_->InitBuffer(weightBrcbBuf_, alignedChannel_ * coordPosition_ * groupSize_ * sizeof(float));
    }

    __aicore__ inline void InitLocalTensor()
    {
        dupOneTensor_ = dupOneBuf_.Get<float>();
        selMask1_ = mask1Buf_.Get<uint8_t>();
        selMask2_ = mask2Buf_.Get<uint8_t>();
        weight_ = weightBuf_.Get<float>();
    }

    __aicore__ inline void InitEvent()
    {
        eventIdMte2ToV_ = pipe_->FetchEventID(HardEvent::MTE2_V);
        eventIdVToMte2_ = pipe_->FetchEventID(HardEvent::V_MTE2);
    }

    __aicore__ inline void CompensatePrecision(
        const LocalTensor<float>& floor, const LocalTensor<float>& dividend, float divisor, int32_t calCount);

    __aicore__ inline void UnnormalizeCoord(const LocalTensor<float>& coord, int32_t size);

    __aicore__ inline void ClipCoord(const LocalTensor<float>& coord, int32_t size);

    __aicore__ inline void ReflectCoord(const LocalTensor<float>& coord, int32_t twiceLow, int32_t twiceHigh);

    __aicore__ inline void SafeDowngradeToIntRange(const LocalTensor<float>& coord);

    __aicore__ inline void ComputeSourceIndex(const LocalTensor<float>& coord, int32_t size);

    __aicore__ inline uint64_t GetGmOffset(const uint64_t index, const int32_t width, const int32_t height,
        const uint64_t strideN, const uint64_t strideH, const uint64_t strideW);

    __aicore__ inline void ComputeCoord(const LocalTensor<float>& inputXFp, const LocalTensor<float>& inputYFp);

    __aicore__ inline void ComputeWeightSub(const LocalTensor<float>& w1, const LocalTensor<float>& w2,
        const LocalTensor<float>& x1, const LocalTensor<float>& x2, const LocalTensor<float>& y1,
        const LocalTensor<float>& y2);

    __aicore__ inline void ComputeBilinearWeight(const LocalTensor<float>& inputXFp, const LocalTensor<float>& inputYFp,
        const LocalTensor<int32_t>& inputXWInt, const LocalTensor<int32_t>& inputXEInt,
        const LocalTensor<int32_t>& inputYNInt, const LocalTensor<int32_t>& inputYSInt,
        const LocalTensor<float>& inputXWFp, const LocalTensor<float>& inputXEFp, const LocalTensor<float>& inputYNFp,
        const LocalTensor<float>& inputYSFp, const LocalTensor<float>& nwWeight, const LocalTensor<float>& neWeight,
        const LocalTensor<float>& swWeight, const LocalTensor<float>& seWeight);

    __aicore__ inline void WithinBounds2d(const LocalTensor<float>& inputXFp, const LocalTensor<float>& inputYFp,
        const LocalTensor<float>& weight, const LocalTensor<float>& selTensor);

    __aicore__ inline void GetInputOffset(const LocalTensor<float>& inputXFp, const LocalTensor<float>& inputYFp,
        const LocalTensor<float>& offset, LocalTensor<float>& flag);

    __aicore__ inline void CopyInXtrans(int32_t groupOffset, int32_t idxInGroup,
        const LocalTensor<float>& inputXCpInPing, const LocalTensor<int32_t>& nwOffsetInt,
        const LocalTensor<int32_t>& neOffsetInt, const LocalTensor<int32_t>& swOffsetInt,
        const LocalTensor<int32_t>& seOffsetInt);

    __aicore__ inline void ComputeBilinearPerGroup(uint8_t& ping, uint8_t& pong, int32_t groupOffset,
        uint16_t groupSize, const LocalTensor<float>& inputXCpInPing, const LocalTensor<float>& weightBrcb,
        const LocalTensor<float>& outputCpOutPong, const LocalTensor<int32_t>& nwOffsetInt,
        const LocalTensor<int32_t>& neOffsetInt, const LocalTensor<int32_t>& swOffsetInt,
        const LocalTensor<int32_t>& seOffsetInt, const LocalTensor<float>& nwWeight, const LocalTensor<float>& neWeight,
        const LocalTensor<float>& swWeight, const LocalTensor<float>& seWeight);

    __aicore__ inline void ComputeBilinear(const LocalTensor<float>& inputXFp, const LocalTensor<float>& inputYFp);

    __aicore__ inline void ProcessPerLoop();

private:
    TPipe* pipe_;

    GlobalTensor<float> xTransGm_;
    GlobalTensor<float> gridGm_;
    GlobalTensor<float> yTransGm_;

    TBuf<TPosition::VECCALC> gridBuf_;
    TBuf<TPosition::VECCALC> inputXYFpBuf_;
    TBuf<TPosition::VECCALC> inputXFpBuf_;
    TBuf<TPosition::VECCALC> inputYFpBuf_;
    TBuf<TPosition::VECCALC> extraFpPointBuf_;
    TBuf<TPosition::VECCALC> flipFloatBuf_;
    TBuf<TPosition::VECCALC> modFloatBuf_;
    TBuf<TPosition::VECCALC> mask1Buf_;
    TBuf<TPosition::VECCALC> mask2Buf_;
    TBuf<TPosition::VECCALC> selTensor1Buf_;
    TBuf<TPosition::VECCALC> selTensor2Buf_;
    TBuf<TPosition::VECCALC> selTensor3Buf_;
    TBuf<TPosition::VECCALC> selTensor4Buf_;
    TBuf<TPosition::VECCALC> tmpFloatBuf_;
    TBuf<TPosition::VECCALC> tmpIntBuf_;
    TBuf<TPosition::VECCALC> dupOneBuf_;
    TBuf<TPosition::VECCALC> inputXWFpBuf_;
    TBuf<TPosition::VECCALC> inputXEFpBuf_;
    TBuf<TPosition::VECCALC> inputYNFpBuf_;
    TBuf<TPosition::VECCALC> inputYSFpBuf_;
    TBuf<TPosition::VECCALC> inputXWIntBuf_;
    TBuf<TPosition::VECCALC> inputXEIntBuf_;
    TBuf<TPosition::VECCALC> inputYNIntBuf_;
    TBuf<TPosition::VECCALC> inputYSIntBuf_;
    TBuf<TPosition::VECCALC> nwOffsetIntBuf_;
    TBuf<TPosition::VECCALC> neOffsetIntBuf_;
    TBuf<TPosition::VECCALC> swOffsetIntBuf_;
    TBuf<TPosition::VECCALC> seOffsetIntBuf_;
    TBuf<TPosition::VECCALC> inputXCpInBuf_;
    TBuf<TPosition::VECCALC> outputCpOutBuf_;
    TBuf<TPosition::VECCALC> weightBuf_;
    TBuf<TPosition::VECCALC> weightBrcbBuf_;

    TEventID eventIdMte2ToV_;
    TEventID eventIdVToMte2_;

    int64_t interpolationMode_;
    int64_t paddingMode_;
    bool alignCorners_;
    int32_t batchSize_;
    int32_t channel_;
    int32_t inHeight_;
    int32_t inWidth_;
    int32_t outHeight_;
    int32_t outWidth_;
    int32_t taskNumPerCore_;
    int32_t usedCoreNum_;
    int32_t alignedChannel_;
    int32_t alignedTaskNumPerLoop_;
    int32_t copyLoop_;
    int32_t copyTail_;
    int32_t lastCopyLoop_;
    int32_t lastCopyTail_;
    int32_t coordPosition_;
    int32_t groupSize_;

    uint64_t gridStrideN_;
    uint64_t gridStrideH_;
    uint64_t gridStrideW_;
    int32_t inStrideN_;
    int32_t inStrideH_;
    int32_t inStrideW_;
    int32_t nOffsetStride_;
    int32_t alignedGridNumPerCore_;
    int32_t alignedCompareNumPerCore_;
    int32_t alignedMaskNum_;
    int32_t alignedChannelBlk_;
    uint16_t outputCpOutBlockLen_;

    uint64_t coreId_;
    uint64_t startTaskIdx_;
    uint64_t endTaskIdx_;
    int32_t computeCount_;
    int32_t nwWeightStart_;
    int32_t neWeightStart_;
    int32_t swWeightStart_;
    int32_t seWeightStart_;
    int32_t nwCpInOffsetStart_;
    int32_t neCpInOffsetStart_;
    int32_t swCpInOffsetStart_;
    int32_t seCpInOffsetStart_;

    GatherMaskParams gatherParams_;
    DataCopyParams inputCpParams_ {1, 0, 0, 0};
    uint8_t brcbRptTimes_;
    BrcbRepeatParams brcbRptParams_;
    uint8_t weightAddsRptCount_;
    UnaryRepeatParams weightAddsRptParams_;

    LocalTensor<uint8_t> selMask1_;
    LocalTensor<uint8_t> selMask2_;
    LocalTensor<uint16_t> int8ToInt16Mask1_;
    LocalTensor<uint16_t> int8ToInt16Mask2_;
    LocalTensor<float> dupOneTensor_;
    LocalTensor<float> weight_;
};

__aicore__ inline void GridSampler2dV2Kernel::UnnormalizeCoord(const LocalTensor<float>& coord, const int32_t size)
{
    if (alignCorners_) {
        Muls(coord, coord, (size - 1.f) * 0.5f, alignedTaskNumPerLoop_);
    } else {
        Muls(coord, coord, size * 0.5f, alignedTaskNumPerLoop_);
        Adds(coord, coord, -0.5f, alignedTaskNumPerLoop_);
    }
}

__aicore__ inline void GridSampler2dV2Kernel::ClipCoord(const LocalTensor<float>& coord, const int32_t size)
{
    Maxs(coord, coord, 0.f, alignedTaskNumPerLoop_);
    Mins(coord, coord, (float)(size - 1), alignedTaskNumPerLoop_);
}

__aicore__ inline void GridSampler2dV2Kernel::CompensatePrecision(
    const LocalTensor<float>& floor, const LocalTensor<float>& dividend, const float divisor, const int32_t calCount)
{
    // c = a / b, c_floor = floor(a / b)
    // result = (c_floor * b > a) ? (c_floor - 1) : c_floor
    LocalTensor<float> tmpFloat = tmpFloatBuf_.Get<float>();
    Muls(tmpFloat, floor, divisor, calCount);
    LocalTensor<uint8_t> selMask = mask1Buf_.Get<uint8_t>();
    Compare(selMask, tmpFloat, dividend, CMPMODE::GT, alignedCompareNumPerCore_);
    Adds(tmpFloat, floor, -1.f, calCount);
    Select(floor, selMask, tmpFloat, floor, SELMODE::VSEL_TENSOR_TENSOR_MODE, calCount);
}

__aicore__ inline void GridSampler2dV2Kernel::ReflectCoord(
    const LocalTensor<float>& coord, const int32_t twiceLow, const int32_t twiceHigh)
{
    if (twiceLow == twiceHigh) {
        Duplicate(coord, 0.f, alignedTaskNumPerLoop_);
        return;
    }
    float min = static_cast<float>(twiceLow) / 2;
    float negMin = -1.f * min;
    float span = static_cast<float>(twiceHigh - twiceLow) / 2;
    LocalTensor<float> tmpFloat = tmpFloatBuf_.Get<float>();
    Adds(coord, coord, negMin, alignedTaskNumPerLoop_);
    Abs(coord, coord, alignedTaskNumPerLoop_);

    LocalTensor<float> extraCoord = extraFpPointBuf_.Get<float>();
    Duplicate(tmpFloat, span, alignedTaskNumPerLoop_);
    Fmod(extraCoord, coord, tmpFloat, alignedTaskNumPerLoop_);

    // flip
    LocalTensor<float> flip = flipFloatBuf_.Get<float>();
    Muls(flip, coord, (1.f / span), alignedTaskNumPerLoop_);
    LocalTensor<int32_t> tmpInt = tmpIntBuf_.Get<int32_t>();
    Cast(flip, flip, RoundMode::CAST_FLOOR, alignedTaskNumPerLoop_);
    Cast(flip, tmpInt, RoundMode::CAST_NONE, alignedTaskNumPerLoop_);
    CompensatePrecision(flip, coord, span, alignedTaskNumPerLoop_);

    LocalTensor<uint8_t> selMask = mask1Buf_.Get<uint8_t>();
    LocalTensor<float> mod = modFloatBuf_.Get<float>();
    Duplicate(tmpFloat, 2.f, alignedTaskNumPerLoop_);
    Fmod(mod, flip, tmpFloat, alignedTaskNumPerLoop_);
    CompareScalar(selMask, mod, 0.f, CMPMODE::EQ, alignedCompareNumPerCore_);

    // out1 = extra + min, out2 = span - extra + min
    LocalTensor<float> out1 = tmpFloat;
    LocalTensor<float> out2 = extraCoord;
    Adds(out1, extraCoord, min, alignedTaskNumPerLoop_);
    Muls(out2, extraCoord, -1.f, alignedTaskNumPerLoop_);
    Adds(out2, out2, span, alignedTaskNumPerLoop_);
    Adds(out2, out2, min, alignedTaskNumPerLoop_);
    Select(coord, selMask, out1, out2, SELMODE::VSEL_TENSOR_TENSOR_MODE, alignedTaskNumPerLoop_);
}

__aicore__ inline void GridSampler2dV2Kernel::SafeDowngradeToIntRange(const LocalTensor<float>& coord)
{
    LocalTensor<uint8_t> selMask = mask1Buf_.Get<uint8_t>();
    CompareScalar(selMask, coord, static_cast<float>(INT_MAX - 1), CMPMODE::LE, alignedCompareNumPerCore_);
    Select(coord, selMask, coord, -100.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedTaskNumPerLoop_);
    CompareScalar(selMask, coord, static_cast<float>(INT_MIN), CMPMODE::GE, alignedCompareNumPerCore_);
    Select(coord, selMask, coord, -100.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedTaskNumPerLoop_);
    // is_nan
    Compare(selMask, coord, coord, CMPMODE::EQ, alignedCompareNumPerCore_);
    Select(coord, selMask, coord, -100.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedTaskNumPerLoop_);
}

__aicore__ inline void GridSampler2dV2Kernel::ComputeSourceIndex(const LocalTensor<float>& coord, const int32_t size)
{
    UnnormalizeCoord(coord, size);
    if (paddingMode_ == PaddingMode::BORDER) {
        ClipCoord(coord, size);
    } else if (paddingMode_ == PaddingMode::REFLECTION) {
        int32_t twiceLow = alignCorners_ ? 0 : -1;
        int32_t twiceHigh = alignCorners_ ? (2 * (size - 1)) : (2 * size - 1);
        ReflectCoord(coord, twiceLow, twiceHigh);
        ClipCoord(coord, size);
    }
    SafeDowngradeToIntRange(coord);
}

__aicore__ inline uint64_t GridSampler2dV2Kernel::GetGmOffset(const uint64_t index, const int32_t width,
    const int32_t height, const uint64_t strideN, const uint64_t strideH, const uint64_t strideW)
{
    uint64_t wOffset, hOffset, nOffset, gmOffset;
    wOffset = index % width;
    hOffset = (index / width) % height;
    nOffset = index / (height * width);
    gmOffset = nOffset * strideN + hOffset * strideH + wOffset * strideW;
    return gmOffset;
}

__aicore__ inline void GridSampler2dV2Kernel::ComputeCoord(
    const LocalTensor<float>& inputXFp, const LocalTensor<float>& inputYFp)
{
    LocalTensor<float> grid = gridBuf_.Get<float>();
    LocalTensor<float> inputXYFp = inputXYFpBuf_.Get<float>();
    uint64_t gridGmOffset = GetGmOffset(startTaskIdx_, outWidth_, outHeight_, gridStrideN_, gridStrideH_, gridStrideW_);

    Duplicate<float>(grid, 0.f, alignedGridNumPerCore_);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
    DataCopyPad(grid, gridGm_[gridGmOffset], {1, (uint32_t)(computeCount_ * GRID_CHANNEL * sizeof(float)), 0, 0, 0},
        {true, 0, 0, 0});
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
    Adds(inputXYFp, grid, 1.0f, alignedGridNumPerCore_);

    uint64_t cnt;
    GatherMask(inputXFp, inputXYFp, 1, false, MASK_PLACEHOLDER, gatherParams_, cnt);
    GatherMask(inputYFp, inputXYFp, 2, false, MASK_PLACEHOLDER, gatherParams_, cnt);

    ComputeSourceIndex(inputXFp, inWidth_);
    ComputeSourceIndex(inputYFp, inHeight_);
}

__aicore__ inline void GridSampler2dV2Kernel::ComputeWeightSub(const LocalTensor<float>& w1,
    const LocalTensor<float>& w2, const LocalTensor<float>& x1, const LocalTensor<float>& x2,
    const LocalTensor<float>& y1, const LocalTensor<float>& y2)
{
    Sub(w1, x1, x2, alignedTaskNumPerLoop_);
    Sub(w2, y1, y2, alignedTaskNumPerLoop_);
}

__aicore__ inline void GridSampler2dV2Kernel::ComputeBilinearWeight(const LocalTensor<float>& inputXFp,
    const LocalTensor<float>& inputYFp, const LocalTensor<int32_t>& inputXWInt, const LocalTensor<int32_t>& inputXEInt,
    const LocalTensor<int32_t>& inputYNInt, const LocalTensor<int32_t>& inputYSInt, const LocalTensor<float>& inputXWFp,
    const LocalTensor<float>& inputXEFp, const LocalTensor<float>& inputYNFp, const LocalTensor<float>& inputYSFp,
    const LocalTensor<float>& nwWeight, const LocalTensor<float>& neWeight, const LocalTensor<float>& swWeight,
    const LocalTensor<float>& seWeight)
{
    Cast(inputXWInt, inputXFp, RoundMode::CAST_FLOOR, alignedTaskNumPerLoop_);
    Adds(inputXEInt, inputXWInt, 1, alignedTaskNumPerLoop_);
    Cast(inputXWFp, inputXWInt, RoundMode::CAST_NONE, alignedTaskNumPerLoop_);
    Cast(inputXEFp, inputXEInt, RoundMode::CAST_NONE, alignedTaskNumPerLoop_);

    Cast(inputYNInt, inputYFp, RoundMode::CAST_FLOOR, alignedTaskNumPerLoop_);
    Adds(inputYSInt, inputYNInt, 1, alignedTaskNumPerLoop_);
    Cast(inputYNFp, inputYNInt, RoundMode::CAST_NONE, alignedTaskNumPerLoop_);
    Cast(inputYSFp, inputYSInt, RoundMode::CAST_NONE, alignedTaskNumPerLoop_);

    LocalTensor<float> tmpFloat = tmpFloatBuf_.Get<float>();
    ComputeWeightSub(nwWeight, tmpFloat, inputXEFp, inputXFp, inputYSFp, inputYFp);
    Mul(nwWeight, nwWeight, tmpFloat, alignedTaskNumPerLoop_);
    ComputeWeightSub(neWeight, tmpFloat, inputXFp, inputXWFp, inputYSFp, inputYFp);
    Mul(neWeight, neWeight, tmpFloat, alignedTaskNumPerLoop_);
    ComputeWeightSub(swWeight, tmpFloat, inputXEFp, inputXFp, inputYFp, inputYNFp);
    Mul(swWeight, swWeight, tmpFloat, alignedTaskNumPerLoop_);
    ComputeWeightSub(seWeight, tmpFloat, inputXFp, inputXWFp, inputYFp, inputYNFp);
    Mul(seWeight, seWeight, tmpFloat, alignedTaskNumPerLoop_);
}

__aicore__ inline void GridSampler2dV2Kernel::WithinBounds2d(const LocalTensor<float>& inputXFp,
    const LocalTensor<float>& inputYFp, const LocalTensor<float>& weight, const LocalTensor<float>& selTensor)
{
    CompareScalar(selMask1_, inputXFp, 0.f, CMPMODE::GE, alignedCompareNumPerCore_);
    CompareScalar(selMask2_, inputXFp, static_cast<float>(inWidth_), CMPMODE::LT, alignedCompareNumPerCore_);

    int8ToInt16Mask1_ = selMask1_.ReinterpretCast<uint16_t>();
    int8ToInt16Mask2_ = selMask2_.ReinterpretCast<uint16_t>();
    And(int8ToInt16Mask1_, int8ToInt16Mask1_, int8ToInt16Mask2_, alignedMaskNum_ / 2);
    CompareScalar(selMask2_, inputYFp, 0.f, CMPMODE::GE, alignedCompareNumPerCore_);
    And(int8ToInt16Mask1_, int8ToInt16Mask1_, int8ToInt16Mask2_, alignedMaskNum_ / 2);
    CompareScalar(selMask2_, inputYFp, static_cast<float>(inHeight_), CMPMODE::LT, alignedCompareNumPerCore_);
    And(int8ToInt16Mask1_, int8ToInt16Mask1_, int8ToInt16Mask2_, alignedMaskNum_ / 2);
    // if x, y in range, selTensor = 1
    Select(selTensor, int8ToInt16Mask1_, dupOneTensor_, 0.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedTaskNumPerLoop_);
    Select(weight, int8ToInt16Mask1_, weight, 0.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, alignedTaskNumPerLoop_);
}

__aicore__ inline void GridSampler2dV2Kernel::GetInputOffset(const LocalTensor<float>& inputXFp,
    const LocalTensor<float>& inputYFp, const LocalTensor<float>& offset, LocalTensor<float>& flag)
{
    Add(offset, inputXFp, inputYFp, alignedTaskNumPerLoop_);
    Mul(offset, offset, flag, alignedTaskNumPerLoop_);
}

__aicore__ inline void GridSampler2dV2Kernel::CopyInXtrans(const int32_t groupOffset, const int32_t idxInGroup,
    const LocalTensor<float>& inputXCpInPing, const LocalTensor<int32_t>& nwOffsetInt,
    const LocalTensor<int32_t>& neOffsetInt, const LocalTensor<int32_t>& swOffsetInt,
    const LocalTensor<int32_t>& seOffsetInt)
{
    uint64_t taskIds = startTaskIdx_ + groupOffset + idxInGroup;
    uint64_t inputBaseOffset = taskIds / nOffsetStride_ * inStrideN_;
    uint64_t taskOffset = taskIds - startTaskIdx_;
    int32_t nwOffset = nwOffsetInt.GetValue(taskOffset);
    int32_t neOffset = neOffsetInt.GetValue(taskOffset);
    int32_t swOffset = swOffsetInt.GetValue(taskOffset);
    int32_t seOffset = seOffsetInt.GetValue(taskOffset);

    int32_t cpInOffset = alignedChannel_ * idxInGroup;
    int32_t nwCpInOffset = nwCpInOffsetStart_ + cpInOffset;
    int32_t neCpInOffset = neCpInOffsetStart_ + cpInOffset;
    int32_t swCpInOffset = swCpInOffsetStart_ + cpInOffset;
    int32_t seCpInOffset = seCpInOffsetStart_ + cpInOffset;

    DataCopy(inputXCpInPing[nwCpInOffset], xTransGm_[inputBaseOffset + nwOffset], inputCpParams_);
    DataCopy(inputXCpInPing[neCpInOffset], xTransGm_[inputBaseOffset + neOffset], inputCpParams_);
    DataCopy(inputXCpInPing[swCpInOffset], xTransGm_[inputBaseOffset + swOffset], inputCpParams_);
    DataCopy(inputXCpInPing[seCpInOffset], xTransGm_[inputBaseOffset + seOffset], inputCpParams_);
}

__aicore__ inline void GridSampler2dV2Kernel::ComputeBilinearPerGroup(uint8_t& ping, uint8_t& pong, int32_t groupOffset,
    uint16_t groupSize, const LocalTensor<float>& inputXCpInPing, const LocalTensor<float>& weightBrcb,
    const LocalTensor<float>& outputCpOutPong, const LocalTensor<int32_t>& nwOffsetInt,
    const LocalTensor<int32_t>& neOffsetInt, const LocalTensor<int32_t>& swOffsetInt,
    const LocalTensor<int32_t>& seOffsetInt, const LocalTensor<float>& nwWeight, const LocalTensor<float>& neWeight,
    const LocalTensor<float>& swWeight, const LocalTensor<float>& seWeight)
{
    WaitFlag<HardEvent::V_MTE2>(ping);
    for (int32_t i = 0; i < groupSize; i++) {
        CopyInXtrans(groupOffset, i, inputXCpInPing, nwOffsetInt, neOffsetInt, swOffsetInt, seOffsetInt);
    }
    SetFlag<HardEvent::MTE2_V>(ping);

    Duplicate<float>(weightBrcb, 0.f, alignedChannel_ * coordPosition_ * groupSize_);
    Brcb(weightBrcb[nwCpInOffsetStart_], nwWeight[groupOffset], brcbRptTimes_, brcbRptParams_);
    Brcb(weightBrcb[neCpInOffsetStart_], neWeight[groupOffset], brcbRptTimes_, brcbRptParams_);
    Brcb(weightBrcb[swCpInOffsetStart_], swWeight[groupOffset], brcbRptTimes_, brcbRptParams_);
    Brcb(weightBrcb[seCpInOffsetStart_], seWeight[groupOffset], brcbRptTimes_, brcbRptParams_);
    for (int i = 1; i < alignedChannelBlk_; i++) {
        Adds(weightBrcb[i * B32_DATA_NUM_PER_BLOCK], weightBrcb, 0.f, B32_DATA_NUM_PER_REPEAT, weightAddsRptCount_,
            weightAddsRptParams_);
    }

    WaitFlag<HardEvent::MTE2_V>(ping);
    Mul(weightBrcb, inputXCpInPing, weightBrcb, groupSize_ * alignedChannel_ * coordPosition_);
    Duplicate<float>(inputXCpInPing, 0.f, alignedChannel_ * coordPosition_ * groupSize_);
    SetFlag<HardEvent::V_MTE2>(ping);
    ping = 1 - ping;

    Add(weightBrcb, weightBrcb, weightBrcb[groupSize_ * alignedChannel_ * 2], groupSize_ * alignedChannel_ * 2);
    WaitFlag<HardEvent::MTE3_V>(pong);
    Add(outputCpOutPong, weightBrcb, weightBrcb[groupSize_ * alignedChannel_], groupSize_ * alignedChannel_);
    SetFlag<HardEvent::V_MTE3>(pong);

    uint64_t outputOffset = (static_cast<uint64_t>(startTaskIdx_) + groupOffset) * alignedChannel_;
    WaitFlag<HardEvent::V_MTE3>(pong);
    DataCopy(yTransGm_[outputOffset], outputCpOutPong, {groupSize, outputCpOutBlockLen_, 0, 0});
    SetFlag<HardEvent::MTE3_V>(pong);
    pong = 1 - pong;
}

__aicore__ inline void GridSampler2dV2Kernel::ComputeBilinear(
    const LocalTensor<float>& inputXFp, const LocalTensor<float>& inputYFp)
{
    float inStrideWFp = static_cast<float>(inStrideW_);
    float inStrideHFp = static_cast<float>(inStrideH_);
    uint8_t ping = 0;
    uint8_t pong = 0;
    int32_t groupOffset = 0;
    int32_t groupNum = computeCount_ / groupSize_;
    int32_t tailNums = computeCount_ % groupSize_;
    LocalTensor<float> inputXWFp = inputXWFpBuf_.Get<float>();
    LocalTensor<float> inputXEFp = inputXEFpBuf_.Get<float>();
    LocalTensor<float> inputYNFp = inputYNFpBuf_.Get<float>();
    LocalTensor<float> inputYSFp = inputYSFpBuf_.Get<float>();
    LocalTensor<int32_t> inputXWInt = inputXWIntBuf_.Get<int32_t>();
    LocalTensor<int32_t> inputXEInt = inputXEIntBuf_.Get<int32_t>();
    LocalTensor<int32_t> inputYNInt = inputYNIntBuf_.Get<int32_t>();
    LocalTensor<int32_t> inputYSInt = inputYSIntBuf_.Get<int32_t>();
    LocalTensor<float> nwPointSelTensor = selTensor1Buf_.Get<float>();
    LocalTensor<float> nePointSelTensor = selTensor2Buf_.Get<float>();
    LocalTensor<float> swPointSelTensor = selTensor3Buf_.Get<float>();
    LocalTensor<float> sePointSelTensor = selTensor4Buf_.Get<float>();
    LocalTensor<float> nwOffsetFp = nwOffsetIntBuf_.Get<int32_t>().ReinterpretCast<float>();
    LocalTensor<float> neOffsetFp = neOffsetIntBuf_.Get<int32_t>().ReinterpretCast<float>();
    LocalTensor<float> swOffsetFp = swOffsetIntBuf_.Get<int32_t>().ReinterpretCast<float>();
    LocalTensor<float> seOffsetFp = seOffsetIntBuf_.Get<int32_t>().ReinterpretCast<float>();
    LocalTensor<int32_t> nwOffsetInt = nwOffsetIntBuf_.Get<int32_t>();
    LocalTensor<int32_t> neOffsetInt = neOffsetIntBuf_.Get<int32_t>();
    LocalTensor<int32_t> swOffsetInt = swOffsetIntBuf_.Get<int32_t>();
    LocalTensor<int32_t> seOffsetInt = seOffsetIntBuf_.Get<int32_t>();
    LocalTensor<float> nwWeight = weight_[nwWeightStart_];
    LocalTensor<float> neWeight = weight_[neWeightStart_];
    LocalTensor<float> swWeight = weight_[swWeightStart_];
    LocalTensor<float> seWeight = weight_[seWeightStart_];
    LocalTensor<float> inputXCpIn = inputXCpInBuf_.Get<float>();
    LocalTensor<float> weightBrcb = weightBrcbBuf_.Get<float>();
    LocalTensor<float> outputCpOut = outputCpOutBuf_.Get<float>();

    ComputeBilinearWeight(inputXFp, inputYFp, inputXWInt, inputXEInt, inputYNInt, inputYSInt, inputXWFp, inputXEFp,
        inputYNFp, inputYSFp, nwWeight, neWeight, swWeight, seWeight);
    WithinBounds2d(inputXWFp, inputYNFp, nwWeight, nwPointSelTensor);
    WithinBounds2d(inputXEFp, inputYNFp, neWeight, nePointSelTensor);
    WithinBounds2d(inputXWFp, inputYSFp, swWeight, swPointSelTensor);
    WithinBounds2d(inputXEFp, inputYSFp, seWeight, sePointSelTensor);
    Muls(inputXWFp, inputXWFp, inStrideWFp, alignedTaskNumPerLoop_);
    Muls(inputXEFp, inputXEFp, inStrideWFp, alignedTaskNumPerLoop_);
    Muls(inputYNFp, inputYNFp, inStrideHFp, alignedTaskNumPerLoop_);
    Muls(inputYSFp, inputYSFp, inStrideHFp, alignedTaskNumPerLoop_);
    GetInputOffset(inputXWFp, inputYNFp, nwOffsetFp, nwPointSelTensor);
    GetInputOffset(inputXEFp, inputYNFp, neOffsetFp, nePointSelTensor);
    GetInputOffset(inputXWFp, inputYSFp, swOffsetFp, swPointSelTensor);
    GetInputOffset(inputXEFp, inputYSFp, seOffsetFp, sePointSelTensor);
    Cast(nwOffsetInt, nwOffsetFp, RoundMode::CAST_RINT, alignedTaskNumPerLoop_);
    Cast(neOffsetInt, neOffsetFp, RoundMode::CAST_RINT, alignedTaskNumPerLoop_);
    Cast(swOffsetInt, swOffsetFp, RoundMode::CAST_RINT, alignedTaskNumPerLoop_);
    Cast(seOffsetInt, seOffsetFp, RoundMode::CAST_RINT, alignedTaskNumPerLoop_);

    Duplicate<float>(inputXCpIn, 0.f, alignedChannel_ * coordPosition_ * groupSize_ * BUFFER_NUM);
    SetFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE2>(1);
    SetFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::MTE3_V>(1);
    for (int32_t i = 0; i < groupNum; i++) {
        groupOffset = groupSize_ * i;
        LocalTensor<float> inputXCpPing = inputXCpIn[alignedChannel_ * coordPosition_ * groupSize_ * ping];
        LocalTensor<float> outputCpOutPong = outputCpOut[alignedChannel_ * coordPosition_ * groupSize_ * pong];
        ComputeBilinearPerGroup(ping, pong, groupOffset, static_cast<uint16_t>(groupSize_), inputXCpPing, weightBrcb,
            outputCpOutPong, nwOffsetInt, neOffsetInt, swOffsetInt, seOffsetInt, nwWeight, neWeight, swWeight,
            seWeight);
    }

    if (tailNums != 0) {
        groupOffset = groupSize_ * groupNum;
        LocalTensor<float> inputXCpPing = inputXCpIn[alignedChannel_ * coordPosition_ * groupSize_ * ping];
        LocalTensor<float> outputCpOutPong = outputCpOut[alignedChannel_ * coordPosition_ * groupSize_ * pong];
        ComputeBilinearPerGroup(ping, pong, groupOffset, static_cast<uint16_t>(tailNums), inputXCpPing, weightBrcb,
            outputCpOutPong, nwOffsetInt, neOffsetInt, swOffsetInt, seOffsetInt, nwWeight, neWeight, swWeight,
            seWeight);
    }
    WaitFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(1);
    WaitFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(1);
}

__aicore__ inline void GridSampler2dV2Kernel::ProcessPerLoop()
{
    LocalTensor<float> inputXFp = inputXFpBuf_.Get<float>();
    LocalTensor<float> inputYFp = inputYFpBuf_.Get<float>();
    ComputeCoord(inputXFp, inputYFp);
    ComputeBilinear(inputXFp, inputYFp);
}

__aicore__ inline void GridSampler2dV2Kernel::Process()
{
    Duplicate<float>(dupOneTensor_, 1.f, alignedTaskNumPerLoop_);

    uint64_t baseOffset = startTaskIdx_;
    for (int i = 0; i < copyLoop_; i++) {
        startTaskIdx_ = baseOffset + i * alignedTaskNumPerLoop_;
        endTaskIdx_ = startTaskIdx_ + alignedTaskNumPerLoop_;
        ProcessPerLoop();
    }
    if (copyTail_ != 0) {
        startTaskIdx_ = baseOffset + copyLoop_ * alignedTaskNumPerLoop_;
        endTaskIdx_ = startTaskIdx_ + copyTail_;
        computeCount_ = copyTail_;
        ProcessPerLoop();
    }
}

extern "C" __global__ __aicore__ void grid_sampler2d_v2(
    GM_ADDR xTrans, GM_ADDR grid, GM_ADDR yTrans, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    GridSampler2dV2Kernel op(xTrans, grid, yTrans, &tilingData, &pipe);
    op.Process();
}