#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace {
constexpr uint8_t DOUBLE_SPACE = 2;
constexpr uint8_t FOUR_CORNERS = 4;
constexpr uint8_t X_OFFSET_SIZE = 9;
constexpr uint8_t OFFSET_SIZE = 18;
constexpr uint8_t X_OFFSET_ALIGNED_SIZE = 16;
constexpr uint8_t OFFSET_ALIGNED_SIZE = 24;

constexpr uint8_t FP32_BYTE_SIZE = 4;
constexpr uint8_t BLOCK_BYTE_SIZE = 32;
constexpr uint8_t BLOCK_SIZE_PER_REPEAT = 8;
constexpr uint8_t DATA_BLOCK_SIZE = BLOCK_BYTE_SIZE / FP32_BYTE_SIZE;

constexpr uint8_t DATA_SIZE_PER_REPEAT = 256 / FP32_BYTE_SIZE;
constexpr uint8_t IN_GLOBAL_BUF_SIZE = 64; // double 4 * dataPerBlock
} // namespace

constexpr MatmulConfig DEFORMABLE_CONV2D_CFG = GetNormalConfig();

template<bool modulated>
class DeformableConv2dV2Kernel {
public:
    using AType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;

    matmul::Matmul<AType, BType, CType, CType, DEFORMABLE_CONV2D_CFG> mm_;

    __aicore__ inline DeformableConv2dV2Kernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR offset, GM_ADDR mask, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
        GM_ADDR offsetOutput, GM_ADDR workspace, const DeformableConv2dV2TilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        InitTiling(tilingData);
        InitGM(x, offset, mask, weight, bias, y, offsetOutput, workspace);
        InitBuffer();

        InitConstLocal();
    }

    __aicore__ inline void Process();

protected:
    TPipe* pipe_;
    GlobalTensor<float> xGm_, offsetGm_, weightGm_, maskGm_, biasGm_;
    GlobalTensor<float> yGm_, img2colMatGm_;

    // Buffers
    TBuf<TPosition::VECCALC> offsetBuf_, FeatureBuf_, constKernelIdxBuf_, samplePointPosBuf_, inGlobalBuf_,
        inGlobalFlagBuf_, FracBuf_, maskBuf_, tmpBuf_, outFeatureBuf_, FracBroadcastBuf_;

    // LocalTensors
    LocalTensor<float> copyInOffsetLocal_, xOffsetLocal_, yOffsetLocal_, constKWIdxLocal_, constKHIdxLocal_, tmpLocal_,
        maskLocal_, outFeatureLocal_, topLeftFeatureLocal_, topRightFeatureLocal_, bottomLeftFeatureLocal_,
        bottomRightFeatureLocal_;

    LocalTensor<float> fracHLocal_, fracWLocal_, oneSubFracHLocal_, oneSubFracWLocal_, fracWBroadcastLocal_,
        fracHBroadcastLocal_, oneSubFracHBroadcastLocal_, oneSubFracWBroadcastLocal_, topLeftWeightLocal_,
        topRightWeightLocal_, bottomLeftWeightLocal_, bottomRightWeightLocal_;

    LocalTensor<float> topPosLocal_, leftPosLocal_, rightPosLocal_, bottomPosLocal_, topLeftOffsetLocal_,
        topRightOffsetLocal_, bottomLeftOffsetLocal_, bottomRightOffsetLocal_;

    LocalTensor<uint32_t> inGlobalLocal_;

    BrcbRepeatParams brcbParams_;
    CopyRepeatParams copyParams_;

    // tiling
    uint32_t n_, cIn_, hIn_, wIn_, cOut_, hOut_, wOut_, kH_, kW_, kernelSize_;
    uint32_t start_, end_, cubeTileTaskCount_, elementsCountPerTask_, featureMapSize_, featureMapElementsSize_;
    uint16_t coreCount_, dataBlockPerInputChannel_;
    int8_t padH_, padW_, strideH_, strideW_, dilationH_, dilationW_, groups_;

    // for vector params
    uint64_t cnt_ = 0, mask_ = DATA_SIZE_PER_REPEAT, maskForBroadcast_;
    uint32_t maskForGatherMask_ = OFFSET_ALIGNED_SIZE;
    uint8_t repeatTimes_;
    TEventID copyInOffsetEventID, copyInMaskEventID, copyInFeatureEventID, copyOutEventID, V_SEventID, MTE3_VEventID;

private:
    __aicore__ inline void ProcessVector(uint32_t taskIdx);

    __aicore__ inline void ProcessCube(uint32_t taskIdx, const int32_t& innerCubeTaskIdx);

    __aicore__ inline void CopyInFeature();

    __aicore__ inline void InitTiling(const DeformableConv2dV2TilingData* tilingData)
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
        padH_ = tilingData->padH;
        padW_ = tilingData->padW;
        strideH_ = tilingData->strideH;
        strideW_ = tilingData->strideW;
        dilationH_ = tilingData->dilationH;
        dilationW_ = tilingData->dilationW;
        groups_ = tilingData->groups;
        coreCount_ = tilingData->coreCount;
        cubeTileTaskCount_ = tilingData->cubeTileTaskCount;

        uint32_t blkIdx_ = GetBlockIdx();
        uint32_t avgTasks_ = tilingData->avgTasks;
        uint32_t bigCoreCount_ = tilingData->bigCoreCount;

        start_ = avgTasks_ * blkIdx_ + (blkIdx_ < bigCoreCount_ ? blkIdx_ : bigCoreCount_);
        end_ = start_ + avgTasks_ + (blkIdx_ < bigCoreCount_ ? 1 : 0);

        kernelSize_ = kH_ * kW_;
        dataBlockPerInputChannel_ = (cIn_ * FP32_BYTE_SIZE) / BLOCK_BYTE_SIZE;
        elementsCountPerTask_ = kernelSize_ * cIn_;
        featureMapSize_ = hOut_ * wOut_;
        featureMapElementsSize_ = featureMapSize_ * cIn_;

        repeatTimes_ = X_OFFSET_SIZE * dataBlockPerInputChannel_ / BLOCK_SIZE_PER_REPEAT;

        uint16_t blkStride = dataBlockPerInputChannel_ / BLOCK_SIZE_PER_REPEAT;
        uint16_t repeatStride = BLOCK_SIZE_PER_REPEAT * blkStride;
        brcbParams_ = {blkStride, repeatStride};
        copyParams_ = {1, 0, blkStride, blkStride};
        maskForBroadcast_ = dataBlockPerInputChannel_ - DATA_BLOCK_SIZE;

        copyInOffsetEventID = pipe_->FetchEventID<HardEvent::MTE2_V>();
        copyInMaskEventID = pipe_->FetchEventID<HardEvent::MTE2_V>();
        copyInFeatureEventID = pipe_->FetchEventID<HardEvent::MTE2_V>();
        copyOutEventID = pipe_->FetchEventID<HardEvent::V_MTE3>();
        V_SEventID = pipe_->FetchEventID<HardEvent::V_S>();
        MTE3_VEventID = pipe_->FetchEventID<HardEvent::MTE3_V>();
    }

    __aicore__ inline void InitGM(GM_ADDR x, GM_ADDR offset, GM_ADDR mask, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
        GM_ADDR offsetOutput, GM_ADDR workspace)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight));
        biasGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bias));
        offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(offset));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        img2colMatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(offsetOutput));

        if (modulated) {
            maskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(mask));
        }
    }

    __aicore__ inline void InitBuffer()
    {
        pipe_->InitBuffer(offsetBuf_, (OFFSET_ALIGNED_SIZE + DOUBLE_SPACE * X_OFFSET_ALIGNED_SIZE) * FP32_BYTE_SIZE);
        pipe_->InitBuffer(constKernelIdxBuf_, DOUBLE_SPACE * X_OFFSET_ALIGNED_SIZE * FP32_BYTE_SIZE);
        pipe_->InitBuffer(samplePointPosBuf_, DOUBLE_SPACE * FOUR_CORNERS * X_OFFSET_ALIGNED_SIZE * FP32_BYTE_SIZE);
        pipe_->InitBuffer(inGlobalBuf_, IN_GLOBAL_BUF_SIZE * sizeof(uint32_t));
        pipe_->InitBuffer(FracBuf_, FOUR_CORNERS * X_OFFSET_ALIGNED_SIZE * FP32_BYTE_SIZE);
        pipe_->InitBuffer(FracBroadcastBuf_,
            DOUBLE_SPACE * FOUR_CORNERS * X_OFFSET_SIZE * dataBlockPerInputChannel_ * FP32_BYTE_SIZE);
        pipe_->InitBuffer(tmpBuf_, X_OFFSET_ALIGNED_SIZE * FP32_BYTE_SIZE);
        pipe_->InitBuffer(
            outFeatureBuf_, ((1 + FOUR_CORNERS) * elementsCountPerTask_ + DATA_BLOCK_SIZE) * FP32_BYTE_SIZE);
        if (modulated) {
            pipe_->InitBuffer(maskBuf_, X_OFFSET_ALIGNED_SIZE * FP32_BYTE_SIZE);
            maskLocal_ = maskBuf_.Get<float>();
        }

        outFeatureLocal_ = outFeatureBuf_.Get<float>();
        topLeftFeatureLocal_ = outFeatureLocal_[elementsCountPerTask_ + DATA_BLOCK_SIZE];
        topRightFeatureLocal_ = topLeftFeatureLocal_[elementsCountPerTask_];
        bottomLeftFeatureLocal_ = topLeftFeatureLocal_[2 * elementsCountPerTask_];
        bottomRightFeatureLocal_ = topLeftFeatureLocal_[3 * elementsCountPerTask_];

        copyInOffsetLocal_ = offsetBuf_.Get<float>();
        xOffsetLocal_ = copyInOffsetLocal_[OFFSET_ALIGNED_SIZE];
        yOffsetLocal_ = copyInOffsetLocal_[OFFSET_ALIGNED_SIZE + X_OFFSET_ALIGNED_SIZE];

        constKHIdxLocal_ = constKernelIdxBuf_.Get<float>();
        constKWIdxLocal_ = constKHIdxLocal_[X_OFFSET_ALIGNED_SIZE];

        topPosLocal_ = samplePointPosBuf_.Get<float>();
        bottomPosLocal_ = topPosLocal_[X_OFFSET_ALIGNED_SIZE];
        leftPosLocal_ = topPosLocal_[X_OFFSET_ALIGNED_SIZE * 2];
        rightPosLocal_ = topPosLocal_[X_OFFSET_ALIGNED_SIZE * 3];
        topLeftOffsetLocal_ = topPosLocal_[X_OFFSET_ALIGNED_SIZE * 4];
        topRightOffsetLocal_ = topPosLocal_[X_OFFSET_ALIGNED_SIZE * 5];
        bottomLeftOffsetLocal_ = topPosLocal_[X_OFFSET_ALIGNED_SIZE * 6];
        bottomRightOffsetLocal_ = topPosLocal_[X_OFFSET_ALIGNED_SIZE * 7];

        fracHLocal_ = FracBuf_.Get<float>();
        fracWLocal_ = fracHLocal_[X_OFFSET_ALIGNED_SIZE];
        oneSubFracHLocal_ = fracHLocal_[X_OFFSET_ALIGNED_SIZE * 2];
        oneSubFracWLocal_ = fracHLocal_[X_OFFSET_ALIGNED_SIZE * 3];

        fracHBroadcastLocal_ = FracBroadcastBuf_.Get<float>();
        fracWBroadcastLocal_ = fracHBroadcastLocal_[X_OFFSET_SIZE * dataBlockPerInputChannel_];
        oneSubFracHBroadcastLocal_ = fracHBroadcastLocal_[2 * X_OFFSET_SIZE * dataBlockPerInputChannel_];
        oneSubFracWBroadcastLocal_ = fracHBroadcastLocal_[3 * X_OFFSET_SIZE * dataBlockPerInputChannel_];
        topLeftWeightLocal_ = fracHBroadcastLocal_[4 * X_OFFSET_SIZE * dataBlockPerInputChannel_];
        topRightWeightLocal_ = fracHBroadcastLocal_[5 * X_OFFSET_SIZE * dataBlockPerInputChannel_];
        bottomLeftWeightLocal_ = fracHBroadcastLocal_[6 * X_OFFSET_SIZE * dataBlockPerInputChannel_];
        bottomRightWeightLocal_ = fracHBroadcastLocal_[7 * X_OFFSET_SIZE * dataBlockPerInputChannel_];

        tmpLocal_ = tmpBuf_.Get<float>();
        inGlobalLocal_ = inGlobalBuf_.Get<uint32_t>();
    }

    __aicore__ inline void InitConstLocal()
    {
        CreateVecIndex(constKWIdxLocal_, 0.0f, X_OFFSET_ALIGNED_SIZE);
        Muls(tmpLocal_, constKWIdxLocal_, 1.0f / 3.0f, X_OFFSET_ALIGNED_SIZE);
        Floor(constKHIdxLocal_, tmpLocal_, X_OFFSET_ALIGNED_SIZE);

        Muls(tmpLocal_, constKHIdxLocal_, 3.0f, X_OFFSET_ALIGNED_SIZE);
        Sub(constKWIdxLocal_, constKWIdxLocal_, tmpLocal_, X_OFFSET_ALIGNED_SIZE);
    }
};

template<bool modulated>
__aicore__ inline void DeformableConv2dV2Kernel<modulated>::CopyInFeature()
{
    int32_t topLeft0 = topLeftOffsetLocal_.GetValue(0);
    int32_t topLeft1 = topLeftOffsetLocal_.GetValue(1);
    int32_t topLeft2 = topLeftOffsetLocal_.GetValue(2);
    int32_t topLeft3 = topLeftOffsetLocal_.GetValue(3);
    int32_t topLeft4 = topLeftOffsetLocal_.GetValue(4);
    int32_t topLeft5 = topLeftOffsetLocal_.GetValue(5);
    int32_t topLeft6 = topLeftOffsetLocal_.GetValue(6);
    int32_t topLeft7 = topLeftOffsetLocal_.GetValue(7);
    int32_t topLeft8 = topLeftOffsetLocal_.GetValue(8);

    (topLeft0 == -1.0f) ? Duplicate(topLeftFeatureLocal_[0 * cIn_], 0.0f, cIn_) :
                          DataCopy(topLeftFeatureLocal_[0 * cIn_], xGm_[topLeft0], cIn_);
    (topLeft1 == -1.0f) ? Duplicate(topLeftFeatureLocal_[1 * cIn_], 0.0f, cIn_) :
                          DataCopy(topLeftFeatureLocal_[1 * cIn_], xGm_[topLeft1], cIn_);
    (topLeft2 == -1.0f) ? Duplicate(topLeftFeatureLocal_[2 * cIn_], 0.0f, cIn_) :
                          DataCopy(topLeftFeatureLocal_[2 * cIn_], xGm_[topLeft2], cIn_);
    (topLeft3 == -1.0f) ? Duplicate(topLeftFeatureLocal_[3 * cIn_], 0.0f, cIn_) :
                          DataCopy(topLeftFeatureLocal_[3 * cIn_], xGm_[topLeft3], cIn_);
    (topLeft4 == -1.0f) ? Duplicate(topLeftFeatureLocal_[4 * cIn_], 0.0f, cIn_) :
                          DataCopy(topLeftFeatureLocal_[4 * cIn_], xGm_[topLeft4], cIn_);
    (topLeft5 == -1.0f) ? Duplicate(topLeftFeatureLocal_[5 * cIn_], 0.0f, cIn_) :
                          DataCopy(topLeftFeatureLocal_[5 * cIn_], xGm_[topLeft5], cIn_);
    (topLeft6 == -1.0f) ? Duplicate(topLeftFeatureLocal_[6 * cIn_], 0.0f, cIn_) :
                          DataCopy(topLeftFeatureLocal_[6 * cIn_], xGm_[topLeft6], cIn_);
    (topLeft7 == -1.0f) ? Duplicate(topLeftFeatureLocal_[7 * cIn_], 0.0f, cIn_) :
                          DataCopy(topLeftFeatureLocal_[7 * cIn_], xGm_[topLeft7], cIn_);
    (topLeft8 == -1.0f) ? Duplicate(topLeftFeatureLocal_[8 * cIn_], 0.0f, cIn_) :
                          DataCopy(topLeftFeatureLocal_[8 * cIn_], xGm_[topLeft8], cIn_);

    Mul(topLeftWeightLocal_, oneSubFracHBroadcastLocal_, oneSubFracWBroadcastLocal_, 9 * dataBlockPerInputChannel_);
    SetFlag<HardEvent::MTE3_V>(MTE3_VEventID);
    WaitFlag<HardEvent::MTE3_V>(MTE3_VEventID);
    SetFlag<HardEvent::MTE2_V>(copyInFeatureEventID);
    WaitFlag<HardEvent::MTE2_V>(copyInFeatureEventID);
    Mul(outFeatureLocal_, topLeftFeatureLocal_, topLeftWeightLocal_, mask_, repeatTimes_, {1, 1, 0, 8, 8, 1});

    int32_t topRight0 = topRightOffsetLocal_.GetValue(0);
    int32_t topRight1 = topRightOffsetLocal_.GetValue(1);
    int32_t topRight2 = topRightOffsetLocal_.GetValue(2);
    int32_t topRight3 = topRightOffsetLocal_.GetValue(3);
    int32_t topRight4 = topRightOffsetLocal_.GetValue(4);
    int32_t topRight5 = topRightOffsetLocal_.GetValue(5);
    int32_t topRight6 = topRightOffsetLocal_.GetValue(6);
    int32_t topRight7 = topRightOffsetLocal_.GetValue(7);
    int32_t topRight8 = topRightOffsetLocal_.GetValue(8);

    (topRight0 == -1.0f) ? Duplicate(topRightFeatureLocal_[0 * cIn_], 0.0f, cIn_) :
                           DataCopy(topRightFeatureLocal_[0 * cIn_], xGm_[topRight0], cIn_);
    (topRight1 == -1.0f) ? Duplicate(topRightFeatureLocal_[1 * cIn_], 0.0f, cIn_) :
                           DataCopy(topRightFeatureLocal_[1 * cIn_], xGm_[topRight1], cIn_);
    (topRight2 == -1.0f) ? Duplicate(topRightFeatureLocal_[2 * cIn_], 0.0f, cIn_) :
                           DataCopy(topRightFeatureLocal_[2 * cIn_], xGm_[topRight2], cIn_);
    (topRight3 == -1.0f) ? Duplicate(topRightFeatureLocal_[3 * cIn_], 0.0f, cIn_) :
                           DataCopy(topRightFeatureLocal_[3 * cIn_], xGm_[topRight3], cIn_);
    (topRight4 == -1.0f) ? Duplicate(topRightFeatureLocal_[4 * cIn_], 0.0f, cIn_) :
                           DataCopy(topRightFeatureLocal_[4 * cIn_], xGm_[topRight4], cIn_);
    (topRight5 == -1.0f) ? Duplicate(topRightFeatureLocal_[5 * cIn_], 0.0f, cIn_) :
                           DataCopy(topRightFeatureLocal_[5 * cIn_], xGm_[topRight5], cIn_);
    (topRight6 == -1.0f) ? Duplicate(topRightFeatureLocal_[6 * cIn_], 0.0f, cIn_) :
                           DataCopy(topRightFeatureLocal_[6 * cIn_], xGm_[topRight6], cIn_);
    (topRight7 == -1.0f) ? Duplicate(topRightFeatureLocal_[7 * cIn_], 0.0f, cIn_) :
                           DataCopy(topRightFeatureLocal_[7 * cIn_], xGm_[topRight7], cIn_);
    (topRight8 == -1.0f) ? Duplicate(topRightFeatureLocal_[8 * cIn_], 0.0f, cIn_) :
                           DataCopy(topRightFeatureLocal_[8 * cIn_], xGm_[topRight8], cIn_);

    Mul(topRightWeightLocal_, oneSubFracHBroadcastLocal_, fracWBroadcastLocal_, 9 * dataBlockPerInputChannel_);
    SetFlag<HardEvent::MTE2_V>(copyInFeatureEventID);
    WaitFlag<HardEvent::MTE2_V>(copyInFeatureEventID);
    MulAddDst(outFeatureLocal_, topRightFeatureLocal_, topRightWeightLocal_, mask_, repeatTimes_, {1, 1, 0, 8, 8, 1});

    int32_t bottomLeft0 = bottomLeftOffsetLocal_.GetValue(0);
    int32_t bottomLeft1 = bottomLeftOffsetLocal_.GetValue(1);
    int32_t bottomLeft2 = bottomLeftOffsetLocal_.GetValue(2);
    int32_t bottomLeft3 = bottomLeftOffsetLocal_.GetValue(3);
    int32_t bottomLeft4 = bottomLeftOffsetLocal_.GetValue(4);
    int32_t bottomLeft5 = bottomLeftOffsetLocal_.GetValue(5);
    int32_t bottomLeft6 = bottomLeftOffsetLocal_.GetValue(6);
    int32_t bottomLeft7 = bottomLeftOffsetLocal_.GetValue(7);
    int32_t bottomLeft8 = bottomLeftOffsetLocal_.GetValue(8);

    (bottomLeft0 == -1.0f) ? Duplicate(bottomLeftFeatureLocal_[0 * cIn_], 0.0f, cIn_) :
                             DataCopy(bottomLeftFeatureLocal_[0 * cIn_], xGm_[bottomLeft0], cIn_);
    (bottomLeft1 == -1.0f) ? Duplicate(bottomLeftFeatureLocal_[1 * cIn_], 0.0f, cIn_) :
                             DataCopy(bottomLeftFeatureLocal_[1 * cIn_], xGm_[bottomLeft1], cIn_);
    (bottomLeft2 == -1.0f) ? Duplicate(bottomLeftFeatureLocal_[2 * cIn_], 0.0f, cIn_) :
                             DataCopy(bottomLeftFeatureLocal_[2 * cIn_], xGm_[bottomLeft2], cIn_);
    (bottomLeft3 == -1.0f) ? Duplicate(bottomLeftFeatureLocal_[3 * cIn_], 0.0f, cIn_) :
                             DataCopy(bottomLeftFeatureLocal_[3 * cIn_], xGm_[bottomLeft3], cIn_);
    (bottomLeft4 == -1.0f) ? Duplicate(bottomLeftFeatureLocal_[4 * cIn_], 0.0f, cIn_) :
                             DataCopy(bottomLeftFeatureLocal_[4 * cIn_], xGm_[bottomLeft4], cIn_);
    (bottomLeft5 == -1.0f) ? Duplicate(bottomLeftFeatureLocal_[5 * cIn_], 0.0f, cIn_) :
                             DataCopy(bottomLeftFeatureLocal_[5 * cIn_], xGm_[bottomLeft5], cIn_);
    (bottomLeft6 == -1.0f) ? Duplicate(bottomLeftFeatureLocal_[6 * cIn_], 0.0f, cIn_) :
                             DataCopy(bottomLeftFeatureLocal_[6 * cIn_], xGm_[bottomLeft6], cIn_);
    (bottomLeft7 == -1.0f) ? Duplicate(bottomLeftFeatureLocal_[7 * cIn_], 0.0f, cIn_) :
                             DataCopy(bottomLeftFeatureLocal_[7 * cIn_], xGm_[bottomLeft7], cIn_);
    (bottomLeft8 == -1.0f) ? Duplicate(bottomLeftFeatureLocal_[8 * cIn_], 0.0f, cIn_) :
                             DataCopy(bottomLeftFeatureLocal_[8 * cIn_], xGm_[bottomLeft8], cIn_);

    Mul(bottomLeftWeightLocal_, oneSubFracWBroadcastLocal_, fracHBroadcastLocal_, 9 * dataBlockPerInputChannel_);
    SetFlag<HardEvent::MTE2_V>(copyInFeatureEventID);
    WaitFlag<HardEvent::MTE2_V>(copyInFeatureEventID);
    MulAddDst(
        outFeatureLocal_, bottomLeftFeatureLocal_, bottomLeftWeightLocal_, mask_, repeatTimes_, {1, 1, 0, 8, 8, 1});

    int32_t bottomRight0 = bottomRightOffsetLocal_.GetValue(0);
    int32_t bottomRight1 = bottomRightOffsetLocal_.GetValue(1);
    int32_t bottomRight2 = bottomRightOffsetLocal_.GetValue(2);
    int32_t bottomRight3 = bottomRightOffsetLocal_.GetValue(3);
    int32_t bottomRight4 = bottomRightOffsetLocal_.GetValue(4);
    int32_t bottomRight5 = bottomRightOffsetLocal_.GetValue(5);
    int32_t bottomRight6 = bottomRightOffsetLocal_.GetValue(6);
    int32_t bottomRight7 = bottomRightOffsetLocal_.GetValue(7);
    int32_t bottomRight8 = bottomRightOffsetLocal_.GetValue(8);

    (bottomRight0 == -1.0f) ? Duplicate(bottomRightFeatureLocal_[0 * cIn_], 0.0f, cIn_) :
                              DataCopy(bottomRightFeatureLocal_[0 * cIn_], xGm_[bottomRight0], cIn_);
    (bottomRight1 == -1.0f) ? Duplicate(bottomRightFeatureLocal_[1 * cIn_], 0.0f, cIn_) :
                              DataCopy(bottomRightFeatureLocal_[1 * cIn_], xGm_[bottomRight1], cIn_);
    (bottomRight2 == -1.0f) ? Duplicate(bottomRightFeatureLocal_[2 * cIn_], 0.0f, cIn_) :
                              DataCopy(bottomRightFeatureLocal_[2 * cIn_], xGm_[bottomRight2], cIn_);
    (bottomRight3 == -1.0f) ? Duplicate(bottomRightFeatureLocal_[3 * cIn_], 0.0f, cIn_) :
                              DataCopy(bottomRightFeatureLocal_[3 * cIn_], xGm_[bottomRight3], cIn_);
    (bottomRight4 == -1.0f) ? Duplicate(bottomRightFeatureLocal_[4 * cIn_], 0.0f, cIn_) :
                              DataCopy(bottomRightFeatureLocal_[4 * cIn_], xGm_[bottomRight4], cIn_);
    (bottomRight5 == -1.0f) ? Duplicate(bottomRightFeatureLocal_[5 * cIn_], 0.0f, cIn_) :
                              DataCopy(bottomRightFeatureLocal_[5 * cIn_], xGm_[bottomRight5], cIn_);
    (bottomRight6 == -1.0f) ? Duplicate(bottomRightFeatureLocal_[6 * cIn_], 0.0f, cIn_) :
                              DataCopy(bottomRightFeatureLocal_[6 * cIn_], xGm_[bottomRight6], cIn_);
    (bottomRight7 == -1.0f) ? Duplicate(bottomRightFeatureLocal_[7 * cIn_], 0.0f, cIn_) :
                              DataCopy(bottomRightFeatureLocal_[7 * cIn_], xGm_[bottomRight7], cIn_);
    (bottomRight8 == -1.0f) ? Duplicate(bottomRightFeatureLocal_[8 * cIn_], 0.0f, cIn_) :
                              DataCopy(bottomRightFeatureLocal_[8 * cIn_], xGm_[bottomRight8], cIn_);

    Mul(bottomRightWeightLocal_, fracHBroadcastLocal_, fracWBroadcastLocal_, 9 * dataBlockPerInputChannel_);
    SetFlag<HardEvent::MTE2_V>(copyInFeatureEventID);
    WaitFlag<HardEvent::MTE2_V>(copyInFeatureEventID);
    MulAddDst(
        outFeatureLocal_, bottomRightFeatureLocal_, bottomRightWeightLocal_, mask_, repeatTimes_, {1, 1, 0, 8, 8, 1});
}

template<bool modulated>
__aicore__ inline void DeformableConv2dV2Kernel<modulated>::ProcessVector(uint32_t taskIdx)
{
    int16_t batchIdx = taskIdx / (featureMapSize_);
    int16_t hOutIdx = (taskIdx % (featureMapSize_)) / wOut_;
    int16_t wOutIdx = taskIdx % wOut_;

    // CopyIn Offset
    DataCopy(copyInOffsetLocal_, offsetGm_[taskIdx * OFFSET_SIZE], OFFSET_ALIGNED_SIZE);
    SetFlag<HardEvent::MTE2_V>(copyInOffsetEventID);
    if (modulated) {
        DataCopy(maskLocal_, maskGm_[taskIdx * X_OFFSET_SIZE], X_OFFSET_ALIGNED_SIZE);
        SetFlag<HardEvent::MTE2_V>(copyInMaskEventID);
    }

    WaitFlag<HardEvent::MTE2_V>(copyInOffsetEventID);
    GatherMask(xOffsetLocal_, copyInOffsetLocal_, 1, true, maskForGatherMask_, {1, 1, 8, 0}, cnt_);
    GatherMask(yOffsetLocal_, copyInOffsetLocal_, 2, true, maskForGatherMask_, {1, 1, 8, 0}, cnt_);

    Cast(topPosLocal_, xOffsetLocal_, RoundMode::CAST_FLOOR, X_OFFSET_ALIGNED_SIZE);
    Cast(leftPosLocal_, yOffsetLocal_, RoundMode::CAST_FLOOR, X_OFFSET_ALIGNED_SIZE);

    Sub(fracHLocal_, xOffsetLocal_, topPosLocal_, X_OFFSET_ALIGNED_SIZE);
    Sub(fracWLocal_, yOffsetLocal_, leftPosLocal_, X_OFFSET_ALIGNED_SIZE);

    Add(topPosLocal_, topPosLocal_, constKHIdxLocal_, X_OFFSET_ALIGNED_SIZE);
    Add(leftPosLocal_, leftPosLocal_, constKWIdxLocal_, X_OFFSET_ALIGNED_SIZE);
    Adds(bottomPosLocal_, topPosLocal_, 1.0f, X_OFFSET_ALIGNED_SIZE);
    Adds(rightPosLocal_, leftPosLocal_, 1.0f, X_OFFSET_ALIGNED_SIZE);

    // global position
    Adds(topPosLocal_, topPosLocal_, hOutIdx - kH_ / 2 + 0.0f, 2 * X_OFFSET_ALIGNED_SIZE);
    Adds(leftPosLocal_, leftPosLocal_, wOutIdx - kW_ / 2 + 0.0f, 2 * X_OFFSET_ALIGNED_SIZE);

    // global Offset
    Muls(topPosLocal_, topPosLocal_, wOut_ + 0.0f, 2 * X_OFFSET_ALIGNED_SIZE);

    Add(topLeftOffsetLocal_, topPosLocal_, leftPosLocal_, X_OFFSET_ALIGNED_SIZE); // global (h * wOut + w)
    Add(topRightOffsetLocal_, topPosLocal_, rightPosLocal_, X_OFFSET_ALIGNED_SIZE);
    Add(bottomLeftOffsetLocal_, bottomPosLocal_, leftPosLocal_, X_OFFSET_ALIGNED_SIZE);
    Add(bottomRightOffsetLocal_, bottomPosLocal_, rightPosLocal_, X_OFFSET_ALIGNED_SIZE);

    Muls(topLeftOffsetLocal_, topLeftOffsetLocal_, cIn_ + 0.0f, 4 * X_OFFSET_ALIGNED_SIZE);
    Adds(topLeftOffsetLocal_, topLeftOffsetLocal_, batchIdx * featureMapElementsSize_ + 0.0f,
        4 * X_OFFSET_ALIGNED_SIZE); // global offset

    // in global flag
    CompareScalar(inGlobalLocal_.ReinterpretCast<uint8_t>(), topPosLocal_, 0.0f, CMPMODE::GE, 64);
    CompareScalar(inGlobalLocal_[8].ReinterpretCast<uint8_t>(), bottomPosLocal_, 0.0f, CMPMODE::GE, 64);
    CompareScalar(inGlobalLocal_[16].ReinterpretCast<uint8_t>(), leftPosLocal_, 0.0f, CMPMODE::GE, 64);
    CompareScalar(inGlobalLocal_[24].ReinterpretCast<uint8_t>(), rightPosLocal_, 0.0f, CMPMODE::GE, 64);

    CompareScalar(inGlobalLocal_[32].ReinterpretCast<uint8_t>(), topPosLocal_, featureMapSize_ + 0.0f, CMPMODE::LT, 64);
    CompareScalar(
        inGlobalLocal_[40].ReinterpretCast<uint8_t>(), bottomPosLocal_, featureMapSize_ + 0.0f, CMPMODE::LT, 64);
    CompareScalar(inGlobalLocal_[48].ReinterpretCast<uint8_t>(), leftPosLocal_, wOut_ + 0.0f, CMPMODE::LT, 64);
    CompareScalar(inGlobalLocal_[56].ReinterpretCast<uint8_t>(), rightPosLocal_, wOut_ + 0.0f, CMPMODE::LT, 64);

    And(inGlobalLocal_[32].ReinterpretCast<uint16_t>(), inGlobalLocal_.ReinterpretCast<uint16_t>(),
        inGlobalLocal_[32].ReinterpretCast<uint16_t>(), 64);

    And(inGlobalLocal_.ReinterpretCast<uint16_t>(), inGlobalLocal_[32].ReinterpretCast<uint16_t>(),
        inGlobalLocal_[48].ReinterpretCast<uint16_t>(), 32); // TopLeft, BottomRight
    And(inGlobalLocal_[16].ReinterpretCast<uint16_t>(), inGlobalLocal_[32].ReinterpretCast<uint16_t>(),
        inGlobalLocal_[56].ReinterpretCast<uint16_t>(), 16); // TopRight
    And(inGlobalLocal_[24].ReinterpretCast<uint16_t>(), inGlobalLocal_[40].ReinterpretCast<uint16_t>(),
        inGlobalLocal_[48].ReinterpretCast<uint16_t>(), 16); // BottomLeft

    Select(topLeftOffsetLocal_, inGlobalLocal_.ReinterpretCast<uint16_t>(), topLeftOffsetLocal_, -1.0f,
        SELMODE::VSEL_TENSOR_SCALAR_MODE, 16);
    Select(bottomRightOffsetLocal_, inGlobalLocal_[8].ReinterpretCast<uint16_t>(), bottomRightOffsetLocal_, -1.0f,
        SELMODE::VSEL_TENSOR_SCALAR_MODE, 16);
    Select(topRightOffsetLocal_, inGlobalLocal_[16].ReinterpretCast<uint16_t>(), topRightOffsetLocal_, -1.0f,
        SELMODE::VSEL_TENSOR_SCALAR_MODE, 16);
    Select(bottomLeftOffsetLocal_, inGlobalLocal_[24].ReinterpretCast<uint16_t>(), bottomLeftOffsetLocal_, -1.0f,
        SELMODE::VSEL_TENSOR_SCALAR_MODE, 16);

    SetFlag<HardEvent::V_S>(V_SEventID);
    WaitFlag<HardEvent::V_S>(V_SEventID);

    Muls(oneSubFracHLocal_, fracHLocal_, -1.0f, 2 * X_OFFSET_ALIGNED_SIZE);
    Adds(oneSubFracHLocal_, oneSubFracHLocal_, 1.0f, 2 * X_OFFSET_ALIGNED_SIZE); // 1-fracH, 1-fracW
    if (modulated) {
        WaitFlag<HardEvent::MTE2_V>(copyInMaskEventID);
        Mul(fracHLocal_, fracHLocal_, maskLocal_, X_OFFSET_ALIGNED_SIZE);
        Mul(oneSubFracHLocal_, oneSubFracHLocal_, maskLocal_, X_OFFSET_ALIGNED_SIZE);
    }
    // Broadcast
    Brcb(fracHBroadcastLocal_, fracHLocal_, 2, brcbParams_);
    Brcb(fracWBroadcastLocal_, fracWLocal_, 2, brcbParams_);
    Brcb(oneSubFracHBroadcastLocal_, oneSubFracHLocal_, 2, brcbParams_);
    Brcb(oneSubFracWBroadcastLocal_, oneSubFracWLocal_, 2, brcbParams_);
    Copy(fracHBroadcastLocal_[DATA_BLOCK_SIZE], fracHBroadcastLocal_, maskForBroadcast_, FOUR_CORNERS * X_OFFSET_SIZE,
        copyParams_);

    CopyInFeature();

    SetFlag<HardEvent::V_MTE3>(copyOutEventID);
    WaitFlag<HardEvent::V_MTE3>(copyOutEventID);
    DataCopyPad(img2colMatGm_[taskIdx * elementsCountPerTask_], outFeatureLocal_,
        {1, static_cast<uint32_t>(elementsCountPerTask_ * FP32_BYTE_SIZE), 0, 0, 0});
}

template<bool modulated>
__aicore__ inline void DeformableConv2dV2Kernel<modulated>::ProcessCube(
    uint32_t taskIdx, const int32_t& innerCubeTaskIdx)
{
    int32_t cubeTaskCount = innerCubeTaskIdx + 1;
    uint64_t aOffset = (taskIdx - innerCubeTaskIdx) * elementsCountPerTask_;
    uint64_t cOffset = (taskIdx - innerCubeTaskIdx) * cOut_;

    mm_.SetTensorA(img2colMatGm_[aOffset]);
    mm_.SetTensorB(weightGm_, true);
    mm_.SetSingleShape(cubeTaskCount, cOut_, elementsCountPerTask_);
    mm_.template IterateAll<false>(yGm_[cOffset]);
}

template<bool modulated>
__aicore__ inline void DeformableConv2dV2Kernel<modulated>::Process()
{
    for (int32_t taskIdx = start_; taskIdx < end_; taskIdx++) {
        ProcessVector(taskIdx);

        int32_t innerCubeTaskIdx = (taskIdx - start_) % cubeTileTaskCount_;
        bool startCubeFlag = (innerCubeTaskIdx == cubeTileTaskCount_ - 1) || (taskIdx == end_ - 1);
        if (startCubeFlag) {
            ProcessCube(taskIdx, innerCubeTaskIdx);
        }
    }
    mm_.End();
}


extern "C" __global__ __aicore__ void deformable_conv2d_v2(GM_ADDR x, GM_ADDR offset, GM_ADDR mask, GM_ADDR weight,
    GM_ADDR bias, GM_ADDR y, GM_ADDR offsetOutput, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }

    TPipe pipe;

    if (TILING_KEY_IS(0)) {
        DeformableConv2dV2Kernel<false> op;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm_, &(tilingData.mmTilingData));
        op.Init(x, offset, mask, weight, bias, y, offsetOutput, usrWorkspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        DeformableConv2dV2Kernel<true> op;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm_, &(tilingData.mmTilingData));
        op.Init(x, offset, mask, weight, bias, y, offsetOutput, usrWorkspace, &tilingData, &pipe);
        op.Process();
    }
}
