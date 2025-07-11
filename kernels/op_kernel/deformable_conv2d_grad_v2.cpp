#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
#define K_MAX_SHAPE_DIM 0
namespace {
constexpr uint8_t OFFSET_SIZE = 18;
constexpr uint8_t MASK_SIZE = 9;
constexpr uint8_t X_OFFSET_SIZE = 9;
constexpr uint8_t OFFSET_BUFFER_SIZE = 24;
constexpr uint8_t MASK_BUFFER_SIZE = 16;
constexpr uint8_t X_OFFSET_BUFFER_SIZE = 16;
constexpr uint8_t FP32_BYTE_SIZE = 4;
constexpr uint8_t INT32_BYTE_SIZE = 4;
constexpr uint8_t BLOCK_BYTE_SIZE = 32;
constexpr uint8_t BLOCK_COUNT_PER_REPEAT = 8;
constexpr int32_t INT_TWO = 2;
}

constexpr MatmulConfig DEFORMABLE_CONV2D_CFG = GetNormalConfig();

template<bool modulated>
class DeformableConv2dGradV2Kernel {
public:
    using A0Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using A1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;

    matmul::Matmul<A0Type, BType, CType, CType, DEFORMABLE_CONV2D_CFG> mm0_;
    matmul::Matmul<A1Type, BType, CType, CType, DEFORMABLE_CONV2D_CFG> mm1_;

    __aicore__ inline DeformableConv2dGradV2Kernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR offset, GM_ADDR mask,
        GM_ADDR gradY, GM_ADDR gradX, GM_ADDR gradWeight, GM_ADDR gradBias, GM_ADDR gradOffset,
        GM_ADDR gradMask, GM_ADDR workspace, const DeformableConv2dGradV2TilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        blkIdx_ = GetBlockIdx();
        InitTiling(tilingData);
        InitGM(x, weight, bias, offset, mask, gradY, gradX, gradWeight, gradBias, gradOffset, gradMask,
            workspace);
        InitBuffer();
        InitConstLocal();
    }

    __aicore__ inline void Process();

protected:
    TPipe* pipe_;
    GlobalTensor<float> xGm_, offsetGm_, weightGm_, gradYGm_;
    GlobalTensor<float> maskGm_, gradMaskGm_;
    GlobalTensor<float> gradXGm_, gradOffsetGm_, gradWeightGm_;
    GlobalTensor<float> img2colMatGradGm_, img2colMatGm_;
    
    // Buffers
    TBuf<TPosition::VECCALC> offsetBuf_, constInnerBufferKIdxBuf_, samplePointPosBuf_, FracBuf_,
        img2colMatGradBuf_, xGradBuf_, maskBuf_, offsetGradBuf_, maskGradBuf_, tmpBuf1_, tmpFeatureBuf_, FracBroadcastBuf_, inGlobalFlagBuf_;
    
    // LocalTensors
    LocalTensor<float> copyInOffsetLocal_, xOffsetLocal_, yOffsetLocal_, constInnerBufferKWIdxLocal_, constInnerBufferKHIdxLocal_, tmpConstLocal1_, topPosLocal_, leftPosLocal_,
        rightPosLocal_, bottomPosLocal_, fracHLocal_, fracWLocal_, img2colMatGradLocal_, xGradBufferLocal_, maskLocal_, offsetGradLocal_, maskGradLocal_, tmpFeatureLocal_,
        topLeftOffsetLocal_, topRightOffsetLocal_, bottomleftOffsetLocal_, bottomRightOffsetLocal_, oneSubFracHLocal_, oneSubFracWLocal_, pointWeightBroadcastLocal_,
        fracWBroadcastLocal_, fracHBroadcastLocal_, oneSubFracHBroadcastLocal_, oneSubFracWBroadcastLocal_, tmpWeightBroadcastLocal1_, tmpWeightBroadcastLocal2_,
        negFracWBroadcastLocal_, negFracHBroadcastLocal_, negOneSubFracHBroadcastLocal_, negOneSubFracWBroadcastLocal_, offsetTmpFeatureLocal1_, offsetTmpFeatureLocal2_, img2colTmpFeatureLocal_;
    LocalTensor<uint32_t> inGlobalLocal_;
    BrcbRepeatParams brcbParams_;
    CopyRepeatParams copyParams_;
    uint32_t reduceShape1_[2], reduceShape2_[2];
    
    // tiling
    uint32_t n_, cIn_, hIn_, wIn_, cOut_, hOut_, wOut_, kH_, kW_;
    uint32_t totalTaskCount_, coreTaskCount_, globalTaskOffset_, bigCoreCount_, cubeTileTaskCount_, elementsCountPerTask_, featureMapSize_, featureMapElementsSize_;
    uint32_t blkIdx_;
    uint16_t coreCount_, dataBlockPerInputChannel_;
    int8_t padH_, padW_, strideH_, strideW_, dilationH_, dilationW_, groups_, ping1_ = 0, ping2_ = 0;
    
    // for vector params
    uint64_t cnt_ = 0, mask_ = 64, maskForBroadcast_;
    uint32_t maskForGatherMask_ = OFFSET_BUFFER_SIZE;
    uint8_t repeatTimes_;
    uint32_t cubeCurIterTaskCount1_ = 0, cubeCurIterTaskCount2_ = 0, cubeTsakOffset1_ = 0, cubeTsakOffset2_ = 0;
    float hConstOffset_;
    float wConstOffset_;

private:
    __aicore__ inline void ComputeImg2colMatGradInCube();
    __aicore__ inline void ComputeWeightGradInCube(const int32_t &taskIdx);
    __aicore__ inline void ComputeGlobalOffsetFlag();
    __aicore__ inline void CopyOut(const int32_t& taskIdx, const int32_t& innerCubeTaskIdx);
    __aicore__ inline void ProcessNinePoint(const LocalTensor<float>& hWeightBroadcast1Local, const LocalTensor<float>& wWeightBroadcast1Local,
    const LocalTensor<float>& hWeightBroadcast2Local, const LocalTensor<float>& wWeightBroadcast2Local, const LocalTensor<float>& copyOutFeatureLocal, const bool &mulFlag);

    __aicore__ inline void InitTiling(const DeformableConv2dGradV2TilingData* tilingData)
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
        totalTaskCount_ = n_ * hOut_ * wOut_;
        bigCoreCount_ = totalTaskCount_ % coreCount_;
        coreTaskCount_ = totalTaskCount_ / coreCount_;
        dataBlockPerInputChannel_ = (cIn_ * FP32_BYTE_SIZE) / BLOCK_BYTE_SIZE;
        elementsCountPerTask_ = kH_ * kW_ * cIn_;
        if (blkIdx_ < bigCoreCount_) {
            globalTaskOffset_ = (coreTaskCount_ + 1) * blkIdx_;
            coreTaskCount_ += 1;
        } else {
            globalTaskOffset_ = (coreTaskCount_ + 1) * bigCoreCount_ + coreTaskCount_ * (blkIdx_ - bigCoreCount_);
        }
        cubeTsakOffset1_ = globalTaskOffset_;
        cubeTsakOffset2_ = globalTaskOffset_;
        featureMapSize_ = hOut_ * wOut_;
        featureMapElementsSize_ = featureMapSize_ * cIn_;
        
        reduceShape1_[0] = kH_ * kW_;
        reduceShape1_[1] = cIn_;
        reduceShape2_[0] = kH_ * kW_ + kH_ * kW_;
        reduceShape2_[1] = cIn_;
        
        uint32_t repeatTimesPerInputChannel_ = dataBlockPerInputChannel_ / BLOCK_COUNT_PER_REPEAT;
        repeatTimes_ = kH_ * kW_ * repeatTimesPerInputChannel_;

        uint16_t blkStride = repeatTimesPerInputChannel_;
        uint16_t repeatStride = BLOCK_COUNT_PER_REPEAT * blkStride;
        brcbParams_ = {blkStride, repeatStride};
        copyParams_ = {1, 0, blkStride, blkStride};
        hConstOffset_ = kH_ / INT_TWO + 0.0f;
        wConstOffset_ = kW_ / INT_TWO + 0.0f;
        maskForBroadcast_ = dataBlockPerInputChannel_ - BLOCK_COUNT_PER_REPEAT;
    }

    __aicore__ inline void InitGM(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR offset, GM_ADDR mask, GM_ADDR gradY,
        GM_ADDR gradX, GM_ADDR gradWeight, GM_ADDR gradBias, GM_ADDR gradOffset, GM_ADDR gradMask, GM_ADDR workspace)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        weightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weight));
        offsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(offset));
        gradYGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradY));
        gradXGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradX));
        gradOffsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradOffset));
        gradWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradWeight));

        img2colMatGradGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        img2colMatGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace) + n_ * featureMapSize_ * elementsCountPerTask_);

        if (modulated) {
            maskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(mask));
            gradMaskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gradMask));
        }
    }

    __aicore__ inline void InitBuffer()
    {
        pipe_->InitBuffer(offsetBuf_, (OFFSET_BUFFER_SIZE + 2 * X_OFFSET_BUFFER_SIZE) * FP32_BYTE_SIZE);
        pipe_->InitBuffer(constInnerBufferKIdxBuf_, 2 * X_OFFSET_BUFFER_SIZE * FP32_BYTE_SIZE);
        pipe_->InitBuffer(samplePointPosBuf_, 8 * X_OFFSET_BUFFER_SIZE * FP32_BYTE_SIZE);
        pipe_->InitBuffer(FracBuf_, 4 * X_OFFSET_BUFFER_SIZE * FP32_BYTE_SIZE);
        pipe_->InitBuffer(xGradBuf_, 2 * elementsCountPerTask_ * FP32_BYTE_SIZE);
        pipe_->InitBuffer(offsetGradBuf_,  OFFSET_BUFFER_SIZE * FP32_BYTE_SIZE);
        pipe_->InitBuffer(tmpBuf1_,  X_OFFSET_BUFFER_SIZE * FP32_BYTE_SIZE);
        pipe_->InitBuffer(img2colMatGradBuf_, 2 * elementsCountPerTask_ * FP32_BYTE_SIZE);
        pipe_->InitBuffer(tmpFeatureBuf_, (5 * elementsCountPerTask_ + 16) * FP32_BYTE_SIZE);
        pipe_->InitBuffer(FracBroadcastBuf_, 11 * X_OFFSET_SIZE * dataBlockPerInputChannel_ * FP32_BYTE_SIZE);
        pipe_->InitBuffer(inGlobalFlagBuf_, 64 * INT32_BYTE_SIZE);

        copyInOffsetLocal_ = offsetBuf_.Get<float>();
        xOffsetLocal_ = copyInOffsetLocal_[OFFSET_BUFFER_SIZE];
        yOffsetLocal_ = copyInOffsetLocal_[OFFSET_BUFFER_SIZE + X_OFFSET_BUFFER_SIZE];
        constInnerBufferKHIdxLocal_ = constInnerBufferKIdxBuf_.Get<float>();
        constInnerBufferKWIdxLocal_ = constInnerBufferKHIdxLocal_[X_OFFSET_BUFFER_SIZE];
        inGlobalLocal_ = inGlobalFlagBuf_.Get<uint32_t>();
        topPosLocal_ = samplePointPosBuf_.Get<float>();
        bottomPosLocal_ = topPosLocal_[X_OFFSET_BUFFER_SIZE];
        leftPosLocal_ = topPosLocal_[X_OFFSET_BUFFER_SIZE * 2];
        rightPosLocal_ = topPosLocal_[X_OFFSET_BUFFER_SIZE * 3];
        topLeftOffsetLocal_ = topPosLocal_[X_OFFSET_BUFFER_SIZE * 4];
        topRightOffsetLocal_ = topPosLocal_[X_OFFSET_BUFFER_SIZE * 5];
        bottomleftOffsetLocal_ = topPosLocal_[X_OFFSET_BUFFER_SIZE * 6];
        bottomRightOffsetLocal_ = topPosLocal_[X_OFFSET_BUFFER_SIZE * 7];
        fracHLocal_ = FracBuf_.Get<float>();
        fracWLocal_ = fracHLocal_[X_OFFSET_BUFFER_SIZE];
        oneSubFracHLocal_ = fracHLocal_[X_OFFSET_BUFFER_SIZE * 2];
        oneSubFracWLocal_ = fracHLocal_[X_OFFSET_BUFFER_SIZE * 3];
        img2colMatGradLocal_ = img2colMatGradBuf_.Get<float>();
        xGradBufferLocal_ = xGradBuf_.Get<float>();
        offsetGradLocal_ = offsetGradBuf_.Get<float>();
        tmpConstLocal1_ = tmpBuf1_.Get<float>();
        tmpFeatureLocal_ = tmpFeatureBuf_.Get<float>();
        offsetTmpFeatureLocal1_ = tmpFeatureLocal_[2 * elementsCountPerTask_ + 8];
        offsetTmpFeatureLocal2_ = tmpFeatureLocal_[3 * elementsCountPerTask_ + 8];
        img2colTmpFeatureLocal_ = tmpFeatureLocal_[4 * elementsCountPerTask_ + 16];
        fracHBroadcastLocal_ = FracBroadcastBuf_.Get<float>();
        fracWBroadcastLocal_ = fracHBroadcastLocal_[X_OFFSET_SIZE * dataBlockPerInputChannel_];
        oneSubFracHBroadcastLocal_ = fracHBroadcastLocal_[2 * X_OFFSET_SIZE * dataBlockPerInputChannel_];
        oneSubFracWBroadcastLocal_ = fracHBroadcastLocal_[3 * X_OFFSET_SIZE * dataBlockPerInputChannel_];
        negFracHBroadcastLocal_ = fracHBroadcastLocal_[4 * X_OFFSET_SIZE * dataBlockPerInputChannel_];
        negFracWBroadcastLocal_ = fracHBroadcastLocal_[5 * X_OFFSET_SIZE * dataBlockPerInputChannel_];
        negOneSubFracHBroadcastLocal_ = fracHBroadcastLocal_[6 * X_OFFSET_SIZE * dataBlockPerInputChannel_];
        negOneSubFracWBroadcastLocal_ = fracHBroadcastLocal_[7 * X_OFFSET_SIZE * dataBlockPerInputChannel_];
        pointWeightBroadcastLocal_ = fracHBroadcastLocal_[8 * X_OFFSET_SIZE * dataBlockPerInputChannel_];
        tmpWeightBroadcastLocal1_ = fracHBroadcastLocal_[9 * X_OFFSET_SIZE * dataBlockPerInputChannel_];
        tmpWeightBroadcastLocal2_ = fracHBroadcastLocal_[10 * X_OFFSET_SIZE * dataBlockPerInputChannel_];

        if (modulated) {
            pipe_->InitBuffer(maskGradBuf_,  MASK_BUFFER_SIZE * FP32_BYTE_SIZE);
            pipe_->InitBuffer(maskBuf_,  MASK_BUFFER_SIZE * FP32_BYTE_SIZE);
            maskLocal_ = maskBuf_.Get<float>();
            maskGradLocal_ = maskGradBuf_.Get<float>();
        }
    }

    __aicore__ inline void InitConstLocal()
    {
        CreateVecIndex(constInnerBufferKWIdxLocal_, 0.0f, X_OFFSET_BUFFER_SIZE);
        Muls(tmpConstLocal1_, constInnerBufferKWIdxLocal_, 1.0f / 3.0f, X_OFFSET_BUFFER_SIZE);
        Floor(constInnerBufferKHIdxLocal_, tmpConstLocal1_, X_OFFSET_BUFFER_SIZE);
        
        Muls(tmpConstLocal1_, constInnerBufferKHIdxLocal_, 3.0f, X_OFFSET_BUFFER_SIZE);
        Sub(constInnerBufferKWIdxLocal_, constInnerBufferKWIdxLocal_, tmpConstLocal1_, X_OFFSET_BUFFER_SIZE);
    }
};

template<bool modulated>
__aicore__ inline void DeformableConv2dGradV2Kernel<modulated>::CopyOut(const int32_t& taskIdx, const int32_t& innerCubeTaskIdx)
{
    DataCopyPad(img2colMatGm_[taskIdx * elementsCountPerTask_], img2colTmpFeatureLocal_, {1, static_cast<uint32_t>(elementsCountPerTask_ * FP32_BYTE_SIZE), 0, 0, 0});
    DataCopyPad(gradOffsetGm_[taskIdx * OFFSET_SIZE], offsetGradLocal_, {1, static_cast<uint32_t>(OFFSET_SIZE * FP32_BYTE_SIZE), 0, 0, 0});
    DataCopyPad(gradMaskGm_[taskIdx * MASK_SIZE], maskGradLocal_, {1, static_cast<uint32_t>(MASK_SIZE * FP32_BYTE_SIZE), 0, 0, 0});
}

template<bool modulated>
__aicore__ inline void DeformableConv2dGradV2Kernel<modulated>::ComputeImg2colMatGradInCube()
{
    if (cubeTsakOffset1_ - globalTaskOffset_ >= coreTaskCount_) {
        return;
    }
    cubeCurIterTaskCount1_ = min(cubeTileTaskCount_, coreTaskCount_ + globalTaskOffset_ - cubeTsakOffset1_);

    mm0_.SetTensorA(gradYGm_[cubeTsakOffset1_ * cOut_]);
    mm0_.SetTensorB(weightGm_);
    mm0_.SetSingleShape(cubeCurIterTaskCount1_, elementsCountPerTask_, cOut_);
    mm0_.template IterateAll<false>(img2colMatGradGm_[cubeTsakOffset1_ * elementsCountPerTask_], 0, false, true);
    cubeTsakOffset1_ += cubeCurIterTaskCount1_;
}

template<bool modulated>
__aicore__ inline void DeformableConv2dGradV2Kernel<modulated>::ComputeWeightGradInCube(const int32_t &taskIdx)
{
    cubeCurIterTaskCount2_ = taskIdx - cubeTsakOffset2_ + 1;
    mm1_.SetTensorA(gradYGm_[cubeTsakOffset2_ * cOut_], true);
    mm1_.SetTensorB(img2colMatGm_[cubeTsakOffset2_ * elementsCountPerTask_]);
    mm1_.SetSingleShape(cOut_, elementsCountPerTask_, cubeCurIterTaskCount2_);
    mm1_.template IterateAll<false>(gradWeightGm_, 1);
    cubeTsakOffset2_ += cubeCurIterTaskCount2_;
}

template<bool modulated>
__aicore__ inline void DeformableConv2dGradV2Kernel<modulated>::ComputeGlobalOffsetFlag()
{
    uint64_t mask = 9;
    uint8_t repeat = 1;
    CompareScalar(inGlobalLocal_.ReinterpretCast<uint8_t>(), topPosLocal_, 0.0f, CMPMODE::GE, mask, repeat, {1, 1, 8, 8});
    CompareScalar(inGlobalLocal_[8].ReinterpretCast<uint8_t>(), bottomPosLocal_, 0.0f, CMPMODE::GE, mask, repeat, {1, 1, 8, 8});
    CompareScalar(inGlobalLocal_[16].ReinterpretCast<uint8_t>(), leftPosLocal_, 0.0f, CMPMODE::GE, mask, repeat, {1, 1, 8, 8});
    CompareScalar(inGlobalLocal_[24].ReinterpretCast<uint8_t>(), rightPosLocal_, 0.0f, CMPMODE::GE, mask, repeat, {1, 1, 8, 8});

    CompareScalar(inGlobalLocal_[32].ReinterpretCast<uint8_t>(), topPosLocal_, featureMapSize_ + 0.0f, CMPMODE::LT, mask, repeat, {1, 1, 8, 8});
    CompareScalar(inGlobalLocal_[40].ReinterpretCast<uint8_t>(), bottomPosLocal_, featureMapSize_ + 0.0f, CMPMODE::LT, mask, repeat, {1, 1, 8, 8});
    CompareScalar(inGlobalLocal_[48].ReinterpretCast<uint8_t>(), leftPosLocal_, wOut_ + 0.0f, CMPMODE::LT, mask, repeat, {1, 1, 8, 8});
    CompareScalar(inGlobalLocal_[56].ReinterpretCast<uint8_t>(), rightPosLocal_, wOut_ + 0.0f, CMPMODE::LT, mask, repeat, {1, 1, 8, 8});

    And(inGlobalLocal_[32].ReinterpretCast<uint16_t>(), inGlobalLocal_.ReinterpretCast<uint16_t>(), inGlobalLocal_[32].ReinterpretCast<uint16_t>(), 64);
    
    And(inGlobalLocal_.ReinterpretCast<uint16_t>(), inGlobalLocal_[32].ReinterpretCast<uint16_t>(), inGlobalLocal_[48].ReinterpretCast<uint16_t>(), 32);    // TopLeft, BottomRight
    And(inGlobalLocal_[16].ReinterpretCast<uint16_t>(), inGlobalLocal_[32].ReinterpretCast<uint16_t>(), inGlobalLocal_[56].ReinterpretCast<uint16_t>(), 16);     // TopRight
    And(inGlobalLocal_[24].ReinterpretCast<uint16_t>(), inGlobalLocal_[40].ReinterpretCast<uint16_t>(), inGlobalLocal_[48].ReinterpretCast<uint16_t>(), 16);     // BottomLeft

    Select(topLeftOffsetLocal_, inGlobalLocal_.ReinterpretCast<uint16_t>(), topLeftOffsetLocal_, -1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, 16);
    Select(bottomRightOffsetLocal_, inGlobalLocal_[8].ReinterpretCast<uint16_t>(), bottomRightOffsetLocal_, -1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, 16);
    Select(topRightOffsetLocal_, inGlobalLocal_[16].ReinterpretCast<uint16_t>(), topRightOffsetLocal_, -1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, 16);
    Select(bottomleftOffsetLocal_, inGlobalLocal_[24].ReinterpretCast<uint16_t>(), bottomleftOffsetLocal_, -1.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, 16);
}

template<bool modulated>
__aicore__ inline void DeformableConv2dGradV2Kernel<modulated>::ProcessNinePoint(const LocalTensor<float>& hWeightBroadcast1Local, const LocalTensor<float>& wWeightBroadcast1Local,
    const LocalTensor<float>& hWeightBroadcast2Local, const LocalTensor<float>& wWeightBroadcast2Local, const LocalTensor<float>& copyOutOffsetLocal, const bool &mulFlag)
{
    uint32_t tmpFeatureOffset = ping1_ * elementsCountPerTask_;
    LocalTensor<float> copyInFeatureLocal = tmpFeatureLocal_[tmpFeatureOffset];
    LocalTensor<float> copyOutFeatureLocal = xGradBufferLocal_[tmpFeatureOffset];

    Mul(tmpWeightBroadcastLocal1_, hWeightBroadcast1Local, wWeightBroadcast1Local, 9 * dataBlockPerInputChannel_);
    if (modulated) {
        Mul(tmpWeightBroadcastLocal2_, pointWeightBroadcastLocal_, hWeightBroadcast2Local, 9 * dataBlockPerInputChannel_);
    } else {
        tmpWeightBroadcastLocal2_ = hWeightBroadcast2Local;
    }

    WaitFlag<HardEvent::MTE3_V>(ping1_);
    Mul(xGradBufferLocal_[tmpFeatureOffset], img2colMatGradLocal_[ping2_ * elementsCountPerTask_], tmpWeightBroadcastLocal1_, mask_, repeatTimes_, {1, 1, 0, 8, 8, 1});

    int32_t gmOffset0 = copyOutOffsetLocal.GetValue(0);
    int32_t gmOffset1 = copyOutOffsetLocal.GetValue(1);
    int32_t gmOffset2 = copyOutOffsetLocal.GetValue(2);
    int32_t gmOffset3 = copyOutOffsetLocal.GetValue(3);
    int32_t gmOffset4 = copyOutOffsetLocal.GetValue(4);
    int32_t gmOffset5 = copyOutOffsetLocal.GetValue(5);
    int32_t gmOffset6 = copyOutOffsetLocal.GetValue(6);
    int32_t gmOffset7 = copyOutOffsetLocal.GetValue(7);
    int32_t gmOffset8 = copyOutOffsetLocal.GetValue(8);
    
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE2>(ping1_);
    
    SetAtomicAdd<float>();
    gmOffset0 != -1.0f? (DataCopy(gradXGm_[gmOffset0], copyOutFeatureLocal[0 * cIn_], cIn_), DataCopy(copyInFeatureLocal[0 * cIn_], xGm_[gmOffset0], cIn_)) : Duplicate(copyInFeatureLocal[0 * cIn_], 0.0f, cIn_);
    gmOffset1 != -1.0f? (DataCopy(gradXGm_[gmOffset1], copyOutFeatureLocal[1 * cIn_], cIn_), DataCopy(copyInFeatureLocal[1 * cIn_], xGm_[gmOffset1], cIn_)) : Duplicate(copyInFeatureLocal[1 * cIn_], 0.0f, cIn_);
    gmOffset2 != -1.0f? (DataCopy(gradXGm_[gmOffset2], copyOutFeatureLocal[2 * cIn_], cIn_), DataCopy(copyInFeatureLocal[2 * cIn_], xGm_[gmOffset2], cIn_)) : Duplicate(copyInFeatureLocal[2 * cIn_], 0.0f, cIn_);
    gmOffset3 != -1.0f? (DataCopy(gradXGm_[gmOffset3], copyOutFeatureLocal[3 * cIn_], cIn_), DataCopy(copyInFeatureLocal[3 * cIn_], xGm_[gmOffset3], cIn_)) : Duplicate(copyInFeatureLocal[3 * cIn_], 0.0f, cIn_);
    gmOffset4 != -1.0f? (DataCopy(gradXGm_[gmOffset4], copyOutFeatureLocal[4 * cIn_], cIn_), DataCopy(copyInFeatureLocal[4 * cIn_], xGm_[gmOffset4], cIn_)) : Duplicate(copyInFeatureLocal[4 * cIn_], 0.0f, cIn_);
    gmOffset5 != -1.0f? (DataCopy(gradXGm_[gmOffset5], copyOutFeatureLocal[5 * cIn_], cIn_), DataCopy(copyInFeatureLocal[5 * cIn_], xGm_[gmOffset5], cIn_)) : Duplicate(copyInFeatureLocal[5 * cIn_], 0.0f, cIn_);
    gmOffset6 != -1.0f? (DataCopy(gradXGm_[gmOffset6], copyOutFeatureLocal[6 * cIn_], cIn_), DataCopy(copyInFeatureLocal[6 * cIn_], xGm_[gmOffset6], cIn_)) : Duplicate(copyInFeatureLocal[6 * cIn_], 0.0f, cIn_);
    gmOffset7 != -1.0f? (DataCopy(gradXGm_[gmOffset7], copyOutFeatureLocal[7 * cIn_], cIn_), DataCopy(copyInFeatureLocal[7 * cIn_], xGm_[gmOffset7], cIn_)) : Duplicate(copyInFeatureLocal[7 * cIn_], 0.0f, cIn_);
    gmOffset8 != -1.0f? (DataCopy(gradXGm_[gmOffset8], copyOutFeatureLocal[8 * cIn_], cIn_), DataCopy(copyInFeatureLocal[8 * cIn_], xGm_[gmOffset8], cIn_)) : Duplicate(copyInFeatureLocal[8 * cIn_], 0.0f, cIn_);
    SetAtomicNone();
    SetFlag<HardEvent::MTE3_V>(ping1_);
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    if (mulFlag) {
        Mul(offsetTmpFeatureLocal1_, tmpFeatureLocal_[tmpFeatureOffset], tmpWeightBroadcastLocal2_, mask_, repeatTimes_, {1, 1, 0, 8, 8, 1});
        Mul(offsetTmpFeatureLocal2_, tmpFeatureLocal_[tmpFeatureOffset], wWeightBroadcast2Local, mask_, repeatTimes_, {1, 1, 0, 8, 8, 1});
        Mul(img2colTmpFeatureLocal_, tmpFeatureLocal_[tmpFeatureOffset], tmpWeightBroadcastLocal1_, mask_, repeatTimes_, {1, 1, 0, 8, 8, 1});
    } else {
        MulAddDst(offsetTmpFeatureLocal1_, tmpFeatureLocal_[tmpFeatureOffset], tmpWeightBroadcastLocal2_, mask_, repeatTimes_, {1, 1, 0, 8, 8, 1});
        MulAddDst(offsetTmpFeatureLocal2_, tmpFeatureLocal_[tmpFeatureOffset], wWeightBroadcast2Local, mask_, repeatTimes_, {1, 1, 0, 8, 8, 1});
        MulAddDst(img2colTmpFeatureLocal_, tmpFeatureLocal_[tmpFeatureOffset], tmpWeightBroadcastLocal1_, mask_, repeatTimes_, {1, 1, 0, 8, 8, 1});
    }
    SetFlag<HardEvent::V_MTE2>(ping1_);
    ping1_ = 1 - ping1_;
}

template<bool modulated>
__aicore__ inline void DeformableConv2dGradV2Kernel<modulated>::Process()
{
    ComputeImg2colMatGradInCube();

    SetFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE2>(1);
    for (int32_t coreTaskIdx = 0; coreTaskIdx < coreTaskCount_; coreTaskIdx++) {
        int32_t taskIdx = coreTaskIdx + globalTaskOffset_;
        int16_t batchIdx = taskIdx / (featureMapSize_);
        int16_t hOutIdx = (taskIdx % (featureMapSize_)) / wOut_;
        int16_t wOutIdx = taskIdx % wOut_;
        int32_t innerCubeTaskIdx = coreTaskIdx % cubeTileTaskCount_;
        bool ComputeWeightGradFlag = (innerCubeTaskIdx == cubeTileTaskCount_ - 1) || (coreTaskIdx == coreTaskCount_ - 1);

        // CopyIn Offset
        DataCopy(copyInOffsetLocal_, offsetGm_[taskIdx * OFFSET_SIZE], OFFSET_BUFFER_SIZE);
        if (modulated) {
            DataCopy(maskLocal_, maskGm_[taskIdx * MASK_SIZE], MASK_BUFFER_SIZE);
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        if (modulated) {
            Brcb(pointWeightBroadcastLocal_, maskLocal_, 2, brcbParams_);
        }
        GatherMask(xOffsetLocal_, copyInOffsetLocal_, 1, true, maskForGatherMask_, {1, 1, 8, 0}, cnt_);
        GatherMask(yOffsetLocal_, copyInOffsetLocal_, 2, true, maskForGatherMask_, {1, 1, 8, 0}, cnt_);

        Add(xOffsetLocal_, xOffsetLocal_, constInnerBufferKHIdxLocal_, X_OFFSET_BUFFER_SIZE);
        Add(yOffsetLocal_, yOffsetLocal_, constInnerBufferKWIdxLocal_, X_OFFSET_BUFFER_SIZE);
        Adds(xOffsetLocal_, xOffsetLocal_, hOutIdx - hConstOffset_, X_OFFSET_BUFFER_SIZE);
        Adds(yOffsetLocal_, yOffsetLocal_, wOutIdx - wConstOffset_, X_OFFSET_BUFFER_SIZE);

        Floor(topPosLocal_, xOffsetLocal_, X_OFFSET_BUFFER_SIZE);
        Floor(leftPosLocal_, yOffsetLocal_, X_OFFSET_BUFFER_SIZE);
        Adds(bottomPosLocal_, topPosLocal_, 1.0f, X_OFFSET_BUFFER_SIZE);
        Adds(rightPosLocal_, leftPosLocal_, 1.0f, X_OFFSET_BUFFER_SIZE);

        Sub(fracHLocal_, xOffsetLocal_, topPosLocal_, X_OFFSET_BUFFER_SIZE);
        Sub(fracWLocal_, yOffsetLocal_, leftPosLocal_, X_OFFSET_BUFFER_SIZE);
        Muls(oneSubFracHLocal_, fracHLocal_, -1.0f, X_OFFSET_BUFFER_SIZE * 2);
        Adds(oneSubFracHLocal_, oneSubFracHLocal_, 1.0f, X_OFFSET_BUFFER_SIZE * 2);

        Mul(fracHLocal_, fracHLocal_, maskLocal_, X_OFFSET_BUFFER_SIZE);
        Mul(oneSubFracHLocal_, oneSubFracHLocal_, maskLocal_, X_OFFSET_BUFFER_SIZE);
        
        // 计算 Offset
        Muls(topPosLocal_, topPosLocal_, wOut_ + 0.0f, 2 * X_OFFSET_BUFFER_SIZE);

        Add(topLeftOffsetLocal_, topPosLocal_, leftPosLocal_, X_OFFSET_BUFFER_SIZE);
        Add(topRightOffsetLocal_, topPosLocal_, rightPosLocal_, X_OFFSET_BUFFER_SIZE);
        Add(bottomleftOffsetLocal_, bottomPosLocal_, leftPosLocal_, X_OFFSET_BUFFER_SIZE);
        Add(bottomRightOffsetLocal_, bottomPosLocal_, rightPosLocal_, X_OFFSET_BUFFER_SIZE);
        Muls(topLeftOffsetLocal_, topLeftOffsetLocal_, cIn_ + 0.0f, 4 * X_OFFSET_BUFFER_SIZE);
        Adds(topLeftOffsetLocal_, topLeftOffsetLocal_, batchIdx * featureMapElementsSize_ + 0.0f, 4 * X_OFFSET_BUFFER_SIZE);
    
        ComputeGlobalOffsetFlag();
        SetFlag<HardEvent::V_S>(0);

        Brcb(fracHBroadcastLocal_, fracHLocal_, 2, brcbParams_);
        Brcb(fracWBroadcastLocal_, fracWLocal_, 2, brcbParams_);
        Brcb(oneSubFracHBroadcastLocal_, oneSubFracHLocal_, 2, brcbParams_);
        Brcb(oneSubFracWBroadcastLocal_, oneSubFracWLocal_, 2, brcbParams_);
        if (modulated) {
            Copy(pointWeightBroadcastLocal_[8], pointWeightBroadcastLocal_, maskForBroadcast_, 1 * 9, copyParams_);
        }
        Copy(fracHBroadcastLocal_[8], fracHBroadcastLocal_, maskForBroadcast_, 4 * 9, copyParams_);
        Muls(negFracHBroadcastLocal_, fracHBroadcastLocal_, -1.0f, 4 * X_OFFSET_SIZE * dataBlockPerInputChannel_);

        if (innerCubeTaskIdx == 0) {
            mm0_.WaitIterateAll();
            ComputeImg2colMatGradInCube();
        }
        WaitFlag<HardEvent::V_MTE2>(ping2_);
        DataCopy(img2colMatGradLocal_[ping2_ * elementsCountPerTask_], img2colMatGradGm_[taskIdx * elementsCountPerTask_], elementsCountPerTask_);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        SetFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::MTE3_V>(1);
        SetFlag<HardEvent::V_MTE2>(ping2_);
        WaitFlag<HardEvent::V_S>(0);
        ProcessNinePoint(oneSubFracHBroadcastLocal_, oneSubFracWBroadcastLocal_, negOneSubFracWBroadcastLocal_, negOneSubFracHBroadcastLocal_, topLeftOffsetLocal_, true);
        ProcessNinePoint(oneSubFracHBroadcastLocal_, fracWBroadcastLocal_, negFracWBroadcastLocal_, oneSubFracHBroadcastLocal_, topRightOffsetLocal_, false);
        ProcessNinePoint(oneSubFracWBroadcastLocal_, fracHBroadcastLocal_, oneSubFracWBroadcastLocal_, negFracHBroadcastLocal_, bottomleftOffsetLocal_, false);
        ProcessNinePoint(fracHBroadcastLocal_, fracWBroadcastLocal_, fracWBroadcastLocal_, fracHBroadcastLocal_, bottomRightOffsetLocal_, false);
        Mul(offsetTmpFeatureLocal1_, offsetTmpFeatureLocal1_, img2colMatGradLocal_[ping2_ * elementsCountPerTask_],
             mask_, repeatTimes_, { 1, 1, 1, 8, 8, 8 });
        Mul(tmpFeatureLocal_, img2colTmpFeatureLocal_, img2colMatGradLocal_[ping2_ * elementsCountPerTask_],  mask_, repeatTimes_, { 1, 1, 1, 8, 8, 8 });
        Mul(offsetTmpFeatureLocal2_, offsetTmpFeatureLocal2_, img2colMatGradLocal_[ping2_ * elementsCountPerTask_],
             mask_, repeatTimes_, { 1, 1, 1, 8, 8, 8 });
        if (modulated) {
            Div(tmpFeatureLocal_, tmpFeatureLocal_, pointWeightBroadcastLocal_, mask_, repeatTimes_, {1, 1, 0, 8, 8, 1});
            ReduceSum<float, Pattern::Reduce::AR, true>(maskGradLocal_, tmpFeatureLocal_, reduceShape1_, true);
        }
        ReduceSum<float, Pattern::Reduce::AR, true>(offsetGradLocal_, offsetTmpFeatureLocal1_, reduceShape2_, true);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        CopyOut(taskIdx, innerCubeTaskIdx);
        if (ComputeWeightGradFlag) {
            ComputeWeightGradInCube(taskIdx);
        }
        ping2_ = 1 - ping2_;
        WaitFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(1);
    }
    WaitFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(1);
    mm0_.End();
    mm1_.End();
}

extern "C" __global__ __aicore__ void deformable_conv2d_grad_v2(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR offset,
    GM_ADDR mask, GM_ADDR gradY, GM_ADDR gradX, GM_ADDR gradWeight, GM_ADDR gradBias, GM_ADDR gradOffset, GM_ADDR gradMask,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }

    TPipe pipe;

    if (TILING_KEY_IS(0)) {
        DeformableConv2dGradV2Kernel<false> op;
        REGIST_MATMUL_OBJ(
            &pipe, GetSysWorkSpacePtr(), op.mm0_, &(tilingData.mm0TilingData), op.mm1_, &(tilingData.mm1TilingData));
        op.Init(x, weight, bias, offset, mask, gradY, gradX, gradWeight, gradBias, gradOffset, gradMask,
            usrWorkspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        DeformableConv2dGradV2Kernel<true> op;
        REGIST_MATMUL_OBJ(
            &pipe, GetSysWorkSpacePtr(), op.mm0_, &(tilingData.mm0TilingData), op.mm1_, &(tilingData.mm1TilingData));

        op.Init(x, weight, bias, offset, mask, gradY, gradX, gradWeight, gradBias, gradOffset, gradMask,
            usrWorkspace, &tilingData, &pipe);
        op.Process();
    }
}
