#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t ELEM_BYTE_SIZE = sizeof(float);
constexpr uint32_t DATA_BLOCK_SIZE = 32;
constexpr uint32_t COMPARE_ALIGN_BYTE = 512;
constexpr uint32_t LAST_DIM_SIZE = 2;

class CalAnchorsHeading {
public:
    __aicore__ inline CalAnchorsHeading() {}
    __aicore__ inline void Init(TPipe *pipe, const GM_ADDR anchors, const GM_ADDR originPos, GM_ADDR heading,
        const CalAnchorsHeadingTilingData *tiling)
    {
        this->pipe_ = pipe;
        this->blkIdx_ = GetBlockIdx();
        this->InitTiling(tiling);
        this->InitGM(anchors, originPos, heading);
        this->InitUB();
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> anchorsLocal = anchorsBuf_.Get<float>();
        LocalTensor<float> anchorsLocalLeftShift = anchorLeftShiftBuf_.Get<float>();
        float originXPos = -1, originYPos = -1, firstXDiff = 0, firstYDiff = 0;
        uint32_t taskCount = 0;
        uint32_t taskOffset = coreStartTaskIdx_;
        for (int taskIdx = 0; taskIdx < coreAnchorNumTask_; taskIdx += taskCount, taskOffset += taskCount) {
            taskCount = min(min(singleLoopTask_, anchorsNum_ - taskOffset % anchorsNum_),
                coreAnchorNumTask_ - taskIdx);
            uint32_t batchSizeIdx = taskOffset / (anchorsNum_);

            uint64_t originPosOffset = static_cast<uint64_t>(batchSizeIdx) * LAST_DIM_SIZE;
            if (taskOffset % anchorsNum_ == 0 || (originXPos == -1)) {
                originXPos = originPosGm_.GetValue(originPosOffset);
                originYPos = originPosGm_.GetValue(originPosOffset + 1);
            }

            CopyIn(taskCount, taskOffset, anchorsLocal, anchorsLocalLeftShift, originXPos, originYPos);

            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV_);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV_);

            LocalTensor<float> outputLocal = Compute(anchorsLocal, anchorsLocalLeftShift, taskCount);

            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3_);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3_);

            CopyOut(outputLocal, taskCount, taskOffset);
        }
    }

private:
    __aicore__ inline void InitTiling(const CalAnchorsHeadingTilingData *tiling)
    {
        this->batchSize_ = tiling->batchSize;
        this->anchorsNum_ = tiling->anchorsNum;
        this->seqLength_ = tiling->seqLength;
        this->coreAnchorNumTask_ = tiling->coreAnchorNumTask;
        this->taskMemAlignedByte_ = tiling->taskMemAlignedByte;
        this->taskElemCountAligned_ = tiling->taskElemCountAligned;
        this->bigCoreCount_ = tiling->bigCoreCount;
        this->singleLoopTask_ = tiling->singleLoopTask;
        this->copyInLocalStride_ = tiling->copyInLocalStride;
        this->copyInDataBlockElemCountAligned_ = tiling->copyInDataBlockElemCountAligned;
        this->alignedAnchorsBufferByte_ = Ceil(singleLoopTask_ * taskMemAlignedByte_ * LAST_DIM_SIZE,
            COMPARE_ALIGN_BYTE) * COMPARE_ALIGN_BYTE;
        this->eventIDVToMTE3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        this->eventIDMTE2ToV_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        this->eventIDMTE3ToV_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));

        if (blkIdx_ < bigCoreCount_) {
            // big core
            coreStartTaskIdx_ = coreAnchorNumTask_ * blkIdx_;
        } else {
            // small core
            coreStartTaskIdx_ = (coreAnchorNumTask_ * bigCoreCount_) +
                (coreAnchorNumTask_ - 1) * (blkIdx_ - bigCoreCount_);
            coreAnchorNumTask_ -= 1;
        }
    }

    __aicore__ inline void InitGM(const GM_ADDR anchors, const GM_ADDR originPos, GM_ADDR heading)
    {
        this->anchorsGm_.SetGlobalBuffer((__gm__ float*) anchors,
            static_cast<uint64_t>(batchSize_) * anchorsNum_ * seqLength_ * LAST_DIM_SIZE * ELEM_BYTE_SIZE);
        this->originPosGm_.SetGlobalBuffer((__gm__ float*) originPos,
            static_cast<uint64_t>(batchSize_) * LAST_DIM_SIZE * ELEM_BYTE_SIZE);
        this->headingGm_.SetGlobalBuffer((__gm__ float*) heading,
            static_cast<uint64_t>(batchSize_)* anchorsNum_ * seqLength_ * ELEM_BYTE_SIZE);
    }

    __aicore__ inline void InitUB()
    {
        this->pipe_->InitBuffer(anchorsBuf_, alignedAnchorsBufferByte_);
        this->pipe_->InitBuffer(anchorLeftShiftBuf_, singleLoopTask_ * taskMemAlignedByte_ * LAST_DIM_SIZE);
        this->pipe_->InitBuffer(tmpBuf_, singleLoopTask_ * taskMemAlignedByte_ * LAST_DIM_SIZE);

        uint32_t tmpMaskBufferByteLength = DATA_BLOCK_SIZE * Ceil(Ceil(singleLoopTask_ * taskMemAlignedByte_ * LAST_DIM_SIZE,
            AscendCUtils::GetBitSize(sizeof(uint8_t))), DATA_BLOCK_SIZE);
        this->pipe_->InitBuffer(tmpMaskBuf_, tmpMaskBufferByteLength);
    }

    __aicore__ inline void CopyIn(const uint32_t &taskCount, const uint64_t &taskOffset,
        LocalTensor<float> &anchorsLocal, LocalTensor<float> &anchorsLocalLeftShift,
        const float &originXPos, const float &originYPos);
    
    __aicore__ inline LocalTensor<float> Compute(LocalTensor<float> &anchorsLocal,
        LocalTensor<float> &anchorsLocalLeftShift, const uint32_t &taskCount);
    
    __aicore__ inline void CopyOut(LocalTensor<float> &outputLocal, const uint32_t &taskCount,
        const uint64_t &taskOffset);

    __aicore__ inline void FillOriginPos(LocalTensor<float> &anchorsLocal, const float &originPos,
        const uint32_t taskCount, bool isY);

    __aicore__ inline LocalTensor<float> ComputeXYDiff(LocalTensor<float> &anchorsLocal,
        LocalTensor<float> &anchorsLocalLeftShift, const uint32_t taskCount);

    __aicore__ inline LocalTensor<float> ComputeGatheredXYDiff(LocalTensor<float> &xyDiffLocal,
        LocalTensor<float> &xyDiffGatheredLocal, const uint32_t taskCount);

    __aicore__ inline void GeneralDiv(LocalTensor<float> destLocal, const LocalTensor<float> src1Local,
        const LocalTensor<float> src2Local, const uint32_t elementCount);
    
    __aicore__ inline LocalTensor<float> ComputeHeading(LocalTensor<float> &xyDiffGatheredLocal,
        LocalTensor<float> &headingLocal, const uint32_t taskCount);
    
    __aicore__ inline LocalTensor<float> ComputeOutput(LocalTensor<float> &headingLocal,
        LocalTensor<float> &xyDiffGatheredLocal, const uint32_t &taskCount);

private:
    uint64_t blkIdx_;
    TPipe *pipe_;
    GlobalTensor<float> anchorsGm_, originPosGm_, headingGm_;

    uint32_t batchSize_, anchorsNum_, seqLength_, coreAnchorNumTask_, copyInLocalStride_,
        alignedAnchorsBufferByte_, taskMemAlignedByte_, bigCoreCount_, coreStartTaskIdx_,
        singleLoopTask_, taskElemCountAligned_, copyInDataBlockElemCountAligned_;

    int32_t eventIDVToMTE3_, eventIDMTE2ToV_, eventIDMTE3ToV_;

    TBuf<TPosition::VECCALC> anchorsBuf_, anchorLeftShiftBuf_, tmpMaskBuf_, tmpBuf_;
    DataCopyExtParams anchorsCopyInParams_, anchorsLeftShiftCopyInParams_;
    DataCopyPadExtParams<float> anchorsLeftShiftCopyInPadParams_{false, 0, 0, 0};
};

__aicore__ inline void CalAnchorsHeading::CopyIn(const uint32_t &taskCount, const uint64_t &taskOffset,
    LocalTensor<float> &anchorsLocal, LocalTensor<float> &anchorsLocalLeftShift, const float &originXPos, const float &originYPos)
{
    uint64_t anchorGlobalOffset = taskOffset * LAST_DIM_SIZE * seqLength_;

    anchorsCopyInParams_ = {static_cast<uint16_t>(taskCount), LAST_DIM_SIZE * (seqLength_ - 1) * ELEM_BYTE_SIZE,
        static_cast<uint16_t>(LAST_DIM_SIZE * ELEM_BYTE_SIZE), copyInLocalStride_, 0};
    DataCopyPadExtParams<float> anchorsCopyInPadParams{true, 2, 0, originYPos};
    DataCopyPad(anchorsLocal, anchorsGm_[anchorGlobalOffset], anchorsCopyInParams_, anchorsCopyInPadParams);

    anchorsLeftShiftCopyInParams_ = {static_cast<uint16_t>(taskCount), LAST_DIM_SIZE * seqLength_ * ELEM_BYTE_SIZE,
        0, copyInLocalStride_, 0};
    DataCopyPad(anchorsLocalLeftShift, anchorsGm_[anchorGlobalOffset], anchorsLeftShiftCopyInParams_,
        anchorsLeftShiftCopyInPadParams_);

    FillOriginPos(anchorsLocal, originXPos, taskCount, false);
    if (seqLength_ == 1)
        FillOriginPos(anchorsLocal, originYPos, taskCount, true);
}

__aicore__ inline LocalTensor<float> CalAnchorsHeading::Compute(LocalTensor<float> &anchorsLocal,
    LocalTensor<float> &anchorsLocalLeftShift, const uint32_t &taskCount)
{
    // reused buffer anchorsBuf_
    LocalTensor<float> xyDiffLocal = ComputeXYDiff(anchorsLocal, anchorsLocalLeftShift, taskCount);

    // reused buffer anchorLeftShiftBuf_
    LocalTensor<float> xyDiffGatheredLocal = ComputeGatheredXYDiff(xyDiffLocal, anchorsLocalLeftShift, taskCount);

    SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV_);
    WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV_);

    // reused buffer anchorsBuf_ (variable xyDiffLocal)
    LocalTensor<float> headingLocal = ComputeHeading(xyDiffGatheredLocal, xyDiffLocal, taskCount);

    LocalTensor<float> outputLocal = ComputeOutput(headingLocal, xyDiffGatheredLocal, taskCount);

    return outputLocal;
}

__aicore__ inline void CalAnchorsHeading::CopyOut(LocalTensor<float> &outputLocal, const uint32_t &taskCount,
    const uint64_t &taskOffset)
{
    DataCopyExtParams CopyOutParams = {static_cast<uint16_t>(taskCount), seqLength_ * ELEM_BYTE_SIZE, 0, 0, 0};
    DataCopyPad(headingGm_[taskOffset * seqLength_], outputLocal, CopyOutParams);
}

__aicore__ inline void CalAnchorsHeading::FillOriginPos(LocalTensor<float> &anchorsLocal, const float &originPos,
    const uint32_t taskCount, bool isY = true)
{
    uint32_t offset = 0;
    for (uint32_t taskIdx = 0; taskIdx < taskCount; ++taskIdx) {
        anchorsLocal.SetValue(offset + isY, originPos);
        offset += copyInDataBlockElemCountAligned_;
    }
}

__aicore__ inline LocalTensor<float> CalAnchorsHeading::ComputeXYDiff(LocalTensor<float> &anchorsLocal,
    LocalTensor<float> &anchorsLocalLeftShift, const uint32_t taskCount)
{
    Sub(anchorsLocal, anchorsLocalLeftShift, anchorsLocal, copyInDataBlockElemCountAligned_ * taskCount);
    return anchorsLocal;
}

__aicore__ inline LocalTensor<float> CalAnchorsHeading::ComputeGatheredXYDiff(LocalTensor<float> &xyDiffLocal,
    LocalTensor<float> &xyDiffGatheredLocal, const uint32_t taskCount)
{
    uint32_t mask = 0;
    uint64_t rsvdCnt = taskCount * taskElemCountAligned_;
    uint8_t src1Pattern = 1;
    uint16_t repeatTimes = (taskCount * copyInDataBlockElemCountAligned_ + 64 - 1) / 64;       // repeatTimes <= 255
    GatherMask(xyDiffGatheredLocal, xyDiffLocal, src1Pattern, false, mask, { 1, repeatTimes, 8, 0 }, rsvdCnt);
    src1Pattern = 2;
    GatherMask(xyDiffGatheredLocal[taskCount * taskElemCountAligned_], xyDiffLocal, src1Pattern, false, mask,
        { 1, repeatTimes, 8, 0 }, rsvdCnt);
    return xyDiffGatheredLocal;
}

/**
 *  z = y / x
 *  if x == 0: z = 0
 */
__aicore__ inline void CalAnchorsHeading::GeneralDiv(LocalTensor<float> destLocal, const LocalTensor<float> src1Local,
    const LocalTensor<float> src2Local, const uint32_t elementCount)
{
    LocalTensor<uint8_t> maskLocal = tmpMaskBuf_.Get<uint8_t>();

    Div(destLocal, src1Local, src2Local, elementCount);
    CompareScalar(maskLocal, src2Local, static_cast<float>(0), CMPMODE::NE, alignedAnchorsBufferByte_ / sizeof(float));
    PipeBarrier<PIPE_V>();
    Select(destLocal, maskLocal, destLocal,
            static_cast<float>(0.0), SELMODE::VSEL_TENSOR_SCALAR_MODE, elementCount);
}

__aicore__ inline LocalTensor<float> CalAnchorsHeading::ComputeHeading(LocalTensor<float> &xyDiffGatheredLocal,
    LocalTensor<float> &headingLocal, const uint32_t taskCount)
{
    uint32_t calCount = taskCount * taskElemCountAligned_;
    LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();

    // normal arctan
    GeneralDiv(headingLocal, xyDiffGatheredLocal[calCount], xyDiffGatheredLocal, calCount);
    PipeBarrier<PIPE_V>();
    Atan(headingLocal, headingLocal, calCount);
    
    // MULS(SIGN(Y), PI/2) stored in tmpLocal
    Sign(tmpLocal, xyDiffGatheredLocal[calCount], calCount);
    PipeBarrier<PIPE_V>();
    Muls(tmpLocal, tmpLocal, PI/2, calCount);

    // ATAN(GeneralDiv(Y, X)) + MULS(SIGN(Y), PI/2)
    PipeBarrier<PIPE_V>();
    Add(headingLocal, headingLocal, tmpLocal, calCount);
    PipeBarrier<PIPE_V>();
    GeneralDiv(headingLocal[calCount], xyDiffGatheredLocal, xyDiffGatheredLocal, calCount);
    PipeBarrier<PIPE_V>();
    Mul(headingLocal, headingLocal, headingLocal[calCount], calCount);
    PipeBarrier<PIPE_V>();
    Add(headingLocal, headingLocal, tmpLocal, calCount);

    // scale headingLocal to [-PI, PI]
    Muls(tmpLocal, headingLocal, 1/PI, calCount);
    PipeBarrier<PIPE_V>();
    Trunc(tmpLocal, tmpLocal, calCount);
    PipeBarrier<PIPE_V>();
    Muls(tmpLocal, tmpLocal, PI, calCount);
    PipeBarrier<PIPE_V>();
    Sub(headingLocal, headingLocal, tmpLocal, calCount);
    return headingLocal;
}

__aicore__ inline LocalTensor<float> CalAnchorsHeading::ComputeOutput(LocalTensor<float> &headingLocal,
    LocalTensor<float> &xyDiffGatheredLocal, const uint32_t &taskCount)
{
    LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();

    uint32_t calCount = taskCount * taskElemCountAligned_;

    // compute heading_valid
    Adds(xyDiffGatheredLocal, xyDiffGatheredLocal, static_cast<float>(-0.1), calCount * LAST_DIM_SIZE);
    PipeBarrier<PIPE_V>();
    ClampMin(tmpLocal, xyDiffGatheredLocal, static_cast<float>(0.0), calCount * LAST_DIM_SIZE);
    PipeBarrier<PIPE_V>();
    Sign(xyDiffGatheredLocal, tmpLocal, calCount * LAST_DIM_SIZE);
    PipeBarrier<PIPE_V>();

    // xyDiffGatheredLocal OR xyDiffGatheredLocal[calCount]
    Muls(xyDiffGatheredLocal, xyDiffGatheredLocal, static_cast<float>(-1.0), calCount * LAST_DIM_SIZE);
    PipeBarrier<PIPE_V>();
    Adds(xyDiffGatheredLocal, xyDiffGatheredLocal, static_cast<float>(1.0), calCount * LAST_DIM_SIZE);
    PipeBarrier<PIPE_V>();
    Mul(xyDiffGatheredLocal, xyDiffGatheredLocal, xyDiffGatheredLocal[calCount], calCount);
    PipeBarrier<PIPE_V>();
    Muls(xyDiffGatheredLocal, xyDiffGatheredLocal, static_cast<float>(-1.0), calCount);
    PipeBarrier<PIPE_V>();
    Adds(xyDiffGatheredLocal, xyDiffGatheredLocal, static_cast<float>(1.0), calCount);
    Adds(tmpLocal, headingLocal, static_cast<float>(0.0), calCount);
    PipeBarrier<PIPE_V>();
    
    // compute output
    int32_t offset = 0;
    float flag = 0;
    for (int32_t taskIdx = 0; taskIdx < taskCount; ++taskIdx) {
        flag = xyDiffGatheredLocal.GetValue(offset);
        if (flag == 0) {
            tmpLocal.SetValue(offset, static_cast<float>(0));
        }
        for (int32_t seqIdx = 1; seqIdx < seqLength_; ++seqIdx) {
            flag = xyDiffGatheredLocal.GetValue(offset + seqIdx);
            if (flag == 0) {
                tmpLocal.SetValue(offset + seqIdx, tmpLocal.GetValue(offset + seqIdx - 1));
            }
        }
        offset += taskElemCountAligned_;
    }
    return tmpLocal;
}


extern "C" __global__ __aicore__ void cal_anchors_heading(GM_ADDR anchors, GM_ADDR originPos, GM_ADDR heading,
    GM_ADDR workspace, GM_ADDR tiling_data)
{
    GET_TILING_DATA(tiling, tiling_data);
    SetSysWorkspace(workspace);
    CalAnchorsHeading op;
    TPipe pipe;
    op.Init(&pipe, anchors, originPos, heading, &tiling);
    op.Process();
}