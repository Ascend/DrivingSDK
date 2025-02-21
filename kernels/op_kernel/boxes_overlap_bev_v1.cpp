#include "kernel_operator.h"
#include "boxes_operator_utils.h"
#include <type_traits>
using namespace AscendC;


namespace {
constexpr uint32_t INT32_BYTE_SIZE = 4;
constexpr uint32_t TASK_SIZE = 7;
constexpr uint32_t PATTERN8_0 = 16843009;
constexpr uint32_t PATTERN8_1 = 33686018;
constexpr uint32_t PATTERN8_3 = 134744072;
constexpr uint32_t PATTERN8_4 = 269488144;
constexpr uint32_t PATTERN8_6 = 1077952576;

constexpr float EPS = 1e-8;

constexpr uint8_t PATTERN4_0 = 3;
constexpr uint8_t PATTERN4_1 = 4;
constexpr uint8_t PATTERN4_2 = 5;
constexpr uint8_t PATTERN4_3 = 6;

// mode flags
constexpr int32_t MODE_FLAG_OVERLAP = 0;
constexpr int32_t MODE_FLAG_IOU = 1;
};

class KernelBoxesOverlapBevV1 {
public:
    __aicore__ inline KernelBoxesOverlapBevV1() {}

    __aicore__ inline void Init(TPipe *pipe, GM_ADDR boxes_a, GM_ADDR boxes_b, GM_ADDR res,
        const BoxesOverlapBevV1TilingData* tiling)
    {
        pipe_ = pipe;
        blkIdx_ = GetBlockIdx();
        InitTiling(tiling);
        InitGlobal(boxes_a, boxes_b, res);
        InitUB();
        InitConstLocal();
        InitEvent();
    }

    __aicore__ inline void InitTiling(const BoxesOverlapBevV1TilingData* tiling)
    {
        M_ = tiling->M;
        N_ = tiling->N;
        totalCoreCount_ = tiling->totalCoreCount;
        tileCountN_ = tiling->tileCountN;
        tileCountM_ = tiling->tileCountM;
        tileN_ = tiling->tileN;
        modeFlag_ = tiling->modeFlag;
        margin_ = tiling->margin;
        
        tileNAligned_ = Ceil(tileN_, TASK_SIZE_ALIGNED) * TASK_SIZE_ALIGNED;
        tileTaskCount_ = 1 * tileN_;
        tileTaskCountAligned_ = Ceil(tileTaskCount_, TASK_SIZE_ALIGNED) * TASK_SIZE_ALIGNED;
        fourTileTaskCount_ = 4 * tileTaskCount_;
        fourTileTaskCountAligned_ = Ceil(fourTileTaskCount_, TASK_SIZE_ALIGNED) * TASK_SIZE_ALIGNED;
    }

    __aicore__ inline void InitGlobal(GM_ADDR boxes_a, GM_ADDR boxes_b, GM_ADDR res)
    {
        this->box1Gm_.SetGlobalBuffer((__gm__ float*) boxes_a);
        this->box2Gm_.SetGlobalBuffer((__gm__ float*) boxes_b);
        this->outputGm_.SetGlobalBuffer((__gm__ float*) res);
    }

    __aicore__ inline void InitEvent()
    {
        eventMTE2V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        eventVMTE3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    }

    __aicore__ inline void InitUB()
    {
        pipe_->InitBuffer(box1Buf_, 8 * tileTaskCountAligned_ * FLOAT_BYTE_SIZE);
        pipe_->InitBuffer(box2Buf_, 8 * tileTaskCountAligned_ * FLOAT_BYTE_SIZE);
        pipe_->InitBuffer(numValidBuf_, tileTaskCountAligned_ * FLOAT_BYTE_SIZE);
        pipe_->InitBuffer(cornersBuf_, 2 * 32 * tileTaskCountAligned_ * FLOAT_BYTE_SIZE);
        pipe_->InitBuffer(maskBuf_, 32 * tileTaskCountAligned_ * FLOAT_BYTE_SIZE);
        pipe_->InitBuffer(boxAreaBuf_, 2 * tileTaskCountAligned_ * FLOAT_BYTE_SIZE);

        // tmp buf
        pipe_->InitBuffer(tmpBuf_, 156 * tileTaskCountAligned_ * FLOAT_BYTE_SIZE);

        // some const var
        pipe_->InitBuffer(sortIdxBuf_, 32 * tileTaskCountAligned_ * FLOAT_BYTE_SIZE);
        pipe_->InitBuffer(patternBuf_, 5 * 64 * INT32_BYTE_SIZE);
        pipe_->InitBuffer(constCornersBuf_, 2 * 4 * tileTaskCountAligned_ * FLOAT_BYTE_SIZE);
        pipe_->InitBuffer(gatherBuf_, 64 * tileTaskCountAligned_ * INT32_BYTE_SIZE);
        pipe_->InitBuffer(offsetBuf_, 16 * tileTaskCountAligned_ * INT32_BYTE_SIZE);
        
        // output buf
        pipe_->InitBuffer(outputBuf_, tileTaskCountAligned_ * FLOAT_BYTE_SIZE);
        
        box1Local_ = box1Buf_.Get<float>();
        box2Local_ = box2Buf_.Get<float>();
        patternLocal_ = patternBuf_.Get<uint32_t>();
        xCorners1Local_ = cornersBuf_.Get<float>();
        xCorners2Local_ = xCorners1Local_[4 * tileTaskCountAligned_];
        yCorners1Local_ = xCorners1Local_[32 * tileTaskCountAligned_];
        yCorners2Local_ = xCorners1Local_[36 * tileTaskCountAligned_];
        xConstCornersLocal_ = constCornersBuf_.Get<float>();
        yConstCornersLocal_ = xConstCornersLocal_[4 * tileTaskCountAligned_];
        numValidLocal_ = numValidBuf_.Get<int32_t>();
        sortIdxLocal_ = sortIdxBuf_.Get<int32_t>();
        offsetLocal_ = offsetBuf_.Get<uint32_t>();
        outputLocal_ = outputBuf_.Get<float>();
        box1AreaLocal_ = boxAreaBuf_.Get<float>();
        box2AreaLocal_ = box1AreaLocal_[tileTaskCountAligned_];
        box1ParsedLocal_ = tmpBuf_.Get<float>();
        box2ParsedLocal_ = box1ParsedLocal_[20 * tileTaskCountAligned_];
        sin1Local_ = box1ParsedLocal_[40 * tileTaskCountAligned_];
        cos1Local_ = box1ParsedLocal_[44 * tileTaskCountAligned_];
        sin2Local_ = box1ParsedLocal_[48 * tileTaskCountAligned_];
        cos2Local_ = box1ParsedLocal_[52 * tileTaskCountAligned_];
        tmpLocal1_  = box1ParsedLocal_[56 * tileTaskCountAligned_];
        tmpLocal2_ = box1ParsedLocal_[60 * tileTaskCountAligned_];
        tmpLocal3_ = box1ParsedLocal_[64 * tileTaskCountAligned_];
        tmpLocal4_ = box1ParsedLocal_[68 * tileTaskCountAligned_];
        tmpLocal5_ = box1ParsedLocal_[72 * tileTaskCountAligned_];
        tmpLocal6_ = box1ParsedLocal_[76 * tileTaskCountAligned_];
        maskLocal_  = maskBuf_.Get<float>();
        gatherOffsetLocal1_ = gatherBuf_.Get<uint32_t>();
        gatherOffsetLocal3_ = gatherOffsetLocal1_[32 * tileTaskCountAligned_];
        gatherOffsetLocal4_ = gatherOffsetLocal1_[32 * tileTaskCountAligned_ + 16 * tileTaskCountAligned_];
        x1Local = box1ParsedLocal_[0 * tileTaskCountAligned_];
        y1Local = box1ParsedLocal_[4 * tileTaskCountAligned_];
        r1Local = box1ParsedLocal_[8 * tileTaskCountAligned_];
        w1Local = box1ParsedLocal_[12 * tileTaskCountAligned_];
        h1Local = box1ParsedLocal_[16 * tileTaskCountAligned_];
        x2Local = box2ParsedLocal_[0 * tileTaskCountAligned_];
        y2Local = box2ParsedLocal_[4 * tileTaskCountAligned_];
        r2Local = box2ParsedLocal_[8 * tileTaskCountAligned_];
        w2Local = box2ParsedLocal_[12 * tileTaskCountAligned_];
        h2Local = box2ParsedLocal_[16 * tileTaskCountAligned_];
        x1PointsRotated = xCorners1Local_;
        x2PointsRotated = xCorners2Local_;
        y1PointsRotated = yCorners1Local_;
        y2PointsRotated = yCorners2Local_;
        corners1Mask = maskLocal_;
        corners2Mask = maskLocal_[4 * tileTaskCountAligned_];
        x1RotatedCorners_1 = box1ParsedLocal_[0 * tileTaskCountAligned_];
        x1RotatedCorners_2 = box1ParsedLocal_[1 * tileTaskCountAligned_];
        x1RotatedCorners_3 = box1ParsedLocal_[2 * tileTaskCountAligned_];
        x1RotatedCorners_4 = box1ParsedLocal_[3 * tileTaskCountAligned_];
        x1RotatedCorners_5 = box1ParsedLocal_[4 * tileTaskCountAligned_];
        y1RotatedCorners_1 = box1ParsedLocal_[5 * tileTaskCountAligned_];
        y1RotatedCorners_2 = box1ParsedLocal_[6 * tileTaskCountAligned_];
        y1RotatedCorners_3 = box1ParsedLocal_[7 * tileTaskCountAligned_];
        y1RotatedCorners_4 = box1ParsedLocal_[8 * tileTaskCountAligned_];
        y1RotatedCorners_5 = box1ParsedLocal_[9 * tileTaskCountAligned_];
        x2RotatedCorners_1 = box2ParsedLocal_[0 * tileTaskCountAligned_];
        x2RotatedCorners_2 = box2ParsedLocal_[1 * tileTaskCountAligned_];
        x2RotatedCorners_3 = box2ParsedLocal_[2 * tileTaskCountAligned_];
        x2RotatedCorners_4 = box2ParsedLocal_[3 * tileTaskCountAligned_];
        x2RotatedCorners_5 = box2ParsedLocal_[4 * tileTaskCountAligned_];
        y2RotatedCorners_1 = box2ParsedLocal_[5 * tileTaskCountAligned_];
        y2RotatedCorners_2 = box2ParsedLocal_[6 * tileTaskCountAligned_];
        y2RotatedCorners_3 = box2ParsedLocal_[7 * tileTaskCountAligned_];
        y2RotatedCorners_4 = box2ParsedLocal_[8 * tileTaskCountAligned_];
        y2RotatedCorners_5 = box2ParsedLocal_[9 * tileTaskCountAligned_];
        intersectionMask = maskLocal_[8 * tileTaskCountAligned_];
        xIntersectionCorners = xCorners1Local_[8 * tileTaskCountAligned_];
        yIntersectionCorners = xCorners1Local_[40 * tileTaskCountAligned_];
        xVertices = xCorners1Local_;
        yVertices = xCorners1Local_[32 * tileTaskCountAligned_];
        tmpLocal = box1ParsedLocal_;
        sortedVerticesIdxLocal = box1ParsedLocal_.ReinterpretCast<int32_t>();
        xFrom1ForRepeat = sin1Local_[0 * tileTaskCountAligned_];
        yFrom1ForRepeat = sin1Local_[4 * tileTaskCountAligned_];
        xTo1ForRepeat = sin1Local_[8 * tileTaskCountAligned_];
        yTo1ForRepeat = sin1Local_[12 * tileTaskCountAligned_];
        s1 = sin1Local_[16 * tileTaskCountAligned_];
        s3 = sin1Local_[20 * tileTaskCountAligned_];
        s2 = sin1Local_[24 * tileTaskCountAligned_];
        s4 = sin1Local_[28 * tileTaskCountAligned_];
        s5 = sin1Local_[32 * tileTaskCountAligned_];
        xMax2 = sin1Local_[36 * tileTaskCountAligned_];
        xMin2 = sin1Local_[40 * tileTaskCountAligned_];
        yMax2 = sin1Local_[44 * tileTaskCountAligned_];
        yMin2 = sin1Local_[48 * tileTaskCountAligned_];
        innerTmpLocal1 = sin1Local_[52 * tileTaskCountAligned_];
        innerTmpLocal2 = sin1Local_[56 * tileTaskCountAligned_];
        innerTmpLocal3 = sin1Local_[60 * tileTaskCountAligned_];
        innerTmpLocal4 = sin1Local_[64 * tileTaskCountAligned_];
        innerTmpLocal5 = sin1Local_[68 * tileTaskCountAligned_];
        innerTmpLocal6 = sin1Local_[72 * tileTaskCountAligned_];
        innerTmpLocal7 = sin1Local_[76 * tileTaskCountAligned_];
        innerTmpLocal8 = sin1Local_[80 * tileTaskCountAligned_];
        innerTmpLocal9 = sin1Local_[84 * tileTaskCountAligned_];
        innerTmpLocal10 = sin1Local_[88 * tileTaskCountAligned_];
    }

    __aicore__ inline void InitConstLocal()
    {
        // init GatherMask Pattern, Can optimization
        for (int32_t i = 0; i < 64; i++) {
            patternLocal_.SetValue(0 + i, PATTERN8_0);
            patternLocal_.SetValue(64 + i, PATTERN8_1);
            patternLocal_.SetValue(128 + i, PATTERN8_3);
            patternLocal_.SetValue(192 + i, PATTERN8_4);
            patternLocal_.SetValue(256 + i, PATTERN8_6);
        }

        // Init corners
        box1Local_.SetValue(0, 0.5f);
        box1Local_.SetValue(1, -0.5f);
        box1Local_.SetValue(2, -0.5f);
        box1Local_.SetValue(3, 0.5f);
        box1Local_.SetValue(4, 0.5f);
        box1Local_.SetValue(5, -0.5f);
        box1Local_.SetValue(6, -0.5f);
        box1Local_.SetValue(7, 0.5f);

        box2Local_.SetValue(0, 0.5f);
        box2Local_.SetValue(1, 0.5f);
        box2Local_.SetValue(2, -0.5f);
        box2Local_.SetValue(3, -0.5f);
        box2Local_.SetValue(4, 0.5f);
        box2Local_.SetValue(5, 0.5f);
        box2Local_.SetValue(6, -0.5f);
        box2Local_.SetValue(7, -0.5f);

        uint32_t srcShape[2] = {1, 8};
        uint32_t dstShape[2] = {Ceil(tileN_, 2), 8};
        
        BroadCast<float, 2, 0, false>(xConstCornersLocal_, box1Local_, dstShape, srcShape);
        BroadCast<float, 2, 0, false>(yConstCornersLocal_, box2Local_, dstShape, srcShape);

        // init scatterOffsetLocal
        uint32_t offset = 0u;
        Duplicate(gatherOffsetLocal1_[24], 0u, 8, tileTaskCount_, 1, 4);
        for (int32_t i = 0; i < tileTaskCount_; i++) {
            uint32_t offset2 = 0u;
            for (int32_t j = 0; j < 2; j++) {
                gatherOffsetLocal1_.SetValue(i * 32 + j * 4 + 0, offset * 4 + offset2 + 0u);
                gatherOffsetLocal1_.SetValue(i * 32 + j * 4 + 1, offset * 4 + offset2 + 4u);
                gatherOffsetLocal1_.SetValue(i * 32 + j * 4 + 2, offset * 4 + offset2 + 8u);
                gatherOffsetLocal1_.SetValue(i * 32 + j * 4 + 3, offset * 4 + offset2 + 12u);
                offset2 += 4 * tileTaskCountAligned_ * FLOAT_BYTE_SIZE;
            }

            for (int32_t j = 0; j < 16; j++) {
                gatherOffsetLocal1_.SetValue(i * 32 + j + 8, ((j + 8) * tileTaskCountAligned_) * FLOAT_BYTE_SIZE + offset);
            }
            offset += 1 * FLOAT_BYTE_SIZE;
        }

        offset = 0u;
        for (int32_t i = 0; i < tileTaskCount_; i++) {
            for (int32_t j = 0; j < 16; j++) {
                gatherOffsetLocal3_.SetValue(i * 16 + j, 4 * (i * 32 + j));
                gatherOffsetLocal4_.SetValue(i * 16 + j, 4 * (i * 32 + j + 1));
            }
        }
        // init offset buf
        offset = 0u;
        for (int32_t i = 0; i < tileTaskCount_; i++) {
            Duplicate(offsetLocal_[16 * i], offset, 16);
            offset += 32 * FLOAT_BYTE_SIZE;
        }
        
        // Init some const data for sort
        uint32_t srcShape1[2] = {1, VERTICES_ALIGNED};
        uint32_t dstShape1[2] = {tileTaskCount_, VERTICES_ALIGNED};
        LocalTensor<int32_t> tmpInt32Local = tmpLocal1_.ReinterpretCast<int32_t>();
        CreateVecIndex(tmpInt32Local, 0, VERTICES_ALIGNED, 1, 1, 4);
        BroadCast<int32_t, 2, 0, false>(sortIdxLocal_, tmpInt32Local, dstShape1, srcShape1);
    }

    __aicore__ inline void CopyIn(LocalTensor<float>& boxLocal, GlobalTensor<float>& boxGlobal, const uint64_t globalTensorOffset, const uint32_t taskCount);
    __aicore__ inline void CopyOut(const uint32_t box2TaskCount, const uint64_t box1Offset, const uint64_t box2Offset);
    __aicore__ inline void Process();
    __aicore__ inline void Compute(uint32_t box2TaskCount);
    
    __aicore__ inline void Box2Corners(uint32_t taskCount, LocalTensor<float>& xLocal, LocalTensor<float>& yLocal,
        LocalTensor<float>& wLocal, LocalTensor<float>& hLocal, LocalTensor<float>& rLocal, LocalTensor<float>& xCornersLocal,
        LocalTensor<float>& yCornersLocal, LocalTensor<float>& sinLocal, LocalTensor<float>& cosLocal);
    
    __aicore__ inline void PointsInBox(uint32_t taskCount, LocalTensor<float>& maskLocal, LocalTensor<float>& xLocal, LocalTensor<float>& yLocal, LocalTensor<float>& wLocal, LocalTensor<float>& hLocal,
        LocalTensor<float>& xPointsRotated, LocalTensor<float>& yPointsRotated, LocalTensor<float>& sinLocal, LocalTensor<float>& cosLocal, const float margin);
    
    __aicore__ inline void Intersection(LocalTensor<float>& xRotatedCorners1, LocalTensor<float>& yRotatedCorners1,
        LocalTensor<float>& xRotatedCorners2, LocalTensor<float>& yRotatedCorners2, LocalTensor<float>& intersectionMask,
        LocalTensor<float>& xIntersectionCorners, LocalTensor<float>& yIntersectionCorners, const uint32_t boxCount);
    
    __aicore__ inline void ComputeIntersectionMask(LocalTensor<float>& mask, const LocalTensor<float>& xFrom1, const LocalTensor<float>& yFrom1, const LocalTensor<float>& xTo1,
        const LocalTensor<float>& yTo1, LocalTensor<float>& xMax2, LocalTensor<float>& xMin2, LocalTensor<float>& yMax2, LocalTensor<float>& yMin2, LocalTensor<float>& s1,
        LocalTensor<float>& s2, LocalTensor<float>& s3, LocalTensor<float>& s4, LocalTensor<float>& tmpLocal1, LocalTensor<float>& tmpLocal2, LocalTensor<float>& tmpLocal3, LocalTensor<float>& tmpLocal4,
        LocalTensor<float>& tmpLocal5, const uint32_t calCount);

    __aicore__ inline void Cross(LocalTensor<float>& res, const LocalTensor<float>& xP1, const LocalTensor<float>& yP1, const LocalTensor<float>& xP2,
        const LocalTensor<float>& yP2, LocalTensor<float>& tmpLocal1, LocalTensor<float>& tmpLocal2, const uint32_t calCount);

    __aicore__ inline void ComputeIntersectionCorners(LocalTensor<float>& xCorners, LocalTensor<float>& yCorners, LocalTensor<float>& s1, LocalTensor<float>& s5,
        const LocalTensor<float>& xFrom1, const LocalTensor<float>& yFrom1, const LocalTensor<float>& xTo1, const LocalTensor<float>& yTo1, const LocalTensor<float>& xFrom2, const LocalTensor<float>& yFrom2,
        const LocalTensor<float>& xTo2, const LocalTensor<float>& yTo2, LocalTensor<float>& tmpLocal1, LocalTensor<float>& tmpLocal2, LocalTensor<float>& tmpLocal3, const uint32_t calCount);

    __aicore__ inline void ComputeOverlapArea(LocalTensor<float> &overlapAreaLocal, LocalTensor<float> &xVertices, LocalTensor<float> &yVertices,
        LocalTensor<int32_t> &sortedVerticesIdxLocal, LocalTensor<float> &tmpLocal, const uint32_t boxCount);
private:
    TPipe* pipe_;
    float margin_;
    uint32_t M_, N_, tileCountN_, tileCountM_, tileN_, blkIdx_, tileTaskCount_, tileTaskCountAligned_, fourTileTaskCountAligned_,
        fourTileTaskCount_, modeFlag_, tileNAligned_, totalCoreCount_;
    GlobalTensor<float> box1Gm_, box2Gm_, outputGm_;
    LocalTensor<float> box1Local_, box2Local_, box1ParsedLocal_, box2ParsedLocal_, xCorners1Local_, xCorners2Local_,
        yCorners1Local_, yCorners2Local_, sin1Local_, cos1Local_, sin2Local_, cos2Local_, tmpLocal1_, tmpLocal2_,
        tmpLocal3_, tmpLocal4_, tmpLocal5_, maskLocal_, xConstCornersLocal_, yConstCornersLocal_, outputLocal_,
        box1AreaLocal_, box2AreaLocal_, tmpLocal6_;
    LocalTensor<float> x1Local, y1Local, r1Local, w1Local, h1Local, x2Local, y2Local, r2Local, w2Local, h2Local;
    LocalTensor<float> x1PointsRotated, x2PointsRotated, y1PointsRotated, y2PointsRotated;
    LocalTensor<float> corners1Mask, corners2Mask;
    LocalTensor<float> x1RotatedCorners_1, x1RotatedCorners_2, x1RotatedCorners_3, x1RotatedCorners_4, x1RotatedCorners_5, y1RotatedCorners_1, y1RotatedCorners_2, y1RotatedCorners_3,
        y1RotatedCorners_4, y1RotatedCorners_5, x2RotatedCorners_1, x2RotatedCorners_2, x2RotatedCorners_3, x2RotatedCorners_4, x2RotatedCorners_5, y2RotatedCorners_1, y2RotatedCorners_2,
        y2RotatedCorners_3, y2RotatedCorners_4, y2RotatedCorners_5;
    LocalTensor<float> intersectionMask;
    LocalTensor<float> xIntersectionCorners, yIntersectionCorners;
    LocalTensor<float> xVertices, yVertices;
    LocalTensor<float> tmpLocal;
    LocalTensor<int32_t> sortedVerticesIdxLocal;
    LocalTensor<float> xFrom1ForRepeat, yFrom1ForRepeat, xTo1ForRepeat, yTo1ForRepeat;
    LocalTensor<float> s1, s2, s3, s4, s5;
    LocalTensor<float> xMax2, xMin2, yMax2, yMin2;
    LocalTensor<float> innerTmpLocal1, innerTmpLocal2, innerTmpLocal3, innerTmpLocal4, innerTmpLocal5, innerTmpLocal6, innerTmpLocal7, innerTmpLocal8, innerTmpLocal9, innerTmpLocal10;

    uint32_t eventMTE2V_, eventVMTE3_;

    LocalTensor<uint32_t> patternLocal_, gatherOffsetLocal1_, gatherOffsetLocal3_, gatherOffsetLocal4_, offsetLocal_;
    LocalTensor<int32_t> numValidLocal_, sortIdxLocal_;

    TBuf<TPosition::VECCALC> box1Buf_, box2Buf_, patternBuf_, cornersBuf_, tmpBuf_, maskBuf_, constCornersBuf_, gatherBuf_, numValidBuf_, sortIdxBuf_, offsetBuf_, outputBuf_, boxAreaBuf_;
};

__aicore__ inline void KernelBoxesOverlapBevV1::CopyIn(LocalTensor<float>& boxLocal, GlobalTensor<float>& boxGlobal, const uint64_t globalTensorOffset, const uint32_t taskCount)
{
    DataCopyExtParams copyParams{static_cast<uint16_t>(taskCount), TASK_SIZE * FLOAT_BYTE_SIZE, 0, 0, 0};
    DataCopyPad(boxLocal, boxGlobal[globalTensorOffset], copyParams, {false, 0, 0, 0});
}

__aicore__ inline void KernelBoxesOverlapBevV1::CopyOut(const uint32_t box2TaskCount, const uint64_t box1Offset, const uint64_t box2Offset)
{
    DataCopyExtParams copyParams{static_cast<uint16_t>(1), box2TaskCount * FLOAT_BYTE_SIZE, 0, 0, 0};
    DataCopyPad(outputGm_[box1Offset * N_ + box2Offset], outputLocal_, copyParams);
}

__aicore__ inline void KernelBoxesOverlapBevV1::Process()
{
    uint32_t curTileIdx = blkIdx_;
    uint32_t totalTileCount = tileCountN_ * tileCountM_;

    while (curTileIdx < totalTileCount) {
        uint32_t curTileRowIdx = curTileIdx / tileCountN_;
        uint32_t curTileColIdx = curTileIdx % tileCountN_;

        uint32_t box1Offset = 1 * curTileRowIdx;
        uint32_t box2Offset = tileN_ * curTileColIdx;
        
        uint32_t box1TaskCount = min(1u, M_ - box1Offset);
        uint32_t box2TaskCount = min(tileN_, N_ - box2Offset);

        CopyIn(box1Local_, box1Gm_, box1Offset * TASK_SIZE, box1TaskCount);
        CopyIn(box2Local_, box2Gm_, box2Offset * TASK_SIZE, box2TaskCount);

        SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);

        Compute(box2TaskCount);

        SetFlag<HardEvent::V_MTE3>(eventVMTE3_);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3_);

        CopyOut(box2TaskCount, box1Offset, box2Offset);
        
        curTileIdx += totalCoreCount_;
    }
}

__aicore__ inline void KernelBoxesOverlapBevV1::Compute(uint32_t box2TaskCount)
{
    uint32_t src1Shape[2] = {1, 1};
    uint32_t dst1Shape[2] = {1, box2TaskCount * 4};

    uint32_t src2Shape[2] = {box2TaskCount, 1};
    uint32_t dst2Shape[2] = {box2TaskCount, 4};

    uint32_t curTaskSize = 1 * box2TaskCount;
    uint32_t curTaskSizeAligned = Ceil(curTaskSize, TASK_SIZE_ALIGNED) * TASK_SIZE_ALIGNED;

    ParseXYZWHDRBox(tmpLocal1_, tmpLocal2_, tmpLocal3_, tmpLocal4_, tmpLocal5_, box1Local_, patternLocal_, 1);
    
    BroadCast<float, 2, 1, false>(x1Local, tmpLocal1_, dst1Shape, src1Shape);
    BroadCast<float, 2, 1, false>(y1Local, tmpLocal2_, dst1Shape, src1Shape);
    BroadCast<float, 2, 1, false>(w1Local, tmpLocal3_, dst1Shape, src1Shape);
    BroadCast<float, 2, 1, false>(h1Local, tmpLocal4_, dst1Shape, src1Shape);
    BroadCast<float, 2, 1, false>(r1Local, tmpLocal5_, dst1Shape, src1Shape);

    LocalTensor<float> unionLocal;
    if (modeFlag_ == MODE_FLAG_IOU) {
        // compute box1 area
        unionLocal = box1AreaLocal_;

        Mul(box1AreaLocal_, w1Local, h1Local, curTaskSize);
    }

    ParseXYZWHDRBox(tmpLocal1_, tmpLocal2_, tmpLocal3_, tmpLocal4_, tmpLocal5_, box2Local_, patternLocal_, box2TaskCount);

    if (modeFlag_ == MODE_FLAG_IOU) {
        // compute box2 area
        Mul(box2AreaLocal_, tmpLocal3_, tmpLocal4_, curTaskSize);
        Add(unionLocal, box1AreaLocal_, box2AreaLocal_, curTaskSize);
    }

    BroadCast<float, 2, 1, false>(x2Local, tmpLocal1_, dst2Shape, src2Shape);
    BroadCast<float, 2, 1, false>(y2Local, tmpLocal2_, dst2Shape, src2Shape);
    BroadCast<float, 2, 1, false>(w2Local, tmpLocal3_, dst2Shape, src2Shape);
    BroadCast<float, 2, 1, false>(h2Local, tmpLocal4_, dst2Shape, src2Shape);
    BroadCast<float, 2, 1, false>(r2Local, tmpLocal5_, dst2Shape, src2Shape);

    Box2Corners(curTaskSize, x1Local, y1Local, w1Local, h1Local, r1Local,
        xCorners1Local_, yCorners1Local_, sin1Local_, cos1Local_);
    Box2Corners(curTaskSize, x2Local, y2Local, w2Local, h2Local, r2Local,
        xCorners2Local_, yCorners2Local_, sin2Local_, cos2Local_);

    PointsInBox(curTaskSize, corners1Mask, x2Local, y2Local, w2Local, h2Local,
        x1PointsRotated, y1PointsRotated, sin2Local_, cos2Local_, margin_);
    PointsInBox(curTaskSize, corners2Mask, x1Local, y1Local, w1Local, h1Local,
        x2PointsRotated, y2PointsRotated, sin1Local_, cos1Local_, margin_);

    uint32_t mask = 0;
    uint64_t rsvdCnt = 0;
    uint16_t repeatTimesForGatherMask = Ceil(fourTileTaskCount_, 64);
    uint8_t repeatTimesForCopy = Ceil(tileTaskCountAligned_, 64);

    GatherMask(x1RotatedCorners_1, x1PointsRotated, PATTERN4_0, false, mask, { 1, repeatTimesForGatherMask, 8, 0 }, rsvdCnt);
    GatherMask(x1RotatedCorners_2, x1PointsRotated, PATTERN4_1, false, mask, { 1, repeatTimesForGatherMask, 8, 0 }, rsvdCnt);
    GatherMask(x1RotatedCorners_3, x1PointsRotated, PATTERN4_2, false, mask, { 1, repeatTimesForGatherMask, 8, 0 }, rsvdCnt);
    GatherMask(x1RotatedCorners_4, x1PointsRotated, PATTERN4_3, false, mask, { 1, repeatTimesForGatherMask, 8, 0 }, rsvdCnt);
    Copy(x1RotatedCorners_5, x1RotatedCorners_1, static_cast<uint64_t>(64), repeatTimesForCopy, {1, 1, 8, 8});

    GatherMask(y1RotatedCorners_1, y1PointsRotated, PATTERN4_0, false, mask, { 1, repeatTimesForGatherMask, 8, 0 }, rsvdCnt);
    GatherMask(y1RotatedCorners_2, y1PointsRotated, PATTERN4_1, false, mask, { 1, repeatTimesForGatherMask, 8, 0 }, rsvdCnt);
    GatherMask(y1RotatedCorners_3, y1PointsRotated, PATTERN4_2, false, mask, { 1, repeatTimesForGatherMask, 8, 0 }, rsvdCnt);
    GatherMask(y1RotatedCorners_4, y1PointsRotated, PATTERN4_3, false, mask, { 1, repeatTimesForGatherMask, 8, 0 }, rsvdCnt);
    Copy(y1RotatedCorners_5, y1RotatedCorners_1, static_cast<uint64_t>(64), repeatTimesForCopy, {1, 1, 8, 8});

    GatherMask(x2RotatedCorners_1, x2PointsRotated, PATTERN4_0, false, mask, { 1, repeatTimesForGatherMask, 8, 0 }, rsvdCnt);
    GatherMask(x2RotatedCorners_2, x2PointsRotated, PATTERN4_1, false, mask, { 1, repeatTimesForGatherMask, 8, 0 }, rsvdCnt);
    GatherMask(x2RotatedCorners_3, x2PointsRotated, PATTERN4_2, false, mask, { 1, repeatTimesForGatherMask, 8, 0 }, rsvdCnt);
    GatherMask(x2RotatedCorners_4, x2PointsRotated, PATTERN4_3, false, mask, { 1, repeatTimesForGatherMask, 8, 0 }, rsvdCnt);
    Copy(x2RotatedCorners_5, x2RotatedCorners_1, static_cast<uint64_t>(64), repeatTimesForCopy, {1, 1, 8, 8});

    GatherMask(y2RotatedCorners_1, y2PointsRotated, PATTERN4_0, false, mask, { 1, repeatTimesForGatherMask, 8, 0 }, rsvdCnt);
    GatherMask(y2RotatedCorners_2, y2PointsRotated, PATTERN4_1, false, mask, { 1, repeatTimesForGatherMask, 8, 0 }, rsvdCnt);
    GatherMask(y2RotatedCorners_3, y2PointsRotated, PATTERN4_2, false, mask, { 1, repeatTimesForGatherMask, 8, 0 }, rsvdCnt);
    GatherMask(y2RotatedCorners_4, y2PointsRotated, PATTERN4_3, false, mask, { 1, repeatTimesForGatherMask, 8, 0 }, rsvdCnt);
    Copy(y2RotatedCorners_5, y2RotatedCorners_1, static_cast<uint64_t>(64), repeatTimesForCopy, {1, 1, 8, 8});
    
    Intersection(x1RotatedCorners_1, y1RotatedCorners_1, x2RotatedCorners_1, y2RotatedCorners_1, intersectionMask, xIntersectionCorners, yIntersectionCorners, curTaskSize);

    // Build Mask
    repeatTimesForCopy = Ceil(24 * tileTaskCountAligned_, 64);
    uint8_t repeatTimesForGather = Ceil(curTaskSize, 2);
    Copy(tmpLocal1_, maskLocal_, static_cast<uint64_t>(64), repeatTimesForCopy, {1, 1, 8, 8});
    Gather(maskLocal_, tmpLocal1_, gatherOffsetLocal1_, 0u, 64, repeatTimesForGather, 8);
    
    // Build x Vertice
    Copy(tmpLocal1_, xVertices, static_cast<uint64_t>(64), repeatTimesForCopy, {1, 1, 8, 8});
    Gather(xVertices, tmpLocal1_, gatherOffsetLocal1_, 0u, 64, repeatTimesForGather, 8);
    
    // Build y Vertice
    Copy(tmpLocal1_, yVertices, static_cast<uint64_t>(64), repeatTimesForCopy, {1, 1, 8, 8});
    Gather(yVertices, tmpLocal1_, gatherOffsetLocal1_, 0u, 64, repeatTimesForGather, 8);

    // Compute numValid
    WholeReduceSum(numValidLocal_.ReinterpretCast<float>(), maskLocal_, 24u, curTaskSize, 1u, 1u, 4u);
    Cast(numValidLocal_, numValidLocal_.ReinterpretCast<float>(), RoundMode::CAST_CEIL, curTaskSize);
    
    SortVertices(sortedVerticesIdxLocal, xVertices, yVertices, maskLocal_, numValidLocal_, sortIdxLocal_, tmpLocal, curTaskSize, false);

    Muls(sortedVerticesIdxLocal, sortedVerticesIdxLocal, static_cast<int32_t>(4), curTaskSize * VERTICES_ALIGNED);

    ComputeOverlapArea(outputLocal_, xVertices, yVertices, sortedVerticesIdxLocal, tmpLocal1_, curTaskSize);

    if (modeFlag_ == MODE_FLAG_IOU) {
        Sub(unionLocal, unionLocal, outputLocal_, curTaskSize);
        Maxs(unionLocal, unionLocal, EPS, curTaskSize);
        Div(outputLocal_, outputLocal_, unionLocal, curTaskSize);
    }
}

__aicore__ inline void KernelBoxesOverlapBevV1::Box2Corners(uint32_t taskCount, LocalTensor<float>& xLocal, LocalTensor<float>& yLocal,
    LocalTensor<float>& wLocal, LocalTensor<float>& hLocal, LocalTensor<float>& rLocal, LocalTensor<float>& xCornersLocal,
    LocalTensor<float>& yCornersLocal, LocalTensor<float>& sinLocal, LocalTensor<float>& cosLocal)
{
    Mul(xCornersLocal, wLocal, xConstCornersLocal_, taskCount * 4);
    Mul(yCornersLocal, hLocal, yConstCornersLocal_, taskCount * 4);

    Sin(sinLocal, rLocal, taskCount * 4);
    Cos(cosLocal, rLocal, taskCount * 4);

    Mul(tmpLocal1_, xCornersLocal, cosLocal, taskCount * 4);
    Mul(tmpLocal2_, yCornersLocal, sinLocal, taskCount * 4);
    Sub(tmpLocal1_, tmpLocal1_, tmpLocal2_, taskCount * 4);
    Add(rLocal, tmpLocal1_, xLocal, taskCount * 4);      // store in rLocal temporally

    Mul(tmpLocal1_, xCornersLocal, sinLocal, taskCount * 4);
    Mul(tmpLocal2_, yCornersLocal, cosLocal, taskCount * 4);
    Add(tmpLocal1_, tmpLocal1_, tmpLocal2_, taskCount * 4);
    Add(yCornersLocal, tmpLocal1_, yLocal, taskCount * 4);      // store in rLocal temporally

    Adds(xCornersLocal, rLocal, 0.f, taskCount * 4);
}


__aicore__ inline void KernelBoxesOverlapBevV1::PointsInBox(uint32_t taskCount, LocalTensor<float>& maskLocal, LocalTensor<float>& xLocal, LocalTensor<float>& yLocal, LocalTensor<float>& wLocal, LocalTensor<float>& hLocal,
    LocalTensor<float>& xPointsRotated, LocalTensor<float>& yPointsRotated, LocalTensor<float>& sinLocal, LocalTensor<float>& cosLocal, const float margin)
{
    Sub(tmpLocal1_, xPointsRotated, xLocal, taskCount * 4);
    Sub(tmpLocal2_, yPointsRotated, yLocal, taskCount * 4);

    Mul(tmpLocal3_, tmpLocal1_, cosLocal, taskCount * 4);
    Mul(tmpLocal4_, tmpLocal2_, sinLocal, taskCount * 4);

    Add(tmpLocal3_, tmpLocal3_, tmpLocal4_, taskCount * 4);
    Add(tmpLocal3_, tmpLocal3_, xLocal, taskCount * 4);
    
    Mul(tmpLocal1_, tmpLocal1_, sinLocal, taskCount * 4);
    Mul(tmpLocal2_, tmpLocal2_, cosLocal, taskCount * 4);

    Sub(tmpLocal1_, tmpLocal2_, tmpLocal1_, taskCount * 4);
    Add(tmpLocal1_, tmpLocal1_, yLocal, taskCount * 4);

    // xCorners store in tmpLocal3_
    // yCorners store in tmpLocal1_
    Muls(wLocal, wLocal, static_cast<float>(0.5), taskCount * 4);
    Muls(hLocal, hLocal, static_cast<float>(0.5), taskCount * 4);

    LocalTensor<float> mask1Local = maskLocal[0 * tileTaskCountAligned_];
    LocalTensor<float> mask2Local = maskLocal[4 * tileTaskCountAligned_];
    LocalTensor<float> mask3Local = maskLocal[8 * tileTaskCountAligned_];
    LocalTensor<float> mask4Local = maskLocal[12 * tileTaskCountAligned_];

    // xCorners < right + margin
    Add(tmpLocal2_, xLocal, wLocal, taskCount * 4);
    Adds(tmpLocal2_, tmpLocal2_, margin, taskCount * 4);
    CompareLess(mask1Local, tmpLocal3_, tmpLocal2_, tmpLocal4_, taskCount * 4);

    // left - margin < xCorners
    Sub(tmpLocal2_, xLocal, wLocal, taskCount * 4);
    Adds(tmpLocal2_, tmpLocal2_, -margin, taskCount * 4);
    CompareLess(mask2Local, tmpLocal2_, tmpLocal3_, tmpLocal4_, taskCount * 4);

    // yCorners < top + margin
    Add(tmpLocal2_, yLocal, hLocal, taskCount * 4);
    Adds(tmpLocal2_, tmpLocal2_, margin, taskCount * 4);
    CompareLess(mask3Local, tmpLocal1_, tmpLocal2_, tmpLocal4_, taskCount * 4);

    // bottom - margin < yCorners
    Sub(tmpLocal2_, yLocal, hLocal, taskCount * 4);
    Adds(tmpLocal2_, tmpLocal2_, -margin, taskCount * 4);
    CompareLess(mask4Local, tmpLocal2_, tmpLocal1_, tmpLocal4_, taskCount * 4);

    Mul(mask1Local, mask1Local, mask2Local, taskCount * 4);
    Mul(mask1Local, mask1Local, mask3Local, taskCount * 4);
    Mul(mask1Local, mask1Local, mask4Local, taskCount * 4);
}

__aicore__ inline void KernelBoxesOverlapBevV1::Cross(LocalTensor<float>& res, const LocalTensor<float>& xP1, const LocalTensor<float>& yP1, const LocalTensor<float>& xP2,
    const LocalTensor<float>& yP2, LocalTensor<float>& tmpLocal1, LocalTensor<float>& tmpLocal2, const uint32_t calCount)
{
    Mul(tmpLocal1, xP1, yP2, calCount);
    Mul(tmpLocal2, xP2, yP1, calCount);

    Sub(res, tmpLocal1, tmpLocal2, calCount);
}

__aicore__ inline void KernelBoxesOverlapBevV1::ComputeIntersectionMask(LocalTensor<float>& mask, const LocalTensor<float>& xFrom1, const LocalTensor<float>& yFrom1, const LocalTensor<float>& xTo1,
    const LocalTensor<float>& yTo1, LocalTensor<float>& xMax2, LocalTensor<float>& xMin2, LocalTensor<float>& yMax2, LocalTensor<float>& yMin2, LocalTensor<float>& s1,
    LocalTensor<float>& s2, LocalTensor<float>& s3, LocalTensor<float>& s4, LocalTensor<float>& tmpLocal1, LocalTensor<float>& tmpLocal2, LocalTensor<float>& tmpLocal3, LocalTensor<float>& tmpLocal4, LocalTensor<float>& tmpLocal5,
    const uint32_t calCount)
{
    // // Compute mask1, store in mask
    Min(tmpLocal1, xFrom1, xTo1, calCount);
    CompareLessEqual(tmpLocal1, tmpLocal1, xMax2, tmpLocal5, calCount);

    Max(tmpLocal4, xFrom1, xTo1, calCount);
    CompareLessEqual(tmpLocal2, xMin2, tmpLocal4, tmpLocal5, calCount);

    Mul(tmpLocal1, tmpLocal1, tmpLocal2, calCount);

    Min(tmpLocal2, yFrom1, yTo1, calCount);
    CompareLessEqual(tmpLocal2, tmpLocal2, yMax2, tmpLocal5, calCount);

    Max(tmpLocal4, yFrom1, yTo1, calCount);
    CompareLessEqual(tmpLocal3, yMin2, tmpLocal4, tmpLocal5, calCount);

    Mul(tmpLocal2, tmpLocal2, tmpLocal3, calCount);
    
    Mul(mask, tmpLocal1, tmpLocal2, calCount);
    
    // Compute mask2, store in tmpLocal1
    Mul(tmpLocal1, s1, s2, calCount * 2);
    Sign(tmpLocal3, tmpLocal1, calCount * 2);
    Maxs(tmpLocal3, tmpLocal3, 0.0f, calCount * 2);
    Mul(tmpLocal1, tmpLocal3, tmpLocal4, calCount);

    // mask = mask1 and mask2
    Mul(mask, mask, tmpLocal1, calCount);
}

__aicore__ inline void KernelBoxesOverlapBevV1::ComputeIntersectionCorners(LocalTensor<float>& xCorners, LocalTensor<float>& yCorners, LocalTensor<float>& s1, LocalTensor<float>& s5,
    const LocalTensor<float>& xFrom1, const LocalTensor<float>& yFrom1, const LocalTensor<float>& xTo1, const LocalTensor<float>& yTo1, const LocalTensor<float>& xFrom2, const LocalTensor<float>& yFrom2,
    const LocalTensor<float>& xTo2, const LocalTensor<float>& yTo2, LocalTensor<float>& tmpLocal1, LocalTensor<float>& tmpLocal2, LocalTensor<float>& tmpLocal3, const uint32_t calCount)
{
    // Compute xCorners
    Sub(tmpLocal1, s5, s1, calCount);

    Mul(tmpLocal2, s5, xFrom2, calCount);
    Mul(tmpLocal3, s1, xTo2, calCount);
    Sub(tmpLocal2, tmpLocal2, tmpLocal3, calCount);
    Div(xCorners, tmpLocal2, tmpLocal1, calCount);

    // Compute yCorners
    Mul(tmpLocal2, s5, yFrom2, calCount);
    Mul(tmpLocal3, s1, yTo2, calCount);
    Sub(tmpLocal2, tmpLocal2, tmpLocal3, calCount);
    Div(yCorners, tmpLocal2, tmpLocal1, calCount);
}

__aicore__ inline void KernelBoxesOverlapBevV1::Intersection(LocalTensor<float>& xRotatedCorners1, LocalTensor<float>& yRotatedCorners1,
    LocalTensor<float>& xRotatedCorners2, LocalTensor<float>& yRotatedCorners2, LocalTensor<float>& intersectionMask,
    LocalTensor<float>& xIntersectionCorners, LocalTensor<float>& yIntersectionCorners, const uint32_t boxCount)
{
    LocalTensor<float> xFrom2 = xRotatedCorners2;
    LocalTensor<float> yFrom2 = yRotatedCorners2;
    LocalTensor<float> xTo2 = xRotatedCorners2[tileTaskCountAligned_];
    LocalTensor<float> yTo2 = yRotatedCorners2[tileTaskCountAligned_];
    LocalTensor<float> xFrom1;
    LocalTensor<float> yFrom1;
    LocalTensor<float> xTo1;
    LocalTensor<float> yTo1;

    Max(xMax2, xFrom2, xTo2, tileTaskCountAligned_ * 4);
    Min(xMin2, xFrom2, xTo2, tileTaskCountAligned_ * 4);
    Max(yMax2, yFrom2, yTo2, tileTaskCountAligned_ * 4);
    Min(yMin2, yFrom2, yTo2, tileTaskCountAligned_ * 4);
    Sub(innerTmpLocal9, xTo2, xFrom2, 4 * tileTaskCountAligned_);
    Sub(innerTmpLocal10, yTo2, yFrom2, 4 * tileTaskCountAligned_);

    uint32_t broadCastSrcShape[2] = {1, tileTaskCountAligned_};
    uint32_t broadCastDstShape[2] = {4, tileTaskCountAligned_};

    for (int32_t i = 0; i < 4; i++) {
        xFrom1 = xRotatedCorners1[i * tileTaskCountAligned_];
        yFrom1 = yRotatedCorners1[i * tileTaskCountAligned_];

        xTo1 = xRotatedCorners1[(i + 1) * tileTaskCountAligned_];
        yTo1 = yRotatedCorners1[(i + 1) * tileTaskCountAligned_];

        BroadCast<float, 2, 0, false>(xFrom1ForRepeat, xFrom1, broadCastDstShape, broadCastSrcShape);
        BroadCast<float, 2, 0, false>(yFrom1ForRepeat, yFrom1, broadCastDstShape, broadCastSrcShape);
        BroadCast<float, 2, 0, false>(xTo1ForRepeat, xTo1, broadCastDstShape, broadCastSrcShape);
        BroadCast<float, 2, 0, false>(yTo1ForRepeat, yTo1, broadCastDstShape, broadCastSrcShape);

        LocalTensor<float> mask = intersectionMask[(i * 4) * tileTaskCountAligned_];
        LocalTensor<float> xCorners = xIntersectionCorners[(i * 4) * tileTaskCountAligned_];
        LocalTensor<float> yCorners = yIntersectionCorners[(i * 4) * tileTaskCountAligned_];

        Sub(innerTmpLocal3, xTo1ForRepeat, xFrom1ForRepeat, 4 * tileTaskCountAligned_);
        Sub(innerTmpLocal4, yTo1ForRepeat, yFrom1ForRepeat, 4 * tileTaskCountAligned_);

        Sub(innerTmpLocal5, xFrom2, xFrom1ForRepeat, 4 * tileTaskCountAligned_);
        Sub(innerTmpLocal6, yFrom2, yFrom1ForRepeat, 4 * tileTaskCountAligned_);

        Sub(innerTmpLocal7, xTo2, xFrom1ForRepeat, 4 * tileTaskCountAligned_);
        Sub(innerTmpLocal8, yTo2, yFrom1ForRepeat, 4 * tileTaskCountAligned_);

        // Cross(From2, To1ForRepeat, From1ForRepeat)
        Cross(s1, innerTmpLocal5, innerTmpLocal6, innerTmpLocal3, innerTmpLocal4, innerTmpLocal1, innerTmpLocal2, 4 * tileTaskCountAligned_);

        // Cross(To1ForRepeat, To2, From1ForRepeat)
        Cross(s2, innerTmpLocal3, innerTmpLocal4, innerTmpLocal7, innerTmpLocal8, innerTmpLocal1, innerTmpLocal2, 4 * tileTaskCountAligned_);

        // Cross(To2, To1ForRepeat, From1ForRepeat)
        Cross(s5, innerTmpLocal7, innerTmpLocal8, innerTmpLocal3, innerTmpLocal4, innerTmpLocal1, innerTmpLocal2, 4 * tileTaskCountAligned_);

        Sub(innerTmpLocal3, xFrom1ForRepeat, xFrom2, 4 * tileTaskCountAligned_);
        Sub(innerTmpLocal4, yFrom1ForRepeat, yFrom2, 4 * tileTaskCountAligned_);

        Sub(innerTmpLocal7, xTo1ForRepeat, xFrom2, 4 * tileTaskCountAligned_);
        Sub(innerTmpLocal8, yTo1ForRepeat, yFrom2, 4 * tileTaskCountAligned_);

        // Cross(From1ForRepeat, To2, From2)
        Cross(s3, innerTmpLocal3, innerTmpLocal4, innerTmpLocal9, innerTmpLocal10, innerTmpLocal1, innerTmpLocal2, 4 * tileTaskCountAligned_);

        // Cross(To2, To1ForRepeat, From2)
        Cross(s4, innerTmpLocal9, innerTmpLocal10, innerTmpLocal7, innerTmpLocal8, innerTmpLocal1, innerTmpLocal2, 4 * tileTaskCountAligned_);

        // Compute mask
        ComputeIntersectionMask(mask, xFrom1ForRepeat, yFrom1ForRepeat, xTo1ForRepeat, yTo1ForRepeat, xMax2, xMin2, yMax2, yMin2,
            s1, s2, s3, s4, innerTmpLocal1, innerTmpLocal2, innerTmpLocal3, innerTmpLocal4, innerTmpLocal5, 4 * tileTaskCountAligned_);

        // Compute xIntersectionCorners & yIntersectionCorners
        ComputeIntersectionCorners(xCorners, yCorners, s1, s5, xFrom1ForRepeat, yFrom1ForRepeat, xTo1ForRepeat, yTo1ForRepeat, xFrom2, yFrom2, xTo2, yTo2, innerTmpLocal1, innerTmpLocal2, innerTmpLocal3, 4 * tileTaskCountAligned_);
    }
}

__aicore__ inline void KernelBoxesOverlapBevV1::ComputeOverlapArea(LocalTensor<float> &overlapAreaLocal, LocalTensor<float> &xVertices, LocalTensor<float> &yVertices,
    LocalTensor<int32_t> &sortedVerticesIdxLocal, LocalTensor<float> &tmpLocal, const uint32_t boxCount)
{
    LocalTensor<int32_t> sortedVerticesIdx1Local = tmpLocal.ReinterpretCast<int32_t>();
    LocalTensor<int32_t> sortedVerticesIdx2Local = tmpLocal[16 * boxCount].ReinterpretCast<int32_t>();
    LocalTensor<float> xVerticesSorted1Local = tmpLocal[32 * boxCount];
    LocalTensor<float> xVerticesSorted2Local = tmpLocal[48 * boxCount];
    LocalTensor<float> yVerticesSorted2Local = tmpLocal[64 * boxCount];
    LocalTensor<float> yVerticesSorted1Local = tmpLocal[80 * boxCount];

    Gather(sortedVerticesIdx1Local, sortedVerticesIdxLocal, gatherOffsetLocal3_, 0u, 16 * boxCount);
    Gather(sortedVerticesIdx2Local, sortedVerticesIdxLocal, gatherOffsetLocal4_, 0u, 16 * boxCount);

    Add(sortedVerticesIdx1Local, sortedVerticesIdx1Local, offsetLocal_.ReinterpretCast<int32_t>(), 16 * boxCount);
    Add(sortedVerticesIdx2Local, sortedVerticesIdx2Local, offsetLocal_.ReinterpretCast<int32_t>(), 16 * boxCount);

    Gather(xVerticesSorted1Local, xVertices, sortedVerticesIdx1Local.ReinterpretCast<uint32_t>(), 0u, 16 * boxCount);
    Gather(yVerticesSorted1Local, yVertices, sortedVerticesIdx1Local.ReinterpretCast<uint32_t>(), 0u, 16 * boxCount);
    Gather(xVerticesSorted2Local, xVertices, sortedVerticesIdx2Local.ReinterpretCast<uint32_t>(), 0u, 16 * boxCount);
    Gather(yVerticesSorted2Local, yVertices, sortedVerticesIdx2Local.ReinterpretCast<uint32_t>(), 0u, 16 * boxCount);
    
    // Compute area
    Mul(xVerticesSorted1Local, xVerticesSorted1Local, yVerticesSorted2Local, 32 * boxCount);
    Sub(xVerticesSorted1Local, xVerticesSorted1Local, xVerticesSorted2Local, 16 * boxCount);
    WholeReduceSum(xVerticesSorted1Local, xVerticesSorted1Local, 12u, boxCount, 1u, 1u, 2u);
    Abs(overlapAreaLocal, xVerticesSorted1Local, boxCount);
    Muls(overlapAreaLocal, overlapAreaLocal, 0.5f, boxCount);
}

extern "C" __global__ __aicore__ void boxes_overlap_bev_v1(GM_ADDR boxes_a, GM_ADDR boxes_b, GM_ADDR res,
    GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    KernelBoxesOverlapBevV1 op;
    op.Init(&pipe, boxes_a, boxes_b, res, &tilingData);
    op.Process();
}