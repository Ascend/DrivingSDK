#include "kernel_operator.h"
#include "boxes_operator_utils.h"

using namespace AscendC;

constexpr uint32_t INT32_BYTE_SIZE = 4;

// 最小任务块中的查询点数 / 字节数
constexpr uint32_t BLOCK_POINT_SIZE = 8;
constexpr uint32_t BLOCK_BYTE_SIZE = 96;
constexpr int32_t REPEAT_STRIDE_0 = 3;
constexpr int32_t REPEAT_STRIDE_1 = 0;

constexpr uint32_t MASK_BEGIN_AT_0 = 1227133513;
constexpr uint32_t MASK_BEGIN_AT_1 = 2454267026;
constexpr uint32_t MASK_BEGIN_AT_2 = 613566756;

constexpr uint32_t DUBBLE_BUFFER = 2;
constexpr uint32_t POINT_DIM = 3;
constexpr uint32_t ROT_SIZE = 9;

constexpr uint32_t MASK_PATTERN_OFFSET_0 = 0;
constexpr uint32_t MASK_PATTERN_OFFSET_1 = 1;
constexpr uint32_t MASK_PATTERN_OFFSET_2 = 2;

constexpr uint32_t POINT_X_OFFSET = 0;
constexpr uint32_t POINT_Y_OFFSET = 1;
constexpr uint32_t POINT_Z_OFFSET = 2;

constexpr uint32_t ROT_VALUE_OFFSET_0 = 0;
constexpr uint32_t ROT_VALUE_OFFSET_1 = 1;
constexpr uint32_t ROT_VALUE_OFFSET_2 = 2;
constexpr uint32_t ROT_VALUE_OFFSET_3 = 3;
constexpr uint32_t ROT_VALUE_OFFSET_4 = 4;
constexpr uint32_t ROT_VALUE_OFFSET_5 = 5;
constexpr uint32_t ROT_VALUE_OFFSET_6 = 6;
constexpr uint32_t ROT_VALUE_OFFSET_7 = 7;
constexpr uint32_t ROT_VALUE_OFFSET_8 = 8;


#define Ceil32(num) (((num) + 31) / 32 * 32)
#define Ceil64(num) (((num) + 63) / 64 * 64)
#define CeilDiv8(num) (((num) + 7) / 8)

class CylinderQuery {
public:
    __aicore__ inline CylinderQuery() {}

    __aicore__ inline void Init(TPipe *pipe, GM_ADDR newXyz, GM_ADDR xyz, GM_ADDR rot, GM_ADDR origin_index,
        GM_ADDR res, const CylinderQueryTilingData* tiling)
    {
        this->pipe_ = pipe;
        this->blkIdx_ = GetBlockIdx();
        InitTiling(tiling);
        InitUB();
        InitMask();
        InitGM(newXyz, xyz, rot, origin_index, res);
        InitEvent();
    }
    __aicore__ inline void Process()
    {
        for (int i = 0; i < this->coreTask_; i++) {
            uint32_t offset = this->taskOffset_ + i;
            this->batchIdx_ = offset / this->queryPointSize_;
            CopyNewXyzIn(offset);
            SetFlag<HardEvent::S_V>(eventSV_);
            WaitFlag<HardEvent::S_V>(eventSV_);
            this->processDataNum_ = this->tileDataNum_; // 本轮运算实际参与计算的元素个数

            for (int j = 0; j < this->finalSmallTileNum_; j++) {
                SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2_);
                WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2_);
                int resOffset = j * processDataNum_;
                this->processBlockNum_ = this->tileBlockNum_;
                if (j == finalSmallTileNum_ - 1) {
                    this->processBlockNum_ = this->smallTileBlockNum_;
                    this->processDataNum_ = smallTileDataNum_;
                }
                InitRes(resOffset);
                CopyIn(this->tileDataNum_ * j);
                SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
                WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
                Compute(i, j);
                SetFlag<HardEvent::V_MTE3>(eventVMTE3_);
                WaitFlag<HardEvent::V_MTE3>(eventVMTE3_);
                CopyOut(offset, this->tileDataNum_ * j, this->processBlockNum_);
            }
        }
    }

private:
    __aicore__ inline void InitTiling(const CylinderQueryTilingData* tiling)
    {
        this->coreTask_ = tiling->coreTask;
        if (blkIdx_ < tiling->bigCoreCount) {
            this->taskOffset_ = blkIdx_ * coreTask_;
        } else {
            this->taskOffset_ = tiling->bigCoreCount * coreTask_ +
                (blkIdx_ - tiling->bigCoreCount) * (coreTask_ - 1);
            this->coreTask_ = this->coreTask_ - 1;
        }
        this->tileBlockNum_ = tiling->tileBlockNum;
        this->smallTileBlockNum_ = tiling->smallTileBlockNum;
        // copy tiling数据与计算
        this->radius2_ = tiling->radius * tiling->radius;
        this->hmin_ = tiling->hmin;
        this->hmax_ = tiling->hmax;
        this->nsample_ = tiling->nsample;
        this->pointCloudSize_ = tiling->pointCloudSize;
        this->queryPointSize_ = tiling->queryPointSize;
        this->finalSmallTileNum_ = tiling->finalSmallTileNum;
        this->smallTileDataNum_ = tiling->smallTileDataNum; // 最后一次循环计算的元素个数
        this->tileDataNum_ = tiling->tileDataNum; // 每次循环计算的元素个数
    }

    // 将cpu侧ternsor搬运到kernel侧
    __aicore__ inline void InitGM(GM_ADDR newXyz, GM_ADDR xyz, GM_ADDR rot, GM_ADDR origin_index, GM_ADDR res)
    {
        this->newXyzGm_.SetGlobalBuffer((__gm__ float*) newXyz);
        this->xyzGm_.SetGlobalBuffer((__gm__ float*) xyz);
        this->rotGm_.SetGlobalBuffer((__gm__ float*) rot);
        this->resGm_.SetGlobalBuffer((__gm__ float*) res);
        this->originIndexGm_.SetGlobalBuffer((__gm__ float*) origin_index);
    }

    __aicore__ inline void InitUB()
    {
        // total: 336x + 736 + nsample_ * 8
        // 96x + 96x + 9 * FLOAT_BYTE_SIZE = 192x + 9 * FLOAT_BYTE_SIZE
        pipe_->InitBuffer(xyzBuf_, DUBBLE_BUFFER * tileBlockNum_ * BLOCK_BYTE_SIZE); // 2(3xf + 95)
        pipe_->InitBuffer(xBuf_, DUBBLE_BUFFER * tileBlockNum_ * BLOCK_BYTE_SIZE / POINT_DIM); // 2xf
        pipe_->InitBuffer(yBuf_, DUBBLE_BUFFER * tileBlockNum_ * BLOCK_BYTE_SIZE / POINT_DIM); // 2xf
        pipe_->InitBuffer(zBuf_, DUBBLE_BUFFER * tileBlockNum_ * BLOCK_BYTE_SIZE / POINT_DIM); // 2xf
        pipe_->InitBuffer(rotBuf_, DUBBLE_BUFFER * ROT_SIZE * FLOAT_BYTE_SIZE); // 18f

        // 32x * 5 = 96x
        pipe_->InitBuffer(rotXBuf_, DUBBLE_BUFFER * tileBlockNum_ * BLOCK_BYTE_SIZE / POINT_DIM); // 2(xf + 255)
        pipe_->InitBuffer(rotYBuf_, DUBBLE_BUFFER * tileBlockNum_ * BLOCK_BYTE_SIZE / POINT_DIM); // 2(xf + 255)
        pipe_->InitBuffer(rotZBuf_, DUBBLE_BUFFER * tileBlockNum_ * BLOCK_BYTE_SIZE / POINT_DIM); // 2(xf + 255)

        // 16x
        // compare的时候count个元素占用的字节需要时256字节，即float向64个对齐
        pipe_->InitBuffer(maskD2Buf_, DUBBLE_BUFFER * 1 * CeilDiv8(Ceil64(this->tileDataNum_))); // 2((x + 63) /32)
        pipe_->InitBuffer(maskHBuf_, DUBBLE_BUFFER * 1 * CeilDiv8(Ceil64(this->tileDataNum_))); // 2((x + 63) /32)

        // nsample_ * INT32_BYTE_SIZE + (x * 8 + 31) * 4 = nsample_ * 4 + 32 * x + 124
        pipe_->InitBuffer(scr1PatternBuf_, DUBBLE_BUFFER * nsample_ * INT32_BYTE_SIZE);
        pipe_->InitBuffer(resBuf_, DUBBLE_BUFFER * Ceil32(tileDataNum_) * FLOAT_BYTE_SIZE);

        // 96 * 3 * 2 = 576
        pipe_->InitBuffer(bufferMaskXBuf, DUBBLE_BUFFER * BLOCK_BYTE_SIZE);
        pipe_->InitBuffer(bufferMaskYBuf, DUBBLE_BUFFER * BLOCK_BYTE_SIZE);
        pipe_->InitBuffer(bufferMaskZBuf, DUBBLE_BUFFER * BLOCK_BYTE_SIZE);

        xyzLocal_ = xyzBuf_.Get<float>();
        xLocal_ = xBuf_.Get<float>();
        yLocal_ = yBuf_.Get<float>();
        zLocal_ = zBuf_.Get<float>();
        rotLocal_ = rotBuf_.Get<float>();

        rotXLocal_ = rotXBuf_.Get<float>();
        rotYLocal_ = rotYBuf_.Get<float>();
        rotZLocal_ = rotZBuf_.Get<float>();

        maskD2Local_ = maskD2Buf_.Get<uint8_t>();
        maskHLocal_ = maskHBuf_.Get<uint8_t>();

        resLocal_ = resBuf_.Get<float>();
    }

    __aicore__ inline void InitRes(int offset)
    {
        DataCopyExtParams originIndexDataCopyParams{static_cast<uint16_t>(1), Ceil32(this->processDataNum_) * FLOAT_BYTE_SIZE, 0, 0, 0};
        DataCopyPad(resLocal_, originIndexGm_[offset], originIndexDataCopyParams, this->originIndexPadParams_);
    }

    __aicore__ inline void InitMask()
    {
        xPattern_ = bufferMaskXBuf.Get<uint32_t>();
        yPattern_ = bufferMaskYBuf.Get<uint32_t>();
        zPattern_ = bufferMaskZBuf.Get<uint32_t>();

        // Set pattern values for x to select first element of three
        xPattern_.SetValue(MASK_PATTERN_OFFSET_0, MASK_BEGIN_AT_0);
        xPattern_.SetValue(MASK_PATTERN_OFFSET_1, MASK_BEGIN_AT_1);
        xPattern_.SetValue(MASK_PATTERN_OFFSET_2, MASK_BEGIN_AT_2);

        // // Set pattern values for y to select second element of three
        yPattern_.SetValue(MASK_PATTERN_OFFSET_0, MASK_BEGIN_AT_1);
        yPattern_.SetValue(MASK_PATTERN_OFFSET_1, MASK_BEGIN_AT_2);
        yPattern_.SetValue(MASK_PATTERN_OFFSET_2, MASK_BEGIN_AT_0);

        // // Set pattern values for z to select third element of three
        zPattern_.SetValue(MASK_PATTERN_OFFSET_0, MASK_BEGIN_AT_2);
        zPattern_.SetValue(MASK_PATTERN_OFFSET_1, MASK_BEGIN_AT_0);
        zPattern_.SetValue(MASK_PATTERN_OFFSET_2, MASK_BEGIN_AT_1);
    }

    __aicore__ inline void InitEvent()
    {
        eventMTE2V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        eventVMTE3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        eventMTE2S_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        eventMTE3MTE2_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        eventSV_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    }

    // 根据查询点偏移量读取数据
    __aicore__ inline void CopyNewXyzIn(uint32_t offset);
    __aicore__ inline void CopyIn(uint32_t offset);
    __aicore__ inline void CopyOut(uint32_t taskOffset, uint32_t dataOffset, uint32_t taskCount);
    __aicore__ inline void Compute(int taskOffset, int xyzOffset);

private:
    TPipe* pipe_;
    int32_t eventMTE2V_, eventVMTE3_, eventMTE2S_, eventMTE3MTE2_, eventSV_;
    uint32_t blkIdx_;
    uint32_t processBlockNum_, processDataNum_;
    uint32_t coreTask_, taskOffset_;
    uint32_t finalSmallTileNum_, smallTileDataNum_, tileDataNum_;
    uint32_t tileBlockNum_, smallTileBlockNum_;

    uint32_t batchIdx_; // 记录查询点在哪一个batch

    GlobalTensor<float> newXyzGm_;
    GlobalTensor<float> xyzGm_;
    GlobalTensor<float> rotGm_;
    GlobalTensor<float> resGm_;
    GlobalTensor<float> originIndexGm_;

    float radius2_;
    float hmin_, hmax_;
    uint32_t nsample_, pointCloudSize_, queryPointSize_;

    TBuf<TPosition::VECCALC> xyzBuf_, xBuf_, yBuf_, zBuf_, rotBuf_, resBuf_, scr1PatternBuf_;
    TBuf<TPosition::VECCALC> bufferMaskXBuf, bufferMaskYBuf, bufferMaskZBuf;
    TBuf<TPosition::VECCALC> rotXBuf_, rotYBuf_, rotZBuf_;
    TBuf<TPosition::VECCALC> maskD2Buf_, maskHBuf_;

    LocalTensor<float> xyzLocal_, xLocal_, yLocal_, zLocal_, rotLocal_;
    LocalTensor<float> rotXLocal_, rotYLocal_, rotZLocal_;
    LocalTensor<uint8_t> maskD2Local_, maskHLocal_;
    float r0, r1, r2, r3, r4, r5, r6, r7, r8;

    LocalTensor<float> resLocal_;
    LocalTensor<uint32_t> xPattern_, yPattern_, zPattern_;
    
    uint32_t mask_ = 0;
    float newX_, newY_, newZ_;

    // 不填充数据
    DataCopyPadExtParams<float> newXyzPadParams_{false, 0, 0, 0};
    DataCopyPadExtParams<float> xyzPadParams_{false, 0, 0, 0};
    DataCopyPadExtParams<float> rotPadParams_{false, 0, 0, 0};
    DataCopyPadExtParams<float> originIndexPadParams_{false, 0, 0, 0};
};


__aicore__ inline void CylinderQuery::CopyNewXyzIn(uint32_t offset)
{
    this->newX_ = newXyzGm_.GetValue(offset * POINT_DIM + POINT_X_OFFSET);
    this->newY_ = newXyzGm_.GetValue(offset * POINT_DIM + POINT_Y_OFFSET);
    this->newZ_ = newXyzGm_.GetValue(offset * POINT_DIM + POINT_Z_OFFSET);
    DataCopyExtParams rotDataCopyParams{static_cast<uint16_t>(1), ROT_SIZE * FLOAT_BYTE_SIZE, 0, 0, 0};
    // MTE2
    DataCopyPad(rotLocal_, rotGm_[static_cast<uint64_t>(offset * ROT_SIZE)], rotDataCopyParams, this->rotPadParams_);
    SetFlag<HardEvent::MTE2_S>(eventMTE2S_);
    WaitFlag<HardEvent::MTE2_S>(eventMTE2S_);

    r0 = rotLocal_.GetValue(ROT_VALUE_OFFSET_0);
    r1 = rotLocal_.GetValue(ROT_VALUE_OFFSET_1);
    r2 = rotLocal_.GetValue(ROT_VALUE_OFFSET_2);
    r3 = rotLocal_.GetValue(ROT_VALUE_OFFSET_3);
    r4 = rotLocal_.GetValue(ROT_VALUE_OFFSET_4);
    r5 = rotLocal_.GetValue(ROT_VALUE_OFFSET_5);
    r6 = rotLocal_.GetValue(ROT_VALUE_OFFSET_6);
    r7 = rotLocal_.GetValue(ROT_VALUE_OFFSET_7);
    r8 = rotLocal_.GetValue(ROT_VALUE_OFFSET_8);
}

// offset：查询点偏移量
__aicore__ inline void CylinderQuery::CopyIn(uint32_t offset)
{
    DataCopyExtParams xyzDataCopyParams{static_cast<uint16_t>(1), processDataNum_ * POINT_DIM * FLOAT_BYTE_SIZE, 0, 0, 0};
    DataCopyPad(xyzLocal_, xyzGm_[POINT_DIM * (static_cast<uint64_t>(offset) + this->batchIdx_ * this->pointCloudSize_)], xyzDataCopyParams, this->xyzPadParams_);
}

__aicore__ inline void CylinderQuery::CopyOut(uint32_t taskOffset, uint32_t dataOffset, uint32_t blockCount)
{
    DataCopyExtParams xyzDataCopyParams{static_cast<uint16_t>(1), processDataNum_ * FLOAT_BYTE_SIZE, 0, 0, 0};
    DataCopyPad(resGm_[static_cast<uint64_t>(dataOffset) + taskOffset * pointCloudSize_],
        resLocal_, xyzDataCopyParams);
}

__aicore__ inline void CylinderQuery::Compute(int taskOffset, int xyzOffset)
{
    // 需要先将x，y，z分别取出
    uint32_t processDataNumAlign8 = this->processBlockNum_ * BLOCK_POINT_SIZE;
    uint32_t processDataNumAlign64 = Ceil64(processDataNumAlign8);

    bool reduceMode = true;
    uint32_t mask = BLOCK_POINT_SIZE * 3;
    uint8_t src1Pattern = 2;

    uint8_t src0BlockStride = 1;
    uint16_t repeatTimes = this->processBlockNum_;
    uint8_t src0RepeatStride = REPEAT_STRIDE_0;
    uint8_t src1RepeatStride = REPEAT_STRIDE_1;
    
    uint64_t rsvdCnt = 0;
    GatherMask(xLocal_, xyzLocal_, xPattern_, reduceMode, mask,
               {1, repeatTimes, src0RepeatStride, src1RepeatStride}, rsvdCnt);
    GatherMask(yLocal_, xyzLocal_, yPattern_, reduceMode, mask,
               {1, repeatTimes, src0RepeatStride, src1RepeatStride}, rsvdCnt);
    GatherMask(zLocal_, xyzLocal_, zPattern_, reduceMode, mask,
               {1, repeatTimes, src0RepeatStride, src1RepeatStride}, rsvdCnt);
    // 计算相对位置

    Adds(xLocal_, xLocal_, -newX_, processDataNumAlign8);
    Adds(yLocal_, yLocal_, -newY_, processDataNumAlign8);
    Adds(zLocal_, zLocal_, -newZ_, processDataNumAlign8);

    Muls(rotXLocal_, xLocal_, r0, processDataNumAlign8);
    Axpy(rotXLocal_, yLocal_, r3, processDataNumAlign8);
    Axpy(rotXLocal_, zLocal_, r6, processDataNumAlign8);

    Muls(rotYLocal_, xLocal_, r1, processDataNumAlign8);
    Axpy(rotYLocal_, yLocal_, r4, processDataNumAlign8);
    Axpy(rotYLocal_, zLocal_, r7, processDataNumAlign8);

    Muls(rotZLocal_, xLocal_, r2, processDataNumAlign8);
    Axpy(rotZLocal_, yLocal_, r5, processDataNumAlign8);
    Axpy(rotZLocal_, zLocal_, r8, processDataNumAlign8);

    // 节省空间服用tensor
    Mul(rotYLocal_, rotYLocal_, rotYLocal_, processDataNumAlign8);
    Mul(rotZLocal_, rotZLocal_, rotZLocal_, processDataNumAlign8);
    Add(rotYLocal_, rotYLocal_, rotZLocal_, processDataNumAlign8);
    CompareScalar(maskD2Local_, rotYLocal_, this->radius2_, AscendC::CMPMODE::LT, processDataNumAlign64);
    CompareScalar(maskHLocal_, rotXLocal_, this->hmin_, AscendC::CMPMODE::GT, processDataNumAlign64);
    And(maskD2Local_, maskD2Local_, maskHLocal_, CeilDiv8(processDataNumAlign64));
    CompareScalar(maskHLocal_, rotXLocal_, this->hmax_, AscendC::CMPMODE::LT, processDataNumAlign64);
    And(maskD2Local_, maskD2Local_, maskHLocal_, CeilDiv8(processDataNumAlign64));
    
    Select(resLocal_, maskD2Local_, resLocal_, float(int32_t(pointCloudSize_)), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum_);
}


extern "C" __global__ __aicore__ void cylinder_query(GM_ADDR newXyz, GM_ADDR xyz, GM_ADDR rot, GM_ADDR origin_index, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(cylinderQueryTiling, tiling);
    TPipe pipe;
    CylinderQuery op;
    op.Init(&pipe, newXyz, xyz, rot, origin_index, out, &cylinderQueryTiling);
    op.Process();
}