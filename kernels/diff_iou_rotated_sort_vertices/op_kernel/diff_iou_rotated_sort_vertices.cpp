#include "kernel_operator.h"
#include "boxes_operator_utils.h"

using namespace AscendC;
constexpr uint32_t UB_ALIGNED_BYTE_SIZE = 32;
constexpr uint32_t VERTICES_CORR = 2;
constexpr uint32_t INT32_BYTE_SIZE = 4;
constexpr uint32_t OUTPUT_IDX_COUNT = 9;
constexpr uint32_t MASK_ALIGNED = 32;
constexpr uint32_t VERTICE_XY_ALIGNED = 64;
constexpr float EPS = 1e-12;

class DiffIouRotatedSortVertices {
public:
    __aicore__ inline DiffIouRotatedSortVertices() {}
    __aicore__ inline void Init(TPipe *pipe, GM_ADDR vertices, GM_ADDR mask, GM_ADDR num_valid,
        GM_ADDR sortedIdx, const DiffIouRotatedSortVerticesTilingData* tiling)
    {
        pipe_ = pipe;
        blkIdx_ = GetBlockIdx();
        InitTiling(tiling);
        InitUB();
        InitGM(vertices, mask, num_valid, sortedIdx);
        InitEvent();
    }
    __aicore__ inline void Process()
    {
        // Compute some const idx
        CreateVecIndex(sortIdxLocal1_, 0, VERTICES_ALIGNED, 1, 1, 4);
        BroadCast<int32_t, 2, 0, false>(sortIdxLocal_, sortIdxLocal1_, broadCastDstShape2_, broadCastSrcShape2_);
        
        uint32_t endTaskOffset = taskOffset_ + coreTask_;
        for (int32_t offset = taskOffset_; offset < endTaskOffset; offset += singleLoopTaskCount_) {
            uint32_t taskCount = min(singleLoopTaskCount_, endTaskOffset - offset);

            SetFlag<HardEvent::V_MTE2>(eventVMTE2_);
            WaitFlag<HardEvent::V_MTE2>(eventVMTE2_);

            CopyIn(offset, taskCount);
            
            SetFlag<HardEvent::MTE2_V>(eventMTE2V_);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V_);
            
            Compute();
            
            SetFlag<HardEvent::V_MTE3>(eventVMTE3_);
            WaitFlag<HardEvent::V_MTE3>(eventVMTE3_);

            CopyOut(offset, taskCount);
        }
    }

private:
    __aicore__ inline void InitTiling(const DiffIouRotatedSortVerticesTilingData* tiling)
    {
        this->coreTask_ = tiling->coreTask;
        if (blkIdx_ < tiling->bigCoreCount) {
            this->taskOffset_ = blkIdx_ * coreTask_;
        } else {
            this->taskOffset_ = tiling->bigCoreCount * coreTask_ +
                (blkIdx_ - tiling->bigCoreCount) * (coreTask_ - 1);
            this->coreTask_ = this->coreTask_ - 1;
        }
        this->singleLoopTaskCount_ = tiling->singleLoopTaskCount;
        rsvdCnt_ = singleLoopTaskCount_ * VERTICES_ALIGNED;
        repeatTimes_ = Ceil(singleLoopTaskCount_ * VERTICE_XY_ALIGNED, static_cast<uint32_t>(64));       // repeatTimes <= 255
        calCount_ = singleLoopTaskCount_ * VERTICES_ALIGNED;
        singleLoopTaskCountAligned_ = Ceil(singleLoopTaskCount_, 8) * 8;

        broadCastSrcShape2_[0] = 1;
        broadCastSrcShape2_[1] = VERTICES_ALIGNED;

        broadCastDstShape2_[0] = singleLoopTaskCount_;
        broadCastDstShape2_[1] = VERTICES_ALIGNED;
    }

    __aicore__ inline void InitGM(GM_ADDR vertices, GM_ADDR mask, GM_ADDR num_valid, GM_ADDR sortedIdx)
    {
        this->verticesGm_.SetGlobalBuffer((__gm__ float*) vertices);
        this->maskGm_.SetGlobalBuffer((__gm__ float*) mask);
        this->numValidGm_.SetGlobalBuffer((__gm__ int32_t*) num_valid);
        this->sortedIdxGm_.SetGlobalBuffer((__gm__ int32_t*) sortedIdx);
    }

    __aicore__ inline void InitUB()
    {
        pipe_->InitBuffer(verticesBuf_, 2 * calCount_ * FLOAT_BYTE_SIZE);
        pipe_->InitBuffer(posBuf_, 2 * calCount_ * FLOAT_BYTE_SIZE);
        pipe_->InitBuffer(outputBuf_, 2 * calCount_ * FLOAT_BYTE_SIZE);
        pipe_->InitBuffer(sortIdxBuf_, 2 * calCount_ * INT32_BYTE_SIZE);
        pipe_->InitBuffer(maskBuf_, singleLoopTaskCount_ * MASK_ALIGNED * FLOAT_BYTE_SIZE);
        pipe_->InitBuffer(numValidBuf_, Ceil(singleLoopTaskCountAligned_ * INT32_BYTE_SIZE, UB_ALIGNED_BYTE_SIZE) * UB_ALIGNED_BYTE_SIZE);
        pipe_->InitBuffer(tmpBuf_, 8 * singleLoopTaskCountAligned_ * FLOAT_BYTE_SIZE + 2 * VERTICES_ALIGNED * singleLoopTaskCountAligned_ * FLOAT_BYTE_SIZE);

        verticesLocal_ = verticesBuf_.Get<float>();
        maskLocal_ = maskBuf_.Get<float>();
        numValidLocal_ = numValidBuf_.Get<int32_t>();
        posLocal_ = posBuf_.Get<float>();
        outputLocal_ = outputBuf_.Get<int32_t>();
        sortIdxLocal_ = sortIdxBuf_.Get<int32_t>();
        sortIdxLocal1_ = sortIdxLocal_[calCount_];
        tmpLocal_ = tmpBuf_.Get<float>();
    }

    __aicore__ inline void InitEvent()
    {
        eventMTE2V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        eventVMTE3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        eventMTE3V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        eventVMTE2_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t taskCount);
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t taskCount);
    __aicore__ inline void Compute();

private:
    uint16_t repeatTimes_;
    TPipe* pipe_;
    int32_t eventMTE2V_, eventVMTE3_, eventMTE3V_, eventVMTE2_;
    uint32_t blkIdx_;
    uint32_t coreTask_, taskOffset_, singleLoopTaskCount_, calCount_, singleLoopTaskCountAligned_;
    uint32_t broadCastSrcShape2_[2];
    uint32_t broadCastDstShape2_[2];
    uint32_t mask_ = 0;
    uint64_t rsvdCnt_;

    GlobalTensor<float> verticesGm_;
    GlobalTensor<float> maskGm_;
    GlobalTensor<int32_t> numValidGm_;
    GlobalTensor<int32_t> sortedIdxGm_;

    TBuf<TPosition::VECCALC> verticesBuf_, maskBuf_, numValidBuf_, sortIdxBuf_,
        posBuf_, outputBuf_, tmpBuf_;
    LocalTensor<float> verticesLocal_, posLocal_, maskLocal_, tmpLocal_;
    LocalTensor<int32_t> numValidLocal_, sortIdxLocal_, sortIdxLocal1_, outputLocal_;
    DataCopyPadExtParams<float> verticesPadParams_{false, 0, 0, 0};
    DataCopyPadExtParams<float> maskPadParams_{false, 0, 0, 0};
    DataCopyPadExtParams<int32_t> numValidPadParams_{false, 0, 0, 0};
    DataCopyPadExtParams<int32_t> sortedIdxPadParams_{false, 0, 0, 0};
};

__aicore__ inline void DiffIouRotatedSortVertices::CopyIn(uint32_t offset, uint32_t taskCount)
{
    DataCopyExtParams verticesDataCopyParams{static_cast<uint16_t>(taskCount), VERTICES_COUNT * VERTICES_CORR * FLOAT_BYTE_SIZE, 0, 2, 0};
    DataCopyExtParams maskDataCopyParams{static_cast<uint16_t>(taskCount), VERTICES_COUNT * FLOAT_BYTE_SIZE, 0, 1, 0};
    DataCopyExtParams numValidDataCopyParams{1, taskCount * INT32_BYTE_SIZE, 0, 0, 0};

    DataCopyPad(verticesLocal_, verticesGm_[static_cast<uint64_t>(offset) * VERTICES_COUNT * VERTICES_CORR], verticesDataCopyParams, verticesPadParams_);
    DataCopyPad(maskLocal_, maskGm_[static_cast<uint64_t>(offset) * VERTICES_COUNT], maskDataCopyParams, maskPadParams_);
    DataCopyPad(numValidLocal_, numValidGm_[offset], numValidDataCopyParams, numValidPadParams_);
}

__aicore__ inline void DiffIouRotatedSortVertices::CopyOut(uint32_t offset, uint32_t taskCount)
{
    DataCopyExtParams copyOutParams{static_cast<uint16_t>(taskCount),  OUTPUT_IDX_COUNT * FLOAT_BYTE_SIZE, 2, 0, 0};
    DataCopyPad(sortedIdxGm_[static_cast<uint64_t>(offset) * OUTPUT_IDX_COUNT], outputLocal_, copyOutParams);
}

__aicore__ inline void DiffIouRotatedSortVertices::Compute()
{
    uint8_t src1Pattern = 1;
    GatherMask(posLocal_, verticesLocal_, src1Pattern, false, mask_, { 1, repeatTimes_, 8, 0 }, rsvdCnt_);
    src1Pattern = 2;
    GatherMask(posLocal_[calCount_], verticesLocal_, src1Pattern, false, mask_, { 1, repeatTimes_, 8, 0 }, rsvdCnt_);

    LocalTensor<float> xVertices = posLocal_;
    LocalTensor<float> yVertices = posLocal_[calCount_];

    SortVertices(outputLocal_, xVertices, yVertices, maskLocal_, numValidLocal_, sortIdxLocal_, tmpLocal_, singleLoopTaskCount_, true);
}

extern "C" __global__ __aicore__ void diff_iou_rotated_sort_vertices(GM_ADDR vertices, GM_ADDR mask, GM_ADDR num_valid,
    GM_ADDR sortedIdx, GM_ADDR workspace, GM_ADDR tiling_data)
{
    GET_TILING_DATA(tiling, tiling_data);
    SetSysWorkspace(workspace);
    TPipe pipe;
    DiffIouRotatedSortVertices op;
    op.Init(&pipe, vertices, mask, num_valid, sortedIdx, &tiling);
    op.Process();
}