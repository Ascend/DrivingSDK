#include "kernel_operator.h"

using namespace AscendC;
constexpr float POSITIVE_INF = 1e24;
constexpr uint32_t UB_ALIGNED_BYTE_SIZE = 32;
constexpr uint32_t VERTICES_COUNT = 24;
constexpr uint32_t VERTICES_CORR = 2;
constexpr uint32_t FLOAT_BYTE_SIZE = 4;
constexpr uint32_t INT32_BYTE_SIZE = 4;
constexpr uint32_t OUTPUT_IDX_COUNT = 9;
constexpr uint32_t VERTICES_ALIGNED = 32;
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
        CreateVecIndex(vecIdxLocal_, static_cast<int32_t>(0), singleLoopTaskCount_);
        CreateVecIndex(sortIdxLocal_, 0, VERTICES_ALIGNED, 1, 1, 4);
        Muls(vecIdxLocal_, vecIdxLocal_, static_cast<int32_t>(VERTICES_ALIGNED * FLOAT_BYTE_SIZE + 0.0f), singleLoopTaskCount_);
        BroadCast<int32_t, 2, 0, false>(sortIdxLocal2_, sortIdxLocal_, broadCastDstShape2_, broadCastSrcShape2_);

        uint32_t endTaskOffset = taskOffset_ + coreTask_;
        for (int32_t offset = taskOffset_; offset < endTaskOffset; offset += singleLoopTaskCount_) {
            uint32_t taskCount = min(singleLoopTaskCount_, endTaskOffset - offset);

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
        
        broadCastSrcShape1_[0] = singleLoopTaskCount_;
        broadCastSrcShape1_[1] = 1;

        broadCastDstShape1_[0] = singleLoopTaskCount_;
        broadCastDstShape1_[1] = VERTICES_ALIGNED;

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
        pipe_->InitBuffer(verticesBuf_, singleLoopTaskCount_ * VERTICES_ALIGNED * FLOAT_BYTE_SIZE * 2);
        pipe_->InitBuffer(posBuf_, singleLoopTaskCount_ * VERTICES_ALIGNED * FLOAT_BYTE_SIZE * 2);
        pipe_->InitBuffer(outputBuf_, singleLoopTaskCount_ * VERTICES_ALIGNED * FLOAT_BYTE_SIZE * 2);
        pipe_->InitBuffer(sortIdxBuf_, singleLoopTaskCount_ * VERTICES_ALIGNED * INT32_BYTE_SIZE * 3);
        pipe_->InitBuffer(maskBuf_, singleLoopTaskCount_ * MASK_ALIGNED * FLOAT_BYTE_SIZE);
        pipe_->InitBuffer(numValidBuf_, Ceil(singleLoopTaskCount_ * INT32_BYTE_SIZE, UB_ALIGNED_BYTE_SIZE) * UB_ALIGNED_BYTE_SIZE);
        pipe_->InitBuffer(argminBuf_, Ceil(singleLoopTaskCount_ * FLOAT_BYTE_SIZE, UB_ALIGNED_BYTE_SIZE) * UB_ALIGNED_BYTE_SIZE);
        pipe_->InitBuffer(vecIdxBuf_, Ceil(singleLoopTaskCount_ * INT32_BYTE_SIZE, UB_ALIGNED_BYTE_SIZE) * UB_ALIGNED_BYTE_SIZE);
        pipe_->InitBuffer(minValBuf_, Ceil(singleLoopTaskCount_ * FLOAT_BYTE_SIZE, UB_ALIGNED_BYTE_SIZE) * UB_ALIGNED_BYTE_SIZE * 3);

        verticesLocal_ = verticesBuf_.Get<float>();
        maskLocal_ = maskBuf_.Get<float>();
        numValidLocal_ = numValidBuf_.Get<int32_t>();
        posLocal_ = posBuf_.Get<float>();
        outputLocal_ = outputBuf_.Get<int32_t>();
        argminLocal_ = argminBuf_.Get<float>();
        vecIdxLocal_ = vecIdxBuf_.Get<int32_t>();
        minValLocal_ = minValBuf_.Get<float>();
        sortIdxLocal_ = sortIdxBuf_.Get<int32_t>();
        sortIdxLocal1_ = sortIdxLocal_[singleLoopTaskCount_ * VERTICES_ALIGNED];
        sortIdxLocal2_ = sortIdxLocal_[singleLoopTaskCount_ * VERTICES_ALIGNED * 2];
    }

    __aicore__ inline void InitEvent()
    {
        eventMTE2V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        eventVMTE3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        eventMTE3V_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t taskCount);
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t taskCount);
    __aicore__ inline void Compute();
    __aicore__ inline void TransferAndMask();
    __aicore__ inline void ComputeArgmin();
    __aicore__ inline void ComputeRadian();
    __aicore__ inline void SortVertices();
    __aicore__ inline void SelectFrontNineIdx();

private:
    uint16_t repeatTimes_;
    TPipe* pipe_;
    int32_t eventMTE2V_, eventVMTE3_, eventMTE3V_;
    uint32_t blkIdx_;
    uint32_t coreTask_, taskOffset_, singleLoopTaskCount_;
    uint32_t broadCastSrcShape1_[2];
    uint32_t broadCastDstShape1_[2];
    uint32_t broadCastSrcShape2_[2];
    uint32_t broadCastDstShape2_[2];
    uint32_t mask_ = 0;
    uint64_t rsvdCnt_;

    GlobalTensor<float> verticesGm_;
    GlobalTensor<float> maskGm_;
    GlobalTensor<int32_t> numValidGm_;
    GlobalTensor<int32_t> sortedIdxGm_;

    TBuf<TPosition::VECCALC> verticesBuf_, maskBuf_, numValidBuf_, sortIdxBuf_,
        posBuf_, argminBuf_, vecIdxBuf_, minValBuf_, outputBuf_;
    LocalTensor<float> verticesLocal_, posLocal_, argminLocal_, maskLocal_, minValLocal_;
    LocalTensor<int32_t> numValidLocal_, vecIdxLocal_, sortIdxLocal_, sortIdxLocal1_, sortIdxLocal2_, outputLocal_;
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

__aicore__ inline void DiffIouRotatedSortVertices::TransferAndMask()
{
    // xyxy... --> xx...yy...
    uint8_t src1Pattern = 1;
    GatherMask(posLocal_, verticesLocal_, src1Pattern, false, mask_, { 1, repeatTimes_, 8, 0 }, rsvdCnt_);
    src1Pattern = 2;
    GatherMask(posLocal_[singleLoopTaskCount_ * VERTICES_ALIGNED], verticesLocal_, src1Pattern, false, mask_, { 1, repeatTimes_, 8, 0 }, rsvdCnt_);

    // vertices[~mask] = INF
    Muls(maskLocal_, maskLocal_, -POSITIVE_INF, singleLoopTaskCount_ * VERTICES_ALIGNED);
    PipeBarrier<PIPE_V>();
    Adds(maskLocal_, maskLocal_, static_cast<float>(POSITIVE_INF), singleLoopTaskCount_ * VERTICES_ALIGNED);
    PipeBarrier<PIPE_V>();
    Add(posLocal_, maskLocal_, posLocal_, singleLoopTaskCount_ * VERTICES_ALIGNED);
    Add(posLocal_[singleLoopTaskCount_ * VERTICES_ALIGNED], maskLocal_, posLocal_[singleLoopTaskCount_ * VERTICES_ALIGNED], singleLoopTaskCount_ * VERTICES_ALIGNED);
    PipeBarrier<PIPE_V>();
    // masked result store in posLocal_
}

__aicore__ inline void DiffIouRotatedSortVertices::ComputeArgmin()
{
    // Compte argmin
    WholeReduceMin<float>(argminLocal_, posLocal_[singleLoopTaskCount_ * VERTICES_ALIGNED], VERTICES_COUNT, singleLoopTaskCount_, 1, 1, 4, ReduceOrder::ORDER_ONLY_INDEX);
    
    // modify the corr, through the min y pos vertice
    PipeBarrier<PIPE_V>();
    Muls(argminLocal_.ReinterpretCast<int32_t>(), argminLocal_.ReinterpretCast<int32_t>(), static_cast<int32_t>(FLOAT_BYTE_SIZE), singleLoopTaskCount_);
    PipeBarrier<PIPE_V>();
    Add(argminLocal_.ReinterpretCast<int32_t>(), argminLocal_.ReinterpretCast<int32_t>(), vecIdxLocal_, singleLoopTaskCount_);
    Gather(minValLocal_, posLocal_, argminLocal_.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), singleLoopTaskCount_);
    Gather(minValLocal_[singleLoopTaskCount_], posLocal_[singleLoopTaskCount_ * VERTICES_ALIGNED],
        argminLocal_.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), singleLoopTaskCount_);
    PipeBarrier<PIPE_V>();
    BroadCast<float, 2, 1, false>(verticesLocal_, minValLocal_, broadCastDstShape1_, broadCastSrcShape1_);
    PipeBarrier<PIPE_V>();
    Sub(posLocal_, posLocal_, verticesLocal_, singleLoopTaskCount_ * VERTICES_ALIGNED);
    PipeBarrier<PIPE_V>();
    BroadCast<float, 2, 1, false>(verticesLocal_, minValLocal_[singleLoopTaskCount_], broadCastDstShape1_, broadCastSrcShape1_);
    PipeBarrier<PIPE_V>();
    Sub(posLocal_[singleLoopTaskCount_ * VERTICES_ALIGNED], posLocal_[singleLoopTaskCount_ * VERTICES_ALIGNED], verticesLocal_, singleLoopTaskCount_ * VERTICES_ALIGNED);
    PipeBarrier<PIPE_V>();
    // store the result in posLocal_
}

__aicore__ inline void DiffIouRotatedSortVertices::ComputeRadian()
{
    Adds(posLocal_, posLocal_, static_cast<float>(EPS), singleLoopTaskCount_ * VERTICES_ALIGNED);
    PipeBarrier<PIPE_V>();
    Div(verticesLocal_[singleLoopTaskCount_ * VERTICES_ALIGNED], posLocal_[singleLoopTaskCount_ * VERTICES_ALIGNED],
        posLocal_, singleLoopTaskCount_ * VERTICES_ALIGNED);
    Adds(posLocal_, posLocal_, static_cast<float>(EPS), singleLoopTaskCount_ * VERTICES_ALIGNED);
    Atan(verticesLocal_, verticesLocal_[singleLoopTaskCount_ * VERTICES_ALIGNED], singleLoopTaskCount_ * VERTICES_ALIGNED);
    Sign(posLocal_[singleLoopTaskCount_ * VERTICES_ALIGNED], posLocal_, singleLoopTaskCount_ * VERTICES_ALIGNED);
    PipeBarrier<PIPE_V>();
    Muls(posLocal_, posLocal_[singleLoopTaskCount_ * VERTICES_ALIGNED], static_cast<float>(-1), singleLoopTaskCount_ * VERTICES_ALIGNED);
    PipeBarrier<PIPE_V>();
    Relu(posLocal_, posLocal_, singleLoopTaskCount_ * VERTICES_ALIGNED);
    PipeBarrier<PIPE_V>();
    Muls(posLocal_, posLocal_, static_cast<float>(PI), singleLoopTaskCount_ * VERTICES_ALIGNED);
    PipeBarrier<PIPE_V>();
    Add(verticesLocal_, verticesLocal_, posLocal_, singleLoopTaskCount_ * VERTICES_ALIGNED);
    // store the result in verticesLocal_
}

__aicore__ inline void DiffIouRotatedSortVertices::SortVertices()
{
    // vertices_radian[~mask] = INF
    Add(verticesLocal_, maskLocal_, verticesLocal_, singleLoopTaskCount_ * VERTICES_ALIGNED);

    // argsort
    Duplicate(verticesLocal_[VERTICES_COUNT], POSITIVE_INF, 8, singleLoopTaskCount_, 1, 4);    // 24 - 32 padding pos fill inf
    PipeBarrier<PIPE_V>();
    Muls(verticesLocal_, verticesLocal_, static_cast<float>(-1), singleLoopTaskCount_ * VERTICES_ALIGNED); // decending
    PipeBarrier<PIPE_V>();
    Sort32(posLocal_, verticesLocal_, sortIdxLocal2_.ReinterpretCast<uint32_t>(), singleLoopTaskCount_);
    PipeBarrier<PIPE_V>();
    uint8_t src1Pattern = 2;
    GatherMask(posLocal_, posLocal_, src1Pattern, false, mask_, { 1, repeatTimes_, 8, 0 }, rsvdCnt_);
    // store the result in posLocal_
}

__aicore__ inline void DiffIouRotatedSortVertices::SelectFrontNineIdx()
{
    BroadCast<int32_t, 2, 1, false>(sortIdxLocal1_, numValidLocal_, broadCastDstShape1_, broadCastSrcShape1_);
    PipeBarrier<PIPE_V>();
    Sub(sortIdxLocal_, sortIdxLocal1_, sortIdxLocal2_, singleLoopTaskCount_ * VERTICES_ALIGNED);
    PipeBarrier<PIPE_V>();
    Cast(verticesLocal_, sortIdxLocal_, RoundMode::CAST_CEIL, singleLoopTaskCount_ * VERTICES_ALIGNED);
    PipeBarrier<PIPE_V>();
    Sign(verticesLocal_[singleLoopTaskCount_ * VERTICES_ALIGNED], verticesLocal_, singleLoopTaskCount_ * VERTICES_ALIGNED);
    PipeBarrier<PIPE_V>();
    Cast(sortIdxLocal_, verticesLocal_[singleLoopTaskCount_ * VERTICES_ALIGNED], RoundMode::CAST_CEIL, singleLoopTaskCount_ * VERTICES_ALIGNED);
    PipeBarrier<PIPE_V>();
    Relu(sortIdxLocal_, sortIdxLocal_, singleLoopTaskCount_ * VERTICES_ALIGNED);
    PipeBarrier<PIPE_V>();
    Mul(posLocal_.ReinterpretCast<int32_t>(), posLocal_.ReinterpretCast<int32_t>(), sortIdxLocal_, singleLoopTaskCount_ * VERTICES_ALIGNED);
    Muls(sortIdxLocal_, sortIdxLocal_, static_cast<int32_t>(-1), singleLoopTaskCount_ * VERTICES_ALIGNED);
    PipeBarrier<PIPE_V>();
    Sub(argminLocal_.ReinterpretCast<int32_t>(), argminLocal_.ReinterpretCast<int32_t>(), vecIdxLocal_, singleLoopTaskCount_);
    PipeBarrier<PIPE_V>();
    Muls(argminLocal_, argminLocal_, 0.25f, singleLoopTaskCount_);
    PipeBarrier<PIPE_V>();
    Adds(sortIdxLocal_, sortIdxLocal_, static_cast<int32_t>(1), singleLoopTaskCount_ * VERTICES_ALIGNED);
    BroadCast<int32_t, 2, 1, false>(sortIdxLocal1_, argminLocal_.ReinterpretCast<int32_t>(), broadCastDstShape1_, broadCastSrcShape1_);
    PipeBarrier<PIPE_V>();
    Mul(sortIdxLocal_, sortIdxLocal_, sortIdxLocal1_, singleLoopTaskCount_ * VERTICES_ALIGNED);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::MTE3_V>(eventMTE3V_);
    WaitFlag<HardEvent::MTE3_V>(eventMTE3V_);
    Add(outputLocal_, posLocal_.ReinterpretCast<int32_t>(), sortIdxLocal_, singleLoopTaskCount_ * VERTICES_ALIGNED);
    // store the result in outputLocal_
}

__aicore__ inline void DiffIouRotatedSortVertices::Compute()
{
    TransferAndMask();
    ComputeArgmin();
    ComputeRadian();
    SortVertices();
    SelectFrontNineIdx();
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