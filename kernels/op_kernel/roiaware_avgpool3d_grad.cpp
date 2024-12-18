#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BUFFER_ALIGN_BYTE = 32;

template<typename dataType>
class RoiAwareAvgpool3dGrad {
public:
    __aicore__ inline RoiAwareAvgpool3dGrad() {}

    __aicore__ inline void Init(TPipe* pipe, const GM_ADDR pts_idx_of_voxels, const GM_ADDR grad_out, const GM_ADDR grad_in,
        const RoiawareAvgpool3dGradTilingData* tiling)
    {
        pipe_ = pipe;
        blkIdx_ = GetBlockIdx();
        InitTiling(tiling);
        InitGM(pts_idx_of_voxels, grad_out, grad_in);
        InitUB();
    }

     __aicore__ inline void Process()
    {
        uint32_t localTaskIdx = 0;
        for (; localTaskIdx < singleCoreTask_; localTaskIdx += singleLoopTask_) {
            uint32_t curLoopTaskCount = min(singleCoreTask_ - localTaskIdx, singleLoopTask_);
            uint32_t curGlobalTaskIdx = localTaskIdx + globalTaskIdx_;

            CopyInNTask(curLoopTaskCount, curGlobalTaskIdx);
            ComputeNTask(curLoopTaskCount, curGlobalTaskIdx);
        }
    }

protected:
    uint64_t blkIdx_;
    TPipe* pipe_;
    GlobalTensor<dataType> goutGm_, ginGm_;
    GlobalTensor<int32_t> ptsIdxOfsVoxelsGm_;
    TQue<QuePosition::VECIN, BUFFER_NUM> goutQue_, ptsIdxQue_;
    TBuf<TPosition::VECCALC> ginBuf_, iterIdxBuf_;
    LocalTensor<int32_t> iterIdxLocal_;

    // tiling
    uint32_t singleCoreTask_, singleLoopTask_, outputLoopTask_;
    uint32_t channels_, channelAligned_, maxPtsPerVoxel_, maxPtsPerVoxelAligned_;
    uint32_t globalTaskIdx_;
    uint32_t totalTask_, npoints_;

    // copy params
    DataCopyExtParams gradOutCopyInParams, ptsIdxCopyInParams, ginCopyOutParams;
    DataCopyPadExtParams<dataType> gradOutCopyInPadParams{false, 0, 0, 0};
    DataCopyPadExtParams<int32_t> ptsIdxCopyInPadParams{false, 0, 0, 0};

private:
    __aicore__ inline void InitTiling(const RoiawareAvgpool3dGradTilingData* tiling)
    {
        singleCoreTask_ = tiling->singleCoreTask;
        singleLoopTask_ = tiling->singleLoopTask;
        outputLoopTask_ = tiling->outputLoopTask;
        channels_ = tiling->channels;
        channelAligned_ = tiling->channelAligned;
        maxPtsPerVoxel_ = tiling->maxPtsPerVoxel;
        maxPtsPerVoxelAligned_ = tiling->maxPtsPerVoxelAligned;
        totalTask_ = tiling->totalTask;
        npoints_ = tiling->npoints;

        if (blkIdx_ < tiling->bigCoreCount) {
            // big core
            globalTaskIdx_ = singleCoreTask_ * blkIdx_;
        } else {
            // small core
            globalTaskIdx_ = (singleCoreTask_ * tiling->bigCoreCount) +
                (singleCoreTask_ - 1) * (blkIdx_ - tiling->bigCoreCount);
            singleCoreTask_ = singleCoreTask_ - 1;
        }
    }

    __aicore__ inline void InitGM(const GM_ADDR pts_idx_of_voxels, const GM_ADDR grad_out, const GM_ADDR grad_in)
    {
        ptsIdxOfsVoxelsGm_.SetGlobalBuffer((__gm__ int32_t*) pts_idx_of_voxels,
            static_cast<uint64_t>(totalTask_) * maxPtsPerVoxelAligned_ * sizeof(int32_t));
        goutGm_.SetGlobalBuffer((__gm__ dataType*) grad_out,
            static_cast<uint64_t>(totalTask_) * channelAligned_ * sizeof(dataType));
        ginGm_.SetGlobalBuffer((__gm__ dataType*) grad_in,
            static_cast<uint64_t>(npoints_) * channelAligned_ * sizeof(dataType));
    }

    __aicore__ inline void InitUB()
    {
        pipe_->InitBuffer(goutQue_, BUFFER_NUM, singleLoopTask_ * channelAligned_ * sizeof(dataType));
        pipe_->InitBuffer(ptsIdxQue_, BUFFER_NUM, singleLoopTask_ * maxPtsPerVoxelAligned_ * sizeof(int32_t));
        pipe_->InitBuffer(ginBuf_, outputLoopTask_ * channelAligned_ * sizeof(dataType));
        pipe_->InitBuffer(iterIdxBuf_, min(singleLoopTask_, totalTask_) * sizeof(int32_t));
        iterIdxLocal_ = iterIdxBuf_.AllocTensor<int32_t>();
    }

    __aicore__ inline void CopyInNTask(const uint32_t &taskCount, const uint64_t &globalTaskIdx)
    {
        LocalTensor<dataType> goutLocal = goutQue_.AllocTensor<dataType>();
        LocalTensor<int32_t> ptsIdxLocal = ptsIdxQue_.AllocTensor<int32_t>();

        uint64_t globalGradOutIdx = globalTaskIdx * channels_;
        uint64_t globalPtsIdxOutIdx = globalTaskIdx * (maxPtsPerVoxel_ + 1);

        gradOutCopyInParams = {static_cast<uint16_t>(taskCount), channels_ * sizeof(dataType), 0, 0, 0};
        DataCopyPad(goutLocal, goutGm_[globalGradOutIdx], gradOutCopyInParams, gradOutCopyInPadParams);

        ptsIdxCopyInParams = {static_cast<uint16_t>(taskCount), maxPtsPerVoxel_ * sizeof(int32_t),
            static_cast<uint32_t>(1 * sizeof(int32_t)), 0, 0};
        DataCopyPad(ptsIdxLocal, ptsIdxOfsVoxelsGm_[globalPtsIdxOutIdx + 1], ptsIdxCopyInParams, ptsIdxCopyInPadParams);

        goutQue_.EnQue<dataType>(goutLocal);
        ptsIdxQue_.EnQue<int32_t>(ptsIdxLocal);
    }

    __aicore__ inline void ComputeNTask(const uint32_t &taskCount, const uint64_t &globalTaskIdx)
    {
        LocalTensor<dataType> goutLocal = goutQue_.DeQue<dataType>();
        LocalTensor<int32_t> ptsIdxLocal = ptsIdxQue_.DeQue<int32_t>();
        LocalTensor<dataType> ginLocal = ginBuf_.AllocTensor<dataType>();

        Duplicate<int32_t>(iterIdxLocal_, static_cast<int32_t>(0), min(totalTask_, singleLoopTask_));

        int32_t gradInChannelsIdx = 0;
        for (; gradInChannelsIdx < npoints_; gradInChannelsIdx += outputLoopTask_) {
            int32_t eventID1 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventID1);
            WaitFlag<HardEvent::MTE3_V>(eventID1);
            
            Duplicate<dataType>(ginLocal, static_cast<dataType>(0), outputLoopTask_ * channelAligned_);
            uint32_t curLoopOutputTaskCount = min(outputLoopTask_, npoints_ - gradInChannelsIdx);
            
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < taskCount; i++) {
                int32_t curPtsIdxCount = ptsIdxOfsVoxelsGm_.GetValue((i + globalTaskIdx) * (maxPtsPerVoxel_ + 1));
                if (gradInChannelsIdx == 0) {
                    Muls(goutLocal[i * channelAligned_], goutLocal[i * channelAligned_], static_cast<dataType>(1.0f / max(curPtsIdxCount, 1)), channelAligned_);
                }
                int32_t curPtsIdx = iterIdxLocal_.GetValue(i);
                ComputeOneTask(i, curPtsIdxCount, gradInChannelsIdx, curPtsIdx, curLoopOutputTaskCount, ptsIdxLocal, goutLocal, ginLocal);
                iterIdxLocal_.SetValue(i, curPtsIdx);
            }

            int32_t eventID2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventID2);
            WaitFlag<HardEvent::V_MTE3>(eventID2);

            ginCopyOutParams = {static_cast<uint16_t>(curLoopOutputTaskCount), static_cast<uint32_t>(channels_ * sizeof(dataType)), 0, 0, 0};
            SetAtomicAdd<dataType>();
            DataCopyPad(ginGm_[static_cast<uint64_t>(gradInChannelsIdx) * channels_], ginLocal, ginCopyOutParams);
            SetAtomicNone();
        }
        
        goutQue_.FreeTensor(goutLocal);
        ptsIdxQue_.FreeTensor(ptsIdxLocal);
    }

    __aicore__ inline void ComputeOneTask(const int32_t &goutLocalTaskIdx, const int32_t &curPtsIdxCount, const int32_t &gradInChannelsIdx,
        int32_t &curPtsIdx, const uint32_t &curLoopOutputTaskCount, const LocalTensor<int32_t> &ptsIdxLocal,
        const LocalTensor<dataType> &goutLocal, LocalTensor<dataType> &ginLocal)
    {
        uint32_t offset = goutLocalTaskIdx * maxPtsPerVoxelAligned_;
        while (curPtsIdx < curPtsIdxCount) {
            uint32_t outputIdx = ptsIdxLocal.GetValue(offset + curPtsIdx);
            if (outputIdx >= gradInChannelsIdx + curLoopOutputTaskCount) {
                break;
            }
            outputIdx -= gradInChannelsIdx;
            Add(ginLocal[outputIdx * channelAligned_], goutLocal[goutLocalTaskIdx * channelAligned_],
                ginLocal[outputIdx * channelAligned_], channelAligned_);
            curPtsIdx++;
        }
    }
};

extern "C" __global__ __aicore__ void roiaware_avgpool3d_grad(GM_ADDR pts_idx_of_voxels, GM_ADDR grad_out, GM_ADDR grad_in,
                                                               GM_ADDR workspace, GM_ADDR tiling_data) {
    GET_TILING_DATA(tiling, tiling_data);
    SetSysWorkspace(workspace);

    RoiAwareAvgpool3dGrad<float> op;
    TPipe pipe;
    op.Init(&pipe, pts_idx_of_voxels, grad_out, grad_in, &tiling);
    op.Process();
}