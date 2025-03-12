/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

template<typename DTYPE_F, typename DTYPE_I>
class KernelDeformableAggregation {
public:
    __aicore__ inline KernelDeformableAggregation() {}
    __aicore__ inline void Init(GM_ADDR mc_ms_feat, GM_ADDR spatial_shape, GM_ADDR scale_start_index,
        GM_ADDR sampling_location, GM_ADDR weights, GM_ADDR out, const DeformableAggregationTilingData* tiling_data, TPipe *tmpPipe)
    {
        pipe_ = tmpPipe;
        bs_ = tiling_data->bs;
        numFeats_ = tiling_data->numFeats;
        numEmbeds_ = tiling_data->numEmbeds;
        numAnchors_ = tiling_data->numAnchors;
        numPoints_ = tiling_data->numPoints;
        numCams_ = tiling_data->numCams;
        numScales_ = tiling_data->numScales;
        numGroups_ = tiling_data->numGroups;
        cAligned_ = tiling_data->cAligned;
        memoryFlag_ = tiling_data->memoryFlag;
        coreNum_ = tiling_data->coreNum;
        numChannels_ = numEmbeds_ / numGroups_;

        weightBufSize_ = memoryFlag_ ? numPoints_ * numCams_ * numScales_ * numGroups_ : numCams_ * numScales_ * numGroups_;
        weightBufSize_ = AlignUp(weightBufSize_, blockAlign_);
        locBufSize_ = AlignUp(numPoints_ * numCams_ * 2, blockAlign_);
        scaleStartBufSize_ = AlignUp(numCams_ * numScales_, blockAlign_);
        spatialShapeBufSize_ = AlignUp(numCams_ * numScales_ * 2, blockAlign_);

        copyOutParams_ = {1, static_cast<uint32_t>(numEmbeds_ * sizeof(DTYPE_F)), 0, 0, 0};
        srcShape_[0] = numGroups_;
        srcShape_[1] = 1;
        dstShape_[0] = numGroups_;
        dstShape_[1] = numChannels_;

        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        uint64_t mcMsFeatGmLength = bs_ * numFeats_ * numEmbeds_;
        uint64_t scaleStartIndexLength = numCams_ * numScales_;
        uint64_t spatialShapeGmLength = scaleStartIndexLength * 2;
        uint64_t samplingLocationGmLength = bs_ * numAnchors_ * numPoints_ * numCams_ * 2;
        uint64_t weightsGmLength = bs_ * numAnchors_ * numPoints_ * numCams_ * numScales_ * numGroups_;
        uint64_t outGmLength = bs_ * numAnchors_ * numEmbeds_;

        mcMsFeatGm_.SetGlobalBuffer((__gm__ DTYPE_F*)mc_ms_feat, mcMsFeatGmLength);
        samplingLocationGm_.SetGlobalBuffer((__gm__ DTYPE_F*)sampling_location, samplingLocationGmLength);
        weightsGm_.SetGlobalBuffer((__gm__ DTYPE_F*)weights, weightsGmLength);
        outGm_.SetGlobalBuffer((__gm__ DTYPE_F*)out, outGmLength);
        spatialShapesGm_.SetGlobalBuffer((__gm__ DTYPE_I*)spatial_shape, spatialShapeGmLength);
        scaleStartIndexGm_.SetGlobalBuffer((__gm__ DTYPE_I*)scale_start_index, scaleStartIndexLength);
    }

    __aicore__ inline void GetLocalTensor()
    {
        pipe_->InitBuffer(weightBuf_, weightBufSize_ * sizeof(DTYPE_F));
        pipe_->InitBuffer(locationBuf_, locBufSize_ * sizeof(DTYPE_F));
        pipe_->InitBuffer(scaleStartBuf_, scaleStartBufSize_ * sizeof(DTYPE_I));
        pipe_->InitBuffer(spatialShapeBuf_, spatialShapeBufSize_ * sizeof(DTYPE_I));
        pipe_->InitBuffer(vBuf_, 4 * cAligned_ * sizeof(DTYPE_F));
        pipe_->InitBuffer(weightMulBuf_, cAligned_ * sizeof(DTYPE_F));
        pipe_->InitBuffer(resBuf_, cAligned_ * sizeof(DTYPE_F));

        weightLocal_ = weightBuf_.Get<DTYPE_F>();
        locationLocal_ = locationBuf_.Get<DTYPE_F>();
        scaleStartLocal_ = scaleStartBuf_.Get<DTYPE_I>();
        spatialShapeLocal_ = spatialShapeBuf_.Get<DTYPE_I>();
        vLocal_ = vBuf_.Get<DTYPE_F>();
        weightMulLocal_ = weightMulBuf_.Get<DTYPE_F>();
        resLocal_ = resBuf_.Get<DTYPE_F>();
    }

    __aicore__ inline void Process()
    {
        taskNum_ = bs_ * numAnchors_;
        taskNumPerCore_ = DivCeil(taskNum_, coreNum_);
        curBlockIdx_ = GetBlockIdx();
        startOffset_ = curBlockIdx_ * taskNumPerCore_;
        endOffset_ = (curBlockIdx_ + 1) * taskNumPerCore_;
        if (endOffset_ > taskNum_) {
            endOffset_ = taskNum_;
        }

        DataCopy(scaleStartLocal_, scaleStartIndexGm_, scaleStartBufSize_);
        DataCopy(spatialShapeLocal_, spatialShapesGm_, spatialShapeBufSize_);
        for (uint32_t taskIdx = startOffset_; taskIdx < endOffset_; ++taskIdx) {
            ComputeAndCopyOut(taskIdx);
        }
    }

    __aicore__ inline void ComputeAndCopyOut(int32_t taskIdx)
    {
        uint32_t batchIdx = taskIdx / numAnchors_;
        uint32_t anchorIdx = taskIdx % numAnchors_;
        uint64_t refOffsetGm = (batchIdx * numAnchors_ + anchorIdx) * numEmbeds_;
        uint64_t locationOffsetGm = (batchIdx * numAnchors_ +
                                     anchorIdx) * numPoints_ * numCams_ * 2;
        if (memoryFlag_) {
            uint64_t weightOffsetGm = (batchIdx * numAnchors_ +
                                       anchorIdx) * numPoints_ * numCams_ * numScales_ * numGroups_;
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
            DataCopy(weightLocal_, weightsGm_[weightOffsetGm], weightBufSize_);
        }
        DataCopy(locationLocal_, samplingLocationGm_[locationOffsetGm], locBufSize_);
        Duplicate(resLocal_, 0.0f, cAligned_);
        for (uint32_t pointIdx = 0; pointIdx < numPoints_; ++pointIdx) {
            if (!memoryFlag_) {
                uint64_t weightOffsetGm = (batchIdx * numAnchors_ * numPoints_ +
                                           anchorIdx * numPoints_ + pointIdx) * numCams_ * numScales_ * numGroups_;
                SetFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(0);
                DataCopy(weightLocal_, weightsGm_[weightOffsetGm], weightBufSize_);
            }
            uint32_t weightBaseOffsetLocal = memoryFlag_ ? pointIdx * numCams_ * numScales_ * numGroups_ : 0;
            for (uint32_t camIdx = 0; camIdx < numCams_; ++camIdx) {
                uint32_t locationOffsetLocal = (pointIdx * numCams_ + camIdx) * 2;
                DTYPE_F locW = locationLocal_.GetValue(locationOffsetLocal);
                if (locW <= 0 || locW >= 1) {
                    continue;
                }
                DTYPE_F locH = locationLocal_.GetValue(locationOffsetLocal + 1);
                if (locH <= 0 || locH >= 1) {
                    continue;
                }
                for (uint32_t scaleIdx = 0; scaleIdx < numScales_; ++scaleIdx) {
                    uint32_t weightOffsetLocal = weightBaseOffsetLocal + (camIdx * numScales_ + scaleIdx) * numGroups_;
                    uint32_t scaleStartOffset = camIdx * numScales_ + scaleIdx;
                    uint32_t spatialShapeOffset = scaleStartOffset * 2;
                    uint32_t scaleStartIdx = scaleStartLocal_.GetValue(scaleStartOffset);
                    uint32_t valueOffset = (batchIdx * numFeats_ + scaleStartIdx) * numEmbeds_;

                    DTYPE_I h = spatialShapeLocal_.GetValue(spatialShapeOffset);
                    DTYPE_I w = spatialShapeLocal_.GetValue(spatialShapeOffset + 1);

                    DTYPE_F hIm = locH * h - 0.5f;
                    DTYPE_F wIm = locW * w - 0.5f;

                    DTYPE_I hLow = ScalarCast<DTYPE_F, DTYPE_I, RoundMode::CAST_FLOOR>(hIm);
                    DTYPE_I wLow = ScalarCast<DTYPE_F, DTYPE_I, RoundMode::CAST_FLOOR>(wIm);
                    DTYPE_I hHigh = hLow + 1;
                    DTYPE_I wHigh = wLow + 1;

                    DTYPE_F lh = hIm - hLow;
                    DTYPE_F lw = wIm - wLow;
                    DTYPE_F hh = 1 - lh;
                    DTYPE_F hw = 1 - lw;

                    DTYPE_I wStride = numEmbeds_;
                    DTYPE_I hStride = w * wStride;
                    DTYPE_I hLowPtrOffset = hLow * hStride;
                    DTYPE_I hHighPtrOffset = hLowPtrOffset + hStride;
                    DTYPE_I wLowPtrOffset = wLow * wStride;
                    DTYPE_I wHighPtrOffset = wLowPtrOffset + wStride;

                    DTYPE_F w1 = hh * hw;
                    DTYPE_F w2 = hh * lw;
                    DTYPE_F w3 = lh * hw;
                    DTYPE_F w4 = lh * lw;

                    Duplicate(vLocal_, 0.0f, 4 * cAligned_);

                    SetFlag<HardEvent::V_MTE2>(0);
                    WaitFlag<HardEvent::V_MTE2>(0);

                    if (hLow >= 0) {
                        basePtr_ = valueOffset + hLowPtrOffset;
                        if (wLow >= 0) {
                            realPtr_ = basePtr_ + wLowPtrOffset;
                            DataCopy(vLocal_[v1Offset_ * cAligned_], mcMsFeatGm_[realPtr_], cAligned_);
                        }
                        if (wHigh <= w - 1) {
                            realPtr_ = basePtr_ + wHighPtrOffset;
                            DataCopy(vLocal_[v2Offset_ * cAligned_], mcMsFeatGm_[realPtr_], cAligned_);
                        }
                    }

                    if (hHigh <= h - 1) {
                        basePtr_ = valueOffset + hHighPtrOffset;
                        if (wLow >= 0) {
                            realPtr_ = basePtr_ + wLowPtrOffset;
                            DataCopy(vLocal_[v3Offset_ * cAligned_], mcMsFeatGm_[realPtr_], cAligned_);
                        }
                        if (wHigh <= w - 1) {
                            realPtr_ = basePtr_ + wHighPtrOffset;
                            DataCopy(vLocal_[v4Offset_ * cAligned_], mcMsFeatGm_[realPtr_], cAligned_);
                        }
                    }

                    SetFlag<HardEvent::MTE2_V>(0);
                    WaitFlag<HardEvent::MTE2_V>(0);
                    Muls(vLocal_[v1Offset_ * cAligned_], vLocal_[v1Offset_ * cAligned_], w1, cAligned_);
                    Axpy(vLocal_[v1Offset_ * cAligned_], vLocal_[v2Offset_ * cAligned_], w2, cAligned_);
                    Axpy(vLocal_[v1Offset_ * cAligned_], vLocal_[v3Offset_ * cAligned_], w3, cAligned_);
                    Axpy(vLocal_[v1Offset_ * cAligned_], vLocal_[v4Offset_ * cAligned_], w4, cAligned_);
                    
                    BroadCast<DTYPE_F, 2, 1>(weightMulLocal_, weightLocal_[weightOffsetLocal], dstShape_, srcShape_);
                    MulAddDst(resLocal_, vLocal_[v1Offset_ * cAligned_], weightMulLocal_, cAligned_);
                }
            }
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(outGm_[refOffsetGm], resLocal_, copyOutParams_);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }

private:
    TPipe *pipe_;

    TBuf<TPosition::VECCALC> weightBuf_, locationBuf_, scaleStartBuf_, spatialShapeBuf_;
    TBuf<TPosition::VECCALC> vBuf_, weightMulBuf_, resBuf_;

    GlobalTensor<DTYPE_F> mcMsFeatGm_, samplingLocationGm_, weightsGm_, outGm_;
    GlobalTensor<DTYPE_I> spatialShapesGm_, scaleStartIndexGm_;

    LocalTensor<DTYPE_F> locationLocal_, weightLocal_;
    LocalTensor<DTYPE_I> spatialShapeLocal_, scaleStartLocal_;
    LocalTensor<DTYPE_F> vLocal_, weightMulLocal_, resLocal_;

    bool memoryFlag_;
    uint32_t basePtr_, realPtr_;
    uint32_t coreNum_, curBlockIdx_;
    uint32_t taskNum_, taskNumPerCore_, startOffset_, endOffset_;
    uint32_t weightBufSize_, locBufSize_, scaleStartBufSize_, spatialShapeBufSize_;
    uint32_t bs_, numFeats_, numEmbeds_, numAnchors_, numPoints_, numCams_, numScales_, numGroups_, numChannels_, cAligned_;
    uint32_t blockAlign_ = 8;
    uint32_t v1Offset_ = 0, v2Offset_ = 1, v3Offset_ = 2, v4Offset_ = 3;
    
    uint32_t srcShape_[2], dstShape_[2];
    DataCopyExtParams copyOutParams_;
};

extern "C" __global__ __aicore__ void deformable_aggregation(GM_ADDR mc_ms_feat, GM_ADDR spatial_shape,
    GM_ADDR scale_start_index, GM_ADDR sampling_location, GM_ADDR weights, GM_ADDR out, GM_ADDR workspace,
    GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    KernelDeformableAggregation<float, int32_t> op;
    op.Init(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights, out, &tiling_data, &pipe);
    op.GetLocalTensor();
    op.Process();
}
