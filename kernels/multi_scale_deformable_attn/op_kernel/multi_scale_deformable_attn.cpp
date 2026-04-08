/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 */

#include "lib/matmul_intf.h"
#include "kernel_utils.h"
#include "msda.h"

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnKernel<aligned, fastMode>::UpdateParams(uint32_t tailCompNum)
{
    this->compTaskNum_ = tailCompNum;
    if constexpr (fastMode) {
        this->outerLoops_ = this->compTaskNum_;
    } else {
        this->outerLoops_ = this->compTaskNum_ * this->numHeads_ * DivCeil(this->oneHeadNum_, this->innerLoopsAligned_);
    }
    this->cpOutParams_.blockCount = this->compTaskNum_ * this->numHeads_;
    if (fastMode) {
        this->cpSampleParams_.blockCount = this->compTaskNum_;
        this->cpDoubleSampleParams_.blockCount = this->compTaskNum_;
    } else {
        this->cpSampleParams_.blockCount = this->compTaskNum_ * this->numHeads_;
        this->cpDoubleSampleParams_.blockCount = this->compTaskNum_ * this->numHeads_;
    }
}

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnKernel<aligned, fastMode>::CopyFullPoint(
    const LocalTensor<int32_t>& location, const LocalTensor<float>& value,
    uint64_t valid, uint32_t baseIdx, uint32_t innerLoops)
{
    for (int32_t i = ScalarGetSFFValue<1>(valid); i < innerLoops && i >= 0;
        i = ScalarGetSFFValue<1>(valid)) {
        valid = sbitset0(valid, i);
        uint32_t idx = baseIdx + i;
        // WARN: dangerous!
        int32_t gmY0Offset = location.GetValue(idx);
        int32_t gmY1Offset = location.GetValue(idx + this->alignedOneTaskNum_);
        this->CopyInValue(value[i * this->alignedEmbedDims_], this->valueGm_[gmY0Offset], this->cpRowDoubleParams_);
        this->CopyInValue(value[i * this->alignedEmbedDims_ + 2 * this->alignedCornerEmbedDims_],
            this->valueGm_[gmY1Offset], this->cpRowDoubleParams_);
    }
}

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnKernel<aligned, fastMode>::CopyBorderPoint(
    const LocalTensor<int32_t>& location, const LocalTensor<float>& value,
    const LocalTensor<int32_t>& shapeInt, const LocalTensor<int32_t>& loc,
    uint64_t valid, uint32_t baseIdx, uint32_t innerLoops)
{
    for (int32_t i = ScalarGetSFFValue<0>(valid); i < innerLoops && i >= 0;
        i = ScalarGetSFFValue<0>(valid)) {
        valid = sbitset1(valid, i);
        uint32_t idx = baseIdx + i;
        int32_t w = shapeInt.GetValue(idx);
        int32_t x = loc.GetValue(idx);
        // WARN: dangerous!
        int32_t gmOffset = location.GetValue(idx);
        if (x != -1) {
            this->CopyInValue(value[i * this->alignedEmbedDims_], this->valueGm_[gmOffset], this->cpOneValParams_);
        }
        if (x != w - 1) {
            this->CopyInValue(value[i * this->alignedEmbedDims_ + this->alignedCornerEmbedDims_],
                this->valueGm_[gmOffset + this->outDims_], this->cpOneValParams_);
        }
    }
}

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnKernel<aligned, fastMode>::CumsumOutput(
    const LocalTensor<float>& output, const LocalTensor<float>& cornerWeightBrc,
    uint32_t outOffset, uint32_t innerLoops)
{
    uint32_t embedOffset = 0;
    if (fastMode) {
        for (uint32_t embedIdx = 0; embedIdx < this->embedLoops_; ++embedIdx) {
            uint32_t embedOffset = embedIdx * B32_DATA_NUM_PER_REPEAT;
            this->SetVectorMask4MSDA(embedIdx);
            for (uint32_t i = 0; i < this->numHeads_; ++i) {
                uint32_t innerOffset = embedOffset + outOffset + i * this->alignedEmbedDims_;
                Add<float, false>(output[innerOffset], cornerWeightBrc[embedOffset + i * this->oneHeadNum_ * this->alignedEmbedDims_],
                    output[innerOffset], MASK_PLACEHOLDER, this->oneHeadNum_, {1, 1, 1, 0, static_cast<uint8_t>(this->embedBlk_), 0});
            }
        }
    } else {
        for (uint32_t embedIdx = 0; embedIdx < this->embedLoops_; ++embedIdx) {
            uint32_t embedOffset = embedIdx * B32_DATA_NUM_PER_REPEAT;
            uint32_t innerOffset = embedOffset + outOffset;
            this->SetVectorMask4MSDA(embedIdx);
            Add<float, false>(output[innerOffset], cornerWeightBrc[embedOffset], output[innerOffset], MASK_PLACEHOLDER,
                innerLoops, {1, 1, 1, 0, static_cast<uint8_t>(this->embedBlk_), 0});
        }
    }
    ResetMask();
}

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnKernel<aligned, fastMode>::BroadEmbedBlk(
    const LocalTensor<float>& cornerWeightBrc, uint32_t innerLoops)
{
    if (this->embedBlk_ >= 32) {
        for (uint32_t i = 1; i < this->embedBlk_; ++i) {
            DataCopy<float>(cornerWeightBrc[i * B32_DATA_NUM_PER_BLOCK], cornerWeightBrc,
                {static_cast<uint16_t>(4 * this->innerLoops_), 1, static_cast<uint16_t>(this->embedBlk_ - 1), static_cast<uint16_t>(this->embedBlk_ - 1)});
        }
    } else {
        for (uint32_t i = 1; i < this->embedBlk_; ++i) {
            Adds<float, false>(cornerWeightBrc[i * B32_DATA_NUM_PER_BLOCK], cornerWeightBrc, 0.f, MASK_PLACEHOLDER, this->brcRpt_,
                {this->embedBlk_, this->embedBlk_, static_cast<uint8_t>(8 * this->embedBlk_), static_cast<uint8_t>(8 * this->embedBlk_)});
        }
    }
}

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnKernel<aligned, fastMode>::ComputeBilinearInterpolation(
    const LocalTensor<uint64_t>& validFlag, const LocalTensor<int32_t>& shapeInt, const LocalTensor<int32_t>& location,
    const LocalTensor<int32_t>& loc, const LocalTensor<float>& shapeFloat, const LocalTensor<float>& production,
    const LocalTensor<float>& value, const LocalTensor<float>& locFloat, const LocalTensor<float>& weight,
    const LocalTensor<float>& attentionWeight, const LocalTensor<float>& cornerWeightBrc, const LocalTensor<float>& output, uint32_t round)
{
    uint64_t bottomOffset = 2 * this->validFlagMaskLen_ / 8;
    uint64_t topOffset = 3 * this->validFlagMaskLen_ / 8;
    WaitFlag<HardEvent::V_MTE2>(this->biEvt_);
    WaitFlag<HardEvent::V_MTE3>(this->biEvt_);
    uint32_t divPoint = this->outerLoops_ / 5 * 4;
    for (uint32_t head = 0; head < this->outerLoops_; ++head) {
        uint32_t bufferFlag = head % 2;
        uint32_t baseIdx = head * this->innerLoopsAligned_;
        uint64_t headOffset = baseIdx / B32_DATA_NUM_PER_REPEAT;
        uint64_t byteOffset = baseIdx - headOffset * B32_DATA_NUM_PER_REPEAT;
        uint64_t valid = validFlag.GetValue(headOffset) >> byteOffset;
        uint64_t bottomValid = validFlag.GetValue(headOffset + bottomOffset) >> byteOffset;
        uint64_t topValid = validFlag.GetValue(headOffset + topOffset) >> byteOffset;
        uint32_t outOffset = head / this->innerTotalGroup_ * this->innerEmbedDims_;
        uint32_t innerLoops = min(this->innerLoops_, this->innerTotal_ - (head % this->innerTotalGroup_) * this->innerLoops_);
        LocalTensor<float> valueSrc = value[bufferFlag * this->cornerRpt_ * B32_DATA_NUM_PER_REPEAT];

        WaitFlag<HardEvent::V_MTE2>(bufferFlag);

        if(fastMode || !aligned || round>=this->coopRound || head < divPoint) {
            CopyFullPoint(location, valueSrc, valid, baseIdx, innerLoops);
        } else {
            if(head == divPoint) {
                AscendC::CrossCoreWaitFlag(2);
            }
            DataCopy(
                valueSrc, 
                this->assembleGm_[(this->blkIdx_* this->outerLoops_ + head) * this->cornerRpt_ * B32_DATA_NUM_PER_REPEAT], 
                this->cornerRpt_ * B32_DATA_NUM_PER_REPEAT
            );
            // 必须加，否则报错
            PipeBarrier<PIPE_MTE2>();
        } 
        if (head == 0) {
            this->ComputeWeight(locFloat, shapeFloat, production, weight, attentionWeight);
        }
        for (uint32_t i = 0; i < 4; ++i) {
            Brcb(cornerWeightBrc[i * this->alignedCornerEmbedDims_], weight[baseIdx + i * this->alignedOneTaskNum_],
                DivCeil(innerLoops, B32_DATA_NUM_PER_BLOCK), {this->embedBlk_, static_cast<uint16_t>(8 * this->embedBlk_)});
        }
        CopyBorderPoint(location, valueSrc, shapeInt, loc, bottomValid, baseIdx, innerLoops);
        CopyBorderPoint(location[this->alignedOneTaskNum_], valueSrc[2 * this->alignedCornerEmbedDims_],
            shapeInt, loc, topValid, baseIdx, innerLoops);
        SetFlag<HardEvent::MTE2_V>(bufferFlag);
        BroadEmbedBlk(cornerWeightBrc, innerLoops);
        WaitFlag<HardEvent::MTE2_V>(bufferFlag);
        Mul<float, false>(cornerWeightBrc, valueSrc, cornerWeightBrc, MASK_PLACEHOLDER, this->cornerRpt_, {1, 1, 1, 8, 8, 8});
        Duplicate<float, false>(valueSrc, 0.f, MASK_PLACEHOLDER, this->cornerRpt_, 1, 8);
        SetFlag<HardEvent::V_MTE2>(bufferFlag);

        Add<float>(cornerWeightBrc, cornerWeightBrc[2 * this->alignedCornerEmbedDims_], cornerWeightBrc,
            2 * this->alignedCornerEmbedDims_);
        Add<float>(cornerWeightBrc, cornerWeightBrc[this->alignedCornerEmbedDims_], cornerWeightBrc,
            this->alignedCornerEmbedDims_);
        if (unlikely(head == 0)) {
            WaitFlag<HardEvent::MTE3_V>(0);
            Duplicate<float>(output, 0.f, this->compTaskNum_ * this->numHeads_ * this->alignedEmbedDims_);
            if(!fastMode && aligned && round<this->coopRound) {
                DataCopy(this->validFlagSwapGm_[this->blkIdx_ * this->validFlagMaskLen_], validFlag, this->validFlagMaskLen_); // 256个uint64
                DataCopy(this->locationSwapGm_[this->blkIdx_ * 4*this->alignedOneTaskNum_], location, 4 * this->alignedOneTaskNum_); // 4096个int32
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(1);
            }
        }
        CumsumOutput(output, cornerWeightBrc, outOffset, innerLoops);
    }
    SetFlag<HardEvent::V_MTE3>(0);
}

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnKernel<aligned, fastMode>::Process()
{
    LocalTensor<float> locationFloat = this->locationQue_.template Get<float>();
    LocalTensor<int32_t> locationInt = this->locationQue_.template Get<int32_t>();
    LocalTensor<float> attentionWeight = this->attentionWeightsQue_.template Get<float>();
    LocalTensor<int32_t> shapes = this->shapeQue_.template Get<int32_t>();
    LocalTensor<int32_t> offset = this->offsetQue_.template Get<int32_t>();
    LocalTensor<float> shapeFloat = this->shapeFloatBuf_.template Get<float>();
    LocalTensor<int32_t> shapeInt = this->shapeIntBuf_.template Get<int32_t>();
    LocalTensor<int32_t> offsetInt = this->offsetIntBuf_.template Get<int32_t>();
    LocalTensor<float> value = this->valueQue_.template Get<float>();
    LocalTensor<float> cornerWeightBrc = this->cornerWeightBrcBuf_.template Get<float>();
    LocalTensor<float> output = this->outputQue_.template Get<float>();
    LocalTensor<uint64_t> validFlag = this->validFlagBuf_.template Get<uint64_t>();

    LocalTensor<int32_t> locInt = this->locIntBuf_.template Get<int32_t>();
    LocalTensor<float> locFloat = this->locFloatBuf_.template Get<float>();
    LocalTensor<float> production = this->productionBuf_.template Get<float>();
    LocalTensor<float> weight = this->weightBuf_.template Get<float>();
    uint32_t round = 0;

    this->PrepareShape(shapes, shapeInt, shapeFloat, offset, offsetInt);
    Duplicate<float, false>(value, 0.f, MASK_PLACEHOLDER, this->cornerRpt_, 1, 8);
    Duplicate<float, false>(value[this->cornerRpt_ * B32_DATA_NUM_PER_REPEAT], 0.f, MASK_PLACEHOLDER, this->cornerRpt_, 1, 8);

    if (!fastMode && aligned && this->blkIdx_ == 0) {
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(this->zeroGm_, value, this->cornerRpt_ * B32_DATA_NUM_PER_REPEAT);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll();

    SetFlag<HardEvent::V_MTE2>(this->copyEvt_);
    SetFlag<HardEvent::V_MTE2>(0);
    SetFlag<HardEvent::V_MTE2>(1);
    SetFlag<HardEvent::MTE3_V>(0);

    for (uint32_t taskLoopId = 0; taskLoopId < this->taskLoops_; taskLoopId++) {
        uint32_t taskIdx = this->compTaskNum_ * (taskLoopId * this->coreNum_ + this->blkIdx_);
        if (unlikely(taskLoopId == (this->taskLoops_ - 1) && (this->batchSize_ * this->numQueries_ - this->tailStart_) > 0)) {
            if (unlikely(this->blockTailTask_ == 0)) {
                break;
            }
            taskIdx = this->blockTailStart_;
            UpdateParams(this->blockTailTask_);
        }
        this->CopyInSample(locationFloat[2 * this->alignedOneTaskNum_], attentionWeight, taskIdx);
        this->ComputeLocation(taskIdx, locationFloat, attentionWeight, locationInt, shapeFloat, shapeInt, locFloat, locInt, offsetInt,
            validFlag.ReinterpretCast<uint8_t>());
        ComputeBilinearInterpolation(validFlag, shapeInt, locationInt, locInt, shapeFloat, production, value, locFloat, weight, attentionWeight, cornerWeightBrc, output, round);
        round++;
        CopyOut(output, taskIdx);
    }
    WaitFlag<HardEvent::V_MTE2>(this->copyEvt_);
    WaitFlag<HardEvent::V_MTE2>(0);
    WaitFlag<HardEvent::V_MTE2>(1);
    WaitFlag<HardEvent::MTE3_V>(0);
}

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnCubeKernel<aligned, fastMode>::UpdateParams(uint32_t tailCompNum)
{
    this->compTaskNum_ = tailCompNum;
    if constexpr (fastMode) {
        this->outerLoops_ = this->compTaskNum_;
    } else {
        this->outerLoops_ = this->compTaskNum_ * this->numHeads_;
    }
    if (fastMode) {
        this->cpSampleParams_.blockCount = this->compTaskNum_;
        this->cpDoubleSampleParams_.blockCount = this->compTaskNum_;
    } else {
        this->cpSampleParams_.blockCount = this->compTaskNum_ * this->numHeads_;
        this->cpDoubleSampleParams_.blockCount = this->compTaskNum_ * this->numHeads_;
    }
}

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnCubeKernel<aligned, fastMode>::CopyFullPointToCube(
    uint64_t valid, 
    uint32_t baseIdx, uint32_t innerLoops, uint32_t vecCoreIdx, uint32_t bufferFlag, const LocalTensor<float>& value)
{
    for (int32_t i = ScalarGetSFFValue<1>(valid); i < innerLoops && i >= 0;
        i = ScalarGetSFFValue<1>(valid)) {
        valid = sbitset0(valid, i);
        uint32_t idx = baseIdx + i;

        int32_t gmY0Offset = this->locationGm_[vecCoreIdx * 4*this->alignedOneTaskNum_ + idx].GetValue(0);
        int32_t gmY1Offset = this->locationGm_[vecCoreIdx * 4*this->alignedOneTaskNum_ + idx+this->alignedOneTaskNum_].GetValue(0);

        this->CopyInValue(value[i * this->alignedEmbedDims_],  this->valueGm_[gmY0Offset], this->cpRowDoubleParams_);

        this->CopyInValue(value[i * this->alignedEmbedDims_ + 2 * this->alignedCornerEmbedDims_], 
            this->valueGm_[gmY1Offset], this->cpRowDoubleParams_);
    }
}

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnCubeKernel<aligned, fastMode>::CopyFullPointFromCube(
    uint32_t head, uint32_t innerLoops, uint32_t assembleOffsetFromTbuf, uint32_t vecCoreIdx, const LocalTensor<float>& value)
{
    // Cube MTE3
    DataCopy(
        this->assembleGm_[(vecCoreIdx * this->outerLoops_ + head) * this->cornerRpt_ * B32_DATA_NUM_PER_REPEAT], 
        value, this->cornerRpt_ * B32_DATA_NUM_PER_REPEAT);
}

template<bool aligned, bool fastMode>
__aicore__ inline void MultiScaleDeformableAttnCubeKernel<aligned, fastMode>::Process()
{
    LocalTensor<float> value = this->assembleMBuf_.template Get<float>();
    uint32_t round = 0;
    uint32_t bufferFlag = 0;
    SetFlag<HardEvent::MTE3_MTE2>(0);
    SetFlag<HardEvent::MTE3_MTE2>(1);
    for (uint32_t taskLoopId = 0; taskLoopId < this->taskLoops_; taskLoopId++) {
        if(round >= this->coopRound) break;
        uint32_t taskIdx = this->compTaskNum_ * (taskLoopId * this->coreNum_ + this->blkIdx_);
        AscendC::CrossCoreWaitFlag(1);
        PipeBarrier<PIPE_ALL>();

        AscendC::DataCacheCleanAndInvalid<uint64_t, 
            AscendC::CacheLine::ENTIRE_DATA_CACHE,
            AscendC::DcciDst::CACHELINE_OUT>(this->validFlagGm_[0]);

        uint32_t divPoint = this->outerLoops_ / 5 * 4; 

        uint32_t vecCore[2]= {(uint32_t)this->blkIdx_ * 2, (uint32_t)this->blkIdx_ * 2 + 1};
        for (uint32_t coreidx = 0; coreidx < 2; ++coreidx) {  
            for (uint32_t head = divPoint; head < this->outerLoops_; ++head) {
                uint64_t baseIdx = head * this->innerLoopsAligned_;
                uint64_t headOffset = baseIdx / B32_DATA_NUM_PER_REPEAT;
                uint64_t byteOffset = baseIdx - headOffset * B32_DATA_NUM_PER_REPEAT;
                uint64_t valid = this->validFlagGm_[vecCore[coreidx] * this->validFlagMaskLen_ + headOffset].GetValue(0) >> byteOffset;
                uint32_t assembleOffsetFromTbuf = bufferFlag * this->innerLoopsAligned_*this->embedDims_*4*B32_BYTE_SIZE;
                uint32_t innerLoops = min(this->innerLoops_, this->innerTotal_ - (head % this->innerTotalGroup_) * this->innerLoops_);
                LocalTensor<float> valueSrc = value[bufferFlag * this->cornerRpt_ * B32_DATA_NUM_PER_REPEAT];
                WaitFlag<HardEvent::MTE3_MTE2>(bufferFlag);
                DataCopy(valueSrc, this->zeroGm_, this->cornerRpt_ * B32_DATA_NUM_PER_REPEAT);
                PipeBarrier<PIPE_MTE2>();
                CopyFullPointToCube(valid, baseIdx, innerLoops, vecCore[coreidx], bufferFlag, valueSrc);
                SetFlag<HardEvent::MTE2_MTE3>(bufferFlag);
                WaitFlag<HardEvent::MTE2_MTE3>(bufferFlag);
                CopyFullPointFromCube(head, this->innerLoops_, assembleOffsetFromTbuf, vecCore[coreidx], valueSrc);
                SetFlag<HardEvent::MTE3_MTE2>(bufferFlag);
                bufferFlag = 1 - bufferFlag;
            }
        }
        PipeBarrier<PIPE_ALL>();
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(2);
        round++;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(0);
    WaitFlag<HardEvent::MTE3_MTE2>(1);
}

extern "C" __global__ __aicore__ void multi_scale_deformable_attn(GM_ADDR value, GM_ADDR valueSpatialShapes,
    GM_ADDR valueLevelStartIndex, GM_ADDR samplingLocations, GM_ADDR attentionWeights, GM_ADDR output,
    GM_ADDR workspace, GM_ADDR tiling)
{

    GM_ADDR user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if ASCEND_IS_AIV{
        TPipe pipe;
        GET_TILING_DATA(tilingData, tiling);
        if (TILING_KEY_IS(11)) {
            MultiScaleDeformableAttnKernel<true, true> op(value, valueSpatialShapes, valueLevelStartIndex,
                samplingLocations, attentionWeights, output, user, &tilingData, &pipe);
            op.Process();
        } else if (TILING_KEY_IS(01)) {
            MultiScaleDeformableAttnKernel<false, true> op(value, valueSpatialShapes, valueLevelStartIndex,
                samplingLocations, attentionWeights, output, user, &tilingData, &pipe);
            op.Process();
        } else if (TILING_KEY_IS(10)) {
            MultiScaleDeformableAttnKernel<true, false> op(value, valueSpatialShapes, valueLevelStartIndex,
                samplingLocations, attentionWeights, output, user, &tilingData, &pipe);
            op.Process();
        } else if (TILING_KEY_IS(00)) {
            MultiScaleDeformableAttnKernel<false, false> op(value, valueSpatialShapes, valueLevelStartIndex,
                samplingLocations, attentionWeights, output, user, &tilingData, &pipe);
            op.Process();
        }
    }

    if ASCEND_IS_AIC{
        TPipe pipe;
        GET_TILING_DATA(tilingData, tiling);
        if (TILING_KEY_IS(10)) {
            MultiScaleDeformableAttnCubeKernel<true, false> op(value, valueSpatialShapes, valueLevelStartIndex,
                samplingLocations, attentionWeights, output, user, &tilingData, &pipe);
        op.Process();
        }
    }
}
