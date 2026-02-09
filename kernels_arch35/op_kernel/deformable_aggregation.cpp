/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 */
// v1.5.2-AscendC::Simt-outer-scalar

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;
using namespace MicroAPI;

namespace {
    constexpr uint32_t BLOCK_BYTE_SIZE = 32;
    constexpr uint32_t INT32_BYTE_SIZE = 4;
    constexpr uint32_t INT64_BYTE_SIZE = 8;
    constexpr uint32_t SINGLE_LOOP_COMPARE_UB = 256;
    constexpr uint32_t VEC_LEN = AscendC::GetVecLen();
    constexpr uint32_t BUFFER_NUM = 4;

    constexpr int32_t SHAPE_DIM = 2;
    constexpr int32_t HH_OFFSET = 0;
    constexpr int32_t HL_OFFSET = 1;
    constexpr int32_t LH_OFFSET = 2;
    constexpr int32_t LL_OFFSET = 3;
}

template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void prepareDataSIMT(
    __gm__ T* samplingLocationGm_,
    __ubuf__ int32_t* usedTaskOffset_, 
    __ubuf__ int32_t* usedWeightOffset_,
    __ubuf__ T* bilinearWeightLocal_, 
    __ubuf__ int32_t* usedFeatOffset_, 
    __ubuf__ uint8_t* taskPtr_, 
    __ubuf__ int32_t* scaleStartLocal_, 
    __ubuf__ int32_t* spatialShapeLocal_,
    int32_t taskIdx, 
    uint64_t baseOffset, 
    uint32_t actualCompNum, 
    uint32_t numAnchors_, 
    uint32_t numPoints_, 
    uint32_t numCams_, 
    uint32_t numScales_, 
    uint32_t numGroups_, 
    uint32_t numFeats_, 
    uint32_t numEmbeds_)
{
    // taskCompNum * numPoints * numCams_ * numScales_ = 312 * taskCompNum
    for (uint32_t i = AscendC::Simt::GetThreadIdx(); i < actualCompNum; i += AscendC::Simt::GetThreadNum()) {
        float locW = samplingLocationGm_[2 * i];
        float locH = samplingLocationGm_[2 * i + 1];
        if (locW <= 0 || locH <= 0 || locW >= 1 || locH >= 1) {
            taskPtr_[i] = 0;
            continue;
        }

        int32_t taskOffset = i / (numPoints_ * numCams_);
        int32_t batchIdx = (taskIdx + taskOffset) / numAnchors_;
        int32_t camIdx = i % numCams_;

        int32_t weightOffset = (i + baseOffset) * numScales_ * numGroups_;
        // 记录weightOffset和taskOffset
        usedTaskOffset_[i] = taskOffset;
        usedWeightOffset_[i] = weightOffset;
        taskPtr_[i] = 1;
        
        for (uint32_t scaleIdx = 0; scaleIdx < numScales_; scaleIdx++) {
            int32_t scaleStartOffset = camIdx * numScales_ + scaleIdx;
            int32_t spatialShapeOffset = scaleStartOffset * 2;
            int32_t scaleStartIdx = scaleStartLocal_[scaleStartOffset];
            int32_t valueOffset = (batchIdx * numFeats_ + scaleStartIdx) * numEmbeds_;

            int32_t h = spatialShapeLocal_[spatialShapeOffset];
            int32_t w = spatialShapeLocal_[spatialShapeOffset + 1];

            float hIm = locH * h - 0.5f;
            float wIm = locW * w - 0.5f;

            // 不能直接用Simt：：Cast，会有奇怪的报错，直接用=也是强制类型转换
            int32_t hLow = static_cast<int32_t>(AscendC::Simt::Floor(hIm));
            int32_t wLow = static_cast<int32_t>(AscendC::Simt::Floor(wIm));
            int32_t hHigh = hLow + 1;
            int32_t wHigh = wLow + 1;

            float lh = hIm - hLow;
            float lw = wIm - wLow;
            float hh = 1 - lh;
            float hw = 1 - lw;

            T w1 = hh * hw;
            T w2 = hh * lw;
            T w3 = lh * hw;
            T w4 = lh * lw;

            int32_t hStride = w * numEmbeds_;
            int32_t hLowPtrOffset = hLow * hStride;
            int32_t hHighPtrOffset = hLowPtrOffset + hStride;
            int32_t wLowPtrOffset = wLow * numEmbeds_;
            int32_t wHighPtrOffset = wLowPtrOffset + numEmbeds_;

            int32_t hhPtr = hLow >= 0 && wLow >= 0 ? valueOffset + hLowPtrOffset + wLowPtrOffset : -1;
            int32_t hlPtr = hLow >= 0 && wHigh <= w - 1 ? valueOffset + hLowPtrOffset + wHighPtrOffset : -1;
            int32_t lhPtr = hHigh <= h - 1 && wLow >= 0 ? valueOffset + hHighPtrOffset + wLowPtrOffset : -1;
            int32_t llPtr = hHigh <= h - 1 && wHigh <= w - 1 ? valueOffset + hHighPtrOffset + wHighPtrOffset : -1;

            // 记录4个权重
            bilinearWeightLocal_[4 * (i * numScales_ + scaleIdx) + 0] = w1;
            bilinearWeightLocal_[4 * (i * numScales_ + scaleIdx) + 1] = w2;
            bilinearWeightLocal_[4 * (i * numScales_ + scaleIdx) + 2] = w3;
            bilinearWeightLocal_[4 * (i * numScales_ + scaleIdx) + 3] = w4;

            // 记录4个offset，若在框外，则置为-1
            usedFeatOffset_[4 * (i * numScales_ + scaleIdx) + 0] = hhPtr;
            usedFeatOffset_[4 * (i * numScales_ + scaleIdx) + 1] = hlPtr;
            usedFeatOffset_[4 * (i * numScales_ + scaleIdx) + 2] = lhPtr;
            usedFeatOffset_[4 * (i * numScales_ + scaleIdx) + 3] = llPtr;
        }
    }
}


template<typename T>
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
        coreNum_ = tiling_data->coreNum;
        numChannels_ = numEmbeds_ / numGroups_;

        taskNum_ = bs_ * numAnchors_;
        taskNumPerCore_ = taskNum_ / coreNum_;
        uint32_t bigCoreCount = taskNum_ % coreNum_;
        curBlockIdx_ = AscendC::GetBlockIdx();
        if (curBlockIdx_ < bigCoreCount) {
            taskNumPerCore_ += 1;
            startOffset_ = curBlockIdx_ * taskNumPerCore_;
        } else {
            startOffset_ = (taskNumPerCore_ + 1) * bigCoreCount + taskNumPerCore_ * (curBlockIdx_ - bigCoreCount);
        }
        endOffset_ = startOffset_ + taskNumPerCore_;

        featByteSize_ = sizeof(T);
        blockAlignFloat_ = BLOCK_BYTE_SIZE / featByteSize_;
        vecAlignFLoat_ = VEC_LEN / featByteSize_;
        vecAlignInt_ = VEC_LEN / INT32_BYTE_SIZE;
        vecAlignInt64_ = VEC_LEN / INT64_BYTE_SIZE;
        repeatAlignFloat_ = SINGLE_LOOP_COMPARE_UB / featByteSize_;

        // full load
        scaleStartBufSize_ = AlignUp(numCams_ * numScales_, vecAlignInt_);
        spatialShapeBufSize_ = AlignUp(numCams_ * numScales_ * SHAPE_DIM, vecAlignInt_);

        // inner loop
        weightBufSize_ = AlignUp(numScales_ * numGroups_, vecAlignFLoat_);
        weightMulBufSize_ = AlignUp(numScales_ * cAligned_, vecAlignFLoat_);    // 4KB
        vLocalBufSize_ = 4 * weightMulBufSize_; // 16KB

        // outer loop, *taskCompNum
        bilinearWeightSize_ = 4 * numPoints_ * numCams_ * numScales_; // 4 * 13 * 6 * 4 = 1248, 1 veclen
        resLocalSize_ = cAligned_;  // 256, 1 veclen
        usedTaskOffsetSize_ = numPoints_ * numCams_; // 13 * 6 = 78, 1 veclen
        usedFeatBufSize_ = 4 * numPoints_ * numCams_ * numScales_;  // 4 * 13 * 6 * 4 = 1248, 1 veclen
        usedWeightOffsetSize_ = numPoints_ * numCams_ * numScales_;   // 13 * 6 * 4 = 312, 1 veclen
        taskCount_ = numPoints_ * numCams_; // 13 * 6 = 78, 1 veclen

        ubSize_ = tiling_data->ubSize;  // for AscendC::Simt
        uint32_t ubTotalSize = static_cast<uint32_t>(ubSize_);
        uint32_t usedUbSize = (scaleStartBufSize_ + spatialShapeBufSize_) * INT32_BYTE_SIZE + 
            BUFFER_NUM * (weightBufSize_ + weightMulBufSize_ + vLocalBufSize_) * featByteSize_ + 10 * VEC_LEN;

        uint32_t oneTaskUbSize = (bilinearWeightSize_ + resLocalSize_) * featByteSize_ + usedTaskOffsetSize_ * INT32_BYTE_SIZE + 
            (usedFeatBufSize_ + usedWeightOffsetSize_) * INT32_BYTE_SIZE + taskCount_;
        taskCompNum_ = (ubTotalSize - usedUbSize) / oneTaskUbSize;
        taskCompNum_ = max(taskCompNum_, (uint32_t)1);

        taskRpt_ = DivCeil(taskCompNum_ * numPoints_ * numCams_, repeatAlignFloat_);
        alignedOneTaskNum_ = taskRpt_ * repeatAlignFloat_;  // align 64

        srcShape_[0] = numScales_ * numGroups_;
        srcShape_[1] = 1;
        dstShape_[0] = numScales_ * numGroups_;
        dstShape_[1] = numChannels_;

        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        InitGlobalTensor(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights, out);
    }

    __aicore__ inline void InitGlobalTensor(GM_ADDR mc_ms_feat, GM_ADDR spatial_shape, GM_ADDR scale_start_index,
                                            GM_ADDR sampling_location, GM_ADDR weights, GM_ADDR out)
    {
        uint64_t mcMsFeatGmLength = bs_ * numFeats_ * numEmbeds_;
        uint64_t scaleStartIndexLength = numCams_ * numScales_;
        uint64_t spatialShapeGmLength = scaleStartIndexLength * 2;
        uint64_t samplingLocationGmLength = bs_ * numAnchors_ * numPoints_ * numCams_ * 2;
        uint64_t weightsGmLength = bs_ * numAnchors_ * numPoints_ * numCams_ * numScales_ * numGroups_;
        uint64_t outGmLength = bs_ * numAnchors_ * numEmbeds_;

        mcMsFeatGm_.SetGlobalBuffer((__gm__ T*)mc_ms_feat, mcMsFeatGmLength);
        samplingLocationGm_.SetGlobalBuffer((__gm__ T*)sampling_location, samplingLocationGmLength);
        weightsGm_.SetGlobalBuffer((__gm__ T*)weights, weightsGmLength);
        outGm_.SetGlobalBuffer((__gm__ T*)out, outGmLength);
        spatialShapesGm_.SetGlobalBuffer((__gm__ int32_t*)spatial_shape, spatialShapeGmLength);
        scaleStartIndexGm_.SetGlobalBuffer((__gm__ int32_t*)scale_start_index, scaleStartIndexLength);
    }

    __aicore__ inline void GetLocalTensor()
    {
        // numCams_ * numScales_ * 4 = 6 * 4 * 4 = 96B
        pipe_->InitBuffer(scaleStartBuf_, AlignUp(scaleStartBufSize_ * INT32_BYTE_SIZE, BLOCK_BYTE_SIZE));
        // numCams_ * numScales_ * 2 * 4 = 6 * 4 * 2 * 4 = 192B
        pipe_->InitBuffer(spatialShapeBuf_, AlignUp(spatialShapeBufSize_ * INT32_BYTE_SIZE, BLOCK_BYTE_SIZE));
        // numScales_ * numGroups_ * 4 = 24 * 4 = 96B
        pipe_->InitBuffer(weightBuf_, AlignUp(BUFFER_NUM * weightBufSize_ * featByteSize_, BLOCK_BYTE_SIZE));
        // numScales_ * cAligned * 4 = 4KB
        pipe_->InitBuffer(weightMulBuf_, AlignUp(BUFFER_NUM * weightMulBufSize_ * featByteSize_, BLOCK_BYTE_SIZE));
        // numScales_ * cAligned_ * 4 * 4 = 4 * 256 * 16 = 16384 = 16KB
        pipe_->InitBuffer(vBuf_, AlignUp(BUFFER_NUM * vLocalBufSize_ * featByteSize_, BLOCK_BYTE_SIZE));
        
        // taskCompNum_ * 256 * 4 = taskCompNum_ * 1024
        pipe_->InitBuffer(resBuf_, AlignUp(taskCompNum_ * cAligned_ * featByteSize_, VEC_LEN));

        // 4 * numScales_ * 4 = 64 * taskCompNum_, 0 veclen
        pipe_->InitBuffer(bilinearWeightBuf_,  AlignUp(taskCompNum_ * bilinearWeightSize_ * featByteSize_, VEC_LEN));
        // taskCompNum_ * usedTaskOffsetSize_ * 4
        pipe_->InitBuffer(usedTaskBuf_, AlignUp(taskCompNum_ * usedTaskOffsetSize_ * INT32_BYTE_SIZE, VEC_LEN));
        // taskCompNum_ * usedFeatBufSize_ * 8
        pipe_->InitBuffer(usedFeatBuf_, AlignUp(taskCompNum_ * usedFeatBufSize_ * INT32_BYTE_SIZE, VEC_LEN));
        // taskCompNum_ * usedWeightOffsetSize_ * 8
        pipe_->InitBuffer(usedWeightOffsetBuf_, AlignUp(taskCompNum_ * usedWeightOffsetSize_ * INT32_BYTE_SIZE, VEC_LEN));
        // taskCompNum_ * taskCount_
        pipe_->InitBuffer(taskPtrBuf_, AlignUp(taskCompNum_ * taskCount_, VEC_LEN));
        
        weightLocal_ = weightBuf_.Get<T>();
        scaleStartLocal_ = scaleStartBuf_.Get<int32_t>();
        spatialShapeLocal_ = spatialShapeBuf_.Get<int32_t>();
        weightMulLocal_ = weightMulBuf_.Get<T>();
        vLocal_ = vBuf_.Get<T>(); 
        
        bilinearWeightLocal_ = bilinearWeightBuf_.Get<T>();
        resLocal_ = resBuf_.Get<T>();
        usedTaskOffset_ = usedTaskBuf_.Get<int32_t>();
        usedFeatOffset_ = usedFeatBuf_.Get<int32_t>();
        usedWeightOffset_ = usedWeightOffsetBuf_.Get<int32_t>();
        taskPtr_ = taskPtrBuf_.Get<uint8_t>();
    }

    __aicore__ inline void Process()
    {   
        // load const values
        Duplicate(vLocal_, static_cast<T>(0.0f), BUFFER_NUM * vLocalBufSize_);
        DataCopy(scaleStartLocal_, scaleStartIndexGm_, scaleStartBufSize_);
        DataCopy(spatialShapeLocal_, spatialShapesGm_, spatialShapeBufSize_);

        for (uint32_t taskIdx = curBlockIdx_ * taskCompNum_; taskIdx < taskNum_; taskIdx += coreNum_ * taskCompNum_) {
            ComputeAndCopyOut(taskIdx, min(taskCompNum_, taskNum_ - taskIdx));
        }
    }

    __aicore__ inline void ComputeAndCopyOut(int32_t taskIdx, uint32_t actualTaskNum)
    {
        uint64_t outOffset = taskIdx * numEmbeds_;
        uint64_t baseOffset = taskIdx * numPoints_ * numCams_;
        uint32_t outerLoops = DivCeil(actualTaskNum * numPoints_ * numCams_, 64);
        uint32_t actualCompNum = actualTaskNum * numPoints_ * numCams_;
        
        // simt的优势是处理scalar，离散访存
        PipeBarrier<PIPE_ALL>();
        AscendC::Simt::VF_CALL<prepareDataSIMT<T>>(
            AscendC::Simt::Dim3{1024}, 
            (__gm__ T*) samplingLocationGm_[baseOffset * 2].GetPhyAddr(),
            (__ubuf__ int32_t*) usedTaskOffset_.GetPhyAddr(), 
            (__ubuf__ int32_t*) usedWeightOffset_.GetPhyAddr(), 
            (__ubuf__ T*) bilinearWeightLocal_.GetPhyAddr(), 
            (__ubuf__ int32_t*) usedFeatOffset_.GetPhyAddr(), 
            (__ubuf__ uint8_t*) taskPtr_.GetPhyAddr(), 
            (__ubuf__ int32_t*) scaleStartLocal_.GetPhyAddr(),
            (__ubuf__ int32_t*) spatialShapeLocal_.GetPhyAddr(), 
            taskIdx, 
            baseOffset, 
            actualCompNum, 
            numAnchors_, 
            numPoints_, 
            numCams_, 
            numScales_, 
            numGroups_, 
            numFeats_, 
            numEmbeds_);
        PipeBarrier<PIPE_ALL>();

        Duplicate(resLocal_, static_cast<T>(0.0f), actualTaskNum * cAligned_);
        CompareScalar(taskPtr_, taskPtr_, static_cast<uint8_t>(1), AscendC::CMPMODE::EQ, actualCompNum);

        for (uint8_t i = 0; i < BUFFER_NUM; i++) {
            SetFlag<HardEvent::V_MTE2>(i);
        }

        for (uint32_t outerIdx = 0; outerIdx < outerLoops; outerIdx++) {
            uint64_t valid = taskPtr_.ReinterpretCast<uint64_t>().GetValue(outerIdx);
            uint32_t innerLoops = min(actualCompNum - 64 * outerIdx, static_cast<uint32_t>(64));
            for (uint32_t innerIdx = ScalarGetSFFValue<1>(valid); innerIdx < innerLoops && innerIdx >= 0; innerIdx = ScalarGetSFFValue<1>(valid)) {
                valid = sbitset0(valid, innerIdx);

                uint32_t idx = outerIdx * 64 + innerIdx;
                int32_t taskOffset = usedTaskOffset_.GetValue(idx);
                int32_t weightOffset = usedWeightOffset_.GetValue(idx);

                // numScales * numGroups
                DataCopy(weightLocal_[bufIdx_ * weightBufSize_], weightsGm_[weightOffset], weightBufSize_);
                SetFlag<HardEvent::MTE2_V>(bufIdx_);
                WaitFlag<HardEvent::MTE2_V>(bufIdx_);
                // numScales * cAligned
                BroadCast<T, 2, 1>(weightMulLocal_[bufIdx_ * weightMulBufSize_], weightLocal_[bufIdx_ * weightBufSize_], dstShape_, srcShape_);

                WaitFlag<HardEvent::V_MTE2>(bufIdx_);
                // 4 * numScales * cAligned
                copyInFeat(vLocal_[bufIdx_ * vLocalBufSize_], usedFeatOffset_[4 * idx * numScales_]);

                SetFlag<HardEvent::MTE2_V>(BUFFER_NUM + bufIdx_);
                WaitFlag<HardEvent::MTE2_V>(BUFFER_NUM + bufIdx_);
                SetFlag<HardEvent::V_S>(bufIdx_);
                WaitFlag<HardEvent::V_S>(bufIdx_);

                // 一次性计算4 * numScales个点
                computeAggregationVF(resLocal_[taskOffset * cAligned_], weightMulLocal_[bufIdx_ * weightMulBufSize_], 
                    bilinearWeightLocal_[4 * idx * numScales_], vLocal_[bufIdx_ * vLocalBufSize_]);
                SetFlag<HardEvent::V_MTE2>(bufIdx_);

                bufIdx_ = (bufIdx_ + 1) % BUFFER_NUM;
            }
        }

        for (uint8_t i = 0; i < BUFFER_NUM; i++) {
            WaitFlag<HardEvent::V_MTE2>(i);
        }

        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(outGm_[outOffset], resLocal_, {1, static_cast<uint16_t>(actualTaskNum * numEmbeds_ * featByteSize_), 0, 0});
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void copyInFeat(LocalTensor<T> vLocal, LocalTensor<int32_t> usedFeatOffset)
    {
        // 4 * numScales * cAligned
        for (uint32_t scaleIdx = 0; scaleIdx < numScales_; scaleIdx++) {
            uint32_t scaleOffset = 4 * scaleIdx;
            int32_t hhPtr = usedFeatOffset.GetValue(scaleOffset + 0);
            int32_t hlPtr = usedFeatOffset.GetValue(scaleOffset + 1);
            int32_t lhPtr = usedFeatOffset.GetValue(scaleOffset + 2);
            int32_t llPtr = usedFeatOffset.GetValue(scaleOffset + 3);

            if (hhPtr != -1 && hlPtr != -1 && lhPtr != -1 && llPtr != -1) {
                DataCopy(vLocal[scaleOffset * cAligned_], mcMsFeatGm_[hhPtr], 
                    {2, static_cast<uint16_t>(DivCeil(2 * cAligned_, blockAlignFloat_)),
                    static_cast<uint16_t>(DivCeil(lhPtr - hlPtr - cAligned_, blockAlignFloat_)), 0});
                continue;
            }
            if (hhPtr != -1 && hlPtr != -1) {
                DataCopy(vLocal[(scaleOffset + HH_OFFSET) * cAligned_], mcMsFeatGm_[hhPtr], TWO * cAligned_);
            } else if (hhPtr != -1) {
                DataCopy(vLocal[(scaleOffset + HH_OFFSET) * cAligned_], mcMsFeatGm_[hhPtr], cAligned_);
            } else if (hlPtr != -1) {
                DataCopy(vLocal[(scaleOffset + HL_OFFSET) * cAligned_], mcMsFeatGm_[hlPtr], cAligned_);
            }
            if (lhPtr != -1 && llPtr != -1) {
                DataCopy(vLocal[(scaleOffset + LH_OFFSET) * cAligned_], mcMsFeatGm_[lhPtr], TWO * cAligned_);
            } else if (lhPtr != -1) {
                DataCopy(vLocal[(scaleOffset + LH_OFFSET) * cAligned_], mcMsFeatGm_[lhPtr], cAligned_);
            } else if (llPtr != -1) {
                DataCopy(vLocal[(scaleOffset + LL_OFFSET) * cAligned_], mcMsFeatGm_[llPtr], cAligned_);
            }
        }
    }

    __aicore__ inline void computeAggregationVF(LocalTensor<T> resLocal, LocalTensor<T> weightLocal, 
        LocalTensor<T> bilinearWeightLocal, LocalTensor<T> vLocal) 
    {
        // 256 / 4 = 64
        uint32_t repeatSizeT = VEC_LEN / sizeof(T);
        // 256 / 64 = 4
        uint16_t repeatTimes = DivCeil(cAligned_, repeatSizeT);
        
        // numScales_ * numGroups_ * numChannels = 4 * 8 * 32 = numScales_ * 4 * repeatSizeT
        __local_mem__ T* weightPtr = (__local_mem__ T*) weightLocal.GetPhyAddr();
        // 4 * numScales_ * cAligned_ = 4 * 4 * 256 = 4个顶点 * numScales_ * repeatTimes * repeatSizeT
        __local_mem__ T* vLocalPtr = (__local_mem__ T*) vLocal.GetPhyAddr();
        // cAligned_ = repeatTimes * repeatSizeT
        __local_mem__ T* resLocalPtr = (__local_mem__ T*) resLocal.GetPhyAddr();
        // 4 * numScales_ = 4个顶点 * numScales_
        __local_mem__ T* bilinearWeighPtr = (__local_mem__ T*) bilinearWeightLocal.GetPhyAddr();

        __VEC_SCOPE__ {
            MicroAPI::RegTensor<T> weightReg;

            MicroAPI::RegTensor<T> hhMulReg;
            MicroAPI::RegTensor<T> hlMulReg;
            MicroAPI::RegTensor<T> lhMulReg;
            MicroAPI::RegTensor<T> llMulReg;

            MicroAPI::RegTensor<T> hhWeightReg;
            MicroAPI::RegTensor<T> hlWeightReg;
            MicroAPI::RegTensor<T> lhWeightReg;
            MicroAPI::RegTensor<T> llWeightReg;

            MicroAPI::RegTensor<T> bilinearWeighthh;
            MicroAPI::RegTensor<T> bilinearWeighthl;
            MicroAPI::RegTensor<T> bilinearWeightlh;
            MicroAPI::RegTensor<T> bilinearWeightll;

            MicroAPI::RegTensor<T> hhVReg;
            MicroAPI::RegTensor<T> hlVReg;
            MicroAPI::RegTensor<T> lhVReg;
            MicroAPI::RegTensor<T> llVReg;

            MicroAPI::RegTensor<T> hResReg;
            MicroAPI::RegTensor<T> lResReg;
            MicroAPI::RegTensor<T> resReg;
            MicroAPI::RegTensor<T> resValueReg;

            MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();

            MicroAPI::RegTensor<T> zeroReg;
            MicroAPI::Duplicate<T>(zeroReg, static_cast<T>(0.0f));

            for (uint16_t i = 0; i < repeatTimes; i++) {
                // 搬入resLocal
                MicroAPI::DataCopy(resValueReg, resLocalPtr + i * repeatSizeT);

                for (uint16_t scaleIdx = 0; scaleIdx < static_cast<uint16_t>(numScales_); scaleIdx++) {
                    // cAligned = 256, repeatTimes = 4
                    if (sizeof(T) == 4) {
                        MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B32>(bilinearWeighthh, bilinearWeighPtr + scaleIdx * 4 + 0);
                        MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B32>(bilinearWeighthl, bilinearWeighPtr + scaleIdx * 4 + 1);
                        MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B32>(bilinearWeightlh, bilinearWeighPtr + scaleIdx * 4 + 2);
                        MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B32>(bilinearWeightll, bilinearWeighPtr + scaleIdx * 4 + 3);
                    } else {
                        MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B16>(bilinearWeighthh, bilinearWeighPtr + scaleIdx * 4 + 0);
                        MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B16>(bilinearWeighthl, bilinearWeighPtr + scaleIdx * 4 + 1);
                        MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B16>(bilinearWeightlh, bilinearWeighPtr + scaleIdx * 4 + 2);
                        MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B16>(bilinearWeightll, bilinearWeighPtr + scaleIdx * 4 + 3);
                    }
                    // 搬入weightMul, numScales_ * cAligned
                    MicroAPI::DataCopy(weightReg, weightPtr + scaleIdx * cAligned_ + i * repeatSizeT);
                    
                    // 搬入featWeight, 4 * numScales_ * cAligned
                    MicroAPI::DataCopy(hhVReg, vLocalPtr + (4 * scaleIdx + 0) * cAligned_ + i * repeatSizeT);
                    MicroAPI::DataCopy(hlVReg, vLocalPtr + (4 * scaleIdx + 1) * cAligned_ + i * repeatSizeT);
                    MicroAPI::DataCopy(lhVReg, vLocalPtr + (4 * scaleIdx + 2) * cAligned_ + i * repeatSizeT);
                    MicroAPI::DataCopy(llVReg, vLocalPtr + (4 * scaleIdx + 3) * cAligned_ + i * repeatSizeT);

                    // 双线性weight
                    MicroAPI::Mul(hhMulReg, weightReg, bilinearWeighthh, mask);
                    MicroAPI::Mul(hlMulReg, weightReg, bilinearWeighthl, mask);
                    MicroAPI::Mul(lhMulReg, weightReg, bilinearWeightlh, mask);
                    MicroAPI::Mul(llMulReg, weightReg, bilinearWeightll, mask);

                    // weight和featWeight相乘
                    MicroAPI::Mul(hhWeightReg, hhMulReg, hhVReg, mask);
                    MicroAPI::Mul(hlWeightReg, hlMulReg, hlVReg, mask);
                    MicroAPI::Mul(lhWeightReg, lhMulReg, lhVReg, mask);
                    MicroAPI::Mul(llWeightReg, llMulReg, llVReg, mask);

                    // 计算resLocal
                    MicroAPI::Add(hResReg, hhWeightReg, hlWeightReg, mask);
                    MicroAPI::Add(lResReg, lhWeightReg, llWeightReg, mask);
                    MicroAPI::Add(resReg, hResReg, lResReg, mask);
                    MicroAPI::Add(resValueReg, resValueReg, resReg, mask);

                    // 置零vLocal
                    MicroAPI::DataCopy(vLocalPtr + (4 * scaleIdx + 0) * cAligned_ + i * repeatSizeT, zeroReg, mask);
                    MicroAPI::DataCopy(vLocalPtr + (4 * scaleIdx + 1) * cAligned_ + i * repeatSizeT, zeroReg, mask);
                    MicroAPI::DataCopy(vLocalPtr + (4 * scaleIdx + 2) * cAligned_ + i * repeatSizeT, zeroReg, mask);
                    MicroAPI::DataCopy(vLocalPtr + (4 * scaleIdx + 3) * cAligned_ + i * repeatSizeT, zeroReg, mask);
                }
                // 写回resLocal
                MicroAPI::DataCopy(resLocalPtr + i * repeatSizeT, resValueReg, mask);
            }
        }
    }


private:
    TPipe *pipe_;

    TBuf<TPosition::VECCALC> weightBuf_, locationBuf_, scaleStartBuf_, spatialShapeBuf_;
    TBuf<TPosition::VECCALC> vBuf_, weightMulBuf_, resBuf_;
    TBuf<TPosition::VECCALC> bilinearWeightBuf_;

    GlobalTensor<T> mcMsFeatGm_, samplingLocationGm_, weightsGm_, outGm_;
    GlobalTensor<int32_t> spatialShapesGm_, scaleStartIndexGm_;

    LocalTensor<T> locationLocal_, weightLocal_;
    LocalTensor<int32_t> spatialShapeLocal_, scaleStartLocal_;
    LocalTensor<T> vLocal_, weightMulLocal_, resLocal_;
    LocalTensor<T> bilinearWeightLocal_;
    
    // for used points
    TBuf<TPosition::VECCALC> usedTaskBuf_, usedFeatBuf_, usedWeightOffsetBuf_, taskPtrBuf_;
    LocalTensor<int32_t> usedTaskOffset_, usedWeightOffset_, usedFeatOffset_;
    LocalTensor<uint8_t> taskPtr_;

    uint32_t basePtr_, realPtr_;
    uint32_t coreNum_, curBlockIdx_;
    uint32_t taskNum_, taskCompNum_, taskRpt_, taskNumPerCore_, startOffset_, endOffset_;
    uint32_t alignedOneTaskNum_;
    uint32_t weightBufSize_, scaleStartBufSize_, spatialShapeBufSize_, weightMulBufSize_, vLocalBufSize_;
    uint32_t bs_, numFeats_, numEmbeds_, numAnchors_, numPoints_, numCams_, numScales_, numGroups_, numChannels_, cAligned_;
    uint32_t vecAlignFLoat_, vecAlignInt_, repeatAlignFloat_, blockAlignFloat_, vecAlignInt64_;
    uint32_t featByteSize_;
    uint64_t ubSize_;
    uint32_t bilinearWeightSize_, locationSize_, resLocalSize_, usedFeatBufSize_, usedTaskOffsetSize_, usedWeightOffsetSize_, taskCount_;

    uint32_t bufIdx_ = 0;
    uint32_t srcShape_[2], dstShape_[2];
};

extern "C" __global__ __aicore__ void deformable_aggregation(GM_ADDR mc_ms_feat, GM_ADDR spatial_shape,
    GM_ADDR scale_start_index, GM_ADDR sampling_location, GM_ADDR weights, GM_ADDR out, GM_ADDR workspace,
    GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    KernelDeformableAggregation<DTYPE_MC_MS_FEAT> op;
    op.Init(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights, out, &tiling_data, &pipe);
    op.GetLocalTensor();
    op.Process();
}
