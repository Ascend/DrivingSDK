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
constexpr uint32_t BUFFER_NUM = 2;

constexpr int32_t SHAPE_DIM = 2;
constexpr int32_t HH_OFFSET = 0;
constexpr int32_t HL_OFFSET = 1;
constexpr int32_t LH_OFFSET = 2;
constexpr int32_t LL_OFFSET = 3;
constexpr uint32_t THREAD_NUM = 512;
constexpr int32_t MAX_SINGLE_LOOP_POINTS = 128;
constexpr int32_t SINGLE_LOOP_COMPARE_POINTS = 64;
constexpr int32_t UINT8_BIT = 8;
constexpr int32_t VERTEX_NUM = 4;
} // namespace

// GetValidPointSIMT:每次处理一个anchor的所有point的validPoint
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void GetValidPointSIMT(
    __gm__ T *samplingLocationGm_, __ubuf__ uint8_t *validPoint_, uint32_t actualCompNum) {
    for (uint32_t i = AscendC::Simt::GetThreadIdx(); i < actualCompNum; i += AscendC::Simt::GetThreadNum()) {
        float locW = samplingLocationGm_[2 * i];
        float locH = samplingLocationGm_[2 * i + 1];

        validPoint_[i] = locW > 0 && locH > 0 && locW < 1 && locH < 1;
    }
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void GetInnerLoopDataSIMT(__gm__ T *samplingLocationGm_,
    __ubuf__ uint8_t *validPoint_, __ubuf__ int32_t *scaleStartLocal_, __ubuf__ int32_t *spatialShapeLocal_,
    __ubuf__ T *bilinearWeightLocal_, __ubuf__ int4 *usedFeatOffset_, int32_t batchIdx, int32_t outerOffset,
    uint32_t actualCompNum, uint32_t numCams_, uint32_t numScales_, uint32_t numGroups_, uint32_t numFeats_,
    uint32_t numEmbeds_) {
    for (uint32_t i = Simt::GetThreadIdx(); i < actualCompNum; i += Simt::GetThreadNum()) {
        if (validPoint_[i] == 0) {
            continue;
        }

        float locW = samplingLocationGm_[2 * i];
        float locH = samplingLocationGm_[2 * i + 1];

        int32_t camIdx = (i + outerOffset) % numCams_;

        for (uint32_t scaleIdx = 0; scaleIdx < numScales_; scaleIdx++) {
            int32_t scaleStartOffset = camIdx * numScales_ + scaleIdx;
            int32_t spatialShapeOffset = scaleStartOffset * 2;
            int32_t scaleStartIdx = scaleStartLocal_[scaleStartOffset];
            int32_t valueOffset = (batchIdx * numFeats_ + scaleStartIdx) * numEmbeds_;
            int32_t localOffset = i * numScales_ + scaleIdx;

            int32_t h = spatialShapeLocal_[spatialShapeOffset];
            int32_t w = spatialShapeLocal_[spatialShapeOffset + 1];

            float hIm = locH * h - 0.5f;
            float wIm = locW * w - 0.5f;

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
            int4 featOffset = {hhPtr, hlPtr, lhPtr, llPtr};

            // 记录4个权重
            bilinearWeightLocal_[VERTEX_NUM * localOffset + 0] = w1;
            bilinearWeightLocal_[VERTEX_NUM * localOffset + 1] = w2;
            bilinearWeightLocal_[VERTEX_NUM * localOffset + 2] = w3;
            bilinearWeightLocal_[VERTEX_NUM * localOffset + 3] = w4;

            // 记录4个offset，若在框外，则置为-1
            usedFeatOffset_[localOffset] = featOffset;
        }
    }
}

template <typename T> class KernelDeformableAggregation {
  public:
    __aicore__ inline KernelDeformableAggregation() {}
    __aicore__ inline void Init(GM_ADDR mc_ms_feat, GM_ADDR spatial_shape, GM_ADDR scale_start_index,
        GM_ADDR sampling_location, GM_ADDR weights, GM_ADDR out, const DeformableAggregationTilingData *tiling_data,
        TPipe *tmpPipe) {
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
        curBlockIdx_ = AscendC::GetBlockIdx();

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
        weightMulBufSize_ = AlignUp(numScales_ * cAligned_, vecAlignFLoat_); // 4KB
        vLocalBufSize_ = VERTEX_NUM * weightMulBufSize_; // 16KB

        ubSize_ = tiling_data->ubSize; // for AscendC::Simt
        int32_t ubTotalSize = ubSize_;
        // usedUBSize约等于24kb
        int32_t usedUbSize = (scaleStartBufSize_ + spatialShapeBufSize_) * INT32_BYTE_SIZE +
            BUFFER_NUM * (weightBufSize_ + weightMulBufSize_ + vLocalBufSize_) * featByteSize_ + 8 * VEC_LEN;

        resLocalSize_ = cAligned_; // 256, 1 veclen
        taskCount_ = numPoints_ * numCams_; // 13 * 6 = 78, 1 veclen, taskCount_是可以整个做完的

        // no generalization for numCams or numScales
        bilinearWeightSize_ = VERTEX_NUM * MAX_SINGLE_LOOP_POINTS * numCams_ * numScales_;
        usedFeatBufSize_ = VERTEX_NUM * MAX_SINGLE_LOOP_POINTS * numCams_ * numScales_;

        outerLoopTimes_ = (MAX_SINGLE_LOOP_POINTS * numCams_) / SINGLE_LOOP_COMPARE_POINTS; // 12

        srcShape_[0] = numScales_ * numGroups_;
        srcShape_[1] = 1;
        dstShape_[0] = numScales_ * numGroups_;
        dstShape_[1] = numChannels_;

        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        InitGlobalTensor(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights, out);
    }

    __aicore__ inline void InitGlobalTensor(GM_ADDR mc_ms_feat, GM_ADDR spatial_shape, GM_ADDR scale_start_index,
        GM_ADDR sampling_location, GM_ADDR weights, GM_ADDR out) {
        uint64_t mcMsFeatGmLength = bs_ * numFeats_ * numEmbeds_;
        uint64_t scaleStartIndexLength = numCams_ * numScales_;
        uint64_t spatialShapeGmLength = scaleStartIndexLength * 2;
        uint64_t samplingLocationGmLength = bs_ * numAnchors_ * numPoints_ * numCams_ * 2;
        uint64_t weightsGmLength = bs_ * numAnchors_ * numPoints_ * numCams_ * numScales_ * numGroups_;
        uint64_t outGmLength = bs_ * numAnchors_ * numEmbeds_;

        mcMsFeatGm_.SetGlobalBuffer((__gm__ T *)mc_ms_feat, mcMsFeatGmLength);
        samplingLocationGm_.SetGlobalBuffer((__gm__ T *)sampling_location, samplingLocationGmLength);
        weightsGm_.SetGlobalBuffer((__gm__ T *)weights, weightsGmLength);
        outGm_.SetGlobalBuffer((__gm__ T *)out, outGmLength);
        spatialShapesGm_.SetGlobalBuffer((__gm__ int32_t *)spatial_shape, spatialShapeGmLength);
        scaleStartIndexGm_.SetGlobalBuffer((__gm__ int32_t *)scale_start_index, scaleStartIndexLength);
    }

    __aicore__ inline void GetLocalTensor() {
        // numScales_: S
        // numCams_: M
        // numPoints_: P
        // numGroups_: G
        // cAligned_: C
        pipe_->InitBuffer(
            scaleStartBuf_, AlignUp(scaleStartBufSize_ * INT32_BYTE_SIZE, BLOCK_BYTE_SIZE)); // M * S * 4 = 96B
        pipe_->InitBuffer(
            spatialShapeBuf_, AlignUp(spatialShapeBufSize_ * INT32_BYTE_SIZE, BLOCK_BYTE_SIZE)); // M * S * 2 * 4 = 192B
        pipe_->InitBuffer(
            weightBuf_, AlignUp(BUFFER_NUM * weightBufSize_ * featByteSize_, BLOCK_BYTE_SIZE)); // 2 * 64 * 4 = 512B
        pipe_->InitBuffer(
            weightMulBuf_, AlignUp(BUFFER_NUM * weightMulBufSize_ * featByteSize_, BLOCK_BYTE_SIZE)); // S * C * 4 = 4KB
        pipe_->InitBuffer(
            vBuf_, AlignUp(BUFFER_NUM * vLocalBufSize_ * featByteSize_, BLOCK_BYTE_SIZE)); // S * C * 4 * 4 = 16KB

        pipe_->InitBuffer(resBuf_, AlignUp(cAligned_ * featByteSize_, VEC_LEN)); // C * 4 = 1024T
        pipe_->InitBuffer(
            bilinearWeightBuf_, AlignUp(bilinearWeightSize_ * featByteSize_, VEC_LEN)); // P * M * S * 4 * 4 = 4992T
        pipe_->InitBuffer(
            usedFeatBuf_, AlignUp(usedFeatBufSize_ * INT32_BYTE_SIZE, VEC_LEN)); // P * M * S * 4 * 4 = 4992T
        pipe_->InitBuffer(validPointBuf_, AlignUp(taskCount_, VEC_LEN) * 2); // 2 * P * M

        weightLocal_ = weightBuf_.Get<T>();
        scaleStartLocal_ = scaleStartBuf_.Get<int32_t>();
        spatialShapeLocal_ = spatialShapeBuf_.Get<int32_t>();
        weightMulLocal_ = weightMulBuf_.Get<T>();
        vLocal_ = vBuf_.Get<T>();

        bilinearWeightLocal_ = bilinearWeightBuf_.Get<T>();
        resLocal_ = resBuf_.Get<T>();
        usedFeatOffset_ = usedFeatBuf_.Get<int32_t>();
        validPointMask_ = validPointBuf_.Get<uint8_t>();
        validPoint_ = validPointMask_[AlignUp(taskCount_, VEC_LEN)];
    }

    __aicore__ inline void Process() {
        // load const values
        Duplicate(vLocal_, static_cast<T>(0.0f), BUFFER_NUM * vLocalBufSize_);
        DataCopy(scaleStartLocal_, scaleStartIndexGm_, scaleStartBufSize_);
        DataCopy(spatialShapeLocal_, spatialShapesGm_, spatialShapeBufSize_);

        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);

        for (uint32_t taskIdx = curBlockIdx_; taskIdx < taskNum_; taskIdx += coreNum_) {
            ComputeAndCopyOut(taskIdx);
        }
    }

    // 需要多次循环才能处理一个anchor
    __aicore__ inline void ComputeAndCopyOut(int32_t taskIdx) {
        uint64_t outOffset = taskIdx * numEmbeds_;
        uint64_t baseOffset = taskIdx * numPoints_ * numCams_;
        int32_t outerLoops = DivCeil(numPoints_ * numCams_, SINGLE_LOOP_COMPARE_POINTS);
        int32_t actualCompNum = numPoints_ * numCams_; // SparseDrive case: 300*6 = 1800
        int32_t batchIdx = taskIdx / numAnchors_;

        // 统计单个anchor的所有point的valid信息
        AscendC::Simt::VF_CALL<GetValidPointSIMT<T>>(AscendC::Simt::Dim3{THREAD_NUM},
            (__gm__ T *)samplingLocationGm_[baseOffset * 2].GetPhyAddr(), (__ubuf__ uint8_t *)validPoint_.GetPhyAddr(),
            actualCompNum);

        Duplicate(resLocal_, static_cast<T>(0.0f), cAligned_);
        CompareScalar(validPointMask_, validPoint_, static_cast<uint8_t>(1), AscendC::CMPMODE::EQ, actualCompNum);

        // numPoints * numCams / SINGLE_LOOP_COMPARE_POINTS
        for (int32_t outerIdx = 0; outerIdx < outerLoops; outerIdx += outerLoopTimes_) {
            // SINGLE_LOOP_COMPARE_POINTS * numCams
            int32_t outerOffset = SINGLE_LOOP_COMPARE_POINTS * outerIdx;
            int32_t pointNum =
                min(actualCompNum - outerOffset, static_cast<int32_t>(MAX_SINGLE_LOOP_POINTS * numCams_));
            int32_t curLoopTimes = min(outerLoops - outerIdx, outerLoopTimes_);

            // SINGLE_LOOP_COMPARE_POINTS * numCams = outerLoopTimes_ * SINGLE_LOOP_COMPARE_POINTS
            AscendC::Simt::VF_CALL<GetInnerLoopDataSIMT<T>>(AscendC::Simt::Dim3{THREAD_NUM},
                (__gm__ T *)samplingLocationGm_[(baseOffset + outerOffset) * 2].GetPhyAddr(),
                (__ubuf__ uint8_t *)validPoint_[outerOffset].GetPhyAddr(),
                (__ubuf__ int32_t *)scaleStartLocal_.GetPhyAddr(), (__ubuf__ int32_t *)spatialShapeLocal_.GetPhyAddr(),
                (__ubuf__ T *)bilinearWeightLocal_.GetPhyAddr(), (__ubuf__ int4 *)usedFeatOffset_.GetPhyAddr(),
                batchIdx, outerOffset, pointNum, numCams_, numScales_, numGroups_, numFeats_, numEmbeds_);

            SetFlag<HardEvent::V_S>(0);
            WaitFlag<HardEvent::V_S>(0);

            ComputeInnerLoop(baseOffset, outerIdx, curLoopTimes, actualCompNum);
        }

        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(outGm_[outOffset], resLocal_, {1, static_cast<uint16_t>(numEmbeds_ * featByteSize_), 0, 0});
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void ComputeInnerLoop(
        uint64_t baseOffset, int32_t batchStartIdx, int32_t curLoopTimes, int32_t actualCompNum) {
        for (uint8_t i = 0; i < BUFFER_NUM; i++) {
            SetFlag<HardEvent::V_MTE2>(i);
        }

        for (int32_t innerLoopIdx = 0; innerLoopIdx < curLoopTimes; innerLoopIdx++) {
            int32_t outerIdx = batchStartIdx + innerLoopIdx;
            int32_t outerOffset = SINGLE_LOOP_COMPARE_POINTS * outerIdx;
            uint64_t valid = validPointMask_.ReinterpretCast<uint64_t>().GetValue(outerIdx);
            int32_t innerLoops = min(actualCompNum - SINGLE_LOOP_COMPARE_POINTS * outerIdx, SINGLE_LOOP_COMPARE_POINTS);

            // SINGLE_LOOP_COMPARE_POINTS
            for (int32_t innerIdx = ScalarGetSFFValue<1>(valid); innerIdx < innerLoops && innerIdx >= 0;
                 innerIdx = ScalarGetSFFValue<1>(valid)) {
                valid = sbitset0(valid, innerIdx);
                SetFlag<HardEvent::V_S>(BUFFER_NUM + bufIdx_);
                int32_t localIdx = innerLoopIdx * SINGLE_LOOP_COMPARE_POINTS + innerIdx;
                uint64_t weightOffset = (baseOffset + outerOffset + innerIdx) * numScales_ * numGroups_;

                // numScales * numGroups
                DataCopy(weightLocal_[bufIdx_ * weightBufSize_], weightsGm_[weightOffset], weightBufSize_);

                SetFlag<HardEvent::MTE2_V>(bufIdx_);
                WaitFlag<HardEvent::MTE2_V>(bufIdx_);

                // numScales * cAligned
                BroadCast<T, SHAPE_DIM, 1>(weightMulLocal_[bufIdx_ * weightMulBufSize_],
                    weightLocal_[bufIdx_ * weightBufSize_], dstShape_, srcShape_);

                WaitFlag<HardEvent::V_MTE2>(bufIdx_);
                // 4 * numScales * cAligned
                copyInFeat(vLocal_[bufIdx_ * vLocalBufSize_], usedFeatOffset_[VERTEX_NUM * localIdx * numScales_]);

                SetFlag<HardEvent::MTE2_V>(BUFFER_NUM + bufIdx_);
                WaitFlag<HardEvent::MTE2_V>(BUFFER_NUM + bufIdx_);

                // 一次计算4 * numScales个点
                ComputeAggregationVF(resLocal_, weightMulLocal_[bufIdx_ * weightMulBufSize_],
                    bilinearWeightLocal_[VERTEX_NUM * localIdx * numScales_], vLocal_[bufIdx_ * vLocalBufSize_]);

                SetFlag<HardEvent::V_MTE2>(bufIdx_);
                WaitFlag<HardEvent::V_S>(BUFFER_NUM + bufIdx_);

                bufIdx_ = (bufIdx_ + 1) % BUFFER_NUM;
            }
        }

        for (uint8_t i = 0; i < BUFFER_NUM; i++) {
            WaitFlag<HardEvent::V_MTE2>(i);
        }
    }

    __aicore__ inline void copyInFeat(LocalTensor<T> vLocal, LocalTensor<int32_t> usedFeatOffset) {
        // 4 * numScales * cAligned
        for (uint32_t scaleIdx = 0; scaleIdx < numScales_; scaleIdx++) {
            uint32_t scaleOffset = VERTEX_NUM * scaleIdx;
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

    __aicore__ inline void ComputeAggregationVF(LocalTensor<T> resLocal, LocalTensor<T> weightLocal,
        LocalTensor<T> bilinearWeightLocal, LocalTensor<T> vLocal) {
        // 256 / 4 = 64
        uint32_t repeatSizeT = VEC_LEN / sizeof(T);
        // 256 / 64 = 4
        uint16_t repeatTimes = DivCeil(cAligned_, repeatSizeT);
        uint32_t maskSize = cAligned_;

        // numScales_ * numGroups_ * numChannels = 4 * 8 * 32 = numScales_ * 4 * repeatSizeT
        __local_mem__ T *weightPtr = (__local_mem__ T *)weightLocal.GetPhyAddr();
        // 4 * numScales_ * cAligned_ = 4 * 4 * 256 = 4个顶点 * numScales_ * repeatTimes * repeatSizeT
        __local_mem__ T *vLocalPtr = (__local_mem__ T *)vLocal.GetPhyAddr();
        // cAligned_ = repeatTimes * repeatSizeT
        __local_mem__ T *resLocalPtr = (__local_mem__ T *)resLocal.GetPhyAddr();
        // 4 * numScales_ = 4个顶点 * numScales_
        __local_mem__ T *bilinearWeighPtr = (__local_mem__ T *)bilinearWeightLocal.GetPhyAddr();

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

            MicroAPI::MaskReg mask;

            MicroAPI::RegTensor<T> zeroReg;
            MicroAPI::Duplicate<T>(zeroReg, static_cast<T>(0.0f));

            for (uint16_t i = 0; i < repeatTimes; i++) {
                // 搬入resLocal
                MicroAPI::DataCopy(resValueReg, resLocalPtr + i * repeatSizeT);
                mask = MicroAPI::UpdateMask<T>(maskSize);

                for (uint16_t scaleIdx = 0; scaleIdx < static_cast<uint16_t>(numScales_); scaleIdx++) {
                    // cAligned = 256, repeatTimes = 4
                    if (sizeof(T) == 4) {
                        MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B32>(
                            bilinearWeighthh, bilinearWeighPtr + scaleIdx * 4 + 0);
                        MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B32>(
                            bilinearWeighthl, bilinearWeighPtr + scaleIdx * 4 + 1);
                        MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B32>(
                            bilinearWeightlh, bilinearWeighPtr + scaleIdx * 4 + 2);
                        MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B32>(
                            bilinearWeightll, bilinearWeighPtr + scaleIdx * 4 + 3);
                    } else {
                        MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B16>(
                            bilinearWeighthh, bilinearWeighPtr + scaleIdx * 4 + 0);
                        MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B16>(
                            bilinearWeighthl, bilinearWeighPtr + scaleIdx * 4 + 1);
                        MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B16>(
                            bilinearWeightlh, bilinearWeighPtr + scaleIdx * 4 + 2);
                        MicroAPI::DataCopy<T, LoadDist::DIST_BRC_B16>(
                            bilinearWeightll, bilinearWeighPtr + scaleIdx * 4 + 3);
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
    TBuf<TPosition::VECCALC> usedFeatBuf_, validPointBuf_;
    LocalTensor<int32_t> usedFeatOffset_;
    LocalTensor<uint8_t> validPoint_, validPointMask_;

    uint32_t basePtr_, realPtr_;
    uint32_t coreNum_, curBlockIdx_;
    uint32_t taskNum_, taskNumPerCore_, startOffset_, endOffset_;
    uint32_t weightBufSize_, scaleStartBufSize_, spatialShapeBufSize_, weightMulBufSize_, vLocalBufSize_;
    uint32_t bs_, numFeats_, numEmbeds_, numAnchors_, numPoints_, numCams_, numScales_, numGroups_, numChannels_,
        cAligned_;
    uint32_t vecAlignFLoat_, vecAlignInt_, repeatAlignFloat_, blockAlignFloat_, vecAlignInt64_;
    uint32_t featByteSize_;
    uint64_t ubSize_;
    uint32_t bilinearWeightSize_, locationSize_, resLocalSize_, usedFeatBufSize_, taskCount_;
    int32_t outerLoopTimes_;

    uint32_t bufIdx_ = 0;
    uint32_t srcShape_[2], dstShape_[2];
};

extern "C" __global__ __aicore__ void deformable_aggregation(GM_ADDR mc_ms_feat, GM_ADDR spatial_shape,
    GM_ADDR scale_start_index, GM_ADDR sampling_location, GM_ADDR weights, GM_ADDR out, GM_ADDR workspace,
    GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    KernelDeformableAggregation<DTYPE_MC_MS_FEAT> op;
    op.Init(mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights, out, &tiling_data, &pipe);
    op.GetLocalTensor();
    op.Process();
}
