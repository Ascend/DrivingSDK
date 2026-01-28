/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef DEFORMABLE_CONV2D_UTILS_TILING_H
#define DEFORMABLE_CONV2D_UTILS_TILING_H
#include "kernel_operator.h"

using namespace AscendC;
using namespace MicroAPI;

constexpr int32_t INT32_BYTE_SIZE = 4;
constexpr int32_t UB_BLOCK_BYTE_SIZE = 32;
constexpr int32_t VEC_LENGTH = 256;

__aicore__ inline void InitConstLocalVf(LocalTensor<int32_t>& innerKernelHeightOffsetLocal, LocalTensor<int32_t>& innerKernelWidthOffsetLocal, int32_t kh, int32_t kw, int32_t count)
{
    __local_mem__ int32_t* innerKernelHeightoffsetPtr = (__local_mem__ int32_t*) innerKernelHeightOffsetLocal.GetPhyAddr();
    __local_mem__ int32_t* innerKernelWidthoffsetPtr = (__local_mem__ int32_t*) innerKernelWidthOffsetLocal.GetPhyAddr();

    __VEC_SCOPE__ {
        RegTensor<float> kHReg;
        RegTensor<float> kWReg;
        RegTensor<float> tmpReg;

        RegTensor<int32_t> kHRegInt32;
        RegTensor<int32_t> kWRegInt32;

        MaskReg maskReg = CreateMask<float, MaskPattern::ALL>();
        MaskReg maskRegInt32 = CreateMask<int32_t, MaskPattern::ALL>();

        MicroAPI::Arange(kWReg, 0);
        MicroAPI::Muls(tmpReg, kWReg, static_cast<float>(1.0f / kw), maskReg);
        MicroAPI::Truncate<float, RoundMode::CAST_FLOOR, MaskMergeMode::ZEROING>(kHReg, tmpReg, maskReg);

        MicroAPI::Muls(tmpReg, kHReg, static_cast<float>(kw), maskReg);
        MicroAPI::Sub(kWReg, kWReg, tmpReg, maskReg);

        static constexpr MicroAPI::CastTrait castTrait = 
            {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};
        MicroAPI::Cast<int32_t, float, castTrait>(kHRegInt32, kHReg, maskReg);
        MicroAPI::Cast<int32_t, float, castTrait>(kWRegInt32, kWReg, maskReg);

        MicroAPI::DataCopy(innerKernelHeightoffsetPtr, kHRegInt32, maskRegInt32);
        MicroAPI::DataCopy(innerKernelWidthoffsetPtr, kWRegInt32, maskRegInt32);
    }
}


template<typename T>
__aicore__ inline void interpolationVF(LocalTensor<T> outputFeaturesLocal, LocalTensor<T> topLeftFeaturesLocal, LocalTensor<T> topRightFeaturesLocal,
        LocalTensor<T> bottomLeftFeaturesLocal, LocalTensor<T> bottomRightFeaturesLocal, LocalTensor<T> fracHeightLocal, LocalTensor<T> fracWidthLocal,
        LocalTensor<T> pointWeightLocal, int32_t inChannelsAligned, int32_t kh, int32_t kw, uint16_t repeatTimes, uint16_t vecElementsCount)
{
    __local_mem__ T* pointWeightPtr = (__local_mem__ T*) pointWeightLocal.GetPhyAddr();
    __local_mem__ T* topLeftFeaturesPtr = (__local_mem__ T*) topLeftFeaturesLocal.GetPhyAddr();
    __local_mem__ T* topRightFeaturesPtr = (__local_mem__ T*) topRightFeaturesLocal.GetPhyAddr();
    __local_mem__ T* bottomLeftFeaturesPtr = (__local_mem__ T*) bottomLeftFeaturesLocal.GetPhyAddr();
    __local_mem__ T* bottomRightFeaturesPtr = (__local_mem__ T*) bottomRightFeaturesLocal.GetPhyAddr();
    __local_mem__ T* outputFeaturesPtr = (__local_mem__ T*) outputFeaturesLocal.GetPhyAddr();
    __local_mem__ T* fracHeightPtr = (__local_mem__ T*) fracHeightLocal.GetPhyAddr();
    __local_mem__ T* fracWeightPtr = (__local_mem__ T*) fracWidthLocal.GetPhyAddr();

    uint16_t kernelSize = kw * kh;

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<T> bottomWeightReg;
        MicroAPI::RegTensor<T> rightWeightReg;
        MicroAPI::RegTensor<T> topWeightReg;
        MicroAPI::RegTensor<T> leftWeightReg;

        MicroAPI::RegTensor<T> topLeftFeaturesReg;
        MicroAPI::RegTensor<T> topRightFeaturesReg;
        MicroAPI::RegTensor<T> bottomLeftFeaturesReg;
        MicroAPI::RegTensor<T> bottomRightFeaturesReg;

        MicroAPI::RegTensor<T> topLeftWeightReg;
        MicroAPI::RegTensor<T> topRightWeightReg;
        MicroAPI::RegTensor<T> bottomLeftWeightReg;
        MicroAPI::RegTensor<T> bottomRightWeightReg;

        MicroAPI::RegTensor<T> pointWeightReg;
        
        MicroAPI::RegTensor<T> outputFeatureReg;
        
        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        MicroAPI::MaskReg cmpMaskReg;

        for (uint16_t k = 0; k < kernelSize; k++) {
            
            if (std::is_same<T, float>::value) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(bottomWeightReg, fracHeightPtr + k);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(rightWeightReg, fracWeightPtr + k);
                
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(pointWeightReg, pointWeightPtr + k);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B16>(bottomWeightReg, fracHeightPtr + k);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B16>(rightWeightReg, fracWeightPtr + k);
                
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B16>(pointWeightReg, pointWeightPtr + k);
            }
            
            MicroAPI::Muls(topWeightReg, bottomWeightReg, -1, mask);
            MicroAPI::Adds(topWeightReg, topWeightReg, 1, mask);
            MicroAPI::Muls(leftWeightReg, rightWeightReg, -1, mask);
            MicroAPI::Adds(leftWeightReg, leftWeightReg, 1, mask);

            MicroAPI::Mul(topLeftWeightReg, topWeightReg, leftWeightReg, mask);
            MicroAPI::Mul(topRightWeightReg, topWeightReg, rightWeightReg, mask);
            MicroAPI::Mul(bottomLeftWeightReg, bottomWeightReg, leftWeightReg, mask);
            MicroAPI::Mul(bottomRightWeightReg, bottomWeightReg, rightWeightReg, mask);
            
            for (uint16_t i = 0; i < repeatTimes; i++) {
                MicroAPI::DataCopy(topLeftFeaturesReg, topLeftFeaturesPtr + (repeatTimes * k + i) * vecElementsCount);
                MicroAPI::DataCopy(topRightFeaturesReg, topRightFeaturesPtr + (repeatTimes * k + i) * vecElementsCount);
                MicroAPI::DataCopy(bottomLeftFeaturesReg, bottomLeftFeaturesPtr + (repeatTimes * k + i) * vecElementsCount);
                MicroAPI::DataCopy(bottomRightFeaturesReg, bottomRightFeaturesPtr + (repeatTimes * k + i) * vecElementsCount);

                MicroAPI::Mul(outputFeatureReg, topLeftFeaturesReg, topLeftWeightReg, mask);
                MicroAPI::MulAddDst(outputFeatureReg, topRightFeaturesReg, topRightWeightReg, mask);
                MicroAPI::MulAddDst(outputFeatureReg, bottomLeftFeaturesReg, bottomLeftWeightReg, mask);
                MicroAPI::MulAddDst(outputFeatureReg, bottomRightFeaturesReg, bottomRightWeightReg, mask);

                MicroAPI::Mul(outputFeatureReg, outputFeatureReg, pointWeightReg, mask);

                MicroAPI::DataCopy(outputFeaturesPtr + (repeatTimes * k + i) * vecElementsCount, outputFeatureReg, mask);
            }
        }
    }
}


template<typename T>
__aicore__ inline void ComputeOffsetAndWeightVf(LocalTensor<int32_t> topLeftOffsetLocal, LocalTensor<int32_t> topRightOffsetLocal, LocalTensor<int32_t> bottomLeftOffsetLocal, LocalTensor<int32_t> bottomRightOffsetLocal,
        LocalTensor<T> fracHeightLocal, LocalTensor<T> fracWidthLocal, LocalTensor<T> inputOffsetLocal, LocalTensor<int32_t>& innerKernelHeightOffsetLocal, LocalTensor<int32_t>& innerKernelWidthOffsetLocal,
        int32_t taskOffset, int32_t featureMapSize, int32_t outHeightSize, int32_t outWidthSize, int32_t kh, int32_t kw)
{
    __local_mem__ int32_t* topLeftOffsetPtr = (__local_mem__ int32_t*) topLeftOffsetLocal.GetPhyAddr();
    __local_mem__ int32_t* topRightOffsetPtr = (__local_mem__ int32_t*) topRightOffsetLocal.GetPhyAddr();
    __local_mem__ int32_t* bottomLeftOffsetPtr = (__local_mem__ int32_t*) bottomLeftOffsetLocal.GetPhyAddr();
    __local_mem__ int32_t* bottomRightOffsetPtr = (__local_mem__ int32_t*) bottomRightOffsetLocal.GetPhyAddr();
    __local_mem__ T* fracHeightPtr = (__local_mem__ T*) fracHeightLocal.GetPhyAddr();
    __local_mem__ T* fracWeightPtr = (__local_mem__ T*) fracWidthLocal.GetPhyAddr();
    __local_mem__ T* inputOffsetPtr = (__local_mem__ T*) inputOffsetLocal.GetPhyAddr();
    __local_mem__ int32_t* innerKernelHeightOffsetPtr = (__local_mem__ int32_t*) innerKernelHeightOffsetLocal.GetPhyAddr();
    __local_mem__ int32_t* innerKernelWidthOffsetPtr = (__local_mem__ int32_t*) innerKernelWidthOffsetLocal.GetPhyAddr();
    
    int32_t batchIdx = taskOffset / featureMapSize;
    int32_t hOutIdx = (taskOffset % (featureMapSize)) / outWidthSize;
    int32_t wOutIdx = taskOffset % outWidthSize;
    int32_t batchOffset = batchIdx * featureMapSize;
    int32_t hOffset = hOutIdx - kh / 2;
    int32_t wOffset = wOutIdx - kw / 2;

    __VEC_SCOPE__  {
        MicroAPI::RegTensor<T> vReg0;
        MicroAPI::RegTensor<T> vReg1;
        MicroAPI::RegTensor<T> vReg2;
        MicroAPI::RegTensor<T> vReg3;
        MicroAPI::RegTensor<T> vReg4;
        MicroAPI::RegTensor<T> vReg5;

        MicroAPI::RegTensor<int32_t> topPosReg;
        MicroAPI::RegTensor<int32_t> rightPosReg;
        MicroAPI::RegTensor<int32_t> bottomPosReg;
        MicroAPI::RegTensor<int32_t> leftPosReg;

        MicroAPI::RegTensor<int32_t> topLeftReg;
        MicroAPI::RegTensor<int32_t> topRightReg;
        MicroAPI::RegTensor<int32_t> bottomLeftReg;
        MicroAPI::RegTensor<int32_t> bottomRightReg;
        MicroAPI::RegTensor<int32_t> vConstReg0; // for const innerKHOffset
        MicroAPI::RegTensor<int32_t> vConstReg1; // for const innerKWOffset
        MicroAPI::RegTensor<int32_t> vConstReg2; // for const value -1

        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        MaskReg maskInt32 = MicroAPI::CreateMask<int32_t, MaskPattern::ALL>();
        MicroAPI::MaskReg topPosCmpMaskReg;
        MicroAPI::MaskReg leftPosCmpMaskReg;
        MicroAPI::MaskReg bottomPosCmpMaskReg;
        MicroAPI::MaskReg rightPosCmpMaskReg;
        MicroAPI::MaskReg topLeftCmpMaskReg;
        MicroAPI::MaskReg topRightCmpMaskReg;
        MicroAPI::MaskReg bottomLeftCmpMaskReg;
        MicroAPI::MaskReg bottomRightCmpMaskReg;

        MicroAPI::DataCopy(vConstReg0, innerKernelHeightOffsetPtr);
        MicroAPI::DataCopy(vConstReg1, innerKernelWidthOffsetPtr);
        MicroAPI::DataCopy(vReg0, inputOffsetPtr);
        MicroAPI::Duplicate(vConstReg2, -1, mask);

        MicroAPI::DeInterleave(vReg2, vReg3, vReg0, vReg1); // vReg0: xOffset, vReg1: yOffset
        
        MicroAPI::Truncate<T, RoundMode::CAST_FLOOR, MaskMergeMode::ZEROING>(vReg4, vReg2, mask);  // vReg2: topPos
        MicroAPI::Truncate<T, RoundMode::CAST_FLOOR, MaskMergeMode::ZEROING>(vReg5, vReg3, mask);  // vReg3: leftPos

        MicroAPI::Sub(vReg2, vReg2, vReg4, mask);  // vReg0: fracH
        MicroAPI::Sub(vReg3, vReg3, vReg5, mask);  // vReg1: fracW

        // 搬出 weight
        MicroAPI::DataCopy(fracHeightPtr, vReg2, mask);
        MicroAPI::DataCopy(fracWeightPtr, vReg3, mask);

        if (std::is_same<T, float>::value) {
            static constexpr MicroAPI::CastTrait castTrait0 =
                {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};
            MicroAPI::Cast<int32_t, T, castTrait0>(topPosReg, vReg4, mask);
            static constexpr MicroAPI::CastTrait castTrait1 =
                {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};
            MicroAPI::Cast<int32_t, T, castTrait1>(leftPosReg, vReg5, mask);
        } else {
            static constexpr MicroAPI::CastTrait castTrait0 =
                {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};
            MicroAPI::Cast<int32_t, T, castTrait0>(topPosReg, vReg0, mask);
            static constexpr MicroAPI::CastTrait castTrait1 =
                {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};
            MicroAPI::Cast<int32_t, T, castTrait1>(leftPosReg, vReg0, mask);
        }
        
        MicroAPI::Add(topPosReg, topPosReg, vConstReg0, maskInt32);
        MicroAPI::Add(leftPosReg, leftPosReg, vConstReg1, maskInt32);
        MicroAPI::Adds(topPosReg, topPosReg, hOffset, maskInt32);     // topPos + hOutIdx - kH_ / 2
        MicroAPI::Adds(leftPosReg, leftPosReg, wOffset, maskInt32);     // leftPos + wOutIdx - kW_ / 2
        
        MicroAPI::Adds(bottomPosReg, topPosReg, 1, maskInt32);     // leftPos + wOutIdx - kW_ / 2
        MicroAPI::Adds(rightPosReg, leftPosReg, 1, maskInt32);     // topPos + hOutIdx - kH_ / 2
        
        MicroAPI::CompareScalar<int32_t, CMPMODE::GE>(topPosCmpMaskReg, topPosReg, 0, maskInt32);
        MicroAPI::CompareScalar<int32_t, CMPMODE::GE>(leftPosCmpMaskReg, leftPosReg, 0, maskInt32);
        MicroAPI::CompareScalar<int32_t, CMPMODE::GE>(rightPosCmpMaskReg, rightPosReg, 0, maskInt32);
        MicroAPI::CompareScalar<int32_t, CMPMODE::GE>(bottomPosCmpMaskReg, bottomPosReg, 0, maskInt32);

        MicroAPI::CompareScalar<int32_t, CMPMODE::LT>(topLeftCmpMaskReg, topPosReg, outHeightSize, maskInt32);
        MicroAPI::CompareScalar<int32_t, CMPMODE::LT>(topRightCmpMaskReg, leftPosReg, outWidthSize, maskInt32);
        MicroAPI::CompareScalar<int32_t, CMPMODE::LT>(bottomLeftCmpMaskReg, rightPosReg, outWidthSize, maskInt32);
        MicroAPI::CompareScalar<int32_t, CMPMODE::LT>(bottomRightCmpMaskReg, bottomPosReg, outHeightSize, maskInt32);

        // 0 <= topPosReg < H, 0 <= leftPosReg < W, 0 <= rightPosReg < W, 0 <= bottomPosReg < H
        MicroAPI::MaskAnd(topPosCmpMaskReg, topLeftCmpMaskReg, topPosCmpMaskReg, maskInt32);
        MicroAPI::MaskAnd(leftPosCmpMaskReg, topRightCmpMaskReg, leftPosCmpMaskReg, maskInt32);
        MicroAPI::MaskAnd(rightPosCmpMaskReg, bottomLeftCmpMaskReg, rightPosCmpMaskReg, maskInt32);
        MicroAPI::MaskAnd(bottomPosCmpMaskReg, bottomRightCmpMaskReg, bottomPosCmpMaskReg, maskInt32);

        // topleftMask = topMask & leftMask, toprightMask = topMask & rightMask, bottomleftMask = bottomMask & leftMask, bottomrightMask = bottomMask & rightMask,
        MicroAPI::MaskAnd(topLeftCmpMaskReg, topPosCmpMaskReg, leftPosCmpMaskReg, maskInt32);
        MicroAPI::MaskAnd(topRightCmpMaskReg, topPosCmpMaskReg, rightPosCmpMaskReg, maskInt32);
        MicroAPI::MaskAnd(bottomLeftCmpMaskReg, bottomPosCmpMaskReg, leftPosCmpMaskReg, maskInt32);
        MicroAPI::MaskAnd(bottomRightCmpMaskReg, bottomPosCmpMaskReg, rightPosCmpMaskReg, maskInt32);

        MicroAPI::Muls(topPosReg, topPosReg, outWidthSize, maskInt32);
        MicroAPI::Muls(bottomPosReg, bottomPosReg, outWidthSize, maskInt32);

        MicroAPI::Adds(topPosReg, topPosReg, batchOffset, maskInt32);
        MicroAPI::Adds(bottomPosReg, bottomPosReg, batchOffset, maskInt32);

        MicroAPI::Add(topLeftReg, topPosReg, leftPosReg, maskInt32);
        MicroAPI::Select(topLeftReg, topLeftReg, vConstReg2, topLeftCmpMaskReg);
        MicroAPI::DataCopy(topLeftOffsetPtr, topLeftReg, maskInt32);

        MicroAPI::Add(topRightReg, topPosReg, rightPosReg, maskInt32);
        MicroAPI::Select(topRightReg, topRightReg, vConstReg2, topRightCmpMaskReg);
        MicroAPI::DataCopy(topRightOffsetPtr, topRightReg, maskInt32);

        MicroAPI::Add(bottomLeftReg, bottomPosReg, leftPosReg, maskInt32);
        MicroAPI::Select(bottomLeftReg, bottomLeftReg, vConstReg2, bottomLeftCmpMaskReg);
        MicroAPI::DataCopy(bottomLeftOffsetPtr, bottomLeftReg, maskInt32);
        
        MicroAPI::Add(bottomRightReg, bottomPosReg, rightPosReg, maskInt32);
        MicroAPI::Select(bottomRightReg, bottomRightReg, vConstReg2, bottomRightCmpMaskReg);
        MicroAPI::DataCopy(bottomRightOffsetPtr, bottomRightReg, maskInt32);
    }
}

template<typename T>
__aicore__ inline void CopyInFeature(const LocalTensor<int32_t>& offsetLocal, const LocalTensor<T>& featuresLocal, const GlobalTensor<T>& featuresGlobal,
    const int32_t& innerOffset, const int32_t& ubOffset, const int32_t& inChannels) {
    
    int32_t offset0 = offsetLocal.GetValue(innerOffset * 16 + 0);
    int32_t offset1 = offsetLocal.GetValue(innerOffset * 16 + 1);
    int32_t offset2 = offsetLocal.GetValue(innerOffset * 16 + 2);
    int32_t offset3 = offsetLocal.GetValue(innerOffset * 16 + 3);
    int32_t offset4 = offsetLocal.GetValue(innerOffset * 16 + 4);
    int32_t offset5 = offsetLocal.GetValue(innerOffset * 16 + 5);
    int32_t offset6 = offsetLocal.GetValue(innerOffset * 16 + 6);
    int32_t offset7 = offsetLocal.GetValue(innerOffset * 16 + 7);
    int32_t offset8 = offsetLocal.GetValue(innerOffset * 16 + 8);

    offset0 >= 0 ? DataCopy(featuresLocal[ubOffset + 0 * inChannels], featuresGlobal[offset0 * inChannels], inChannels) :
        AscendC::Duplicate(featuresLocal[ubOffset + 0 * inChannels], static_cast<T>(0), inChannels);
    offset1 >= 0 ? DataCopy(featuresLocal[ubOffset + 1 * inChannels], featuresGlobal[offset1 * inChannels], inChannels) :
        AscendC::Duplicate(featuresLocal[ubOffset + 1 * inChannels], static_cast<T>(0), inChannels);
    offset2 >= 0 ? DataCopy(featuresLocal[ubOffset + 2 * inChannels], featuresGlobal[offset2 * inChannels], inChannels) :
        AscendC::Duplicate(featuresLocal[ubOffset + 2 * inChannels], static_cast<T>(0), inChannels);
    offset3 >= 0 ? DataCopy(featuresLocal[ubOffset + 3 * inChannels], featuresGlobal[offset3 * inChannels], inChannels) :
        AscendC::Duplicate(featuresLocal[ubOffset + 3 * inChannels], static_cast<T>(0), inChannels);
    offset4 >= 0 ? DataCopy(featuresLocal[ubOffset + 4 * inChannels], featuresGlobal[offset4 * inChannels], inChannels) :
        AscendC::Duplicate(featuresLocal[ubOffset + 4 * inChannels], static_cast<T>(0), inChannels);
    offset5 >= 0 ? DataCopy(featuresLocal[ubOffset + 5 * inChannels], featuresGlobal[offset5 * inChannels], inChannels) :
        AscendC::Duplicate(featuresLocal[ubOffset + 5 * inChannels], static_cast<T>(0), inChannels);
    offset6 >= 0 ? DataCopy(featuresLocal[ubOffset + 6 * inChannels], featuresGlobal[offset6 * inChannels], inChannels) :
        AscendC::Duplicate(featuresLocal[ubOffset + 6 * inChannels], static_cast<T>(0), inChannels);
    offset7 >= 0 ? DataCopy(featuresLocal[ubOffset + 7 * inChannels], featuresGlobal[offset7 * inChannels], inChannels) :
        AscendC::Duplicate(featuresLocal[ubOffset + 7 * inChannels], static_cast<T>(0), inChannels);
    offset8 >= 0 ? DataCopy(featuresLocal[ubOffset + 8 * inChannels], featuresGlobal[offset8 * inChannels], inChannels) :
        AscendC::Duplicate(featuresLocal[ubOffset + 8 * inChannels], static_cast<T>(0), inChannels);
}

template<typename T>
__aicore__ inline void CopyInAndCopyOutFeature(const LocalTensor<int32_t>& offsetLocal, const LocalTensor<T>& inFeaturesLocal, const LocalTensor<T>& outFeaturesLocal, const GlobalTensor<T>& inputFeaturesGlobal,
    const GlobalTensor<T>& outputFeaturesGlobal, const int32_t& innerOffset, const int32_t& ubOffset, const int32_t& inChannels) {
    int32_t offset0 = offsetLocal.GetValue(innerOffset * 16 + 0);
    int32_t offset1 = offsetLocal.GetValue(innerOffset * 16 + 1);
    int32_t offset2 = offsetLocal.GetValue(innerOffset * 16 + 2);
    int32_t offset3 = offsetLocal.GetValue(innerOffset * 16 + 3);
    int32_t offset4 = offsetLocal.GetValue(innerOffset * 16 + 4);
    int32_t offset5 = offsetLocal.GetValue(innerOffset * 16 + 5);
    int32_t offset6 = offsetLocal.GetValue(innerOffset * 16 + 6);
    int32_t offset7 = offsetLocal.GetValue(innerOffset * 16 + 7);
    int32_t offset8 = offsetLocal.GetValue(innerOffset * 16 + 8);

    SetAtomicAdd<T>();
    offset0 >= 0 ? DataCopy(inFeaturesLocal[ubOffset + 0 * inChannels], inputFeaturesGlobal[offset0 * inChannels], inChannels),
        DataCopy(outputFeaturesGlobal[offset0 * inChannels], outFeaturesLocal[ubOffset + 0 * inChannels], inChannels) :
        AscendC::Duplicate(inFeaturesLocal[ubOffset + 0 * inChannels], static_cast<T>(0), inChannels);

    offset1 >= 0 ? DataCopy(inFeaturesLocal[ubOffset + 1 * inChannels], inputFeaturesGlobal[offset1 * inChannels], inChannels),
        DataCopy(outputFeaturesGlobal[offset1 * inChannels], outFeaturesLocal[ubOffset + 1 * inChannels], inChannels) :
        AscendC::Duplicate(inFeaturesLocal[ubOffset + 1 * inChannels], static_cast<T>(0), inChannels);

    offset2 >= 0 ? DataCopy(inFeaturesLocal[ubOffset + 2 * inChannels], inputFeaturesGlobal[offset2 * inChannels], inChannels),
        DataCopy(outputFeaturesGlobal[offset2 * inChannels], outFeaturesLocal[ubOffset + 2 * inChannels], inChannels) :
        AscendC::Duplicate(inFeaturesLocal[ubOffset + 2 * inChannels], static_cast<T>(0), inChannels);

    offset3 >= 0 ? DataCopy(inFeaturesLocal[ubOffset + 3 * inChannels], inputFeaturesGlobal[offset3 * inChannels], inChannels),
        DataCopy(outputFeaturesGlobal[offset3 * inChannels], outFeaturesLocal[ubOffset + 3 * inChannels], inChannels) :
        AscendC::Duplicate(inFeaturesLocal[ubOffset + 3 * inChannels], static_cast<T>(0), inChannels);

    offset4 >= 0 ? DataCopy(inFeaturesLocal[ubOffset + 4 * inChannels], inputFeaturesGlobal[offset4 * inChannels], inChannels),
        DataCopy(outputFeaturesGlobal[offset4 * inChannels], outFeaturesLocal[ubOffset + 4 * inChannels], inChannels) :
        AscendC::Duplicate(inFeaturesLocal[ubOffset + 4 * inChannels], static_cast<T>(0), inChannels);

    offset5 >= 0 ? DataCopy(inFeaturesLocal[ubOffset + 5 * inChannels], inputFeaturesGlobal[offset5 * inChannels], inChannels),
        DataCopy(outputFeaturesGlobal[offset5 * inChannels], outFeaturesLocal[ubOffset + 5 * inChannels], inChannels) :
        AscendC::Duplicate(inFeaturesLocal[ubOffset + 5 * inChannels], static_cast<T>(0), inChannels);

    offset6 >= 0 ? DataCopy(inFeaturesLocal[ubOffset + 6 * inChannels], inputFeaturesGlobal[offset6 * inChannels], inChannels),
        DataCopy(outputFeaturesGlobal[offset6 * inChannels], outFeaturesLocal[ubOffset + 6 * inChannels], inChannels) :
        AscendC::Duplicate(inFeaturesLocal[ubOffset + 6 * inChannels], static_cast<T>(0), inChannels);

    offset7 >= 0 ? DataCopy(inFeaturesLocal[ubOffset + 7 * inChannels], inputFeaturesGlobal[offset7 * inChannels], inChannels),
        DataCopy(outputFeaturesGlobal[offset7 * inChannels], outFeaturesLocal[ubOffset + 7 * inChannels], inChannels) :
        AscendC::Duplicate(inFeaturesLocal[ubOffset + 7 * inChannels], static_cast<T>(0), inChannels);

    offset8 >= 0 ? DataCopy(inFeaturesLocal[ubOffset + 8 * inChannels], inputFeaturesGlobal[offset8 * inChannels], inChannels),
        DataCopy(outputFeaturesGlobal[offset8 * inChannels], outFeaturesLocal[ubOffset + 8 * inChannels], inChannels) :
        AscendC::Duplicate(inFeaturesLocal[ubOffset + 8 * inChannels], static_cast<T>(0), inChannels);
    SetAtomicNone();
}

template<typename T>
__aicore__ inline void ComputeInputFeatureGrad(LocalTensor<T> topLeftFeaturesGradLocal, LocalTensor<T> topRightFeaturesGradLocal,LocalTensor<T> bottomLeftFeaturesGradLocal, LocalTensor<T> bottomRightFeaturesGradLocal,
    LocalTensor<T> img2colGradLocal, LocalTensor<T> fracHeightLocal, LocalTensor<T> fracWidthLocal,LocalTensor<T> pointWeightLocal, int32_t inChannelsAligned, int32_t kh, int32_t kw, uint16_t repeatTimes, uint16_t vecElementsCount)
{
    __local_mem__ T* pointWeightPtr = (__local_mem__ T*) pointWeightLocal.GetPhyAddr();
    __local_mem__ T* topLeftFeaturesGradPtr = (__local_mem__ T*) topLeftFeaturesGradLocal.GetPhyAddr();
    __local_mem__ T* topRightFeaturesGradPtr = (__local_mem__ T*) topRightFeaturesGradLocal.GetPhyAddr();
    __local_mem__ T* bottomLeftFeaturesGradPtr = (__local_mem__ T*) bottomLeftFeaturesGradLocal.GetPhyAddr();
    __local_mem__ T* bottomRightFeaturesGradPtr = (__local_mem__ T*) bottomRightFeaturesGradLocal.GetPhyAddr();

    __local_mem__ T* fracHeightPtr = (__local_mem__ T*) fracHeightLocal.GetPhyAddr();
    __local_mem__ T* fracWeightPtr = (__local_mem__ T*) fracWidthLocal.GetPhyAddr();

    __local_mem__ T* img2colGradPtr = (__local_mem__ T*) img2colGradLocal.GetPhyAddr();
    
    uint16_t kernelSize = kw * kh;

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<T> bottomWeightReg;
        MicroAPI::RegTensor<T> rightWeightReg;
        MicroAPI::RegTensor<T> topWeightReg;
        MicroAPI::RegTensor<T> leftWeightReg;

        MicroAPI::RegTensor<T> topLeftWeightReg;
        MicroAPI::RegTensor<T> topRightWeightReg;
        MicroAPI::RegTensor<T> bottomLeftWeightReg;
        MicroAPI::RegTensor<T> bottomRightWeightReg;

        MicroAPI::RegTensor<T> topLeftGradFeatureReg;
        MicroAPI::RegTensor<T> topRightGradFeatureReg;
        MicroAPI::RegTensor<T> bottomLeftGradFeatureReg;
        MicroAPI::RegTensor<T> bottomRightGradFeatureReg;

        MicroAPI::RegTensor<T> pointWeightReg;
        MicroAPI::RegTensor<T> img2colGradReg;

        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        MicroAPI::MaskReg cmpMaskReg;

        for (uint16_t k = 0; k < kernelSize; k++) {
            
            if (std::is_same<T, float>::value) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(bottomWeightReg, fracHeightPtr + k);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(rightWeightReg, fracWeightPtr + k);
                
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(pointWeightReg, pointWeightPtr + k);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B16>(bottomWeightReg, fracHeightPtr + k);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B16>(rightWeightReg, fracWeightPtr + k);
                
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B16>(pointWeightReg, pointWeightPtr + k);
            }
            
            MicroAPI::Muls(topWeightReg, bottomWeightReg, -1, mask);
            MicroAPI::Adds(topWeightReg, topWeightReg, 1, mask);
            MicroAPI::Muls(leftWeightReg, rightWeightReg, -1, mask);
            MicroAPI::Adds(leftWeightReg, leftWeightReg, 1, mask);

            MicroAPI::Mul(topLeftWeightReg, topWeightReg, leftWeightReg, mask);
            MicroAPI::Mul(topRightWeightReg, topWeightReg, rightWeightReg, mask);
            MicroAPI::Mul(bottomLeftWeightReg, bottomWeightReg, leftWeightReg, mask);
            MicroAPI::Mul(bottomRightWeightReg, bottomWeightReg, rightWeightReg, mask);

            for (uint16_t i = 0; i < repeatTimes; i++) {
                MicroAPI::DataCopy(img2colGradReg, img2colGradPtr + (repeatTimes * k + i) * vecElementsCount);

                // compute inputFeatureGrad
                MicroAPI::Mul(img2colGradReg, img2colGradReg, pointWeightReg, mask);

                MicroAPI::Mul(topLeftGradFeatureReg, img2colGradReg, topLeftWeightReg, mask);
                MicroAPI::DataCopy(topLeftFeaturesGradPtr + k * inChannelsAligned + i * vecElementsCount, topLeftGradFeatureReg, mask);

                MicroAPI::Mul(topRightGradFeatureReg, img2colGradReg, topRightWeightReg, mask);
                MicroAPI::DataCopy(topRightFeaturesGradPtr + k * inChannelsAligned + i * vecElementsCount, topRightGradFeatureReg, mask);

                MicroAPI::Mul(bottomLeftGradFeatureReg, img2colGradReg, bottomLeftWeightReg, mask);
                MicroAPI::DataCopy(bottomLeftFeaturesGradPtr + k * inChannelsAligned + i * vecElementsCount, bottomLeftGradFeatureReg, mask);

                MicroAPI::Mul(bottomRightGradFeatureReg, img2colGradReg, bottomRightWeightReg, mask);
                MicroAPI::DataCopy(bottomRightFeaturesGradPtr + k * inChannelsAligned + i * vecElementsCount, bottomRightGradFeatureReg, mask);
            }
        }
    }
}

template<typename T>
__aicore__ inline void ComputeOffsetGradAndMaskGrad(LocalTensor<T> gradOffsetLocal, LocalTensor<T> gradMaskLocal, LocalTensor<T> img2colLocal, LocalTensor<T> img2colGradLocal,
    LocalTensor<T> topLeftFeaturesLocal, LocalTensor<T> topRightFeaturesLocal,LocalTensor<T> bottomLeftFeaturesLocal, LocalTensor<T> bottomRightFeaturesLocal,
    LocalTensor<T> fracHeightLocal, LocalTensor<T> fracWidthLocal,LocalTensor<T> pointWeightLocal, int32_t inChannelsAligned, int32_t kh, int32_t kw, uint16_t repeatTimes, uint16_t vecElementsCount)
{
    __local_mem__ T* pointWeightPtr = (__local_mem__ T*) pointWeightLocal.GetPhyAddr();
    __local_mem__ T* topLeftFeaturesPtr = (__local_mem__ T*) topLeftFeaturesLocal.GetPhyAddr();
    __local_mem__ T* topRightFeaturesPtr = (__local_mem__ T*) topRightFeaturesLocal.GetPhyAddr();
    __local_mem__ T* bottomLeftFeaturesPtr = (__local_mem__ T*) bottomLeftFeaturesLocal.GetPhyAddr();
    __local_mem__ T* bottomRightFeaturesPtr = (__local_mem__ T*) bottomRightFeaturesLocal.GetPhyAddr();

    __local_mem__ T* gradXOffsetPtr = (__local_mem__ T*) gradOffsetLocal.GetPhyAddr();
    __local_mem__ T* gradMaskPtr = (__local_mem__ T*) gradMaskLocal.GetPhyAddr();

    __local_mem__ T* img2colGradPtr = (__local_mem__ T*) img2colGradLocal.GetPhyAddr();
    __local_mem__ T* img2colPtr = (__local_mem__ T*) img2colLocal.GetPhyAddr();

    __local_mem__ T* fracHeightPtr = (__local_mem__ T*) fracHeightLocal.GetPhyAddr();
    __local_mem__ T* fracWeightPtr = (__local_mem__ T*) fracWidthLocal.GetPhyAddr();

    uint16_t kernelSize = kw * kh;

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<T> bottomWeightReg;
        MicroAPI::RegTensor<T> rightWeightReg;
        MicroAPI::RegTensor<T> topWeightReg;
        MicroAPI::RegTensor<T> leftWeightReg;

        MicroAPI::RegTensor<T> negBottomWeightReg;
        MicroAPI::RegTensor<T> negRightWeightReg;
        MicroAPI::RegTensor<T> negTopWeightReg;
        MicroAPI::RegTensor<T> negLeftWeightReg;

        MicroAPI::RegTensor<T> topLeftWeightReg;
        MicroAPI::RegTensor<T> topRightWeightReg;
        MicroAPI::RegTensor<T> bottomLeftWeightReg;
        MicroAPI::RegTensor<T> bottomRightWeightReg;

        MicroAPI::RegTensor<T> topLeftFeaturesReg;
        MicroAPI::RegTensor<T> topRightFeaturesReg;
        MicroAPI::RegTensor<T> bottomLeftFeaturesReg;
        MicroAPI::RegTensor<T> bottomRightFeaturesReg;

        MicroAPI::RegTensor<T> pointWeightReg;
        
        MicroAPI::RegTensor<T> gradOffsetReg;
        MicroAPI::RegTensor<T> gradXOffsetReg;
        MicroAPI::RegTensor<T> gradYOffsetReg;
        MicroAPI::RegTensor<T> gradPointWeightReg;

        MicroAPI::RegTensor<T> img2colReg;
        MicroAPI::RegTensor<T> img2colGradReg;

        MaskReg mask = MicroAPI::CreateMask<T, MaskPattern::ALL>();
        MicroAPI::MaskReg cmpMaskReg;

        for (uint16_t k = 0; k < kernelSize; k++) {
            
            if (std::is_same<T, float>::value) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(bottomWeightReg, fracHeightPtr + k);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(rightWeightReg, fracWeightPtr + k);
                
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(pointWeightReg, pointWeightPtr + k);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B16>(bottomWeightReg, fracHeightPtr + k);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B16>(rightWeightReg, fracWeightPtr + k);
                
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B16>(pointWeightReg, pointWeightPtr + k);
            }
            
            MicroAPI::Muls(topWeightReg, bottomWeightReg, -1, mask);
            MicroAPI::Adds(topWeightReg, topWeightReg, 1, mask);
            MicroAPI::Muls(leftWeightReg, rightWeightReg, -1, mask);
            MicroAPI::Adds(leftWeightReg, leftWeightReg, 1, mask);

            MicroAPI::Muls(negLeftWeightReg, leftWeightReg, -1, mask);
            MicroAPI::Muls(negRightWeightReg, rightWeightReg, -1, mask);
            MicroAPI::Muls(negTopWeightReg, topWeightReg, -1, mask);
            MicroAPI::Muls(negBottomWeightReg, bottomWeightReg, -1, mask);

            MicroAPI::Mul(topLeftWeightReg, topWeightReg, leftWeightReg, mask);
            MicroAPI::Mul(topRightWeightReg, topWeightReg, rightWeightReg, mask);
            MicroAPI::Mul(bottomLeftWeightReg, bottomWeightReg, leftWeightReg, mask);
            MicroAPI::Mul(bottomRightWeightReg, bottomWeightReg, rightWeightReg, mask);

            for (uint16_t i = 0; i < repeatTimes; i++) {
                MicroAPI::DataCopy(topLeftFeaturesReg, topLeftFeaturesPtr + (repeatTimes * k + i) * vecElementsCount);
                MicroAPI::DataCopy(topRightFeaturesReg, topRightFeaturesPtr + (repeatTimes * k + i) * vecElementsCount);
                MicroAPI::DataCopy(bottomLeftFeaturesReg, bottomLeftFeaturesPtr + (repeatTimes * k + i) * vecElementsCount);
                MicroAPI::DataCopy(bottomRightFeaturesReg, bottomRightFeaturesPtr + (repeatTimes * k + i) * vecElementsCount);
                MicroAPI::DataCopy(img2colGradReg, img2colGradPtr + (repeatTimes * k + i) * vecElementsCount);

                // compute img2colMat
                MicroAPI::Mul(img2colReg, topLeftFeaturesReg, topLeftWeightReg, mask);
                MicroAPI::MulAddDst(img2colReg, topRightFeaturesReg, topRightWeightReg, mask);
                MicroAPI::MulAddDst(img2colReg, bottomLeftFeaturesReg, bottomLeftWeightReg, mask);
                MicroAPI::MulAddDst(img2colReg, bottomRightFeaturesReg, bottomRightWeightReg, mask);

                // compute offset
                // Bottom Right
                MicroAPI::Mul(gradXOffsetReg, bottomRightFeaturesReg, rightWeightReg, mask);
                MicroAPI::Mul(gradYOffsetReg, bottomRightFeaturesReg, bottomWeightReg, mask);

                // Bottom Left
                MicroAPI::MulAddDst(gradXOffsetReg, bottomLeftFeaturesReg, leftWeightReg, mask);
                MicroAPI::MulAddDst(gradYOffsetReg, bottomLeftFeaturesReg, negBottomWeightReg, mask);

                // top right
                MicroAPI::MulAddDst(gradXOffsetReg, topRightFeaturesReg, negRightWeightReg, mask);
                MicroAPI::MulAddDst(gradYOffsetReg, topRightFeaturesReg, topWeightReg, mask);
                
                // top left
                MicroAPI::MulAddDst(gradXOffsetReg, topLeftFeaturesReg, negLeftWeightReg, mask);
                MicroAPI::MulAddDst(gradYOffsetReg, topLeftFeaturesReg, negTopWeightReg, mask);

                MicroAPI::Mul(gradXOffsetReg, gradXOffsetReg, pointWeightReg, mask);
                MicroAPI::Mul(gradYOffsetReg, gradYOffsetReg, pointWeightReg, mask);

                MicroAPI::Mul(gradXOffsetReg, gradXOffsetReg, img2colGradReg, mask);
                MicroAPI::Mul(gradYOffsetReg, gradYOffsetReg, img2colGradReg, mask);

                MicroAPI::Mul(gradPointWeightReg, img2colReg, img2colGradReg, mask);

                MicroAPI::DataCopy(gradXOffsetPtr + k * inChannelsAligned + i * vecElementsCount, gradXOffsetReg, mask);
                MicroAPI::DataCopy(gradXOffsetPtr + (kernelSize + k) * inChannelsAligned + i * vecElementsCount, gradYOffsetReg, mask);
                MicroAPI::DataCopy(gradMaskPtr + k * inChannelsAligned + i * vecElementsCount, gradPointWeightReg, mask);
                MicroAPI::Mul(img2colReg, img2colReg, pointWeightReg, mask);
                MicroAPI::DataCopy(img2colPtr + inChannelsAligned * k + i * vecElementsCount, img2colReg, mask);
            }
        }
    }
}

#endif // DEFORMABLE_CONV2D_UTILS_TILING_H