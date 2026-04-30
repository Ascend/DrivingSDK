/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 */
#ifndef MSDA_H
#define MSDA_H

#include "kernel_operator.h"

using namespace AscendC;
using namespace MicroAPI;

template<typename T, typename U>
__aicore__ inline void ComputeGmOffsetVF(uint16_t taskRpt_, uint32_t numHeads_, uint32_t embedDims_,
        uint32_t baseOffset, uint32_t nextOffset, uint32_t baseCount, const LocalTensor<T> locationFloat,
        const LocalTensor<T> shapeFloat, const LocalTensor<U> offsetInt, const LocalTensor<U> gmOffset,
        const LocalTensor<U> validMaskTensor)
{
    __local_mem__ T* locationFloatPtr = (__local_mem__ T*) locationFloat.GetPhyAddr();
    __local_mem__ T* locationInputsPtr = (__local_mem__ T*) locationFloat[2 * taskRpt_ * B32_DATA_NUM_PER_REPEAT].GetPhyAddr();
    __local_mem__ T* shapeFloatPtr = (__local_mem__ T*) shapeFloat.GetPhyAddr();
    __local_mem__ U* offsetIntPtr = (__local_mem__ U*) offsetInt.GetPhyAddr();
    __local_mem__ U* gmOffsetPtr = (__local_mem__ U*) gmOffset.GetPhyAddr();
    __local_mem__ U* validMaskPtr = (__local_mem__ U*) validMaskTensor.GetPhyAddr();

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<T> locationXY1Reg, locationXY2Reg, shapeInput1Reg, shapeInput2Reg;
        MicroAPI::RegTensor<T> locationXReg, locationYReg, widthFloatReg, heightFloatReg;

        MicroAPI::RegTensor<U> offsetReg;
        MicroAPI::RegTensor<U> widthIntReg;
        MicroAPI::RegTensor<U> locationXIntReg, locationYIntReg;
        MicroAPI::RegTensor<U> gmOffset1Reg, gmOffset2Reg;
        MicroAPI::RegTensor<U> baseOffsetReg;

        MicroAPI::RegTensor<U> validMaskReg;
        MicroAPI::RegTensor<U> bilinearValidPoint1Reg, bilinearValidPoint2Reg, bilinearValidPoint3Reg, bilinearValidPoint4Reg;

        MicroAPI::RegTensor<T> constOffsetReg;
        MicroAPI::RegTensor<U> zeroReg;

        MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();

        static constexpr AscendC::MicroAPI::CastTrait castF2ITrait = 
            {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};
        static constexpr AscendC::MicroAPI::CastTrait castI2FTrait = 
            {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        Duplicate(constOffsetReg, -0.5, mask);
        Duplicate(zeroReg, 0, mask);

        uint32_t taskOffset_ = taskRpt_ * B32_DATA_NUM_PER_REPEAT;
        for (uint16_t taskIdx = 0; taskIdx < taskRpt_; ++taskIdx) {
            uint32_t localOffset = taskIdx * B32_DATA_NUM_PER_REPEAT;
            MicroAPI::DataCopy(locationXY1Reg, locationInputsPtr + 2 * localOffset);
            MicroAPI::DataCopy(locationXY2Reg, locationInputsPtr + 2 * localOffset + B32_DATA_NUM_PER_REPEAT);
            MicroAPI::DataCopy(shapeInput1Reg, shapeFloatPtr + 2 * localOffset);
            MicroAPI::DataCopy(shapeInput2Reg, shapeFloatPtr + 2 * localOffset + B32_DATA_NUM_PER_REPEAT);
            MicroAPI::DataCopy(offsetReg, offsetIntPtr + localOffset);

            MicroAPI::DeInterleave(locationXReg, locationYReg, locationXY1Reg, locationXY2Reg);
            MicroAPI::DeInterleave(widthFloatReg, heightFloatReg, shapeInput1Reg, shapeInput2Reg);
            MicroAPI::FusedMulDstAdd(locationXReg, widthFloatReg, constOffsetReg, mask);
            MicroAPI::FusedMulDstAdd(locationYReg, heightFloatReg, constOffsetReg, mask);
            
            MicroAPI::Interleave(locationXY1Reg, locationXY2Reg, locationXReg, locationYReg);
            MicroAPI::DataCopy(locationFloatPtr + 2 * localOffset, locationXY1Reg, mask);
            MicroAPI::DataCopy(locationFloatPtr + 2 * localOffset + B32_DATA_NUM_PER_REPEAT, locationXY2Reg, mask);

            MicroAPI::Cast<U, T, castF2ITrait>(widthIntReg, widthFloatReg, mask);
            MicroAPI::Cast<U, T, castF2ITrait>(locationXIntReg, locationXReg, mask);
            MicroAPI::Cast<U, T, castF2ITrait>(locationYIntReg, locationYReg, mask);

            MicroAPI::Mul(gmOffset1Reg, locationYIntReg, widthIntReg, mask);
            MicroAPI::Add(gmOffset1Reg, gmOffset1Reg, locationXIntReg, mask);
            MicroAPI::Muls(gmOffset1Reg, gmOffset1Reg, numHeads_, mask);
            MicroAPI::Add(gmOffset1Reg, gmOffset1Reg, offsetReg, mask);

            MicroAPI::MaskReg baseMask = MicroAPI::UpdateMask<T>(baseCount);
            MicroAPI::Duplicate<U, MaskMergeMode::MERGING>(baseOffsetReg, nextOffset, mask);
            MicroAPI::Duplicate<U, MaskMergeMode::MERGING>(baseOffsetReg, baseOffset, baseMask);
            MicroAPI::Add(gmOffset1Reg, gmOffset1Reg, baseOffsetReg, mask);
            MicroAPI::Muls(gmOffset1Reg, gmOffset1Reg, embedDims_, mask);
            MicroAPI::Muls(offsetReg, widthIntReg, numHeads_ * embedDims_, mask);
            MicroAPI::Add(gmOffset2Reg, gmOffset1Reg, offsetReg, mask);
            MicroAPI::Interleave(locationXIntReg, locationYIntReg, gmOffset1Reg, gmOffset2Reg);
            MicroAPI::DataCopy(gmOffsetPtr + 2 * localOffset, locationXIntReg, mask);
            MicroAPI::DataCopy(gmOffsetPtr + 2 * localOffset + B32_DATA_NUM_PER_REPEAT, locationYIntReg, mask);

            MicroAPI::MaskReg validMask, tmpMask;
            MicroAPI::Compares<T, CMPMODE::GT>(validMask, locationXReg, -1.0f, mask);
            MicroAPI::Compares<T, CMPMODE::GT>(tmpMask, locationYReg, -1.0f, mask);
            MicroAPI::And(validMask, validMask, tmpMask, mask);
            MicroAPI::Compare<T, CMPMODE::LT>(tmpMask, locationXReg, widthFloatReg, mask);
            MicroAPI::And(validMask, validMask, tmpMask, mask);
            MicroAPI::Compare<T, CMPMODE::LT>(tmpMask, locationYReg, heightFloatReg, mask);
            MicroAPI::And(validMask, validMask, tmpMask, mask);

            MicroAPI::MaskReg leftMask, rightMask, bottomMask, topMask;
            MicroAPI::Adds(widthFloatReg, widthFloatReg, -1.0f, mask);
            MicroAPI::Adds(heightFloatReg, heightFloatReg, -1.0f, mask);
            MicroAPI::Compares<T, CMPMODE::GT>(leftMask, locationXReg, 0.0f, mask);
            MicroAPI::Compares<T, CMPMODE::GT>(bottomMask, locationYReg, 0.0f, mask);
            MicroAPI::Compare<T, CMPMODE::LT>(rightMask, locationXReg, widthFloatReg, mask);
            MicroAPI::Compare<T, CMPMODE::LT>(topMask, locationYReg, heightFloatReg, mask);

            MicroAPI::Duplicate(bilinearValidPoint1Reg, 1, leftMask);
            MicroAPI::Select(bilinearValidPoint1Reg, bilinearValidPoint1Reg, zeroReg, bottomMask);
            MicroAPI::Duplicate(bilinearValidPoint2Reg, 2, rightMask);
            MicroAPI::Select(bilinearValidPoint2Reg, bilinearValidPoint2Reg, zeroReg, bottomMask);
            MicroAPI::Duplicate(bilinearValidPoint3Reg, 4, leftMask);
            MicroAPI::Select(bilinearValidPoint3Reg, bilinearValidPoint3Reg, zeroReg, topMask);
            MicroAPI::Duplicate(bilinearValidPoint4Reg, 8, rightMask);
            MicroAPI::Select(bilinearValidPoint4Reg, bilinearValidPoint4Reg, zeroReg, topMask);
            MicroAPI::Add(bilinearValidPoint1Reg, bilinearValidPoint1Reg, bilinearValidPoint2Reg, mask);
            MicroAPI::Add(bilinearValidPoint3Reg, bilinearValidPoint3Reg, bilinearValidPoint4Reg, mask);
            MicroAPI::Add(validMaskReg, bilinearValidPoint1Reg, bilinearValidPoint3Reg, mask);
            MicroAPI::Not(tmpMask, validMask, mask);
            MicroAPI::Duplicate<U, MaskMergeMode::MERGING>(validMaskReg, 0, tmpMask);

            MicroAPI::DataCopy(validMaskPtr + localOffset, validMaskReg, mask);
        }
    }
}
#endif // MSDA_H