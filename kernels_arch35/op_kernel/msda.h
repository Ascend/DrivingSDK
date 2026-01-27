/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 */
#ifndef MSDA_H
#define MSDA_H

#include "kernel_utils.h"
#include "kernel_operator.h"
#include "kernel_tpipe_impl.h"

using namespace AscendC;
using namespace MicroAPI;

template<typename T, typename U>
__aicore__ inline void ComputeGmOffsetVF(uint16_t taskRpt_, uint32_t numHeads_, uint32_t embedDims_,
        uint32_t baseOffset, uint32_t nextOffset, uint32_t baseCount, const LocalTensor<T> locationFloat,
        const LocalTensor<T> shapeFloat, const LocalTensor<U> offsetInt, const LocalTensor<U> gmOffset)
{
    __local_mem__ T* locationFloatPtr = (__local_mem__ T*) locationFloat.GetPhyAddr();
    __local_mem__ T* locationInputsPtr = (__local_mem__ T*) locationFloat[2 * taskRpt_ * B32_DATA_NUM_PER_REPEAT].GetPhyAddr();
    __local_mem__ T* shapeFloatPtr = (__local_mem__ T*) shapeFloat.GetPhyAddr();
    __local_mem__ U* offsetIntPtr = (__local_mem__ U*) offsetInt.GetPhyAddr();
    __local_mem__ U* gmOffsetPtr = (__local_mem__ U*) gmOffset.GetPhyAddr();

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<T> locationXY1Reg;
        MicroAPI::RegTensor<T> locationXY2Reg;
        MicroAPI::RegTensor<T> locationXReg;
        MicroAPI::RegTensor<T> locationYReg;
        MicroAPI::RegTensor<T> widthFloatReg;
        MicroAPI::RegTensor<T> heightFloatReg;
        MicroAPI::RegTensor<T> constOffsetReg;

        MicroAPI::RegTensor<U> offsetReg;
        MicroAPI::RegTensor<U> widthIntReg;
        MicroAPI::RegTensor<U> locationXIntReg;
        MicroAPI::RegTensor<U> locationYIntReg;
        MicroAPI::RegTensor<U> gmOffsetReg;
        MicroAPI::RegTensor<U> baseOffsetReg;

        MicroAPI::MaskReg mask = MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();

        static constexpr AscendC::MicroAPI::CastTrait castF2ITrait = 
            {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};
        static constexpr AscendC::MicroAPI::CastTrait castI2FTrait = 
            {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        Duplicate(constOffsetReg, -0.5, mask);

        uint32_t taskOffset_ = taskRpt_ * B32_DATA_NUM_PER_REPEAT;
        for (uint16_t taskIdx = 0; taskIdx < taskRpt_; ++taskIdx) {
            uint32_t localOffset = taskIdx * B32_DATA_NUM_PER_REPEAT;
            MicroAPI::DataCopy(locationXY1Reg, locationInputsPtr + 2 * localOffset);
            MicroAPI::DataCopy(locationXY2Reg, locationInputsPtr + 2 * localOffset + B32_DATA_NUM_PER_REPEAT);
            MicroAPI::DataCopy(widthFloatReg, shapeFloatPtr + localOffset);
            MicroAPI::DataCopy(heightFloatReg, shapeFloatPtr + localOffset + taskOffset_);
            MicroAPI::DataCopy(offsetReg, offsetIntPtr + localOffset);

            MicroAPI::DeInterleave(locationXReg, locationYReg, locationXY1Reg, locationXY2Reg);
            MicroAPI::FusedMulDstAdd(locationXReg, widthFloatReg, constOffsetReg, mask);
            MicroAPI::FusedMulDstAdd(locationYReg, heightFloatReg, constOffsetReg, mask);

            MicroAPI::DataCopy(locationFloatPtr + localOffset, locationXReg, mask);
            MicroAPI::DataCopy(locationFloatPtr + localOffset + taskOffset_, locationYReg, mask);

            MicroAPI::Cast<U, T, castF2ITrait>(widthIntReg, widthFloatReg, mask);
            MicroAPI::Cast<U, T, castF2ITrait>(locationXIntReg, locationXReg, mask);
            MicroAPI::Cast<U, T, castF2ITrait>(locationYIntReg, locationYReg, mask);

            MicroAPI::Mul(gmOffsetReg, locationYIntReg, widthIntReg, mask);
            MicroAPI::Add(gmOffsetReg, gmOffsetReg, locationXIntReg, mask);
            MicroAPI::Muls(gmOffsetReg, gmOffsetReg, numHeads_, mask);
            MicroAPI::Add(gmOffsetReg, gmOffsetReg, offsetReg, mask);

            MicroAPI::MaskReg baseMask = MicroAPI::UpdateMask<T>(baseCount);
            MicroAPI::Duplicate<U, MaskMergeMode::MERGING>(baseOffsetReg, nextOffset, mask);
            MicroAPI::Duplicate<U, MaskMergeMode::MERGING>(baseOffsetReg, baseOffset, baseMask);
            MicroAPI::Add(gmOffsetReg, gmOffsetReg, baseOffsetReg, mask);
            MicroAPI::Muls(gmOffsetReg, gmOffsetReg, embedDims_, mask);

            MicroAPI::DataCopy(gmOffsetPtr + localOffset, gmOffsetReg, mask);
            MicroAPI::Muls(offsetReg, widthIntReg, numHeads_ * embedDims_, mask);
            MicroAPI::Add(gmOffsetReg, gmOffsetReg, offsetReg, mask);
            MicroAPI::DataCopy(gmOffsetPtr + localOffset + taskOffset_, gmOffsetReg, mask);
        }
    }
}
#endif // MSDA_H