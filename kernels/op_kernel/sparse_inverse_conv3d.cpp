/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#include "kernel_operator.h"
using namespace AscendC;

namespace {
constexpr uint32_t BYTE_SIZE_PER_BLOCK = 32;
constexpr uint32_t NUM_BUFFER = 2;
}; // namespace

class SparseInverseConv3dKernel {
public:
    __aicore__ inline SparseInverseConv3dKernel() {}
    __aicore__ inline void Init(TPipe* pipe, GM_ADDR features, GM_ADDR indices_offset, GM_ADDR former_sorted_indices,
        GM_ADDR output_img2col, GM_ADDR workspace, SparseInverseConv3dTilingData* tiling_data)
    {
        curBlockIdx = GetBlockIdx();
        InitTilingData(tiling_data);

        valueBlockNum = BYTE_SIZE_PER_BLOCK / sizeof(DTYPE_FEATURES);
        idxBlockNum = BYTE_SIZE_PER_BLOCK / sizeof(DTYPE_INDICES);
        uint64_t beginOffset = curBlockIdx * (uint64_t)vectorCoreTask;

        featuresGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_FEATURES*>(features) + (uint64_t)beginOffset * inChannel);
        indicesOffsetGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES*>(indices_offset) + beginOffset);
        formerSortedIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES*>(former_sorted_indices));
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_FEATURES*>(output_img2col));

        uint32_t moveLenAligned = AlignUp(moveLen + 1, idxBlockNum);
        uint32_t moveLenAlignedSize = moveLenAligned * sizeof(DTYPE_INDICES);
        uint32_t sortLenAligned = AlignUp(moveLen * kernelSize, idxBlockNum);
        uint32_t sortLenAlignedSize = sortLenAligned * sizeof(DTYPE_INDICES);
        uint32_t featureLenAligned = AlignUp(moveLen * inChannel, valueBlockNum);
        uint32_t featureLenAlignedSize = featureLenAligned * sizeof(DTYPE_FEATURES);
        pipe->InitBuffer(indicesOffsetBuf, NUM_BUFFER * moveLenAlignedSize);
        pipe->InitBuffer(formerSortedIndicesBuf, NUM_BUFFER * sortLenAlignedSize);
        pipe->InitBuffer(featureBuf, NUM_BUFFER * featureLenAlignedSize);
        indicesOffsetLocal = indicesOffsetBuf.Get<DTYPE_INDICES>();
        sortLocal = formerSortedIndicesBuf.Get<DTYPE_INDICES>();
        featureLocal = featureBuf.Get<DTYPE_FEATURES>();
        uniqueOffset[0] = 0;
        uniqueOffset[1] = moveLenAligned;
        sortOffset[0] = 0;
        sortOffset[1] = sortLenAligned;
        featureOffset[0] = 0;
        featureOffset[1] = featureLenAligned;
    }

    __aicore__ inline void Process()
    {
        if (curBlockIdx < usedVectorCoreNum) {
            CopyInAndToGm();
        }
    }

private:
    __aicore__ inline void InitTilingData(SparseInverseConv3dTilingData* tiling_data)
    {
        usedVectorCoreNum = GetBlockNum();
        inChannel = tiling_data->inChannel;
        kernelSize = tiling_data->kernelSize;
        moveLen = tiling_data->moveLen;
        vectorCoreTask = tiling_data->vectorCoreTask;
        vectorLastCoreTask = tiling_data->vectorLastCoreTask;
        coreRepeatTimes = tiling_data->coreRepeatTimes;
        coreMoveLenTail = tiling_data->coreMoveLenTail;
        lastCoreRepeatTimes = tiling_data->lastCoreRepeatTimes;
        lastCoreMoveLenTail = tiling_data->lastCoreMoveLenTail;
    }

    __aicore__ inline void CopyInAndToGm()
    {
        uint32_t repeatTimes = coreRepeatTimes;
        uint32_t movelenTail = coreMoveLenTail;
        if (curBlockIdx == usedVectorCoreNum - 1) { // tail core
            repeatTimes = lastCoreRepeatTimes;
            movelenTail = lastCoreMoveLenTail;
        }
        uint16_t pingpong = 0;
        uint32_t indicesCopyBlockNum = DivCeil((moveLen + 1) * sizeof(DTYPE_INDICES), BYTE_SIZE_PER_BLOCK);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
        for (uint32_t computeRound = 0; computeRound < repeatTimes; computeRound++) {
            uint64_t repeatBeginOffset = computeRound * moveLen;
            uint32_t realMoveLen = moveLen;
            if (computeRound == repeatTimes - 1) {
                realMoveLen = movelenTail;
                indicesCopyBlockNum = DivCeil((realMoveLen + 1) * sizeof(DTYPE_INDICES), BYTE_SIZE_PER_BLOCK);
            }
            // 1.copyin
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(pingpong);
            DataCopy(featureLocal[featureOffset[pingpong]], featuresGm[repeatBeginOffset * inChannel],
                realMoveLen * inChannel);
            DataCopy(indicesOffsetLocal[uniqueOffset[pingpong]], indicesOffsetGm[repeatBeginOffset],
                {1, static_cast<uint16_t>(indicesCopyBlockNum), 0, 0});
            uint32_t beginIndicesOffset = indicesOffsetLocal[uniqueOffset[pingpong]].GetValue(0);
            uint32_t endIndicesOffset = indicesOffsetLocal[uniqueOffset[pingpong]].GetValue(realMoveLen);
            DataCopy(sortLocal[sortOffset[pingpong]], formerSortedIndicesGm[beginIndicesOffset],
                {1,
                    static_cast<uint16_t>(
                        DivCeil((endIndicesOffset - beginIndicesOffset) * sizeof(DTYPE_INDICES), BYTE_SIZE_PER_BLOCK)),
                    0, 0});

            // 2.copyout: img2col
            for (uint32_t i = 0; i < realMoveLen; i++) {
                uint32_t beginIndicesOffsetCur =
                    indicesOffsetLocal[uniqueOffset[pingpong]].GetValue(i) - beginIndicesOffset;
                uint32_t endIndicesOffsetCur =
                    indicesOffsetLocal[uniqueOffset[pingpong]].GetValue(i + 1) - beginIndicesOffset;
                for (uint32_t j = beginIndicesOffsetCur; j < endIndicesOffsetCur; j++) {
                    uint64_t globalTaskOffset = sortLocal[sortOffset[pingpong]].GetValue(j);
                    DataCopy(outputGm[globalTaskOffset * inChannel],
                        featureLocal[featureOffset[pingpong]][i * inChannel], inChannel);
                }
            }
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(pingpong);
            pingpong = 1 - pingpong;
        }
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    }

private:
    GlobalTensor<DTYPE_FEATURES> featuresGm, outputGm;
    GlobalTensor<DTYPE_INDICES> indicesOffsetGm, formerSortedIndicesGm;
    LocalTensor<DTYPE_INDICES> sortLocal, indicesOffsetLocal;
    LocalTensor<DTYPE_FEATURES> featureLocal;
    TBuf<TPosition::VECCALC> indicesOffsetBuf, formerSortedIndicesBuf, featureBuf;

    uint32_t curBlockIdx;
    uint32_t valueBlockNum;
    uint32_t idxBlockNum;

    uint32_t usedVectorCoreNum;
    uint32_t inChannel;
    uint32_t kernelSize;
    uint32_t moveLen;
    uint32_t vectorCoreTask;
    uint32_t vectorLastCoreTask;
    uint32_t coreRepeatTimes;
    uint32_t coreMoveLenTail;
    uint32_t lastCoreRepeatTimes;
    uint32_t lastCoreMoveLenTail;

    uint32_t uniqueOffset[2];
    uint32_t sortOffset[2];
    uint32_t featureOffset[2];
};

extern "C" __global__ __aicore__ void sparse_inverse_conv3d(GM_ADDR features, GM_ADDR indices,
    GM_ADDR unique_indices_offset, GM_ADDR sorted_idx_to_former_indices, GM_ADDR output_img2col, GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (!GetSysWorkSpacePtr()) {
        return;
    }
    TPipe pipe;
    SparseInverseConv3dKernel op;
    op.Init(
        &pipe, features, unique_indices_offset, sorted_idx_to_former_indices, output_img2col, workspace, &tiling_data);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    op.Process();
}