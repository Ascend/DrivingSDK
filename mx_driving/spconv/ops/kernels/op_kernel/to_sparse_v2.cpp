/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#include "kernel_operator.h"
using namespace AscendC;

namespace {
constexpr static int32_t BUFFER_NUM = 1;
};

class KernelToSparseV2 {
public:
    __aicore__ inline KernelToSparseV2() {}
    __aicore__ inline void Init(GM_ADDR features, GM_ADDR weight, GM_ADDR indices_offset, GM_ADDR former_sorted_indices, GM_ADDR indices, GM_ADDR sparse_value, GM_ADDR sparse_indices, ToSparseV2TilingData tiling_data, TPipe *tmpPipe)
    {
        pipe = tmpPipe;
        curBlockIdx = GetBlockIdx();
        initTilingData(&tiling_data);
        uint64_t beginOffset = curBlockIdx * coreTask;

        valueBlockNum = blockBytes / sizeof(DTYPE_FEATURES);
        idxBlockNum = blockBytes / sizeof(DTYPE_INDICES);
        kernelICAlign = AlignUp(kernelIC, valueBlockNum);
        kernelSizeAlign = AlignUp(kernelSize, valueBlockNum);

        repeatOffset = repeatBlockByte / sizeof(DTYPE_FEATURES);
        kernelICBlock = kernelICAlign / valueBlockNum;
        wholeRedusumMask = kernelICAlign;
        if (wholeRedusumMask > repeatOffset) {
            wholeRedusumMask = repeatOffset;
        }

        if (curBlockIdx < usedCoreNum - 1) {
            coreRepeatTimes = repeatTimes;
            coreMoveTail = moveTail;
        } else {
            coreRepeatTimes = lastRepeatTimes;
            coreMoveTail = lastMoveTail;
        }

        featuresGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_FEATURES *>(features));
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_FEATURES *>(weight));
        indicesOffsetGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES *>(indices_offset) + beginOffset);
        formerSortedIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES *>(former_sorted_indices));
        indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES *>(indices));

        sparseValueGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_FEATURES *>(sparse_value) + beginOffset * kernelOC);
        sparseIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES *>(sparse_indices) + beginOffset * 8);

        pipe->InitBuffer(weightQueue, BUFFER_NUM, kernelOneLen * kernelOC * kernelICAlign * sizeof(DTYPE_FEATURES));
        pipe->InitBuffer(mulTmpUB, kernelOC * kernelICAlign * sizeof(DTYPE_FEATURES));
        pipe->InitBuffer(featrueQueue, BUFFER_NUM, kernelICAlign * sizeof(DTYPE_FEATURES));
        pipe->InitBuffer(featureTmpUB, kernelOC * sizeof(DTYPE_FEATURES));

        pipe->InitBuffer(indicesOffsetQueue, BUFFER_NUM, AlignUp(moveLen + 1, idxBlockNum) * sizeof(DTYPE_INDICES));
        pipe->InitBuffer(formerSortedIndicesQueue, BUFFER_NUM, moveLen * kernelSizeAlign * sizeof(DTYPE_INDICES));
        pipe->InitBuffer(indicesQueue, BUFFER_NUM, moveLen * 8 * sizeof(DTYPE_INDICES));
        pipe->InitBuffer(sumTmpUB, moveLen * kernelOC * sizeof(DTYPE_FEATURES));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < coreRepeatTimes; i++) {
            Compute(i);
            pipe_barrier(PIPE_ALL);
        }
    }

private:
    __aicore__ inline void initTilingData(ToSparseV2TilingData *tiling_data)
    {
        usedCoreNum = tiling_data->usedCoreNum;
        coreTask = tiling_data->coreTask;
        lastCoreTask = tiling_data->lastCoreTask;

        moveLen = tiling_data->moveLen;

        repeatTimes = tiling_data->repeatTimes;
        moveTail = tiling_data->moveTail;
        lastRepeatTimes = tiling_data->lastRepeatTimes;
        lastMoveTail = tiling_data->lastMoveTail;
        kernelIC = tiling_data->kernelIC;
        kernelOC = tiling_data->kernelOC;
        kernelSize = tiling_data->kernelSize;
        kernelOneLen = tiling_data->kernelOneLen;
        kernelRepeateTimes = tiling_data->kernelRepeateTimes;
        kernelLastLen = tiling_data->kernelLastLen;
    }

    __aicore__ inline void Compute(uint32_t query)
    {
        uint32_t taskOffset = query * moveLen;
        uint32_t forMoveLen = moveLen;
        if (query == coreRepeatTimes - 1) {
            forMoveLen = coreMoveTail;
        }

        DataCopyExtParams indicesOffsetCopyParams {1, (uint32_t)((forMoveLen + 1) * sizeof(DTYPE_INDICES)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_INDICES> indicesOffsetPadParams{true, 0, 0, 0};

        DataCopyExtParams indicesCopyParams {1, (uint32_t)(4 * sizeof(DTYPE_INDICES)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_INDICES> indicesPadParams{true, 0, 0, 0};

        DataCopyExtParams featureCopyParams {1, (uint32_t)(kernelIC * sizeof(DTYPE_FEATURES)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_FEATURES> featureCopyPadParams{true, 0, 0, 0};

        DataCopyExtParams outIndicesCopyParams {1, (uint32_t)(forMoveLen * 8 * sizeof(DTYPE_INDICES)), 0, 0, 0};
        DataCopyExtParams sumCopyParams {1, (uint32_t)(forMoveLen * kernelOC * sizeof(DTYPE_FEATURES)), 0, 0, 0};

        event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        event_t eventIDMTE2ToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

        LocalTensor<DTYPE_INDICES> indicesOffsetLocal = indicesOffsetQueue.AllocTensor<DTYPE_INDICES>();
        LocalTensor<DTYPE_INDICES> sortLocal = formerSortedIndicesQueue.AllocTensor<DTYPE_INDICES>();
        LocalTensor<DTYPE_INDICES> indicesLocal = indicesQueue.AllocTensor<DTYPE_INDICES>();

        LocalTensor<DTYPE_FEATURES> featureLocal = featrueQueue.AllocTensor<DTYPE_FEATURES>();
        LocalTensor<DTYPE_FEATURES> weightLocal = weightQueue.AllocTensor<DTYPE_FEATURES>();

        LocalTensor<DTYPE_FEATURES> mulTemp = mulTmpUB.Get<DTYPE_FEATURES>();
        LocalTensor<DTYPE_FEATURES> featureTmpLocal = featureTmpUB.Get<DTYPE_FEATURES>();
        LocalTensor<DTYPE_FEATURES> sumValueLocal = sumTmpUB.Get<DTYPE_FEATURES>();

        uint32_t srcShape_[2] = {1, kernelICAlign};
        uint32_t dstShape_[2] = {kernelOC, kernelICAlign};

        DTYPE_FEATURES zeroVal = 0.0;
        Duplicate<DTYPE_FEATURES>(sumValueLocal, zeroVal, moveLen * kernelOC);

        SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
        WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
        DataCopyPad(indicesOffsetLocal, indicesOffsetGm[taskOffset], indicesOffsetCopyParams, indicesOffsetPadParams);
        pipe_barrier(PIPE_MTE2);

        DataCopyPadExtParams<DTYPE_INDICES> sortPadParams{true, 0, 0, 0};

        for (uint32_t idx = 0; idx < forMoveLen; idx++) {
            uint32_t beginIndicesOffset = indicesOffsetLocal.GetValue(idx);
            uint32_t endIndicesOffset = indicesOffsetLocal.GetValue(idx + 1);
            DataCopyExtParams sortCopyParams {1, (uint32_t)((endIndicesOffset - beginIndicesOffset) * sizeof(DTYPE_INDICES_OFFSET)), 0, 0, 0};
            DataCopyPad(sortLocal[idx * kernelSizeAlign], formerSortedIndicesGm[beginIndicesOffset], sortCopyParams, sortPadParams);
        }

        SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        for (uint32_t idx = 0; idx < forMoveLen; idx++) {
            uint32_t sortOffset = sortLocal.GetValue(idx * kernelSizeAlign);
            DataCopyPad(indicesLocal[idx * 8], indicesGm[sortOffset * 4], indicesCopyParams, indicesPadParams);
        }
        SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
        WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
        DataCopyPad(sparseIndicesGm[taskOffset * 8], indicesLocal, outIndicesCopyParams);

        uint32_t kernelMoveLen = kernelOneLen;

        for (uint32_t kernelMove = 0; kernelMove < kernelRepeateTimes; kernelMove++) {
            if (kernelMove == kernelRepeateTimes - 1) {
                kernelMoveLen = kernelLastLen;
            }
            uint32_t kernelBegin = kernelMove * kernelOneLen;
            uint32_t kernelEnd = kernelBegin + kernelMoveLen;
            DataCopyExtParams weightCopyParams {(uint16_t)(kernelMoveLen * kernelOC), (uint32_t)(kernelIC * sizeof(DTYPE_WEIGHT)), 0, 0, 0};
            DataCopyPadExtParams<DTYPE_WEIGHT> weightCopyPadParams{true, 0, 0, 0};
            DataCopyPad(weightLocal, weightGm[kernelBegin * kernelOC * kernelIC], weightCopyParams, weightCopyPadParams);
            for (uint32_t idx = 0; idx < forMoveLen; idx++) {
                uint32_t beginIndicesOffset = indicesOffsetLocal.GetValue(idx);
                uint32_t endIndicesOffset = indicesOffsetLocal.GetValue(idx + 1);
                for (uint32_t j = 0; j < endIndicesOffset - beginIndicesOffset; j++) {
                    SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
                    uint32_t sortOffset = sortLocal.GetValue(j + idx * kernelSizeAlign);
                    uint32_t weightOffset = sortOffset % kernelSize;
                    if (weightOffset >= kernelBegin && weightOffset < kernelEnd) {
                        WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
                        uint32_t featureOffset = sortOffset / kernelSize;
                        DataCopyPad(featureLocal, featuresGm[featureOffset * kernelIC], featureCopyParams, featureCopyPadParams);
                        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                        BroadCast<DTYPE_FEATURES, 2, 0>(mulTemp, featureLocal, dstShape_, srcShape_);
                        SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
                        Mul(mulTemp, mulTemp, weightLocal[(weightOffset - kernelBegin) * kernelOC * kernelICAlign], kernelOC * kernelICAlign);
                        if (kernelICAlign > repeatOffset) {
                            WholeReduceSum(featureTmpLocal, mulTemp[repeatOffset], kernelICAlign - repeatOffset, kernelOC, 1, 1, kernelICBlock);
                            Add(sumValueLocal[idx * kernelOC], sumValueLocal[idx * kernelOC], featureTmpLocal, kernelOC);
                        }
                        WholeReduceSum(featureTmpLocal, mulTemp, wholeRedusumMask, kernelOC, 1, 1, kernelICBlock);
                        Add(sumValueLocal[idx * kernelOC], sumValueLocal[idx * kernelOC], featureTmpLocal, kernelOC);
                    }
                    WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
                }
                pipe_barrier(PIPE_ALL);
            }
            SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
            WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
        }
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        DataCopyPad(sparseValueGm[taskOffset * kernelOC], sumValueLocal, sumCopyParams);
        weightQueue.FreeTensor(weightLocal);
        indicesOffsetQueue.FreeTensor(indicesOffsetLocal);
        formerSortedIndicesQueue.FreeTensor(sortLocal);
        indicesQueue.FreeTensor(indicesLocal);
        featrueQueue.FreeTensor(featureLocal);
    }

private:
    TPipe *pipe;
    GlobalTensor<DTYPE_FEATURES> featuresGm, weightGm, sparseValueGm;
    GlobalTensor<DTYPE_INDICES> indicesOffsetGm, formerSortedIndicesGm, indicesGm, sparseIndicesGm;
    TQue<QuePosition::VECIN, 1> indicesOffsetQueue, formerSortedIndicesQueue, featrueQueue, weightQueue, indicesQueue;
    TBuf<TPosition::VECCALC> mulTmpUB, workUB, reduceSumUB, featureTmpUB, sumTmpUB;

    uint32_t usedCoreNum;
    uint32_t coreTask;
    uint32_t lastCoreTask;

    uint32_t moveLen;

    uint32_t repeatTimes;
    uint32_t moveTail;
    uint32_t lastRepeatTimes;
    uint32_t lastMoveTail;
    uint32_t kernelIC;
    uint32_t kernelOC;
    uint32_t kernelSize;

    uint32_t blockBytes{32};
    uint32_t repeatBlockByte{256};
    uint32_t curBlockIdx;
    uint32_t coreRepeatTimes;
    uint32_t coreMoveTail;
    uint32_t kernelOneLen;
    uint32_t kernelRepeateTimes;
    uint32_t kernelLastLen;

    uint32_t kernelICAlign;
    uint32_t kernelSizeAlign;
    uint32_t valueBlockNum;
    uint32_t idxBlockNum;

    uint32_t kernelICBlock;
    uint32_t repeatOffset;
    uint32_t wholeRedusumMask;
};

extern "C" __global__ __aicore__ void to_sparse_v2(GM_ADDR features, GM_ADDR weight, GM_ADDR indices_offset, GM_ADDR former_sorted_indices, GM_ADDR indices, GM_ADDR sparse_value, GM_ADDR sparse_indices, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    KernelToSparseV2 op;
    op.Init(features, weight, indices_offset, former_sorted_indices, indices, sparse_value, sparse_indices, tiling_data, &pipe);
    op.Process();
}