/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#include "kernel_operator.h"
using namespace AscendC;

namespace {
constexpr static int32_t BUFFER_NUM = 1;
};

class KernelToSparse {
public:
    __aicore__ inline KernelToSparse() {}
    __aicore__ inline void Init(GM_ADDR indices_offset, GM_ADDR value, GM_ADDR former_sorted_indices, GM_ADDR indices, GM_ADDR sparse_value, GM_ADDR sparse_indices, ToSparseTilingData *tiling_data, TPipe *tmpPipe)
    {
        pipe = tmpPipe;
        curBlockIdx = GetBlockIdx();
        initTilingData(tiling_data);
        uint64_t beginOffset = curBlockIdx * coreTask;

        uint32_t valueBlockNum = blockBytes / sizeof(DTYPE_VALUE);
        uint32_t idxBlockNum = blockBytes / sizeof(DTYPE_INDICES);

        if (curBlockIdx < usedCoreNum - 1) {
            coreRepeatTimes = repeatTimes;
            coreMoveTail = moveTail;
        } else {
            coreRepeatTimes = lastRepeatTimes;
            coreMoveTail = lastMoveTail;
        }

        indicesOffsetGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES *>(indices_offset) + beginOffset);
        formerSortedIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES *>(former_sorted_indices));
        valueGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(value));
        indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES *>(indices));

        sparseValueGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_VALUE *>(sparse_value) + beginOffset * outChannels);
        sparseIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES *>(sparse_indices) + beginOffset * 8);

        pipe->InitBuffer(indicesOffsetQueue, BUFFER_NUM, AlignUp(moveLen + 1, idxBlockNum) * sizeof(DTYPE_INDICES));
        pipe->InitBuffer(formerSortedIndicesQueue, BUFFER_NUM, AlignUp(27, idxBlockNum) * sizeof(DTYPE_INDICES));
        pipe->InitBuffer(valueQueue, BUFFER_NUM, AlignUp(outChannels, valueBlockNum) * sizeof(DTYPE_VALUE));
        pipe->InitBuffer(indicesQueue, BUFFER_NUM, AlignUp(moveLen * 8, idxBlockNum) * sizeof(DTYPE_INDICES));
        pipe->InitBuffer(sumTmpUB, AlignUp(moveLen * outChannels, valueBlockNum) * sizeof(DTYPE_VALUE));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < coreRepeatTimes; i++) {
            Compute(i);
            pipe_barrier(PIPE_ALL);
        }
    }

private:
    __aicore__ inline void initTilingData(ToSparseTilingData *tiling_data)
    {
        usedCoreNum = tiling_data->usedCoreNum;
        coreTask = tiling_data->coreTask;
        lastCoreTask = tiling_data->lastCoreTask;

        moveLen = tiling_data->moveLen;

        repeatTimes = tiling_data->repeatTimes;
        moveTail = tiling_data->moveTail;
        lastRepeatTimes = tiling_data->lastRepeatTimes;
        lastMoveTail = tiling_data->lastMoveTail;
        outChannels = tiling_data->outChannels;
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
        DataCopyExtParams valueCopyParams {1, (uint32_t)(outChannels * sizeof(DTYPE_VALUE)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_VALUE> valuePadParams{true, 0, 0, 0};

        DataCopyExtParams outIndicesCopyParams {1, (uint32_t)(forMoveLen * 8 * sizeof(DTYPE_INDICES)), 0, 0, 0};
        DataCopyExtParams sumCopyParams {1, (uint32_t)(forMoveLen * outChannels * sizeof(DTYPE_VALUE)), 0, 0, 0};

        event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

        LocalTensor<DTYPE_INDICES> indicesOffsetLocal = indicesOffsetQueue.AllocTensor<DTYPE_INDICES>();
        LocalTensor<DTYPE_INDICES> sortLocal = formerSortedIndicesQueue.AllocTensor<DTYPE_INDICES>();
        LocalTensor<DTYPE_INDICES> indicesLocal = indicesQueue.AllocTensor<DTYPE_INDICES>();
        LocalTensor<DTYPE_VALUE> valueLocal = valueQueue.AllocTensor<DTYPE_VALUE>();
        LocalTensor<DTYPE_VALUE> sumValueLocal = sumTmpUB.Get<DTYPE_VALUE>();

        DTYPE_VALUE zeroVal = 0.0;
        Duplicate<DTYPE_VALUE>(sumValueLocal, zeroVal, moveLen * outChannels);

        SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
        WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
        DataCopyPad(indicesOffsetLocal, indicesOffsetGm[taskOffset], indicesOffsetCopyParams, indicesOffsetPadParams);
        pipe_barrier(PIPE_MTE2);

        for (uint32_t i = 0; i < forMoveLen; i++) {
            uint32_t beginIndicesOffset = indicesOffsetLocal.GetValue(i);
            uint32_t endIndicesOffset = indicesOffsetLocal.GetValue(i + 1);
            DataCopyExtParams sortCopyParams {1, (uint32_t)((endIndicesOffset - beginIndicesOffset) * sizeof(DTYPE_INDICES)), 0, 0, 0};
            DataCopyPadExtParams<DTYPE_INDICES> sortPadParams{true, 0, 0, 0};
            SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
            WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
            DataCopyPad(sortLocal, formerSortedIndicesGm[beginIndicesOffset], sortCopyParams, sortPadParams);
            SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            uint32_t sortOffset = sortLocal.GetValue(0);
            SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
            WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
            DataCopyPad(indicesLocal[i * 8], indicesGm[sortOffset * 4], indicesCopyParams, indicesPadParams);
            for (uint32_t j = 0; j < endIndicesOffset - beginIndicesOffset; j++) {
                sortOffset = sortLocal.GetValue(j);
                SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
                WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
                DataCopyPad(valueLocal, valueGm[sortOffset * outChannels], valueCopyParams, valuePadParams);
                SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                Add(sumValueLocal[i * outChannels], sumValueLocal[i * outChannels], valueLocal, outChannels);
                SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
                WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
            }
        }
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        DataCopyPad(sparseIndicesGm[taskOffset * 8], indicesLocal, outIndicesCopyParams);
        DataCopyPad(sparseValueGm[taskOffset * outChannels], sumValueLocal, sumCopyParams);

        indicesOffsetQueue.FreeTensor(indicesOffsetLocal);
        formerSortedIndicesQueue.FreeTensor(sortLocal);
        indicesQueue.FreeTensor(indicesLocal);
        valueQueue.FreeTensor(valueLocal);
    }

private:
    TPipe *pipe;
    GlobalTensor<DTYPE_VALUE> valueGm, sparseValueGm;
    GlobalTensor<DTYPE_INDICES> indicesOffsetGm, formerSortedIndicesGm, indicesGm, sparseIndicesGm;
    TQue<QuePosition::VECIN, 1> indicesOffsetQueue, formerSortedIndicesQueue, valueQueue, indicesQueue;
    TBuf<TPosition::VECCALC> sumTmpUB;

    uint32_t usedCoreNum;
    uint32_t coreTask;
    uint32_t lastCoreTask;

    uint32_t moveLen;

    uint32_t repeatTimes;
    uint32_t moveTail;
    uint32_t lastRepeatTimes;
    uint32_t lastMoveTail;
    uint32_t outChannels;

    uint32_t blockBytes{32};
    uint32_t curBlockIdx;
    uint32_t coreRepeatTimes;
    uint32_t coreMoveTail;
};

extern "C" __global__ __aicore__ void to_sparse(GM_ADDR indices_offset, GM_ADDR value, GM_ADDR former_sorted_indices, GM_ADDR indices, GM_ADDR sparse_value, GM_ADDR sparse_indices, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    KernelToSparse op;
    op.Init(indices_offset, value, former_sorted_indices, indices, sparse_value, sparse_indices, &tiling_data, &pipe);
    op.Process();
}