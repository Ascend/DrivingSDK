/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 */
#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t BLOCK_SIZE = 32;

class KernelScatterMaxWithArgmaxV2 {
public:
    __aicore__ inline KernelScatterMaxWithArgmaxV2() {}
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR out, GM_ADDR argmax, ScatterMaxTilingData *tiling_data, TPipe* tmpPipe)
    {
        pipe = tmpPipe;
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        usedCoreNum = tiling_data->usedCoreNum;
        tilingMode = tiling_data->tilingMode;
        outTailNum = tiling_data->outTailNum;
        outEachCore = tiling_data->outEachCore;
        indicesNum = tiling_data->indicesNum;
        updatesNum = tiling_data->updatesNum;
        outNum = tiling_data->outNum;
        ubIndicesNum = tiling_data->ubIndicesNum;
        ubUpdatesNum = tiling_data->ubUpdatesNum;
        indicesLoop = tiling_data->indicesLoop;
        indicesLastNum = tiling_data->indicesLastNum;
        unpdatesLastNum = tiling_data->unpdatesLastNum;
        updatesTail = tiling_data->updatesTail;
        argmaxGap = tiling_data->argmaxGap;
        initArgmax = tiling_data->initArgmax;
        isAligned = tiling_data->isAligned;
        outTaskNum = tiling_data->outLineEachTask * outTailNum;
        taskNum = tiling_data->taskNumPerCore;
        outLastTaskNum = tiling_data->outeachCoreLastNum;
        outTaskLine = tiling_data->outLineEachTask;

        curBlockIdx = GetBlockIdx();
        outEachLine = outEachCore / outTailNum;
        // last Core
        if ((usedCoreNum == curBlockIdx + 1) && (tiling_data->taskNumLastCore != 0)) {
            taskNum = tiling_data->taskNumLastCore;
            outLastTaskNum = tiling_data->outLastCoreLastNum;
        }

        outrepeat = AlignUp(outTaskNum / updatesTail, dataEachBlock);
        repeatTimes = (updatesTail + dataEachBlock - 1) / dataEachBlock;
        updatesTailCopy = AlignUp(updatesTail, dataEachBlock);

        indicesEachBlock = BLOCK_SIZE / sizeof(DTYPE_INDICES);
        dataEachBlock = BLOCK_SIZE / sizeof(DTYPE_UPDATES);

        eventIdMte2ToV_0 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdMte2ToV_1 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_V>());
        eventIdVToMte3_0 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdVToMte3_1 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdMte3ToV_0 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());
        eventIdMte3ToMte2_0 = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_MTE2>());
        eventIdSToV_0 = static_cast<event_t>(pipe->AllocEventID<HardEvent::S_V>());

        varGm.SetGlobalBuffer((__gm__ DTYPE_VAR*)var, outNum);
        indicesGm.SetGlobalBuffer((__gm__ DTYPE_INDICES*)indices, indicesNum);
        updatesGm.SetGlobalBuffer((__gm__ DTYPE_UPDATES*)updates, updatesNum);
        outGm.SetGlobalBuffer((__gm__ DTYPE_OUT*)out, outNum);
        argmaxGm.SetGlobalBuffer((__gm__ DTYPE_ARGMAX*)argmax, outNum);
    
        pipe->InitBuffer(inQueueIndices, AlignUp(ubIndicesNum, indicesEachBlock) * sizeof(DTYPE_INDICES));
        pipe->InitBuffer(inQueueUpdates, updatesTailCopy * sizeof(DTYPE_UPDATES));
        pipe->InitBuffer(outQueueArgmax, updatesTailCopy * sizeof(DTYPE_ARGMAX));
        pipe->InitBuffer(inQueueVar, updatesTailCopy * sizeof(DTYPE_VAR));
        pipe->InitBuffer(outQueueOut, updatesTailCopy * sizeof(DTYPE_OUT));
        pipe->InitBuffer(tempArgmaxFloat, updatesTailCopy * sizeof(float));
        pipe->InitBuffer(tempArgmaxFloat2, updatesTailCopy * sizeof(float));
        pipe->InitBuffer(outSetNote, outrepeat * sizeof(int32_t));
        pipe->InitBuffer(mask1Buf, updatesTailCopy * sizeof(DTYPE_UPDATES));
    }
    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < taskNum; i++) {
            CopyIn(i);
            Compute(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        outThisNum = outTaskNum;
        if ((progress == taskNum - 1) && (outLastTaskNum != 0)) {
            outThisNum = outLastTaskNum;
        }
        outThisLine = outThisNum / outTailNum; // DataNum in this task
        varLocalTemp = inQueueVar.Get<DTYPE_VAR>();
        outLocalTemp = outQueueOut.Get<DTYPE_OUT>();
        argmaxLocalTemp = outQueueArgmax.Get<DTYPE_ARGMAX>();
    }

    __aicore__ inline void CopyParamasInit(const uint32_t calCount)
    {
        copyParamsOut.blockCount = 1;
        copyParamsOut.blockLen = static_cast<uint32_t>(calCount * sizeof(float)); // float 或者 int32
        copyParamsOut.srcStride = 0;
        copyParamsOut.dstStride = 0;
        copyParamsOut.rsv = 0;
    }

    template <typename T>
    __aicore__ inline void ComputeDataCopy(const GlobalTensor<T>& dstLocal, const LocalTensor<T>& srcGlobal, const uint32_t calCount)
    {
        if (isAligned) {
            DataCopy(dstLocal, srcGlobal, calCount);
        } else {
            CopyParamasInit(calCount);
            DataCopyPad(dstLocal, srcGlobal, copyParamsOut);
        }
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<int32_t>localSetNote = outSetNote.Get<int32_t>();
        LocalTensor<uint8_t> mask1Tensor = mask1Buf.Get<uint8_t>();
        LocalTensor<float> tempArgmaxFloatT = tempArgmaxFloat.Get<float>();
        LocalTensor<float> tempArgmaxFloatT2 = tempArgmaxFloat2.Get<float>();
        Duplicate(localSetNote, 0, outrepeat);

        for (uint32_t loop = 0; loop < indicesLoop; loop++) {
            LocalTensor<DTYPE_INDICES>indicesLocal = inQueueIndices.Get<DTYPE_INDICES>();
            LocalTensor<DTYPE_UPDATES>updatesLocal = inQueueUpdates.Get<DTYPE_UPDATES>();
            DataCopy(indicesLocal, indicesGm[loop * ubIndicesNum], (ubIndicesNum + indicesEachBlock - 1) / indicesEachBlock * indicesEachBlock);

            int64_t indicesStart = curBlockIdx * outEachLine + outTaskLine * progress;
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);
            for (uint32_t idx = 0; idx < ubIndicesNum; idx++) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
                WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);
                DTYPE_INDICES dataInIndices = indicesLocal.GetValue(idx);
                // if this indices should be processed in this task
                if (dataInIndices >= indicesStart && dataInIndices < indicesStart + outThisLine) {
                    int64_t outBlockOffset = dataInIndices - indicesStart;
                    int64_t outLineOffset = idx % (outTailNum / updatesTail);
                    int64_t outActualOffset = outBlockOffset * outTailNum + outLineOffset * updatesTail;
                    int64_t updatesOffset = idx * updatesTail;
                    DataCopy(updatesLocal, updatesGm[loop * ubUpdatesNum + updatesOffset], updatesTailCopy);
                    // offset in localSetNote
                    int64_t localSetNoteOffset = dataInIndices % (outTaskNum / updatesTail);
                    DTYPE_INDICES argmaxValue = (idx + ubIndicesNum * loop) / argmaxGap;
                    SetFlag<HardEvent::S_V>(eventIdSToV_0);

                    int64_t offsetInOut = curBlockIdx * outEachCore + progress * outTaskNum + outActualOffset;
                    if (localSetNote.GetValue(localSetNoteOffset) == 0) {
                        localSetNote.SetValue(localSetNoteOffset, 1);
                        BinaryRepeatParams repeatParams = {1, 1, 1, 8, 8, 8};

                        DataCopy(argmaxLocalTemp, argmaxGm[offsetInOut], updatesTailCopy);
                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);

                        DataCopy(outLocalTemp, varGm[offsetInOut], updatesTailCopy);

                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);
                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);

                        Compare(mask1Tensor, updatesLocal, outLocalTemp, CMPMODE::LT, 8, repeatTimes, repeatParams);
                        Select(outLocalTemp, mask1Tensor, outLocalTemp, updatesLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, updatesTail);

                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);

                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);
                        Cast(tempArgmaxFloatT, argmaxLocalTemp, RoundMode::CAST_ROUND, updatesTail);
                        WaitFlag<HardEvent::S_V>(eventIdSToV_0);
                        Duplicate(tempArgmaxFloatT2, (float)argmaxValue, updatesTail);

                        Select(tempArgmaxFloatT, mask1Tensor, tempArgmaxFloatT, tempArgmaxFloatT2, SELMODE::VSEL_TENSOR_TENSOR_MODE, updatesTail);
                        Cast(argmaxLocalTemp, tempArgmaxFloatT, RoundMode::CAST_ROUND, updatesTail);
                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_1);

                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);
                        ComputeDataCopy<float>(outGm[offsetInOut], outLocalTemp, updatesTail);

                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_1);
                        ComputeDataCopy<int32_t>(argmaxGm[offsetInOut], argmaxLocalTemp, updatesTail);
                    } else {
                        BinaryRepeatParams repeatParams = {1, 1, 1, 8, 8, 8};

                        DataCopy(argmaxLocalTemp, argmaxGm[offsetInOut], updatesTailCopy);
                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);

                        DataCopy(outLocalTemp, outGm[offsetInOut], updatesTailCopy);

                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);
                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);

                        Compare(mask1Tensor, updatesLocal, outLocalTemp, CMPMODE::LT, 8, repeatTimes, repeatParams);
                        Select(outLocalTemp, mask1Tensor, outLocalTemp, updatesLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, updatesTail);

                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);

                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);
                        Cast(tempArgmaxFloatT, argmaxLocalTemp, RoundMode::CAST_ROUND, updatesTail);
                        WaitFlag<HardEvent::S_V>(eventIdSToV_0);
                        Duplicate(tempArgmaxFloatT2, (float)argmaxValue, updatesTail);

                        Select(tempArgmaxFloatT, mask1Tensor, tempArgmaxFloatT, tempArgmaxFloatT2, SELMODE::VSEL_TENSOR_TENSOR_MODE, updatesTail);
                        Cast(argmaxLocalTemp, tempArgmaxFloatT, RoundMode::CAST_ROUND, updatesTail);
                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_1);

                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);
                        ComputeDataCopy<float>(outGm[offsetInOut], outLocalTemp, updatesTail);

                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_1);
                        ComputeDataCopy<int32_t>(argmaxGm[offsetInOut], argmaxLocalTemp, updatesTail);
                    }
                }
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
                SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);
            }
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);
        }

        if (indicesLastNum != 0) {
            LocalTensor<DTYPE_INDICES>indicesLocal = inQueueIndices.Get<DTYPE_INDICES>();
            LocalTensor<DTYPE_UPDATES>updatesLocal = inQueueUpdates.Get<DTYPE_UPDATES>();
            DataCopy(indicesLocal, indicesGm[indicesLoop * ubIndicesNum], (indicesLastNum + indicesEachBlock - 1) / indicesEachBlock * indicesEachBlock);

            int64_t indicesStart = curBlockIdx * outEachLine + outTaskLine * progress;
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);
            for (uint32_t idx = 0; idx < indicesLastNum; idx++) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
                WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);
                DTYPE_INDICES dataInIndices = indicesLocal.GetValue(idx);
                // if this indices should be processed in this task
                if (dataInIndices >= indicesStart && dataInIndices < indicesStart + outThisLine) {
                    int64_t outBlockOffset = dataInIndices - indicesStart;
                    int64_t outLineOffset = idx % (outTailNum / updatesTail);
                    int64_t outActualOffset = outBlockOffset * outTailNum + outLineOffset * updatesTail;
                    int64_t updatesOffset = idx * updatesTail;
                    DataCopy(updatesLocal, updatesGm[indicesLoop * ubUpdatesNum + updatesOffset], updatesTailCopy);
                    // offset in localSetNote
                    int64_t localSetNoteOffset = dataInIndices % (outTaskNum / updatesTail);
                    DTYPE_INDICES argmaxValue = (idx + ubIndicesNum * indicesLoop) / argmaxGap;
                    SetFlag<HardEvent::S_V>(eventIdSToV_0);

                    int64_t offsetInOut = curBlockIdx * outEachCore + progress * outTaskNum + outActualOffset;
                    if (localSetNote.GetValue(localSetNoteOffset) == 0) {
                        localSetNote.SetValue(localSetNoteOffset, 1);

                        BinaryRepeatParams repeatParams = {1, 1, 1, 8, 8, 8};

                        DataCopy(argmaxLocalTemp, argmaxGm[offsetInOut], updatesTailCopy);
                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);

                        DataCopy(outLocalTemp, varGm[offsetInOut], updatesTailCopy);

                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);
                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);

                        Compare(mask1Tensor, updatesLocal, outLocalTemp, CMPMODE::LT, 8, repeatTimes, repeatParams);
                        Select(outLocalTemp, mask1Tensor, outLocalTemp, updatesLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, updatesTail);

                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);

                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);
                        Cast(tempArgmaxFloatT, argmaxLocalTemp, RoundMode::CAST_ROUND, updatesTail);
                        WaitFlag<HardEvent::S_V>(eventIdSToV_0);
                        Duplicate(tempArgmaxFloatT2, (float)argmaxValue, updatesTail);

                        Select(tempArgmaxFloatT, mask1Tensor, tempArgmaxFloatT, tempArgmaxFloatT2, SELMODE::VSEL_TENSOR_TENSOR_MODE, updatesTail);
                        Cast(argmaxLocalTemp, tempArgmaxFloatT, RoundMode::CAST_ROUND, updatesTail);
                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_1);

                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);
                        ComputeDataCopy<float>(outGm[offsetInOut], outLocalTemp, updatesTail);

                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_1);
                        ComputeDataCopy<int32_t>(argmaxGm[offsetInOut], argmaxLocalTemp, updatesTail);
                    } else {
                        BinaryRepeatParams repeatParams = {1, 1, 1, 8, 8, 8};

                        DataCopy(argmaxLocalTemp, argmaxGm[offsetInOut], updatesTailCopy);
                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);

                        DataCopy(outLocalTemp, outGm[offsetInOut], updatesTailCopy);

                        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);
                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_0);

                        Compare(mask1Tensor, updatesLocal, outLocalTemp, CMPMODE::LT, 8, repeatTimes, repeatParams);
                        Select(outLocalTemp, mask1Tensor, outLocalTemp, updatesLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, updatesTail);

                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);

                        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_1);
                        Cast(tempArgmaxFloatT, argmaxLocalTemp, RoundMode::CAST_ROUND, updatesTail);
                        WaitFlag<HardEvent::S_V>(eventIdSToV_0);
                        Duplicate(tempArgmaxFloatT2, (float)argmaxValue, updatesTail);

                        Select(tempArgmaxFloatT, mask1Tensor, tempArgmaxFloatT, tempArgmaxFloatT2, SELMODE::VSEL_TENSOR_TENSOR_MODE, updatesTail);
                        Cast(argmaxLocalTemp, tempArgmaxFloatT, RoundMode::CAST_ROUND, updatesTail);
                        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_1);

                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_0);
                        ComputeDataCopy<float>(outGm[offsetInOut], outLocalTemp, updatesTail);

                        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_1);
                        ComputeDataCopy<int32_t>(argmaxGm[offsetInOut], argmaxLocalTemp, updatesTail);
                    }
                }
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
                SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);
            }
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2_0);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_0);
        }

        outSetNote.FreeTensor(localSetNote);
        mask1Buf.FreeTensor(mask1Tensor);
        tempArgmaxFloat.FreeTensor(tempArgmaxFloatT);
        tempArgmaxFloat2.FreeTensor(tempArgmaxFloatT2);
    }

private:
    TPipe* pipe;
    TBuf<TPosition::VECCALC> inQueueVar, inQueueIndices, inQueueUpdates;
    TBuf<TPosition::VECCALC> outQueueOut, outQueueArgmax;
    TBuf<TPosition::VECCALC> tempArgmaxFloat, tempArgmaxFloat2, outSetNote, mask1Buf;

    GlobalTensor<DTYPE_VAR> varGm;
    GlobalTensor<DTYPE_INDICES> indicesGm;
    GlobalTensor<DTYPE_UPDATES> updatesGm;
    GlobalTensor<DTYPE_OUT> outGm;
    GlobalTensor<DTYPE_ARGMAX> argmaxGm;

    LocalTensor<DTYPE_VAR> varLocalTemp;
    LocalTensor<DTYPE_OUT> outLocalTemp;
    LocalTensor<DTYPE_ARGMAX> argmaxLocalTemp;

    DataCopyExtParams copyParamsOut;
    uint64_t curBlockIdx;
    uint64_t argmaxGap;
    int32_t initArgmax;
    bool isAligned;
    uint64_t usedCoreNum, tilingMode;
    uint64_t indicesNum, updatesNum, outNum, updatesTail, outTailNum, outEachCore;
    uint64_t indicesLoop, ubIndicesNum, ubUpdatesNum, indicesLastNum, unpdatesLastNum;
    uint64_t outTaskNum, taskNum, outLastTaskNum, outrepeat, outEachLine, repeatTimes;
    uint64_t outTaskLine, outThisNum, outThisLine, updatesTailCopy, dataEachBlock, indicesEachBlock;

    event_t eventIdMte2ToV_0, eventIdMte2ToV_1, eventIdVToMte3_0, eventIdVToMte3_1, eventIdSToV_0, eventIdMte3ToMte2_0, eventIdMte3ToV_0;
};

extern "C" __global__ __aicore__ void scatter_max_with_argmax_v2(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR out,
                                                              GM_ADDR argmax, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    KernelScatterMaxWithArgmaxV2 op;
    op.Init(var, indices, updates, out, argmax, &tiling_data, &pipe);
    op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void scatter_max_with_argmax_v2_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* var, uint8_t* indices, uint8_t* updates,
    uint8_t* out, uint8_t* argmax, uint8_t* workspace, uint8_t* tiling)
{
    scatter_max_with_argmax_v2<<<blockDim, l2ctrl, stream>>>(var, indices, updates, out, argmax, workspace, tiling);
}
#endif