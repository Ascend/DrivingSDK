/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef _KNN_H_
#define _KNN_H_
#include <cmath>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace AscendC {
// T is the dtype of input and output dist2(float32 or float16) while U is for the output idx(only int32_t)
template<typename T, typename U>
class KnnKernel {
public:
    __aicore__ inline KnnKernel(GM_ADDR xyz, GM_ADDR center_xyz, GM_ADDR dist, GM_ADDR idx, const KnnTilingData* tiling_data, TPipe *tmpPipe)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        batch = tiling_data->batch;
        nPoint = tiling_data->nPoint;
        nSource = tiling_data->nSource;
        coreNum = tiling_data->coreNum;
        isFromKnn = tiling_data->isFromKnn;
        k = tiling_data->k;

        formerTaskNum = Ceil(batch * nPoint, coreNum);

        coreId = GetBlockIdx();
        InitGm(xyz, center_xyz, dist, idx, tmpPipe);
        InitBuffer();
    }
    __aicore__ inline void InitGm(GM_ADDR xyz, GM_ADDR center_xyz, GM_ADDR dist, GM_ADDR idx, TPipe *tmpPipe)
    {
        pipe = tmpPipe;
        startTask = coreId * formerTaskNum;
        endTask = startTask + formerTaskNum;
        if (endTask > (batch * nPoint)) {
            endTask = batch * nPoint;
        }

        sourceGm.SetGlobalBuffer((__gm__ T *)xyz, static_cast<uint64_t>(batch) * nSource * 3);
        targetGm.SetGlobalBuffer((__gm__ T *)center_xyz, static_cast<uint64_t>(batch) * nPoint * 3);
        distGm.SetGlobalBuffer((__gm__ T *)dist, static_cast<uint64_t>(batch) * nPoint * k);
        idxGm.SetGlobalBuffer((__gm__ int32_t*)idx, static_cast<uint64_t>(batch) * nPoint * k);
    }

    __aicore__ inline void InitBuffer()
    {
        pipe->InitBuffer(targetUb, 32);
        pipe->InitBuffer(sourceBackupUb, compNum * sizeof(T) * 3);
        pipe->InitBuffer(sourceUb, compNum * sizeof(T) * 3);
        pipe->InitBuffer(distUb, compNum * sizeof(T));
        pipe->InitBuffer(idxUb, compNum * sizeof(int32_t));
        pipe->InitBuffer(constIdxUb, compNum * sizeof(int32_t));

        pipe->InitBuffer(bestDistUb, compNum * sizeof(T));
        pipe->InitBuffer(bestIdxUb, compNum * sizeof(int32_t));

        pipe->InitBuffer(sortSrcUb, compNum * sizeof(T) * 2);
        pipe->InitBuffer(sortTmp1Ub, mergeLength * sizeof(T) * 4);
        pipe->InitBuffer(sortTmp2Ub, mergeLength * sizeof(T) * 4);
    }
    __aicore__ inline void Process()
    {
        // 计算loop time
        uint32_t loopTimes = nSource / compNum;
        uint32_t tailNum = nSource % compNum;
        sourceBackupLocal = sourceBackupUb.Get<T>();
        sourceLocal = sourceUb.Get<T>();
        targetLocal = targetUb.Get<T>();
        distLocal = distUb.Get<T>();
        idxLocal = idxUb.Get<int32_t>();
        bestDistLocal = bestDistUb.Get<T>();
        bestIdxLocal = bestIdxUb.Get<int32_t>();

        constIdxLocal = constIdxUb.Get<int32_t>();
        for (int32_t index = 0; index < compNum; index++) {
            constIdxLocal.SetValue((uint32_t)index, index);
        }

        sortSrcLocal = sortSrcUb.Get<T>();
        sortTmp1Local = sortTmp1Ub.Get<T>();
        sortTmp2Local = sortTmp2Ub.Get<T>();

        for (uint32_t currentTask = startTask; currentTask < endTask; currentTask++) {
            uint64_t currentBatch = currentTask / nPoint;
            uint64_t sourceOffset = currentBatch * nSource * 3; // B 3 N
            uint64_t targetOffset = currentTask * 3; // B M 3
            uint64_t copyOutOffset = currentTask * k; // B M N

            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            DataCopy(targetLocal, targetGm[targetOffset], 8);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            Duplicate<T>(sourceBackupLocal, targetLocal.GetValue(0), (int32_t)compNum);
            Duplicate<T>(sourceBackupLocal[compNum], targetLocal.GetValue(1), (int32_t)compNum);
            Duplicate<T>(sourceBackupLocal[compNum * 2], targetLocal.GetValue(2), (int32_t)compNum);

            Duplicate(sortTmp1Local, minFloatValue, mergeLength * 4);
            Duplicate(sortTmp2Local, minFloatValue, mergeLength * 4);

            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            for (uint32_t currentLoop = 0; currentLoop < loopTimes; currentLoop++) {
                Compute(currentLoop, compNum, sourceOffset);
            }
            if (tailNum > 0) {
                Compute(loopTimes, tailNum, sourceOffset);
            }
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

            CopyOut(copyOutOffset);
        }
    }

    __aicore__ inline void Compute(uint32_t currentLoop, uint32_t copySize, uint64_t sourceOffset)
    {
        uint32_t copyInLength = static_cast<uint32_t>(copySize * sizeof(T));
        uint32_t loopOffset = currentLoop * compNum;

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        DataCopyPad(sourceLocal, sourceGm[sourceOffset + loopOffset],
                    {1, copyInLength, 0, 0, 0}, {false, 0, 0, 0});
        DataCopyPad(sourceLocal[compNum], sourceGm[sourceOffset + loopOffset + nSource],
                    {1, copyInLength, 0, 0, 0}, {false, 0, 0, 0});
        DataCopyPad(sourceLocal[compNum * 2], sourceGm[sourceOffset + loopOffset + nSource * 2],
                    {1, copyInLength, 0, 0, 0}, {false, 0, 0, 0});

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        Sub<T>(sourceLocal, sourceLocal, sourceBackupLocal, compNum * 3);
        Mul<T>(sourceLocal, sourceLocal, sourceLocal, compNum * 3);

        Duplicate(distLocal, maxFloatValue, compNum);
        Add<T>(distLocal, sourceLocal, sourceLocal[compNum], copySize);
        Add<T>(distLocal, distLocal, sourceLocal[compNum * 2], copySize);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

        if (isFromKnn) {
            Mins<T>(distLocal, distLocal, static_cast<T>(1e10f), compNum);
        }

        Adds(idxLocal, constIdxLocal, static_cast<int32_t>(loopOffset), compNum);
        Muls(distLocal, distLocal, -1.0f, compNum);

        SortDist(currentLoop);
    }

    __aicore__ inline void SortDist(uint32_t currentLoop)
    {
        const uint16_t sortCountList[4] = {(uint16_t)sort32Elements, (uint16_t)sort32Elements, (uint16_t)sort32Elements, (uint16_t)sort32Elements};
        const uint16_t mergeCountList[4] = {(uint16_t)k, (uint16_t)k, (uint16_t)k, (uint16_t)k};
        AscendC::LocalTensor<uint32_t> interpreIdxInTensor = idxLocal.ReinterpretCast<uint32_t>();
        Sort32(sortSrcLocal, distLocal, interpreIdxInTensor, sort32RepeatTimes);
        AscendC::MrgSortSrcList sortList = AscendC::MrgSortSrcList(sortSrcLocal, sortSrcLocal[sort32Offset], sortSrcLocal[sort32Offset * 2], sortSrcLocal[sort32Offset * 3]);
        if ((currentLoop % 2) == 0) {
            MrgSort<T>(sortTmp1Local[mergeLength], sortList, {sortCountList, false, 0b1111, sort32MergeRepeatTimes});
            AscendC::MrgSortSrcList mergeList = AscendC::MrgSortSrcList(sortTmp1Local, sortTmp1Local[mergeLength], sortTmp1Local[mergeLength * 2], sortTmp1Local[mergeLength * 3]);
            MrgSort<T>(sortTmp2Local, mergeList, {mergeCountList, false, 0b1111, 1});
        } else {
            MrgSort<T>(sortTmp2Local[mergeLength], sortList, {sortCountList, false, 0b1111, sort32MergeRepeatTimes});
            AscendC::MrgSortSrcList mergeList = AscendC::MrgSortSrcList(sortTmp2Local, sortTmp2Local[mergeLength], sortTmp2Local[mergeLength * 2], sortTmp2Local[mergeLength * 3]);
            MrgSort<T>(sortTmp1Local, mergeList, {mergeCountList, false, 0b1111, 1});
        }
    }

    __aicore__ inline void CopyOut(uint64_t offset)
    {
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        AscendC::LocalTensor<uint32_t> interpreIdxOutTensor = bestIdxLocal.ReinterpretCast<uint32_t>();
        if (((nSource + compNum - 1) / compNum) % 2 == 1) {
            Extract(bestDistLocal, interpreIdxOutTensor, sortTmp2Local, sort32RepeatTimes);
        } else {
            Extract(bestDistLocal, interpreIdxOutTensor, sortTmp1Local, sort32RepeatTimes);
        }
        Muls(bestDistLocal, bestDistLocal, -1.0f, k);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        DataCopyPad(distGm[offset], bestDistLocal,
            {1, static_cast<uint32_t>(k * sizeof(T)), 0, 0, 0});
        DataCopyPad(idxGm[offset], bestIdxLocal,
            {1, static_cast<uint32_t>(k * sizeof(int32_t)), 0, 0, 0});
    }

public:
    TPipe *pipe;
    GlobalTensor<T> sourceGm, targetGm, distGm;
    GlobalTensor<int32_t> idxGm;
    TBuf<TPosition::VECCALC> sourceUb, sourceBackupUb, targetUb, distUb, idxUb, bestDistUb, bestIdxUb, constIdxUb, sortTmp1Ub, sortTmp2Ub, sortSrcUb;
    LocalTensor<T> sourceLocal, sourceBackupLocal, targetLocal, distLocal, bestDistLocal, sortTmp1Local, sortTmp2Local, sortSrcLocal;
    LocalTensor<int32_t> bestIdxLocal, idxLocal, constIdxLocal;
    uint32_t coreId;
    uint32_t startTask, endTask;
    uint32_t formerTaskNum;

    uint32_t compNum = 384;
    uint32_t mergeLength = 256;
    uint32_t sort32Elements = 32;
    uint32_t sort32Offset = sort32Elements * 2;
    int32_t sort32RepeatTimes = compNum / 32;
    uint16_t sort32MergeRepeatTimes = compNum / 128;

    float minFloatValue = -3.40282347E+38;
    float maxFloatValue = 3.40282347E+38;
public:
    // tiling
    uint32_t batch;
    uint32_t nPoint;
    uint32_t nSource;
    uint32_t coreNum;
    bool isFromKnn;
    int32_t k;
};
} // namespace AscendC

#endif  // _KNN_H_