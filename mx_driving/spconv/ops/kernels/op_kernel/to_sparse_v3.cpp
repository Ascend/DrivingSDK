/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace matmul;
namespace {
};

class ToSparseV3Kernel {
public:
    __aicore__ inline ToSparseV3Kernel() {}
    __aicore__ inline void Init(GM_ADDR features, GM_ADDR weight, GM_ADDR indices_offset, GM_ADDR former_sorted_indices, GM_ADDR indices, GM_ADDR sparse_value, GM_ADDR sparse_indices, GM_ADDR workspace, ToSparseV3TilingData *tiling_data, TPipe *pipe)
    {
        this->cubeTilingData = tiling_data->cubeTilingData;

        curBlockIdx = GetBlockIdx();
        initTilingData(tiling_data);

        valueBlockNum = blockBytes / sizeof(DTYPE_FEATURES);
        idxBlockNum = blockBytes / sizeof(DTYPE_INDICES);
        kernelICAlign = AlignUp(kernelIC, valueBlockNum);
        kernelSizeAlign = AlignUp(kernelSize, valueBlockNum);
        uint32_t beginOffset = curBlockIdx * vectorCoreTask;

        featuresGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_FEATURES *>(features));
        indicesOffsetGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES *>(indices_offset) + beginOffset);
        formerSortedIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES *>(former_sorted_indices));
        indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES *>(indices));
        workspaceGm_Copy.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_FEATURES *>(workspace) + beginOffset * kernelSize * kernelIC);
        sparseIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES *>(sparse_indices) + beginOffset * 8);

        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_FEATURES *>(workspace));
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_FEATURES *>(weight));
        sparseValueGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_FEATURES *>(sparse_value));

        CalcOffset(curBlockIdx, cubeTilingData, offsetA, offsetB, offsetC);

        pipe->InitBuffer(indicesOffsetQueue, 1, AlignUp(moveLen + 1, idxBlockNum) * sizeof(DTYPE_INDICES));
        pipe->InitBuffer(formerSortedIndicesQueue, 1, moveLen * kernelSizeAlign * sizeof(DTYPE_INDICES));
        pipe->InitBuffer(indicesQueue, 1, moveLen * 8 * sizeof(DTYPE_INDICES));
        pipe->InitBuffer(featrueQueue, 1, moveLen * kernelSize * kernelIC * sizeof(DTYPE_FEATURES));
    }

    __aicore__ inline void Process(TPipe *pipe)
    {
        if (curBlockIdx < usedVectorCoreNum) {
            CopyInAndToGm();
        }
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x8);
        CrossCoreWaitFlag(0x8);
        workspaceGm_ = workspaceGm_[offsetA];
        weightGm = weightGm[offsetB];
        sparseValueGm = sparseValueGm[offsetC];
        matmulObj.SetTensorA(workspaceGm_);
        matmulObj.SetTensorB(weightGm);
        matmulObj.IterateAll(sparseValueGm);
    }
    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_FEATURES>, MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_FEATURES>,
           MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_FEATURES>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_FEATURES>>
        matmulObj;

private:

    __aicore__ inline void initTilingData(ToSparseV3TilingData *tiling_data)
    {
        usedVectorCoreNum = tiling_data->usedVectorCoreNum;
        kernelIC = tiling_data->kernelIC;
        kernelSize = tiling_data->kernelSize;
        moveLen = tiling_data->moveLen;
        vectorCoreTask = tiling_data->vectorCoreTask;
        vectorLastCoreTask = tiling_data->vectorLastCoreTask;
        coreRepeatTimes = tiling_data->coreRepeatTimes;
        coreMoveLenTail = tiling_data->coreMoveLenTail;
        lastCoreRepeatTimes = tiling_data->lastCoreRepeatTimes;
        lastCoreMoveLenTail = tiling_data->lastCoreMoveLenTail;
    }
    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                    int64_t &offsetA, int64_t &offsetB, int64_t &offsetC)
    {
        auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
        auto kSingleBlocks = Ceiling(tiling.Ka, tiling.singleCoreK);
        auto nSingleBlocks = Ceiling(tiling.N, tiling.singleCoreN);
        auto divideKcoreNum = usedVectorCoreNum / kSingleBlocks;

        auto mCoreIndx = (blockIdx % divideKcoreNum) % mSingleBlocks;
        auto nCoreIndx = (blockIdx % divideKcoreNum) / mSingleBlocks;
        offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
        offsetB = nCoreIndx * tiling.singleCoreN;
        offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
        // 尾块M
        int32_t gmUseM = tiling.M - mCoreIndx * tiling.singleCoreM;
        int32_t singleCoreM = gmUseM < tiling.singleCoreM ? gmUseM : tiling.singleCoreM;
        // 尾块N
        int32_t gmUseN = tiling.N - nCoreIndx * tiling.singleCoreN;
        int32_t singleCoreN = gmUseN < tiling.singleCoreN ? gmUseN : tiling.singleCoreN;
        // 尾块K
        int32_t gmUseK = tiling.Ka;
        int32_t singleCoreK = gmUseK < tiling.singleCoreK ? gmUseK : tiling.singleCoreK;
        matmulObj.SetTail(singleCoreM, singleCoreN, singleCoreK);
    }

    __aicore__ inline void CopyInAndToGm()
    {
        indicesOffsetLocal = indicesOffsetQueue.AllocTensor<DTYPE_INDICES>();
        sortLocal = formerSortedIndicesQueue.AllocTensor<DTYPE_INDICES>();
        indicesLocal = indicesQueue.AllocTensor<DTYPE_INDICES>();
        featureLocal = featrueQueue.AllocTensor<DTYPE_FEATURES>();

        event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        event_t eventIDMTE2ToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        uint32_t repeatTimes = coreRepeatTimes;
        uint32_t movelenTail = coreMoveLenTail;
        if (curBlockIdx == usedVectorCoreNum - 1) {
            repeatTimes = lastCoreRepeatTimes;
            movelenTail = lastCoreMoveLenTail;
        }
        for (uint32_t computeRound = 0; computeRound < repeatTimes; computeRound++) {
            uint32_t repeatBeginOffset = computeRound * moveLen;
            uint32_t realMoveLen = moveLen;
            if (computeRound == repeatTimes - 1) {
                realMoveLen = movelenTail;
            }

            DataCopyExtParams indicesOffsetCopyParams {1, (uint32_t)((realMoveLen + 1) * sizeof(DTYPE_INDICES)), 0, 0, 0};
            DataCopyPadExtParams<DTYPE_INDICES> indicesOffsetPadParams{true, 0, 0, 0};
            DataCopyExtParams indicesCopyParams {1, (uint32_t)(4 * sizeof(DTYPE_INDICES)), 0, 0, 0};
            DataCopyPadExtParams<DTYPE_INDICES> indicesPadParams{true, 0, 0, 0};
            DataCopyExtParams featureCopyParams {1, (uint32_t)(kernelIC * sizeof(DTYPE_FEATURES)), 0, 0, 0};
            DataCopyPadExtParams<DTYPE_FEATURES> featureCopyPadParams{true, 0, 0, 0};

            DataCopyExtParams workspaceCopyParams {1, (uint32_t)(realMoveLen * kernelSize * kernelIC * sizeof(DTYPE_FEATURES)), 0, 0, 0};
            DataCopyExtParams outIndicesCopyParams {1, (uint32_t)(realMoveLen * 8 * sizeof(DTYPE_INDICES)), 0, 0, 0};

            Duplicate<DTYPE_FEATURES>(featureLocal, 0.0, moveLen * kernelSize * kernelICAlign);
            DataCopyPad(indicesOffsetLocal, indicesOffsetGm[repeatBeginOffset], indicesOffsetCopyParams, indicesOffsetPadParams);

            SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);

            DataCopyPadExtParams<DTYPE_INDICES> sortPadParams{true, 0, 0, 0};
            for (uint32_t idx = 0; idx < realMoveLen; idx++) {
                uint32_t beginIndicesOffset = indicesOffsetLocal.GetValue(idx);
                uint32_t endIndicesOffset = indicesOffsetLocal.GetValue(idx + 1);
                DataCopyExtParams sortCopyParams {1, (uint32_t)((endIndicesOffset - beginIndicesOffset) * sizeof(DTYPE_INDICES_OFFSET)), 0, 0, 0};
                DataCopyPad(sortLocal[idx * kernelSizeAlign], formerSortedIndicesGm[beginIndicesOffset], sortCopyParams, sortPadParams);
            }

            SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            for (uint32_t idx = 0; idx < realMoveLen; idx++) {
                uint32_t sortOffset = sortLocal.GetValue(idx * kernelSizeAlign);
                DataCopyPad(indicesLocal[idx * 8], indicesGm[sortOffset * 4], indicesCopyParams, indicesPadParams);
            }
            SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
            DataCopyPad(sparseIndicesGm[repeatBeginOffset * 8], indicesLocal, outIndicesCopyParams);
            uint32_t workspaceGmOffset = repeatBeginOffset * kernelSize * kernelIC;
            for (uint32_t idx = 0; idx < realMoveLen; idx++) {
                SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
                WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
                uint32_t beginIndicesOffset = indicesOffsetLocal.GetValue(idx);
                uint32_t endIndicesOffset = indicesOffsetLocal.GetValue(idx + 1);
                uint32_t featureLocalOffset = idx * kernelSize * kernelIC;
                for (uint32_t j = 0; j < endIndicesOffset - beginIndicesOffset; j++) {
                    uint32_t sortOffset = sortLocal.GetValue(j + idx * kernelSizeAlign);
                    uint32_t featureOffset = sortOffset / kernelSize;
                    uint32_t weightOffset = sortOffset % kernelSize;
                    DataCopyPad(featureLocal[featureLocalOffset + weightOffset * kernelIC], featuresGm[featureOffset * kernelIC], featureCopyParams, featureCopyPadParams);
                }
            }
            SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
            DataCopyPad(workspaceGm_Copy[workspaceGmOffset], featureLocal, workspaceCopyParams);
            pipe_barrier(PIPE_ALL);
        }
        indicesOffsetQueue.FreeTensor(indicesOffsetLocal);
        formerSortedIndicesQueue.FreeTensor(sortLocal);
        indicesQueue.FreeTensor(indicesLocal);
        featrueQueue.FreeTensor(featureLocal);
    }

    __aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
    {
        return (a + b - 1) / b;
    }

private:
    TCubeTiling cubeTilingData;
    GlobalTensor<DTYPE_FEATURES> featuresGm, weightGm, workspaceGm_, workspaceGm_Copy, sparseValueGm;
    GlobalTensor<DTYPE_INDICES> indicesOffsetGm, formerSortedIndicesGm, indicesGm, sparseIndicesGm;
    LocalTensor<DTYPE_INDICES> sortLocal, indicesLocal, indicesOffsetLocal;
    LocalTensor<DTYPE_FEATURES> featureLocal;
    TQue<QuePosition::VECIN, 1> indicesOffsetQueue, formerSortedIndicesQueue, featrueQueue, indicesQueue;

    uint32_t curBlockIdx;
    uint32_t blockBytes{32};
    uint32_t valueBlockNum;
    uint32_t idxBlockNum;
    uint32_t kernelICAlign;
    uint32_t kernelSizeAlign;
    int64_t offsetA;
    int64_t offsetB;
    int64_t offsetC;

    uint32_t usedCubeCoreNum;
    uint32_t usedVectorCoreNum;
    uint32_t kernelIC;
    uint32_t kernelSize;
    uint32_t moveLen;
    uint32_t vectorCoreTask;
    uint32_t vectorLastCoreTask;
    uint32_t coreRepeatTimes;
    uint32_t coreMoveLenTail;
    uint32_t lastCoreRepeatTimes;
    uint32_t lastCoreMoveLenTail;
    uint32_t tailM;
    uint32_t tailN;
    uint32_t tailK;
};

extern "C" __global__ __aicore__ void to_sparse_v3(GM_ADDR features, GM_ADDR weight, GM_ADDR indices_offset, GM_ADDR former_sorted_indices, GM_ADDR indices, GM_ADDR sparse_value, GM_ADDR sparse_indices, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    TPipe pipe;
    ToSparseV3Kernel op;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.matmulObj, &tiling_data.cubeTilingData);
    op.Init(features, weight, indices_offset, former_sorted_indices, indices, sparse_value, sparse_indices, workspace, &tiling_data, &pipe);
    op.Process(&pipe);
}