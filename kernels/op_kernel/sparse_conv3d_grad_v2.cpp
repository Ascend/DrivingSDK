/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace matmul;

class KernelSparseConv3dGradV2 {
public:
    __aicore__ inline KernelSparseConv3dGradV2() {}
    __aicore__ inline void Init(GM_ADDR indices_offset, GM_ADDR former_sorted_indices, GM_ADDR feature, GM_ADDR weight, GM_ADDR grad, GM_ADDR feature_grad, GM_ADDR weight_grad, GM_ADDR workspace, SparseConv3dGradV2TilingData *tiling_data, TPipe *pipe)
    {
        this->featureCubeTilingData = tiling_data->featureCubeTilingData;
        this->weightCubeTilingData = tiling_data->weightCubeTilingData;
        curBlockIdx = GetBlockIdx();
        initTilingData(tiling_data);
        valueBlockNum = blockBytes / sizeof(DTYPE_WEIGHT);
        idxBlockNum = blockBytes / sizeof(DTYPE_INDICES_OFFSET);
        kernelSizeAlign = AlignUp(kernelSize, idxBlockNum);
        kernelICAlign = AlignUp(kernelIC, valueBlockNum);
        uint64_t beginOffset = curBlockIdx * vectorCoreTask;
        if (usedVectorCoreNum <= 0) {
            return ;
        }
        uint64_t initLen = featureCubeTilingData.M / usedVectorCoreNum;

        indicesOffsetGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES_OFFSET *>(indices_offset) + beginOffset);
        formerSortedIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES_OFFSET *>(former_sorted_indices));
        featureGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_FEATURE *>(feature));
        gradGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_GRAD *>(grad) + beginOffset * kernelOC);

        featureGradGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_FEATURE *>(feature_grad));
        weightGradGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_FEATURE *>(weight_grad));

        uint64_t weightLeftGmOffset = featureCubeTilingData.M * kernelSize * kernelOC;
        weightCoreBeginOffset = beginOffset;

        // featureMatMul featureLeftGm * weightGm = featureGradGm
        featureLeftGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_FEATURE *>(workspace));
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_WEIGHT *>(weight));
        // weightMatMul weightLeftGm * weightRightGm = weightGradGm
        weightLeftGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_WEIGHT *>(workspace) + weightLeftGmOffset);
        weightRightGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_WEIGHT *>(grad));
        // workspace Init
        featureInitGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_FEATURE *>(workspace) + initLen * curBlockIdx * kernelSize * kernelOC);
        const float zeros = 0.0;
        if (curBlockIdx < usedVectorCoreNum) {
            if (usedVectorCoreNum - 1 != curBlockIdx) {
                InitGlobalMemory(featureInitGm, initLen * kernelSize * kernelOC, zeros);
            } else {
                InitGlobalMemory(featureInitGm, (featureCubeTilingData.M - initLen * curBlockIdx) * kernelSize * kernelOC, zeros);
            }
        }
        SyncAll();

        uint64_t featureSingleCoreM, featureSingleCoreN, featureSingleCoreK;
        CalcOffset(curBlockIdx, featureCubeTilingData, 0, 0,
                   featureOffsetA, featureOffsetB, featureOffsetC,
                   featureSingleCoreM, featureSingleCoreN, featureSingleCoreK);
        featureMatmulObj.SetTail(featureSingleCoreM, featureSingleCoreN, featureSingleCoreK);
        uint64_t weightSingleCoreM, weightSingleCoreN, weightSingleCoreK;
        CalcOffset(curBlockIdx, weightCubeTilingData, 1, 0,
                   weightOffsetA, weightOffsetB, weightOffsetC,
                   weightSingleCoreM, weightSingleCoreN, weightSingleCoreK);
        weightMatmulObj.SetTail(weightSingleCoreM, weightSingleCoreN, weightSingleCoreK);

        pipe->InitBuffer(indicesOffsetQueue, 1, AlignUp(moveLen + 1, idxBlockNum) * sizeof(DTYPE_INDICES_OFFSET));
        pipe->InitBuffer(formerSortedIndicesQueue, 1, moveLen * kernelSizeAlign * sizeof(DTYPE_INDICES_OFFSET));
        pipe->InitBuffer(gradQueue, 1, kernelOC * sizeof(DTYPE_WEIGHT));
        pipe->InitBuffer(featureUb, kernelSize * AlignUp(kernelIC, valueBlockNum) * sizeof(DTYPE_WEIGHT));
    }

    __aicore__ inline void Process(TPipe *pipe)
    {
        if (curBlockIdx < usedVectorCoreNum) {
            CopyInAndWeightCalculate();
        }
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x8);
        CrossCoreWaitFlag(0x8);
        if (curBlockIdx < featureCubeNum) {
            featureLeftGm = featureLeftGm[featureOffsetA];
            weightGm = weightGm[featureOffsetB];
            featureGradGm = featureGradGm[featureOffsetC];
            featureMatmulObj.SetTensorA(featureLeftGm);
            featureMatmulObj.SetTensorB(weightGm);
            featureMatmulObj.IterateAll(featureGradGm);
        }
        if (curBlockIdx < weightCubeNum) {
            weightLeftGm = weightLeftGm[weightOffsetA];
            weightRightGm = weightRightGm[weightOffsetB];
            weightGradGm = weightGradGm[weightOffsetC];
            weightMatmulObj.SetTensorA(weightLeftGm, true);
            weightMatmulObj.SetTensorB(weightRightGm);
            weightMatmulObj.IterateAll(weightGradGm);
        }
    }

    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_FEATURE>, MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_FEATURE>,
        MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_FEATURE>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_FEATURE>>
    featureMatmulObj;

    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_FEATURE, true>, MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_FEATURE>,
        MatmulType<TPosition::GM, CubeFormat::ND, DTYPE_FEATURE>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_FEATURE>>
    weightMatmulObj;

private:
    __aicore__ inline void initTilingData(SparseConv3dGradV2TilingData *tiling_data)
    {
        usedVectorCoreNum = tiling_data->usedVectorCoreNum;
        featureCubeNum = tiling_data->featureCubeNum;
        weightCubeNum = tiling_data->weightCubeNum;
        kernelIC = tiling_data->kernelIC;
        kernelOC = tiling_data->kernelOC;
        kernelSize = tiling_data->kernelSize;
        moveLen = tiling_data->moveLen;
        vectorActualNum = tiling_data->vectorActualNum;
        vectorCoreTask = tiling_data->vectorCoreTask;
        vectorLastCoreTask = tiling_data->vectorLastCoreTask;
        coreRepeatTimes = tiling_data->coreRepeatTimes;
        coreMoveLenTail = tiling_data->coreMoveLenTail;
        lastCoreRepeatTimes = tiling_data->lastCoreRepeatTimes;
        lastCoreMoveLenTail = tiling_data->lastCoreMoveLenTail;
    }

    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                      uint32_t isTransposeAIn, uint32_t isTransposeBIn,
                                      uint64_t &offsetA, uint64_t &offsetB, uint64_t &offsetC,
                                      uint64_t &singleCoreM, uint64_t &singleCoreN, uint64_t &singleCoreK)
    {
        auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
        auto kSingleBlocks = Ceiling(tiling.Ka, tiling.singleCoreK);
        auto nSingleBlocks = Ceiling(tiling.N, tiling.singleCoreN);
        auto divideKcoreNum = usedVectorCoreNum / kSingleBlocks;

        auto mCoreIndx = (blockIdx % divideKcoreNum) % mSingleBlocks;
        auto nCoreIndx = (blockIdx % divideKcoreNum) / mSingleBlocks;

        if (isTransposeAIn == 0) {
            offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
        } else {
            offsetA = mCoreIndx * tiling.singleCoreM;
        }
        if (isTransposeBIn == 0) {
            offsetB = nCoreIndx * tiling.singleCoreN;
        } else {
            offsetB = nCoreIndx * tiling.Kb * tiling.singleCoreN;
        }
        offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
        // 尾块M
        uint64_t gmUseM = tiling.M - mCoreIndx * tiling.singleCoreM;
        singleCoreM = gmUseM < tiling.singleCoreM ? gmUseM : tiling.singleCoreM;
        // 尾块N
        uint64_t gmUseN = tiling.N - nCoreIndx * tiling.singleCoreN;
        singleCoreN = gmUseN < tiling.singleCoreN ? gmUseN : tiling.singleCoreN;
        // 尾块K
        uint64_t gmUseK = tiling.Ka;
        singleCoreK = gmUseK < tiling.singleCoreK ? gmUseK : tiling.singleCoreK;
    }

    __aicore__ inline void CopyInAndWeightCalculate()
    {
        indicesOffsetLocal = indicesOffsetQueue.AllocTensor<DTYPE_INDICES_OFFSET>();
        sortLocal = formerSortedIndicesQueue.AllocTensor<DTYPE_INDICES_OFFSET>();
        gradLocal = gradQueue.AllocTensor<DTYPE_WEIGHT>();
        featureLocal = featureUb.Get<DTYPE_WEIGHT>();

        DataCopyExtParams gradCopyParams {1, (uint32_t)(kernelOC * sizeof(DTYPE_WEIGHT)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_WEIGHT> gradCopyPadParams{true, 0, 0, 0};
        DataCopyExtParams featureCopyParams {1, (uint32_t)(kernelIC * sizeof(DTYPE_WEIGHT)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_WEIGHT> featureCopyPadParams{true, 0, 0, 0};
        DataCopyExtParams weightCopyParams {1, (uint32_t)(kernelIC * kernelOC * sizeof(DTYPE_WEIGHT)), 0, 0, 0};
        DataCopyExtParams weightLeftCopyParams {(uint16_t)kernelSize, (uint32_t)(kernelIC * sizeof(DTYPE_WEIGHT)), 0, 0, 0};

        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        event_t eventIDMTE2ToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));

        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        uint64_t repeatTimes = coreRepeatTimes;
        uint64_t movelenTail = coreMoveLenTail;
        if (curBlockIdx == usedVectorCoreNum - 1) {
            repeatTimes = lastCoreRepeatTimes;
            movelenTail = lastCoreMoveLenTail;
        }
        for (uint64_t computeRound = 0; computeRound < repeatTimes; computeRound++) {
            uint64_t repeatBeginOffset = computeRound * moveLen;
            uint64_t realMoveLen = moveLen;
            if (computeRound == repeatTimes - 1) {
                realMoveLen = movelenTail;
            }

            DataCopyExtParams indicesOffsetCopyParams {1, (uint32_t)((realMoveLen + 1) * sizeof(DTYPE_INDICES_OFFSET)), 0, 0, 0};
            DataCopyPadExtParams<DTYPE_INDICES_OFFSET> indicesOffsetPadParams{true, 0, 0, 0};

            DataCopyPad(indicesOffsetLocal, indicesOffsetGm[repeatBeginOffset], indicesOffsetCopyParams, indicesOffsetPadParams);
            SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            DataCopyPadExtParams<DTYPE_INDICES_OFFSET> sortPadParams{true, 0, 0, 0};
            for (uint64_t idx = 0; idx < realMoveLen; idx++) {
                uint32_t beginIndicesOffset = indicesOffsetLocal.GetValue(idx);
                uint32_t endIndicesOffset = indicesOffsetLocal.GetValue(idx + 1);
                DataCopyExtParams sortCopyParams {1, (uint32_t)((endIndicesOffset - beginIndicesOffset) * sizeof(DTYPE_INDICES_OFFSET)), 0, 0, 0};
                DataCopyPad(sortLocal[idx * kernelSizeAlign], formerSortedIndicesGm[beginIndicesOffset], sortCopyParams, sortPadParams);
            }
            SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            for (uint64_t idx = 0; idx < realMoveLen; idx++) {
                uint32_t beginIndicesOffset = indicesOffsetLocal.GetValue(idx);
                uint32_t endIndicesOffset = indicesOffsetLocal.GetValue(idx + 1);
                Duplicate<DTYPE_WEIGHT>(featureLocal, 0.0, kernelSize * AlignUp(kernelIC, valueBlockNum));
                DataCopyPad(gradLocal, gradGm[(repeatBeginOffset + idx) * kernelOC], gradCopyParams, gradCopyPadParams);
                SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
                WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
                SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
                WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
                for (uint32_t j = 0; j < endIndicesOffset - beginIndicesOffset; j++) {
                    uint32_t sortOffset = sortLocal.GetValue(j + idx * kernelSizeAlign);
                    uint32_t featureOffset = sortOffset / kernelSize;
                    uint32_t weightOffset = sortOffset % kernelSize;
                    DataCopyPad(featureLeftGm[featureOffset * kernelSize * kernelOC + weightOffset * kernelOC], gradLocal, gradCopyParams);
                    DataCopyPad(featureLocal[weightOffset * kernelICAlign], featureGm[featureOffset * kernelIC], featureCopyParams, featureCopyPadParams);
                }
                SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
                WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
                DataCopyPad(weightLeftGm[(weightCoreBeginOffset + repeatBeginOffset + idx) * kernelSize * kernelIC], featureLocal, weightLeftCopyParams);
                SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
                WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
            }
            pipe_barrier(PIPE_ALL);
        }

        indicesOffsetQueue.FreeTensor(indicesOffsetLocal);
        formerSortedIndicesQueue.FreeTensor(sortLocal);
        gradQueue.FreeTensor(gradLocal);
    }

    __aicore__ inline uint64_t Ceiling(uint64_t a, uint64_t b)
    {
        return (a + b - 1) / b;
    }

private:
    TCubeTiling featureCubeTilingData, weightCubeTilingData;
    GlobalTensor<DTYPE_WEIGHT> featureInitGm, featureGm, gradGm, featureGradGm, weightGradGm, featureLeftGm, weightGm, weightLeftGm, weightRightGm;
    GlobalTensor<DTYPE_INDICES_OFFSET> indicesOffsetGm, formerSortedIndicesGm;

    LocalTensor<DTYPE_WEIGHT> gradLocal, featureLocal, mulTemp;
    LocalTensor<DTYPE_INDICES_OFFSET> indicesOffsetLocal, sortLocal;

    TQue<QuePosition::VECIN, 1> indicesOffsetQueue, formerSortedIndicesQueue, gradQueue;
    TBuf<TPosition::VECCALC> featureUb;

    uint32_t usedVectorCoreNum;
    uint32_t featureCubeNum;
    uint32_t weightCubeNum;
    uint64_t kernelIC;
    uint64_t kernelOC;
    uint64_t kernelSize;
    uint64_t moveLen;
    uint64_t vectorActualNum;
    uint64_t vectorCoreTask;
    uint64_t vectorLastCoreTask;
    uint64_t coreRepeatTimes;
    uint64_t coreMoveLenTail;
    uint64_t lastCoreRepeatTimes;
    uint64_t lastCoreMoveLenTail;

    uint64_t blockBytes{32};
    uint64_t repeatBlockByte{256};

    uint32_t curBlockIdx;
    uint64_t valueBlockNum;
    uint64_t idxBlockNum;
    uint64_t weightCoreBeginOffset;
    uint64_t kernelSizeAlign;
    uint64_t kernelICAlign;

    uint64_t featureOffsetA;
    uint64_t featureOffsetB;
    uint64_t featureOffsetC;
    uint64_t weightOffsetA;
    uint64_t weightOffsetB;
    uint64_t weightOffsetC;
};

extern "C" __global__ __aicore__ void sparse_conv3d_grad_v2(GM_ADDR indices_offset, GM_ADDR former_sorted_indices,
                                                        GM_ADDR feature, GM_ADDR weight, GM_ADDR grad,
                                                        GM_ADDR feature_grad, GM_ADDR weight_grad,
                                                        GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    TPipe pipe;
    KernelSparseConv3dGradV2 op;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.featureMatmulObj, op.weightMatmulObj);
    op.featureMatmulObj.Init(&tiling_data.featureCubeTilingData);
    op.weightMatmulObj.Init(&tiling_data.weightCubeTilingData);
    op.Init(indices_offset, former_sorted_indices, feature, weight, grad,
            feature_grad, weight_grad, workspace, &tiling_data, &pipe);
    op.Process(&pipe);
}