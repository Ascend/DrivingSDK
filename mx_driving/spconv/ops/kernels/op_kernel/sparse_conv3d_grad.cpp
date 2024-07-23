/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */

#include "kernel_operator.h"
using namespace AscendC;

namespace {
constexpr static int32_t BUFFER_NUM = 1;
};

class KernelSparseConv3dGrad {
public:
    __aicore__ inline KernelSparseConv3dGrad() {}
    __aicore__ inline void Init(GM_ADDR indices_offset, GM_ADDR former_sorted_indices, GM_ADDR feature, GM_ADDR weight, GM_ADDR grad, GM_ADDR feature_grad, GM_ADDR weight_grad, SparseConv3dGradTilingData *tiling_data, TPipe *tmpPipe)
    {
        pipe = tmpPipe;
        curBlockIdx = GetBlockIdx();
        dataAlign = blockBytes / sizeof(DTYPE_FEATURE);
        maskAlign = dataAlign * 8;
        initTilingData(tiling_data);
        calculateReduceSum();
        uint64_t beginOffset = curBlockIdx * coreTask;

        uint32_t valueBlockNum = blockBytes / sizeof(DTYPE_WEIGHT);
        uint32_t idxBlockNum = blockBytes / sizeof(DTYPE_INDICES_OFFSET);

        if (curBlockIdx < usedCoreNum - 1) {
            coreRepeatTimes = repeatTimes;
            coreMoveTail = moveTail;
        } else {
            coreRepeatTimes = lastRepeatTimes;
            coreMoveTail = lastMoveTail;
        }

        indicesOffsetGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES_OFFSET *>(indices_offset) + beginOffset);
        formerSortedIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES_OFFSET *>(former_sorted_indices));
        featureGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_FEATURE *>(feature));
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_WEIGHT *>(weight));
        gradGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_GRAD *>(grad));

        featureGradGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_FEATURE *>(feature_grad));
        weightGradGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_WEIGHT *>(weight_grad));

        pipe->InitBuffer(indicesOffsetQueue, BUFFER_NUM, AlignUp(moveLen, idxBlockNum) * sizeof(DTYPE_INDICES_OFFSET));
        pipe->InitBuffer(gradQueue, BUFFER_NUM, AlignUp(kernelOC * moveLen, idxBlockNum) * sizeof(DTYPE_INDICES_OFFSET));
        pipe->InitBuffer(formerSortedIndicesQueue, BUFFER_NUM, AlignUp(kernelSize, valueBlockNum) * sizeof(DTYPE_WEIGHT));
        pipe->InitBuffer(featureUb, AlignUp(kernelIC, valueBlockNum) * sizeof(DTYPE_WEIGHT));
        pipe->InitBuffer(weightUB, AlignUp(kernelIC * kernelOC, valueBlockNum) * sizeof(DTYPE_WEIGHT));
        pipe->InitBuffer(reduceSumTmpUB, AlignUp(kernelIC, valueBlockNum) * sizeof(DTYPE_WEIGHT));
        pipe->InitBuffer(workUB, workSize * sizeof(DTYPE_WEIGHT));
        pipe->InitBuffer(featureTmpUB, AlignUp(kernelIC, valueBlockNum) * sizeof(DTYPE_WEIGHT));
        pipe->InitBuffer(weightTmpUB, AlignUp(kernelIC * kernelOC, valueBlockNum) * sizeof(DTYPE_WEIGHT));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < coreRepeatTimes; i++) {
            Compute(i);
            pipe_barrier(PIPE_ALL);
        }
    }

private:
    __aicore__ inline void initTilingData(SparseConv3dGradTilingData *tiling_data)
    {
        usedCoreNum = tiling_data->usedCoreNum;
        coreTask = tiling_data->coreTask;
        lastCoreTask = tiling_data->lastCoreTask;
        moveLen = tiling_data->moveLen;
        repeatTimes = tiling_data->repeatTimes;
        moveTail = tiling_data->moveTail;
        lastRepeatTimes = tiling_data->lastRepeatTimes;
        lastMoveTail = tiling_data->lastMoveTail;
        kernelSize = tiling_data->kernelSize;
        kernelIC = tiling_data->kernelIC;
        kernelOC = tiling_data->kernelOC;
    }

    __aicore__ inline void Compute(uint32_t query)
    {
        uint32_t taskOffset = query * moveLen;
        uint32_t forMoveLen = moveLen;
        if (query == coreRepeatTimes - 1) {
            forMoveLen = coreMoveTail;
        }

        DataCopyExtParams indicesOffsetCopyParams {1, (uint32_t)((forMoveLen + 1) * sizeof(DTYPE_INDICES_OFFSET)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_INDICES_OFFSET> indicesOffsetPadParams{true, 0, 0, 0};
        DataCopyExtParams gradCopyParams {1, (uint32_t)(forMoveLen * kernelOC * sizeof(DTYPE_WEIGHT)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_WEIGHT> gradCopyPadParams{true, 0, 0, 0};
        DataCopyExtParams featureCopyParams {1, (uint32_t)(kernelIC * sizeof(DTYPE_WEIGHT)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_WEIGHT> featureCopyPadParams{true, 0, 0, 0};
        DataCopyExtParams weightCopyParams {1, (uint32_t)(kernelIC * kernelOC * sizeof(DTYPE_WEIGHT)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_WEIGHT> weightCopyPadParams{true, 0, 0, 0};

        DataCopyExtParams featureTmpCopyParams {1, (uint32_t)(uint32_t)(kernelIC * sizeof(DTYPE_WEIGHT)), 0, 0, 0};
        DataCopyExtParams weightTmpCopyParams {1, (uint32_t)(kernelIC * kernelOC * sizeof(DTYPE_WEIGHT)), 0, 0, 0};

        event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));

        LocalTensor<DTYPE_INDICES_OFFSET> indicesOffsetLocal = indicesOffsetQueue.AllocTensor<DTYPE_INDICES_OFFSET>();
        LocalTensor<DTYPE_WEIGHT> gradLocal = gradQueue.AllocTensor<DTYPE_WEIGHT>();
        LocalTensor<DTYPE_INDICES_OFFSET> sortLocal = formerSortedIndicesQueue.AllocTensor<DTYPE_INDICES_OFFSET>();

        LocalTensor<DTYPE_WEIGHT> featureLocal = featureUb.Get<DTYPE_WEIGHT>();
        LocalTensor<DTYPE_WEIGHT> weightLocal = weightUB.Get<DTYPE_WEIGHT>();
        LocalTensor<DTYPE_WEIGHT> reduceSumTemp = reduceSumTmpUB.Get<DTYPE_WEIGHT>();
        LocalTensor<DTYPE_WEIGHT> workLocal = workUB.Get<DTYPE_WEIGHT>();
        LocalTensor<DTYPE_WEIGHT> featureTmpLocal = featureTmpUB.Get<DTYPE_WEIGHT>();
        LocalTensor<DTYPE_WEIGHT> weightTmp = weightTmpUB.Get<DTYPE_WEIGHT>();

        SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
        WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
        DataCopyPad(indicesOffsetLocal, indicesOffsetGm[taskOffset], indicesOffsetCopyParams, indicesOffsetPadParams);
        DataCopyPad(gradLocal, gradGm[taskOffset * kernelOC], gradCopyParams, gradCopyPadParams);
        pipe_barrier(PIPE_MTE2);
        for (uint32_t i = 0; i < forMoveLen; i++) {
            uint32_t beginIndicesOffset = indicesOffsetLocal.GetValue(i);
            uint32_t endIndicesOffset = indicesOffsetLocal.GetValue(i + 1);
            DataCopyExtParams sortCopyParams {1, (uint32_t)((endIndicesOffset - beginIndicesOffset) * sizeof(DTYPE_INDICES_OFFSET)), 0, 0, 0};
            DataCopyPadExtParams<DTYPE_INDICES_OFFSET> sortPadParams{true, 0, 0, 0};
            SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
            WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
            DataCopyPad(sortLocal, formerSortedIndicesGm[beginIndicesOffset], sortCopyParams, sortPadParams);
            SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
            for (uint32_t j = 0; j < endIndicesOffset - beginIndicesOffset; j++) {
                uint32_t sortOffset = sortLocal.GetValue(j);
                uint32_t weightOffset = sortOffset % kernelSize;
                uint32_t featureOffset = sortOffset / kernelSize;
                SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
                WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
                DataCopyPad(featureLocal, featureGm[featureOffset * kernelIC], featureCopyParams, featureCopyPadParams);
                DataCopyPad(weightLocal, weightGm[weightOffset * kernelIC * kernelOC], weightCopyParams, weightCopyPadParams);
                SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
                WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
                for (uint32_t ic = 0; ic < kernelIC; ic++) {
                    DTYPE_WEIGHT featureValue = featureLocal.GetValue(ic);
                    Muls(weightTmp[ic * kernelOC], gradLocal[i * kernelOC], featureValue, kernelOC);
                    Mul(weightLocal[ic * kernelOC], weightLocal[ic * kernelOC], gradLocal[i * kernelOC], kernelOC);
                    pipe_barrier(PIPE_V);
                    ReduceSum<DTYPE_FEATURE>(reduceSumTemp, weightLocal[ic * kernelOC], workLocal, kernelOC);
                    SetFlag<HardEvent::V_S>(eventIDVToS);
                    WaitFlag<HardEvent::V_S>(eventIDVToS);
                    featureTmpLocal.SetValue(ic, reduceSumTemp.GetValue(0));
                }
                SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
                WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
                SetAtomicAdd<DTYPE_WEIGHT>();
                DataCopyPad(featureGradGm[featureOffset * kernelIC], featureTmpLocal, featureTmpCopyParams);
                DataCopyPad(weightGradGm[weightOffset * kernelIC * kernelOC], weightTmp, weightTmpCopyParams);
                SetAtomicNone();
            }
        }
        indicesOffsetQueue.FreeTensor(indicesOffsetLocal);
        gradQueue.FreeTensor(gradLocal);
        formerSortedIndicesQueue.FreeTensor(sortLocal);
    }
    __aicore__ inline void calculateReduceSum()
    {
        mulmask = maskAlign;
        if (mulmask > kernelIC) {
            mulmask = kernelIC;
        }
        mulRepeatTimes = DivCeil(kernelIC, mulmask);
        int workSize = AlignUp(mulRepeatTimes, dataAlign);
    }
private:
    TPipe *pipe;
    GlobalTensor<DTYPE_WEIGHT> featureGm, weightGm, gradGm, featureGradGm, weightGradGm;
    GlobalTensor<DTYPE_INDICES_OFFSET> indicesOffsetGm, formerSortedIndicesGm;
    TQue<QuePosition::VECIN, 1> indicesOffsetQueue, gradQueue, formerSortedIndicesQueue, indicesQueue;
    TBuf<TPosition::VECCALC> featureUb, weightUB, reduceSumTmpUB, workUB, featureTmpUB, weightTmpUB;

    uint32_t usedCoreNum;
    uint32_t coreTask;
    uint32_t lastCoreTask;
    uint32_t moveLen;
    uint32_t repeatTimes;
    uint32_t moveTail;
    uint32_t lastRepeatTimes;
    uint32_t lastMoveTail;
    uint32_t kernelSize;
    uint32_t kernelIC;
    uint32_t kernelOC;

    uint32_t blockBytes{32};
    uint32_t curBlockIdx;
    uint32_t dataAlign;
    uint32_t coreRepeatTimes;
    uint32_t coreMoveTail;
    uint32_t maskAlign;
    uint32_t mulmask;
    uint32_t mulRepeatTimes;
    uint32_t workSize;
};

extern "C" __global__ __aicore__ void sparse_conv3d_grad(GM_ADDR indices_offset, GM_ADDR former_sorted_indices,
                                                        GM_ADDR feature, GM_ADDR weight, GM_ADDR grad,
                                                        GM_ADDR feature_grad, GM_ADDR weight_grad,
                                                        GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    KernelSparseConv3dGrad op;
    op.Init(indices_offset, former_sorted_indices, feature, weight, grad,
            feature_grad, weight_grad,
            &tiling_data, &pipe);
    op.Process();
}