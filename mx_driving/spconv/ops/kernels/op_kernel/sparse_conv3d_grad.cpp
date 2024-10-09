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

        initTilingData(tiling_data);

        uint32_t valueBlockNum = blockBytes / sizeof(DTYPE_WEIGHT);
        uint32_t idxBlockNum = blockBytes / sizeof(DTYPE_INDICES_OFFSET);
        kernelSizeAlign = AlignUp(kernelSize, idxBlockNum);
        repeatOffset = repeatBlockByte / sizeof(DTYPE_FEATURE);
        kernelOCBlock = kernelOC / idxBlockNum;
        wholeRedusumMask = kernelOC;
        if (wholeRedusumMask > repeatOffset) {
            wholeRedusumMask = repeatOffset;
        }

        uint64_t beginOffset = curBlockIdx * coreTask;

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
        gradGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_GRAD *>(grad) + beginOffset * kernelOC);

        featureGradGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_FEATURE *>(feature_grad));
        weightGradGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_WEIGHT *>(weight_grad));

        pipe->InitBuffer(indicesOffsetQueue, BUFFER_NUM, AlignUp(moveLen + 1, idxBlockNum) * sizeof(DTYPE_INDICES_OFFSET));
        pipe->InitBuffer(formerSortedIndicesQueue, BUFFER_NUM, moveLen * kernelSizeAlign * sizeof(DTYPE_INDICES_OFFSET));
        pipe->InitBuffer(gradQueue, BUFFER_NUM, kernelOC * sizeof(DTYPE_WEIGHT));

        pipe->InitBuffer(featureUb, AlignUp(kernelIC, valueBlockNum) * sizeof(DTYPE_WEIGHT));
        pipe->InitBuffer(gradBroadTmpUB, kernelIC * kernelOC * sizeof(DTYPE_WEIGHT));
        pipe->InitBuffer(mulTmpUB, kernelIC * kernelOC * sizeof(DTYPE_WEIGHT));
        pipe->InitBuffer(reduceSumTmpUB, AlignUp(kernelIC, valueBlockNum) * sizeof(DTYPE_WEIGHT));
    }

    __aicore__ inline void Process()
    {
        SetAtomicAdd<DTYPE_WEIGHT>();
        for (uint32_t i = 0; i < coreRepeatTimes; i++) {
            Compute(i);
            pipe_barrier(PIPE_ALL);
        }
        SetAtomicNone();
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
        DataCopyExtParams gradCopyParams {1, (uint32_t)(kernelOC * sizeof(DTYPE_WEIGHT)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_WEIGHT> gradCopyPadParams{true, 0, 0, 0};
        DataCopyExtParams featureCopyParams {1, (uint32_t)(kernelIC * sizeof(DTYPE_WEIGHT)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_WEIGHT> featureCopyPadParams{true, 0, 0, 0};
        DataCopyExtParams weightCopyParams {1, (uint32_t)(kernelIC * kernelOC * sizeof(DTYPE_WEIGHT)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_WEIGHT> weightCopyPadParams{true, 0, 0, 0};

        DataCopyExtParams featureTmpCopyParams {1, (uint32_t)(uint32_t)(kernelIC * sizeof(DTYPE_WEIGHT)), 0, 0, 0};
        DataCopyExtParams weightTmpCopyParams {1, (uint32_t)(kernelIC * kernelOC * sizeof(DTYPE_WEIGHT)), 0, 0, 0};

        event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

        event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));

        LocalTensor<DTYPE_INDICES_OFFSET> indicesOffsetLocal = indicesOffsetQueue.AllocTensor<DTYPE_INDICES_OFFSET>();
        LocalTensor<DTYPE_INDICES_OFFSET> sortLocal = formerSortedIndicesQueue.AllocTensor<DTYPE_INDICES_OFFSET>();

        LocalTensor<DTYPE_WEIGHT> gradLocal = gradQueue.AllocTensor<DTYPE_WEIGHT>();
        LocalTensor<DTYPE_WEIGHT> gradBroadTmp = gradBroadTmpUB.Get<DTYPE_WEIGHT>();

        LocalTensor<DTYPE_WEIGHT> featureLocal = featureUb.Get<DTYPE_WEIGHT>();
        LocalTensor<DTYPE_WEIGHT> mulTemp = mulTmpUB.Get<DTYPE_WEIGHT>();
        LocalTensor<DTYPE_WEIGHT> reduceSumTemp = reduceSumTmpUB.Get<DTYPE_WEIGHT>();

        uint32_t srcShapeFeature[2] = {kernelIC, 1};
        uint32_t srcShapeGrad[2] = {1, kernelOC};
        uint32_t dstShape_[2] = {kernelIC, kernelOC};

        SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
        WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
        DataCopyPad(indicesOffsetLocal, indicesOffsetGm[taskOffset], indicesOffsetCopyParams, indicesOffsetPadParams);
        pipe_barrier(PIPE_MTE2);

        DataCopyPadExtParams<DTYPE_INDICES_OFFSET> sortPadParams{true, 0, 0, 0};

        for (uint32_t idx = 0; idx < forMoveLen; idx++) {
            uint32_t beginIndicesOffset = indicesOffsetLocal.GetValue(idx);
            uint32_t endIndicesOffset = indicesOffsetLocal.GetValue(idx + 1);
            DataCopyExtParams sortCopyParams {1, (uint32_t)((endIndicesOffset - beginIndicesOffset) * sizeof(DTYPE_INDICES_OFFSET)), 0, 0, 0};
            DataCopyPad(sortLocal[idx * kernelSizeAlign], formerSortedIndicesGm[beginIndicesOffset], sortCopyParams, sortPadParams);
        }
        pipe_barrier(PIPE_MTE2);

        for (uint32_t idx = 0; idx < forMoveLen; idx++) {
            uint32_t beginIndicesOffset = indicesOffsetLocal.GetValue(idx);
            uint32_t endIndicesOffset = indicesOffsetLocal.GetValue(idx + 1);
            DataCopyPad(gradLocal, gradGm[(taskOffset + idx) * kernelOC], gradCopyParams, gradCopyPadParams);
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            BroadCast<DTYPE_FEATURE, 2, 0>(gradBroadTmp, gradLocal, dstShape_, srcShapeGrad);
            for (uint32_t j = 0; j < endIndicesOffset - beginIndicesOffset; j++) {
                uint32_t sortOffset = sortLocal.GetValue(j + idx * kernelSizeAlign);
                uint32_t featureOffset = sortOffset / kernelSize;
                uint32_t weightOffset = sortOffset % kernelSize;

                DataCopyPad(mulTemp, weightGm[weightOffset * kernelIC * kernelOC], weightCopyParams, weightCopyPadParams);

                SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);

                Mul(mulTemp, mulTemp, gradBroadTmp, kernelIC * kernelOC);

                if (kernelOC > repeatOffset) {
                    WholeReduceSum(reduceSumTemp, mulTemp[repeatOffset], kernelOC - repeatOffset, kernelIC, 1, 1, kernelOCBlock);
                    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                    DataCopyPad(featureGradGm[featureOffset * kernelIC], reduceSumTemp, featureTmpCopyParams);
                    SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
                    WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
                }
                WholeReduceSum(reduceSumTemp, mulTemp, wholeRedusumMask, kernelIC, 1, 1, kernelOCBlock);
                SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                DataCopyPad(featureGradGm[featureOffset * kernelIC], reduceSumTemp, featureTmpCopyParams);
                SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);

                DataCopyPad(featureLocal, featureGm[featureOffset * kernelIC], featureCopyParams, featureCopyPadParams);
                SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
                BroadCast<DTYPE_FEATURE, 2, 1>(mulTemp, featureLocal, dstShape_, srcShapeFeature);
                Mul(mulTemp, mulTemp, gradBroadTmp, kernelIC * kernelOC);
                SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                DataCopyPad(weightGradGm[weightOffset * kernelIC * kernelOC], mulTemp, weightTmpCopyParams);
                SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            }
        }

        indicesOffsetQueue.FreeTensor(indicesOffsetLocal);
        formerSortedIndicesQueue.FreeTensor(sortLocal);
        gradQueue.FreeTensor(gradLocal);
    }

private:
    TPipe *pipe;
    GlobalTensor<DTYPE_WEIGHT> featureGm, weightGm, gradGm, featureGradGm, weightGradGm;
    GlobalTensor<DTYPE_INDICES_OFFSET> indicesOffsetGm, formerSortedIndicesGm;
    TQue<QuePosition::VECIN, 1> indicesOffsetQueue, formerSortedIndicesQueue, gradQueue;
    TBuf<TPosition::VECCALC> featureUb, gradBroadTmpUB, mulTmpUB, reduceSumTmpUB;

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

    uint32_t kernelSizeAlign;
    uint32_t kernelOCBlock;
    uint32_t repeatOffset;
    uint32_t wholeRedusumMask;

    uint32_t blockBytes{32};
    uint32_t repeatBlockByte{256};
    uint32_t curBlockIdx;
    uint32_t coreRepeatTimes;
    uint32_t coreMoveTail;
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