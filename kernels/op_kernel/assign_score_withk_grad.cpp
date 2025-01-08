/*
Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
*/
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"
using namespace AscendC;
constexpr uint32_t BUFFER_NUM = 1;

template <typename T>
class AssignScoreWithkGrad {
public:
    __aicore__ inline AssignScoreWithkGrad(TPipe* pipe, GM_ADDR grad_out, GM_ADDR points, GM_ADDR centers, GM_ADDR scores, GM_ADDR knn_idx, GM_ADDR gradScores, GM_ADDR gradPoints, GM_ADDR gradCenters,
                                        GM_ADDR workspace, const AssignScoreWithkTilingData* tilingData)
    {
        ASSERT(GetBlockNum() != 0 && "block num can not be zero");
        InitTask(tilingData);
        InitGM(grad_out, points, centers, scores, knn_idx, gradScores, gradPoints, gradCenters);
        InitBuffer(pipe);
    }
    
    __aicore__ inline void InitTask(const AssignScoreWithkTilingData* tilingData)
    {
        batchSize = tilingData->batchSize;
        nsource = tilingData->nsource;
        npoint = tilingData->npoint;
        numWeights= tilingData->numWeights;
        numNeighbors = tilingData->numNeighbors;
        numFeatures= tilingData->numFeatures;
        aggregate = tilingData->aggregate;
        dataAlign = ONE_BLK_SIZE / sizeof(T);
        featureAlign = AlignUp(numFeatures, dataAlign);

        ndataPerCore = tilingData->npointPerCore;
        ndataRemained = tilingData->npointRemained;

        coreId = GetBlockIdx();
        if (coreId < ndataRemained) {
            ndataInCore = ndataPerCore + 1;
            startDataIdx = coreId * ndataInCore;
        } else {
            ndataInCore = ndataPerCore;
            startDataIdx = (ndataPerCore + 1) * ndataRemained + ndataPerCore * (coreId - ndataRemained);
        }

        startBatchIdx = startDataIdx / npoint;
        numBatchInCore = (startDataIdx + ndataInCore + npoint - 1) / npoint - startBatchIdx;
    }

    __aicore__ inline void InitGM(GM_ADDR grad_out, GM_ADDR points, GM_ADDR centers, GM_ADDR scores, GM_ADDR knn_idx, GM_ADDR gradScores, GM_ADDR gradPoints, GM_ADDR gradCenters)
    {
        pointsGm.SetGlobalBuffer((__gm__ T *)points + startBatchIdx * nsource * numWeights * numFeatures,
                                                numBatchInCore * nsource * numWeights * numFeatures);

        centersGm.SetGlobalBuffer((__gm__ T *)centers + startBatchIdx * nsource * numWeights * numFeatures,
                                                numBatchInCore * nsource * numWeights * numFeatures);

        scoresGm.SetGlobalBuffer((__gm__ T *)scores + startDataIdx * numNeighbors * numWeights,
                                                ndataInCore * numNeighbors * numWeights);

        knnIdxGm.SetGlobalBuffer((__gm__ int64_t *)knn_idx + startDataIdx * numNeighbors,
                                                ndataInCore * numNeighbors);

        gradOutGm.SetGlobalBuffer((__gm__ T *)grad_out + startDataIdx * numNeighbors * numFeatures,
                                                ndataInCore * numNeighbors * numFeatures);

        gradScoresGm.SetGlobalBuffer((__gm__ T *)gradScores + startDataIdx * numNeighbors * numWeights,
                                                ndataInCore * numNeighbors * numWeights);
        
        gradPointsGm.SetGlobalBuffer((__gm__ T *)gradPoints + startBatchIdx * nsource * numWeights * numFeatures,
                                                    numBatchInCore * nsource * numWeights * numFeatures);
        
        gradCentersGm.SetGlobalBuffer((__gm__ T *)gradCenters + startBatchIdx * nsource * numWeights * numFeatures,
                                                    numBatchInCore * nsource * numWeights * numFeatures);
    }

    __aicore__ inline void InitBuffer(TPipe* pipe)
    {
        pipe->InitBuffer(pointsQue, BUFFER_NUM, numWeights * featureAlign * sizeof(T));
        pipe->InitBuffer(centersQue, BUFFER_NUM, numWeights * featureAlign * sizeof(T));
        pipe->InitBuffer(scoresQue, BUFFER_NUM, numWeights * sizeof(T));
        pipe->InitBuffer(knnIdxQue, BUFFER_NUM, numNeighbors * sizeof(int64_t));
        pipe->InitBuffer(gradOutQue, BUFFER_NUM, featureAlign * sizeof(T));

        pipe->InitBuffer(gradScoresQue, BUFFER_NUM, numWeights * sizeof(T));
        pipe->InitBuffer(gradPointsQue, BUFFER_NUM, numWeights * numFeatures * sizeof(T));
        pipe->InitBuffer(gradCentersQue, BUFFER_NUM, numWeights * numFeatures * sizeof(T));
        
        pipe->InitBuffer(tempBuf, numWeights * featureAlign * sizeof(T));
        pipe->InitBuffer(tempBuf1, numWeights * featureAlign * sizeof(T));
        pipe->InitBuffer(tempBuf2, numWeights * featureAlign * sizeof(T));
    }

    __aicore__ inline void Process(AssignScoreWithkTilingData* tilingData)
    {
        for (uint32_t taskId = 0; taskId < ndataInCore; taskId++) {
            uint32_t batchId = (taskId + startDataIdx) / npoint - startBatchIdx;
            knnIdxLocal = knnIdxQue.AllocTensor<int64_t>();
            centersLocal = centersQue.AllocTensor<T>();
            DataCopyPad(knnIdxLocal, knnIdxGm[taskId * numNeighbors],
                        {1, static_cast<uint32_t>(numNeighbors * sizeof(int64_t)), 0, 0, 0},
                        {false, 0, 0, 0});
            knnIdxQue.EnQue(knnIdxLocal);
            knnIdxLocal = knnIdxQue.DeQue<int64_t>();

            DataCopyPad(centersLocal, centersGm[batchId * nsource * numWeights * numFeatures + knnIdxLocal.GetValue(0) * numWeights * numFeatures],
                        {static_cast<uint16_t>(numWeights), static_cast<uint32_t>(numFeatures * sizeof(T)), 0, 0, 0},
                        {true, 0, static_cast<uint8_t>(featureAlign - numFeatures), 0});
            centersQue.EnQue(centersLocal);
            centersLocal = centersQue.DeQue<T>();

            for (uint32_t k = 0; k < numNeighbors; k++) {
                CopyIn(taskId, batchId, k);
                Compute(tilingData);
                CopyOut(taskId, batchId, k);
            }
            
            knnIdxQue.FreeTensor<int64_t>(knnIdxLocal);
            centersQue.FreeTensor<T>(centersLocal);
        }
    }

    __aicore__ inline void CopyIn(uint32_t taskId, uint32_t batchId, uint32_t k)
    {
        uint32_t idx = knnIdxLocal.GetValue(k);
        pointsLocal = pointsQue.AllocTensor<T>();
        gradOutLocal = gradOutQue.AllocTensor<T>();
        scoresLocal = scoresQue.AllocTensor<T>();
        DataCopyPad(pointsLocal, pointsGm[batchId * nsource * numWeights * numFeatures + idx * numWeights * numFeatures],
                    {static_cast<uint16_t>(numWeights), static_cast<uint32_t>(numFeatures * sizeof(T)), 0, 0, 0},
                    {true, 0, static_cast<uint8_t>(featureAlign - numFeatures), 0});
        pointsQue.EnQue(pointsLocal);

        DataCopyPad(gradOutLocal, gradOutGm[taskId * numNeighbors * numFeatures + k * numFeatures],
                    {1, static_cast<uint32_t>(numFeatures * sizeof(T)), 0, 0, 0},
                    {true, 0, static_cast<uint8_t>(featureAlign - numFeatures), 0});
        gradOutQue.EnQue(gradOutLocal);

        DataCopyPad(scoresLocal, scoresGm[taskId * numNeighbors * numWeights + k * numWeights],
                    {1, static_cast<uint32_t>(numWeights * sizeof(T)), 0, 0, 0},
                    {false, 0, 0, 0});
        scoresQue.EnQue(scoresLocal);
    }

    __aicore__ inline void Compute(AssignScoreWithkTilingData* tilingData)
    {
        srcShape[0][0] = 1;
        srcShape[0][1] = featureAlign;
        dstShape[0][0] = numWeights;
        dstShape[0][1] = featureAlign;

        srcShape[1][0] = numWeights;
        srcShape[1][1] = 1;
        dstShape[1][0] = numWeights;
        dstShape[1][1] = featureAlign;

        gradOutLocal = gradOutQue.DeQue<T>();
        ComputeGradScore();
        ComputeGradPointsAndCenters(tilingData);
        gradOutQue.FreeTensor<T>(gradOutLocal);
    }

    __aicore__ inline void ComputeGradScore()
    {
        pointsLocal = pointsQue.DeQue<T>();
        gradScoresLocal = gradScoresQue.AllocTensor<T>();
        tempLocal = tempBuf.Get<T>();

        Sub(pointsLocal, pointsLocal, centersLocal, numWeights * featureAlign);
        BroadCast<T, 2, 0>(tempLocal, gradOutLocal, dstShape[0], srcShape[0]);
        Mul(pointsLocal, pointsLocal, tempLocal, numWeights * featureAlign);
        Sum(gradScoresLocal, pointsLocal, {numWeights, featureAlign, featureAlign});
        gradScoresQue.EnQue(gradScoresLocal);
        pointsQue.FreeTensor<T>(pointsLocal);
    }

    __aicore__ inline void ComputeGradPointsAndCenters(AssignScoreWithkTilingData* tilingData)
    {
        scoresLocal = scoresQue.DeQue<T>();
        tempLocal1 = tempBuf1.Get<T>(numWeights * numFeatures);
        tempLocal2 = tempBuf2.Get<T>(numWeights * numFeatures);
        gradPointsLocal = gradPointsQue.AllocTensor<T>();
        gradCentersLocal = gradCentersQue.AllocTensor<T>();

        BroadCast<T, 2, 1>(tempLocal1, scoresLocal, dstShape[1], srcShape[1]);
        BroadCast<T, 2, 0>(tempLocal2, gradOutLocal, dstShape[0], srcShape[0]);
        Mul(tempLocal1, tempLocal1, tempLocal2, numWeights * featureAlign);

        UnPadParams UnPadParams;
        UnPadParams.rightPad = featureAlign - numFeatures;
        UnPad(gradPointsLocal, tempLocal1, UnPadParams, tilingData->unpadTilingData);

        Muls(gradCentersLocal, gradPointsLocal, (T)-1.0, numWeights * numFeatures);
        gradPointsQue.EnQue(gradPointsLocal);
        gradCentersQue.EnQue(gradCentersLocal);
        
        scoresQue.FreeTensor<T>(scoresLocal);
    }

    __aicore__ inline void CopyOut(uint32_t taskId, uint32_t batchId, uint32_t k)
    {
        uint32_t idx = knnIdxLocal.GetValue(k);
        uint32_t idx0 = knnIdxLocal.GetValue(0);
        gradScoresLocal = gradScoresQue.DeQue<T>();
        gradPointsLocal = gradPointsQue.DeQue<T>();
        gradCentersLocal = gradCentersQue.DeQue<T>();
        DataCopyPad(gradScoresGm[taskId * numNeighbors * numWeights + k * numWeights], gradScoresLocal,
                    {1, static_cast<uint32_t>(numWeights * sizeof(T)), 0, 0, 0});
        SetAtomicAdd<T>();
        DataCopyPad(gradPointsGm[batchId * nsource * numWeights * numFeatures + idx * numWeights * numFeatures], gradPointsLocal,
                    {1, static_cast<uint32_t>(numWeights * numFeatures * sizeof(T)), 0, 0, 0});
        DataCopyPad(gradCentersGm[batchId * nsource * numWeights * numFeatures + idx0 * numWeights * numFeatures], gradCentersLocal,
                    {1, static_cast<uint32_t>(numWeights * numFeatures * sizeof(T)), 0, 0, 0});
        SetAtomicNone();
        gradScoresQue.FreeTensor<T>(gradScoresLocal);
        gradPointsQue.FreeTensor<T>(gradPointsLocal);
        gradCentersQue.FreeTensor<T>(gradCentersLocal);
    }

private:
    GlobalTensor<T> pointsGm, centersGm, scoresGm, gradOutGm, gradScoresGm, gradPointsGm, gradCentersGm;
    GlobalTensor<int64_t> knnIdxGm;
    LocalTensor<T> pointsLocal, centersLocal, scoresLocal, gradOutLocal, tempLocal, tempLocal1, tempLocal2;
    LocalTensor<T> gradScoresLocal, gradPointsLocal, gradCentersLocal;
    LocalTensor<int64_t> knnIdxLocal;

    TQue<TPosition::VECIN, BUFFER_NUM> pointsQue, centersQue, scoresQue, knnIdxQue, gradOutQue;
    TQue<TPosition::VECOUT, BUFFER_NUM> gradScoresQue, gradPointsQue, gradCentersQue;
    TBuf<TPosition::VECCALC> tempBuf, tempBuf1, tempBuf2;

    uint32_t aggregate;
    uint32_t batchSize;
    uint32_t nsource;
    uint32_t npoint;
    uint32_t numWeights;
    uint32_t numNeighbors;
    uint32_t numFeatures;
    uint64_t ndataInCore;
    uint32_t coreId;
    uint64_t ndataPerCore, ndataRemained, startDataIdx;
    uint32_t dataAlign;
    uint32_t weightsFeatsAlign;
    uint32_t featureAlign;
    uint64_t startBatchIdx;
    uint64_t numBatchInCore;
    uint32_t dstShape[2][2];
    uint32_t srcShape[2][2];
};

extern "C" __global__ __aicore__ void assign_score_withk_grad(
    GM_ADDR grad_out,
    GM_ADDR points,
    GM_ADDR centers,
    GM_ADDR scores,
    GM_ADDR knnIdx,
    GM_ADDR gradScores,
    GM_ADDR gradPoints,
    GM_ADDR gradCenters,
    GM_ADDR workspace,
    GM_ADDR tiling
)
{
#if CCE_AICORE == 220
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    AssignScoreWithkGrad<float> op(&pipe, grad_out, points, centers, scores, knnIdx, gradScores, gradPoints, gradCenters, workspace, &tilingData);
    op.Process(&tilingData);
}
