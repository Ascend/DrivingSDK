/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

using namespace AscendC;
constexpr uint32_t BUFFER_NUM = 1;

template <typename T>
class AssignScoreWithk {
public:
    __aicore__ inline AssignScoreWithk(GM_ADDR points, GM_ADDR centers, GM_ADDR scores, GM_ADDR knn_idx, GM_ADDR output,
                                            GM_ADDR workspace, const AssignScoreWithkTilingData* __restrict tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "block num can not be zero");

        // points shape: npoint, B, K, O, M
        // centers shape: npoint, B, O, M
        // socres shape: npoint, B, K, O, M
        // output shape: npoint, B, K, O

        batchSize = tiling_data->batch_size;
        nsource = tiling_data->nsource;
        npoint = tiling_data->npoint;
        numWeights = tiling_data->num_weights;
        numNeighbors = tiling_data->num_neighbors;
        numFeatures = tiling_data->num_features;
        aggregate = tiling_data->aggregate;
        dataAlign = ONE_BLK_SIZE / sizeof(T);
        WeightAlign = AlignUp(numWeights, dataAlign);
        weightFeatAlign = numFeatures *  WeightAlign;
        inner = (numWeights * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE / sizeof(T);

        numDataPoint = batchSize * numNeighbors * numWeights * numFeatures;
        numDataPointBatch = numNeighbors * numWeights * numFeatures;
        numDataCenter = batchSize * numWeights * numFeatures;
        numDataCenterBatch = numWeights * numFeatures;
        numDataScore = batchSize * numNeighbors * numWeights * numFeatures;
        numDataScoreBatch = numNeighbors * numWeights * numFeatures;
        numDataOutput = batchSize * numNeighbors * numFeatures;
        numDataOutputBatch = numNeighbors * numFeatures;
        
        uint32_t npointPerCore = tiling_data->npoint_per_core;
        uint32_t npointRemained = tiling_data->npoint_remained;
        uint32_t coreId = GetBlockIdx();
        if (coreId < npointRemained) {
            npointInCore = npointPerCore + 1;
            startPointIdx = coreId * npointInCore;
            endPointIdx = startPointIdx + npointInCore;
        } else {
            npointInCore = npointPerCore;
            startPointIdx = (npointPerCore + 1) * npointRemained + npointPerCore * (coreId - npointRemained);
            endPointIdx = startPointIdx + npointInCore;
        }

        pointsGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(points) + startPointIdx * numDataPoint,
                                    npointInCore * numDataPoint);
        centersGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(centers) + startPointIdx * numDataCenter,
                                    npointInCore * numDataCenter);
        scoresGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(scores) + startPointIdx * numDataScore,
                                    npointInCore * numDataScore);
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output) + startPointIdx * numDataOutput,
                                    npointInCore * numDataOutput);

        pipe.InitBuffer(pointsQue, BUFFER_NUM, weightFeatAlign * sizeof(T));
        pipe.InitBuffer(centersQue, BUFFER_NUM, weightFeatAlign * sizeof(T));
        pipe.InitBuffer(scoresQue, BUFFER_NUM, weightFeatAlign * sizeof(T));
        pipe.InitBuffer(outputQue, BUFFER_NUM, numFeatures * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < npointInCore; i++) {
            for (uint32_t j = 0; j < batchSize; j++) {
                centersLocal = centersQue.AllocTensor<T>();
                DataCopyPad(centersLocal, centersGm[i * numDataCenter + j * numDataCenterBatch],
                            {static_cast<uint16_t>(numFeatures), static_cast<uint32_t>(numWeights * sizeof(T)), 0, 0, 0},
                            {true, 0, static_cast<uint8_t>(WeightAlign - numWeights), 0});
                centersQue.EnQue(centersLocal);
                centersLocal = centersQue.DeQue<T>();
                for (uint32_t k = 0; k < numNeighbors; k++) {
                    pointsLocal = pointsQue.AllocTensor<T>();
                    scoresLocal = scoresQue.AllocTensor<T>();
                    outputLocal = outputQue.AllocTensor<T>();
                    DataCopyPad(pointsLocal, pointsGm[i * numDataPoint + j * numDataPointBatch + k * numWeights * numFeatures],
                                {static_cast<uint16_t>(numFeatures), static_cast<uint32_t>(numWeights * sizeof(T)), 0, 0, 0},
                                {true, 0, static_cast<uint8_t>(WeightAlign - numWeights), 0});
                    DataCopyPad(scoresLocal, scoresGm[i * numDataScore + j * numDataScoreBatch + k * numWeights * numFeatures],
                                {static_cast<uint16_t>(numFeatures), static_cast<uint32_t>(numWeights * sizeof(T)), 0, 0, 0},
                                {true, 0, static_cast<uint8_t>(WeightAlign - numWeights), 0});
                    pointsQue.EnQue(pointsLocal);
                    scoresQue.EnQue(scoresLocal);
                    pointsLocal = pointsQue.DeQue<T>();
                    scoresLocal = scoresQue.DeQue<T>();
                    Sub(pointsLocal, pointsLocal, centersLocal, weightFeatAlign);
                    Mul(pointsLocal, pointsLocal, scoresLocal, weightFeatAlign);
                    Sum(outputLocal, pointsLocal, {numFeatures, inner, numWeights});
                    outputQue.EnQue(outputLocal);
                    outputLocal = outputQue.DeQue<T>();
                    DataCopyPad(outputGm[i * numDataOutput+ j * numDataOutputBatch + k * numFeatures], outputLocal,
                        {1, static_cast<uint32_t>(numFeatures * sizeof(T)), 0, 0, 0});
                    pointsQue.FreeTensor<T>(pointsLocal);
                    scoresQue.FreeTensor<T>(scoresLocal);
                    outputQue.FreeTensor<T>(outputLocal);
                }
                centersQue.FreeTensor<T>(centersLocal);
            }
        }
    }
private:
    TPipe pipe;
    GlobalTensor<T> pointsGm, centersGm, scoresGm, outputGm;
    TQue<TPosition::VECIN, BUFFER_NUM> pointsQue, centersQue, scoresQue;
    TQue<TPosition::VECOUT, BUFFER_NUM> outputQue;
    LocalTensor<T> pointsLocal, centersLocal, scoresLocal, outputLocal, tempLocal;

private:
    uint32_t batchSize;
    uint32_t nsource;
    uint32_t npoint;
    uint32_t numWeights;
    uint32_t numNeighbors;
    uint32_t numFeatures;
    uint32_t npointInCore;
    uint32_t aggregate;
    uint32_t startPointIdx;
    uint32_t endPointIdx;
    uint32_t numDataPoint, numDataPointBatch, numDataPointBatchAlign;
    uint32_t numDataCenter, numDataCenterBatch, numDataCenterBatchAlign;
    uint32_t numDataScore, numDataScoreBatch, numDataScoreBatchAlign;
    uint32_t numDataOutput, numDataOutputBatch, numDataOutputBatchAlign;
    uint32_t dataAlign;
    uint32_t inner;
    uint32_t weightFeatAlign;
    uint32_t WeightAlign;
};

extern "C" __global__ __aicore__ void assign_score_withk(
    GM_ADDR points,
    GM_ADDR centers,
    GM_ADDR scores,
    GM_ADDR knnIdx,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling
)
{
#if CCE_AICORE == 220
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
    GET_TILING_DATA(tilingData, tiling);
    const AssignScoreWithkTilingData* __restrict tiling_data = &tilingData;
    AssignScoreWithk<float> op(points, centers, scores, knnIdx, output, workspace, tiling_data);
    op.Process();
}