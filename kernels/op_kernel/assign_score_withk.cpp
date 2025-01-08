/*
Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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
                                       GM_ADDR workspace, const AssignScoreWithkTilingData* tilingData)
    {
        ASSERT(GetBlockNum() != 0 && "block num can not be zero");

        batchSize = tilingData->batchSize;
        nsource = tilingData->nsource;
        npoint = tilingData->npoint;
        numWeights= tilingData->numWeights;
        numNeighbors = tilingData->numNeighbors;
        numFeatures= tilingData->numFeatures;
        aggregate = tilingData->aggregate;
        dataAlign = ONE_BLK_SIZE / sizeof(T);
        weightsAlign= AlignUp(numWeights, dataAlign);
        inner = (numWeights * sizeof(T) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE / sizeof(T);
        ndataPerCore = tilingData->npointPerCore;
        ndataRemained = tilingData->npointRemained;

        coreId = GetBlockIdx();
        if (coreId < ndataRemained) {
            ndataInCore = ndataPerCore + 1;
            startDataIdx = ndataInCore * coreId;
            endDataIdx = startDataIdx + ndataInCore;
        } else {
            ndataInCore = ndataPerCore;
            startDataIdx = (ndataPerCore + 1) * ndataRemained + ndataPerCore * (coreId - ndataRemained);
            endDataIdx = startDataIdx + ndataInCore;
        }

        startBatchIdx = startDataIdx / (numFeatures * npoint);
        startFeatureIdx = (startDataIdx - startBatchIdx * numFeatures * npoint) / npoint;
        startPointIdx = startDataIdx - startBatchIdx * numFeatures * npoint - startFeatureIdx *npoint;
        numBatchInCore = (startFeatureIdx * npoint + startPointIdx + ndataInCore + numFeatures * npoint - 1) / (numFeatures * npoint) - startBatchIdx;
        numFeaturesInCore = (startPointIdx + ndataInCore + npoint - 1) / npoint;

        pointsGm.SetGlobalBuffer((__gm__ T *)points + startBatchIdx * numFeatures * nsource * numWeights + startFeatureIdx * nsource * numWeights,
                                numFeaturesInCore * nsource * numWeights);
        centersGm.SetGlobalBuffer((__gm__ T *)centers + startBatchIdx * numFeatures * nsource * numWeights + startFeatureIdx * nsource * numWeights,
                                numFeaturesInCore * nsource * numWeights);
        scoresGm.SetGlobalBuffer((__gm__ T *)scores + startBatchIdx * npoint * numNeighbors * numWeights,
                                numBatchInCore * npoint * numNeighbors * numWeights);
        knnIdxGm.SetGlobalBuffer((__gm__ int64_t *)knn_idx + startBatchIdx * npoint * numNeighbors,
                                numBatchInCore * npoint * numNeighbors);
        outputGm.SetGlobalBuffer((__gm__ T *)output + startDataIdx * numNeighbors,
                                ndataInCore * numNeighbors);

        pipe.InitBuffer(pointsQue, BUFFER_NUM, weightsAlign * sizeof(T));
        pipe.InitBuffer(centersQue, BUFFER_NUM, weightsAlign * sizeof(T));
        pipe.InitBuffer(scoresQue, BUFFER_NUM,  weightsAlign * sizeof(T));
        pipe.InitBuffer(knnIdxQue, BUFFER_NUM, numNeighbors * sizeof(int64_t));
        pipe.InitBuffer(outputQue, BUFFER_NUM, numNeighbors * sizeof(T));
        pipe.InitBuffer(tempBuf, weightsAlign * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < ndataInCore; i++) {
            uint32_t batchIdx = (startFeatureIdx * npoint + startPointIdx + i) / (numFeatures * npoint);
            uint32_t featureIdx = (startPointIdx % npoint + i) / npoint;
            uint32_t pointIdx = (startPointIdx + i) % npoint;

            knnIdxLocal = knnIdxQue.AllocTensor<int64_t>();
            centersLocal = centersQue.AllocTensor<T>();
            pointsLocal = pointsQue.AllocTensor<T>();
            scoresLocal = scoresQue.AllocTensor<T>();
            outputLocal = outputQue.AllocTensor<T>();
            tempLocal = tempBuf.Get<T>();

            DataCopyPad(knnIdxLocal, knnIdxGm[batchIdx * npoint * numNeighbors + pointIdx * numNeighbors],
                        {1, static_cast<uint32_t>(numNeighbors * sizeof(int64_t)), 0, 0, 0},
                        {false, 0, 0, 0});
            knnIdxQue.EnQue(knnIdxLocal);
            knnIdxLocal = knnIdxQue.DeQue<int64_t>();

            DataCopyPad(centersLocal, centersGm[featureIdx * nsource * numWeights + knnIdxLocal.GetValue(0) * numWeights],
                        {1, static_cast<uint32_t>(numWeights * sizeof(T)), 0, 0, 0},
                        {true, 0, static_cast<uint8_t>(weightsAlign - numWeights), 0});
            centersQue.EnQue(centersLocal);
            centersLocal = centersQue.DeQue<T>();

            for (uint32_t k = 0; k < numNeighbors; k++) {
                uint32_t idx = knnIdxLocal.GetValue(k);
                DataCopyPad(pointsLocal, pointsGm[featureIdx * nsource * numWeights + idx * numWeights],
                            {1, static_cast<uint32_t>(numWeights * sizeof(T)), 0, 0, 0},
                            {true, 0, static_cast<uint8_t>(weightsAlign - numWeights), 0});
                pointsQue.EnQue(pointsLocal);

                DataCopyPad(scoresLocal, scoresGm[batchIdx * npoint * numNeighbors * numWeights + pointIdx * numNeighbors * numWeights + k * numWeights],
                            {1, static_cast<uint32_t>(numWeights * sizeof(T)), 0, 0, 0},
                            {true, 0, static_cast<uint8_t>(weightsAlign - numWeights), 0});
                scoresQue.EnQue(scoresLocal);

                pointsLocal = pointsQue.DeQue<T>();
                scoresLocal = scoresQue.DeQue<T>();
                Sub(pointsLocal, pointsLocal, centersLocal, weightsAlign);
                Mul(pointsLocal, pointsLocal, scoresLocal, weightsAlign);
                Sum(tempLocal, pointsLocal, {1, inner, numWeights});
                outputLocal.SetValue(k, tempLocal.GetValue(0));
            }
            outputQue.EnQue(outputLocal);
            outputLocal = outputQue.DeQue<T>();
            DataCopyPad(outputGm[i * numNeighbors], outputLocal,
                {1, static_cast<uint32_t>(numNeighbors * sizeof(T)), 0, 0, 0});

            centersQue.FreeTensor<T>(centersLocal);
            knnIdxQue.FreeTensor<int64_t>(knnIdxLocal);
            pointsQue.FreeTensor<T>(pointsLocal);
            scoresQue.FreeTensor<T>(scoresLocal);
            outputQue.FreeTensor<T>(outputLocal);
        }
    }

private:
    TPipe pipe;
    GlobalTensor<T> pointsGm, centersGm, scoresGm, outputGm;
    GlobalTensor<int64_t> knnIdxGm;
    TQue<TPosition::VECIN, BUFFER_NUM> pointsQue, centersQue, scoresQue, knnIdxQue;
    TQue<TPosition::VECOUT, BUFFER_NUM> outputQue;
    TBuf<TPosition::VECCALC> tempBuf;
    LocalTensor<T> pointsLocal, centersLocal, scoresLocal, outputLocal, tempLocal;
    LocalTensor<int64_t> knnIdxLocal;

private:
    uint32_t aggregate;
    uint32_t batchSize;
    uint32_t nsource;
    uint32_t npoint;
    uint32_t numWeights;
    uint32_t numNeighbors;
    uint32_t numFeatures;
    uint64_t ndataInCore;
    uint32_t coreId;
    uint64_t ndataPerCore, ndataRemained, startDataIdx, endDataIdx;
    uint32_t inner;
    uint32_t dataAlign;
    uint32_t weightsFeatsAlign;
    uint32_t weightsAlign;
    uint64_t startBatchIdx;
    uint64_t startFeatureIdx;
    uint64_t startPointIdx;
    uint64_t numFeaturesInCore;
    uint64_t numBatchInCore;
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
    AssignScoreWithk<float> op(points, centers, scores, knnIdx, output, workspace, &tilingData);
    op.Process();
}