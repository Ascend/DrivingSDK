// Copyright (c) 2024 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint32_t IDX_NUM_PER_GROUP = 3;
constexpr uint32_t UB_ALIGN_SIZE = 32;

class KernelVecPoolGrad {
public:
    __aicore__ inline KernelVecPoolGrad() {}

    __aicore__ inline void Init(GM_ADDR grad_new_features, GM_ADDR point_cnt_of_grid,
                                GM_ADDR grouped_idxs, GM_ADDR grad_support_features,
                                const VecPoolGradTilingData& tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->formerCoreGroups = tiling_data.formerCoreGroups;
        this->usedCoreNum = tiling_data.usedCoreNum;
        this->availableUbSize = tiling_data.availableUbSize;
        this->mainGroups = tiling_data.mainGroups;
        this->copyLoop = tiling_data.copyLoop;
        this->copyTail = tiling_data.copyTail;
        this->formerTailGroups = tiling_data.formerTailGroups;
        this->lastCopyLoop = tiling_data.lastCopyLoop;
        this->lastCopyTail = tiling_data.lastCopyTail;
        this->lastTailGroups = tiling_data.lastTailGroups;
        this->m = tiling_data.m;
        this->cOut = tiling_data.cOut;
        this->numTotalGrids = tiling_data.numTotalGrids;
        this->numCEachGrid = tiling_data.numCEachGrid;
        this->gradUBEleNum = tiling_data.gradUBEleNum;
        this->numMaxSumPoints = tiling_data.numMaxSumPoints;
        this->n = tiling_data.n;
        this->cIn = tiling_data.cIn;
        this->repeatTimes = tiling_data.repeatTimes;
        this->tail = tiling_data.tail;
        this->mainCopySize = tiling_data.mainCopySize;
        this->formerCoreTailCopySize = tiling_data.formerCoreTailCopySize;
        this->lastCoreTailCopySize = tiling_data.lastCoreTailCopySize;

        gradNewFeaturesGM.SetGlobalBuffer((__gm__ float*)grad_new_features, this->m * this->cOut);
        pointCntOfGridGM.SetGlobalBuffer((__gm__ int32_t*)point_cnt_of_grid, this->m * this->numTotalGrids);
        groupedIdxsGM.SetGlobalBuffer((__gm__ int32_t*)grouped_idxs, this->numMaxSumPoints * IDX_NUM_PER_GROUP);
        gradSupportFeaturesGM.SetGlobalBuffer((__gm__ float*)grad_support_features, this->n * this->cIn);

        pipe.InitBuffer(inQueueGroupedIdxs, BUFFER_NUM, availableUbSize);
        pipe.InitBuffer(inQueuePointCntOfGrid, BUFFER_NUM, UB_ALIGN_SIZE);
        pipe.InitBuffer(inQueueGradNewFeatures, BUFFER_NUM, this->gradUBEleNum * sizeof(float));
        pipe.InitBuffer(outQueueGradSupportFeatures, BUFFER_NUM, this->gradUBEleNum * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();
        if (coreId > this->usedCoreNum) {
            return;
        }
        if (coreId != (this->usedCoreNum -1)) {
            for (int32_t i = 0; i < this->copyLoop; i++) {
                CopyIn(i, this->mainCopySize);
                Compute(this->mainGroups);
            }
            if (this->copyTail != 0) {
                CopyIn(this->copyLoop, this->formerCoreTailCopySize);
                Compute(this->formerTailGroups);
            }
        } else {
            for (int32_t i = 0; i < this->lastCopyLoop; i++) {
                CopyIn(i, this->mainCopySize);
                Compute(this->mainGroups);
            }
            if (this->lastCopyTail != 0) {
                CopyIn(this->lastCopyLoop, this->lastCoreTailCopySize);
                Compute(this->lastTailGroups);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t copySize)
    {
        LocalTensor<int32_t> groupedIdxsLocal = inQueueGroupedIdxs.AllocTensor<int32_t>();
        DataCopyExtParams groupedIdxsCopyParams{1, copySize, 0, 0, 0};
        DataCopyPadExtParams<int32_t> groupedIdxsPadParams{false, 0, 0, 0};
        DataCopyPad(groupedIdxsLocal, groupedIdxsGM[GetBlockIdx() * this->formerCoreGroups * IDX_NUM_PER_GROUP + progress * this->mainGroups * IDX_NUM_PER_GROUP], groupedIdxsCopyParams, groupedIdxsPadParams);
        pipe_barrier(PIPE_ALL);
        inQueueGroupedIdxs.EnQue(groupedIdxsLocal);
    }

    __aicore__ inline void Compute(uint32_t numGroups)
    {
        LocalTensor<int32_t> groupedIdxsLocal = inQueueGroupedIdxs.DeQue<int32_t>();
        LocalTensor<int32_t> numTotalPtsLocal = inQueuePointCntOfGrid.AllocTensor<int32_t>();
        LocalTensor<float> gradNewFeaturesLocal = inQueueGradNewFeatures.AllocTensor<float>();
        LocalTensor<float> gradSupportFeaturesLocal = outQueueGradSupportFeatures.AllocTensor<float>();
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        for (int32_t i = 0; i < numGroups; i++) {
            int32_t idxOfSupportXyz = groupedIdxsLocal.GetValue(i * IDX_NUM_PER_GROUP);
            int32_t idxOfNewXyz = groupedIdxsLocal.GetValue(i * IDX_NUM_PER_GROUP + 1);
            int32_t idxOfGridIdx = groupedIdxsLocal.GetValue(i * IDX_NUM_PER_GROUP + 2);

            DataCopyExtParams pointCntOfGridCopyParams{1, sizeof(int32_t), 0, 0, 0};
            DataCopyPadExtParams<int32_t> pointCntOfGridPadParams{false, 0, 0, 0};
            DataCopyPad(numTotalPtsLocal, pointCntOfGridGM[idxOfNewXyz * this->numTotalGrids + idxOfGridIdx], pointCntOfGridCopyParams, pointCntOfGridPadParams);
            int32_t num_total_pts = numTotalPtsLocal.GetValue(0);

            float cur_grad = 1 / max(static_cast<float>(num_total_pts), 1.f);

            DataCopyExtParams gradNewFeaturesCopyParams{1, static_cast<uint32_t>(this->numCEachGrid * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> gradNewFeaturesPadParams{false, 0, 0, 0};
            DataCopyPad(gradNewFeaturesLocal, gradNewFeaturesGM[idxOfNewXyz * this->cOut + idxOfGridIdx * this->numCEachGrid], gradNewFeaturesCopyParams, gradNewFeaturesPadParams);
            pipe_barrier(PIPE_ALL);
            Muls(gradSupportFeaturesLocal, gradNewFeaturesLocal, cur_grad, this->numCEachGrid);
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            for (int32_t j = 0; j < this->repeatTimes; j++) {
                SetAtomicAdd<float>();
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->numCEachGrid * sizeof(float)), 0, 0, 0};
                DataCopyPad(gradSupportFeaturesGM[idxOfSupportXyz * this->cIn + j * this->numCEachGrid], gradSupportFeaturesLocal, copyParams);
                SetAtomicNone();
            }
            if (this->tail != 0) {
                SetAtomicAdd<float>();
                DataCopyExtParams copyTailParams{1, static_cast<uint32_t>(this->tail * sizeof(float)), 0, 0, 0};
                DataCopyPad(gradSupportFeaturesGM[idxOfSupportXyz * this->cIn + this->repeatTimes * this->numCEachGrid], gradSupportFeaturesLocal, copyTailParams);
                SetAtomicNone();
            }
        }
        inQueueGroupedIdxs.FreeTensor(groupedIdxsLocal);
        inQueuePointCntOfGrid.FreeTensor(numTotalPtsLocal);
        inQueueGradNewFeatures.FreeTensor(gradNewFeaturesLocal);
        outQueueGradSupportFeatures.FreeTensor(gradSupportFeaturesLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGradNewFeatures, inQueuePointCntOfGrid, inQueueGroupedIdxs;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueGradSupportFeatures;
    GlobalTensor<float> gradNewFeaturesGM, gradSupportFeaturesGM;
    GlobalTensor<int32_t> pointCntOfGridGM, groupedIdxsGM;

    uint32_t formerCoreGroups;
    uint32_t usedCoreNum;
    uint32_t availableUbSize;
    uint32_t mainGroups;
    uint32_t copyLoop;
    uint32_t copyTail;
    uint32_t formerTailGroups;
    uint32_t lastCopyLoop;
    uint32_t lastCopyTail;
    uint32_t lastTailGroups;
    uint32_t m;
    uint32_t cOut;
    uint32_t numTotalGrids;
    uint32_t numCEachGrid;
    uint32_t gradUBEleNum;
    uint32_t numMaxSumPoints;
    uint32_t n;
    uint32_t cIn;
    uint32_t repeatTimes;
    uint32_t tail;
    uint32_t mainCopySize;
    uint32_t formerCoreTailCopySize;
    uint32_t lastCoreTailCopySize;
};


extern "C" __global__ __aicore__ void vec_pool_grad(GM_ADDR grad_new_features, GM_ADDR point_cnt_of_grid, GM_ADDR grouped_idxs, GM_ADDR grad_support_features, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelVecPoolGrad op;
    op.Init(grad_new_features, point_cnt_of_grid, grouped_idxs, grad_support_features, tiling_data);
    op.Process();
}


#ifndef __CCE_KT_TEST__
// call of kernel function
void vec_pool_grad_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* grad_new_features, uint8_t* point_cnt_of_grid, uint8_t* grouped_idxs, uint8_t* grad_support_features)
{
    vec_pool_grad<<<blockDim, l2ctrl, stream>>>(grad_new_features, point_cnt_of_grid, grouped_idxs, grad_support_features);
}
#endif
