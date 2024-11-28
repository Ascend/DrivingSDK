/*
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
*/
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

class KernelPixelGroup {
public:
    __aicore__ inline KernelPixelGroup() {}
    __aicore__ inline void Init(GM_ADDR score, GM_ADDR mask, GM_ADDR embedding, GM_ADDR kernel_label, GM_ADDR kernel_contour,
                                GM_ADDR point_vector, GM_ADDR label_updated, GM_ADDR workspace, const PixelGroupTilingData* tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        usedCoreNum = tiling_data->core_used;
        totalPixels = tiling_data->total_pixels;
        averagePixels = tiling_data->average_pixels;
        pixelLast = tiling_data->pixel_last;
        embeddingDim = tiling_data->embedding_dim;
        dimAlign = tiling_data->dim_align;
        kernelRegionNum = tiling_data->kernel_region_num;
        distanceThreshold = tiling_data->distance_threshold;
        availableUbSize = tiling_data->available_ub_size;
        loopTimeFront = tiling_data->loop_time_front;
        lastLoopFront = tiling_data->last_loop_front;
        loopTimeRear = tiling_data->loop_time_rear;
        lastLoopRear = tiling_data->last_loop_rear;
        if (embeddingDim % 8 != 0) paddingNum = dimAlign - embeddingDim;

        // GM
        scoreGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_SCORE *>(score), totalPixels);
        maskGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_KERNEL_CONTOUR *>(mask), totalPixels);
        embeddingGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_EMBEDDING *>(embedding), totalPixels * embeddingDim);
        labelGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_KERNEL_LABEL *>(kernel_label), totalPixels);
        contourGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_KERNEL_CONTOUR *>(kernel_contour), totalPixels);
        kernelVectorGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_EMBEDDING *>(workspace));
        countGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_SCORE *>(workspace) + kernelRegionNum * embeddingDim);
        labelUpdatedGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_KERNEL_LABEL *>(label_updated), totalPixels);
        pointVectorGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_POINT_VECTOR *>(point_vector), kernelRegionNum * 2);

        InitGlobalMemory(kernelVectorGm, kernelRegionNum * embeddingDim, float(0));
        InitGlobalMemory(countGm, kernelRegionNum * embeddingDim, float(0));
        eventIdMte2ToV = static_cast<event_t>(pipe.AllocEventID<HardEvent::MTE2_V>());

        // TQue
        pipe.InitBuffer(inQueueScore, BUFFER_NUM, availableUbSize * sizeof(float));
        pipe.InitBuffer(inQueueMask, BUFFER_NUM, Ceil((availableUbSize * sizeof(uint8_t)), ONE_BLK_SIZE));
        pipe.InitBuffer(inQueueEmbedding, BUFFER_NUM, availableUbSize * dimAlign * sizeof(float));
        pipe.InitBuffer(inQueueLabel, BUFFER_NUM, availableUbSize * sizeof(int32_t));

        // TBuf
        pipe.InitBuffer(labelBuf, availableUbSize * sizeof(float));
        pipe.InitBuffer(kernelLabelBuf, availableUbSize * sizeof(float));
        pipe.InitBuffer(kernelVectorBuf, dimAlign * sizeof(float));
        pipe.InitBuffer(numBuf, availableUbSize * sizeof(float));
        pipe.InitBuffer(scoreBuf, availableUbSize * sizeof(float));
        pipe.InitBuffer(maskBuf, availableUbSize * dimAlign * sizeof(uint8_t));
        pipe.InitBuffer(labelBroadBuf, availableUbSize * dimAlign * sizeof(int32_t));
        pipe.InitBuffer(scoreBroadBuf, availableUbSize * dimAlign * sizeof(float));
        pipe.InitBuffer(embeddingBuf, availableUbSize * dimAlign * sizeof(float));
        pipe.InitBuffer(vectorBuf, availableUbSize * dimAlign * sizeof(float));
        pipe.InitBuffer(sumBuf, availableUbSize * dimAlign * sizeof(float));
        pipe.InitBuffer(disBuf, availableUbSize * sizeof(float));
        pipe.InitBuffer(tempBuf, sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();
        if (coreId > usedCoreNum) {
            return;
        }
        if (coreId < pixelLast) {
            for (int32_t i = 0; i < loopTimeFront; ++i) {
                uint64_t offset = coreId * (averagePixels + 1) + i * availableUbSize;
                CopyIn(availableUbSize, offset);
                Compute(availableUbSize, offset);
            }
            if (lastLoopFront != 0) {
                uint64_t offset = coreId * (averagePixels + 1) + loopTimeFront * availableUbSize;
                CopyIn(lastLoopFront, offset);
                Compute(lastLoopFront, offset);
            }
        } else {
            for (int32_t i = 0; i < loopTimeRear; ++i) {
                uint64_t offset = coreId * averagePixels + pixelLast + i * availableUbSize;
                CopyIn(availableUbSize, offset);
                Compute(availableUbSize, offset);
            }
            if (lastLoopRear != 0) {
                uint64_t offset = coreId * averagePixels + pixelLast + loopTimeRear * availableUbSize;
                CopyIn(lastLoopRear, offset);
                Compute(lastLoopRear, offset);
            }
        }
    }
private:
    __aicore__ inline void CopyIn(uint32_t tensorSize, uint64_t offset)
    {
        // alloc tensor from queue memory
        LocalTensor<DTYPE_SCORE> scoreLocal = inQueueScore.AllocTensor<DTYPE_SCORE>();
        LocalTensor<DTYPE_KERNEL_CONTOUR> maskLocal = inQueueMask.AllocTensor<DTYPE_KERNEL_CONTOUR>();
        LocalTensor<DTYPE_EMBEDDING> embeddingLocal = inQueueEmbedding.AllocTensor<DTYPE_EMBEDDING>();
        LocalTensor<DTYPE_KERNEL_LABEL> labelLocal = inQueueLabel.AllocTensor<DTYPE_KERNEL_LABEL>();

        // copy progress_th tile from global tensor to local tensor
        DataCopyParams copyParamsMask{1, uint16_t(tensorSize * sizeof(uint8_t)), 0, 0};
        DataCopyParams copyParamsScore{1, uint16_t(tensorSize * sizeof(float)), 0, 0};
        DataCopyParams copyParamsLable{1, uint16_t(tensorSize * sizeof(int32_t)), 0, 0};
        DataCopyExtParams copyParamsInEmbedding{uint16_t(tensorSize), uint32_t(embeddingDim * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParamsEmbedding {true, 0, paddingNum, 0.f};

        DataCopyPad(scoreLocal, scoreGm[offset], copyParamsScore, padParams);
        DataCopyPad(maskLocal, maskGm[offset], copyParamsMask, padParams);
        DataCopyPad(labelLocal, labelGm[offset], copyParamsLable, padParams);
        DataCopyPad(embeddingLocal, embeddingGm[offset * embeddingDim], copyParamsInEmbedding, padParamsEmbedding);

        inQueueScore.EnQue(scoreLocal);
        inQueueMask.EnQue(maskLocal);
        inQueueLabel.EnQue(labelLocal);
        inQueueEmbedding.EnQue(embeddingLocal);
    }

    __aicore__ inline void Compute(uint32_t tensorSize, uint64_t offset)
    {
        LocalTensor<float> scoreLocal = inQueueScore.DeQue<float>();
        LocalTensor<uint8_t> maskLocal = inQueueMask.DeQue<uint8_t>();
        LocalTensor<int32_t> labelLocal = inQueueLabel.DeQue<int32_t>();
        LocalTensor<float> embeddingLocal = inQueueEmbedding.DeQue<float>();

        LocalTensor<float> kernelLabel = kernelLabelBuf.Get<float>();
        LocalTensor<float> kernelVector = kernelVectorBuf.Get<float>();
        LocalTensor<float> labelEmbeddings = embeddingBuf.Get<float>();
        LocalTensor<int32_t> labelBroadcast = labelBroadBuf.Get<int32_t>();
        LocalTensor<float> scoreBroadcast = scoreBroadBuf.Get<float>();
        LocalTensor<float> validScores = scoreBuf.Get<float>();
        LocalTensor<float> tempVector = vectorBuf.Get<float>();
        LocalTensor<float> validLabels = labelBuf.Get<float>();
        LocalTensor<float> distances = disBuf.Get<float>();
        LocalTensor<float> vectorSquared = sumBuf.Get<float>();
        LocalTensor<float> vectorNum = numBuf.Get<float>();
        LocalTensor<float> tempScore = tempBuf.Get<float>();

        // BroadCast labelLocal to the same shape as embeddingLocal
        uint32_t dstShape_[2] = {AlignUp(tensorSize, 8), dimAlign};
        uint32_t srcShape_[2] = {AlignUp(tensorSize, 8), 1};
        BroadCast<int32_t, 2, 1>(labelBroadcast, labelLocal, dstShape_, srcShape_);
        BroadCast<float, 2, 1>(scoreBroadcast, scoreLocal, dstShape_, srcShape_);

        DataCopyParams copyParamsVectorOut{1, uint16_t(embeddingDim * sizeof(float)), 0, 0};
        DataCopyParams copyParamsVectorIn{1, uint16_t(embeddingDim * sizeof(float)), 0, 0};
        DataCopyParams copyParamsLable{1, uint16_t(tensorSize * sizeof(int32_t)), 0, 0};
        for (int32_t label = 1; label < kernelRegionNum; label++) {
            Duplicate(vectorNum, 1.f, embeddingDim);
            SetAtomicAdd<float>();
            for (int32_t i = 0; i < tensorSize; ++i) {
                if (labelLocal.GetValue(i) == label) {
                    DataCopyPad(kernelVectorGm[label * embeddingDim], embeddingLocal[i * dimAlign], copyParamsVectorOut);
                    DataCopyPad(countGm[label * embeddingDim], vectorNum, copyParamsVectorOut);
                }
            }
            SetAtomicNone();
            SyncAll();

            DataCopyPad(kernelVector, kernelVectorGm[label * embeddingDim], copyParamsVectorIn, padParams);
            DataCopyPad(vectorNum, countGm[label * embeddingDim], copyParamsVectorIn, padParams);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

            Div(kernelVector, kernelVector, vectorNum, embeddingDim);
            uint32_t dstShape[2] = {tensorSize, dimAlign};
            uint32_t srcShape[2] = {1, dimAlign};
            BroadCast<float, 2, 0>(tempVector, kernelVector, dstShape, srcShape);

            LocalTensor<uint8_t> label_mask = maskBuf.Get<uint8_t>();
            CompareScalar(label_mask, labelBroadcast, 0, CMPMODE::EQ, AlignUp(tensorSize * dimAlign, 64));
            Select(labelEmbeddings, label_mask, embeddingLocal, 0.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, tensorSize * dimAlign);
            CompareScalar(label_mask, scoreBroadcast, 0.5f, CMPMODE::GT, AlignUp(tensorSize * dimAlign, 64));
            Select(labelEmbeddings, label_mask, labelEmbeddings, 0.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, tensorSize * dimAlign);
            Sub(tempVector, labelEmbeddings, tempVector, tensorSize * dimAlign);
            Power(vectorSquared, tempVector, 2.f, tensorSize * dimAlign);
            SumParams params{tensorSize, dimAlign, embeddingDim};
            Sum(distances, vectorSquared, params);

            LocalTensor<uint8_t> within_threshold = maskBuf.Get<uint8_t>();
            CompareScalar(within_threshold, distances, distanceThreshold * distanceThreshold, CMPMODE::LT, AlignUp(tensorSize, 64));
            Duplicate<float>(vectorNum, static_cast<float>(label), tensorSize);
            Cast(kernelLabel, labelLocal, RoundMode::CAST_ROUND, tensorSize);
            Select(kernelLabel, within_threshold, vectorNum, kernelLabel, SELMODE::VSEL_CMPMASK_SPR, tensorSize);
            Cast(labelLocal, kernelLabel, RoundMode::CAST_FLOOR, tensorSize);

            DataCopyPad(labelUpdatedGm[offset], labelLocal, copyParamsLable);
        }
        LocalTensor<uint8_t> fmask = maskBuf.Get<uint8_t>();
        CompareScalar(fmask, kernelLabel, 0.f, CMPMODE::GT, AlignUp(tensorSize, 64));
        Select(validLabels, fmask, kernelLabel, 0.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, tensorSize);
        Select(validScores, fmask, scoreLocal, 0.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, tensorSize);
        vectorNum.SetValue(0, 1.f);

        SetAtomicAdd<float>();
        for (int32_t i = 0; i < tensorSize; ++i) {
            addr = static_cast<int32_t>(validLabels.GetValue(i));
            if (addr > 0) {
                tempScore.SetValue(0, validScores.GetValue(i));
                DataCopyPad(pointVectorGm[addr * 2], tempScore, copyParams);
                DataCopyPad(pointVectorGm[addr * 2 + 1], vectorNum, copyParams);
            }
        }
        SetAtomicNone();

        inQueueScore.FreeTensor(scoreLocal);
        inQueueMask.FreeTensor(maskLocal);
        inQueueEmbedding.FreeTensor(embeddingLocal);
        inQueueLabel.FreeTensor(labelLocal);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueScore, inQueueMask, inQueueEmbedding, inQueueLabel;
    TBuf<TPosition::VECCALC> kernelVectorBuf, disBuf, embeddingBuf, vectorBuf, labelBuf, numBuf, scoreBuf;
    TBuf<TPosition::VECCALC> kernelLabelBuf, sumBuf, maskBuf, labelBroadBuf, scoreBroadBuf, tempBuf;
    GlobalTensor<float> scoreGm, embeddingGm, pointVectorGm, kernelVectorGm, countGm;
    GlobalTensor<uint8_t> maskGm, contourGm;
    GlobalTensor<int32_t> labelGm, labelUpdatedGm;

    uint32_t usedCoreNum;
    uint32_t averagePixels;
    uint32_t totalPixels;
    uint32_t pixelLast;
    uint32_t embeddingDim;
    uint32_t dimAlign;
    uint32_t availableUbSize;
    uint32_t loopTimeFront;
    uint32_t lastLoopFront;
    uint32_t loopTimeRear;
    uint32_t lastLoopRear;
    uint32_t addr;
    uint8_t paddingNum = 0;
    int32_t kernelRegionNum;
    float distanceThreshold;

    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams copyParams{1, uint16_t(1 * sizeof(float)), 0, 0};

    event_t eventIdMte2ToV;
};

extern "C" __global__ __aicore__ void pixel_group(GM_ADDR score, GM_ADDR mask, GM_ADDR embedding, GM_ADDR kernel_label,
                                                  GM_ADDR kernel_contour, GM_ADDR point_vector, GM_ADDR label_updated,
                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelPixelGroup op;
    op.Init(score, mask, embedding, kernel_label, kernel_contour, point_vector, label_updated, workspace, &tiling_data);
    op.Process();
}