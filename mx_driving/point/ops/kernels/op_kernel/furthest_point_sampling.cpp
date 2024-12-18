/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */
#include "furthest_point_sampling.h"

using namespace AscendC;

// Entrance of kernel
extern "C" __global__ __aicore__ void furthest_point_sampling(
    GM_ADDR point_xyz,
    GM_ADDR temp,
    GM_ADDR index,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    tilingArgs TA;
    // Since type of tiling_data unknown, create a class out of reliability.
    TA.N              = tiling_data.N;
    TA.batch          = tiling_data.batch;
    TA.numPoints      = tiling_data.numPoints;
    TA.pieces         = tiling_data.pieces;
    TA.formerNum      = tiling_data.formerNum;
    TA.tailNum        = tiling_data.tailNum;
    TA.workSize       = tiling_data.workSize;
    TA.idxTempSize    = tiling_data.idxTempSize;
    TA.bigCoreBatch   = tiling_data.bigCoreBatch;
    TA.smallCoreBatch = tiling_data.smallCoreBatch;
    TA.bigCoreNum     = tiling_data.bigCoreNum;
    TA.repeats        = tiling_data.repeats;

    if (TILING_KEY_IS(0)) {
        furthestPointSamplingKernel<float, int32_t> op(point_xyz, temp, index, workspace, &TA);
        op.Process();
    }
}

template<typename dataType, typename idxType>
__aicore__ inline furthestPointSamplingKernel<dataType, idxType>::furthestPointSamplingKernel(GM_ADDR point_xyz,
    GM_ADDR temp, GM_ADDR index, GM_ADDR workspace, tilingArgs *tiling)
{
    // Init tiling args.
    this->TA = tiling;
    // host tiling have ensured formerNum is aligned with 32bytes and bigger than tailNum.
    this->sizeofFormer = this->TA->formerNum * sizeof(dataType);
    this->sizeofTail = this->TA->tailNum * sizeof(dataType);
    this->dataNumIn32Bytes = 32 / sizeof(dataType);
    this->dataNumIn64Bytes = 64 / sizeof(dataType);
    this->dataNumIn256Bytes = 256 / sizeof(dataType);
    this->dataNumIn1024Bytes = 1024 / sizeof(dataType);
    // Init GM.
    InitGm(point_xyz, temp, index, workspace);

    // Must be aligned with 32bytes.
    this->pipe.InitBuffer(this->pointXQue, BUFFER_NUM, this->sizeofFormer);
    this->pipe.InitBuffer(this->pointYQue, BUFFER_NUM, this->sizeofFormer);
    this->pipe.InitBuffer(this->pointZQue, BUFFER_NUM, this->sizeofFormer);
    this->pipe.InitBuffer(this->pointTempXUb, BUFFER_NUM, this->sizeofFormer);
    this->pipe.InitBuffer(this->pointTempYUb, BUFFER_NUM, this->sizeofFormer);
    this->pipe.InitBuffer(this->pointTempZUb, BUFFER_NUM, this->sizeofFormer);
    this->pipe.InitBuffer(this->nearestDistQue, BUFFER_NUM, this->sizeofFormer);
    this->pipe.InitBuffer(this->distUb, BUFFER_NUM, this->sizeofFormer);
    this->pipe.InitBuffer(this->workUb, BUFFER_NUM, this->TA->workSize);

    this->pipe.InitBuffer(this->idxQue, BUFFER_NUM, 1024); // 1024: copy out 256 fp32s once

    this->pipe.InitBuffer(this->idxTempUb, BUFFER_NUM, this->TA->idxTempSize);
    this->pipe.InitBuffer(this->pointSampled, BUFFER_NUM, 32 * 3);
    // Malloc.
    this->ubBlocks.pointXLocal = pointXQue.AllocTensor<dataType>();
    this->ubBlocks.pointYLocal = pointYQue.AllocTensor<dataType>();
    this->ubBlocks.pointZLocal = pointZQue.AllocTensor<dataType>();
    this->ubBlocks.pointTempXLocal = pointTempXUb.AllocTensor<dataType>();
    this->ubBlocks.pointTempYLocal = pointTempYUb.AllocTensor<dataType>();
    this->ubBlocks.pointTempZLocal = pointTempZUb.AllocTensor<dataType>();
    this->ubBlocks.nearestDistLocal = nearestDistQue.AllocTensor<dataType>();
    this->ubBlocks.distLocal = distUb.AllocTensor<dataType>();
    this->ubBlocks.workLocal = workUb.AllocTensor<dataType>();

    this->ubBlocks.idxLocal = idxQue.AllocTensor<idxType>();

    this->ubBlocks.idxTempLocal = idxTempUb.AllocTensor<dataType>();
    this->ubBlocks.pointSampledLocal = pointSampled.AllocTensor<dataType>();
}

template<typename dataType, typename idxType>
__aicore__ inline void furthestPointSamplingKernel<dataType, idxType>::Process()
{
    uint32_t batch_num = (GetBlockIdx() < this->TA->bigCoreNum) ? (this->TA->bigCoreBatch) : (this->TA->smallCoreBatch);

    for (this->core_batch = 0; this->core_batch < batch_num; this->core_batch++) {
        this->batchOffsetPoint = this->core_batch * this->TA->N * 3;
        this->batchOffsetNearest = this->core_batch * this->TA->N;
        // Set：idxGm[0] = 0
        CopyInIdx(0);
        if (this->TA->numPoints == 1) {
            CopyOut(0); // special case: only one points sampled.
        }
        if (this->TA->pieces == 1) {
            Process_complete_data();
        } else {
            Process_split_data();
        }
    }
}

template<typename dataType, typename idxType>
__aicore__ inline void furthestPointSamplingKernel<dataType, idxType>::CopyInIdx(uint32_t loopNum)
{
    DataCopyParams data_copy_param = {1, 1, 0, 0};
    uint32_t offsetGmX    = this->batchOffsetPoint + this->maxDistIdx;
    uint32_t offsetGmY    = offsetGmX + this->TA->N;
    uint32_t offsetGmZ    = offsetGmY + this->TA->N;
    uint32_t offsetLocalX = 0;
    uint32_t offsetLocalY = this->dataNumIn32Bytes;
    uint32_t offsetLocalZ = this->dataNumIn64Bytes;
    uint32_t offsetIdx    = loopNum & (this->dataNumIn1024Bytes - 1); // aka. loopNum % this->dataNumIn1024Bytes

    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);

#ifndef __GET_CODE_CHANNEL__
    DataCopy(this->ubBlocks.pointSampledLocal[offsetLocalX], pointGm[offsetGmX], data_copy_param);
    DataCopy(this->ubBlocks.pointSampledLocal[offsetLocalY], pointGm[offsetGmY], data_copy_param);
    DataCopy(this->ubBlocks.pointSampledLocal[offsetLocalZ], pointGm[offsetGmZ], data_copy_param);
#endif

    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    this->ubBlocks.idxLocal.SetValue(offsetIdx, this->maxDistIdx);
    this->pointXSampled = -1 * this->ubBlocks.pointSampledLocal.GetValue(offsetLocalX);
    this->pointYSampled = -1 * this->ubBlocks.pointSampledLocal.GetValue(offsetLocalY);
    this->pointZSampled = -1 * this->ubBlocks.pointSampledLocal.GetValue(offsetLocalZ);
    this->maxDist = 0;
    this->maxDistIdx = 0;
}

template<typename dataType, typename idxType>
__aicore__ inline void furthestPointSamplingKernel<dataType, idxType>::Process_complete_data()
{
    uint32_t loopNum;

    for (loopNum = 1; loopNum < this->TA->numPoints; loopNum++) {
        if (loopNum == 1) {
            Process_first_sampling(0);
        } else {
            ComputePointsSquare();

            pipe_barrier(PIPE_V);

            ComputeDist();

            pipe_barrier(PIPE_V);

            ComputeSamplePoints(0, 0);
        }
        pipe_barrier(PIPE_V);

        updateDist();

        CopyInIdx(loopNum);

        CopyOut(loopNum);
    }
}

template<typename dataType, typename idxType>
__aicore__ inline void furthestPointSamplingKernel<dataType, idxType>::Process_split_data()
{
    uint32_t loopNum, loopSplit;

    for (loopNum = 1; loopNum < this->TA->numPoints; loopNum++) {
        for (loopSplit = 0; loopSplit < this->TA->pieces; loopSplit++) {
            if (loopNum == 1) {
                Process_first_sampling(loopSplit);
            } else {
                uint32_t comBlock = (loopSplit + this->TA->pieces - 1) % this->TA->pieces;

                // Cal point_x -> Mov point_x, Cal point_y -> Mov point_y, Cal point_z -> Mov point_z
                ComputePointDeltaSquare(this->ubBlocks.pointXLocal, this->ubBlocks.pointTempXLocal, this->pointXSampled);

                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

                CopyInPointAxis(pointAxis_x, loopSplit);

                ComputePointDeltaSquare(this->ubBlocks.pointYLocal, this->ubBlocks.pointTempYLocal, this->pointYSampled);

                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

                CopyInPointAxis(pointAxis_y, loopSplit);

                ComputePointDeltaSquare(this->ubBlocks.pointZLocal, this->ubBlocks.pointTempZLocal, this->pointZSampled);

                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);

                CopyInPointAxis(pointAxis_z, loopSplit);

                pipe_barrier(PIPE_ALL);

                ComputeDist();

                pipe_barrier(PIPE_ALL);

                ComputeSamplePoints(loopSplit, comBlock);

                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);

                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

                CopyInNearestDistTemp(loopSplit);
            }
        }
        pipe_barrier(PIPE_V);

        updateDist();

        CopyInIdx(loopNum);

        CopyOut(loopNum);
    }
}

template<typename dataType, typename idxType>
__aicore__ inline void furthestPointSamplingKernel<dataType, idxType>::Process_first_sampling(uint32_t loopSplit)
{
    // Mov point_x -> Cal point_x, Mov point_y -> Cal point_y, Mov point_z -> Cal point_z
    CopyInPointAxis(pointAxis_x, loopSplit);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    ComputePointDeltaSquare(this->ubBlocks.pointXLocal, this->ubBlocks.pointTempXLocal, this->pointXSampled);

    CopyInPointAxis(pointAxis_y, loopSplit);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

    ComputePointDeltaSquare(this->ubBlocks.pointYLocal, this->ubBlocks.pointTempYLocal, this->pointYSampled);

    CopyInPointAxis(pointAxis_z, loopSplit);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);

    ComputePointDeltaSquare(this->ubBlocks.pointZLocal, this->ubBlocks.pointTempZLocal, this->pointZSampled);

    pipe_barrier(PIPE_V);

    ComputeDist();

    CopyInNearestDist(loopSplit);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID3);

    ComputeSamplePoints(loopSplit, loopSplit);
}

template<typename dataType, typename idxType>
__aicore__ inline void furthestPointSamplingKernel<dataType, idxType>::CopyInPointAxis(PointAxis pointAxis,
    uint32_t loopSplit)
{
    uint64_t offset;
    DataCopyParams data_copy_param = {1, 0, 0, 0};
    DataCopyPadParams pad_param = {false, 0, 0, 0};

    if (loopSplit == (this->TA->pieces - 1)) {
        data_copy_param.blockLen = this->sizeofTail;
    } else {
        data_copy_param.blockLen = this->sizeofFormer;
    }
    switch (pointAxis) {
        case pointAxis_x:
            offset = this->batchOffsetPoint + this->TA->formerNum * loopSplit;
            break;
        case pointAxis_y:
            offset = this->batchOffsetPoint + this->TA->formerNum * loopSplit + this->TA->N;
            break;
        case pointAxis_z:
            offset = this->batchOffsetPoint + this->TA->formerNum * loopSplit + this->TA->N * 2;
            break;
        default:
            break;
    }

    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID1);

    switch (pointAxis) {
        case pointAxis_x:
#ifndef __GET_CODE_CHANNEL__
            DataCopyPad(this->ubBlocks.pointXLocal, pointGm[offset], data_copy_param, pad_param);
#endif
            break;
        case pointAxis_y:
#ifndef __GET_CODE_CHANNEL__
            DataCopyPad(this->ubBlocks.pointYLocal, pointGm[offset], data_copy_param, pad_param);
#endif
            break;
        case pointAxis_z:
#ifndef __GET_CODE_CHANNEL__
            DataCopyPad(this->ubBlocks.pointZLocal, pointGm[offset], data_copy_param, pad_param);
#endif
            break;
        default:
            break;
    }
}

template<typename dataType, typename idxType>
__aicore__ inline void furthestPointSamplingKernel<dataType, idxType>::CopyInNearestDist(uint32_t loopSplit)
{
    uint64_t offset = this->batchOffsetNearest + this->TA->formerNum * loopSplit;
    DataCopyParams data_copy_param = {1, 0, 0, 0};
    DataCopyPadParams pad_param = {false, 0, 0, 0};

    if (loopSplit == (this->TA->pieces - 1)) {
        data_copy_param.blockLen = this->sizeofTail;
    } else {
        data_copy_param.blockLen = this->sizeofFormer;
    }

    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID2);

#ifndef __GET_CODE_CHANNEL__
    DataCopyPad(this->ubBlocks.nearestDistLocal, nearestDistGm[offset], data_copy_param, pad_param);
#endif
}

template<typename dataType, typename idxType>
__aicore__ inline void furthestPointSamplingKernel<dataType, idxType>::CopyInNearestDistTemp(uint32_t loopSplit)
{
    uint64_t offset = this->batchOffsetNearest + this->TA->formerNum * loopSplit;
    DataCopyParams data_copy_param = {1, 0, 0, 0};
    DataCopyPadParams pad_param = {false, 0, 0, 0};

    if (loopSplit == (this->TA->pieces - 1)) {
        data_copy_param.blockLen = this->sizeofTail;
    } else {
        data_copy_param.blockLen = this->sizeofFormer;
    }

    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID2);

#ifndef __GET_CODE_CHANNEL__
    DataCopyPad(this->ubBlocks.nearestDistLocal, nearestDistTempGm[offset], data_copy_param, pad_param);
#endif
}

template<typename dataType, typename idxType>
__aicore__ inline void furthestPointSamplingKernel<dataType, idxType>::ComputePointsSquare()
{
    uint32_t total_num, dupTime, offset, comp_num;

    // while cal，every data block is aligned with 256 bytes.
    for (offset = 0, total_num = this->TA->formerNum; total_num > 0;
        comp_num = dupTime * this->dataNumIn256Bytes, offset = offset + comp_num, total_num = total_num - comp_num) {
        dupTime = (total_num * sizeof(dataType)) / ALLIGNED_BYTES;
        dupTime = (dupTime > OP_MAX_REPEAT_NUM) ? OP_MAX_REPEAT_NUM : dupTime;

        set_flag(PIPE_S, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID3);

        Adds<dataType>(this->ubBlocks.pointTempXLocal[offset], this->ubBlocks.pointXLocal[offset], this->pointXSampled,
            this->dataNumIn256Bytes, dupTime, {1, 1, 8, 8});
        Adds<dataType>(this->ubBlocks.pointTempYLocal[offset], this->ubBlocks.pointYLocal[offset], this->pointYSampled,
            this->dataNumIn256Bytes, dupTime, {1, 1, 8, 8});
        Adds<dataType>(this->ubBlocks.pointTempZLocal[offset], this->ubBlocks.pointZLocal[offset], this->pointZSampled,
            this->dataNumIn256Bytes, dupTime, {1, 1, 8, 8});

        pipe_barrier(PIPE_V);

        Mul<dataType>(this->ubBlocks.pointTempXLocal[offset], this->ubBlocks.pointTempXLocal[offset],
            this->ubBlocks.pointTempXLocal[offset], this->dataNumIn256Bytes, dupTime, {1, 1, 1, 8, 8, 8});
        Mul<dataType>(this->ubBlocks.pointTempYLocal[offset], this->ubBlocks.pointTempYLocal[offset],
            this->ubBlocks.pointTempYLocal[offset], this->dataNumIn256Bytes, dupTime, {1, 1, 1, 8, 8, 8});
        Mul<dataType>(this->ubBlocks.pointTempZLocal[offset], this->ubBlocks.pointTempZLocal[offset],
            this->ubBlocks.pointTempZLocal[offset], this->dataNumIn256Bytes, dupTime, {1, 1, 1, 8, 8, 8});
    }
}

template<typename dataType, typename idxType>
__aicore__ inline void furthestPointSamplingKernel<dataType, idxType>::ComputePointDeltaSquare(
        LocalTensor<dataType> &pointLocal, LocalTensor<dataType> &pointTempLocal, dataType pointSampled)
{
    uint32_t total_num, dupTime, offset, comp_num;

    // while cal，every data block is aligned with 256 bytes.
    for (offset = 0, total_num = this->TA->formerNum; total_num > 0;
        comp_num = dupTime * this->dataNumIn256Bytes, offset = offset + comp_num, total_num = total_num - comp_num) {
        dupTime = (total_num * sizeof(dataType)) / ALLIGNED_BYTES;
        dupTime = (dupTime > OP_MAX_REPEAT_NUM) ? OP_MAX_REPEAT_NUM : dupTime;

        set_flag(PIPE_S, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID3);

        Adds<dataType>(pointTempLocal[offset], pointLocal[offset], pointSampled, this->dataNumIn256Bytes,
            dupTime, {1, 1, 8, 8});

        pipe_barrier(PIPE_V);

        Mul<dataType>(pointTempLocal[offset], pointTempLocal[offset], pointTempLocal[offset], this->dataNumIn256Bytes,
            dupTime, {1, 1, 1, 8, 8, 8});
    }
}

template<typename dataType, typename idxType>
__aicore__ inline void furthestPointSamplingKernel<dataType, idxType>::ComputeDist()
{
    uint32_t total_num, dupTime, offset, comp_num;

    // while cal，every data block is aligned with 256 bytes.
    for (offset = 0, total_num = this->TA->formerNum; total_num > 0;
        comp_num = dupTime * this->dataNumIn256Bytes, offset = offset + comp_num, total_num = total_num - comp_num) {
        dupTime = (total_num * sizeof(dataType)) / ALLIGNED_BYTES;
        dupTime = (dupTime > OP_MAX_REPEAT_NUM) ? OP_MAX_REPEAT_NUM : dupTime;

        set_flag(PIPE_S, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

        Add<dataType>(this->ubBlocks.distLocal[offset], this->ubBlocks.pointTempXLocal[offset],
            this->ubBlocks.pointTempYLocal[offset], this->dataNumIn256Bytes, dupTime, {1, 1, 1, 8, 8, 8});

        pipe_barrier(PIPE_V);

        Add<dataType>(this->ubBlocks.distLocal[offset], this->ubBlocks.distLocal[offset],
            this->ubBlocks.pointTempZLocal[offset], this->dataNumIn256Bytes, dupTime, {1, 1, 1, 8, 8, 8});
    }
}

template<typename dataType, typename idxType>
__aicore__ inline void furthestPointSamplingKernel<dataType, idxType>::ComputeSamplePoints(uint32_t loopSplit,
    uint32_t comBlock)
{
    uint32_t total_num, dupTime, offset, comp_num, reduceCnt, reduceOffset;

    reduceCnt = ((this->TA->formerNum != this->TA->tailNum) && (comBlock == (this->TA->pieces - 1))) ?
        this->TA->tailNum : this->TA->formerNum;
    reduceOffset = comBlock * 2;

    for (offset = 0, total_num = this->TA->formerNum; total_num > 0;
        comp_num = dupTime * this->dataNumIn256Bytes, offset = offset + comp_num, total_num = total_num - comp_num) {
        dupTime = (total_num * sizeof(dataType)) / ALLIGNED_BYTES;
        dupTime = (dupTime > OP_MAX_REPEAT_NUM) ? OP_MAX_REPEAT_NUM : dupTime;

        set_flag(PIPE_S, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_S, PIPE_V, EVENT_ID1);

        Min<dataType>(this->ubBlocks.nearestDistLocal[offset], this->ubBlocks.nearestDistLocal[offset],
            this->ubBlocks.distLocal[offset], this->dataNumIn256Bytes, dupTime, {1, 1, 1, 8, 8, 8});
    }

    if (this->TA->pieces > 1) {
        // set_flag: After Updated nearestDistLocal, Mov nearestDistLocal to GM.
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        CopyOutNearestDistTemp(comBlock);
    }

    pipe_barrier(PIPE_ALL);

    // ReduceMax
    ReduceMax<dataType>(this->ubBlocks.idxTempLocal[reduceOffset], this->ubBlocks.nearestDistLocal,
        this->ubBlocks.workLocal, reduceCnt, 1);
}

template<typename dataType, typename idxType>
__aicore__ inline void furthestPointSamplingKernel<dataType, idxType>::updateDist()
{
    dataType tempValue;

    // this->TA->pieces >= 1
    for (uint32_t i = 1; i < (2 * this->TA->pieces); i = (i + 2)) {
        tempValue = this->ubBlocks.idxTempLocal.GetValue(i);
        if (this->maxDist < this->ubBlocks.idxTempLocal.GetValue(i-1)) {
            this->maxDist = this->ubBlocks.idxTempLocal.GetValue(i-1);
            this->maxDistIdx = (this->TA->formerNum * (i / 2)) + (*reinterpret_cast<idxType*>(&tempValue));
        }
    }
}

template<typename dataType, typename idxType>
__aicore__ inline void furthestPointSamplingKernel<dataType, idxType>::CopyOut(uint32_t loopNum)
{
    uint32_t elemNum = this->dataNumIn1024Bytes;
    // elemNum is a multiple of 2.
    if ((loopNum != 0) && (((loopNum + 1) & (elemNum - 1)) != 0) && ((loopNum + 1) != this->TA->numPoints)) {
        // when num of sampled < 256 && not last loop, return;
        return ;
    }

    uint64_t offset = this->core_batch * this->TA->numPoints;
    DataCopyExtParams data_copy_param = {1, sizeof(dataType), 0, 0, 0};
    if (((loopNum + 1) & (elemNum - 1)) == 0) {
        data_copy_param.blockLen = 1024;
        offset = offset + loopNum / elemNum * elemNum;
    } else if ((loopNum + 1) == this->TA->numPoints) {
        data_copy_param.blockLen = sizeof(dataType) *
            (this->TA->numPoints - (this->TA->numPoints / elemNum * elemNum));
        offset = offset + (this->TA->numPoints / elemNum * elemNum);
    }

    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);

#ifndef __GET_CODE_CHANNEL__
    DataCopyPad(idxGm[offset], this->ubBlocks.idxLocal, data_copy_param);
#endif
}

template<typename dataType, typename idxType>
__aicore__ inline void furthestPointSamplingKernel<dataType, idxType>::CopyOutNearestDistTemp(uint32_t loopSplit)
{
    uint64_t offset = this->batchOffsetNearest + this->TA->formerNum * loopSplit;
    DataCopyExtParams data_copy_param = {1, 0, 0, 0, 0};

    if (loopSplit == (this->TA->pieces - 1)) {
        data_copy_param.blockLen = this->sizeofTail;
    } else {
        data_copy_param.blockLen = this->sizeofFormer;
    }

    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);

#ifndef __GET_CODE_CHANNEL__
    DataCopyPad(nearestDistTempGm[offset], this->ubBlocks.nearestDistLocal, data_copy_param);
#endif
}

template<typename dataType, typename idxType>
__aicore__ inline void furthestPointSamplingKernel<dataType, idxType>::InitGm(GM_ADDR point_xyz, GM_ADDR temp,
    GM_ADDR index, GM_ADDR workspace)
{
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    uint32_t coreId = GetBlockIdx();
    uint64_t skipData, numData, skipIdx, numIdx;
    uint64_t numDataBigCore = this->TA->bigCoreBatch * this->TA->N;
    uint64_t numIdxBigCore = this->TA->bigCoreBatch * this->TA->numPoints;

    if (coreId < this->TA->bigCoreNum) {
        numData = numDataBigCore;
        numIdx = numIdxBigCore;
        skipData = numData * coreId;
        skipIdx = numIdx * coreId;
    } else {
        numData = this->TA->smallCoreBatch * this->TA->N;
        numIdx = this->TA->smallCoreBatch * this->TA->numPoints;
        skipData = this->TA->bigCoreNum * numDataBigCore + (coreId - this->TA->bigCoreNum) * numData;
        skipIdx = this->TA->bigCoreNum * numIdxBigCore + (coreId - this->TA->bigCoreNum) * numIdx;
    }

    this->pointGm.SetGlobalBuffer((__gm__ dataType*)point_xyz + skipData * 3, numData * 3);
    this->nearestDistGm.SetGlobalBuffer((__gm__ dataType*)temp + skipData, numData);
    this->idxGm.SetGlobalBuffer((__gm__ idxType*)index + skipIdx, numIdx);
    this->nearestDistTempGm.SetGlobalBuffer((__gm__ dataType*)usrWorkspace + skipData, numData);
}

template<typename dataType, typename idxType>
__aicore__ inline furthestPointSamplingKernel<dataType, idxType>::~furthestPointSamplingKernel()
{
    this->pointXQue.FreeTensor(this->ubBlocks.pointXLocal);
    this->pointYQue.FreeTensor(this->ubBlocks.pointYLocal);
    this->pointZQue.FreeTensor(this->ubBlocks.pointZLocal);
    this->pointTempXUb.FreeTensor(this->ubBlocks.pointTempXLocal);
    this->pointTempYUb.FreeTensor(this->ubBlocks.pointTempYLocal);
    this->pointTempZUb.FreeTensor(this->ubBlocks.pointTempZLocal);
    this->nearestDistQue.FreeTensor(this->ubBlocks.nearestDistLocal);
    this->distUb.FreeTensor(this->ubBlocks.distLocal);
    this->workUb.FreeTensor(this->ubBlocks.workLocal);

    this->idxQue.FreeTensor(this->ubBlocks.idxLocal);

    this->idxTempUb.FreeTensor(this->ubBlocks.idxTempLocal);
    this->pointSampled.FreeTensor(this->ubBlocks.pointSampledLocal);
}