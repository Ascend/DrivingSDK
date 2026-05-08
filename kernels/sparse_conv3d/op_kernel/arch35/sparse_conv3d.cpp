/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_tpipe_impl.h"
#include "kernel_utils.h"
using namespace AscendC;
using namespace MicroAPI;

namespace {
constexpr static int32_t BUFFER_NUM = 1;
constexpr int32_t THREAD_NUM = 1024;
constexpr int32_t OUTPUT_DIM = 4;
}; // namespace

template<typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void computeSIMT(__ubuf__ T* indicesLocal_,
    __ubuf__ T* indicesOffsetLocal_, __ubuf__ T* indicesPairLocal_, int32_t taskOffset, int32_t singleLoopTask,
    int32_t kernelSize, int32_t kernelD, int32_t kernelH, int32_t kernelW, int32_t paddingDepth, int32_t paddingHeight,
    int32_t paddingWidth, int32_t outputDepth, int32_t outputHeight, int32_t outputWidth, int32_t strideDepth,
    int32_t strideHeight, int32_t strideWidth)
{
    for (int32_t i = AscendC::Simt::GetThreadIdx(); i < singleLoopTask; i += AscendC::Simt::GetThreadNum()) {
        int32_t idxOffset = i * 4;
        int32_t featureB = indicesLocal_[idxOffset];
        int32_t featureD = indicesLocal_[idxOffset + 1] + paddingDepth;
        int32_t featureH = indicesLocal_[idxOffset + 2] + paddingHeight;
        int32_t featureW = indicesLocal_[idxOffset + 3] + paddingWidth;
        int32_t bOffset = featureB * outputDepth * outputHeight * outputWidth;
        // Calculate the features of this position that affect the positions of the output
        int32_t startD = Simt::Max(featureD - kernelD + 1, 0);
        int32_t startH = Simt::Max(featureH - kernelH + 1, 0);
        int32_t startW = Simt::Max(featureW - kernelW + 1, 0);
        int32_t outBeginD = (startD + strideDepth - 1) / strideDepth;
        int32_t outBeginH = (startH + strideHeight - 1) / strideHeight;
        int32_t outBeginW = (startW + strideWidth - 1) / strideWidth;
        int32_t outEndD = Simt::Min(featureD / strideDepth + 1, outputDepth);
        int32_t outEndH = Simt::Min(featureH / strideHeight + 1, outputHeight);
        int32_t outEndW = Simt::Min(featureW / strideWidth + 1, outputWidth);

        for (int32_t ix = outBeginD; ix < outEndD; ix++) {
            uint32_t xOffset = (uint32_t)ix * outputHeight * outputWidth;
            for (int32_t iy = outBeginH; iy < outEndH; iy++) {
                uint32_t yOffset = (uint32_t)iy * outputWidth;
                for (int32_t iz = outBeginW; iz < outEndW; iz++) {
                    uint32_t zOffset = (uint32_t)iz;
                    uint32_t gmOutValueOffset = bOffset + xOffset + yOffset + zOffset;
                    uint32_t weightD = featureD - ix * strideDepth;
                    uint32_t weightH = featureH - iy * strideHeight;
                    uint32_t weightW = featureW - iz * strideWidth;
                    uint32_t convOffset = weightD * kernelH * kernelW + weightH * kernelW + weightW;
                    int64_t outInidcesOffset = i * kernelSize + convOffset;
                    int64_t outInidcesPairOffset = (i * kernelSize + convOffset) * 4;
                    indicesOffsetLocal_[outInidcesOffset] = gmOutValueOffset;
                    indicesPairLocal_[outInidcesPairOffset] = featureB;
                    indicesPairLocal_[outInidcesPairOffset + 1] = ix;
                    indicesPairLocal_[outInidcesPairOffset + 2] = iy;
                    indicesPairLocal_[outInidcesPairOffset + 3] = iz;
                }
            }
        }
    }
}


class KernelSparseConv3d {
public:
    __aicore__ inline KernelSparseConv3d() {}
    __aicore__ inline void Init(GM_ADDR indices, GM_ADDR indices_out, GM_ADDR indices_pair, GM_ADDR workspace,
        SparseConv3dTilingData* tiling_data, TPipe* tmpPipe)
    {
        pipe = tmpPipe;
        curBlockIdx = GetBlockIdx();
        // features dtype must be same with weight
        initTilingData(tiling_data);

        uint64_t beginOffset = curBlockIdx * coreTask;

        if (curBlockIdx < usedCoreNum - 1) {
            taskNum = coreTask;
            coreRepeatTimes = repeatTimes;
            coreMoveTail = moveTail;
        } else {
            taskNum = lastCoreTask;
            coreRepeatTimes = lastRepeatTimes;
            coreMoveTail = lastMoveTail;
        }

        indicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_INDICES*>(indices) + beginOffset * 4);
        outputIndicesGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_INDICES*>(indices_out) + beginOffset * kernelSize);
        outputIndicesPairGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ DTYPE_INDICES*>(indices_pair) + beginOffset * kernelSize * 4);

        pipe->InitBuffer(indicesUB_, moveLen * OUTPUT_DIM * sizeof(DTYPE_INDICES));
        pipe->InitBuffer(outIndicesUB_, moveLen * kernelSize * sizeof(DTYPE_INDICES));
        pipe->InitBuffer(outIndicesPairUB_, moveLen * kernelSize * OUTPUT_DIM * sizeof(DTYPE_INDICES));

        indicesLocal_ = indicesUB_.Get<DTYPE_INDICES>();
        indicesOffsetLocal_ = outIndicesUB_.Get<DTYPE_INDICES>();
        indicesPairLocal_ = outIndicesPairUB_.Get<DTYPE_INDICES>();
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < coreRepeatTimes; i++) {
            Compute(i);
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    __aicore__ inline void initTilingData(SparseConv3dTilingData* tiling_data)
    {
        usedCoreNum = tiling_data->usedCoreNum;
        coreTask = tiling_data->coreTask;
        lastCoreTask = tiling_data->lastCoreTask;

        moveLen = tiling_data->moveLen;

        repeatTimes = tiling_data->repeatTimes;
        moveTail = tiling_data->moveTail;
        lastRepeatTimes = tiling_data->lastRepeatTimes;
        lastMoveTail = tiling_data->lastMoveTail;

        kernelD = tiling_data->kernelD;
        kernelH = tiling_data->kernelH;
        kernelW = tiling_data->kernelW;
        kernelSize = tiling_data->kernelSize;

        outputDepth = tiling_data->outputDepth;
        outputHeight = tiling_data->outputHeight;
        outputWidth = tiling_data->outputWidth;

        strideDepth = tiling_data->strideDepth;
        strideHeight = tiling_data->strideHeight;
        strideWidth = tiling_data->strideWidth;

        paddingDepth = tiling_data->paddingDepth;
        paddingHeight = tiling_data->paddingHeight;
        paddingWidth = tiling_data->paddingWidth;
    }

    __aicore__ inline void Compute(uint32_t query)
    {
        uint32_t taskOffset = query * moveLen;
        uint32_t singleLoopTask = moveLen;
        if (query == coreRepeatTimes - 1) {
            singleLoopTask = coreMoveTail;
        }

        Duplicate<DTYPE_INDICES>(indicesOffsetLocal_, -1, moveLen * kernelSize);
        Duplicate<DTYPE_INDICES>(indicesPairLocal_, 0, moveLen * kernelSize * OUTPUT_DIM);
        DataCopyPad(indicesLocal_, indicesGm_[taskOffset * OUTPUT_DIM],
            {1, (uint32_t)(singleLoopTask * OUTPUT_DIM * sizeof(DTYPE_INDICES)), 0, 0, 0}, {true, 0, 0, 0});

        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        AscendC::Simt::VF_CALL<computeSIMT<DTYPE_INDICES>>(AscendC::Simt::Dim3 {
                THREAD_NUM
            },
            (__ubuf__ DTYPE_INDICES*)indicesLocal_.GetPhyAddr(),
            (__ubuf__ DTYPE_INDICES*)indicesOffsetLocal_.GetPhyAddr(),
            (__ubuf__ DTYPE_INDICES*)indicesPairLocal_.GetPhyAddr(), taskOffset, singleLoopTask, kernelSize, kernelD,
            kernelH, kernelW, paddingDepth, paddingHeight, paddingWidth, outputDepth, outputHeight, outputWidth,
            strideDepth, strideHeight, strideWidth);

        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        DataCopyPad(outputIndicesGm_[taskOffset * kernelSize], indicesOffsetLocal_,
            {1, (uint32_t)(singleLoopTask * kernelSize * sizeof(DTYPE_INDICES)), 0, 0, 0});
        DataCopyPad(outputIndicesPairGm_[taskOffset * kernelSize * 4], indicesPairLocal_,
            {1, (uint32_t)(singleLoopTask * kernelSize * 4 * sizeof(DTYPE_INDICES)), 0, 0, 0});
    }

    __aicore__ inline uint32_t Max(int32_t a, int32_t b)
    {
        if (a > b)
            return a;
        return b;
    }
    __aicore__ inline uint32_t Min(int32_t a, int32_t b)
    {
        if (a > b)
            return b;
        return a;
    }

private:
    // Private Member
    TPipe* pipe;
    GlobalTensor<DTYPE_INDICES> indicesGm_, outputIndicesGm_, outputIndicesPairGm_;

    TBuf<TPosition::VECCALC> indicesUB_, outIndicesUB_, outIndicesPairUB_;
    LocalTensor<DTYPE_INDICES> indicesLocal_, indicesOffsetLocal_, indicesPairLocal_;

    uint32_t usedCoreNum;
    uint32_t coreTask;
    uint32_t lastCoreTask;

    uint32_t moveLen;

    uint32_t repeatTimes;
    uint32_t moveTail;
    uint32_t lastRepeatTimes;
    uint32_t lastMoveTail;

    uint32_t kernelD;
    uint32_t kernelH;
    uint32_t kernelW;
    uint32_t kernelSize;

    uint32_t outputDepth;
    uint32_t outputHeight;
    uint32_t outputWidth;

    uint32_t strideDepth;
    uint32_t strideHeight;
    uint32_t strideWidth;

    uint32_t paddingDepth;
    uint32_t paddingHeight;
    uint32_t paddingWidth;

    uint32_t curBlockIdx;

    uint32_t taskNum;
    uint32_t coreRepeatTimes;
    uint32_t coreMoveTail;
    uint32_t maskAlign;
    uint32_t mulmask;
    uint32_t mulRepeatTimes;
    uint32_t workSize;
};
extern "C" __global__ __aicore__ void sparse_conv3d(
    GM_ADDR indices, GM_ADDR indices_out, GM_ADDR indices_pair, GM_ADDR workspace, GM_ADDR tiling)
{
    SetSysWorkspace(workspace);
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    KernelSparseConv3d op;
    op.Init(indices, indices_out, indices_pair, workspace, &tiling_data, &pipe);
    op.Process();
}
