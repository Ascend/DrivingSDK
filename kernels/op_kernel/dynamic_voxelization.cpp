/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t N_DIM = 3;
constexpr uint32_t TMP_QUE_SIZE = 5;
constexpr uint32_t UB_SIZE = 176128;
constexpr uint32_t VECORE_PROCESS_SIZE = 256;

using namespace AscendC;
class DynamicVoxKernel {
public:
    __aicore__ inline DynamicVoxKernel() = delete;
    __aicore__ inline DynamicVoxKernel(GM_ADDR points, GM_ADDR coors, GM_ADDR workspace,
                                       const DynamicVoxTilingData *__restrict tiling)
    {
        ASSERT(GetBlockNum() != 0 && "block num can not be zero");
        InitParams(tiling);
        SetCopyGmAddr(points, coors);
        SetUBSizeForData();
        InitBuffers();
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < ptsInUBParam_[0]; i++) {
            CopyIn(i, true);
            Compute(true);
            CopyOut(i, true);
        }
        for (uint32_t i = 0; i < ptsInUBParam_[1]; i++) {
            CopyIn(i, false);
            Compute(false);
            CopyOut(i, false);
        }
    }

private:
    __aicore__ inline void InitParams(const DynamicVoxTilingData *__restrict tiling)
    {
        blockIdx_ = GetBlockIdx();
        lastCoreIdx_ = GetBlockNum() - 1;
        ptsNum_ = tiling->pts_num;
        ptsFeature_ = tiling->pts_feature;
        ptsNumInCore_ = tiling->points_num_in_core;
        ptsNumInLastCore_ = tiling->points_num_in_last_core;

        voxelX_ = tiling->voxel_x;
        voxelY_ = tiling->voxel_y;
        voxelZ_ = tiling->voxel_z;

        coorsMinX_ = tiling->coors_min_x;
        coorsMinY_ = tiling->coors_min_y;
        coorsMinZ_ = tiling->coors_min_z;

        gridX_ = tiling->grid_x;
        gridY_ = tiling->grid_y;
        gridZ_ = tiling->grid_z;
    }

    __aicore__ inline void SetCopyGmAddr(GM_ADDR points, GM_ADDR coors)
    {
        // points
        uint64_t ptsNum = blockIdx_ == lastCoreIdx_ ? static_cast<uint64_t>(ptsNumInLastCore_) : static_cast<uint64_t>(ptsNumInCore_);
        uint64_t ptsAllNum = ptsNum * N_DIM;
        // coors: coors num equal to points num
        uint64_t coorsAllNum = ptsNum * N_DIM;

        ptsGm_.SetGlobalBuffer((__gm__ float *)points + blockIdx_ * ptsNumInCore_, ptsAllNum);
        coorsGm_.SetGlobalBuffer((__gm__ int *)coors + blockIdx_ * ptsNumInCore_, coorsAllNum);
    }

    __aicore__ inline void SetUBSizeForData()
    {
        // UB SIZE: points, coors, tmpTensor
        uint32_t ptOneDimSize = sizeof(float);
        uint32_t ubSizeForPtOneDim = UB_SIZE / (N_DIM + N_DIM + TMP_QUE_SIZE);
        if (blockIdx_ == lastCoreIdx_) {
            ComputeVariableInUBParam(ptsNumInLastCore_, ptOneDimSize, ubSizeForPtOneDim, ptsInUBParam_);
        } else {
            ComputeVariableInUBParam(ptsNumInCore_, ptOneDimSize, ubSizeForPtOneDim, ptsInUBParam_);
        }
    }

    __aicore__ inline void ComputeVariableInUBParam(const uint32_t allVarNum, const uint32_t oneVarSize,
                                                    const uint32_t ubSizeForVar, uint32_t varParam[])
    {
        // 1. compute max data num in ub (round down)
        ASSERT(oneVarSize != 0 && "one var size can not be zero");
        uint32_t varNumInUBMax = ubSizeForVar / oneVarSize;
        // 2. compute repeat time to copy all data (round up)
        uint32_t varCopyTime = (allVarNum + varNumInUBMax - 1) / varNumInUBMax;
        uint32_t alignNum = BLOCK_SIZE / oneVarSize;
        // 3. compute repeat time for former block and tail block
        varParam[0] = allVarNum % varCopyTime;
        varParam[1] = varCopyTime - varParam[0];
        // 4. compute data num once copy
        varParam[2] = (allVarNum + varCopyTime - 1) / varCopyTime;
        varParam[3] = allVarNum / varCopyTime;
        // 5. data num once copy align with 32B
        varParam[4] = ((varParam[2] + alignNum - 1) / alignNum) * alignNum;
        varParam[5] = ((varParam[3] + alignNum - 1) / alignNum) * alignNum;
    }

    __aicore__ inline void InitBuffers()
    {
        uint32_t allPtSize = ptsInUBParam_[4] * N_DIM * sizeof(float);
        pipe_.InitBuffer(ptsQue_, BUFFER_NUM, allPtSize);
        pipe_.InitBuffer(coorsQue_, BUFFER_NUM, allPtSize);
        pipe_.InitBuffer(tmpQue1_, BUFFER_NUM, ptsInUBParam_[4] * sizeof(float));
        pipe_.InitBuffer(tmpQue2_, BUFFER_NUM, ptsInUBParam_[4] * sizeof(float));
        pipe_.InitBuffer(tmpQue3_, BUFFER_NUM, ptsInUBParam_[4] * sizeof(float));
        pipe_.InitBuffer(tmpQue4_, BUFFER_NUM, ptsInUBParam_[4] * sizeof(float));
        pipe_.InitBuffer(tmpQue5_, BUFFER_NUM, ptsInUBParam_[4] * sizeof(float));
        pipe_.InitBuffer(voxelSizeXBuf_, VECORE_PROCESS_SIZE);
        pipe_.InitBuffer(voxelSizeYBuf_, VECORE_PROCESS_SIZE);
        pipe_.InitBuffer(voxelSizeZBuf_, VECORE_PROCESS_SIZE);
    }

    __aicore__ inline void CopyIn(const uint32_t progress, const bool formerFlag)
    {
        LocalTensor<float> ptsLocal = ptsQue_.AllocTensor<float>();
        uint32_t copyNum = 0;
        uint32_t copyNumAlign = 0;
        uint64_t ptsAddrOffset = 0;
        if (formerFlag) {
            copyNum = ptsInUBParam_[2];
            copyNumAlign = ptsInUBParam_[4];
            ptsAddrOffset = progress * copyNum;
        } else {
            copyNum = ptsInUBParam_[3];
            copyNumAlign = ptsInUBParam_[5];
            ptsAddrOffset = ptsInUBParam_[0] * ptsInUBParam_[2] + progress * copyNum;
        }
        DataCopyParams copyParams{1, static_cast<uint16_t>(copyNum * sizeof(float)), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};
        for (uint32_t i = 0; i < N_DIM; i++) {
            DataCopyPad(ptsLocal[i * copyNumAlign], ptsGm_[ptsAddrOffset + i * ptsNum_], copyParams, padParams);
        }
        ptsQue_.EnQue<float>(ptsLocal);
    }

    __aicore__ inline void Compute(const bool formerFlag)
    {
        uint32_t ptNumAlign = formerFlag ? ptsInUBParam_[4] : ptsInUBParam_[5];
        uint32_t ptNum = formerFlag ? ptsInUBParam_[2] : ptsInUBParam_[3];
        ComputeScalarParam();
        ComputePt2VoxCoors(ptNum, ptNumAlign);
    }

    __aicore__ inline void CopyOut(const uint32_t progress, const bool formerFlag)
    {
        LocalTensor<int> coorsLocal = coorsQue_.DeQue<int>();
        uint32_t copyNum = 0;
        uint32_t copyNumAlign = 0;
        uint64_t coorsAddrOffset = 0;
        if (formerFlag) {
            copyNum = ptsInUBParam_[2];
            copyNumAlign = ptsInUBParam_[4];
            coorsAddrOffset = progress * copyNum;
        } else {
            copyNum = ptsInUBParam_[3];
            copyNumAlign = ptsInUBParam_[5];
            coorsAddrOffset = ptsInUBParam_[0] * ptsInUBParam_[2] + progress * copyNum;
        }
        DataCopyParams copyParams{1, static_cast<uint16_t>(copyNum * sizeof(int)), 0, 0};
        for (uint32_t i = 0; i < N_DIM; i++) {
            DataCopyPad(coorsGm_[coorsAddrOffset + i * ptsNum_], coorsLocal[i * copyNumAlign], copyParams);
        }
        coorsQue_.FreeTensor<int>(coorsLocal);
    }

    __aicore__ inline void ComputeScalarParam()
    {
        scalarCoors_[0] = -1.0F * coorsMinX_;
        scalarCoors_[1] = -1.0F * coorsMinY_;
        scalarCoors_[2] = -1.0F * coorsMinZ_;
    }

    __aicore__ inline void DupScalar2Tensor(LocalTensor<float> &voxelSizeXT, LocalTensor<float> &voxelSizeYT,
                                            LocalTensor<float> &voxelSizeZT)
    {
        int32_t calcCnt = VECORE_PROCESS_SIZE / sizeof(float);
        Duplicate<float>(voxelSizeXT, voxelX_, calcCnt);
        Duplicate<float>(voxelSizeYT, voxelY_, calcCnt);
        Duplicate<float>(voxelSizeZT, voxelZ_, calcCnt);
    }

    template <typename T>
    __aicore__ inline void ComputeParamFor0Interface(const uint32_t dataNum, uint64_t &mask, uint8_t &repeatTime,
                                                     uint64_t &formerNum, uint64_t &tailNum)
    {
        mask = VECORE_PROCESS_SIZE / sizeof(T);
        repeatTime = static_cast<uint8_t>(dataNum / mask);
        uint32_t repeatTimeU32 = static_cast<uint32_t>(repeatTime);
        formerNum = static_cast<uint64_t>(repeatTimeU32 * mask);
        tailNum = static_cast<uint64_t>(dataNum - formerNum);
    }

    __aicore__ inline void ComputePt2VoxCoors(const uint32_t ptNum, const uint32_t ptNumAlign)
    {
        // coorsx = (ptx - coorsXmin) / voxX
        // coorsy = (pty - coorsYmin) / voxY
        // coorsz = (ptz - coorsZmin) / voxZ
        LocalTensor<float> ptsLocal = ptsQue_.DeQue<float>();
        LocalTensor<int32_t> coorsLocal = coorsQue_.AllocTensor<int32_t>();
        LocalTensor<float> coorsLocalx = tmpQue1_.AllocTensor<float>();
        LocalTensor<float> coorsLocaly = tmpQue2_.AllocTensor<float>();
        LocalTensor<float> coorsLocalz = tmpQue3_.AllocTensor<float>();
        LocalTensor<float> dstLocal = tmpQue4_.AllocTensor<float>();
        LocalTensor<uint8_t> selMask = tmpQue5_.AllocTensor<uint8_t>();
        LocalTensor<float> voxelSizeXT = voxelSizeXBuf_.Get<float>();
        LocalTensor<float> voxelSizeYT = voxelSizeYBuf_.Get<float>();
        LocalTensor<float> voxelSizeZT = voxelSizeZBuf_.Get<float>();
        DupScalar2Tensor(voxelSizeXT, voxelSizeYT, voxelSizeZT);
        // coors x
        ComputePtOneDim2VoxCoors(coorsLocalx, dstLocal, ptsLocal, voxelSizeXT, 0, scalarCoors_[0], ptNum);
        // coors y
        ComputePtOneDim2VoxCoors(coorsLocaly, dstLocal, ptsLocal, voxelSizeYT, ptNumAlign, scalarCoors_[1], ptNum);
        // coors z
        ComputePtOneDim2VoxCoors(coorsLocalz, dstLocal, ptsLocal, voxelSizeZT, 2 * ptNumAlign, scalarCoors_[2], ptNum);

        FillInvalidData(coorsLocalx, coorsLocaly, coorsLocalz, dstLocal, selMask, -1, gridX_, ptNum); // x
        FillInvalidData(coorsLocaly, coorsLocalx, coorsLocalz, dstLocal, selMask, -1, gridY_, ptNum); // y
        FillInvalidData(coorsLocalz, coorsLocalx, coorsLocaly, dstLocal, selMask, -1, gridZ_, ptNum); // z

        LocalTensor<int32_t> tmpLocal = dstLocal.ReinterpretCast<int32_t>();
        // copy coorsx to coors que
        CastCoorsToInt(coorsLocalz, tmpLocal, ptNum);
        CopyCoors(tmpLocal, coorsLocal, 0, ptNum);
        // copy coorsy to coors que
        CastCoorsToInt(coorsLocaly, tmpLocal, ptNum);
        CopyCoors(tmpLocal, coorsLocal, ptNumAlign, ptNum);
        // copy coorsz to coors que
        CastCoorsToInt(coorsLocalx, tmpLocal, ptNum);
        CopyCoors(tmpLocal, coorsLocal, ptNumAlign * 2, ptNum);

        coorsQue_.EnQue<int32_t>(coorsLocal);
        ptsQue_.FreeTensor<float>(ptsLocal);
        tmpQue1_.FreeTensor<float>(coorsLocalx);
        tmpQue2_.FreeTensor<float>(coorsLocaly);
        tmpQue3_.FreeTensor<float>(coorsLocalz);
        tmpQue4_.FreeTensor<int32_t>(tmpLocal);
        tmpQue5_.FreeTensor<uint8_t>(selMask);
    }

    __aicore__ inline void CastCoorsToInt(const LocalTensor<float> &srcLocal, LocalTensor<int> &dstLocal,
                                          const uint32_t ptNum)
    {
        uint64_t mask = 0;
        uint8_t repeatTime = 0;
        uint64_t formerNum = 0;
        uint64_t tailNum = 0;
        ComputeParamFor0Interface<float>(ptNum, mask, repeatTime, formerNum, tailNum);
        if (repeatTime > 0) {
            Cast(dstLocal, srcLocal, RoundMode::CAST_ROUND, mask, repeatTime, {1, 1, 8, 8});
        }
        if (tailNum > 0) {
            Cast(dstLocal[formerNum], srcLocal[formerNum], RoundMode::CAST_ROUND, tailNum, 1, {1, 1, 0, 0});
        }
    }

    __aicore__ inline void CopyCoors(const LocalTensor<int> &tmpLocal, LocalTensor<int> &coorsLocal,
                                     const uint32_t offset, const uint32_t ptNum)
    {
        uint64_t mask = 0;
        uint8_t repeatTime = 0;
        uint64_t formerNum = 0;
        uint64_t tailNum = 0;
        ComputeParamFor0Interface<float>(ptNum, mask, repeatTime, formerNum, tailNum);
        if (repeatTime > 0) {
            Copy(coorsLocal[offset], tmpLocal, mask, repeatTime, {1, 1, 8, 8});
        }
        if (tailNum > 0) {
            Copy(coorsLocal[offset + formerNum], tmpLocal[formerNum], tailNum, 1, {1, 1, 0, 0});
        }
    }

    __aicore__ inline void FillInvalidData(LocalTensor<float> &src0Local, LocalTensor<float> &src1Local,
                                           LocalTensor<float> &src2Local, LocalTensor<float> &thresholdLocal,
                                           LocalTensor<uint8_t> &selMask, const int minThresh, const int maxThresh,
                                           const uint32_t ptNum)
    {
        float negativeFlag = -1.0F;
        // mask with min threshold
        MaskWithThreshold(src0Local, thresholdLocal, selMask, maxThresh, ptNum, 0);
        SelectByMask<float>(selMask, src0Local, ptNum, negativeFlag);
        SelectByMask<float>(selMask, src1Local, ptNum, negativeFlag);
        SelectByMask<float>(selMask, src2Local, ptNum, negativeFlag);
        // mask with min threshold
        MaskWithThreshold(src0Local, thresholdLocal, selMask, minThresh, ptNum, 1);
        SelectByMask<float>(selMask, src0Local, ptNum, negativeFlag);
        SelectByMask<float>(selMask, src1Local, ptNum, negativeFlag);
        SelectByMask<float>(selMask, src2Local, ptNum, negativeFlag);
    }

    template <typename T>
    __aicore__ inline void SelectByMask(const LocalTensor<uint8_t> &selMask, LocalTensor<T> &srcLocal,
                                        const uint32_t ptNum, const T flag)
    {
        uint64_t mask = 0;
        uint8_t repeatTime = 0;
        uint64_t formerNum = 0;
        uint64_t tailNum = 0;
        ComputeParamFor0Interface<float>(ptNum, mask, repeatTime, formerNum, tailNum);
        if (repeatTime > 0) {
            Select(srcLocal, selMask, srcLocal, flag, SELMODE::VSEL_TENSOR_SCALAR_MODE, mask, repeatTime,
                   {1, 1, 0, 8, 8, 0});
        }
        if (tailNum > 0) {
            Select(srcLocal[formerNum], selMask[formerNum], srcLocal[formerNum], flag, SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   tailNum, 1, {1, 1, 0, 0, 0, 0});
        }
    }

    __aicore__ inline void MaskWithThreshold(LocalTensor<float> &srcLocal, LocalTensor<float> &thresholdLocal,
                                             LocalTensor<uint8_t> &selMask, const int threshold, const uint32_t ptNum,
                                             const int compareFlag)
    {
        float thresholdF = static_cast<float>(threshold);
        CMPMODE compareMode = compareFlag == 0 ? CMPMODE::LT : CMPMODE::GT;

        uint64_t mask = 0;
        uint8_t repeatTime = 0;
        uint64_t formerNum = 0;
        uint64_t tailNum = 0;
        ComputeParamFor0Interface<float>(ptNum, mask, repeatTime, formerNum, tailNum);
        if (repeatTime > 0) {
            Duplicate(thresholdLocal, thresholdF, mask, repeatTime, 1, 8);
            pipe_barrier(PIPE_V);
            Compare(selMask, srcLocal, thresholdLocal, compareMode, mask, repeatTime, {1, 1, 1, 8, 8, 8});
        }
        if (tailNum > 0) {
            Duplicate(thresholdLocal[formerNum], thresholdF, tailNum, 1, 1, 0);
            pipe_barrier(PIPE_V);
            Compare(selMask[formerNum], srcLocal[formerNum], thresholdLocal[formerNum], compareMode, tailNum, 1,
                    {1, 1, 1, 0, 0, 0});
        }
    }

    __aicore__ inline void ComputePtOneDim2VoxCoors(LocalTensor<float> &castLocal, LocalTensor<float> &dstLocal,
                                                    const LocalTensor<float> &ptsLocal, LocalTensor<float> &voxelSizeT,
                                                    const uint32_t offset, const float scalar, const uint32_t ptNum)
    {
        uint64_t mask = 0;
        uint8_t repeatTime = 0;
        uint64_t formerNum = 0;
        uint64_t tailNum = 0;
        ComputeParamFor0Interface<float>(ptNum, mask, repeatTime, formerNum, tailNum);
        if (repeatTime > 0) {
            Adds(dstLocal, ptsLocal[offset], scalar, mask, repeatTime, {1, 1, 8, 8});
            Div(dstLocal, dstLocal, voxelSizeT, mask, repeatTime, {1, 1, 1, 8, 8, 0});
            Cast(castLocal, dstLocal, RoundMode::CAST_FLOOR, mask, repeatTime, {1, 1, 8, 8});
        }
        if (tailNum > 0) {
            Adds(dstLocal[formerNum], ptsLocal[offset + formerNum], scalar, tailNum, 1, {1, 1, 0, 0});
            Div(dstLocal[formerNum], dstLocal[formerNum], voxelSizeT, tailNum, 1, {1, 1, 1, 0, 0, 0});
            Cast(castLocal[formerNum], dstLocal[formerNum], RoundMode::CAST_FLOOR, tailNum, 1, {1, 1, 0, 0});
        }
    }

private:
    TPipe pipe_;
    GlobalTensor<float> ptsGm_;
    GlobalTensor<int> coorsGm_;

    TQue<TPosition::VECIN, BUFFER_NUM> ptsQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> tmpQue1_;
    TQue<TPosition::VECIN, BUFFER_NUM> tmpQue2_;
    TQue<TPosition::VECIN, BUFFER_NUM> tmpQue3_;
    TQue<TPosition::VECIN, BUFFER_NUM> tmpQue4_;
    TQue<TPosition::VECIN, BUFFER_NUM> tmpQue5_;
    TQue<TPosition::VECOUT, BUFFER_NUM> coorsQue_;
    TBuf<TPosition::VECCALC> voxelSizeXBuf_, voxelSizeYBuf_, voxelSizeZBuf_;

    int ptsNum_;
    int ptsFeature_;
    int ptsNumInCore_;
    int ptsNumInLastCore_;
    // vox param
    float voxelX_;
    float voxelY_;
    float voxelZ_;
    // coors param
    float coorsMinX_;
    float coorsMinY_;
    float coorsMinZ_;
    // grid param
    int gridX_;
    int gridY_;
    int gridZ_;

    uint32_t blockIdx_;
    uint32_t lastCoreIdx_;

private:
    // 0：forme num; 1: tail num;
    // 2: data num in former; 3: data num in tail
    // 4: data num align with 32B in former; 5: data num align with 32B in tail
    uint32_t ptsInUBParam_[6];
    // 0: -coorMinX; 1: -coorsMinY; 2: -coorsMinZ
    float scalarCoors_[3];
    // 0: 1.0/voxelX; 1: 1.0/voxelY; 2: 1.0/voxelZ
    float reciproVoxel_[3];
};

extern "C" __global__ __aicore__ void dynamic_voxelization(GM_ADDR points, GM_ADDR coors, GM_ADDR workspace,
                                                           GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);

    GET_TILING_DATA(tilingData, tiling);
    const DynamicVoxTilingData *__restrict tilingDevice = &tilingData;
    DynamicVoxKernel op(points, coors, workspace, tilingDevice);
    op.Process();
}