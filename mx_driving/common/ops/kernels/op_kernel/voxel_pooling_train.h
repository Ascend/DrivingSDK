/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_utils.h"

using namespace AscendC;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t N_DIM = 3;
constexpr uint32_t UB_SIZE = 176128;
constexpr uint32_t UB_PART_FOR_TMP = 5;
constexpr uint32_t UB_PART_FOR_GEOM = 6;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t VECORE_PROCESS_SIZE = 256;


template <typename T>
class VoxelPoolingTrainKernel {
public:
    __aicore__ inline VoxelPoolingTrainKernel() = delete;
    __aicore__ inline VoxelPoolingTrainKernel(GM_ADDR geom, GM_ADDR featuresIn,
                                              GM_ADDR featuresOut, GM_ADDR posMemo,
                                              GM_ADDR workspace, const VoxelPoolingTilingData* __restrict tilingData)
    {
        ASSERT(GetBlockNum() != 0 && "block num can not be zero");
        // 1. 初始化参数
        InitParams(tilingData);
        // 2. 设置对应的gm地址
        SetGmAddr(geom, featuresIn, featuresOut, posMemo);
        // 3. 分配UB空间
        SetUBSizeForData();
        // 4. 初始化队列
        InitBuffers();
    }
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < batchSize_; i++) {
            for (uint32_t j = 0; j < dataInUBParam_[0]; j++) {
                CopyIn(i, j, true);
                Compute(i, true);
                CopyOut(i, j, true);
            }
            for (uint32_t j = 0; j < dataInUBParam_[1]; j++) {
                CopyIn(i, j, false);
                Compute(i, false);
                CopyOut(i, j, false);
            }
        }    
    }
private:
    __aicore__ inline void CopyIn(const uint32_t batchId, const uint32_t progress, const bool formerFlag)
    {
        LocalTensor<T> featureLocal = featuresQue_.AllocTensor<T>();
        LocalTensor<int> geoLocal = geomQue_.AllocTensor<int>();
        uint32_t copyNum = 0;
        uint32_t copyNumAlign = 0;
        uint32_t numChannelAlign = BLOCK_SIZE / sizeof(T);
        numChannelAlign = ((numChannels_ + numChannelAlign - 1) / numChannelAlign) * numChannelAlign;
        uint32_t featureAddrOffset = 0;
        uint32_t geomAddrOffset = 0;
        if (formerFlag) {
            copyNum = dataInUBParam_[2];
            copyNumAlign = dataInUBParam_[4];
            featureAddrOffset = batchId * numPts_ * numChannels_ +  progress * copyNum * numChannels_;
            geomAddrOffset = batchId * numPts_ * N_DIM + progress * copyNum;
        } else {
            copyNum = dataInUBParam_[3];
            copyNumAlign = dataInUBParam_[5];
            featureAddrOffset = (batchId * numPts_ + dataInUBParam_[0] * dataInUBParam_[2] + progress * copyNum) * numChannels_;
            geomAddrOffset = batchId * numPts_ * N_DIM + dataInUBParam_[0] * dataInUBParam_[2] + progress * copyNum;
        }

        DataCopyParams featCopyParams{1, static_cast<uint16_t>(numChannels_ * sizeof(T)), 0, 0};
        DataCopyParams geomCopyParams{1, static_cast<uint16_t>(copyNum * sizeof(int)), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};
        
        // copy in features
        for (uint32_t i = 0; i < copyNum; i++) {
            DataCopyPad(featureLocal[i * numChannelAlign], featuresInGm_[featureAddrOffset + i * numChannels_], featCopyParams, padParams);
        }
        // copy in geom xyz
        for (uint32_t i = 0; i < N_DIM; i++) {
            DataCopyPad(geoLocal[i * copyNumAlign], geomGm_[geomAddrOffset + i * numPts_], geomCopyParams, padParams);
        }

        featuresQue_.EnQue<T>(featureLocal);
        geomQue_.EnQue<int>(geoLocal);
    }

    __aicore__ inline void Compute(const uint32_t batchId, const bool formerFlag)
    {
        uint32_t dataNum = formerFlag ? dataInUBParam_[2] : dataInUBParam_[3];
        uint32_t dataNumAlign = formerFlag ? dataInUBParam_[4] : dataInUBParam_[5];
        // 1. compare geom three dim data with threshold, fill invalid data with -1
        ProcessDataInGeom(batchId, dataNum, dataNumAlign);
        // 2. 编码处理后的geomQue的元素
        EnCodeGeomCoord(dataNum);
        // 3. 获取idxLocal中小于0的mask，根据mask来把结果存到pos_memo中
        StoreData2PosMem(batchId, dataNum, dataNumAlign);
        // 4. 保留mask并转成int类型
        StoreMask2Int(dataNum);
    }

    __aicore__ inline void CopyOut(const uint32_t batchId, const uint32_t progress, const bool formerFlag)
    {
        uint32_t dataNum = formerFlag ? dataInUBParam_[2] : dataInUBParam_[3];
        AtomAddFeat2OutGm(batchId, dataNum);
        CopyGeomOut(batchId, progress, formerFlag);
    }

    __aicore__ inline void CopyGeomOut(const uint32_t batchId, const uint32_t progress, const bool formerFlag)
    {
        LocalTensor<int> posMemoLocal = posMemoQue_.DeQue<int>();
        uint32_t copyNum = 0;
        uint32_t copyNumAlign = 0;
        uint32_t posMemoOffset = 0;
        if (formerFlag) {
            copyNum = dataInUBParam_[2];
            copyNumAlign = dataInUBParam_[4];
            posMemoOffset = batchId * numPts_ * N_DIM + progress * copyNum;
        } else {
            copyNum = dataInUBParam_[3];
            copyNumAlign = dataInUBParam_[5];
            posMemoOffset = batchId * numPts_ * N_DIM + dataInUBParam_[2] * dataInUBParam_[0] + progress * copyNum;
        }

        DataCopyParams copyParams{1, static_cast<uint16_t>(copyNum * sizeof(int)), 0, 0};
        for (uint32_t i = 0; i < N_DIM; i++) {
            DataCopyPad(posMemoGm_[posMemoOffset + i * numPts_], posMemoLocal[i * copyNumAlign], copyParams);
        }
        posMemoQue_.FreeTensor<int>(posMemoLocal);
    }

    template <typename TYPE>
    __aicore__ inline void Compute0InterfaceParam(const uint32_t dataNum, uint64_t &mask, uint8_t &repeatTime,
                                                   uint64_t &formerNum, uint64_t &tailNum)
    {
        mask = VECORE_PROCESS_SIZE / sizeof(TYPE);
        repeatTime = static_cast<uint8_t>(dataNum / mask);
        formerNum = static_cast<uint64_t>(repeatTime) * mask;
        tailNum = static_cast<uint64_t>(dataNum - formerNum);
    }

    __aicore__ inline void StoreMask2Int(const uint32_t dataNum)
    {
        LocalTensor<int> maskLocal = tmpQue1_.AllocTensor<int>();
        LocalTensor<float> tmpLocal = tmpQue2_.AllocTensor<float>();
        LocalTensor<uint8_t> selMask = tmpQue3_.DeQue<uint8_t>();

        uint64_t mask = 0;
        uint8_t repeatTime = 0;
        uint64_t formerNum = 0;
        uint64_t tailNum = 0;
        Compute0InterfaceParam<float>(dataNum, mask, repeatTime, formerNum, tailNum);
        if (repeatTime > 0) {
            Duplicate(tmpLocal, 1.0F, mask, repeatTime, 1, 8);
            Select(tmpLocal, selMask, tmpLocal, 0.0F, SELMODE::VSEL_TENSOR_SCALAR_MODE, mask, repeatTime, {1, 1, 0, 8, 8, 0});
            Cast(maskLocal, tmpLocal, RoundMode::CAST_FLOOR, mask, repeatTime, {1, 1, 8, 8});
        }
        if (tailNum > 0) {
            Duplicate(tmpLocal[formerNum], 1.0F, tailNum, 1, 1, 0);
            Select(tmpLocal[formerNum], selMask[formerNum], tmpLocal[formerNum], 0.0F,
                SELMODE::VSEL_TENSOR_SCALAR_MODE, tailNum, 1, {1, 1, 0, 0, 0, 0});
            Cast(maskLocal[formerNum], tmpLocal[formerNum], RoundMode::CAST_FLOOR, tailNum, 1, {1, 1, 0, 0});
        }

        tmpQue1_.EnQue<int>(maskLocal);
        tmpQue2_.FreeTensor<float>(tmpLocal);
        tmpQue3_.FreeTensor<uint8_t>(selMask);
    }

    __aicore__ inline void AtomAddFeat2OutGm(const uint32_t batchId, const uint32_t dataNum)
    {
        LocalTensor<T> featureLocal = featuresQue_.DeQue<T>();
        LocalTensor<int> maskLocal = tmpQue1_.DeQue<int>();
        LocalTensor<float> idxLocal = tmpQue4_.DeQue<float>();
        float idxOffset = static_cast<float>((int)batchId * numVoxelX_ * numVoxelY_ * numChannels_);
        uint32_t numChannelAlign = BLOCK_SIZE / sizeof(T);
        numChannelAlign = ((numChannels_ + numChannelAlign - 1) / numChannelAlign) * numChannelAlign;

        uint64_t mask = 0;
        uint8_t repeatTime = 0;
        uint64_t formerNum = 0;
        uint64_t tailNum = 0;
        Compute0InterfaceParam<float>(dataNum, mask, repeatTime, formerNum, tailNum);
        if (repeatTime > 0) {
            Adds(idxLocal, idxLocal, idxOffset, mask, repeatTime, {1, 1, 8, 8});
        }
        if (tailNum > 0) {
            Adds(idxLocal[formerNum], idxLocal[formerNum], idxOffset, tailNum, 1, {1, 1, 0, 0});
        }
        DataCopyParams copyParams{1, static_cast<uint16_t>(numChannels_ * sizeof(T)), 0, 0};
        for (uint32_t i = 0; i < dataNum; i++) {
            if (maskLocal(i) == 0) {
                continue;
            }
            int32_t addrOffset = idxLocal(i);
            SetAtomicAdd<T>();
            DataCopyPad(featuresOutGm_[addrOffset], featureLocal[i * numChannelAlign], copyParams);
            SetAtomicNone();
        }

        featuresQue_.FreeTensor<T>(featureLocal);
        tmpQue1_.FreeTensor<int>(maskLocal);
        tmpQue4_.FreeTensor<float>(idxLocal);
    }

    __aicore__ inline void StoreData2PosMem(const int batchId, const uint32_t dataNum, const uint32_t dataNumAlign)
    {
        LocalTensor<float> geomLocalx = tmpQue1_.DeQue<float>();
        LocalTensor<float> geomLocaly = tmpQue2_.DeQue<float>();
        LocalTensor<float> idxLocal = tmpQue4_.DeQue<float>();
        LocalTensor<uint8_t> selMask = tmpQue3_.AllocTensor<uint8_t>();
        LocalTensor<float> threshLocal = tmpQue5_.AllocTensor<float>();

        uint64_t mask = 0;
        uint8_t repeatTime = 0;
        uint64_t formerNum = 0;
        uint64_t tailNum = 0;
        Compute0InterfaceParam<float>(dataNum, mask, repeatTime, formerNum, tailNum);
        GetMaskWithThresh(idxLocal, threshLocal, selMask, -1, dataNum, 1);
        SelectValidData(geomLocalx, selMask, dataNum, 0.0F);
        SelectValidData(geomLocaly, selMask, dataNum, 0.0F);

        tmpQue1_.FreeTensor<float>(geomLocalx);
        tmpQue2_.FreeTensor<float>(geomLocaly);
        tmpQue5_.FreeTensor<float>(threshLocal);
        tmpQue3_.EnQue<uint8_t>(selMask);
        tmpQue4_.EnQue<float>(idxLocal);
    }

    __aicore__ inline void EnCodeGeomCoord(const uint32_t dataNum)
    {
        LocalTensor<float> geomLocalx = tmpQue1_.DeQue<float>();
        LocalTensor<float> geomLocaly = tmpQue2_.DeQue<float>();
        LocalTensor<float> geomLocalz = tmpQue3_.DeQue<float>();
        LocalTensor<float> idxLocal = tmpQue4_.AllocTensor<float>();

        uint64_t mask = 0;
        uint8_t repeatTime = 0;
        uint64_t formerNum = 0;
        uint64_t tailNum = 0;
        Compute0InterfaceParam<float>(dataNum, mask, repeatTime, formerNum, tailNum);
        if (repeatTime > 0) {
            Muls(geomLocalx, geomLocalx, static_cast<float>(numChannels_), mask, repeatTime, {1, 1, 8, 8});
            Muls(geomLocaly, geomLocaly, static_cast<float>(numChannels_ * numVoxelX_), mask, repeatTime, {1, 1, 8, 8});
            Add(idxLocal, geomLocalx, geomLocaly, mask, repeatTime, {1, 1, 1, 8, 8, 8});
        }
        if (tailNum > 0) {
            Muls(geomLocalx[formerNum], geomLocalx[formerNum], static_cast<float>(numChannels_), tailNum, 1, {1, 1, 0, 0});
            Muls(geomLocaly[formerNum], geomLocaly[formerNum], static_cast<float>(numChannels_ * numVoxelX_), tailNum, 1, {1, 1, 0, 0});
            Add(idxLocal[formerNum], geomLocalx[formerNum], geomLocaly[formerNum], tailNum, 1, {1, 1, 1, 0, 0, 0});
        }

        tmpQue1_.EnQue<float>(geomLocalx);
        tmpQue2_.EnQue<float>(geomLocaly);
        tmpQue4_.EnQue<float>(idxLocal);
        tmpQue3_.FreeTensor<float>(geomLocalz);
    }

    __aicore__ inline void GetMaskWithThresh(LocalTensor<float> &src0Local, LocalTensor<float> &thresholdLocal,
                                                LocalTensor<uint8_t> &selMask, const int threshold,
                                                const uint32_t dataNum, const int compareFlag)
    {
        float thresholdF = static_cast<float>(threshold);
        CMPMODE compareMode = compareFlag == 0 ? CMPMODE::LT : CMPMODE::GT;

        uint64_t mask = 0;
        uint8_t repeatTime = 0;
        uint64_t formerNum = 0;
        uint64_t tailNum = 0;
        Compute0InterfaceParam<float>(dataNum, mask, repeatTime, formerNum, tailNum);
        if (repeatTime > 0) {
            Duplicate(thresholdLocal, thresholdF, mask, repeatTime, 1, 8);
            pipe_barrier(PIPE_V);
            Compare(selMask, src0Local, thresholdLocal, compareMode, mask, repeatTime, {1, 1, 1, 8, 8, 8});
        }
        if (tailNum > 0) {
            Duplicate(thresholdLocal[formerNum], thresholdF, tailNum, 1, 1, 0);
            pipe_barrier(PIPE_V);
            Compare(selMask[formerNum], src0Local[formerNum], thresholdLocal[formerNum], compareMode, tailNum, 1, {1, 1, 1, 0, 0, 0});
        }
    }

    __aicore__ inline void CompareWithThreshold(LocalTensor<float> &src0Local, LocalTensor<float> &src1Local,
                                                LocalTensor<float> &src2Local, LocalTensor<float> &threshLocal,
                                                LocalTensor<uint8_t> &selMask, const int minThresh,
                                                const int maxThresh, const uint32_t dataNum)
    {
        // 1. get mask by threshold
        // 2. fill invalid data with -1
        const float fillData = -1.0F;
        // min threshold
        GetMaskWithThresh(src0Local, threshLocal, selMask, minThresh, dataNum, 1);
        SelectValidData(src0Local, selMask, dataNum, fillData);
        SelectValidData(src1Local, selMask, dataNum, fillData);
        SelectValidData(src2Local, selMask, dataNum, fillData);
        // max threshold
        GetMaskWithThresh(src0Local, threshLocal, selMask, maxThresh, dataNum, 0);
        SelectValidData(src0Local, selMask, dataNum, fillData);
        SelectValidData(src1Local, selMask, dataNum, fillData);
        SelectValidData(src2Local, selMask, dataNum, fillData);
    }

    __aicore__ inline void ProcessDataInGeom(const int32_t batchId, const uint32_t dataNum, const uint32_t dataNumAlign)
    {
        LocalTensor<int> geomLocal = geomQue_.DeQue<int>();
        LocalTensor<float> geomLocalx = tmpQue1_.AllocTensor<float>();
        LocalTensor<float> geomLocaly = tmpQue2_.AllocTensor<float>();
        LocalTensor<float> geomLocalz = tmpQue3_.AllocTensor<float>();
        LocalTensor<float> threshLocal = tmpQue4_.AllocTensor<float>();
        LocalTensor<uint8_t> selMask = tmpQue5_.AllocTensor<uint8_t>();
        LocalTensor<int> posMemoLocal = posMemoQue_.AllocTensor<int>();

        uint64_t mask = 0;
        uint8_t repeatTime = 0;
        uint64_t formerNum = 0;
        uint64_t tailNum = 0;
        Compute0InterfaceParam<float>(dataNum, mask, repeatTime, formerNum, tailNum);
        if (repeatTime > 0) {
            Cast(geomLocalx, geomLocal, RoundMode::CAST_FLOOR, mask, repeatTime, {1, 1, 8, 8});
            Cast(geomLocaly, geomLocal[dataNumAlign], RoundMode::CAST_FLOOR, mask, repeatTime, {1, 1, 8, 8});
            Cast(geomLocalz, geomLocal[2 * dataNumAlign], RoundMode::CAST_FLOOR, mask, repeatTime, {1, 1, 8, 8});

            Duplicate(posMemoLocal, batchId, mask, repeatTime, 1, 8);
            Cast(posMemoLocal[dataNumAlign], geomLocaly, RoundMode::CAST_FLOOR, mask, repeatTime, {1, 1, 8, 8});
            Cast(posMemoLocal[2 * dataNumAlign], geomLocalx, RoundMode::CAST_FLOOR, mask, repeatTime, {1, 1, 8, 8});
        }
        if (tailNum > 0) {
            Cast(geomLocalx[formerNum], geomLocal[formerNum], RoundMode::CAST_FLOOR, tailNum, 1, {1, 1, 0, 0});
            Cast(geomLocaly[formerNum], geomLocal[dataNumAlign + formerNum], RoundMode::CAST_FLOOR, tailNum, 1, {1, 1, 0, 0});
            Cast(geomLocalz[formerNum], geomLocal[2 * dataNumAlign + formerNum], RoundMode::CAST_FLOOR, tailNum, 1, {1, 1, 0, 0});

            Duplicate(posMemoLocal[formerNum], batchId, tailNum, 1, 1, 0);
            Cast(posMemoLocal[dataNumAlign + formerNum], geomLocaly[formerNum], RoundMode::CAST_FLOOR, tailNum, 1, {1, 1, 0, 0});
            Cast(posMemoLocal[2 * dataNumAlign + formerNum], geomLocalx[formerNum], RoundMode::CAST_FLOOR, tailNum, 1, {1, 1, 0, 0});
        }

        CompareWithThreshold(geomLocalx, geomLocaly, geomLocalz, threshLocal, selMask, -1, numVoxelX_, dataNum);
        CompareWithThreshold(geomLocaly, geomLocalx, geomLocalz, threshLocal, selMask, -1, numVoxelY_, dataNum);
        CompareWithThreshold(geomLocalz, geomLocalx, geomLocaly, threshLocal, selMask, -1, numVoxelZ_, dataNum);

        tmpQue1_.EnQue<float>(geomLocalx);
        tmpQue2_.EnQue<float>(geomLocaly);
        tmpQue3_.EnQue<float>(geomLocalz);
        geomQue_.FreeTensor<int>(geomLocal);
        tmpQue4_.FreeTensor<float>(threshLocal);
        tmpQue5_.FreeTensor<uint8_t>(selMask);
        posMemoQue_.EnQue<int>(posMemoLocal);
    }

    __aicore__ inline void SelectValidData(LocalTensor<float> &srcLocal, LocalTensor<uint8_t> &selMask,
                                          const uint32_t dataNum, const float invalidFlag)
    {
        uint64_t mask = 0;
        uint8_t repeatTime = 0;
        uint64_t formerNum = 0;
        uint64_t tailNum = 0;
        Compute0InterfaceParam<float>(dataNum, mask, repeatTime, formerNum, tailNum);
        if (repeatTime > 0) {
            Select(srcLocal, selMask, srcLocal, invalidFlag, SELMODE::VSEL_TENSOR_SCALAR_MODE, mask, repeatTime, {1, 1, 0, 8, 8, 0});
        }
        if (tailNum > 0) {
            Select(srcLocal[formerNum], selMask[formerNum], srcLocal[formerNum], invalidFlag, SELMODE::VSEL_TENSOR_SCALAR_MODE, tailNum, 1, {1, 1, 0, 0, 0, 0});
        }
    }

    __aicore__ inline void InitParams(const VoxelPoolingTilingData* __restrict tilingData)
    {
        uint32_t numInBlock = BLOCK_SIZE / sizeof(T);
        blockIdx_ = GetBlockIdx();
        lastCoreIdx_ = GetBlockNum() - 1;
        multiple_ = sizeof(int) / sizeof(T);
        batchSize_ = tilingData->batch_size;
        numPts_ = tilingData->num_points;
        numChannels_ = tilingData->num_channels;
        numChannelAlign_ = ((numChannels_ + numInBlock - 1) / numInBlock) * numInBlock;
        numVoxelX_ = tilingData->num_voxel_x;
        numVoxelY_ = tilingData->num_voxel_y;
        numVoxelZ_ = tilingData->num_voxel_z;
        featuresInCore_ = tilingData->features_num_in_core;
        featuresInLastCore_ = tilingData->features_num_in_last_core;
    }

    __aicore__ inline void SetGmAddr(GM_ADDR geom, GM_ADDR featuresIn, GM_ADDR featuresOut, GM_ADDR posMemo)
    {
        uint32_t featuresNumOneCore = featuresInCore_ * numChannels_;
        geomGm_.SetGlobalBuffer((__gm__ int*)geom + blockIdx_ * featuresInCore_, N_DIM * featuresInCore_);
        posMemoGm_.SetGlobalBuffer((__gm__ int*)posMemo + blockIdx_ * featuresInCore_, N_DIM * featuresInCore_);
        featuresInGm_.SetGlobalBuffer((__gm__ T*)featuresIn + blockIdx_ * featuresNumOneCore, featuresNumOneCore);
        featuresOutGm_.SetGlobalBuffer((__gm__ T*)featuresOut, batchSize_ * numPts_ * numChannels_);
    }

    __aicore__ inline void SetUBSizeForData()
    {
        uint32_t featOneDimSize = sizeof(T);
        uint32_t ubSizeForOneParam = UB_SIZE / (numChannelAlign_ + multiple_ * (UB_PART_FOR_GEOM + UB_PART_FOR_TMP));
        if (blockIdx_ == lastCoreIdx_) {
            ComputeVariableInUBParam(featuresInLastCore_, featOneDimSize, ubSizeForOneParam, dataInUBParam_);
        } else {
            ComputeVariableInUBParam(featuresInCore_, featOneDimSize, ubSizeForOneParam, dataInUBParam_);
        }
    }

    __aicore__ inline void InitBuffers()
    {
        uint32_t allFeatureSize = dataInUBParam_[4] * numChannelAlign_ * sizeof(T);
        uint32_t posMemoSize = multiple_ * dataInUBParam_[4] * N_DIM * sizeof(T);
        pipe_.InitBuffer(featuresQue_, BUFFER_NUM, allFeatureSize);
        pipe_.InitBuffer(posMemoQue_, BUFFER_NUM, posMemoSize);
        pipe_.InitBuffer(geomQue_, BUFFER_NUM, posMemoSize);
        pipe_.InitBuffer(tmpQue1_, BUFFER_NUM, multiple_ * dataInUBParam_[4] * sizeof(T));
        pipe_.InitBuffer(tmpQue2_, BUFFER_NUM, multiple_ * dataInUBParam_[4] * sizeof(T));
        pipe_.InitBuffer(tmpQue3_, BUFFER_NUM, multiple_ * dataInUBParam_[4] * sizeof(T));
        pipe_.InitBuffer(tmpQue4_, BUFFER_NUM, multiple_ * dataInUBParam_[4] * sizeof(T));
        pipe_.InitBuffer(tmpQue5_, BUFFER_NUM, multiple_ * dataInUBParam_[4] * sizeof(T));
    }

    __aicore__ inline void ComputeVariableInUBParam(const uint32_t allDataNum, const uint32_t oneDataSize,
                                                    const uint32_t ubSizeForData, uint32_t dataParam[])
    {
        ASSERT(oneDataSize != 0 && "one data size can not be zero");
        // 1. compute max data num in ub (round down)
        uint32_t dataNumInUBMax = ubSizeForData / oneDataSize;
        // 2. compute repeat time to copy all data (round up)
        uint32_t dataCopyTime = (allDataNum + dataNumInUBMax - 1) / dataNumInUBMax;
        uint32_t alignNum = BLOCK_SIZE / oneDataSize;
        // 3. compute repeat time for former block and tail block
        dataParam[0] = allDataNum % dataCopyTime;
        dataParam[1] = dataCopyTime - dataParam[0];
        // 4. compute data num once copy
        dataParam[2] = (allDataNum + dataCopyTime - 1) / dataCopyTime;
        dataParam[3] = allDataNum / dataCopyTime;
        // 5. data num once copy align with 32B
        dataParam[4] = ((dataParam[2] + alignNum - 1) / alignNum) * alignNum;
        dataParam[5] = ((dataParam[3] + alignNum - 1) / alignNum) * alignNum;
    }

private:
    TPipe pipe_;
    GlobalTensor<T> featuresInGm_;
    GlobalTensor<T> featuresOutGm_;
    GlobalTensor<int> geomGm_;
    GlobalTensor<int> posMemoGm_;

    TQue<TPosition::VECIN, BUFFER_NUM> featuresQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> geomQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> tmpQue1_;
    TQue<TPosition::VECIN, BUFFER_NUM> tmpQue2_;
    TQue<TPosition::VECIN, BUFFER_NUM> tmpQue3_;
    TQue<TPosition::VECIN, BUFFER_NUM> tmpQue4_;
    TQue<TPosition::VECIN, BUFFER_NUM> tmpQue5_;
    TQue<TPosition::VECOUT, BUFFER_NUM> posMemoQue_;

private:
    int blockIdx_;
    int lastCoreIdx_;
    int multiple_;

    int batchSize_;
    int numPts_;
    int numChannels_;
    int numChannelAlign_;
    int numVoxelX_;
    int numVoxelY_;
    int numVoxelZ_;
    int featuresInCore_;
    int featuresInLastCore_;

    uint32_t dataInUBParam_[6];
};
