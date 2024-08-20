/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef SPARSE_INVERSE_CONV3D_TILING_H
#define SPARSE_INVERSE_CONV3D_TILING_H

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SparseInverseConv3dTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
    TILING_DATA_FIELD_DEF(uint32_t, coreTask)
    TILING_DATA_FIELD_DEF(uint32_t, lastCoreTask)
    TILING_DATA_FIELD_DEF(uint32_t, moveLen)
    TILING_DATA_FIELD_DEF(uint32_t, repeatTimes)
    TILING_DATA_FIELD_DEF(uint32_t, moveTail)
    TILING_DATA_FIELD_DEF(uint32_t, lastRepeatTimes)
    TILING_DATA_FIELD_DEF(uint32_t, lastMoveTail)
    TILING_DATA_FIELD_DEF(uint32_t, kernelD)
    TILING_DATA_FIELD_DEF(uint32_t, kernelH)
    TILING_DATA_FIELD_DEF(uint32_t, kernelW)
    TILING_DATA_FIELD_DEF(uint32_t, kernelIC)
    TILING_DATA_FIELD_DEF(uint32_t, kernelOC)
    TILING_DATA_FIELD_DEF(uint32_t, kernelSize)
    TILING_DATA_FIELD_DEF(uint32_t, outfeatureB)
    TILING_DATA_FIELD_DEF(uint32_t, outputDepth)
    TILING_DATA_FIELD_DEF(uint32_t, outputHeight)
    TILING_DATA_FIELD_DEF(uint32_t, outputWidth)
    TILING_DATA_FIELD_DEF(uint32_t, strideDepth)
    TILING_DATA_FIELD_DEF(uint32_t, strideHeight)
    TILING_DATA_FIELD_DEF(uint32_t, strideWidth)
    TILING_DATA_FIELD_DEF(uint32_t, paddingDepth)
    TILING_DATA_FIELD_DEF(uint32_t, paddingHeight)
    TILING_DATA_FIELD_DEF(uint32_t, paddingWidth)
    TILING_DATA_FIELD_DEF(uint32_t, dilationDepth)
    TILING_DATA_FIELD_DEF(uint32_t, dilationHeight)
    TILING_DATA_FIELD_DEF(uint32_t, dilationWidth)
    TILING_DATA_FIELD_DEF(uint32_t, outputPaddingDepth)
    TILING_DATA_FIELD_DEF(uint32_t, outputPaddingHeight)
    TILING_DATA_FIELD_DEF(uint32_t, outputPaddingWidth)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SparseInverseConv3d, SparseInverseConv3dTilingData)

class SparseInverseConv3dTiling {
public:
    explicit SparseInverseConv3dTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();

private:
    void CalUsedCoreNumAndCoreTask();
    void CalAvailableUbTiling();
    void GetIntArrayList();

private:
    SparseInverseConv3dTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    uint32_t coreNum;
    uint32_t usedCoreNum;
    uint32_t coreTask;
    uint32_t lastCoreTask;
    uint32_t actualNum;
    uint32_t kernelD;
    uint32_t kernelH;
    uint32_t kernelW;
    uint32_t kernelOC;
    uint32_t kernelIC;
    uint32_t kernelSize;
    uint64_t ubSizePlatForm;
    uint32_t moveLen;
    uint32_t repeatTimes;
    uint32_t moveTail;
    uint32_t lastRepeatTimes;
    uint32_t lastMoveTail;
};
} // namespace optiling
#endif // SPARSE_INVERSE_CONV3D_TILING_H
