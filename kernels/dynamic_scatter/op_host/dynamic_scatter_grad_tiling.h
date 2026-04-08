/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef DYNAMIC_SCATTER_GRAD_TILING_H
#define DYNAMIC_SCATTER_GRAD_TILING_H

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DynamicScatterGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalPointNum);
TILING_DATA_FIELD_DEF(uint32_t, totalVoxelNum);
TILING_DATA_FIELD_DEF(uint32_t, pointGradNum);
TILING_DATA_FIELD_DEF(uint32_t, voxelNumPerCore);
TILING_DATA_FIELD_DEF(uint32_t, eleNumPerCore);
TILING_DATA_FIELD_DEF(uint32_t, voxelNumLastCore);
TILING_DATA_FIELD_DEF(uint32_t, eleNumLastCore);
TILING_DATA_FIELD_DEF(uint32_t, alignedNum);
TILING_DATA_FIELD_DEF(uint32_t, featDim);
TILING_DATA_FIELD_DEF(uint32_t, featDimAligned);
TILING_DATA_FIELD_DEF(uint64_t, maskNum);
TILING_DATA_FIELD_DEF(uint32_t, maskDim);
TILING_DATA_FIELD_DEF(uint32_t, maskDimAligned);
TILING_DATA_FIELD_DEF(uint32_t, maxPointNum);
TILING_DATA_FIELD_DEF(uint32_t, blockLen);
TILING_DATA_FIELD_DEF(uint32_t, blockLenPad);
TILING_DATA_FIELD_DEF(uint32_t, blockLenMask);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(bool, isFeatsAligned);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DynamicScatterGrad, DynamicScatterGradTilingData)

class DynamicScatterGradTiling {
public:
    explicit DynamicScatterGradTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();

private:
    void SetTilingKeyMode(uint32_t reduceTypeNum) const;
    void CalUsedCoreNum(const uint32_t coreNumPlatform);
    ge::graphStatus CalTilingAligned();
    void CalMaskTiling();

private:
    DynamicScatterGradTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    uint32_t coreNum;
    uint32_t usedCoreNum = 1;
    uint32_t pointGradNum;
    uint32_t totalVoxelNum;
    uint32_t totalPointNum;
    uint32_t voxelNumPerCore;
    uint32_t voxelNumLastCore;
    uint32_t eleNumPerCore;
    uint32_t eleNumLastCore;
    uint32_t alignedNum;
    uint32_t featDim;
    uint32_t featDimAligned;
    uint32_t maskDim;
    uint32_t maskDimAligned;
    uint64_t maskNum = 0;
    uint32_t maxPointNum = 0;
    uint32_t blockLen;
    uint32_t blockLenPad = 0;
    uint32_t blockLenMask = 0;
    uint64_t ubSizePlatForm;
    bool isFeatsAligned = false;
};
} // namespace optiling
#endif // DYNAMIC_SCATTER_GRAD_TILING_H