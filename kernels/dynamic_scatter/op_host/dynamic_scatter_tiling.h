/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef DYNAMIC_SCATTER_TILING_H
#define DYNAMIC_SCATTER_TILING_H

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DynamicScatterTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalPointNum);
TILING_DATA_FIELD_DEF(uint32_t, totalVoxelNum);
TILING_DATA_FIELD_DEF(uint32_t, featsDim);
TILING_DATA_FIELD_DEF(uint32_t, pointFeatsNum);
TILING_DATA_FIELD_DEF(uint32_t, voxelNumPerCore);
TILING_DATA_FIELD_DEF(uint32_t, voxelNumLastCore);
TILING_DATA_FIELD_DEF(uint32_t, voxelFeatsNumPerCore);
TILING_DATA_FIELD_DEF(uint32_t, voxelFeatsNumLastCore);
TILING_DATA_FIELD_DEF(uint32_t, alignedNum);
TILING_DATA_FIELD_DEF(uint32_t, featsDimAligned);
TILING_DATA_FIELD_DEF(uint32_t, availablePointNum);
TILING_DATA_FIELD_DEF(uint64_t, maskNum);
TILING_DATA_FIELD_DEF(uint32_t, maskDim);
TILING_DATA_FIELD_DEF(uint32_t, maskDimAligned);
TILING_DATA_FIELD_DEF(uint32_t, maskDimAlignedB16);
TILING_DATA_FIELD_DEF(uint32_t, blockLen);
TILING_DATA_FIELD_DEF(uint32_t, blockLenPad);
TILING_DATA_FIELD_DEF(uint32_t, blockLenMask);
TILING_DATA_FIELD_DEF(uint32_t, repeatTimes);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(bool, isFeatsAligned);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DynamicScatter, DynamicScatterTilingData)

class DynamicScatterTiling {
public:
    explicit DynamicScatterTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();

private:
    void CalUsedCoreNum();
    ge::graphStatus CalTilingAligned();
    void CalMaskTiling();
    void CalAvailableUbTiling();

private:
    DynamicScatterTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    uint32_t coreNum;
    uint32_t usedCoreNum = 1;
    uint32_t totalPointNum;
    uint32_t totalVoxelNum;
    uint32_t featsDim;
    uint32_t pointFeatsNum;
    uint32_t voxelNumPerCore;
    uint32_t voxelNumLastCore;
    uint32_t voxelFeatsNumPerCore;
    uint32_t voxelFeatsNumLastCore;
    uint32_t alignedNum;
    uint32_t featsDimAligned;
    uint32_t availablePointNum = 1;
    uint64_t maskNum = 0;
    uint32_t maskDim = 0;
    uint32_t maskDimAligned = 0;
    uint32_t maskDimAlignedB16 = 0;
    uint32_t blockLen = 0;
    uint32_t blockLenPad = 0;
    uint32_t blockLenMask = 0;
    uint32_t repeatTimes = 1;
    uint64_t ubSizePlatForm;
    bool isFeatsAligned = false;
};
} // namespace optiling
#endif // DYNAMIC_SCATTER_TILING_H
