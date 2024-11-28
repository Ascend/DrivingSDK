/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef TO_SPARSE_V3_TILING_H
#define TO_SPARSE_V3_TILING_H

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ToSparseV3TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedVectorCoreNum)
    TILING_DATA_FIELD_DEF(uint32_t, kernelIC)
    TILING_DATA_FIELD_DEF(uint32_t, kernelSize)
    TILING_DATA_FIELD_DEF(uint32_t, moveLen)
    TILING_DATA_FIELD_DEF(uint32_t, vectorCoreTask)
    TILING_DATA_FIELD_DEF(uint32_t, vectorLastCoreTask)
    TILING_DATA_FIELD_DEF(uint32_t, coreRepeatTimes)
    TILING_DATA_FIELD_DEF(uint32_t, coreMoveLenTail)
    TILING_DATA_FIELD_DEF(uint32_t, lastCoreRepeatTimes)
    TILING_DATA_FIELD_DEF(uint32_t, lastCoreMoveLenTail)
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ToSparseV3, ToSparseV3TilingData)

class ToSparseV3Tiling {
public:
    explicit ToSparseV3Tiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();

private:
    ge::graphStatus GetVectorTilingData();
    ge::graphStatus GetCubeTilingData();
    ge::graphStatus SetTilingData();

private:
    gert::TilingContext* tilingContext = nullptr;
    ToSparseV3TilingData tilingData;
    uint32_t aivNum;
    uint32_t actualNum;
    uint32_t kernelOC;
    uint64_t availableUbSize;
    // Cube
    int32_t singleM;
    int32_t singleN;
    int32_t singleK;
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
    // TilingSet
    int32_t usedVectorCoreNum;
    uint32_t kernelIC;
    uint32_t kernelSize;
    uint32_t vectorCoreTask;
    uint32_t vectorLastCoreTask;
    uint32_t moveLen;
    uint32_t coreRepeatTimes;
    uint32_t coreMoveLenTail;
    uint32_t lastCoreRepeatTimes;
    uint32_t lastCoreMoveLenTail;
};
} // namespace optiling
#endif // TO_SPARSE_V3_TILING_H