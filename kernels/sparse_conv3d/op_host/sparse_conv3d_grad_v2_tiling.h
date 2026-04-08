/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef SPARSE_CONV3D_GRAD_V2_TILING_H
#define SPARSE_CONV3D_GRAD_V2_TILING_H

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
using namespace matmul_tiling;

namespace optiling {
BEGIN_TILING_DATA_DEF(SparseConv3dGradV2TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, usedVectorCoreNum)
    TILING_DATA_FIELD_DEF(uint32_t, featureCubeNum)
    TILING_DATA_FIELD_DEF(uint32_t, weightCubeNum)
    TILING_DATA_FIELD_DEF(uint64_t, kernelIC)
    TILING_DATA_FIELD_DEF(uint64_t, kernelOC)
    TILING_DATA_FIELD_DEF(uint64_t, kernelSize)
    TILING_DATA_FIELD_DEF(uint64_t, moveLen)
    TILING_DATA_FIELD_DEF(uint64_t, vectorActualNum)
    TILING_DATA_FIELD_DEF(uint64_t, vectorCoreTask)
    TILING_DATA_FIELD_DEF(uint64_t, vectorLastCoreTask)
    TILING_DATA_FIELD_DEF(uint64_t, coreRepeatTimes)
    TILING_DATA_FIELD_DEF(uint64_t, coreMoveLenTail)
    TILING_DATA_FIELD_DEF(uint64_t, lastCoreRepeatTimes)
    TILING_DATA_FIELD_DEF(uint64_t, lastCoreMoveLenTail)
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, featureCubeTilingData)
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, weightCubeTilingData)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SparseConv3dGradV2, SparseConv3dGradV2TilingData)

class SparseConv3dGradV2Tiling {
public:
    explicit SparseConv3dGradV2Tiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();

private:
    ge::graphStatus GetVectorTilingData();
    ge::graphStatus GetCubeTilingData();
    ge::graphStatus SetTilingData();
    ge::graphStatus MatMulGetData(uint64_t M, uint64_t N, uint64_t K, uint32_t mode);

private:
    gert::TilingContext* tilingContext = nullptr;
    SparseConv3dGradV2TilingData tilingData;
    uint32_t aivNum;
    uint32_t featureCubeNum;
    uint32_t weightCubeNum;
    uint64_t availableUbSize;

    // TilingSet
    int32_t actualNum;
    int32_t usedVectorCoreNum;
    uint64_t kernelIC;
    uint64_t kernelOC;
    uint64_t kernelSize;
    uint64_t vectorActualNum;
    uint64_t vectorCoreTask;
    uint64_t vectorLastCoreTask;
    uint64_t moveLen;
    uint64_t coreRepeatTimes;
    uint64_t coreMoveLenTail;
    uint64_t lastCoreRepeatTimes;
    uint64_t lastCoreMoveLenTail;
};
} // namespace optiling
#endif // SPARSE_CONV3D_GRAD_V2_TILING_H