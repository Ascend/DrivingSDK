#ifndef SPARSE_CONV3D_GRAD_TILING_H
#define SPARSE_CONV3D_GRAD_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SparseConv3dGradTillingData)
TILING_DATA_FIELD_DEF(uint32_t, usedVectorNum);
TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum);
TILING_DATA_FIELD_DEF(uint32_t, inChannels);
TILING_DATA_FIELD_DEF(uint32_t, outChannels);
TILING_DATA_FIELD_DEF(uint32_t, sparseRatio);
TILING_DATA_FIELD_DEF(uint32_t, ubMaxTaskNum);
TILING_DATA_FIELD_DEF(uint32_t, mainCoreTask);
TILING_DATA_FIELD_DEF(uint32_t, lastCoreTask);

TILING_DATA_FIELD_DEF(uint64_t, featuresGradSize);
TILING_DATA_FIELD_DEF(uint64_t, weightGradSize);
TILING_DATA_FIELD_DEF(uint64_t, featuresWorkSpaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, tmpGradFeaturesWorkSpaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, startIndicesWorkSpaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, endIndicesWorkSpaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, inputIndicesPtrWorkSpaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, inputIndicesWorkSpaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, kernelIndicesWorkSpaceOffset);

TILING_DATA_FIELD_DEF(uint32_t, tmpSortSize);
TILING_DATA_FIELD_DEF(uint32_t, kernelSizeAlign32);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, featureMatmulTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, weightMatmulTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SparseConv3dGrad, SparseConv3dGradTillingData)
} // namespace optiling
#endif