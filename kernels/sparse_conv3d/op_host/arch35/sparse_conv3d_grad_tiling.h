#ifndef SPARSE_CONV3D_GRAD_TILING_H
#define SPARSE_CONV3D_GRAD_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SparseConv3dGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, usedVectorNum);
TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum);
TILING_DATA_FIELD_DEF(int32_t, totalPointsCount);
TILING_DATA_FIELD_DEF(int32_t, startOffset);
TILING_DATA_FIELD_DEF(uint32_t, inChannels);
TILING_DATA_FIELD_DEF(uint32_t, outChannels);
TILING_DATA_FIELD_DEF(uint32_t, ubMaxTaskNum);
TILING_DATA_FIELD_DEF(uint32_t, loopPointCount);
TILING_DATA_FIELD_DEF(uint64_t, featureWspOffset);
TILING_DATA_FIELD_DEF(uint32_t, sparseWspOffset);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, featureMatmulTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, weightMatmulTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SparseConv3dGrad, SparseConv3dGradTilingData)
} // namespace optiling
#endif
