#ifndef SUMB_SPARSE_CONV3D_GRAD_V2_TILING_H
#define SUMB_SPARSE_CONV3D_GRAD_V2_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SubmConv3dGradV2TillingData)
    TILING_DATA_FIELD_DEF(int64_t, aivNum);
    TILING_DATA_FIELD_DEF(int32_t, k0);
    TILING_DATA_FIELD_DEF(int32_t, k1);
    TILING_DATA_FIELD_DEF(int32_t, k2);
    TILING_DATA_FIELD_DEF(int32_t, inChannels);
    TILING_DATA_FIELD_DEF(int32_t, outChannels);
    TILING_DATA_FIELD_DEF(int32_t, coreTaskCount);
    TILING_DATA_FIELD_DEF(int32_t, bigCoreCount);
    TILING_DATA_FIELD_DEF(int32_t, singleLoopTask);
    TILING_DATA_FIELD_DEF(int32_t, totalTaskCount);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, featureMatmulTilingData);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, weightMatmulTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SubmSparseConv3dGradV2, SubmConv3dGradV2TillingData)
} // namespace optiling
#endif