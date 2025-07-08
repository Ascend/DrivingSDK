#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DeformableConv2dV2TilingData)
TILING_DATA_FIELD_DEF(uint64_t, n)
TILING_DATA_FIELD_DEF(uint64_t, cIn)
TILING_DATA_FIELD_DEF(uint64_t, hIn)
TILING_DATA_FIELD_DEF(uint64_t, wIn)
TILING_DATA_FIELD_DEF(uint64_t, cOut)
TILING_DATA_FIELD_DEF(uint64_t, hOut)
TILING_DATA_FIELD_DEF(uint64_t, wOut)
TILING_DATA_FIELD_DEF(uint64_t, kH)
TILING_DATA_FIELD_DEF(uint64_t, kW)
TILING_DATA_FIELD_DEF(int64_t, padH)
TILING_DATA_FIELD_DEF(int64_t, padW)
TILING_DATA_FIELD_DEF(int64_t, strideH)
TILING_DATA_FIELD_DEF(int64_t, strideW)
TILING_DATA_FIELD_DEF(int64_t, dilationH)
TILING_DATA_FIELD_DEF(int64_t, dilationW)
TILING_DATA_FIELD_DEF(int64_t, groups)
TILING_DATA_FIELD_DEF(uint32_t, coreCount)
TILING_DATA_FIELD_DEF(uint32_t, avgTasks)
TILING_DATA_FIELD_DEF(uint32_t, bigCoreCount)
TILING_DATA_FIELD_DEF(uint32_t, cubeTileTaskCount)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(DeformableConv2dV2, DeformableConv2dV2TilingData)
} // namespace optiling