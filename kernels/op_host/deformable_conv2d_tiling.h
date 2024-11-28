#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DeformableConv2dTilingData)
TILING_DATA_FIELD_DEF(uint32_t, n)
TILING_DATA_FIELD_DEF(uint32_t, cIn)
TILING_DATA_FIELD_DEF(uint32_t, hIn)
TILING_DATA_FIELD_DEF(uint32_t, wIn)
TILING_DATA_FIELD_DEF(uint32_t, cOut)
TILING_DATA_FIELD_DEF(uint32_t, hOut)
TILING_DATA_FIELD_DEF(uint32_t, wOut)
TILING_DATA_FIELD_DEF(uint32_t, kH)
TILING_DATA_FIELD_DEF(uint32_t, kW)
TILING_DATA_FIELD_DEF(int32_t, padH)
TILING_DATA_FIELD_DEF(int32_t, padW)
TILING_DATA_FIELD_DEF(int32_t, strideH)
TILING_DATA_FIELD_DEF(int32_t, strideW)
TILING_DATA_FIELD_DEF(int32_t, dilationH)
TILING_DATA_FIELD_DEF(int32_t, dilationW)
TILING_DATA_FIELD_DEF(uint32_t, usedBlkNum)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTilingData)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(DeformableConv2d, DeformableConv2dTilingData)
} // namespace optiling
