
#ifndef geometric_kernel_attention_tiling_h
#define geometric_kernel_attention_tiling_h
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GeometricKernelAttentionTilingData)
    TILING_DATA_FIELD_DEF(int32_t, batchSize);
    TILING_DATA_FIELD_DEF(int32_t, numKeys);
    TILING_DATA_FIELD_DEF(int32_t, numHeads);
    TILING_DATA_FIELD_DEF(int32_t, numQueries);
    TILING_DATA_FIELD_DEF(int32_t, numLevels);
    TILING_DATA_FIELD_DEF(int32_t, numPoints);
    TILING_DATA_FIELD_DEF(int32_t, dim);
    TILING_DATA_FIELD_DEF(int32_t, alignLevels);
    TILING_DATA_FIELD_DEF(int32_t, alignDim);
    TILING_DATA_FIELD_DEF(int32_t, totalTaskNum);
    TILING_DATA_FIELD_DEF(int32_t, alignTaskNum);
    TILING_DATA_FIELD_DEF(int32_t, tailNum);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    TILING_DATA_FIELD_DEF(uint32_t, taskNumPerScore);
    TILING_DATA_FIELD_DEF(uint32_t, taskNumPerLcore);
    TILING_DATA_FIELD_DEF(uint32_t, scoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, lcoreNum);
    TILING_DATA_FIELD_DEF(uint64_t, ubTotalSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GeometricKernelAttention, GeometricKernelAttentionTilingData)
}
#endif