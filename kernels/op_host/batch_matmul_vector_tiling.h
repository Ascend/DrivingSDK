 /*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BatchMatmulVectorTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, core_used);
    TILING_DATA_FIELD_DEF(uint64_t, core_data);
    TILING_DATA_FIELD_DEF(uint64_t, copy_loop);
    TILING_DATA_FIELD_DEF(uint64_t, copy_tail);
    TILING_DATA_FIELD_DEF(uint64_t, last_copy_loop);
    TILING_DATA_FIELD_DEF(uint64_t, last_copy_tail);
    TILING_DATA_FIELD_DEF(uint64_t, available_ub_size);
    TILING_DATA_FIELD_DEF(uint64_t, totalresult);
    TILING_DATA_FIELD_DEF(uint64_t, ptstotal);
    TILING_DATA_FIELD_DEF(uint64_t, dim4);
    TILING_DATA_FIELD_DEF(uint64_t, dim5);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BatchMatmulVector, BatchMatmulVectorTilingData)
}
#endif // ADD_CUSTOM_TILING_H