 /*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SubmSparseConv3dGradTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, core_used);
    TILING_DATA_FIELD_DEF(uint64_t, core_data);
    TILING_DATA_FIELD_DEF(uint64_t, copy_loop);
    TILING_DATA_FIELD_DEF(uint64_t, copy_tail);
    TILING_DATA_FIELD_DEF(uint64_t, last_copy_loop);
    TILING_DATA_FIELD_DEF(uint64_t, last_copy_tail);
    TILING_DATA_FIELD_DEF(uint64_t, inchannel);
    TILING_DATA_FIELD_DEF(uint64_t, outchannel);
    TILING_DATA_FIELD_DEF(uint64_t, indices_number);
    TILING_DATA_FIELD_DEF(uint64_t, kernel_size);
    TILING_DATA_FIELD_DEF(uint64_t, available_ub_size);
    TILING_DATA_FIELD_DEF(uint64_t, valid_number);
    TILING_DATA_FIELD_DEF(uint64_t, K0);
    TILING_DATA_FIELD_DEF(uint64_t, K1);
    TILING_DATA_FIELD_DEF(uint64_t, K2);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SubmSparseConv3dGrad, SubmSparseConv3dGradTilingData)
}
#endif // ADD_CUSTOM_TILING_H