#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
    BEGIN_TILING_DATA_DEF(DeformableConv2dTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, c_in);
    TILING_DATA_FIELD_DEF(uint32_t, h_in);
    TILING_DATA_FIELD_DEF(uint32_t, w_in);
    TILING_DATA_FIELD_DEF(uint32_t, c_out);
    TILING_DATA_FIELD_DEF(uint32_t, h_out);
    TILING_DATA_FIELD_DEF(uint32_t, w_out);
    TILING_DATA_FIELD_DEF(uint32_t, x_offset_unit);
    TILING_DATA_FIELD_DEF(uint32_t, c_in_aligned);
    TILING_DATA_FIELD_DEF(uint32_t, x_size);
    TILING_DATA_FIELD_DEF(uint32_t, weight_size);
    TILING_DATA_FIELD_DEF(uint32_t, use_core_num);
    TILING_DATA_FIELD_DEF(uint32_t, core_avg_task);
    TILING_DATA_FIELD_DEF(uint32_t, main_core_num);
    TILING_DATA_FIELD_DEF(uint32_t, available_ub_elem);
    TILING_DATA_FIELD_DEF(uint32_t, task_single_loop);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(DeformableConv2d, DeformableConv2dTilingData)
}