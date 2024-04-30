#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterMeanGradTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, size);
    TILING_DATA_FIELD_DEF(uint32_t, paramsPre);
    TILING_DATA_FIELD_DEF(uint32_t, dimRange);
    TILING_DATA_FIELD_DEF(uint32_t, dimRangeOut);
    TILING_DATA_FIELD_DEF(uint32_t, paramsPro);
    TILING_DATA_FIELD_DEF(int32_t, dim);

    TILING_DATA_FIELD_DEF(uint32_t, taskPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, taskTailCore);

    TILING_DATA_FIELD_DEF(uint64_t, ubSize);
    TILING_DATA_FIELD_DEF(uint32_t, gradInUbSize);
    TILING_DATA_FIELD_DEF(uint32_t, indexUbSize);
    TILING_DATA_FIELD_DEF(uint32_t, gradOutUbSize);
    TILING_DATA_FIELD_DEF(uint32_t, indexSumUbSize);

    TILING_DATA_FIELD_DEF(uint32_t, gradInNum);
    TILING_DATA_FIELD_DEF(uint32_t, indexNum);
    TILING_DATA_FIELD_DEF(uint32_t, gradOutNum);

    TILING_DATA_FIELD_DEF(uint32_t, paramsSliceLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterMeanGrad, ScatterMeanGradTilingData)
}
