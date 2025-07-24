#ifndef TIK_TOOLS_TILING_H
#define TIK_TOOLS_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GraphSoftmaxTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, coreTask);
    TILING_DATA_FIELD_DEF(uint32_t, coreWorkspace);
    TILING_DATA_FIELD_DEF(uint32_t, totalTask);
    TILING_DATA_FIELD_DEF(uint32_t, totalWorkspace);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreCount);
    TILING_DATA_FIELD_DEF(uint32_t, singleLoopTaskCount);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GraphSoftmax, GraphSoftmaxTilingData)
}
#endif