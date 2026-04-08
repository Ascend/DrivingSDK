#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RoiawareMaxpool3dGradTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalTask);
    TILING_DATA_FIELD_DEF(uint32_t, coreTask);
    TILING_DATA_FIELD_DEF(uint32_t, firstSmallCoreIdx);
    TILING_DATA_FIELD_DEF(uint32_t, singleLoopTask);
    TILING_DATA_FIELD_DEF(uint32_t, singleLoopOutput);
    TILING_DATA_FIELD_DEF(uint32_t, channelAligned);
    TILING_DATA_FIELD_DEF(uint32_t, channels);
    TILING_DATA_FIELD_DEF(uint32_t, npoints);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RoiawareMaxpool3dGrad, RoiawareMaxpool3dGradTilingData)
}
