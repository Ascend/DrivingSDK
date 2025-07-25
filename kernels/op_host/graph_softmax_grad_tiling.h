#ifndef graph_softmax_grad_tiling_h
#define graph_softmax_grad_tiling_h
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GraphSoftmaxGradTilingData)
    TILING_DATA_FIELD_DEF(int32_t, edgeNum);
    TILING_DATA_FIELD_DEF(int32_t, alignTaskNum);
    TILING_DATA_FIELD_DEF(int32_t, tailNum);
    TILING_DATA_FIELD_DEF(int32_t, nodeNum);
    TILING_DATA_FIELD_DEF(int32_t, taskNumPerLoop);
    TILING_DATA_FIELD_DEF(int32_t, taskLoop);
    TILING_DATA_FIELD_DEF(int32_t, tailCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);
    TILING_DATA_FIELD_DEF(uint32_t, taskNumPerCore);
    TILING_DATA_FIELD_DEF(uint64_t, ubTotalSize);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GraphSoftmaxGrad, GraphSoftmaxGradTilingData)
}
#endif