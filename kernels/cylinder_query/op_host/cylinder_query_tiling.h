#ifndef CYLINDER_QUERY_TILING_H
#define CYLINDER_QUERY_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CylinderQueryTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, pointCloudSize);
TILING_DATA_FIELD_DEF(uint32_t, queryPointSize);
TILING_DATA_FIELD_DEF(uint32_t, nsample);
TILING_DATA_FIELD_DEF(float, radius);
TILING_DATA_FIELD_DEF(float, hmin);
TILING_DATA_FIELD_DEF(float, hmax);
TILING_DATA_FIELD_DEF(uint32_t, coreTask); // 每个核心的任务数
TILING_DATA_FIELD_DEF(uint32_t, bigCoreCount);
TILING_DATA_FIELD_DEF(uint32_t, tailTaskNum); // 尾核的任务数

TILING_DATA_FIELD_DEF(uint32_t, finalSmallTileNum); // 遍历点云过程中需要循环的次数
TILING_DATA_FIELD_DEF(uint32_t, tileDataNum); // 单次搬运点云数据中点的个数，8对齐，将96作为一个数据块，对应八个点(8 * 3 * 4)
TILING_DATA_FIELD_DEF(uint32_t, tileBlockNum); // 单次搬运中数据块个数(最大值)
TILING_DATA_FIELD_DEF(uint32_t, smallTileDataNum); // 最后一次搬运要处理的点云点个数
TILING_DATA_FIELD_DEF(uint32_t, smallTileBlockNum); // 最后一次搬运中数据块的个数

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CylinderQuery, CylinderQueryTilingData)
} // namespace optiling
#endif // CYLINDER_QUERY_TILING_H
