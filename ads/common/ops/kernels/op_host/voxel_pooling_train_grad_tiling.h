/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef VOXEL_POOLING_TRAIN_GRAD_TILING_H
#define VOXEL_POOLING_TRAIN_GRAD_TILING_H
#include "register/tilingdata_base.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(VoxelPoolingTrainGradTilingData)

TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, numPoints);
TILING_DATA_FIELD_DEF(uint32_t, numChannels);
TILING_DATA_FIELD_DEF(uint32_t, h);
TILING_DATA_FIELD_DEF(uint32_t, w);
TILING_DATA_FIELD_DEF(uint32_t, numChannelsAligned);
TILING_DATA_FIELD_DEF(uint32_t, indicesAligned);
TILING_DATA_FIELD_DEF(uint32_t, average);
TILING_DATA_FIELD_DEF(uint32_t, taskLast);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);


END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(VoxelPoolingTrainGrad, VoxelPoolingTrainGradTilingData)
} // namespace optiling
#endif // VOXEL_POOLING_TRAIN_GRAD_TILING_H