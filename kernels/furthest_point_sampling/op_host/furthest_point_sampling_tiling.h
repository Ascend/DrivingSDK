/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file furthest_point_sampling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_FURTHEST_POINT_SAMPLING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_FURTHEST_POINT_SAMPLING_H_
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FurthestPointSamplingTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, N);
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, numPoints);
    TILING_DATA_FIELD_DEF(uint32_t, pieces);
    TILING_DATA_FIELD_DEF(uint32_t, formerNum);
    TILING_DATA_FIELD_DEF(uint32_t, tailNum);
    TILING_DATA_FIELD_DEF(uint32_t, workSize);
    TILING_DATA_FIELD_DEF(uint32_t, idxTempSize);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreBatch);
    TILING_DATA_FIELD_DEF(uint32_t, smallCoreBatch);
    TILING_DATA_FIELD_DEF(uint32_t, bigCoreNum);
    TILING_DATA_FIELD_DEF(uint32_t, repeats);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FurthestPointSampling, FurthestPointSamplingTilingData)
}

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_FURTHEST_POINT_SAMPLING_H_