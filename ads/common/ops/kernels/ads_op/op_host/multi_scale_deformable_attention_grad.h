/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
 * \file multi_scale_deformable_attention_grad.h
 * \brief
 */
 /*!
 * \file multi_scale_deformable_attention_grad.h
 * \brief tiling of MultiScaleDeformableAttentionGrad op
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_MUTISCALEDEFORMABLEATTENTIONGRAD_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_MUTISCALEDEFORMABLEATTENTIONGRAD_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MultiScaleDeformableAttentionGradTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batch_size)
    TILING_DATA_FIELD_DEF(uint32_t, spatial_size)
    TILING_DATA_FIELD_DEF(uint32_t, num_heads)
    TILING_DATA_FIELD_DEF(uint32_t, channels)
    TILING_DATA_FIELD_DEF(uint32_t, num_levels)
    TILING_DATA_FIELD_DEF(uint32_t, num_query)
    TILING_DATA_FIELD_DEF(uint32_t, num_point)
    TILING_DATA_FIELD_DEF(uint32_t, task_per_core)
    TILING_DATA_FIELD_DEF(uint32_t, task_tail_core)
    TILING_DATA_FIELD_DEF(uint32_t, core_used)
    TILING_DATA_FIELD_DEF(uint64_t, ub_size)
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(MultiScaleDeformableAttentionGrad, MultiScaleDeformableAttentionGradTilingData)
}
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_MUTISCALEDEFORMABLEATTENTIONGRAD_H
