// Copyright (c) 2024 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef VEC_POOL_GRAD_TILING_H
#define VEC_POOL_GRAD_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(VecPoolGradTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, formerCoreGroups)
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
    TILING_DATA_FIELD_DEF(uint32_t, availableUbSize)
    TILING_DATA_FIELD_DEF(uint32_t, mainGroups)
    TILING_DATA_FIELD_DEF(uint32_t, copyLoop)
    TILING_DATA_FIELD_DEF(uint32_t, copyTail)
    TILING_DATA_FIELD_DEF(uint32_t, formerTailGroups)
    TILING_DATA_FIELD_DEF(uint32_t, lastCopyLoop)
    TILING_DATA_FIELD_DEF(uint32_t, lastCopyTail)
    TILING_DATA_FIELD_DEF(uint32_t, lastTailGroups)
    TILING_DATA_FIELD_DEF(uint32_t, m)
    TILING_DATA_FIELD_DEF(uint32_t, cOut)
    TILING_DATA_FIELD_DEF(uint32_t, numTotalGrids)
    TILING_DATA_FIELD_DEF(uint32_t, numCEachGrid)
    TILING_DATA_FIELD_DEF(uint32_t, gradUBEleNum)
    TILING_DATA_FIELD_DEF(uint32_t, numMaxSumPoints)
    TILING_DATA_FIELD_DEF(uint32_t, n)
    TILING_DATA_FIELD_DEF(uint32_t, cIn)
    TILING_DATA_FIELD_DEF(uint32_t, repeatTimes)
    TILING_DATA_FIELD_DEF(uint32_t, tail)
    TILING_DATA_FIELD_DEF(uint32_t, mainCopySize)
    TILING_DATA_FIELD_DEF(uint32_t, formerCoreTailCopySize)
    TILING_DATA_FIELD_DEF(uint32_t, lastCoreTailCopySize)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(VecPoolGrad, VecPoolGradTilingData)
} // namespace optiling

#endif // VEC_POOL_GRAD_TILING_H
