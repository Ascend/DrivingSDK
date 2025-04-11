// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

#include "csrc/OpApiCommon.h"
#include "csrc/functions.h"

at::Tensor min_area_polygons(const at::Tensor& pointsets)
{
    int64_t N = pointsets.size(0);
    c10::SmallVector<int64_t, 8> polygons_size = {N, 8};
    at::Tensor polygons = at::zeros(polygons_size, pointsets.options());
    EXEC_NPU_CMD(aclnnMinAreaPolygons, pointsets, polygons);
    return polygons;
}