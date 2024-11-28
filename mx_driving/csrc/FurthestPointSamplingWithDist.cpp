// Copyright (c) 2023-2024 Huawei Technologies Co., Ltd
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

at::Tensor furthest_point_sampling_with_dist(
    const at::Tensor& points_dist, const at::Tensor& nearest_temp, int32_t num_points)
{
    auto points_dist_size = points_dist.sizes();
    int64_t b = points_dist_size[0];
    int64_t num_points_real = num_points;
    auto output_size = {b, num_points_real};
    at::Tensor result = at::empty(output_size, points_dist.options().dtype(at::kInt));
    EXEC_NPU_CMD(aclnnFurthestPointSamplingWithDist, points_dist, nearest_temp, num_points, result);
    return result;
}
