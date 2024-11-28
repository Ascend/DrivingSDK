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

#include "csrc/OpApiCommon.h"
#include "csrc/functions.h"

/*
 * points: (B, N, 3)
 * point_features: (B, N, C)
 * boxes3d: (B, M, 7)
 * pooled_features: (B, M, num, 3+C)
 * pooled_empty_flag: (B, M)
*/
std::tuple<at::Tensor, at::Tensor> npu_roipoint_pool3d_forward(const int32_t num_sampled_points,
    const at::Tensor& points, const at::Tensor& point_features, const at::Tensor& boxes3d)
{
    auto points_trans = points.transpose(1, 2).contiguous();
    auto point_features_trans = point_features.transpose(1, 2).contiguous();
    c10::SmallVector<int64_t, SIZE> features_trans_size = {
        points.size(0), boxes3d.size(1), points.size(2) + point_features.size(2), num_sampled_points};
    at::Tensor pooled_features_trans = at::empty(features_trans_size, points.options());
    c10::SmallVector<int64_t, SIZE> empty_flag_size = {boxes3d.size(0), boxes3d.size(1)};
    at::Tensor pooled_empty_flag = at::empty(empty_flag_size, boxes3d.options().dtype(at::kInt));
    EXEC_NPU_CMD(aclnnRoipointPool3dForward, points_trans, point_features_trans, boxes3d, num_sampled_points,
        pooled_features_trans, pooled_empty_flag);
    auto pooled_features = pooled_features_trans.transpose(2, 3).contiguous();
    return std::tie(pooled_features, pooled_empty_flag);
}
