// Copyright (c) 2024 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License");
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

constexpr int32_t MAX_OBJS = 500;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_gaussian(const at::Tensor& boxes,
    int32_t out_size_factor, float overlap, int32_t min_radius, float size_x, float size_y, float range_x,
    float range_y, int32_t feature_map_size_x, int32_t feature_map_size_y, bool norm_bbox, bool with_velocity)
{
    TORCH_CHECK_NPU(boxes);
    TORCH_CHECK(boxes.dim() == 2, "boxes.dim() must be 2, but got: ", boxes.dim());

    auto num_objs = boxes.size(0);
    num_objs = std::min(static_cast<int32_t>(num_objs), MAX_OBJS);
    c10::SmallVector<int64_t, 8> num_size = {MAX_OBJS};
    c10::SmallVector<int64_t, 8> center_int_size = {2, num_objs};
    c10::SmallVector<int64_t, 8> sub_xy_size = {2, MAX_OBJS};
    c10::SmallVector<int64_t, 8> box_dim_size = {3, num_objs};

    at::Tensor boxes_trans = boxes.permute({1, 0}).contiguous();
    at::Tensor center_int_trans = at::empty(center_int_size, boxes.options().dtype(at::kInt));
    at::Tensor radius = at::empty({num_objs}, boxes.options().dtype(at::kInt));
    at::Tensor mask = at::zeros(num_size, boxes.options().dtype(at::kByte));
    at::Tensor ind = at::zeros(num_size, boxes.options().dtype(at::kInt));
    at::Tensor sub_xy = at::zeros(sub_xy_size, boxes.options());
    at::Tensor box_dim = at::zeros(box_dim_size, boxes.options());
    at::Tensor sin_rot = at::zeros(num_size, boxes.options());
    at::Tensor cos_rot = at::zeros(num_size, boxes.options());
    at::Tensor anno_box_trans = at::empty({0}, boxes.options());

    double gaussian_overlap = overlap;
    double voxel_size_x = size_x;
    double voxel_size_y = size_y;
    double pc_range_x = range_x;
    double pc_range_y = range_y;

    EXEC_NPU_CMD(aclnnGaussian, boxes_trans, out_size_factor, gaussian_overlap, min_radius, voxel_size_x, voxel_size_y,
        pc_range_x, pc_range_y, feature_map_size_x, feature_map_size_y, norm_bbox, with_velocity, center_int_trans,
        radius, mask, ind, sub_xy, box_dim, sin_rot, cos_rot);

    ind = ind.to(at::kLong);
    at::Tensor z = at::zeros(num_size, boxes.options());
    at::Tensor vx = at::zeros(num_size, boxes.options());
    at::Tensor vy = at::zeros(num_size, boxes.options());
    at::Tensor log_box_dim = at::zeros({3, MAX_OBJS}, boxes.options());
    z.slice(0, 0, num_objs) = boxes_trans[2];
    vx.slice(0, 0, num_objs) = boxes_trans[7];
    vy.slice(0, 0, num_objs) = boxes_trans[8];
    log_box_dim.slice(1, 0, num_objs) = box_dim;

    if (with_velocity) {
        anno_box_trans = at::cat({sub_xy, z.unsqueeze(0), log_box_dim, sin_rot.unsqueeze(0), cos_rot.unsqueeze(0),
                                     vx.unsqueeze(0), vy.unsqueeze(0)},
            0);
    } else {
        anno_box_trans = at::cat({sub_xy, z.unsqueeze(0), log_box_dim, sin_rot.unsqueeze(0), cos_rot.unsqueeze(0)}, 0);
    }
    at::Tensor center_int = center_int_trans.permute({1, 0}).contiguous();
    at::Tensor anno_box = anno_box_trans.permute({1, 0}).contiguous();
    return std::tie(center_int, radius, mask, ind, anno_box);
}