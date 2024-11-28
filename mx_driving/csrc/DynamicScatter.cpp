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

namespace {
constexpr uint32_t BLOCK_NUM = 8;
constexpr float DEFAULT_VALUE = -1.0f;

inline void npu_dynamic_scatter_check(const at::Tensor& feats, const at::Tensor& coors)
{
    TORCH_CHECK_NPU(feats);
    TORCH_CHECK_NPU(coors);
    TORCH_CHECK(coors.size(1) == 3, "npu_dynamic_scatter only support coors.size(1) == 3.");
    TORCH_CHECK(feats.size(0) == coors.size(0), "npu_dynamic_scatter: feats.size(0) should equal coors.size(0).");
    TORCH_CHECK(feats.size(1) <= 2048, "npu_dynamic_scatter: feats.size(1) should less than or equal to 2048.");
}
} // namespace

std::tuple<at::Tensor, at::Tensor> npu_dynamic_scatter(const at::Tensor& feats, const at::Tensor& coors,
    const at::Tensor& prefix_sum_point_per_voxel, const at::Tensor& argsort_coor, int32_t num_voxels,
    const char* reduce_type)
{
    // Check inputs
    npu_dynamic_scatter_check(feats, coors);
    uint32_t point_num = feats.size(0);
    uint32_t feats_dim = feats.size(1);
    if (point_num == 0 || feats_dim == 0) {
        return std::make_tuple(feats.clone().detach(), coors.new_empty({0}, at::kByte));
    }

    // Do DynamicScatter
    uint32_t mask_dim = (feats_dim + BLOCK_NUM - 1) / BLOCK_NUM;
    at::Tensor voxel_feats = at::zeros({num_voxels, feats_dim}, feats.options());
    at::Tensor compare_mask = at::empty({point_num, mask_dim}, feats.options().dtype(at::kByte));
    EXEC_NPU_CMD_SYNC(
        aclnnDynamicScatter, feats, prefix_sum_point_per_voxel, argsort_coor, reduce_type, voxel_feats, compare_mask);

    return std::make_tuple(voxel_feats, compare_mask);
}

void npu_dynamic_scatter_grad(at::Tensor& grad_point_feats, const at::Tensor& grad_voxel_feats,
    const at::Tensor& prefix_sum_point_per_voxel, const at::Tensor& argsort_coor, const at::Tensor& compare_mask,
    const char* reduce_type)
{
    auto point_num = grad_point_feats.size(0);
    auto feats_dim = grad_point_feats.size(1);
    if (point_num > 0 && feats_dim > 0) {
        EXEC_NPU_CMD(aclnnDynamicScatterGrad, grad_voxel_feats, prefix_sum_point_per_voxel, argsort_coor, compare_mask,
            reduce_type, grad_point_feats);
    }
}
