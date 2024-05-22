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
#include "functions.h"

namespace {
inline void npu_dynamic_scatter_check(const at::Tensor& feats, const at::Tensor& coors, int64_t reduce_type)
{
    TORCH_CHECK_NPU(feats);
    TORCH_CHECK_NPU(coors);
    TORCH_CHECK(
        reduce_type == 0 || reduce_type == 1 || reduce_type == 2, "reduce_type must be 0(sum) or 1(mean) or 2(max).");
    TORCH_CHECK(coors.size(1) == 3, "npu_dynamic_scatter only support coors.size(1) == 3.");
    TORCH_CHECK(feats.size(0) == coors.size(0), "npu_dynamic_scatter: feats.size(0) should equal coors.size(0).");
    TORCH_CHECK(feats.size(1) <= 2048, "npu_dynamic_scatter: feats.size(1) should less than or equal to 2048.");
}
} // namespace

static std::map<int64_t, std::string> REDUCE_TYPE_MAP = {{0, "sum"}, {1, "mean"}, {2, "max"}};


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_dynamic_scatter(
    at::Tensor& cof_tensor, at::Tensor& out_coors_unique2, at::Tensor& coors_map,
    at::Tensor& reduce_count, const at::Tensor& feats, const at::Tensor& coors, int64_t reduce_type)
{
    npu_dynamic_scatter_check(feats, coors, reduce_type);
    auto num_input = feats.size(0);
    auto num_feats = feats.size(1);
    if (num_input == 0) {
        return {feats.clone().detach(), coors.clone().detach(), coors.new_empty({0}, at::kInt),
            coors.new_empty({0}, at::kInt)};
    }

    c10::optional<c10::string_view> rounding_mode = "floor";
    std::vector<at::Tensor> out_coors_tensors;
    auto out_coors_0 = at::div(out_coors_unique2, cof_tensor[0], rounding_mode);
    out_coors_tensors.push_back(out_coors_0);
    out_coors_unique2 = at::sub(out_coors_unique2, at::mul(out_coors_0, cof_tensor[0]));
    auto out_coors_1 = at::div(out_coors_unique2, cof_tensor[1], rounding_mode);
    out_coors_tensors.push_back(out_coors_1);
    out_coors_unique2 = at::sub(out_coors_unique2, at::mul(out_coors_1, cof_tensor[1]));
    out_coors_tensors.push_back(out_coors_unique2);

    at::Tensor out_coors = at::stack(at::TensorList(out_coors_tensors), 1);
    out_coors = out_coors.to(coors.dtype());

    coors_map = coors_map.to(at::kInt);
    reduce_count = reduce_count.to(at::kInt);
    if (out_coors[0][0].lt(0).item<bool>()) {
        out_coors = out_coors.slice(0, 1);
        reduce_count = reduce_count.slice(0, 1);
        coors_map = coors_map - 1;
    }

    auto reduced_feats = at::zeros({out_coors.size(0), num_feats}, feats.options());
    const char* reduce_type_string = const_cast<char*>(REDUCE_TYPE_MAP[reduce_type] == "max" ? "max" : "sum");
    EXEC_NPU_CMD(aclnnDynamicScatter, feats, coors_map, reduce_type_string, reduced_feats);

    if (reduce_type == 1) {
        reduced_feats /= reduce_count.unsqueeze(-1).to(reduced_feats.dtype());
    }
    return {reduced_feats, out_coors, coors_map, reduce_count};
}

void npu_dynamic_scatter_grad(at::Tensor& grad_point_feats, const at::Tensor& grad_voxel_feats,
    const at::Tensor& prefix_sum_point_per_voxel, const at::Tensor& argsort_coor, const at::Tensor& compare_mask,
    const char* reduce_type)
{
    EXEC_NPU_CMD(aclnnDynamicScatterGrad, grad_voxel_feats, prefix_sum_point_per_voxel, argsort_coor, compare_mask,
        reduce_type, grad_point_feats);
}