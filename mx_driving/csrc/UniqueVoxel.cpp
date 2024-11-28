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

std::tuple<int32_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor> unique_voxel(const at::Tensor& voxels)
{
    TORCH_CHECK_NPU(voxels);
    TORCH_CHECK(voxels.dim() == 1, "voxels.dim() must be 1, but got: ", voxels.dim());
    TORCH_CHECK(voxels.dtype() == at::kFloat || voxels.dtype() == at::kInt,
        "voxels.dtype() must be float or int32, but got: ", voxels.dtype());

    size_t num_points = voxels.size(0);

    auto sorted = voxels.dtype() == at::kFloat ? at::sort(voxels) : at::sort(voxels.view(at::kFloat));
    at::Tensor sorted_voxels = std::get<0>(sorted);
    at::Tensor argsort_indices = std::get<1>(sorted).to(at::kInt);

    at::Scalar start = at::Scalar(1);
    at::Scalar end = at::Scalar(static_cast<int>(num_points + 1));
    at::Scalar step = at::Scalar(1);
    at::Tensor indices = at::range(start, end, step, voxels.options().dtype(at::kInt));

    at::Tensor uni_voxels = at::empty({num_points + 1}, voxels.options().dtype(at::kInt));
    at::Tensor uni_indices = at::empty({num_points + 1}, voxels.options().dtype(at::kInt));
    at::Tensor uni_argsort_indices = at::empty({num_points + 1}, voxels.options().dtype(at::kFloat));
    at::Tensor num_voxels = at::empty({1}, voxels.options().dtype(at::kInt));
    EXEC_NPU_CMD_SYNC(aclnnUniqueVoxel, sorted_voxels, indices, argsort_indices, uni_voxels, uni_indices,
        uni_argsort_indices, num_voxels);

    int32_t num_voxels_ = num_voxels.item().toInt();
    uni_voxels = uni_voxels.slice(0, 0, num_voxels_);
    uni_indices = uni_indices.slice(0, 0, num_voxels_);
    uni_argsort_indices = uni_argsort_indices.slice(0, 0, num_voxels_);
    return std::make_tuple(num_voxels_, uni_voxels, uni_indices, argsort_indices, uni_argsort_indices);
}
