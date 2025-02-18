// Copyright (c) 2024 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "csrc/OpApiCommon.h"
#include "csrc/functions.h"

std::tuple<at::Tensor, at::Tensor> grid_sampler2d_v2_backward(const at::Tensor& grad_output,
    const at::Tensor& input_x, const at::Tensor& input_grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners)
{
    TORCH_CHECK_NPU(grad_output);
    TORCH_CHECK_NPU(input_x);
    TORCH_CHECK_NPU(input_grid);
    TORCH_CHECK(grad_output.dim() == 4, "input must to be a 4D Tensor, but got: ", grad_output.dim());
    TORCH_CHECK(input_x.dim() == 4, "offset has to be a 4D Tensor, but got: ", input_x.dim());
    TORCH_CHECK(input_grid.dim() == 4, "weight has to be a 4D Tensor, but got: ", input_grid.dim());

    auto input_x_sizes = input_x.sizes();
    auto input_grid_sizes = input_grid.sizes();
    at::Tensor grad_x = at::zeros(input_x_sizes, input_x.options());
    at::Tensor grad_grid = at::empty(input_grid_sizes, input_grid.options());

    EXEC_NPU_CMD(aclnnGridSampler2dV2Grad, grad_output, input_x, input_grid, interpolation_mode, padding_mode, align_corners,
        grad_x, grad_grid);
    grad_x = grad_x.permute({0, 3, 1, 2});
    return std::tie(grad_x, grad_grid);
}
