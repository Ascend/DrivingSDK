// Copyright (c) 2025 Huawei Technologies Co., Ltd
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

std::tuple<at::Tensor, at::Tensor> grid_sampler3d_grad_v1(const at::Tensor& grad,
    const at::Tensor& x, const at::Tensor& grid, int64_t interpolation, int64_t padding, bool align)
{
    TORCH_CHECK_NPU(grad);
    TORCH_CHECK_NPU(x);
    TORCH_CHECK_NPU(grid);
    TORCH_CHECK(grad.dim() == 5, "grad.dim() must be 5, but got: ", grad.dim());
    TORCH_CHECK(x.dim() == 5, "x.dim() must be 5, but got: ", x.dim());
    TORCH_CHECK(grid.dim() == 5, "grid.dim() must be 5, but got: ", grid.dim());
    TORCH_CHECK(grid.sizes()[4] == 3, "last dim of grid must be 3, but got: ", grid.sizes()[4]);

    at::Tensor gradTensor = grad.permute({0, 2, 3, 4, 1}).contiguous();
    at::Tensor xTensor = x.permute({0, 2, 3, 4, 1}).contiguous();

    at::Tensor dx = at::zeros(xTensor.sizes(), xTensor.options());
    at::Tensor dgrid = at::zeros(grid.sizes(), grid.options());

    EXEC_NPU_CMD(aclnnGridSampler3dGradV1, gradTensor, xTensor, grid, interpolation, padding, align, dx, dgrid);

    at::Tensor dxTensor = dx.permute({0, 4, 1, 2, 3});

    return std::tie(dxTensor, dgrid);
}