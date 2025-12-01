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

using namespace std;

constexpr uint32_t ALIGN_NUM = 8;
constexpr uint32_t NUM_COORDINATES = 9; // max smaller than 9D x/y
constexpr uint32_t NUM_DIM = 2; // Two-dimensional coordinates

std::tuple<at::Tensor, at::Tensor> radius(at::Tensor& x, at::Tensor& y, at::Tensor& ptr_x, at::Tensor& ptr_y,
                                          double r, int max_num_neighbors)
{
    TORCH_CHECK_NPU(x);
    TORCH_CHECK_NPU(y);
    TORCH_CHECK_NPU(ptr_x);
    TORCH_CHECK_NPU(ptr_y);
    TORCH_CHECK(x.size(1) < NUM_COORDINATES, "x must be a coordinate which is smaller than 9D, but got: ", x.size(1));
    TORCH_CHECK(y.size(1) < NUM_COORDINATES, "y must be a coordinate which is smaller than 9D, but got: ", y.size(1));
    TORCH_CHECK(x.dim() == NUM_DIM, "x must be a 2D Tensor, but got: ", x.dim());
    TORCH_CHECK(y.dim() == NUM_DIM, "y must be a 2D Tensor, but got: ", y.dim());
    
    auto x_shape = x.sizes(); // [num_points_x, Ndim]
    auto y_shape = y.sizes(); // [num_points_y, Ndim]
    auto ptr_x_shape = ptr_x.sizes(); // [batch_size + 1]
    auto x_trans = x.transpose(0, 1).contiguous(); // [Ndim, num_points_x]
    auto y_trans = y.transpose(0, 1).contiguous(); // [Ndim, num_points_y]
    int out_dim = y_shape[0] * max_num_neighbors;
    int batch_size = ptr_x_shape[0] - 1;
    auto out_temp = at::zeros({NUM_DIM, out_dim}, ptr_x.options().dtype(at::kInt)); // [2, num_points_y * max_num_neighbors], used to hold intermediate results for non-continuous distributions.
    auto out_final = at::zeros({NUM_DIM, out_dim}, ptr_x.options().dtype(at::kInt)); // [2, num_points_y * max_num_neighbors], used to store the final results of continuous distributions.
    auto actual_num_neighbors = at::zeros({ALIGN_NUM}, ptr_x.options().dtype(at::kInt));
    EXEC_NPU_CMD(aclnnRadius, x_trans, y_trans, ptr_x, ptr_y, r,  max_num_neighbors, out_temp, out_final, actual_num_neighbors);

    return std::tie(out_final, actual_num_neighbors);
}
