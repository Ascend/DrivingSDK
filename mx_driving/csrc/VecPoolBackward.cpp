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

at::Tensor vec_pool_backward(const at::Tensor& grad_new_features, const at::Tensor& point_cnt_of_grid,
    const at::Tensor& grouped_idxs, const int64_t n, const int64_t num_c_in)
{
    TORCH_CHECK(grad_new_features.dim() == 2, "grad_new_features.dim() must be 2, but got: ", grad_new_features.dim());
    TORCH_CHECK(point_cnt_of_grid.dim() == 2, "point_cnt_of_grid.dim() must be 2, but got: ", point_cnt_of_grid.dim());
    TORCH_CHECK(grouped_idxs.dim() == 2, "grouped_idxs.dim() must be 2, but got: ", grouped_idxs.dim());
    TORCH_CHECK(point_cnt_of_grid.size(1) != 0, "numTotalGrids can not be 0.");
    TORCH_CHECK(grouped_idxs.size(1) == 3, "grouped_idxs.shape[1] must be 3, but got: ", grouped_idxs.size(1));
    auto output_size = {n, num_c_in};
    at::Tensor out = at::zeros(output_size, grad_new_features.options());
    EXEC_NPU_CMD(aclnnVecPoolGrad, grad_new_features, point_cnt_of_grid, grouped_idxs, n, num_c_in, out);
    return out;
}
