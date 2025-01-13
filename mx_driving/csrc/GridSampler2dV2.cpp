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

#include <climits>

#include "csrc/OpApiCommon.h"
#include "csrc/functions.h"
#include "csrc/utils.h"

namespace {
constexpr int32_t B32_DATA_NUM_PER_BLOCK = 8;

void check_npu(const at::Tensor& input, const at::Tensor& grid)
{
    TORCH_CHECK_NPU(input);
    TORCH_CHECK_NPU(grid);
}
} // namespace

at::Tensor grid_sampler2d_v2(const at::Tensor& input, const at::Tensor& grid, int64_t interpolation_mode,
    int64_t padding_mode, bool align_corners)
{
    check_npu(input, grid);

    TORCH_CHECK(input.layout() == at::kStrided && grid.layout() == at::kStrided,
        "grid_sampler2d_v2(): expected input and grid to have torch.strided layout, but input has ", input.layout(),
        " and grid has ", grid.layout());
    TORCH_CHECK(input.scalar_type() == at::kFloat && grid.scalar_type() == at::kFloat,
        "grid_sampler2d_v2(): float32 tensor expected, but got input tensor with dtype: ", input.scalar_type(),
        "and grid tensor with dtype: ", grid.scalar_type());
    TORCH_CHECK(input.size(0) == grid.size(0),
        "grid_sampler2d_v2(): input and grid must have same batch size, but got input with sizes ", input.sizes(),
        " and grid with sizes ", grid.sizes());
    TORCH_CHECK(input.dim() == 4 && grid.dim() == 4,
        "grid_sampler2d_v2(): input and grid must be 4D tensor, but got input with sizes ", input.sizes(),
        " and grid with sizes ", grid.sizes());
    TORCH_CHECK(grid.size(-1) == 2,
        "grid_sampler2d_v2(): grid must have size 2 in last dimension, but got grid with sizes ", grid.sizes());

    int64_t n = input.size(0);
    int64_t c = input.size(1);
    int64_t h_in = input.size(2);
    int64_t w_in = input.size(3);
    int64_t c_out = AlignUp(c, B32_DATA_NUM_PER_BLOCK);
    int64_t h_out = grid.size(1);
    int64_t w_out = grid.size(2);
    TORCH_CHECK(n * c * h_in * w_in <= INT_MAX,
        "grid_sampler2d_v2(): not support for N*C*H*W of input greater than int32 max value.");
    TORCH_CHECK(n * 2 * h_out * w_out <= INT_MAX,
        "grid_sampler2d_v2(): not support for N*C*H*W of grid greater than int32 max value.");
    TORCH_CHECK(n * c * h_out * w_out <= INT_MAX,
        "grid_sampler2d_v2(): not support for N*C*H*W of output greater than int32 max value.");
    at::Tensor output_trans = at::empty({n, h_out, w_out, c_out}, input.options());

    // NCHW -> NHWC
    at::Tensor input_trans = input.permute({0, 2, 3, 1}).contiguous();
    EXEC_NPU_CMD(
        aclnnGridSampler2dV2, input_trans, grid, interpolation_mode, padding_mode, align_corners, output_trans);

    // NHWC -> NCHW
    at::Tensor output = output_trans.permute({0, 3, 1, 2}).contiguous();
    output = output.slice(1, 0, c);
    return output;
}