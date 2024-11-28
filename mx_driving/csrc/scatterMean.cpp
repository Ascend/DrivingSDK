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

static void npu_scatter_mean_shape_check(
    const at::Tensor& src, const at::Tensor& indices, const at::Tensor& out, int dim, int max_index)
{
    auto src_size = src.sizes();
    auto out_size = out.sizes();
    auto indices_size = indices.sizes();
    auto indices_dim = indices.dim();
    TORCH_CHECK(dim < indices_dim, "Dimension out of range, dim expected to be in range of [", -indices_dim, ", ",
        indices_dim - 1, "], but got ", dim);
    TORCH_CHECK(src.dim() == out.dim(), "out's dimension should be equal to src's dimension.");
    TORCH_CHECK(src.dim() >= indices.dim(), "indices's dimension should not larger than src's dimension.");
    // shape of out and src
    for (int i = 0; i < out.dim(); i++) {
        if (i != dim) {
            TORCH_CHECK(src_size[i] == out_size[i], "src and out should have the same size except for dim ", dim);
        }
    }

    uint32_t last_indices_dim = 0;
    if (indices_dim == 0) {
        return;
    }
    for (int i = indices.dim() - 1; i >= 0; i--) {
        if (indices_size[i] == 1) {
            last_indices_dim++;
        }
    }
    for (int i = 0; i < indices.dim() - last_indices_dim; i++) {
        TORCH_CHECK(src_size[i] == indices_size[i], "src and indices should have the same size at dim ", i);
    }
}

static int32_t get_available_dimnum(const at::Tensor& indices)
{
    auto indices_sizes = indices.sizes();
    int64_t indices_dim = indices.dim();
    int32_t last_indices_dim = 0;
    for (int64_t i = indices_dim - 1; i >= 0; i--) {
        if (indices_sizes[i] == 1) {
            last_indices_dim++;
        } else {
            break;
        }
    }
    return indices_dim - last_indices_dim;
}

std::tuple<at::Tensor, at::Tensor> npu_scatter_mean(at::Tensor& src, at::Tensor& indices, c10::optional<at::Tensor> out,
    c10::optional<int> dim, c10::optional<int> dim_size)
{
    TORCH_CHECK_NPU(src);
    TORCH_CHECK_NPU(indices);

    if (indices.numel() == 0 || src.numel() == 0) {
        std::tie(out.value(), out.value());
    }

    auto sizes = src.sizes().vec();
    auto true_dim = dim.value();
    int64_t true_dim_size;
    auto max_index = indices.max().item().toLong();
    if (dim_size.has_value()) {
        true_dim_size = dim_size.value();
    } else {
        true_dim_size = max_index + 1;
    }
    if (true_dim < 0) {
        true_dim = true_dim + indices.dim();
    }
    TORCH_CHECK(true_dim < src.dim(), "dim should not exceed the dimension of input src");
    sizes[true_dim] = true_dim_size;
    at::Tensor true_out = out.value_or(at::zeros(sizes, src.options().dtype(at::kFloat)));
    true_dim_size = true_out.sizes()[true_dim];

    npu_scatter_mean_shape_check(src, indices, true_out, true_dim, max_index);

    int32_t available_indices_dim = get_available_dimnum(indices);
    available_indices_dim = std::max(available_indices_dim, true_dim + 1);
    auto src_size = src.sizes();
    uint64_t tail = 1;
    for (uint64_t i = available_indices_dim; i < src.dim(); i++) {
        tail *= src_size[i];
    }

    if (tail == 1) {
        int32_t dim_input = available_indices_dim == 0 ? 0 : available_indices_dim - 1;
        src = src.transpose(true_dim, dim_input);
        indices = indices.transpose(true_dim, dim_input);
        at::Tensor out_trans = true_out.transpose(true_dim, dim_input).contiguous();
        at::Tensor count = at::zeros(out_trans.sizes(), src.options().dtype(at::kFloat));

        EXEC_NPU_CMD(aclnnScatterMean, src, indices, out_trans, dim_input, out_trans, count);
        count = at::where(count == 0, at::ones({}, count.options()), count);
        out_trans = out_trans / count;

        out_trans = out_trans.transpose(true_dim, dim_input).contiguous();
        count = count.transpose(true_dim, dim_input).contiguous();
        return std::tie(out_trans, count);
    } else {
        auto sizes_count = indices.sizes().vec();
        sizes_count[true_dim] = true_dim_size;
        at::Tensor count = at::zeros(sizes_count, src.options().dtype(at::kFloat));

        EXEC_NPU_CMD(aclnnScatterMean, src, indices, true_out, true_dim, true_out, count);
        EXEC_NPU_CMD(aclnnScatterMeanDiv, true_out, count, true_out);
        return std::tie(true_out, count);
    }
}
