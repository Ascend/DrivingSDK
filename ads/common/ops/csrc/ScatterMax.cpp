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

using namespace std;

namespace {
constexpr uint32_t MAX_INDICES_VALUE = 120000;
constexpr uint32_t SUPPORT_UPDATES = 32;
constexpr uint32_t MAX_SUPPORT_UPDATES = 512;
}

void npu_scatter_max_check(const at::Tensor& updates, const at::Tensor& indices)
{
    auto indicesSizes = indices.sizes();
    auto updatesSizes = updates.sizes();
    int32_t indicesLength = 1;
    int32_t updatesLength = 1;
    for (size_t i = 1; i < updates.dim(); i++) {
        updatesLength *= updatesSizes[i];
    }
    for (size_t i = 1; i < indices.dim(); i++) {
        indicesLength *= indicesSizes[i];
    }
    auto updates_dims = updatesSizes.size();
    auto index_dims = indicesSizes.size();
    TORCH_CHECK(updatesLength % 8 == 0, "The shape of updates after combine the axis should be aligned by 32 Bytes.");
    TORCH_CHECK(updates_dims != 0 && index_dims != 0, "updates and index should not be empty");
    TORCH_CHECK(indicesLength == 1, "All the dims's range except the first dim of input tensor [indices] should be equal to 1.");
    TORCH_CHECK(indices.sizes()[0] == updates.sizes()[0], "input's updates size of dim 0 should be equal to indices's size.");
}

bool npu_scatter_max_v2_support(const at::Tensor& updates, at::Tensor& var)
{
    auto varSizes = var.sizes();
    auto updatesSizes = updates.sizes();
    int32_t indicesLength = 1;
    int32_t updatesLength = 1;
    if (updates.dim() == 1 || updates.dim() >= 3) {
        return false;
    }
    for (size_t i = 1; i < updates.dim(); i++) {
        updatesLength *= updatesSizes[i];
    }
    if ((varSizes[0] > MAX_INDICES_VALUE && updatesLength < SUPPORT_UPDATES) ||
        updatesLength >= MAX_SUPPORT_UPDATES) {
        return false;
    }
    return true;
}

std::tuple<at::Tensor, at::Tensor> scatter_max_with_argmax_v2(const at::Tensor& updates, const at::Tensor& indices,
                                                              c10::optional<at::Tensor> out)
{
    npu_scatter_max_check(updates, indices);
    auto sizes = updates.sizes().vec();
    auto indicesMax = indices.max().item().toLong();
    TORCH_CHECK(indicesMax > 0, "the value of indices is not a valid index.");
    sizes[0] = indicesMax + 1;
    at::Tensor result = out.value_or(at::zeros(sizes, updates.options().dtype(at::kFloat)));
    auto argmax_init = updates.sizes().vec()[0];
    at::Tensor argmax = at::zeros(sizes, result.options().dtype(at::kInt)).fill_(argmax_init);
    at::Tensor var = out.value_or(at::zeros(sizes, updates.options().dtype(at::kFloat)).fill_(-3.4e+38));
    if (npu_scatter_max_v2_support(updates, var)) {
        EXEC_NPU_CMD(aclnnScatterMaxWithArgmaxV2, var, indices, updates, result, argmax);
    } else {
        at_npu::native::OpCommand cmd;
        cmd.Name("ScatterMaxWithArgmax").Input(result).Input(indices).Input(updates).Output(result).Output(argmax).Run();
    }
    return std::tie(result, argmax);
}

at::Tensor npu_scatter_max_backward(const at::Tensor& x, const at::Tensor& segment_ids,
                                    const at::Tensor& num_segments)
{
    c10::SmallVector<int64_t, SIZE> output_size;

    auto num_segments_value = num_segments.item().toLong();
    output_size.push_back(num_segments_value);

    auto x_sizes = x.sizes();
    auto segment_ids_dims = segment_ids.dim();

    copy(x_sizes.begin() + segment_ids_dims, x_sizes.end(), std::back_inserter(output_size));

    at::Tensor out = at::empty(output_size, x.options());
    at_npu::native::OpCommand cmd;
    cmd.Name("UnsortedSegmentSum").Input(x).Input(segment_ids).Input(num_segments).Output(out).Attr("check_ids", true).Run();
    return out;
}