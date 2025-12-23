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


void scatter_max_validate(const at::Tensor& src, const at::Tensor& index, const at::Tensor& res)
{
    auto indexSizes = index.sizes();
    auto srcSizes = src.sizes();
    auto resSizes = res.sizes();
    int32_t indexLength = 1;
    for (size_t i = 1; i < static_cast<size_t>(index.dim()); i++) {
        indexLength *= indexSizes[i];
    }
    auto src_dims = srcSizes.size();
    auto index_dims = indexSizes.size();
    auto res_dims = resSizes.size();
    TORCH_CHECK(src_dims != 0 && index_dims != 0, "src and index should not be empty.");
    TORCH_CHECK(res_dims == src_dims, "out's dimension should be equal to src's dimension.");
    for (size_t i = 1; i < static_cast<size_t>(res.dim()); i++) {
        TORCH_CHECK(srcSizes[i] == resSizes[i], "src and out should have the same size except for dim 0.");
    }
    TORCH_CHECK(indexLength == 1,
        "all the dims's range except the first dim of input tensor [index] should be equal to 1.");
    TORCH_CHECK(
        index.sizes()[0] == src.sizes()[0], "input's src size of dim 0 should be equal to index's size.");
}

std::tuple<at::Tensor, at::Tensor> scatter_max_v3(
    const at::Tensor& src, const at::Tensor& index, c10::optional<at::Tensor> out)
{
    auto sizes = src.sizes().vec();
    auto idxMaxVal = index.max().item().toLong();
    TORCH_CHECK(idxMaxVal >= 0, "invalid index value.");

    sizes[0] = idxMaxVal + 1;
    float ninf = -std::numeric_limits<float>::infinity();
    at::Tensor res = out.value_or(at::empty(sizes, src.options().dtype(at::kFloat)).fill_(ninf));
    at::Tensor argmax = at::empty(res.sizes(), res.options().dtype(at::kInt)).fill_(-1);
    scatter_max_validate(src, index, res);

    EXEC_NPU_CMD(aclnnScatterMaxV3, src, index, res, argmax);
    res.masked_fill_(res == ninf, 0.0f);

    EXEC_NPU_CMD(aclnnScatterMaxArgmaxV3, src, index, res, argmax);
    auto argmaxInvalidVal = src.sizes().vec()[0];
    argmax.masked_fill_(argmax == -1, argmaxInvalidVal);

    return std::tie(res, argmax);
}
