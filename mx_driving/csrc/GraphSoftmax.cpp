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

at::Tensor graph_softmax(const at::Tensor& src, const at::Tensor& index, int N)
{
    TORCH_CHECK_NPU(src);
    TORCH_CHECK_NPU(index);
    TORCH_CHECK(src.dim() == 2, "src must be a 2D Tensor, but got: ", src.dim());
    TORCH_CHECK(index.dim() == 1, "index must be a 1D Tensor, but got: ", index.dim());
    TORCH_CHECK(index.sizes()[0] == src.sizes()[0], "The first dimension of index and src must be of equal size.");
    TORCH_CHECK(src[0].sizes() == 8, "The second dimension of src must be 8, but got: ", src[0].sizes());

    at::Tensor softmax_result = at::zeros_like(src);

    EXEC_NPU_CMD(aclnnGraphSoftmax, src, index, N, softmax_result);
    
    return softmax_result;
}