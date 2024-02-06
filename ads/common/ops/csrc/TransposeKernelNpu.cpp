// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/framework/OpCommand.h"
#include "functions.h"
#include "common.h"

namespace {
at::Tensor &npu_transpose_out_nocheck(
    at::Tensor &result,
    const at::Tensor &self,
    at::IntArrayRef perm,
    bool require_contiguous)
{
    at_npu::native::OpCommand cmd;
    if (require_contiguous) {
        // Any tensor-view(discontiguous) Input Tensor from users should be transformed to be contiguous here.
        cmd.Name("Transpose")
            .Input(self)
            .Input(perm)
            .Output(result)
            .Run();
    } else {
        // For permute-opt in trans-contiguous, it accepts transposed(discontiguous) Input Tensor.
        cmd.Name("Transpose")
            .InputWithoutContiguous(self)
            .Input(perm)
            .Output(result)
            .Run();
    }
    return result;
}
} // namespace

at::Tensor npu_transpose(const at::Tensor &self, at::IntArrayRef perm, bool require_contiguous)
{
    auto output_size = transpose_npu_output_size(self, perm);
    at::Tensor result = at::empty(output_size, self.options());
    npu_transpose_out_nocheck(result, self, perm, require_contiguous);

    return result;
}

at::Tensor &npu_transpose_out(
    const at::Tensor &self,
    at::IntArrayRef perm,
    bool require_contiguous,
    at::Tensor &result)
{
    if (!check_match(result)) {
        at::Tensor contiguous_result = result.contiguous();
        npu_transpose_out_nocheck(contiguous_result, self, perm, require_contiguous);
        format_fresh_view(result, contiguous_result);
    } else {
        npu_transpose_out_nocheck(result, self, perm, require_contiguous);
    }
    return result;
}
