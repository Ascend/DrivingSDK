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

using tensor_tuple = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

namespace {
at::Tensor &rotary_mul_nocheck(at::Tensor &y, const at::Tensor &x, const at::Tensor &r1, const at::Tensor &r2)
{
    if (x.sizes()[3] % 64 != 0) {
        std::vector<at::Tensor> chunkResult = x.chunk(2, -1);
        at::Tensor x_new = at::cat({chunkResult[1] * (-1), chunkResult[0]}, 3);
        y = at::mul(r1, x) + at::mul(r2, x_new);
    } else {
        at_npu::native::OpCommand cmd;
        cmd.Name("RotaryMul").Input(x).Input(r1).Input(r2).Output(y).Run();
    }
    return y;
}

tensor_tuple rotary_mul_backward_nocheck(at::Tensor &dx, at::Tensor &dr1, at::Tensor &dr2, const at::Tensor &x,
    const at::Tensor &r1, const at::Tensor &r2, const at::Tensor &dy)
{
    TORCH_CHECK(x.dim() == 4, "The dim of input tensor [x] shoule equal to four.");
    TORCH_CHECK(r1.dim() == 4, "The dim of input tensor [r1] shoule equal to four.");
    TORCH_CHECK(r2.dim() == 4, "The dim of input tensor [r2] shoule equal to four.");
    if (x.sizes()[3] % 64 != 0) {
        at::Tensor x_grad_mul = at::mul(x, dy);
        at::Tensor x1_grad_mul = at::mul(r1, dy);
        at::Tensor x2_grad_mul = at::mul(r2, dy);
        std::vector<at::Tensor> x2_chunk = x2_grad_mul.chunk(2, -1);
        at::Tensor x2_chunk_cat = at::cat({x2_chunk[1], x2_chunk[0] * (-1)}, 3);
        dx = at::add(x2_chunk_cat, x1_grad_mul);
        c10::SmallVector<int64_t, SIZE> dims;
        for (int i = 0; i < 4; i++) {
            if (x.sizes()[i] != r1.sizes()[i]) {
                dims.emplace_back(i);
            }
        }
        std::vector<at::Tensor> xq_chunk = x_grad_mul.chunk(2, -1);
        at::Tensor xq_chunk_cat = at::cat({xq_chunk[1] * (-1), xq_chunk[0]}, 3);
        dr2 = at::sum(xq_chunk_cat, dims, true);
        dr1 = at::sum(x_grad_mul, dims, true);
    } else {
        at_npu::native::OpCommand cmd;
        cmd.Name("RotaryMulGrad").Input(x).Input(r1).Input(r2).Input(dy).Output(dx).Output(dr1).Output(dr2).Run();
    }
    return std::tie(dx, dr1, dr2);
}
} // namespace

at::Tensor npu_rotary_mul(const at::Tensor &self, const at::Tensor &r1, const at::Tensor &r2)
{
    at::Tensor result = at::empty(self.sizes(), self.options());
    rotary_mul_nocheck(result, self, r1, r2);
    return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_rotary_mul_backward(const at::Tensor &grad, const at::Tensor &self,
                                                                       const at::Tensor &r1, const at::Tensor &r2)
{
    at::Tensor dx = at::empty(self.sizes(), self.options());
    at::Tensor dr1 = at::empty(r1.sizes(), r1.options());
    at::Tensor dr2 = at::empty(r2.sizes(), r2.options());
    rotary_mul_backward_nocheck(dx, dr1, dr2, self, r1, r2, grad);
    return std::tie(dx, dr1, dr2);
}
