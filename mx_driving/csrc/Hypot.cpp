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

at::Tensor npu_hypot(const at::Tensor& x, const at::Tensor& y)
{
    auto out = at::empty_like(x, x.options());
    EXEC_NPU_CMD(aclnnHypot, x, y, out);
    return out;
}

std::tuple<at::Tensor, at::Tensor> npu_hypot_grad(
    const at::Tensor& x, const at::Tensor& y, const at::Tensor& out, const at::Tensor& out_grad)
{
    auto x_grad = at::empty_like(x, x.options());
    auto y_grad = at::empty_like(y, y.options());
    EXEC_NPU_CMD(aclnnHypotGrad, x, y, out, out_grad, x_grad, y_grad);
    return std::make_tuple(x_grad, y_grad);
}
