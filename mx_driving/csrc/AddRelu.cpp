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

at::Tensor npu_add_relu(at::Tensor& x, const at::Tensor& y)
{
    EXEC_NPU_CMD(aclnnAddRelu, x, y);
    return x;
}

at::Tensor npu_add_relu_grad(at::Tensor& self, at::Tensor& grad_output)
{
    auto result = at::empty_like(self, self.options());
    at_npu::native::OpCommand cmd;
    cmd.Name("ReluGrad").Input(grad_output).Input(self).Output(result).Run();
    return result;
}
