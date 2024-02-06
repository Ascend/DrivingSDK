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

#include <ATen/ATen.h>
#include <torch/csrc/autograd/custom_function.h>
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "functions.h"
#include "common.h"
#include "OpApiCommon.h"

using npu_preparation = at_npu::native::OpPreparation;
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_tuple = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

namespace {
inline void npu_moe_tutel_check(
    const at::Tensor& self,
    const at::Tensor& gates,
    const at::Tensor& indices,
    const at::Tensor& locations)
{
    TORCH_CHECK(self.dim() == 2, "The dim of input tensor [x] should equal to (sample, hidden).");
    TORCH_CHECK(gates.dim() == 2, "The dim of gates tensor [x] should equal to (batch, sample).");
    TORCH_CHECK(self.sizes()[0] == gates.sizes()[1], "input's sample size should equal to gates's samples size.");
    TORCH_CHECK((gates.sizes() == indices.sizes()) && (indices.sizes() == locations.sizes()),
        "Shape of gates should match shape of indices and locations.");
}
} // namespace

at::Tensor npu_moe_tutel(
    const at::Tensor& self,
    const at::Tensor& gates,
    const at::Tensor& indices,
    const at::Tensor& locations,
    int64_t capacity)
{
    npu_moe_tutel_check(self, gates, indices, locations);
    auto gates_size = gates.sizes();
    auto self_size = self.sizes();
    auto output_size = {gates_size[0], capacity, self_size[1]};
    at::Tensor result = at::zeros(output_size, self.options());
    EXEC_NPU_CMD(aclnnMoeTutelDispatch, self, gates, indices, locations, capacity, result);
    return result;
}

at::Tensor npu_moe_tutel_data_backward(
    const at::Tensor& y_grad,
    const at::Tensor& gates,
    const at::Tensor& indices,
    const at::Tensor& locations)
{
    auto gates_size = gates.sizes();
    auto grad_size = y_grad.sizes();
    auto output_size = {gates_size[1], grad_size[2]};
    at::Tensor result = at::zeros(output_size, y_grad.options());
    EXEC_NPU_CMD(aclnnMoeTutelCombineX, y_grad, gates, indices, locations, result);
    return result;
}

at::Tensor npu_moe_tutel_gate_backward(
    const at::Tensor& self,
    const at::Tensor& y_grad,
    const at::Tensor& indices,
    const at::Tensor& locations)
{
    at::Tensor result = at::zeros(indices.sizes(), y_grad.options());
    EXEC_NPU_CMD(aclnnMoeTutelCombineGates, self, y_grad, indices, locations, result);
    return result;
}
