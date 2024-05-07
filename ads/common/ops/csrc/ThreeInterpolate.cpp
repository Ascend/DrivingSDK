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

at::Tensor npu_three_interpolate(int b, int c, int m, int n, const at::Tensor& points, const at::Tensor& idx, const at::Tensor& weight)
{
    auto originDtype = points.scalar_type();
    TORCH_CHECK((originDtype == at::kFloat || originDtype == at::kHalf),
                "three_interpolate_forward ascend only support fp32 and fp16.");

    auto point_c_trans = points.transpose(1, 2);

    c10::SmallVector<int64_t, 8> output_size = {b, c, n};
    at::Tensor out = at::zeros(output_size, points.options());

    at_npu::native::OpCommand cmd;
    cmd.Name("ThreeInterpolate")
        .Input(point_c_trans)
        .Input(idx)
        .Input(weight)
        .Output(out)
        .Run();
        
    auto output = out.view({b, n, c}).transpose(1, 2);
    auto res = output.contiguous();
    out.copy_(res);
    
    return out;
}

at::Tensor npu_three_interpolate_backward(int b, int c, int n, int m, const at::Tensor& grad_out, const at::Tensor& idx, const at::Tensor& weight)
{
    auto originDtype = grad_out.scalar_type();
    TORCH_CHECK((originDtype == at::kFloat || originDtype == at::kHalf),
                "three_interpolate_backward ascend only support fp32 and fp16.");

    at::Tensor grad_points = at::zeros({b, c, m}, grad_out.options());
    auto grad_x = at::unsqueeze(grad_out, 3);
    auto grad_y = at::unsqueeze(grad_points, 3);

    EXEC_NPU_CMD(aclnnThreeInterpolateBackward, grad_x, idx, weight, m, grad_y);

    auto output = at::squeeze(grad_y, 3);
    auto res = output.contiguous();
    grad_points.copy_(res);

    return grad_points;
}
