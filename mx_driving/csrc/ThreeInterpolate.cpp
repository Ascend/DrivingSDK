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

at::Tensor npu_three_interpolate(
    int b, int c, int m, int n, const at::Tensor& points, const at::Tensor& idx, const at::Tensor& weight)
{
    TORCH_CHECK_NPU(points);
    TORCH_CHECK_NPU(idx);
    TORCH_CHECK_NPU(weight);

    auto point_dtype = points.scalar_type();
    auto idx_dtype = idx.scalar_type();
    auto weight_dtype = weight.scalar_type();

    TORCH_CHECK((point_dtype == at::kFloat || point_dtype == at::kHalf),
        "three_interpolate_forward ascend only support fp32 and fp16.");
    TORCH_CHECK((weight_dtype == at::kFloat || weight_dtype == at::kHalf),
        "three_interpolate_forward ascend only support fp32 and fp16.");
    TORCH_CHECK((point_dtype == weight_dtype), "input dtype is inconsistent.");
    TORCH_CHECK((idx_dtype == at::kInt), "indices: int32 tensor expected but got a tensor with dtype: ", idx_dtype);

    auto point_size = points.sizes();
    auto idx_size = idx.sizes();
    auto weight_size = weight.sizes();

    TORCH_CHECK(
        (point_size.size() == 3 && idx_size.size() == 3 && weight_size.size() == 3), "input dimension should be 3.");
    TORCH_CHECK((point_size[0] == idx_size[0] && point_size[0] == weight_size[0] && idx_size[0] == weight_size[0]),
        "the first dimension of input should be the same.");
    TORCH_CHECK((idx_size[1] == weight_size[1]), "the second dimension of indices and weight should be the same.");
    TORCH_CHECK((idx_size[2] == 3 && weight_size[2] == 3), "the third dimension of indices and weight should be 3.");

    TORCH_CHECK((b < 10001 && c < 10001 && m < 10001 && n < 10001), "input dimension is too heavy.");

    auto point_c_trans = points.transpose(1, 2).to(at::kFloat);
    auto weight_cast = weight.to(at::kFloat);

    c10::SmallVector<int64_t, 8> output_size = {b, c, n};
    at::Tensor out_cast = at::zeros(output_size, points.options()).to(at::kFloat);

    at_npu::native::OpCommand cmd;
    cmd.Name("ThreeInterpolate").Input(point_c_trans).Input(idx).Input(weight_cast).Output(out_cast).Run();

    auto out = out_cast;
    if (point_dtype == at::kHalf) {
        out = out_cast.to(at::kHalf);
    }
    auto output = out_cast.view({b, n, c}).transpose(1, 2);
    auto res = output.contiguous();
    out.copy_(res);

    return out;
}

at::Tensor npu_three_interpolate_backward(
    int b, int c, int n, int m, const at::Tensor& grad_out, const at::Tensor& idx, const at::Tensor& weight)
{
    TORCH_CHECK_NPU(grad_out);
    TORCH_CHECK_NPU(idx);
    TORCH_CHECK_NPU(weight);

    auto grad_dtype = grad_out.scalar_type();
    auto idx_dtype = idx.scalar_type();
    auto weight_dtype = weight.scalar_type();

    TORCH_CHECK((grad_dtype == at::kFloat || grad_dtype == at::kHalf),
        "three_interpolate_forward ascend only support fp32 and fp16.");
    TORCH_CHECK((weight_dtype == at::kFloat || weight_dtype == at::kHalf),
        "three_interpolate_forward ascend only support fp32 and fp16.");
    TORCH_CHECK((grad_dtype == weight_dtype), "input dtype is inconsistent.");
    TORCH_CHECK((idx_dtype == at::kInt), "indices: int32 tensor expected but got a tensor with dtype: ", idx_dtype);

    auto grad_size = grad_out.sizes();
    auto idx_size = idx.sizes();
    auto weight_size = weight.sizes();

    TORCH_CHECK(
        (grad_size.size() == 3 && idx_size.size() == 3 && weight_size.size() == 3), "the input dimension should be 3.");
    TORCH_CHECK((grad_size[0] == idx_size[0] && grad_size[0] == weight_size[0] && idx_size[0] == weight_size[0]),
        "the first dimension of input should be the same.");
    TORCH_CHECK((grad_size[2] == idx_size[1] && grad_size[2] == weight_size[1] && idx_size[1] == weight_size[1]),
        "the second dimension of indices and weight should be the same.");
    TORCH_CHECK((idx_size[2] == 3 && weight_size[2] == 3), "the third dimension of indices and weight should be 3.");

    TORCH_CHECK((b < 10001 && c < 10001 && m < 10001 && n < 10001), "input dimension is too heavy.");

    at::Tensor grad_points = at::zeros({b, c, m}, grad_out.options());
    auto grad_x = at::unsqueeze(grad_out, 3);
    auto grad_y = at::unsqueeze(grad_points, 3);

    EXEC_NPU_CMD(aclnnThreeInterpolateBackward, grad_x, idx, weight, m, grad_y);

    auto output = at::squeeze(grad_y, 3);
    auto res = output.contiguous();
    grad_points.copy_(res);

    return grad_points;
}
