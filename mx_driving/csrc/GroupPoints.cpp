// Copyright (c) OpenMMLab. All rights reserved.
// Copyright (c) 2024 Huawei Technologies Co., Ltd
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

at::Tensor group_points(
    const at::Tensor& points, const at::Tensor& idx, int64_t b, int64_t c, int64_t n, int64_t npoints, int64_t nsample)
{
    TORCH_CHECK_NPU(points);
    TORCH_CHECK_NPU(idx);
    TORCH_CHECK(points.scalar_type() == at::kHalf || points.scalar_type() == at::kFloat,
        "group_points only support float16 or float32 tensor.")
    TORCH_CHECK(points.dim() == 3, "points.dim() must be 3, but got: ", points.dim());
    TORCH_CHECK(idx.dim() == 3, "idx.dim() must be 3, but got: ", idx.dim());
    TORCH_CHECK(points.size(0) == idx.size(0), "the input first dimension must be the same.")

    at::Tensor trans_features = points.transpose(1, 2);
    at::Tensor features = trans_features.contiguous();
    at::Tensor out = at::empty({b, c, npoints, nsample}, points.options());

    EXEC_NPU_CMD(aclnnGroupPoints, features, idx, b, c, n, npoints, nsample, out);

    at::Tensor output = out.view({b, npoints, nsample, c}).permute({0, 3, 1, 2});
    return output;
}


at::Tensor group_points_backward(const at::Tensor& grad_out, const at::Tensor& idx, int64_t b, int64_t c, int64_t n,
    int64_t npoints, int64_t nsample)
{
    TORCH_CHECK_NPU(grad_out);
    TORCH_CHECK_NPU(idx);
    TORCH_CHECK(grad_out.dim() == 4, "grad_out.dim() must be 4, but got: ", grad_out.dim());
    TORCH_CHECK(idx.dim() == 3, "idx.dim() must be 3, but got: ", idx.dim());

    at::Tensor trans_idx = idx.view({b * npoints * nsample});
    at::Tensor trans_grad_out = grad_out.permute({0, 2, 3, 1});
    at::Tensor grad_out_tensor = trans_grad_out.contiguous();
    grad_out_tensor = grad_out_tensor.view({b * npoints * nsample, c});
    at::Tensor out = at::zeros({b, n, c}, grad_out.options());

    EXEC_NPU_CMD(aclnnGroupPointsGrad, grad_out_tensor, trans_idx, b, c, n, npoints, nsample, out);

    at::Tensor grad_points = out.transpose(1, 2);
    return grad_points;
}
