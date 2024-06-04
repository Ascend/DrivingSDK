# Copyright (c) 2024 Huawei Technologies Co., Ltd
# Copyright (c) 2019, Facebook CORPORATION.
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Tuple

import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class ThreeInterpolateFunction(Function):

    @staticmethod
    def forward(ctx: Any, features: torch.Tensor, indices: torch.Tensor,
                weight: torch.Tensor) -> torch.Tensor:
        
        b, c, m = features.size()
        n = indices.size(1)
        ctx.three_interpolate_for_backward = (indices, weight, m)

        func = ads_c.npu_three_interpolate
        out = func(b, c, m, n, features, indices, weight)

        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        b, c, n = grad_out.size()
        idx, weight, m = ctx.three_interpolate_for_backward

        grad_out_dtype = grad_out.dtype
        grad_out_data = grad_out.data.contiguous().to(torch.float)
        weight = weight.to(torch.float)

        grad_features = ads_c.npu_three_interpolate_backward(b, c, n, m, grad_out_data, idx, weight)

        if grad_out_dtype == torch.half:
            grad_features = grad_features.to(torch.half)
        
        return grad_features, None, None

three_interpolate = ThreeInterpolateFunction.apply
