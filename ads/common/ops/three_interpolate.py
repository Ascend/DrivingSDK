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
        """
        Args:
            features (torch.Tensor): (B, C, M) Features descriptors to be
                interpolated.
            indices (torch.Tensor): (B, n, 3) indices of three nearest
                neighbor features for the target features.
            weight (torch.Tensor): (B, n, 3) weights of three nearest
                neighbor features for the target features.

        Returns:
            torch.Tensor: (B, C, N) tensor of the interpolated features
        """
        B, c, m = features.size()
        n = indices.size(1)
        ctx.three_interpolate_for_backward = (indices, weight, m)

        func = ads_c.npu_three_interpolate
        out = func(B, c, m, n, features, indices, weight)

        return out

    @staticmethod
    def backward(
        ctx, grad_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            grad_out (torch.Tensor): (B, C, N) tensor with gradients of outputs

        Returns:
            torch.Tensor: (B, C, M) tensor with gradients of features
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        # grad_features = grad_out.new_zeros(B, c, m)
        grad_out_data = grad_out.data.contiguous()

        grad_features = ads_c.npu_three_interpolate_backward(B, c, n, m,
            grad_out_data, idx, weight)
        
        return grad_features, None, None

three_interpolate = ThreeInterpolateFunction.apply
