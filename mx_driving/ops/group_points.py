# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2024 Huawei Technologies Co., Ltd
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
import warnings

import numpy as np
import torch
import torch_npu
from torch.autograd import Function

import mx_driving._C


class AdsGroupPoints(Function):
    """Group feature with given index."""

    @staticmethod
    def forward(ctx, features: torch.Tensor, indices: torch.Tensor):
        """
        Args:
            features (Tensor): Tensor of features to group, input shape is (B, C, N).
            indices (Tensor):  The indices of features to group with, input shape is (B, npoints, nsample).

        Returns:
            Tensor: Grouped features, the shape is (B, C, npoints, nsample)
        """
        features = features.contiguous()
        indices = indices.contiguous()

        B, C, N = features.size()
        _, npoints, nsample = indices.size()

        output = mx_driving._C.group_points(features, indices, B, C, N, npoints, nsample)

        ctx.for_backwards = (indices, N)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        Args:
            grad_out (Tensor): (B, C, npoint, nsample) tensor of the gradients
                of the output from forward.

        Returns:
            Tensor: (B, C, N) gradient of the features.
        """
        if (torch.numel(grad_out) == 0):
            raise Exception("Error! Input Tensor can not be a empty Tensor.\n")
        
        idx, N = ctx.for_backwards

        B, C, npoints, nsample = grad_out.size()
        grad_features = mx_driving._C.group_points_backward(grad_out, idx, B, C, N, npoints, nsample)
        return grad_features, None


def group_points(features: torch.Tensor, indices: torch.Tensor):
    return AdsGroupPoints.apply(features, indices)


def npu_group_points(features: torch.Tensor, indices: torch.Tensor):
    warnings.warn(
        "`npu_group_points` will be deprecated in future. Please use `group_points` instead.", DeprecationWarning
    )
    return AdsGroupPoints.apply(features, indices)
