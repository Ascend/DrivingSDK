# Copyright (c) 2024, Huawei Technologies.All rights reserved.
# Copyright 2019 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import torch
import numpy as np
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import ads_c
from . import sparse_ops as ops


class SparseBaseCovFunction(Function):
    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx: Any, features, indices, weight, out_spatial_shape,
                out_channels, batch_size,
                kernel_size, stride, padding, dilation, output_padding,
                groups, bias, subm, inverse=False) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Features that needs to convolute.
            filters (torch.nn.parameter.Parameter): Convolution filters.
            indice_pairs (torch.Tensor): Indice pairs between inputs locations
                and outputs locations.
            indice_pair_num (torch.Tensor): Indice pairs num.
            num_activate_out (torch.Tensor): Output channels num.

        Returns:
            torch.Tensor: Output features from gather-gemm-scatter.
        """
        device = features.device
        out_spatial_shape = [int(i) for i in out_spatial_shape]
        if subm:
            out_features, outidx_pair, ouidx_offset = indice_subm_conv(features, indices, weight,
                                                kernel_size, out_channels,
                                                out_spatial_shape, batch_size)
        elif inverse:
            out_features, outidx_pair, ouidx_offset = indice_inverse_conv(features, indices, weight,
                                                    kernel_size, stride, padding, dilation, output_padding,
                                                    out_channels, out_spatial_shape, batch_size)
        else:
            outidx_pair, ouidx_offset = indice_conv(indices, kernel_size, stride, padding,
                                                    out_channels, out_spatial_shape, batch_size)
        to_insert = torch.tensor(-1).to(device)
        sorted_idx, sorted_idx_to_former_indices = torch.sort(ouidx_offset.view(torch.float32))
        new_sorted_idx = torch.cat((to_insert.view(1), sorted_idx.view(torch.int32)), 0)
        new_sorted_idx_2 = torch.cat((sorted_idx.view(torch.int32), to_insert.view(1)), 0)
        sub_result = new_sorted_idx - new_sorted_idx_2
        unique_indices_offset = torch.nonzero(sub_result != 0)
        if subm or inverse:
            out_features, outidx = multi_to_sparse(out_features, unique_indices_offset.int(),
                                                    sorted_idx_to_former_indices.int(), outidx_pair.int())
        else:
            out_features, outidx = multi_to_sparse_v2(features, weight, unique_indices_offset.int(),
                                                sorted_idx_to_former_indices.int(), outidx_pair.int())

        outidx, outidx_ = torch.chunk(outidx, 2, dim=1)
        if bias is not None:
            out_features += bias
        ctx.save_for_backward(features, weight, sorted_idx_to_former_indices.int(), unique_indices_offset.int())
        return out_features, outidx

    @staticmethod
    @once_differentiable
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def backward(ctx: Any, grad_out_features: torch.Tensor, grad_outidx = None) -> tuple:
        features, weight, sorted_idx_to_former_indices, unique_indices_offset = ctx.saved_tensors
        weight_grad, feature_grad = ads_c.npu_sparse_conv3d_grad(unique_indices_offset,
                                                                 sorted_idx_to_former_indices,
                                                                 features, weight, grad_out_features)
        return feature_grad, None, weight_grad, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class SparseConvFunction(Function):

    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx: Any, indices: torch.Tensor,
                kernel_size, stride, padding,
                out_channels: int, out_spatial_shape,
                batch_size: int) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Features that needs to convolute.
            indices (torch.Tensor):  Indices of feature that needs to convolute.
            kernel_size (Union[List, Tuple]): Kernel size of convolute.
            out_channels (int): Output channels num..
            spatial_shape (Union[List, Tuple]): Output channels num.

        Returns:
            torch.Tensor: Output features from gather-gemm-scatter.
        """

        outidx_pair, ouidx_offset = ads_c.npu_sparse_conv3d(indices, kernel_size, stride, padding,
                                                                    out_channels, out_spatial_shape, batch_size)

        return outidx_pair, ouidx_offset


class SparseInverseConvFunction(Function):

    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx: Any, features: torch.Tensor, indices: torch.Tensor, weight: torch.Tensor,
                kernel_size, stride, padding, dilation, output_padding,
                out_channels: int, out_spatial_shape,
                batch_size: int) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Features that needs to convolute.
            indices (torch.Tensor):  Indices of feature that needs to convolute.
            kernel_size (Union[List, Tuple]): Kernel size of convolute.
            out_channels (int): Output channels num..
            spatial_shape (Union[List, Tuple]): Output channels num.

        Returns:
            torch.Tensor: Output features from gather-gemm-scatter.
        """
        out_features, outidx_pair, ouidx_offset = ads_c.npu_sparse_inverse_conv3d(features, indices, weight,
                                        kernel_size, stride, padding, dilation, output_padding,
                                        out_channels, out_spatial_shape, batch_size)

        return out_features, outidx_pair, ouidx_offset


class SubMConvFunction(Function):

    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx: Any, features: torch.Tensor, indices: torch.Tensor, weight: torch.Tensor,
                kernel_size,
                out_channels: int, out_spatial_shape,
                batch_size: int) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): Features that needs to convolute.
            filters (torch.nn.parameter.Parameter): Convolution filters.
            indice_pairs (torch.Tensor): Indice pairs between inputs locations
                and outputs locations.
            indice_pair_num (torch.Tensor): Indice pairs num.
            num_activate_out (torch.Tensor): Output channels num.

        Returns:
            torch.Tensor: Output features from gather-gemm-scatter.
        """
        out_features, outidx_pair, ouidx_offset = ads_c.npu_subm_sparse_conv3d(features, indices, weight,
                                              kernel_size, out_channels,
                                              out_spatial_shape, batch_size)

        return out_features, outidx_pair, ouidx_offset


class MultiToSparseFunction(Function):

    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx: Any, out_features, unique_indices_offset,
                sorted_idx_to_former_indices, outidx_pair):
        """
        Args:
            features (torch.Tensor): Features that needs to convolute.
            filters (torch.nn.parameter.Parameter): Convolution filters.
            indice_pairs (torch.Tensor): Indice pairs between inputs locations
                and outputs locations.
            indice_pair_num (torch.Tensor): Indice pairs num.
            num_activate_out (torch.Tensor): Output channels num.

        Returns:
            torch.Tensor: Output features from gather-gemm-scatter.
        """
        feature, indices = ads_c.multi_to_sparse(out_features, unique_indices_offset,
                                                 sorted_idx_to_former_indices, outidx_pair)
        return feature, indices


class MultiToSparseV2Function(Function):

    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx: Any, features, weight, unique_indices_offset,
                sorted_idx_to_former_indices, outidx_pair):
        """
        Args:
            features (torch.Tensor): Features that needs to convolute.
            filters (torch.nn.parameter.Parameter): Convolution filters.
            indice_pairs (torch.Tensor): Indice pairs between inputs locations
                and outputs locations.
            indice_pair_num (torch.Tensor): Indice pairs num.
            num_activate_out (torch.Tensor): Output channels num.

        Returns:
            torch.Tensor: Output features from gather-gemm-scatter.
        """
        feature, indices = ads_c.multi_to_sparse_v2(features, weight, unique_indices_offset,
                                                 sorted_idx_to_former_indices, outidx_pair)
        return feature, indices


indice_conv = SparseConvFunction.apply
indice_inverse_conv = SparseInverseConvFunction.apply
indice_subm_conv = SubMConvFunction.apply
multi_to_sparse = MultiToSparseFunction.apply
multi_to_sparse_v2 = MultiToSparseV2Function.apply
indices_conv_base = SparseBaseCovFunction.apply