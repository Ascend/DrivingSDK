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

import numpy as np
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import mx_driving._C


class SparseConvFunction(Function):
    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(
        ctx: Any,
        features,
        indices,
        weight,
        out_spatial_shape,
        out_channels,
        batch_size,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
    ) -> torch.Tensor:

        device = features.device
        weight = weight.data
        # calculate the index pair
        outidx_pair, ouidx_offset = mx_driving._C.npu_sparse_conv3d(
            indices, kernel_size, stride, padding, out_channels, out_spatial_shape, batch_size
        )
        # sort and nonezero
        to_insert = torch.tensor(-1).to(device)
        sorted_idx, sorted_idx_to_former_indices = torch.sort(ouidx_offset.view(torch.float32))
        new_sorted_idx = torch.cat((to_insert.view(1), sorted_idx.view(torch.int32)), 0)
        new_sorted_idx_2 = torch.cat((sorted_idx.view(torch.int32), to_insert.view(1)), 0)
        sub_result = new_sorted_idx - new_sorted_idx_2
        unique_indices_offset = torch.nonzero(sub_result != 0)
        # index_put and matmul
        out_features, outidx = mx_driving._C.multi_to_sparse_v2(
            features, weight, unique_indices_offset.int(), sorted_idx_to_former_indices.int(), outidx_pair.int()
        )
        outidx, outidx_ = torch.chunk(outidx, 2, dim=1)

        ctx.save_for_backward(features, weight, sorted_idx_to_former_indices.int(), unique_indices_offset.int())
        return out_features, outidx

    @staticmethod
    @once_differentiable
    # pylint: disable=too-many-return-values
    def backward(ctx: Any, grad_out_features: torch.Tensor, grad_outidx=None) -> tuple:
        features, weight, sorted_idx_to_former_indices, unique_indices_offset = ctx.saved_tensors
        feature_grad, weight_grad = mx_driving._C.npu_sparse_conv3d_grad(
            unique_indices_offset, sorted_idx_to_former_indices, features, weight, grad_out_features
        )

        return feature_grad, None, weight_grad, None, None, None, None, None, None, None, None, None


def generate_map(coors, spatial_shape, bs):
    spatial_shape_size = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]

    if spatial_shape_size * bs > 200000000:
        spatial_shape1 = (spatial_shape[1] * spatial_shape[0])
        new_coors1 = spatial_shape1 * coors[:, 0] + spatial_shape[1] * coors[:, 1] + coors[:, 2]
        map1 = torch.full((spatial_shape1 * bs, ), -1, dtype=torch.int32, device=coors.device)

        map1[new_coors1] = 1
        mask = (map1 != -1)
        map1Length = map1[mask].shape[0]
        map1[mask] = torch.arange(map1Length, dtype=torch.int32, device=coors.device)

        map2 = torch.full((map1Length, spatial_shape[2]), -1, dtype=torch.int32, device=coors.device)
        map2[map1[new_coors1], coors[:, 3]] = torch.arange(new_coors1.numel(), dtype=torch.int32, device=coors.device)
        return map1, map2
    else:
        flatten_indices = (
            coors[:, 0] * spatial_shape_size
            + coors[:, 1] * (spatial_shape[1] * spatial_shape[2])
            + coors[:, 2] * (spatial_shape[2])
            + coors[:, 3]
        )

        map1 = torch.full((spatial_shape_size * bs, ), -1,
            dtype=torch.int32, device=coors.device)

        map1[flatten_indices] = torch.arange(flatten_indices.numel(), dtype=torch.int32, device=coors.device)
        return map1, torch.Tensor([]).int()


class SubMConvFunction(Function):
    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(
        ctx: Any,
        features,
        indices,
        weight,
        out_spatial_shape,
        out_channels,
        batch_size,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
    ) -> torch.Tensor:
        weight = weight.data
        map1, map2 = generate_map(indices, out_spatial_shape, batch_size)
        output_iml2col, indices_offset = mx_driving._C.npu_subm_sparse_conv3d_v2(features, indices, map1, map2, kernel_size,
            features.shape[1], out_spatial_shape, batch_size)
        out_features = output_iml2col @ weight.reshape(-1, out_channels)

        ctx.kernel_size = kernel_size
        ctx.save_for_backward(features, weight, output_iml2col, indices_offset)
        return out_features, indices, indices_offset

    @staticmethod
    @once_differentiable
    # pylint: disable=too-many-return-values
    def backward(ctx: Any, grad_out_features: torch.Tensor, grad_outidx=None, grad_offset=None) -> tuple:
        features, weight, output_iml2col, ouidx_offset = ctx.saved_tensors
        weight_grad = output_iml2col.T @ grad_out_features
        weight_shape = weight.shape
        kernel_num = weight_shape[0] * weight_shape[1] * weight_shape[2]
        weight_grad = weight_grad.view(
            weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3], weight_shape[4]
        )
        mask = ouidx_offset != -1
        valid_indices = torch.nonzero(mask).view(-1)
        ouidx_offset = torch.index_select(ouidx_offset, 0, valid_indices)
        grad_out_features_iml2col = mx_driving._C.npu_subm_sparse_conv3d_grad(
            ouidx_offset, valid_indices.int(), weight, grad_out_features, features.shape[0], ctx.kernel_size
        )
        grad_out_features_iml2col = grad_out_features_iml2col.view(features.shape[0], -1)
        weight = weight.permute(0, 1, 2, 4, 3).contiguous()
        weight_permute = weight.view(kernel_num * weight_shape[4], weight_shape[3])
        feature_grad = grad_out_features_iml2col @ weight_permute

        return feature_grad, None, weight_grad, None, None, None, None, None, None, None, None, None


class SubMConvWithKeyFunction(Function):

    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(
        ctx: Any,
        features,
        indices,
        weight,
        ouidx_offset,
        out_spatial_shape,
        out_channels,
        batch_size,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
    ) -> torch.Tensor:
        device = features.device
        weight = weight.data
        mask = ouidx_offset != -1
        valid_indices = torch.nonzero(mask).view(-1)
        ouidx_offset = torch.index_select(ouidx_offset, 0, valid_indices)
        output_iml2col = mx_driving._C.npu_subm_sparse_conv3d_with_key(
            ouidx_offset, valid_indices.int(), weight, features, features.shape[0], kernel_size
        )
        weight_flatten = weight.view(kernel_size[0] * kernel_size[1] * kernel_size[2] * features.shape[1], out_channels)
        output_iml2col = output_iml2col.view(features.shape[0], -1)
        out_features = output_iml2col @ weight_flatten
        ctx.kernel_size = kernel_size
        ctx.save_for_backward(features, weight, output_iml2col, ouidx_offset, valid_indices)
        return out_features, indices

    @staticmethod
    @once_differentiable
    # pylint: disable=too-many-return-values
    def backward(ctx: Any, grad_out_features: torch.Tensor, grad_outidx=None) -> tuple:
        features, weight, output_iml2col, ouidx_offset, valid_indices = ctx.saved_tensors
        weight_grad = output_iml2col.T @ grad_out_features
        weight_shape = weight.shape
        kernel_num = weight_shape[0] * weight_shape[1] * weight_shape[2]
        weight_grad = weight_grad.view(
            weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3], weight_shape[4]
        )
        grad_out_features_iml2col = mx_driving._C.npu_subm_sparse_conv3d_grad(
            ouidx_offset, valid_indices.int(), weight, grad_out_features, features.shape[0], ctx.kernel_size
        )
        grad_out_features_iml2col = grad_out_features_iml2col.view(features.shape[0], -1)
        weight = weight.permute(0, 1, 2, 4, 3).contiguous()
        weight_permute = weight.view(kernel_num * weight_shape[4], weight_shape[3])
        feature_grad = grad_out_features_iml2col @ weight_permute

        return feature_grad, None, weight_grad, None, None, None, None, None, None, None, None, None, None


indice_conv = SparseConvFunction.apply
indice_subm_conv = SubMConvFunction.apply
indice_subm_conv_with_key = SubMConvWithKeyFunction.apply
