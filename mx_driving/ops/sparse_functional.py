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
        num_voxels_, uni_voxels, unique_indices_offset, sorted_idx_to_former_indices, uni_argsort_indices = mx_driving._C.unique_voxel(ouidx_offset)
        indices_last = torch.tensor(ouidx_offset.shape).to(unique_indices_offset.device)
        unique_indices_offset = torch.cat((unique_indices_offset, indices_last), dim=0)
        
        # index_put and matmul
        out_features, _ = mx_driving._C.npu_sparse_matmul(
            features, weight, unique_indices_offset.int(), sorted_idx_to_former_indices.int(), outidx_pair.int())

        ctx.save_for_backward(features, weight, sorted_idx_to_former_indices.int(), unique_indices_offset.int())
        return out_features, outidx_pair.int()[uni_argsort_indices]

    @staticmethod
    @once_differentiable
    # pylint: disable=too-many-return-values
    def backward(ctx: Any, grad_out_features: torch.Tensor, grad_outidx=None) -> tuple:
        features, weight, sorted_idx_to_former_indices, unique_indices_offset = ctx.saved_tensors
        feature_grad, weight_grad = mx_driving._C.npu_sparse_conv3d_grad(
            unique_indices_offset, sorted_idx_to_former_indices, features, weight, grad_out_features
        )

        return feature_grad, None, weight_grad, None, None, None, None, None, None, None, None, None


def generate_map(coors, spaned_spatial_shape, bs, kernel_size):
    padding = kernel_size[0] // 2
    spatial_shape_size = spaned_spatial_shape[0] * spaned_spatial_shape[1] * spaned_spatial_shape[2]

    if (spatial_shape_size > 400000000):
        spatial_shape1 = (spaned_spatial_shape[1] * spaned_spatial_shape[0])
        new_coors1 = spatial_shape1 * coors[:, 0] + spaned_spatial_shape[1] * coors[:, 1] + coors[:, 2] + (padding + spaned_spatial_shape[1] * padding)
        map1 = torch.full((spatial_shape1 * bs, ), -1, dtype=torch.int32, device=coors.device)

        map1_length, unique_idx, _, _, _ = mx_driving.unique_voxel(new_coors1)
        map1[unique_idx] = torch.arange(map1_length, dtype=torch.int32, device=coors.device)
            
        map2 = torch.full((map1_length, spaned_spatial_shape[2]), -1, dtype=torch.int32, device=coors.device)
        map2[map1[new_coors1], (coors[:, 3] + padding)] = torch.arange(new_coors1.numel(), dtype=torch.int32, device=coors.device)
    else:
        flatten_indices = (
            coors[:, 0] * spatial_shape_size
            + coors[:, 1] * (spaned_spatial_shape[1] * spaned_spatial_shape[2])
            + coors[:, 2] * (spaned_spatial_shape[2])
            + coors[:, 3] + (((spaned_spatial_shape[1] * spaned_spatial_shape[2]) + (spaned_spatial_shape[2]) + 1) * padding)
        )

        map1 = torch.full((spatial_shape_size * bs, ), -1,
            dtype=torch.int32, device=coors.device)

        map1[flatten_indices] = torch.arange(flatten_indices.numel(), dtype=torch.int32, device=coors.device)
        map2 = torch.Tensor([]).int()

    return map1, map2


class SubMConvFunction(Function):
    @staticmethod
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(
        ctx: Any,
        features,
        indices,
        weight,
        indices_offset,
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
        indices = indices.contiguous()
        spaned_spatial_shape = (out_spatial_shape[0] + 2 * (kernel_size[0] // 2), out_spatial_shape[1] + 2 * (kernel_size[1] // 2),
                     out_spatial_shape[2] + 2 * (kernel_size[2] // 2))
        if indices_offset is None:
            map1, map2 = generate_map(indices, spaned_spatial_shape, batch_size, kernel_size)
            with_key = 0
            indices_offset = torch.Tensor([]).int()
        else:
            map1, map2 = torch.Tensor([]).int(), torch.Tensor([]).int()
            with_key = 1

        out_features, out_indices_offset = mx_driving._C.npu_subm_sparse_conv3d_v3(features, weight, indices, indices_offset, map1, map2, kernel_size,
            features.shape[1], out_channels, spaned_spatial_shape, batch_size, with_key)

        indices_offset = indices_offset if with_key == 1 else out_indices_offset
        ctx.save_for_backward(features, weight, indices_offset)
        return out_features, indices, indices_offset
    
    @staticmethod
    @once_differentiable
    # pylint: disable=too-many-return-values
    def backward(ctx: Any, grad_out_features: torch.Tensor, grad_outidx=None, grad_offset=None) -> tuple:
        features, weight, ouidx_offset = ctx.saved_tensors
        feature_grad, weight_grad = mx_driving._C.npu_subm_sparse_conv3d_grad_v2(
            features, weight, grad_out_features, ouidx_offset
        )

        return feature_grad, None, weight_grad, None, None, None, None, None, None, None, None, None, None


indice_conv = SparseConvFunction.apply
indice_subm_conv = SubMConvFunction.apply
