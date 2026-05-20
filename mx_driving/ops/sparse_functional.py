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
import torch_npu
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from mx_driving._C import (
    npu_sparse_conv3d,
    unique_voxel,
    npu_sparse_matmul,
    npu_sparse_conv3d_grad,
    npu_subm_sparse_conv3d_v3,
    npu_subm_sparse_conv3d_arch35,
    npu_subm_sparse_conv3d_grad_v2,
    npu_subm_sparse_conv3d_grad_arch35,
    npu_sparse_inverse_conv3d,
)


class SparseConvFunction(Function):
    @staticmethod
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
        # calculate the index pair
        # pylint: disable=unpacking-non-sequence
        outidx_pair, ouidx_offset = npu_sparse_conv3d(
            indices, kernel_size, stride, padding, out_channels, out_spatial_shape, batch_size
        )
        # sort and nonezero
        # pylint: disable=unpacking-non-sequence
        num_voxels_, uni_voxels, unique_indices_offset, sorted_idx_to_former_indices, uni_argsort_indices = (
            unique_voxel(ouidx_offset)
        )
        indices_last = torch.tensor(ouidx_offset.shape).to(unique_indices_offset.device)
        unique_indices_offset = torch.cat((unique_indices_offset, indices_last), dim=0)

        # index_put and matmul
        # pylint: disable=unpacking-non-sequence
        out_features, _ = npu_sparse_matmul(
            features, weight, unique_indices_offset.int(), sorted_idx_to_former_indices.int(), outidx_pair.int()
        )

        ctx.save_for_backward(features, weight, sorted_idx_to_former_indices.int(), unique_indices_offset.int())
        return (
            out_features,
            outidx_pair.int()[uni_argsort_indices],
            unique_indices_offset,
            sorted_idx_to_former_indices,
            outidx_pair,
        )

    @staticmethod
    @once_differentiable
    def backward(
        ctx: Any,
        grad_out_features: torch.Tensor,
        grad_outidx=None,
        grad_unique_indices_offset=None,
        grad_sorted_idx_to_former_indices=None,
        grad_outidx_pair=None,
    ) -> tuple:
        features, weight, sorted_idx_to_former_indices, unique_indices_offset = ctx.saved_tensors
        # pylint: disable=unpacking-non-sequence
        feature_grad, weight_grad = npu_sparse_conv3d_grad(
            sorted_idx_to_former_indices, unique_indices_offset, features, weight, grad_out_features
        )

        return feature_grad, None, weight_grad, None, None, None, None, None, None, None, None, None


def generate_map(coors, spaned_spatial_shape, bs, kernel_size):
    padding = kernel_size[0] // 2
    spatial_shape_size = spaned_spatial_shape[0] * spaned_spatial_shape[1] * spaned_spatial_shape[2]

    if spatial_shape_size > 40000000:
        spatial_shape1 = spaned_spatial_shape[1] * spaned_spatial_shape[0]
        new_coors1 = (
            spatial_shape1 * coors[:, 0]
            + spaned_spatial_shape[1] * coors[:, 1]
            + coors[:, 2]
            + padding
            + spaned_spatial_shape[1] * padding
        )
        map1 = torch.full((spatial_shape1 * bs,), -1, dtype=torch.int32, device=coors.device)
        # pylint: disable=unpacking-non-sequence
        map1_length, unique_idx, _, _, _ = unique_voxel(new_coors1)
        map1[unique_idx] = torch.arange(map1_length, dtype=torch.int32, device=coors.device)
        map2 = torch.full((map1_length, spaned_spatial_shape[2]), -1, dtype=torch.int32, device=coors.device)
        map2[map1[new_coors1], (coors[:, 3] + padding)] = torch.arange(
            new_coors1.numel(), dtype=torch.int32, device=coors.device
        )
    else:
        map1 = torch.full(
            (bs, spaned_spatial_shape[0], spaned_spatial_shape[1], spaned_spatial_shape[2]),
            -1,
            dtype=torch.int32,
            device=coors.device,
        )
        bs, sp0, sp1, sp2 = coors.split(1, dim=1)
        bs, sp0, sp1, sp2 = bs.flatten(), sp0.flatten(), sp1.flatten(), sp2.flatten()
        map1[bs, sp0 + padding, sp1 + padding, sp2 + padding] = torch.arange(
            coors.shape[0], dtype=torch.int32, device=coors.device
        )
        map1 = map1.flatten().contiguous()
        map2 = torch.Tensor([]).int()

    return map1, map2


class SubMConvFunction(Function):
    @staticmethod
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
        spaned_spatial_shape = (
            out_spatial_shape[0] + 2 * (kernel_size[0] // 2),
            out_spatial_shape[1] + 2 * (kernel_size[1] // 2),
            out_spatial_shape[2] + 2 * (kernel_size[2] // 2),
        )
        if indices_offset is None:
            map1, map2 = generate_map(indices, spaned_spatial_shape, batch_size, kernel_size)
            with_key = 0
            indices_offset = torch.Tensor([]).int()
        else:
            map1, map2 = torch.Tensor([]).int(), torch.Tensor([]).int()
            with_key = 1
        DEVICE_NAME = torch_npu.npu.get_device_name(features.device.index)
        if 'Ascend910' in DEVICE_NAME:
            subm_forward_func = npu_subm_sparse_conv3d_v3
        elif 'Ascend950' in DEVICE_NAME:
            subm_forward_func = npu_subm_sparse_conv3d_arch35
        else:
            raise NotImplementedError(
                'The npu_subm_sparse_conv3d operator currently only supports Ascend910B, Ascend910C and Ascend950.'
            )

        out_features, out_indices_offset = subm_forward_func(
            features,
            weight,
            indices,
            indices_offset,
            map1,
            map2,
            kernel_size,
            features.shape[1],
            out_channels,
            spaned_spatial_shape,
            batch_size,
            with_key,
        )

        indices_offset = indices_offset if with_key == 1 else out_indices_offset
        ctx.save_for_backward(features, weight, indices_offset)
        return out_features, indices, indices_offset

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_out_features: torch.Tensor, grad_outidx=None, grad_offset=None) -> tuple:
        features, weight, ouidx_offset = ctx.saved_tensors
        DEVICE_NAME = torch_npu.npu.get_device_name(features.device.index)
        if 'Ascend910' in DEVICE_NAME:
            subm_grad_func = npu_subm_sparse_conv3d_grad_v2
        elif 'Ascend950' in DEVICE_NAME:
            subm_grad_func = npu_subm_sparse_conv3d_grad_arch35
        else:
            raise NotImplementedError(
                'The npu_subm_sparse_conv3d_grad operator currently only supports Ascend910B, Ascend910C and Ascend950.'
            )

        feature_grad, weight_grad = subm_grad_func(features, weight, grad_out_features, ouidx_offset)

        return feature_grad, None, weight_grad, None, None, None, None, None, None, None, None, None, None


class SparseInverseConvFunction(Function):
    @staticmethod
    def forward(
        ctx: Any,
        features,
        weight,
        in_channels,
        out_channels,
        kernel_size,
        indice_data,
    ) -> torch.Tensor:
        weight = weight.data
        output_img2col = npu_sparse_inverse_conv3d(
            features,
            indice_data.origin_indices,
            indice_data.unique_indices_offset.int(),
            indice_data.sorted_idx_to_former_indices,
            kernel_size,
            in_channels,
        )
        out_features = output_img2col @ weight.reshape(-1, out_channels)
        ctx.save_for_backward(
            weight,
            indice_data.unique_indices_offset.int(),
            indice_data.sorted_idx_to_former_indices,
            indice_data.outidx_pair,
            output_img2col,
        )
        return out_features

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_out_features: torch.Tensor) -> tuple:
        weight, unique_indices_offset, sorted_idx_to_former_indices, outidx_pair, output_img2col = ctx.saved_tensors
        weight_shape = weight.shape
        weight.data = weight.data.permute(0, 1, 2, 4, 3).contiguous()
        # pylint: disable=unpacking-non-sequence
        inverse_feature_grad, _ = npu_sparse_matmul(
            grad_out_features, weight, unique_indices_offset, sorted_idx_to_former_indices, outidx_pair
        )
        inverse_weight_grad = grad_out_features.transpose(0, 1).contiguous() @ output_img2col

        inverse_weight_grad = (
            inverse_weight_grad.transpose(0, 1)
            .contiguous()
            .view(weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3], weight_shape[4])
        )

        return inverse_feature_grad, inverse_weight_grad, None, None, None, None


indice_conv = SparseConvFunction.apply
indice_subm_conv = SubMConvFunction.apply
indice_inverse_conv = SparseInverseConvFunction.apply
