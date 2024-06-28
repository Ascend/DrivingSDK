"""
Copyright (c) OpenMMLab. All rights reserved.
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
Modification by: Huawei Developers
Modification date: 2024-06-04 
Modification Description: 
Modification 1. Add support for Ascend NPU
"""
from typing import Any, Optional, Tuple

import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class DynamicScatterFunction(Function):
    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx: Any, feats: torch.Tensor, coors: torch.Tensor,
                reduce_type: str = 'max') -> Tuple[torch.Tensor, torch.Tensor]:
        if reduce_type not in ('max', 'sum', 'mean'):
            raise ValueError("reduce_type should be 'max', 'sum' or 'mean', but now is %s." % reduce_type)

        voxel_idx = ads_c.point_to_voxel(coors, None, None)
        num_voxels, uniqued_voxel_idx, prefix_sum_point_per_voxel, argsort_coor = ads_c.unique_voxel(voxel_idx)
        voxel_coors = ads_c.voxel_to_point(uniqued_voxel_idx, None, None)
        voxel_feats, compare_mask = ads_c.npu_dynamic_scatter(feats, coors, prefix_sum_point_per_voxel,
                                                              argsort_coor, num_voxels, reduce_type)

        ctx.reduce_type = reduce_type
        ctx.feats_shape = feats.shape
        ctx.save_for_backward(prefix_sum_point_per_voxel, argsort_coor, compare_mask)
        ctx.mark_non_differentiable(voxel_coors)
        return voxel_feats, voxel_coors

    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    # 'pylint: disable=too-many-return-arguments,huawei-too-many-return-arguments
    def backward(ctx: Any,
                 grad_voxel_feats: torch.Tensor,
                 grad_voxel_coors: Optional[torch.Tensor] = None) -> tuple:
        (prefix_sum_point_per_voxel, argsort_coor, compare_mask) = ctx.saved_tensors
        grad_point_feats = torch.zeros(ctx.feats_shape, dtype=grad_voxel_feats.dtype, device=grad_voxel_feats.device)
        ads_c.npu_dynamic_scatter_grad(grad_point_feats, grad_voxel_feats.contiguous(), prefix_sum_point_per_voxel,
                                       argsort_coor, compare_mask, ctx.reduce_type)
        return grad_point_feats, None, None


npu_dynamic_scatter = DynamicScatterFunction.apply
