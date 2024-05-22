from typing import Any, Optional, Tuple

import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class DynamicScatterFunction(Function):
    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx: Any, feats: torch.Tensor, coors: torch.Tensor, reduce_type) -> Tuple[torch.Tensor, torch.Tensor]:

        coors_trans = torch.permute(coors, (1, 0)).long()
        coors_max = torch.max(coors_trans, 1)[0] + 1
        cof_tensor = torch.tensor([coors_max[1] * coors_max[2], coors_max[2], 1]).npu()
        coors_clean = torch.masked_fill(coors, coors < 0, -1).npu()
        hash_key = torch.sum(cof_tensor * coors_clean, 1)

        out_coors_unique2, coors_map, reduce_count = torch._unique2(hash_key, True, True, True)

        result = ads_c.npu_dynamic_scatter(cof_tensor, out_coors_unique2, coors_map,
                                           reduce_count, feats, coors, reduce_type)
        (voxel_feats, voxel_coors, point2voxel_map, voxel_points_count) = result

        ctx.reduce_type = reduce_type
        ctx.save_for_backward(feats, voxel_feats, point2voxel_map, voxel_points_count)
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
