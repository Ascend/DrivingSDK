import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class DynamicScatterFunction(Function):
    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(ctx, feats, coors, reduce_type):

        coors_trans = torch.permute(coors, (1, 0)).long()
        coors_max = torch.max(coors_trans, 1)[0] + 1
        cof_tensor = torch.tensor([coors_max[1] * coors_max[2], coors_max[2], 1]).npu()
        coors_clean = torch.masked_fill(coors, coors < 0, -1).npu()
        hash_key = torch.sum(cof_tensor * coors_clean, 1)

        out_coors_unique2, coors_map, reduce_count = torch._unique2(hash_key, True, True, True)

        (voxel_feats, voxel_coors, point2voxel_map, voxel_points_count) = ads_c.npu_dynamic_scatter(cof_tensor, out_coors_unique2, coors_map, reduce_count, feats, coors,
                                                                                                    reduce_type)
        ctx.reduce_type = reduce_type
        ctx.save_for_backward(feats, voxel_feats, point2voxel_map, voxel_points_count)
        ctx.mark_non_differentiable(voxel_coors)
        return voxel_feats, voxel_coors

    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    # 'pylint: disable=too-many-return-arguments,huawei-too-many-return-arguments
    def backward(ctx, y_grad):
        raise "Error: npu_dynamic_scatter is not currently support backward."


npu_dynamic_scatter = DynamicScatterFunction.apply
