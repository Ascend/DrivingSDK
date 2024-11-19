from typing import Optional, List
import torch
from torch import Tensor
import torch.onnx.symbolic_helper as sym_help
import mx_driving.fused


class NPUMultiScaleDeformableAttnOP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        return mx_driving.fused.multi_scale_deformable_attn(*args, **kwargs)

    @staticmethod
    # 'pylint: disable=too-many-arguments,huawei-too-many-arguments
    def symbolic(g, value: Tensor, value_spatial_shapes: Tensor, value_level_start_index: Tensor,
                                                sampling_locations: Tensor, attention_weights: Tensor):
        return g.op("npu::MultiScaleDeformableAttn",
                    value,
                    value_spatial_shapes,
                    value_level_start_index,
                    sampling_locations,
                    attention_weights)
