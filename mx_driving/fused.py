import warnings

from .ops.deform_conv2d import DeformConv2dFunction, deform_conv2d
from .ops.fused_bias_leaky_relu import npu_fused_bias_leaky_relu
from .ops.modulated_deform_conv2d import (ModulatedDeformConv2dFunction,
                                          modulated_deform_conv2d)
from .ops.multi_scale_deformable_attn import (
    MultiScaleDeformableAttnFunction, multi_scale_deformable_attn,
    npu_multi_scale_deformable_attn_function)
from .ops.npu_add_relu import npu_add_relu
from .ops.npu_deformable_aggregation import npu_deformable_aggregation
from .ops.npu_max_pool2d import npu_max_pool2d

warnings.warn(
    "This package is deprecated and will be removed in future. Please use `mx_driving.api` instead.", DeprecationWarning
)