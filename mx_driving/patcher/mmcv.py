# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import importlib
from types import ModuleType
from typing import Dict


def patch_mmcv_version(expected_version: str):
    try:
        mmcv = importlib.import_module("mmcv")
        origin_version = mmcv.__version__
        if origin_version == expected_version:
            return
        mmcv.__version__ = expected_version
        try:
            # fix mmdet stupid compatibility check
            importlib.import_module("mmdet")
            importlib.import_module("mmdet3d")
        except ImportError:
            return
        finally:
            # restore mmcv version
            mmcv.__version__ = origin_version
    except ImportError:
        return


def msda(mmcvops: ModuleType, options: Dict):
    from mx_driving import MultiScaleDeformableAttnFunction

    def apply_mxdriving_msda_forward_param(function):
        # pylint: disable=too-many-arguments,huawei-too-many-arguments
        def wrapper(ctx, value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step=None):
            return function(ctx, value, spatial_shapes, level_start_index, sampling_locations, attention_weights)
        return wrapper
    
    def apply_mxdriving_msda_backward_param(function):
        def wrapper(ctx, grad_output):
            return *(function(ctx, grad_output)), None
        return wrapper

    if hasattr(mmcvops, "multi_scale_deform_attn"):
        mmcvops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction.forward = apply_mxdriving_msda_forward_param(MultiScaleDeformableAttnFunction.forward)
        mmcvops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction.backward = apply_mxdriving_msda_backward_param(MultiScaleDeformableAttnFunction.backward)
    else:
        raise AttributeError("multi_scale_deform_attn not found")


def dc(mmcvops: ModuleType, options: Dict):
    from mx_driving import DeformConv2dFunction, deform_conv2d

    if hasattr(mmcvops, "deform_conv"):
        mmcvops.deform_conv.DeformConv2dFunction = DeformConv2dFunction
        mmcvops.deform_conv.deform_conv2d = deform_conv2d
    else:
        raise AttributeError("deform_conv not found")


def mdc(mmcvops: ModuleType, options: Dict):
    from mx_driving import ModulatedDeformConv2dFunction, modulated_deform_conv2d

    if hasattr(mmcvops, "modulated_deform_conv"):
        mmcvops.modulated_deform_conv.ModulatedDeformConv2dFunction = ModulatedDeformConv2dFunction
        mmcvops.modulated_deform_conv.modulated_deform_conv2d = modulated_deform_conv2d
    else:
        raise AttributeError("modulated_deform_conv not found")