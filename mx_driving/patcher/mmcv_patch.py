# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import importlib
from types import ModuleType
from typing import List, Optional, Union, Dict

# Since MMCV 2.0 era, modules such as Runner, Hook, Parallel, Config, FileIO, have been moved to MMEngine
# Some of mmcv patches have been moved to mmengine_patch.py to comply with MMCV2.x style


# Special hack for fixing an mmcv v.s. mmdet v.s. mmdet3d compatibility flaw
def patch_mmcv_version(expected_version: str):
    try:
        mmcv = importlib.import_module("mmcv")
        origin_version = mmcv.__version__
        if origin_version == expected_version:
            return
        mmcv.__version__ = expected_version
        try:
            # fix mmdet compatibility check
            importlib.import_module("mmdet")
            importlib.import_module("mmdet3d")
        except ImportError:
            return
        finally:
            # restore mmcv version
            mmcv.__version__ = origin_version
    except ImportError:
        return


def msda(mmcv: ModuleType, options: Dict):
    importlib.import_module(f"{mmcv.__name__}.ops")
    if hasattr(mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction, "forward") \
        and hasattr(mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction, "backward"):
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
        
        mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction.forward = apply_mxdriving_msda_forward_param(MultiScaleDeformableAttnFunction.forward)
        mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction.backward = apply_mxdriving_msda_backward_param(MultiScaleDeformableAttnFunction.backward)
    else:
        raise AttributeError("In mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction, forward or backward not found")


def dc(mmcv: ModuleType, options: Dict):
    importlib.import_module(f"{mmcv.__name__}.ops")
    if hasattr(mmcv.ops.deform_conv, "DeformConv2dFunction") and hasattr(mmcv.ops.deform_conv, "deform_conv2d"):
        from mx_driving import DeformConv2dFunction, deform_conv2d
        
        mmcv.ops.deform_conv.DeformConv2dFunction = DeformConv2dFunction
        mmcv.ops.deform_conv.deform_conv2d = deform_conv2d
    else:
        raise AttributeError("In mmcv.ops.deform_conv, DeformConv2dFunction or deform_conv2d not found")


def mdc(mmcv: ModuleType, options: Dict):
    importlib.import_module(f"{mmcv.__name__}.ops")
    if hasattr(mmcv.ops.modulated_deform_conv, "ModulatedDeformConv2dFunction") \
        and hasattr(mmcv.ops.modulated_deform_conv, "modulated_deform_conv2d"):
        from mx_driving import ModulatedDeformConv2dFunction, modulated_deform_conv2d
            
        mmcv.ops.modulated_deform_conv.ModulatedDeformConv2dFunction = ModulatedDeformConv2dFunction
        mmcv.ops.modulated_deform_conv.modulated_deform_conv2d = modulated_deform_conv2d
    else:
        raise AttributeError("In mmcv.ops.modulated_deform_conv, ModulatedDeformConv2dFunction or modulated_deform_conv not found")


def spconv3d(mmcv: ModuleType, options: Dict):
    mxd = importlib.import_module("mx_driving")
    if ({"ops"}).issubset(mmcv.__dir__()):
        ops = importlib.import_module(f"{mmcv.__name__}.ops")
        ops.SparseConvTensor = mxd.SparseConvTensor
        ops.sparse_structure.SparseConvTensor = mxd.SparseConvTensor
        ops.SparseSequential = mxd.SparseSequential
        ops.sparse_modules.SparseSequential = mxd.SparseSequential
        ops.SparseModule = mxd.SparseModule
        ops.sparse_modules.SparseModule = mxd.SparseModule
        ops.SparseConvolution = mxd.SparseConvolution
        ops.sparse_conv.SparseConvolution = mxd.SparseConvolution
        ops.SubMConv3d = mxd.SubMConv3d
        ops.sparse_conv.SubMConv3d = mxd.SubMConv3d
        ops.SparseConv3d = mxd.SparseConv3d
        ops.sparse_conv.SparseConv3d = mxd.SparseConv3d
        ops.SparseInverseConv3d = mxd.SparseInverseConv3d
        ops.sparse_conv.SparseInverseConv3d = mxd.SparseInverseConv3d

    # support mmcv version 1.7.2
    if ({"cnn"}).issubset(mmcv.__dir__()):
        cnn = importlib.import_module(f"{mmcv.__name__}.cnn")
        if hasattr(cnn, "CONV_LAYERS"):
            cnn.CONV_LAYERS.module_dict['SubMConv3d'] = mxd.SubMConv3d
            cnn.CONV_LAYERS.module_dict['SparseConv3d'] = mxd.SparseConv3d
            cnn.CONV_LAYERS.module_dict['SparseInverseConv3d'] = \
                mxd.SparseInverseConv3d
    # support mmcv version 2.0.0 or above
    if importlib.util.find_spec("mmengine.registry") is not None:
        registry = importlib.import_module("mmengine.registry")
        if hasattr(registry, "MODELS"):
            registry.MODELS.module_dict['SubMConv3d'] = mxd.SubMConv3d
            registry.MODELS.module_dict['SparseConv3d'] = mxd.SparseConv3d
            registry.MODELS.module_dict['SparseInverseConv3d'] = \
                mxd.SparseInverseConv3d
