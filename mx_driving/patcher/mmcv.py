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
    from mx_driving import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn

    if hasattr(mmcvops, "multi_scale_deformable_attn"):
        mmcvops.multi_scale_deformable_attn.MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction
        mmcvops.multi_scale_deformable_attn.multi_scale_deformable_attn = multi_scale_deformable_attn


def dc(mmcvops: ModuleType, options: Dict):
    from mx_driving import DeformConv2dFunction, deform_conv2d

    if hasattr(mmcvops, "deform_conv"):
        mmcvops.deform_conv.DeformConv2dFunction = DeformConv2dFunction
        mmcvops.deform_conv.deform_conv2d = deform_conv2d


def mdc(mmcvops: ModuleType, options: Dict):
    from mx_driving import ModulatedDeformConv2dFunction, modulated_deform_conv2d

    if hasattr(mmcvops, "modulated_deform_conv"):
        mmcvops.modulated_deform_conv.ModulatedDeformConv2dFunction = ModulatedDeformConv2dFunction
        mmcvops.modulated_deform_conv.modulated_deform_conv2d = modulated_deform_conv2d
