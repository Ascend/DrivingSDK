# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Patcher module for non-invasive GPU to NPU model migration.

Quick Start:
    from mx_driving.patcher import default_patcher
    default_patcher.apply()

Custom Usage:
    from mx_driving.patcher import Patcher, AtomicPatch

    patcher = Patcher()
    patcher.add(
        AtomicPatch("mmcv.ops.xxx", npu_replacement),
    )
    patcher.apply()

Disable Patches:
    from mx_driving.patcher import default_patcher, MSDA, BatchMatmul
    default_patcher.disable(MSDA.name, BatchMatmul.name).apply()

Configure Logging:
    from mx_driving.patcher import configure_patcher_logging

    # Strict mode
    configure_patcher_logging(on_fail="exception", on_error="exception")

    # Quiet mode
    configure_patcher_logging(on_skip="silent", on_fail="debug")
"""
from typing import Dict, List, Type

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         Built-in Dependencies                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Pre-import torch and torch_npu to ensure they are available for patches.
# This prevents errors when users forget to import these modules before using patcher.
try:
    import torch
except ImportError:
    pass

try:
    import torch_npu
except ImportError:
    pass
else:
    # Pre-import transfer_to_npu to enable CUDA→NPU compatibility monkey-patching
    # (matches common NPU migration practice where importing it is sufficient).
    #
    # NOTE: This import may have global side effects (e.g. patching torch.cuda/.cuda()).
    # Keep it best-effort and avoid hard dependency.
    try:
        from torch_npu.contrib import transfer_to_npu  # noqa: F401
    except ImportError:
        pass

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                              Core Imports                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

from mx_driving.patcher.patcher_logger import (
    patcher_logger,
    configure_patcher_logging,
    set_patcher_log_level,
)
from mx_driving.patcher.reporting import PatchResult, PatchStatus
from mx_driving.patcher.patch import (
    BasePatch,
    Patch,
    AtomicPatch,
    RegistryPatch,
    LegacyPatch,
    get_version,
    check_version,
    mmcv_version,
    is_mmcv_v1x,
    is_mmcv_v2x,
)
from mx_driving.patcher.patcher import Patcher, replace_with

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          Predefined Patch Classes                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

from mx_driving.patcher.mmcv_patch import (
    MultiScaleDeformableAttention,
    DeformConv,
    ModulatedDeformConv,
    SparseConv3D,
    Stream,
    DDP,
    OptimizerHooks,
)
from mx_driving.patcher.mmengine_patch import OptimizerWrapper
from mx_driving.patcher.mmdet_patch import (
    PseudoSampler,
    ResNetAddRelu,
    ResNetMaxPool,
    ResNetFP16,
)
from mx_driving.patcher.mmdet3d_patch import NuScenesDataset, NuScenesMetric, NuScenes
from mx_driving.patcher.numpy_patch import NumpyCompat
from mx_driving.patcher.torch_patch import TensorIndex, BatchMatmul
from mx_driving.patcher.torch_scatter_patch import TorchScatter

MSDA = MultiScaleDeformableAttention  # alias

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    Patch Registry (Single Source of Truth)                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
# Adding/removing patches here automatically updates:
# - default_patcher configuration
# - Legacy API function generation

_ALL_PATCH_CLASSES: List[Type[Patch]] = [
    # mmcv
    MultiScaleDeformableAttention, DeformConv, ModulatedDeformConv,
    SparseConv3D, Stream, DDP, OptimizerHooks,
    # mmengine
    OptimizerWrapper,
    # mmdet
    PseudoSampler, ResNetAddRelu, ResNetMaxPool, ResNetFP16,
    # numpy
    NumpyCompat,
    # mmdet3d
    NuScenesDataset, NuScenesMetric,
    # torch
    TensorIndex, BatchMatmul,
    # torch_scatter
    TorchScatter,
]

_DEFAULT_PATCH_CLASSES: List[Type[Patch]] = [
    # mmcv
    MultiScaleDeformableAttention, DeformConv, ModulatedDeformConv,
    SparseConv3D, Stream, DDP,
    # mmdet
    ResNetAddRelu, ResNetMaxPool,
    # numpy
    # Keep NumPy compatibility ahead of mmdet3d so import-time dataset/eval
    # code does not touch removed aliases (e.g. np.int) before they are restored.
    NumpyCompat,
    # mmdet3d
    NuScenesDataset, NuScenesMetric,
    # torch
    TensorIndex, BatchMatmul,
]

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                             Default Patcher                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _create_default_patcher() -> Patcher:
    """Create default patcher with all available patches."""
    patcher = Patcher()
    patcher.add(*_DEFAULT_PATCH_CLASSES)
    return patcher

default_patcher = _create_default_patcher()

"""Temporarily override mmcv version for mmdet/mmdet3d compatibility."""
def ensure_mmcv_version(expected_version: str):
    """Special hack for fixing mmcv v.s. mmdet v.s. mmdet3d compatibility."""
    import importlib
    try:
        mmcv = importlib.import_module("mmcv")
        origin_version = mmcv.__version__
        if origin_version == expected_version:
            return
        mmcv.__version__ = expected_version
        try:
            importlib.import_module("mmdet")
            importlib.import_module("mmdet3d")
        except ImportError:
            return
        finally:
            mmcv.__version__ = origin_version
    except ImportError:
        return



# ─────────────────────────────────────────────────────────────────────────────
# Legacy Name Mapping
# ─────────────────────────────────────────────────────────────────────────────

_LEGACY_NAME_TO_CLASS: Dict[str, Type[Patch]] = {
    cls.legacy_name: cls
    for cls in _ALL_PATCH_CLASSES
    if hasattr(cls, 'legacy_name') and cls.legacy_name
}

# ─────────────────────────────────────────────────────────────────────────────
# Auto-generated Legacy Patch Functions
# ─────────────────────────────────────────────────────────────────────────────

def _create_legacy_patch_func(patch_cls: Type[Patch]):
    """Create a legacy-style patch function: func(module, options) -> None"""
    def legacy_func(module, options):
        for atomic in patch_cls.patches(options):
            atomic.apply()
    legacy_func.__name__ = patch_cls.legacy_name
    return legacy_func

msda = _create_legacy_patch_func(MultiScaleDeformableAttention)
dc = _create_legacy_patch_func(DeformConv)
mdc = _create_legacy_patch_func(ModulatedDeformConv)
spconv3d = _create_legacy_patch_func(SparseConv3D)
stream = _create_legacy_patch_func(Stream)
ddp = _create_legacy_patch_func(DDP)
optimizer_hooks = _create_legacy_patch_func(OptimizerHooks)
optimizer_wrapper = _create_legacy_patch_func(OptimizerWrapper)
index = _create_legacy_patch_func(TensorIndex)
batch_matmul = _create_legacy_patch_func(BatchMatmul)
numpy_type = _create_legacy_patch_func(NumpyCompat)
pseudo_sampler = _create_legacy_patch_func(PseudoSampler)
resnet_add_relu = _create_legacy_patch_func(ResNetAddRelu)
resnet_maxpool = _create_legacy_patch_func(ResNetMaxPool)
resnet_fp16 = _create_legacy_patch_func(ResNetFP16)
nuscenes_dataset = _create_legacy_patch_func(NuScenesDataset)
nuscenes_metric = _create_legacy_patch_func(NuScenesMetric)
scatter = _create_legacy_patch_func(TorchScatter)


def patch_mmcv_version(expected_version: str):
    ensure_mmcv_version(expected_version)

# ─────────────────────────────────────────────────────────────────────────────
# Legacy Builder Classes
# ─────────────────────────────────────────────────────────────────────────────

from mx_driving.patcher.legacy import (
    LegacyPatchWrapper,
    LegacyPatcherBuilder,
    LegacyPatcherBuilder as PatcherBuilder,
    default_patcher_builder,
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                              Public API                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

__all__ = [
    # ── Core ──────────────────────────────────────────────────────────────────
    "Patcher", "BasePatch", "Patch", "AtomicPatch", "RegistryPatch", "LegacyPatch",
    "replace_with",
    # ── Logger ────────────────────────────────────────────────────────────────
    "patcher_logger", "configure_patcher_logging", "set_patcher_log_level",
    # ── Reporting ─────────────────────────────────────────────────────────────
    "PatchStatus", "PatchResult",
    # ── Version ───────────────────────────────────────────────────────────────
    "get_version", "check_version", "mmcv_version", "is_mmcv_v1x", "is_mmcv_v2x",
    # ── Default Patcher ───────────────────────────────────────────────────────
    "default_patcher",
    # ── Predefined Patches ────────────────────────────────────────────────────
    "MultiScaleDeformableAttention", "MSDA", "DeformConv", "ModulatedDeformConv",
    "SparseConv3D", "Stream", "DDP", "OptimizerHooks", "OptimizerWrapper",
    "PseudoSampler", "ResNetAddRelu", "ResNetMaxPool", "ResNetFP16",
    "NuScenesDataset", "NuScenesMetric", "NuScenes",
    "NumpyCompat", "TensorIndex", "BatchMatmul", "TorchScatter",
    # ── Legacy API ────────────────────────────────────────────────────────────
    "PatcherBuilder", "LegacyPatcherBuilder", "LegacyPatchWrapper",
    "default_patcher_builder",
    "msda", "dc", "mdc", "spconv3d", "patch_mmcv_version", "ensure_mmcv_version", "stream", "ddp",
    "optimizer_hooks", "optimizer_wrapper", "index", "batch_matmul", "numpy_type",
    "pseudo_sampler", "resnet_add_relu", "resnet_maxpool", "resnet_fp16",
    "nuscenes_dataset", "nuscenes_metric", "scatter",
    # ── Internal ──────────────────────────────────────────────────────────────
    "_ALL_PATCH_CLASSES", "_DEFAULT_PATCH_CLASSES", "_LEGACY_NAME_TO_CLASS",
]
