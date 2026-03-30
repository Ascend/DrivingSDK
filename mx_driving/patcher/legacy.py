# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Legacy compatibility layer for the old patcher API (pre-87ec1895).

This module provides backward compatibility for users who have code written
using the old patcher API style:

Old API (before refactoring):
    from mx_driving.patcher import PatcherBuilder, Patch
    from mx_driving.patcher import batch_matmul, msda

    patcher_builder = (
        PatcherBuilder()
        .add_module_patch("mmcv", Patch(msda))
        .add_module_patch("torch", Patch(batch_matmul))
    )

    with patcher_builder.build() as patcher:
        # train model

This module is a pure interface mediator - it translates old API calls to
the new Patcher implementation. All actual logic is in patcher.py.

Note: The `Patch` class in patch.py has been enhanced with a metaclass that
automatically detects old-style usage (Patch(func)) and returns LegacyPatchWrapper.

IMPORTANT: This module should NOT define any patch functions or mappings.
All patch definitions are in __init__.py (_ALL_PATCH_CLASSES, _LEGACY_NAME_TO_CLASS).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set

from mx_driving.patcher.patch import LegacyPatch
from mx_driving.patcher.patcher import Patcher
from mx_driving.patcher.patcher_logger import patcher_logger


__all__ = [
    # Core classes
    "LegacyPatchWrapper",
    "LegacyPatcherBuilder",
    "PatcherBuilder",  # alias for backward compatibility
    "default_patcher_builder",
]


# =============================================================================
# LegacyPatchWrapper - Core wrapper for old-style patch functions
# =============================================================================

class LegacyPatchWrapper:
    """
    Wrapper for old-style patch functions.

    This class mimics the old Patch class API where a patch was simply
    a wrapper around a function with signature: func(module, options).

    Args:
        func: Callable with signature (module, options) -> None
        options: Optional dict passed to func when applying
        priority: Priority for sorting patches (lower = earlier)
        patch_failure_warning: Whether to warn on patch failure

    Example:
        def my_patch(module, options):
            module.some_attr = new_value

        patch = LegacyPatchWrapper(my_patch)
        # Or using the Patch alias:
        patch = Patch(my_patch)  # Automatically returns LegacyPatchWrapper
    """

    _migration_warning_shown = False

    def __init__(
        self,
        func: Callable,
        options: Optional[Dict] = None,
        priority: int = 0,
        patch_failure_warning: bool = True,
    ):
        self.func = func
        self.name = func.__name__
        self.options = options if options is not None else {}
        self.priority = priority
        self.is_applied = False
        self.patch_failure_warning = patch_failure_warning

        # Show migration warning once per session
        if not LegacyPatchWrapper._migration_warning_shown:
            patcher_logger.info(
                "Using legacy patcher API (Patch(func)). "
                "This is a compatibility-layer notice, not a patch failure. "
                "Consider migrating to the new Patch class API:\n"
                "  class MyPatch(Patch):\n"
                "      @classmethod\n"
                "      def patches(cls, options=None):\n"
                "          return [AtomicPatch('target.path', replacement)]\n"
                "  MyPatch.apply_all()  # or patcher.add(MyPatch).apply()"
            )
            LegacyPatchWrapper._migration_warning_shown = True

    def __lt__(self, other):
        """Support sorting by priority."""
        return self.priority < other.priority

    def __repr__(self) -> str:
        return f"LegacyPatchWrapper({self.name})"


# =============================================================================
# LegacyPatcherBuilder - Interface mediator to new Patcher
# =============================================================================

class LegacyPatcherBuilder:
    """
    Builder class that provides the old PatcherBuilder API.

    This is a pure interface mediator - it collects configuration and
    delegates all actual work to the new Patcher class.

    Example:
        builder = LegacyPatcherBuilder()
        builder.add_module_patch("mmcv", Patch(my_patch))
        builder.add_module_patch("torch", Patch(another_patch))

        with builder.build() as patcher:
            # train model
    """

    def __init__(self):
        # Store configuration only - no actual logic
        self._module_patches: Dict[str, List[LegacyPatchWrapper]] = {}
        self._blacklist: Set[str] = set()
        self._profiling_options: Optional[Dict] = None
        self._brake_step: Optional[int] = None

    def add_module_patch(self, module_name: str, *patches: LegacyPatchWrapper) -> 'LegacyPatcherBuilder':
        """
        Add patches for a specific module.

        Args:
            module_name: The target module name (e.g., "mmcv", "torch")
            patches: LegacyPatchWrapper instances to add

        Returns:
            self for method chaining
        """
        if module_name not in self._module_patches:
            self._module_patches[module_name] = []
        self._module_patches[module_name].extend(patches)
        self._module_patches[module_name].sort()
        return self

    def disable_patches(self, *patch_names: str) -> 'LegacyPatcherBuilder':
        """
        Disable patches by name.

        Args:
            patch_names: Names of patches to disable

        Returns:
            self for method chaining
        """
        self._blacklist.update(patch_names)
        return self

    def with_profiling(
        self,
        path: str,
        level: int = 0,
        skip_first: int = 20,
        wait: int = 1,
        warmup: int = 1,
        active: int = 1,
        repeat: int = 1,
    ) -> 'LegacyPatcherBuilder':
        """
        Configure profiling for training loops.

        Args:
            path: Output path for profiling data
            level: Profiling level (0-2)
            skip_first: Number of initial steps to skip
            wait: Wait steps in schedule
            warmup: Warmup steps in schedule
            active: Active steps in schedule
            repeat: Repeat count in schedule

        Returns:
            self for method chaining
        """
        self._profiling_options = {
            'path': path,
            'level': level,
            'skip_first': skip_first,
            'wait': wait,
            'warmup': warmup,
            'active': active,
            'repeat': repeat,
        }
        return self

    def brake_at(self, brake_step: int) -> 'LegacyPatcherBuilder':
        """
        Configure early stopping at a specific training step.

        Args:
            brake_step: Step number to stop at

        Returns:
            self for method chaining
        """
        self._brake_step = brake_step
        return self

    def build(self, allow_internal_format: bool = False) -> '_LegacyPatcher':
        """
        Build and return a Patcher instance.

        This method creates a new Patcher, translates all old-style patches
        to new-style patches, and returns a _LegacyPatcher wrapper.

        Args:
            allow_internal_format: Whether to allow NPU internal format

        Returns:
            A _LegacyPatcher instance that wraps the new Patcher
        """
        # Create new Patcher instance
        patcher = Patcher()
        from mx_driving.patcher import _LEGACY_NAME_TO_CLASS

        # Translate old-style patches while preserving 1.0 insertion order.
        # Built-in legacy functions expand to their 2.0 child patches so summary
        # output can keep detailed target paths. Custom legacy functions remain
        # LegacyPatch instances for backward compatibility.
        for module_name, patches in self._module_patches.items():
            for wrapper in patches:
                if wrapper.name in self._blacklist:
                    continue

                patch_cls = _LEGACY_NAME_TO_CLASS.get(wrapper.name)
                if patch_cls is not None:
                    for patch in patch_cls.patches(wrapper.options or None):
                        patch._parent_name = patch_cls.name
                        patcher.add(patch)
                    continue

                patcher.add(LegacyPatch(
                    wrapper.func,
                    target_module=module_name,
                    options=wrapper.options,
                ))

        # Apply blacklist to new patcher (for patches added by name)
        for name in self._blacklist:
            patcher.disable(name)

        # Configure profiling if set
        if self._profiling_options:
            patcher.with_profiling(**self._profiling_options)

        # Configure brake if set
        if self._brake_step is not None:
            patcher.brake_at(self._brake_step)

        return _LegacyPatcher(patcher, allow_internal_format)


# =============================================================================
# _LegacyPatcher - Interface mediator wrapping new Patcher
# =============================================================================

class _LegacyPatcher:
    """
    Internal patcher class that provides the old Patcher API.

    This is a pure interface mediator - it wraps the new Patcher and
    translates old API calls to new API calls.
    """

    def __init__(self, patcher: Patcher, allow_internal_format: bool = False):
        self._patcher = patcher
        self._allow_internal_format = allow_internal_format

    def transfer_to_npu(self):
        """Transfer to NPU (called automatically in __enter__)."""
        # Delegate to Patcher._transfer_to_npu via apply()
        # The new Patcher handles this internally
        pass

    def apply(self):
        """Apply all registered patches - delegates to new Patcher."""
        self._patcher.apply(allow_internal_format=self._allow_internal_format)

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @property
    def is_applied(self) -> bool:
        """Check if patches have been applied."""
        return self._patcher.is_applied


# =============================================================================
# Default patcher builder - delegates to default_patcher from __init__.py
# =============================================================================

class _DefaultPatcherBuilderProxy:
    """
    Proxy class that provides the old default_patcher_builder API.

    This proxy delegates to the default_patcher from __init__.py,
    translating old API calls to new API calls.

    IMPORTANT: This class automatically mirrors the configuration from
    _DEFAULT_PATCH_CLASSES in __init__.py. No manual synchronization needed.
    """

    def __init__(self):
        self._builder: Optional[LegacyPatcherBuilder] = None

    def _get_builder(self) -> LegacyPatcherBuilder:
        """Lazily create a builder that mirrors default_patcher configuration."""
        if self._builder is None:
            self._builder = LegacyPatcherBuilder()

            # Import patch classes directly from their modules to avoid
            # triggering mx_driving/__init__.py which requires torch.
            # This list should match _DEFAULT_PATCH_CLASSES in __init__.py
            from mx_driving.patcher.mmcv_patch import (
                MultiScaleDeformableAttention,
                DeformConv,
                ModulatedDeformConv,
                SparseConv3D,
                Stream,
                DDP,
            )
            from mx_driving.patcher.mmdet_patch import ResNetAddRelu, ResNetMaxPool
            from mx_driving.patcher.mmdet3d_patch import NuScenesDataset, NuScenesMetric
            from mx_driving.patcher.numpy_patch import NumpyCompat
            from mx_driving.patcher.torch_patch import TensorIndex, BatchMatmul
            from mx_driving.patcher.patch import Patch

            # Default patch classes - must match _DEFAULT_PATCH_CLASSES in __init__.py
            default_classes = [
                MultiScaleDeformableAttention,
                DeformConv,
                ModulatedDeformConv,
                SparseConv3D,
                Stream,
                DDP,
                ResNetAddRelu,
                ResNetMaxPool,
                NumpyCompat,
                NuScenesDataset,
                NuScenesMetric,
                TensorIndex,
                BatchMatmul,
            ]

            # Create legacy functions that delegate to new Patch classes
            def _create_legacy_func(patch_cls):
                def legacy_func(module, options):
                    for atomic in patch_cls.patches(options):
                        atomic.apply()
                legacy_func.__name__ = patch_cls.legacy_name
                return legacy_func

            # Group patches by target_module and add them
            patches_by_module: Dict[str, List] = {}
            for patch_cls in default_classes:
                target_module = getattr(patch_cls, 'target_module', None)
                legacy_name = getattr(patch_cls, 'legacy_name', None)
                if target_module and legacy_name:
                    legacy_func = _create_legacy_func(patch_cls)
                    if target_module not in patches_by_module:
                        patches_by_module[target_module] = []
                    patches_by_module[target_module].append(Patch(legacy_func))

            # Add patches to builder
            for module_name, patches in patches_by_module.items():
                self._builder.add_module_patch(module_name, *patches)

        return self._builder

    def add_module_patch(self, module_name: str, *patches) -> '_DefaultPatcherBuilderProxy':
        """Add patches for a specific module."""
        self._get_builder().add_module_patch(module_name, *patches)
        return self

    def disable_patches(self, *patch_names: str) -> '_DefaultPatcherBuilderProxy':
        """Disable patches by name."""
        self._get_builder().disable_patches(*patch_names)
        return self

    def with_profiling(self, path: str, **kwargs) -> '_DefaultPatcherBuilderProxy':
        """Configure profiling for training loops."""
        self._get_builder().with_profiling(path, **kwargs)
        return self

    def brake_at(self, brake_step: int) -> '_DefaultPatcherBuilderProxy':
        """Configure early stopping at a specific training step."""
        self._get_builder().brake_at(brake_step)
        return self

    def build(self, allow_internal_format: bool = False) -> _LegacyPatcher:
        """Build and return a Patcher instance."""
        return self._get_builder().build(allow_internal_format)

    def __repr__(self):
        return "<DefaultPatcherBuilderProxy>"


default_patcher_builder = _DefaultPatcherBuilderProxy()

# Alias for backward compatibility
PatcherBuilder = LegacyPatcherBuilder
