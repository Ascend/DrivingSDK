# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Patcher: lightweight monkey-patch framework for NPU adaptation.

================================================================================
Quick Start
================================================================================

    from mx_driving.patcher import default_patcher
    default_patcher.apply()

    # Or with customization
    from mx_driving.patcher import default_patcher, MultiScaleDeformableAttention, BatchMatmul
    default_patcher.disable(MultiScaleDeformableAttention.name, BatchMatmul.name).apply()

================================================================================
Class Hierarchy
================================================================================

    BasePatch (ABC)
    ├── AtomicPatch      - Single attribute replacement (declarative)
    ├── RegistryPatch    - Register classes to mmcv/mmengine Registry
    ├── Patch            - Composite patch, predefined patches inherit this
    └── LegacyPatch      - Function-based patch (backward compatibility)

================================================================================
AtomicPatch - Single Attribute Replacement
================================================================================

Basic usage:
    AtomicPatch("mmcv.ops.msda.forward", npu_forward)
    AtomicPatch("numpy.bool", bool)

With precheck (validate before patching):
    # precheck can accept no argument or patch parameters - auto-detected
    AtomicPatch("mmcv.ops.func", npu_impl, precheck=lambda: is_mmcv_v1x())
    AtomicPatch("mmcv.ops.func", npu_impl, precheck=lambda target: target.startswith("mmcv"))

With runtime_check (conditional dispatch at call time):
    AtomicPatch(
        "torch.matmul",
        npu_matmul,
        runtime_check=lambda a, b: a.dim() >= 4,
    )

With aliases (handle re-export paths):
    AtomicPatch(
        "mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction.forward",
        npu_forward,
        aliases=["mmcv.ops.MultiScaleDeformableAttnFunction.forward"],
    )

With string replacement (lazy resolve):
    AtomicPatch(
        "mmcv.parallel.distributed.MMDistributedDataParallel",
        "mmcv.device.npu.NPUDistributedDataParallel",  # resolved at apply time
    )

================================================================================
Patch - Predefined Composite Patch
================================================================================

Define a Patch:

    class MultiScaleDeformableAttention(Patch):
        '''Multi-Scale Deformable Attention NPU patch'''

        name = "multi_scale_deformable_attention"

        @staticmethod
        def precheck() -> bool:
            return mmcv_version.is_v2x

        # Use @with_imports to lazily import modules
        # String form imports the whole module (most common usage)
        @staticmethod
        @with_imports("torch_npu")
        def forward(ctx, value, spatial_shapes, level_start_index,
                    sampling_locations, attention_weights, im2col_step=None):
            return torch_npu.npu_msda_forward(...)  # noqa: F821

        @classmethod
        def patches(cls, options=None) -> List[AtomicPatch]:
            base = "mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction"
            return [
                AtomicPatch(f"{base}.forward", cls.forward, precheck=cls.precheck),
                AtomicPatch(f"{base}.backward", cls.backward, precheck=cls.precheck),
            ]

With options:

    class MyPatch(Patch):
        name = "my_patch"

        @classmethod
        def patches(cls, options=None) -> List[AtomicPatch]:
            use_fast = options.get('use_fast', True) if options else True
            return [
                AtomicPatch("module.func", cls.fast_impl if use_fast else cls.slow_impl),
            ]

    patcher.add(MyPatch, options={'use_fast': False})

================================================================================
RegistryPatch - Register to Registry
================================================================================

For patches that register classes to mmcv/mmengine registries:

    RegistryPatch(
        "mmcv.runner.hooks.optimizer.HOOKS",
        name="OptimizerHook",
        module_factory=create_optimizer_hook,
        precheck=lambda: mmcv_version.is_v1x,
    )

================================================================================
LegacyPatch - Backward Compatibility
================================================================================

For existing function-based patches:

    def my_patch(module, options):
        module.ops.func = my_replacement

    LegacyPatch(my_patch, target_module="mmcv", options={"key": "value"})

================================================================================
Patcher API
================================================================================

Create and configure:

    patcher = Patcher()

    # Add predefined Patches
    patcher.add(MultiScaleDeformableAttention, DeformConv, SparseConv3D)

    # Add Patch with options
    patcher.add(MyPatch, options={'use_fast': False})

    # Add AtomicPatch directly
    patcher.add(
        AtomicPatch("torch.matmul", npu_matmul),
        AtomicPatch("numpy.bool", bool),
    )

    # Chain calls
    patcher.add(MultiScaleDeformableAttention).add(DeformConv).add(AtomicPatch("torch.bmm", npu_bmm))

    # Legacy patch
    patcher.add(LegacyPatch(my_old_patch_func))

Disable patches:

    # By Patch class (recommended in 2.0)
    patcher.disable(MultiScaleDeformableAttention)

    # By Patch name (recommended)
    patcher.disable(MultiScaleDeformableAttention.name)
    patcher.disable(SparseConv3D.name, BatchMatmul.name)

    # By AtomicPatch target
    patcher.disable("mmcv.ops.multi_scale_deform_attn...forward")

Print info:

    patcher.print_info()            # Basic info
    patcher.print_info(diff=True)   # With source code diff

Apply:

    patcher.apply()

    # Or as context manager
    with patcher:
        train()

================================================================================
Default Patcher
================================================================================

    from mx_driving.patcher import default_patcher

    # Basic usage
    default_patcher.apply()

    # Disable specific patches
    from mx_driving.patcher import MultiScaleDeformableAttention, BatchMatmul
    default_patcher.disable(MultiScaleDeformableAttention.name, BatchMatmul.name).apply()

    # With profiling
    default_patcher.with_profiling("/output/prof").apply()

    # With early stopping
    default_patcher.brake_at(100).apply()

    # Combine options
    (default_patcher
        .disable(MultiScaleDeformableAttention.name)
        .with_profiling("/output/prof", level=1)
        .brake_at(500)
        .apply())

================================================================================
"""
from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from mx_driving.patcher.patch import (
    AtomicPatch,
    BasePatch,
    LegacyPatch,
    Patch,
    mmcv_version,
)
from mx_driving.patcher.patcher_logger import patcher_logger
from mx_driving.patcher.reporting import PatchStatus


# =============================================================================
# Patcher
# =============================================================================

class Patcher:
    """
    Lightweight monkey-patch manager.

    Patches are stored by module and applied in insertion order.
    """

    @staticmethod
    def _is_patch_class(item: Any) -> bool:
        """Check whether an object behaves like a Patch subclass."""
        if not isinstance(item, type):
            return False
        if issubclass(item, Patch):
            return True
        return (
            callable(getattr(item, "patches", None))
            and hasattr(item, "name")
            and callable(getattr(item, "apply_all", None))
        )

    def __init__(self):
        self._entries: List[tuple] = []  # [(kind, item, options)] - unified timeline
        self._blacklist: Set[str] = set()
        self._applied = False
        self._profiling_options: Optional[Dict] = None
        self._brake_step: Optional[int] = None
        self._collected_patches: Optional[List[Tuple[BasePatch, str]]] = None  # Cache for collected patches
        self._resolved_class_patches: Dict[int, List[BasePatch]] = {}  # Per-entry child patch cache
        self._skipped_modules: List[str] = []  # Track skipped modules for logging
        self._injected_imports: List[tuple] = []  # Track injected imports (source, name, target)
        self._allow_internal_format: Optional[bool] = None  # None means use default (False)

    def _invalidate_collection_cache(self) -> None:
        """Invalidate cached child patches/collection when entries or blacklist change."""
        self._collected_patches = None
        self._resolved_class_patches = {}

    def _iter_enabled_patch_classes(self):
        """Yield enabled Patch classes already added to this patcher."""
        for kind, item, _ in self._entries:
            if kind == "class" and item.name not in self._blacklist:
                yield item

    @staticmethod
    def _build_conflict_guidance(preferred_patch: Type[Patch], other_patch_name: str, other_patch_class: Optional[Type[Patch]] = None) -> str:
        """Build a simple, actionable fix message for a conflict."""
        lines = [
            "If one of them comes from default_patcher, disable the default patch you do not want first, then add the patch you want.",
            "Recommended fix:",
            f"  patcher.disable('{other_patch_name}').add({preferred_patch.__name__})",
        ]
        if other_patch_class is not None:
            lines.extend([
                "Or keep the other patch instead:",
                f"  patcher.disable('{preferred_patch.name}').add({other_patch_class.__name__})",
            ])
        return "\n".join(lines)

    def _warn_conflicting_add(self, new_patch: Type[Patch]) -> None:
        """Warn early when a newly-added Patch conflicts with an already-enabled Patch."""
        new_conflicts = set(getattr(new_patch, "conflicts_with", None) or [])
        for existing_patch in self._iter_enabled_patch_classes():
            if existing_patch.name == new_patch.name:
                continue
            existing_conflicts = set(getattr(existing_patch, "conflicts_with", None) or [])
            if existing_patch.name not in new_conflicts and new_patch.name not in existing_conflicts:
                continue
            patcher_logger.warning(
                f"Patch '{new_patch.name}' conflicts with already-enabled patch '{existing_patch.name}'.\n"
                f"{self._build_conflict_guidance(new_patch, existing_patch.name, existing_patch)}"
            )

    def add(self, *items: Union[Type[Patch], BasePatch], options: Optional[Dict] = None) -> 'Patcher':
        """
        Add patches.

        Args:
            items: Patch classes or BasePatch instances.
            options: Optional configuration dict for Patch classes.

        Examples:
            patcher.add(MultiScaleDeformableAttention, DeformConv)
            patcher.add(MyPatch, options={'use_fast': False})
            patcher.add(AtomicPatch("torch.matmul", npu_matmul))
            patcher.add(MultiScaleDeformableAttention).add(DeformConv).add(AtomicPatch(...))
        """
        for item in items:
            if self._is_patch_class(item):
                self._warn_conflicting_add(item)
                self._entries.append(("class", item, options))
            elif isinstance(item, BasePatch):
                self._entries.append(("direct", item, None))
            else:
                raise TypeError(f"Expected Patch class or BasePatch instance, got {type(item)}")
        self._invalidate_collection_cache()
        return self

    @staticmethod
    def _normalize_patch_name(item: Union[str, Type[Patch], BasePatch]) -> str:
        """Normalize a patch identifier to its stable string name."""
        if isinstance(item, str):
            return item
        if Patcher._is_patch_class(item):
            return item.name
        if isinstance(item, BasePatch):
            return item.name
        raise TypeError(f"Expected patch name, Patch class, or BasePatch instance, got {type(item)}")

    def disable(self, *names: Union[str, Type[Patch], BasePatch]) -> 'Patcher':
        """
        Disable patches by name.

        Examples:
            patcher.disable(MultiScaleDeformableAttention)
            patcher.disable(MultiScaleDeformableAttention.name)
            patcher.disable(SparseConv3D.name, BatchMatmul.name)
            patcher.disable("mmcv.ops.some_func")
        """
        self._blacklist.update(self._normalize_patch_name(name) for name in names)
        self._collected_patches = None
        return self

    def _resolve_class_patches(
        self,
        entry_index: int,
        patch_cls: Type[Patch],
        options: Optional[Dict],
    ) -> List[BasePatch]:
        """Resolve child patches for a Patch entry once and reuse the same instances."""
        cached = self._resolved_class_patches.get(entry_index)
        if cached is not None:
            return cached
        patches = list(patch_cls.patches(options))
        self._resolved_class_patches[entry_index] = patches
        return patches

    def _apply_single_patch(self, patch: BasePatch, parent_name: str) -> None:
        """Apply/log a single patch instance."""
        package = patch.module or ""

        if patch.name in self._blacklist:
            patcher_logger.skipped(
                patch.name,
                reason="disabled",
                package=package,
                patch_name=parent_name,
            )
            return

        if patch.is_applied:
            return

        try:
            result = patch.apply()
            if result.status == PatchStatus.APPLIED:
                patcher_logger.applied(
                    patch.name,
                    package=package,
                    patch_name=parent_name,
                )
            elif result.status == PatchStatus.SKIPPED:
                patcher_logger.skipped(
                    patch.name,
                    reason=result.reason or "",
                    package=package,
                    patch_name=parent_name,
                )
            else:
                patcher_logger.failed(
                    patch.name,
                    reason=result.reason or "",
                    package=package,
                    patch_name=parent_name,
                )
        except Exception as e:
            patcher_logger.failed(
                patch.name,
                reason=str(e),
                package=package,
                patch_name=parent_name,
            )

    def _apply_bootstrap_patches(self) -> None:
        """
        Apply opt-in compatibility patches before collecting later Patch classes.

        This is reserved for low-risk environment/bootstrap fixes such as NumPy
        alias restoration, where later Patch.patches() implementations may
        import third-party modules during collection.
        """
        for entry_index, (kind, item, options) in enumerate(self._entries):
            if kind != "class":
                continue
            if item.name in self._blacklist:
                continue
            if not getattr(item, "apply_before_collect", False):
                continue

            for patch in self._resolve_class_patches(entry_index, item, options):
                self._apply_single_patch(patch, item.name)

    def skip_import(self, *module_paths: str) -> 'Patcher':
        """
        Skip importing specified modules by registering stubs in sys.modules.

        Use this to block CUDA-specific dependencies that are not available on NPU.
        When Python tries to import these modules, it will find the stub instead
        of raising ImportError.

        IMPORTANT: Call this BEFORE any code that imports these modules.

        Args:
            module_paths: Module paths to skip (e.g., "flash_attn", "torch_scatter")

        Examples:
            # Block flash_attn (GPU-only package)
            patcher.skip_import("flash_attn")

            # Block multiple modules
            patcher.skip_import(
                "flash_attn",
                "torch_scatter",
                "spconv",
            )

            # The stub handles any attribute access automatically:
            # from flash_attn.bert_padding import unpad_input  # Works, returns stub
            # flash_attn.some_function()  # Works, returns None
        """
        for module_path in module_paths:
            if module_path in sys.modules:
                patcher_logger.debug(f"Module {module_path} already imported, skip_import has no effect")
                continue

            _register_stub_module(module_path)
            self._skipped_modules.append(module_path)
            patcher_logger.add_skipped_module(module_path)
            patcher_logger.debug(f"Registered stub for: {module_path}")

        return self

    def inject_import(
        self,
        source_module: str,
        class_name: str,
        target_module: str,
    ) -> 'Patcher':
        """
        Inject a missing import from one module into another.

        Use this to fix missing imports in third-party code where a class/function
        is defined in one module but not properly exported in the parent module's __all__.

        IMPORTANT: Call this BEFORE any code that imports from the target module.
        The injection is executed immediately when this method is called.

        Args:
            source_module: The module path where the class is defined
            class_name: The name of the class/function to inject
            target_module: The module path where the class should be accessible

        Examples:
            # Make V1SparseDrive accessible from projects.mmdet3d_plugin.models
            patcher.inject_import(
                "projects.mmdet3d_plugin.models.sparsedrive_v1",
                "V1SparseDrive",
                "projects.mmdet3d_plugin.models",
            )
        """
        import importlib

        try:
            # Import source module and get the class
            src_mod = importlib.import_module(source_module)
            cls = getattr(src_mod, class_name, None)
            if cls is None:
                patcher_logger.debug(f"Class {class_name} not found in {source_module}")
                return self

            # Import target module and inject the class
            tgt_mod = importlib.import_module(target_module)
            # Use vars() instead of hasattr() to avoid _StubModule.__getattr__
            # which always returns a new stub for non-private names
            if class_name not in vars(tgt_mod):
                setattr(tgt_mod, class_name, cls)
                # Also add to __all__ if it exists
                if hasattr(tgt_mod, '__all__'):
                    try:
                        all_ = tgt_mod.__all__
                        if class_name not in all_:
                            if isinstance(all_, tuple):
                                tgt_mod.__all__ = all_ + (class_name,)
                            elif isinstance(all_, set):
                                all_.add(class_name)
                            else:
                                all_.append(class_name)
                    except Exception as e:
                        patcher_logger.debug(f"Failed to update {target_module}.__all__: {e}")
                patcher_logger.add_injected_import(f"{target_module}.{class_name}")
                patcher_logger.debug(f"Injected {class_name} into {target_module}")
                self._injected_imports.append((source_module, class_name, target_module))
        except ImportError as e:
            patcher_logger.debug(f"Failed to inject {class_name}: {e}")

        return self

    def replace_import(
        self,
        module_path: str,
        replacement: Optional[str] = None,
        *,
        base_module: Optional[str] = None,
        exports: Optional[Dict[str, Any]] = None,
        **attrs,
    ) -> 'Patcher':
        """
        Replace a module import with another module or custom attributes.

        When code tries to import `module_path`, it will get the replacement instead.

        IMPORTANT: Call this BEFORE any code that imports this module.
        If the module is already in sys.modules, the replacement will be skipped
        and a warning will be logged with guidance to move replace_import()
        earlier in startup.

        Args:
            module_path: The module path to replace (e.g., "cuda_ops.special_op")
            replacement: Replacement module path string. Recommended for the
                        simplest whole-module replacement case:
                        replace_import("old.module", "new.module").
            base_module: Explicit keyword form for the replacement module path.
                        Cannot be used
                        together with `replacement`.
            exports: Dict of {name: object} to set on the replacement module.
                    Can be used alone or combined with base_module.
            **attrs: Custom attributes to set on the replacement module.
                    Equivalent to exports but as keyword arguments.

        Examples:
            # Replace entire module with another (default/simple form)
            patcher.replace_import("old.module", "new.module")

            # Replace with custom exports
            patcher.replace_import(
                "projects.ops.deformable_aggregation",
                exports={"DeformableAggregationFunction": MyWrapper},
            )

            # Replace module and override specific attributes
            patcher.replace_import(
                "old.module",
                base_module="new.module",
                exports={"SpecialClass": CustomImpl},
            )

            # Using replace_with helper for export-oriented forms
            from mx_driving.patcher import replace_with
            patcher.replace_import("old.module", replace_with.module(Foo=Bar))
            patcher.replace_import("old.module", replace_with.module("new.module", Foo=Bar))

            # Explicit keyword form (also supported)
            patcher.replace_import("old.module", base_module="new.module")

            # Compatibility form
            patcher.replace_import("old.module", DeformableAggregationFunction=MyWrapper)
        """
        if module_path in sys.modules:
            patcher_logger.warning(
                f"replace_import skipped for {module_path}: module is already loaded in "
                "sys.modules. Call replace_import() before importing the target module."
            )
            return self

        # --- Resolve replacement spec ---
        # Handle replace_with.module() spec objects
        resolved_base = None
        resolved_exports: Dict[str, Any] = {}

        if replacement is not None and isinstance(replacement, _ImportReplacementSpec):
            # replacement is actually a spec object from replace_with.module(...)
            resolved_base = replacement.base_module
            resolved_exports.update(replacement.exports)
            replacement = None  # clear so it doesn't conflict below

        # Validate: replacement and base_module are mutually exclusive
        if replacement is not None and base_module is not None:
            raise ValueError(
                "Cannot specify both 'replacement' (positional) and 'base_module' (keyword). "
                "Use only one of them."
            )

        # Merge base_module sources
        resolved_base = resolved_base or base_module or replacement

        # Merge exports: explicit exports dict + legacy **attrs
        if exports:
            overlap = set(exports.keys()) & set(attrs.keys())
            if overlap:
                raise ValueError(f"Conflicting keys in exports and **attrs: {overlap}")
            resolved_exports.update(exports)
        resolved_exports.update(attrs)

        # --- Build the replacement module ---
        if resolved_base:
            import importlib
            try:
                src_module = importlib.import_module(resolved_base)
                module = ModuleType(module_path)
                # Snapshot copy, but preserve target identity metadata
                module.__dict__.update(src_module.__dict__)
                module.__name__ = module_path
                module.__package__ = module_path.rsplit('.', 1)[0] if '.' in module_path else module_path
            except ImportError as e:
                patcher_logger.warning(f"Failed to import replacement module {resolved_base}: {e}")
                module = ModuleType(module_path)
        else:
            module = ModuleType(module_path)

        # Apply exports/attrs
        for name, value in resolved_exports.items():
            setattr(module, name, value)

        # Register leaf first so parent package imports can resolve it if needed.
        sys.modules[module_path] = module

        # --- Register module with parent package chain ---
        _register_parent_packages(module_path)

        # Link to parent package
        if '.' in module_path:
            parent_path, child_name = module_path.rsplit('.', 1)
            parent = sys.modules.get(parent_path)
            if parent is not None:
                setattr(parent, child_name, module)

        # --- Logging ---
        desc = module_path
        if resolved_base:
            desc += f" -> {resolved_base}"
        patcher_logger.add_replaced_import(desc)
        patcher_logger.debug(f"Replaced import: {desc}")

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
    ) -> 'Patcher':
        """Configure profiling for training loops."""
        self._profiling_options = {
            'enable_profiler': True,
            'profiling_path': path,
            'profiling_level': level,
            'step_ctrl': (wait, warmup, active, repeat, skip_first),
        }
        return self

    def brake_at(self, step: int) -> 'Patcher':
        """Configure early stopping at a specific training step."""
        self._brake_step = step
        return self

    def _set_npu_internal_format(self, allow: bool) -> None:
        """Best-effort set `torch.npu.config.allow_internal_format`."""
        try:
            import torch  # noqa: F401
            import torch_npu  # noqa: F401

            torch.npu.config.allow_internal_format = allow
            patcher_logger.debug(f"Set torch.npu.config.allow_internal_format={allow}")
        except ImportError:
            patcher_logger.debug("torch_npu not available, cannot set allow_internal_format")
        except AttributeError as e:
            patcher_logger.debug(f"Failed to set allow_internal_format due to attribute error: {e}")

    def allow_internal_format(self, allow: bool = True) -> 'Patcher':
        """
        Configure whether to allow NPU internal format.

        NPU internal format can improve performance but may cause compatibility issues
        with some operations. By default, internal format is disabled (False).

        Args:
            allow: If True, allow NPU internal format. Default is True when this
                   method is called.

        Returns:
            self for method chaining.

        Examples:
            # Enable internal format
            patcher.allow_internal_format().apply()

            # Explicitly disable internal format
            patcher.allow_internal_format(False).apply()
        """
        self._allow_internal_format = allow
        # Apply immediately if torch_npu is available (also useful when called after apply()).
        self._set_npu_internal_format(allow)
        return self

    def disallow_internal_format(self) -> 'Patcher':
        """
        Disable NPU internal format.

        This is a convenience method equivalent to allow_internal_format(False).

        Returns:
            self for method chaining.

        Examples:
            patcher.disallow_internal_format().apply()
        """
        return self.allow_internal_format(False)

    def print_info(self, diff: bool = False) -> 'Patcher':
        """
        Print structured patch information grouped by module.

        Args:
            diff: If True, show source code diff between original and replacement.
        """
        # Collect all patches
        all_patches = self._collect_all_patches()

        # Group by module
        by_module: Dict[str, List[BasePatch]] = {}
        for patch, _ in all_patches:
            module = patch.module or "other"
            if module not in by_module:
                by_module[module] = []
            by_module[module].append(patch)

        total_patches = len(all_patches)
        total_modules = len(by_module)

        lines = [
            "=" * 80,
            f"Patcher Info ({total_modules} modules, {total_patches} patches)",
            "=" * 80,
            "",
        ]

        for module_name, patches in by_module.items():
            lines.append(f"[{module_name}] {len(patches)} patches")

            for patch in patches:
                status = "[applied]" if patch.is_applied else "[pending]"
                if patch.name in self._blacklist:
                    status = "[disabled]"

                info = patch.get_info(show_diff=diff)
                info_lines = info.split("\n")
                lines.append(f"  {status} {info_lines[0]}")
                for line in info_lines[1:]:
                    lines.append(f"    {line}")

            lines.append("")

        patcher_logger.info("\n".join(lines))
        return self

    def _collect_all_patches(self, force_refresh: bool = False) -> List[Tuple[BasePatch, str]]:
        """
        Collect all patches from entries in true insertion order.

        Args:
            force_refresh: If True, force re-collection even if cached.

        Returns:
            List of (patch, parent_name) tuples in insertion order.
            parent_name is the Patch class name for patches from Patch classes,
            or the direct patch's own grouping name.
        """
        # Return cached result if available and not forcing refresh
        if self._collected_patches is not None and not force_refresh:
            return self._collected_patches

        all_patches = []
        order = 0

        for entry_index, (kind, item, options) in enumerate(self._entries):
            if kind == "class":
                if item.name in self._blacklist:
                    continue
                for patch in self._resolve_class_patches(entry_index, item, options):
                    patch._order = order
                    order += 1
                    all_patches.append((patch, item.name))
            else:  # "direct"
                item._order = order
                order += 1
                parent_name = getattr(item, "_parent_name", item.name)
                all_patches.append((item, parent_name))

        # Cache the result
        self._collected_patches = all_patches

        return all_patches

    def apply(self, allow_internal_format: bool = None) -> 'Patcher':
        """
        Apply all registered patches in insertion order.

        Args:
            allow_internal_format: Whether to allow NPU internal format.
                If None (default), uses the value set by allow_internal_format() method,
                or False if not set. This parameter is provided for backward compatibility.

        Returns:
            self for method chaining.
        """
        if self._applied:
            patcher_logger.warning("Patches already applied")
            return self

        # Determine allow_internal_format value:
        # 1. If explicitly passed to apply(), use that value
        # 2. Otherwise, use the value set by allow_internal_format() method
        # 3. If neither is set, default to False
        if allow_internal_format is not None:
            aif_value = allow_internal_format
        elif self._allow_internal_format is not None:
            aif_value = self._allow_internal_format
        else:
            aif_value = False  # Default value

        self._set_npu_internal_format(aif_value)
        self._add_training_loop_patches()

        # Check for conflicts before applying
        self._check_conflicts()

        # Some compatibility patches must take effect before collecting later
        # Patch classes, because Patch.patches() may import heavy third-party
        # modules during collection.
        self._apply_bootstrap_patches()

        # Collect and apply all patches (force refresh to include training loop patches)
        all_patches = self._collect_all_patches(force_refresh=True)

        for patch, parent_name in all_patches:
            self._apply_single_patch(patch, parent_name)

        self._applied = True
        patcher_logger.flush_summary()
        return self

    def _check_conflicts(self):
        """Check for conflicts_with declarations among enabled Patch classes."""
        # Collect enabled Patch class names
        enabled_by_name = {
            item.name: item
            for item in self._iter_enabled_patch_classes()
        }
        enabled_names = set(enabled_by_name)

        # Check each class's conflicts_with
        for kind, item, _ in self._entries:
            if kind != "class" or item.name in self._blacklist:
                continue
            conflicts = getattr(item, "conflicts_with", None)
            if not conflicts:
                continue
            for conflict_name in conflicts:
                if conflict_name in enabled_names and conflict_name not in self._blacklist:
                    conflict_patch = enabled_by_name.get(conflict_name)
                    raise ValueError(
                        f"Patch conflict: '{item.name}' conflicts with '{conflict_name}'. "
                        f"Both are enabled.\n"
                        f"{self._build_conflict_guidance(item, conflict_name, conflict_patch)}"
                    )

    def _add_training_loop_patches(self):
        """Add training loop patches for profiling/brake."""
        if self._profiling_options is None and self._brake_step is None:
            return

        options = {
            'enable_profiler': self._profiling_options is not None,
            'enable_brake': self._brake_step is not None,
        }
        if self._profiling_options:
            options.update(self._profiling_options)
        if self._brake_step:
            options['brake_step'] = self._brake_step

        from mx_driving.patcher.mmcv_patch import (
            build_mmcv_epoch_runner_patch,
            build_mmcv_iter_runner_patch,
        )
        from mx_driving.patcher.mmengine_patch import (
            build_mmengine_epoch_train_loop_patch,
            build_mmengine_iter_train_loop_patch,
        )

        if mmcv_version.is_v1x:
            self.add(
                build_mmcv_epoch_runner_patch(options),
                build_mmcv_iter_runner_patch(options),
            )

        if mmcv_version.is_v2x:
            self.add(
                build_mmengine_epoch_train_loop_patch(options),
                build_mmengine_iter_train_loop_patch(options),
            )

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, *args):
        pass

    @property
    def is_applied(self) -> bool:
        return self._applied


# =============================================================================
# Stub Module Helper (for blocking unavailable imports)
# =============================================================================


class _StubModuleFinder:
    """
    Minimal meta path finder for stub module submodules.

    When Python imports "parent.child" and parent is a _StubModule,
    this finder creates a stub for child and registers it in sys.modules.
    """

    def find_spec(self, fullname: str, path=None, target=None):
        """Find module spec for submodules of stubbed modules."""
        if '.' not in fullname:
            return None

        # Check if parent is a stub module
        parent_name = fullname.rsplit('.', 1)[0]
        parent = sys.modules.get(parent_name)
        if isinstance(parent, _StubModule):
            from importlib.machinery import ModuleSpec
            return ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        """Create the stub module."""
        return _StubModule(spec.name)

    def exec_module(self, module):
        """Register the module in sys.modules."""
        sys.modules[module.__name__] = module


# Install finder once at module load time
_stub_finder = _StubModuleFinder()
if _stub_finder not in sys.meta_path:
    sys.meta_path.insert(0, _stub_finder)


class _StubModule(ModuleType):
    """
    Internal stub module that silently absorbs all attribute access.
    Used by skip_import() to block unavailable CUDA dependencies.

    This stub implements all necessary dunder methods to work with Python's
    import machinery, including __iter__, __path__, and __all__.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.__name__ = name
        self.__file__ = f"<stub:{name}>"
        self.__loader__ = None
        self.__package__ = name.rsplit('.', 1)[0] if '.' in name else name
        # __path__ is required for Python to treat this as a package
        # and to iterate over it when looking for submodules
        self.__path__ = []
        # __all__ is used by "from module import *" syntax
        self.__all__ = []

    def __getattr__(self, name: str) -> Any:
        # Skip private attributes to avoid infinite recursion
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Create submodule name
        submodule_name = f"{self.__name__}.{name}"

        # Check if already registered in sys.modules
        if submodule_name in sys.modules:
            return sys.modules[submodule_name]

        # Create a new stub for the submodule and register it in sys.modules
        # This is critical for Python's import machinery to find submodules
        # when doing "from parent.submodule import something"
        stub = _StubModule(submodule_name)
        sys.modules[submodule_name] = stub
        return stub

    def __call__(self, *args, **kwargs):
        # Allow the stub to be called as a function
        return None

    def __iter__(self):
        # Support iteration (required by Python import machinery)
        return iter([])

    def __contains__(self, item):
        # Support "in" operator
        return False

    def __len__(self):
        # Support len() calls
        return 0

    def __bool__(self):
        # Stub modules are falsy to indicate they are not real
        return False

    def __repr__(self):
        return f"<StubModule '{self.__name__}'>"


def _register_stub_module(module_path: str) -> _StubModule:
    """
    Register a stub module without poisoning real parent packages.

    If a parent package already exists on disk (regular package or namespace
    package), leave it for Python's normal import machinery to load. Only
    synthesize stub parents for package segments that cannot be resolved at all.
    """
    from importlib.machinery import PathFinder

    parts = module_path.split('.')
    search_path = None

    # Create / preserve parent packages (all except leaf)
    for i in range(len(parts) - 1):
        current_path = '.'.join(parts[:i + 1])
        if current_path in sys.modules:
            current_module = sys.modules[current_path]
            search_path = getattr(current_module, "__path__", None)
            continue

        # Use PathFinder directly so we can probe package segments without
        # importing their parent packages and executing parent __init__.py.
        # For child segments, query only the local segment name against the
        # already discovered parent search path.
        query_name = current_path if search_path is None else parts[i]
        spec = PathFinder.find_spec(query_name, search_path)
        if spec is None:
            stub = _StubModule(current_path)
            sys.modules[current_path] = stub
            search_path = stub.__path__

            # Only link synthetic stub parents. Real parents should be linked
            # by Python's import machinery when they are actually imported.
            if i > 0:
                parent_path = '.'.join(parts[:i])
                parent = sys.modules.get(parent_path)
                if parent is not None:
                    setattr(parent, parts[i], stub)
        else:
            search_path = list(spec.submodule_search_locations or [])

    # Register the leaf stub itself
    if module_path not in sys.modules:
        sys.modules[module_path] = _StubModule(module_path)

    # Link leaf to parent only if parent is already loaded.
    if len(parts) > 1:
        parent_path = '.'.join(parts[:-1])
        parent = sys.modules.get(parent_path)
        if parent is not None:
            setattr(parent, parts[-1], sys.modules[module_path])

    return sys.modules[module_path]


def _register_parent_packages(module_path: str) -> None:
    """
    Ensure all parent packages of module_path exist in sys.modules.

    Unlike _register_stub_module which creates stubs for the full path,
    this only creates minimal package stubs for parents that don't exist yet.
    The leaf module itself is NOT created here.
    """
    import importlib
    import importlib.util

    parts = module_path.split('.')
    if len(parts) <= 1:
        return

    # Create parent packages (all except the last segment)
    for i in range(len(parts) - 1):
        current_path = '.'.join(parts[:i + 1])
        if current_path not in sys.modules:
            try:
                importlib.import_module(current_path)
            except Exception:
                spec = importlib.util.find_spec(current_path)
                if spec is not None and spec.submodule_search_locations is not None:
                    # Preserve real filesystem-backed parent packages, including
                    # namespace packages such as "projects/" without __init__.py.
                    pkg = ModuleType(current_path)
                    pkg.__package__ = current_path
                    pkg.__path__ = list(spec.submodule_search_locations)
                    pkg.__spec__ = spec
                    sys.modules[current_path] = pkg
                else:
                    # Fall back to a minimal package stub only when the real
                    # parent package cannot be resolved at all.
                    pkg = ModuleType(current_path)
                    pkg.__package__ = current_path
                    pkg.__path__ = []  # Mark as package
                    sys.modules[current_path] = pkg

        # Link to grandparent for both imported real packages and synthetic stubs.
        if i > 0:
            parent_path = '.'.join(parts[:i])
            parent = sys.modules.get(parent_path)
            child = sys.modules.get(current_path)
            if parent is not None and child is not None:
                setattr(parent, parts[i], child)


# =============================================================================
# replace_with helper for replace_import API
# =============================================================================

class _ImportReplacementSpec:
    """Internal spec object produced by replace_with.module()."""

    __slots__ = ('base_module', 'exports')

    def __init__(self, base_module: Optional[str] = None, exports: Optional[Dict[str, Any]] = None):
        self.base_module = base_module
        self.exports = exports or {}


class _ReplaceWith:
    """
    Helper namespace for constructing replacement specs.

    Usage:
        from mx_driving.patcher import replace_with

        patcher.replace_import("old.module", replace_with.module("new.module"))
        patcher.replace_import("old.module", replace_with.module(Foo=Bar))
        patcher.replace_import("old.module", replace_with.module("new.module", Foo=Bar))
    """

    @staticmethod
    def module(base: Optional[str] = None, **exports) -> _ImportReplacementSpec:
        """
        Construct a module replacement spec.

        Args:
            base: Optional base module path to use as template.
            **exports: Attributes to set on the replacement module.

        Examples:
            replace_with.module("new.module")
            replace_with.module(Foo=FooImpl, Bar=BarImpl)
            replace_with.module("new.module", SpecialClass=CustomImpl)
        """
        return _ImportReplacementSpec(base_module=base, exports=exports)


replace_with = _ReplaceWith()


def __getattr__(name: str) -> Any:
    """
    Provide lazy compatibility aliases for legacy imports.

    Some model examples still import ``PatcherBuilder`` from
    ``mx_driving.patcher.patcher`` instead of from the package root. Resolve
    that alias lazily to avoid introducing an eager circular import with
    ``legacy.py``.
    """
    if name == "PatcherBuilder":
        from mx_driving.patcher.legacy import PatcherBuilder as LegacyPatcherBuilder

        return LegacyPatcherBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
