# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Patch classes for the patcher framework.

Class Hierarchy:

    BasePatch (ABC)
    ├── AtomicPatch         - Single attribute replacement or wrapper
    ├── RegistryPatch       - Register classes to mmcv/mmengine Registry
    ├── Patch               - Composite patch, predefined patches inherit this
    └── LegacyPatch         - Function-based patch (backward compatibility)
"""
from __future__ import annotations

import difflib
import importlib
import inspect
import types
from abc import ABC, ABCMeta, abstractmethod
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from mx_driving.patcher.patcher_logger import patcher_logger
from mx_driving.patcher.reporting import PatchResult, PatchStatus


# =============================================================================
# Core Classes
# =============================================================================

class BasePatch(ABC):
    """Abstract base class for all patches."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier for disable and logging."""
        pass

    @property
    def module(self) -> str:
        """Target module name for grouping."""
        return self.name.split(".")[0] if "." in self.name else ""

    @abstractmethod
    def apply(self) -> PatchResult:
        """Apply this patch."""
        pass

    def get_info(self, show_diff: bool = False) -> str:
        """Get patch info string."""
        return self.name


class _PatchMeta(ABCMeta):
    """
    Metaclass for Patch that enables automatic detection of old vs new usage.

    Inherits from ABCMeta to be compatible with BasePatch (which is an ABC).

    This allows the same `Patch` name to work for both:
    - New style: class MyPatch(Patch): ...
    - Old style: Patch(my_func)  -> returns LegacyPatchWrapper
    """

    def __call__(cls, *args, **kwargs):
        # If called on Patch class directly (not a subclass) with a callable argument,
        # treat it as old-style usage: Patch(func) -> LegacyPatchWrapper
        if cls is Patch and args and callable(args[0]) and not isinstance(args[0], type):
            from mx_driving.patcher.legacy import LegacyPatchWrapper
            return LegacyPatchWrapper(*args, **kwargs)
        # Otherwise, normal class instantiation (new style)
        return super().__call__(*args, **kwargs)


class Patch(BasePatch, metaclass=_PatchMeta):
    """
    Composite patch base class. Predefined patches inherit this.

    This class supports two usage patterns:

    1. New style (recommended) - Inherit and define patches:
        class MultiScaleDeformableAttention(Patch):
            @classmethod
            def patches(cls, options=None) -> List[AtomicPatch]:
                return [
                    AtomicPatch("mmcv.ops.msda.forward", cls.forward),
                    AtomicPatch("mmcv.ops.msda.backward", cls.backward),
                ]

        If `name` is omitted, it defaults to the class name. Set it explicitly
        when you need a stable external identifier across refactors.

    2. Old style (backward compatibility) - Wrap a function:
        def my_patch(module, options):
            module.some_attr = new_value

        patch = Patch(my_patch)  # Returns LegacyPatchWrapper

    The old style is automatically detected when Patch is called with a
    callable argument, and returns a LegacyPatchWrapper instance instead.

    Import Handling:
        Use the @with_imports decorator on replacement functions to lazily
        import modules at first call. The imports are injected into the
        function's globals, so you can use them directly in the function body.

        Two forms are supported:
        - String form (most common): imports the whole module
        - Tuple form: imports specific names from a module

        @staticmethod
        @with_imports("torch_npu")  # import torch_npu
        def replacement(self, x):
            return torch_npu.npu_exp(x)  # noqa: F821

        @staticmethod
        @with_imports(("module.path", "Name1", "Name2"))  # from module.path import Name1, Name2
        def replacement(self, ...):
            return Name1 + Name2  # noqa: F821

        This is optional - you can also use regular imports inside functions.
        For IDE warnings about undefined names, use # noqa: F821 comments.
    """

    name: Optional[str] = None
    # Opt-in escape hatch for compatibility patches that must take effect
    # before collecting later Patch classes would trigger heavyweight imports.
    apply_before_collect: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is Patch:
            return
        if cls.__dict__.get("name") is None:
            cls.name = cls.__name__

    @classmethod
    @abstractmethod
    def patches(cls, options: Optional[Dict] = None) -> List[BasePatch]:
        """Return list of patches. Called at apply time.

        Args:
            options: Optional configuration dict for customizing patch behavior.
        """
        pass

    @classmethod
    def apply_all(cls, options: Optional[Dict] = None) -> List[PatchResult]:
        """
        Apply all patches in this Patch class directly.

        This is a convenience method that allows applying a Patch class
        without going through Patcher. Useful for legacy compatibility.

        Args:
            options: Optional configuration dict for customizing patch behavior.

        Returns:
            List of PatchResult for each atomic patch.
        """
        results = []
        for patch in cls.patches(options):
            result = patch.apply()
            results.append(result)
        return results

    def apply(self) -> PatchResult:
        """Apply all patches in this set.

        Note: For Patch classes, use apply_all() classmethod instead,
        or add to a Patcher instance.
        """
        # This is typically handled by Patcher, but we provide a fallback
        # that applies all patches and returns a summary result
        results = self.__class__.apply_all()
        applied = sum(1 for r in results if r.status == PatchStatus.APPLIED)
        failed = sum(1 for r in results if r.status == PatchStatus.FAILED)

        if failed > 0:
            return PatchResult(PatchStatus.FAILED, self.name, "", f"{failed} patches failed")
        elif applied > 0:
            return PatchResult(PatchStatus.APPLIED, self.name, "", f"{applied} patches applied")
        else:
            return PatchResult(PatchStatus.SKIPPED, self.name, "", "all patches skipped")

    def __iter__(self):
        return iter(self.patches())

    def __repr__(self) -> str:
        return f"Patch({self.name})"


class AtomicPatch(BasePatch):
    """
    Single attribute replacement or wrapper patch.

    Args:
        target: Full dot-separated path to the attribute to replace/wrap.
        replacement: The new object, or a string path to resolve at apply time.
                     If not provided, target_wrapper must be set.
        aliases: Additional paths to patch (for re-export handling).
        precheck: Optional callback -> bool. Return False to skip.
                  Supports two patterns:
                  - precheck() -> bool: No arguments
                  - precheck(target=..., replacement=..., ...) -> bool:
                    Keyword arguments matching AtomicPatch constructor parameters
        runtime_check: Optional callback(*args, **kwargs) -> bool for conditional dispatch.
        replacement_wrapper: Optional callable(replacement) -> wrapped.
                             Wraps the replacement before applying.
        target_wrapper: Optional callable(original) -> wrapped.
                        Wraps the original target. Used when replacement is None.

    Examples:
        # Simple replacement
        AtomicPatch("module.func", new_func)

        # target_wrapper mode - wrap original with custom logic
        def wrap_init(original):
            def new_init(self, *args, **kwargs):
                original(self, *args, **kwargs)
                self.extra = "added"
            return new_init

        AtomicPatch("module.Class.__init__", target_wrapper=wrap_init)

        # replacement_wrapper - wrap the replacement
        AtomicPatch("module.func", new_func, replacement_wrapper=add_logging)

        # With precheck - no arguments (most common)
        AtomicPatch(
            "mmcv.ops.func",
            npu_func,
            precheck=lambda: mmcv_version.is_v2x
        )

        # With precheck - check target path
        AtomicPatch(
            "mmcv.ops.func",
            npu_func,
            precheck=lambda target: target.startswith("mmcv.ops")
        )
    """

    def __init__(
        self,
        target: str,
        replacement: Any = None,
        *,
        aliases: List[str] = None,
        precheck: Optional[Callable[..., bool]] = None,
        runtime_check: Optional[Callable[..., bool]] = None,
        replacement_wrapper: Optional[Callable[[Callable], Callable]] = None,
        target_wrapper: Optional[Callable[[Callable], Callable]] = None,
    ):
        if replacement is None and target_wrapper is None:
            raise ValueError("Either replacement or target_wrapper must be provided")

        self.target = target
        self._replacement = replacement
        self._replacement_wrapper = replacement_wrapper
        self._target_wrapper = target_wrapper
        self.aliases = aliases or []
        self.precheck = precheck
        self.runtime_check = runtime_check
        self.is_applied = False
        self._order = 0
        self._original = None

    @property
    def name(self) -> str:
        return self.target

    @property
    def replacement(self) -> Any:
        """Resolve replacement, supporting string paths."""
        if isinstance(self._replacement, str):
            return _get_by_path(self._replacement)
        return self._replacement

    def apply(self) -> PatchResult:
        """Apply this patch to target and all aliases."""
        result = self._apply_to_target(self.target)

        # Apply to aliases
        for alias in self.aliases:
            self._apply_to_target(alias)

        if result.status == PatchStatus.APPLIED:
            self.is_applied = True

        return result

    def _apply_to_target(self, target: str) -> PatchResult:
        """Apply patch to a single target path."""
        parts = target.split(".")
        if len(parts) < 2:
            return PatchResult(PatchStatus.FAILED, self.name, "", "invalid target path")

        # Import root module
        module_name = parts[0]
        module = _import_module(module_name)
        if module is None:
            return PatchResult(PatchStatus.SKIPPED, self.name, module_name,
                               f"module not found: {module_name}")

        # Resolve parent and attr_name
        attr_path = ".".join(parts[1:])
        path_parts = attr_path.rsplit(".", 1)
        if len(path_parts) == 1:
            parent, attr_name = module, path_parts[0]
        else:
            parent_path = f"{module_name}.{path_parts[0]}"
            parent = _get_by_path(parent_path)
            attr_name = path_parts[1]

        if parent is None:
            return PatchResult(PatchStatus.SKIPPED, self.name, module_name,
                               f"target not found: {target}")

        # Check existence
        is_dict = isinstance(parent, dict)
        exists = attr_name in parent if is_dict else hasattr(parent, attr_name)
        original = (parent.get(attr_name) if is_dict else getattr(parent, attr_name, None)) if exists else None

        # Fail-safe semantics: if target attribute/key does not exist, skip.
        if not exists:
            return PatchResult(PatchStatus.SKIPPED, self.name, module_name,
                               f"target not found: {target}")

        # Precheck - detect signature and pass matching parameters
        if self.precheck:
            try:
                sig = inspect.signature(self.precheck)
                params = sig.parameters

                if not params:
                    # precheck() - no arguments
                    if not self.precheck():
                        return PatchResult(PatchStatus.SKIPPED, self.name, module_name, "precheck failed")
                else:
                    # precheck(**kwargs) - pass matching parameters
                    kwargs = {}
                    available = {
                        'target': self.target,
                        'replacement': self._replacement,
                        'replacement_wrapper': self._replacement_wrapper,
                        'target_wrapper': self._target_wrapper,
                        'aliases': self.aliases,
                    }
                    for param_name in params:
                        if param_name in available:
                            kwargs[param_name] = available[param_name]
                    if not self.precheck(**kwargs):
                        return PatchResult(PatchStatus.SKIPPED, self.name, module_name, "precheck failed")
            except Exception as e:
                # Precheck raised exception - this is a code bug, should be FAILED
                return PatchResult(PatchStatus.FAILED, self.name, module_name, f"precheck error: {e}")

        # Store original for info/diff
        self._original = original

        # Resolve replacement
        replacement = self.replacement
        if replacement is None:
            # target_wrapper mode: wrap the original
            if self._target_wrapper is None:
                return PatchResult(PatchStatus.SKIPPED, self.name, module_name, "replacement not found")
            if original is None:
                return PatchResult(PatchStatus.SKIPPED, self.name, module_name, "original not found for target_wrapper")
            try:
                replacement = self._target_wrapper(original)
            except Exception as e:
                return PatchResult(PatchStatus.FAILED, self.name, module_name, f"target_wrapper error: {e}")
        else:
            # Apply replacement_wrapper if provided
            if self._replacement_wrapper is not None:
                try:
                    replacement = self._replacement_wrapper(replacement)
                except Exception as e:
                    return PatchResult(PatchStatus.FAILED, self.name, module_name, f"replacement_wrapper error: {e}")

        # Wrap with runtime check if needed
        if self.runtime_check and callable(original) and callable(replacement):
            replacement = self._wrap_with_runtime_check(original, replacement)

        # Apply
        if is_dict:
            parent[attr_name] = replacement
        else:
            setattr(parent, attr_name, replacement)

        return PatchResult(PatchStatus.APPLIED, self.name, module_name)

    def _wrap_with_runtime_check(self, original: Callable, replacement: Callable) -> Callable:
        """Wrap replacement with runtime conditional dispatch."""
        check = self.runtime_check
        target_name = self.target

        @wraps(replacement)
        def wrapper(*args, **kwargs):
            try:
                if check(*args, **kwargs):
                    return replacement(*args, **kwargs)
            except Exception:
                # Runtime check exception - use original (debug level to avoid performance impact)
                patcher_logger.debug(f"Runtime check exception for {target_name}, using original")
            return original(*args, **kwargs)

        return wrapper

    def get_info(self, show_diff: bool = False) -> str:
        """Get patch info string."""
        original_name = _get_callable_name(self._original) if self._original else "<missing>"
        replacement_name = _get_callable_name(self.replacement)
        info = f"{self.target}: {original_name} -> {replacement_name}"

        if show_diff and self._original and callable(self._original) and callable(self.replacement):
            diff = _get_source_diff(self._original, self.replacement)
            if diff:
                info += f"\n{diff}"

        return info

    def __repr__(self) -> str:
        return f"AtomicPatch({self.target})"


class RegistryPatch(BasePatch):
    """
    Register a class/function to mmcv/mmengine Registry.

    A declarative way to register modules to mmcv/mmengine registries.

    Args:
        registry: Registry path, e.g., "mmcv.runner.HOOKS" or "mmengine.registry.MODELS"
        module_cls: The class or function to register. Can be None if module_factory is provided.
        name: Registration name. Required if module_factory is used, otherwise defaults to module_cls.__name__.
        force: Whether to force overwrite existing registration. Default True.
        precheck: Optional callable -> bool. Return False to skip.
                  Supports two patterns:
                  - precheck() -> bool: No arguments
                  - precheck(registry=..., name=..., ...) -> bool:
                    Keyword arguments matching RegistryPatch constructor parameters
        module_factory: Optional callable() -> type. Called at apply time to create the class.
                        Use this when the class needs to be defined dynamically (e.g., inheriting
                        from classes that are only available at runtime).

    Examples:
        # Register a pre-defined class
        RegistryPatch(
            "mmcv.runner.HOOKS",
            MyOptimizerHook,
            name="OptimizerHook",
            precheck=lambda: mmcv_version.is_v1x,
        )

        # Register a dynamically created class
        def create_hook():
            from mmcv.runner.hooks import Hook, HOOKS
            class MyHook(Hook):
                def after_train_iter(self, runner):
                    pass
            return MyHook

        RegistryPatch(
            "mmcv.runner.HOOKS",
            name="MyHook",
            module_factory=create_hook,
        )
    """

    def __init__(
        self,
        registry: str,
        module_cls: type = None,
        *,
        name: Optional[str] = None,
        force: bool = True,
        precheck: Optional[Callable[..., bool]] = None,
        module_factory: Optional[Callable[[], type]] = None,
    ):
        if module_cls is None and module_factory is None:
            raise ValueError("Either module_cls or module_factory must be provided")
        if module_cls is None and name is None:
            raise ValueError("name is required when using module_factory")

        self.registry = registry
        self.module_cls = module_cls
        self.module_factory = module_factory
        self.register_name = name or (module_cls.__name__ if module_cls else None)
        self.force = force
        self.precheck = precheck
        self.is_applied = False
        self._order = 0

    @property
    def name(self) -> str:
        return f"{self.registry}.{self.register_name}"

    def apply(self) -> PatchResult:
        """Register the module to the registry."""
        if self.precheck:
            try:
                sig = inspect.signature(self.precheck)
                params = sig.parameters

                if not params:
                    # precheck() - no arguments
                    if not self.precheck():
                        return PatchResult(PatchStatus.SKIPPED, self.name, "", "precheck failed")
                else:
                    # precheck(**kwargs) - pass matching parameters
                    kwargs = {}
                    available = {
                        'registry': self.registry,
                        'module_cls': self.module_cls,
                        'name': self.register_name,
                        'force': self.force,
                        'module_factory': self.module_factory,
                    }
                    for param_name in params:
                        if param_name in available:
                            kwargs[param_name] = available[param_name]
                    if not self.precheck(**kwargs):
                        return PatchResult(PatchStatus.SKIPPED, self.name, "", "precheck failed")
            except Exception as e:
                # Precheck raised exception - this is a code bug, should be FAILED
                return PatchResult(PatchStatus.FAILED, self.name, "", f"precheck error: {e}")

        # Resolve registry
        registry = _get_by_path(self.registry)
        if registry is None:
            return PatchResult(PatchStatus.SKIPPED, self.name, "",
                               f"registry not found: {self.registry}")

        # Check if registry has register_module method
        if not hasattr(registry, "register_module"):
            return PatchResult(PatchStatus.SKIPPED, self.name, "", "invalid registry")

        # Get the module class (either directly or via factory)
        try:
            module_cls = self.module_factory() if self.module_factory else self.module_cls
        except Exception as e:
            return PatchResult(PatchStatus.FAILED, self.name, "", f"factory error: {e}")

        try:
            registry.register_module(name=self.register_name, force=self.force, module=module_cls)
            self.is_applied = True
            return PatchResult(PatchStatus.APPLIED, self.name, "")
        except Exception as e:
            return PatchResult(PatchStatus.FAILED, self.name, "", str(e))

    def get_info(self, show_diff: bool = False) -> str:
        cls_name = self.module_cls.__name__ if self.module_cls else "<factory>"
        return f"{self.registry} <- {self.register_name} ({cls_name})"

    def __repr__(self) -> str:
        return f"RegistryPatch({self.registry}, {self.register_name})"


class LegacyPatch(BasePatch):
    """
    Function-based patch for backward compatibility.

    Args:
        func: Callable(module, options) that performs the patching.
        options: Optional dict passed to func.
        target_module: The module this patch targets. Required for the patch to be applied.

    Raises:
        ValueError: If target_module is not provided.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        options: Optional[Dict] = None,
        target_module: str = None,
    ):
        if target_module is None:
            raise ValueError(
                f"LegacyPatch requires target_module parameter. "
                f"Usage: LegacyPatch({func.__name__}, target_module='module_name')"
            )
        self.func = func
        self.options = options or {}
        self.target_module = target_module
        self.is_applied = False
        self._order = 0

    @staticmethod
    def _infer_display_name(func: Callable[..., Any]) -> str:
        """Infer a readable display name for anonymous legacy helper functions."""
        explicit = getattr(func, "__patch_name__", None)
        if explicit:
            return explicit

        raw_name = getattr(func, "__name__", "") or "legacy_patch"
        if raw_name != "_apply":
            return raw_name

        qualname = getattr(func, "__qualname__", "")
        if ".<locals>." in qualname:
            parts = qualname.split(".<locals>.")
            outer = parts[-2].split(".")[-1] if len(parts) >= 2 else qualname
        else:
            outer = qualname.split(".")[-2] if "." in qualname else qualname
        if outer.startswith("build_") and outer.endswith("_patch"):
            return outer[len("build_"):-len("_patch")]
        if outer.endswith("_patch"):
            return outer[:-len("_patch")]
        return raw_name

    @property
    def name(self) -> str:
        return self._infer_display_name(self.func)

    @property
    def module(self) -> str:
        return self.target_module

    def apply(self) -> PatchResult:
        """Apply this legacy patch."""
        module = _import_module(self.target_module)
        if module is None:
            return PatchResult(PatchStatus.SKIPPED, self.name, self.target_module, "module not found")

        try:
            self.func(module, self.options)
            self.is_applied = True
            return PatchResult(PatchStatus.APPLIED, self.name, self.target_module)
        except AttributeError as e:
            return PatchResult(PatchStatus.SKIPPED, self.name, self.target_module, str(e))
        except Exception as e:
            return PatchResult(PatchStatus.FAILED, self.name, self.target_module, str(e))

    def __repr__(self) -> str:
        return f"LegacyPatch({self.name})"


# =============================================================================
# Lazy Import Decorator
# =============================================================================

def with_imports(*import_specs: Union[str, Tuple[str, ...]],
                 apply_decorators: Optional[List[Tuple[str, dict]]] = None):
    """
    Decorator for lazy importing modules into a function's global namespace.

    This decorator delays imports until the function is first called, then
    injects the imported names into the function's globals. This allows the
    function body to use the imported names directly, just like regular imports.

    Args:
        import_specs: Variable number of specifications. Three forms supported:

            - String form: "module_path" - imports the whole module
              Example: "torch_npu" -> import torch_npu

            - Tuple form: (module_path, name1, name2, ...) - imports specific names
              Example: ("torch", "sin", "cos") -> from torch import sin, cos

            - Decorator string: "@expression" - lazily applies a decorator.
              The expression is evaluated in the resolved import namespace after
              all imports are done. Names used in the expression must be
              imported by prior import specs.
              Example: "@auto_fp16(apply_to=('q',), out_fp32=True)"

        apply_decorators: (Legacy) Optional list of (decorator_path, kwargs) tuples.
            Prefer using "@expression" strings instead.

    Examples:
        # Import whole modules
        @with_imports("math", "torch_npu")
        def my_func(x, sigma):
            return torch_npu.npu_exp(x) * math.sqrt(sigma)  # noqa: F821

        # Import specific names from module
        @with_imports(("torch.nn.functional", "relu", "softmax"))
        def my_func(x):
            return softmax(relu(x))  # noqa: F821

        # Use with @staticmethod (must be placed AFTER @staticmethod)
        @staticmethod
        @with_imports("torch_npu")
        def process(data):
            return torch_npu.npu_exp(data)  # noqa: F821

        # Apply target-module decorators with @ expression
        @staticmethod
        @with_imports(
            ("projects.module", "rearrange", "auto_fp16"),
            "@auto_fp16(apply_to=('q', 'k', 'v'), out_fp32=True)",
        )
        def forward(self, q, k, v):
            return rearrange(q, '...')  # noqa: F821

        # No-arg decorator
        @with_imports("torch", "@torch.no_grad()")
        def inference(self, x):
            return torch.relu(x)  # noqa: F821

    Note:
        - The decorated function can use imported names directly in its body
        - Imports are cached after first call, no repeated import overhead
        - For IDE warnings about undefined names, use # noqa: F821 comments
        - Do NOT stack multiple @with_imports on the same function; combine all
          imports into a single @with_imports() call instead.
    """
    # Separate import specs from decorator expressions
    pure_imports = []
    decorator_exprs = []  # "@expr" strings
    legacy_decorators = apply_decorators or []
    for spec in import_specs:
        if isinstance(spec, str) and spec.startswith("@"):
            decorator_exprs.append(spec[1:])  # strip @
        else:
            pure_imports.append(spec)

    def decorator(func):
        # Handle staticmethod/classmethod wrapped functions
        actual_func = func
        wrapper_type = None
        if isinstance(func, staticmethod):
            actual_func = func.__func__
            wrapper_type = staticmethod
        elif isinstance(func, classmethod):
            actual_func = func.__func__
            wrapper_type = classmethod

        # Detect stacked @with_imports (would not compose correctly)
        if getattr(actual_func, '_with_imports_decorated', False):
            patcher_logger.warning(
                f"with_imports: stacking multiple @with_imports on '{actual_func.__name__}' "
                f"is not supported. Combine all imports into a single @with_imports() call."
            )

        # State for lazy resolution
        resolved = [False]
        resolved_func = [None]

        @wraps(actual_func)
        def wrapper(*args, **kwargs):
            if not resolved[0]:
                # Build new globals with imports
                new_globals = actual_func.__globals__.copy()

                for spec in pure_imports:
                    if isinstance(spec, str):
                        module_path = spec
                        names = ()
                    else:
                        module_path = spec[0]
                        names = spec[1:]

                    try:
                        module = importlib.import_module(module_path)
                        if not names:
                            module_name = module_path.split(".")[-1]
                            new_globals[module_name] = module
                        else:
                            for name in names:
                                if hasattr(module, name):
                                    new_globals[name] = getattr(module, name)
                                else:
                                    patcher_logger.debug(
                                        f"with_imports: {name} not found in {module_path}"
                                    )
                    except ImportError as e:
                        patcher_logger.warning(
                            f"with_imports: failed to import {module_path}: {e}"
                        )

                # Create new function with updated globals
                resolved_func[0] = types.FunctionType(
                    actual_func.__code__,
                    new_globals,
                    actual_func.__name__,
                    actual_func.__defaults__,
                    actual_func.__closure__
                )
                resolved_func[0].__kwdefaults__ = actual_func.__kwdefaults__
                resolved_func[0].__annotations__ = getattr(actual_func, '__annotations__', {})
                resolved_func[0].__dict__.update(getattr(actual_func, '__dict__', {}))

                # Apply @ decorator expressions (evaluated in resolved namespace)
                for expr in decorator_exprs:
                    try:
                        dec = eval(expr, new_globals)  # noqa: S307
                        resolved_func[0] = dec(resolved_func[0])
                    except Exception as e:
                        patcher_logger.warning(
                            f"with_imports: failed to apply @{expr}: {e}"
                        )

                # Apply legacy (path, kwargs) decorators
                for dec_path, dec_kwargs in legacy_decorators:
                    try:
                        dec_func = _get_by_path(dec_path)
                        if dec_func is not None:
                            if dec_kwargs:
                                resolved_func[0] = dec_func(**dec_kwargs)(resolved_func[0])
                            else:
                                resolved_func[0] = dec_func(resolved_func[0])
                        else:
                            patcher_logger.warning(
                                f"with_imports: decorator not found: {dec_path}"
                            )
                    except Exception as e:
                        patcher_logger.warning(
                            f"with_imports: failed to apply decorator {dec_path}: {e}"
                        )

                resolved[0] = True

            return resolved_func[0](*args, **kwargs)

        wrapper._with_imports_decorated = True

        if wrapper_type is not None:
            return wrapper_type(wrapper)
        return wrapper

    return decorator


# =============================================================================
# Helper Functions
# =============================================================================

def _import_module(name: str) -> Optional[Any]:
    """Import a module by name, returning None if not found."""
    try:
        return importlib.import_module(name)
    except (ModuleNotFoundError, ImportError):
        return None


def _get_by_path(path: str, ensure_import: bool = True) -> Optional[Any]:
    """
    Resolve a dot-separated path to an object.

    Args:
        path: Dot-separated path like "mmcv.ops.SparseConv3d"
        ensure_import: If True, try to import intermediate submodules to handle lazy loading.

    Example: "mmcv.ops.SparseConv3d" -> mmcv.ops.SparseConv3d
    """
    parts = path.split(".")
    if not parts:
        return None

    module = _import_module(parts[0])
    if module is None:
        return None

    obj = module
    for i, part in enumerate(parts[1:], start=1):
        if isinstance(obj, dict):
            obj = obj.get(part)
        elif hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            # Try importing as submodule to handle lazy loading
            if ensure_import:
                submodule_path = ".".join(parts[:i + 1])
                submodule = _import_module(submodule_path)
                if submodule is not None:
                    obj = submodule
                    continue
            return None
        if obj is None:
            return None
    return obj


def _get_callable_name(obj: Any) -> str:
    """Get a readable name for a callable object."""
    if obj is None:
        return "<None>"
    if hasattr(obj, "__qualname__"):
        return obj.__qualname__
    if hasattr(obj, "__name__"):
        return obj.__name__
    return type(obj).__name__


def _get_source_diff(original: Any, replacement: Any) -> str:
    """Get unified diff between two callable objects' source code."""
    try:
        orig_source = inspect.getsource(original).splitlines(keepends=True)
        repl_source = inspect.getsource(replacement).splitlines(keepends=True)
        diff = difflib.unified_diff(
            orig_source,
            repl_source,
            fromfile=f"original: {_get_callable_name(original)}",
            tofile=f"replacement: {_get_callable_name(replacement)}",
            lineterm="",
        )
        return "".join(diff)
    except (TypeError, OSError):
        return ""


# =============================================================================
# Version Detection (re-exported from version module)
# =============================================================================

from mx_driving.patcher.version import (
    get_version,
    check_version,
    mmcv_version,
    is_mmcv_v1x,
    is_mmcv_v2x,
)
