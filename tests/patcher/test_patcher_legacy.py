# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Legacy patcher API compatibility tests.

This module tests backward compatibility with the old patcher API (pre-87ec1895):
- LegacyPatchWrapper (Patch(func) syntax)
- LegacyPatcherBuilder (PatcherBuilder class)
- Old-style patch functions (msda, dc, mdc, batch_matmul, etc.)
- Mixed usage of old and new APIs
- Delegation to new Patcher implementation
"""
import importlib.util
import io
import os
import sys
import types
import unittest
from typing import Dict, List
from types import ModuleType
from unittest.mock import MagicMock, patch

# Get project root
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_patcher_dir = os.path.join(_project_root, "mx_driving", "patcher")


def _load_module_from_file(module_name: str, file_path: str):
    """Load a module directly from file without triggering parent package __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load patcher modules directly to avoid torch dependency
_patcher_logger = _load_module_from_file(
    "mx_driving.patcher.patcher_logger",
    os.path.join(_patcher_dir, "patcher_logger.py")
)
_reporting = _load_module_from_file(
    "mx_driving.patcher.reporting",
    os.path.join(_patcher_dir, "reporting.py")
)
_version_module = _load_module_from_file(
    "mx_driving.patcher.version",
    os.path.join(_patcher_dir, "version.py")
)
_patch_module = _load_module_from_file(
    "mx_driving.patcher.patch",
    os.path.join(_patcher_dir, "patch.py")
)
_patcher_module = _load_module_from_file(
    "mx_driving.patcher.patcher",
    os.path.join(_patcher_dir, "patcher.py")
)
_legacy_module = _load_module_from_file(
    "mx_driving.patcher.legacy",
    os.path.join(_patcher_dir, "legacy.py")
)

# Load __init__.py to get legacy patch functions
# We need to load it after the other modules are loaded
_patcher_init_module = _load_module_from_file(
    "mx_driving.patcher",
    os.path.join(_patcher_dir, "__init__.py")
)

# Import classes from loaded modules
AtomicPatch = _patch_module.AtomicPatch
BasePatch = _patch_module.BasePatch
LegacyPatch = _patch_module.LegacyPatch
Patch = _patch_module.Patch
Patcher = _patcher_module.Patcher
PatchResult = _reporting.PatchResult
PatchStatus = _reporting.PatchStatus

# Legacy API imports
LegacyPatchWrapper = _legacy_module.LegacyPatchWrapper
LegacyPatcherBuilder = _legacy_module.LegacyPatcherBuilder
PatcherBuilder = _legacy_module.PatcherBuilder


class TestLegacyPatchWrapper(unittest.TestCase):
    """
    Test LegacyPatchWrapper class.

    LegacyPatchWrapper wraps old-style patch functions with signature (module, options).
    """

    def setUp(self):
        """Reset migration warning flag before each test."""
        LegacyPatchWrapper._migration_warning_shown = False

    def test_basic_wrapper(self):
        """Test basic LegacyPatchWrapper creation."""
        def my_patch(module, options):
            pass

        wrapper = LegacyPatchWrapper(my_patch)

        self.assertEqual(wrapper.name, "my_patch")
        self.assertEqual(wrapper.func, my_patch)
        self.assertEqual(wrapper.options, {})
        self.assertEqual(wrapper.priority, 0)
        self.assertFalse(wrapper.is_applied)

    def test_wrapper_with_options(self):
        """Test LegacyPatchWrapper with options."""
        def my_patch(module, options):
            pass

        wrapper = LegacyPatchWrapper(my_patch, options={'key': 'value'})

        self.assertEqual(wrapper.options, {'key': 'value'})

    def test_wrapper_with_priority(self):
        """Test LegacyPatchWrapper with priority."""
        def patch1(module, options):
            pass

        def patch2(module, options):
            pass

        wrapper1 = LegacyPatchWrapper(patch1, priority=10)
        wrapper2 = LegacyPatchWrapper(patch2, priority=5)

        # Lower priority should come first
        self.assertTrue(wrapper2 < wrapper1)

    def test_wrapper_sorting(self):
        """Test that wrappers can be sorted by priority."""
        def patch1(module, options):
            pass

        def patch2(module, options):
            pass

        def patch3(module, options):
            pass

        wrappers = [
            LegacyPatchWrapper(patch1, priority=10),
            LegacyPatchWrapper(patch2, priority=5),
            LegacyPatchWrapper(patch3, priority=15),
        ]
        sorted_wrappers = sorted(wrappers)

        self.assertEqual(sorted_wrappers[0].name, "patch2")
        self.assertEqual(sorted_wrappers[1].name, "patch1")
        self.assertEqual(sorted_wrappers[2].name, "patch3")


class TestPatchMetaclass(unittest.TestCase):
    """
    Test Patch metaclass for automatic old/new style detection.

    The Patch class uses a metaclass that detects:
    - Patch(func) -> returns LegacyPatchWrapper (old style)
    - class MyPatch(Patch): ... -> normal class inheritance (new style)
    """

    def setUp(self):
        """Reset migration warning flag before each test."""
        LegacyPatchWrapper._migration_warning_shown = False

    def test_patch_with_callable_returns_wrapper(self):
        """Test that Patch(func) returns LegacyPatchWrapper."""
        def my_patch(module, options):
            pass

        result = Patch(my_patch)

        self.assertIsInstance(result, LegacyPatchWrapper)
        self.assertEqual(result.name, "my_patch")

    def test_patch_subclass_works_normally(self):
        """Test that subclassing Patch works normally."""
        class MyPatch(Patch):
            name = "my_patch"

            @classmethod
            def patches(cls, options=None):
                return []

        # Should be a class, not an instance
        self.assertTrue(isinstance(MyPatch, type))
        self.assertTrue(issubclass(MyPatch, Patch))
        self.assertEqual(MyPatch.name, "my_patch")

    def test_patch_with_options(self):
        """Test Patch(func, options=...) passes options to wrapper."""
        def my_patch(module, options):
            pass

        result = Patch(my_patch, options={'key': 'value'})

        self.assertIsInstance(result, LegacyPatchWrapper)
        self.assertEqual(result.options, {'key': 'value'})


class TestLegacyPatcherBuilder(unittest.TestCase):
    """
    Test LegacyPatcherBuilder class.

    LegacyPatcherBuilder provides the old PatcherBuilder API and
    delegates to the new Patcher implementation.
    """

    def setUp(self):
        """Create mock module for testing."""
        self.mock_module = types.ModuleType('legacy_builder_test')
        self.mock_module.func = lambda x: x
        sys.modules['legacy_builder_test'] = self.mock_module
        LegacyPatchWrapper._migration_warning_shown = False

    def tearDown(self):
        """Clean up mock module."""
        if 'legacy_builder_test' in sys.modules:
            del sys.modules['legacy_builder_test']

    def test_basic_builder(self):
        """Test basic builder creation."""
        builder = LegacyPatcherBuilder()

        self.assertEqual(builder._module_patches, {})
        self.assertEqual(builder._blacklist, set())

    def test_add_module_patch(self):
        """Test add_module_patch method."""
        def my_patch(module, options):
            module.func = lambda x: x * 2

        builder = LegacyPatcherBuilder()
        builder.add_module_patch("legacy_builder_test", Patch(my_patch))

        self.assertIn("legacy_builder_test", builder._module_patches)
        self.assertEqual(len(builder._module_patches["legacy_builder_test"]), 1)

    def test_chained_add_module_patch(self):
        """Test chained add_module_patch calls."""
        def patch1(module, options):
            pass

        def patch2(module, options):
            pass

        builder = (
            LegacyPatcherBuilder()
            .add_module_patch("module1", Patch(patch1))
            .add_module_patch("module2", Patch(patch2))
        )

        self.assertIn("module1", builder._module_patches)
        self.assertIn("module2", builder._module_patches)

    def test_disable_patches(self):
        """Test disable_patches method."""
        builder = LegacyPatcherBuilder()
        builder.disable_patches("patch1", "patch2")

        self.assertIn("patch1", builder._blacklist)
        self.assertIn("patch2", builder._blacklist)

    def test_with_profiling(self):
        """Test with_profiling method."""
        builder = LegacyPatcherBuilder()
        builder.with_profiling("/path/to/prof", level=1, skip_first=50)

        self.assertIsNotNone(builder._profiling_options)
        self.assertEqual(builder._profiling_options['path'], "/path/to/prof")
        self.assertEqual(builder._profiling_options['level'], 1)
        self.assertEqual(builder._profiling_options['skip_first'], 50)

    def test_brake_at(self):
        """Test brake_at method."""
        builder = LegacyPatcherBuilder()
        builder.brake_at(100)

        self.assertEqual(builder._brake_step, 100)

    def test_build_returns_legacy_patcher(self):
        """Test build method returns _LegacyPatcher."""
        builder = LegacyPatcherBuilder()
        patcher = builder.build()

        self.assertIsNotNone(patcher)
        self.assertTrue(hasattr(patcher, 'apply'))
        self.assertTrue(hasattr(patcher, '__enter__'))
        self.assertTrue(hasattr(patcher, '__exit__'))

    def test_build_creates_new_patcher_internally(self):
        """Test that build() creates a new Patcher internally."""
        builder = LegacyPatcherBuilder()
        legacy_patcher = builder.build()

        # _LegacyPatcher should wrap a new Patcher
        self.assertIsInstance(legacy_patcher._patcher, Patcher)

    def test_full_workflow(self):
        """Test full legacy workflow: build -> apply."""
        def my_patch(module, options):
            module.func = lambda x: x * 10

        builder = LegacyPatcherBuilder()
        builder.add_module_patch("legacy_builder_test", Patch(my_patch))

        with builder.build() as patcher:
            self.assertEqual(self.mock_module.func(5), 50)


class TestPatcherBuilderAlias(unittest.TestCase):
    """Test that PatcherBuilder is an alias for LegacyPatcherBuilder."""

    def test_patcherbuilder_is_alias(self):
        """Test PatcherBuilder is LegacyPatcherBuilder."""
        self.assertIs(PatcherBuilder, LegacyPatcherBuilder)

    def test_patcher_module_exposes_legacy_alias(self):
        """Test mx_driving.patcher.patcher lazily exposes PatcherBuilder."""
        self.assertIs(getattr(_patcher_module, "PatcherBuilder"), LegacyPatcherBuilder)


class TestLegacyPatcherDelegation(unittest.TestCase):
    """
    Test that _LegacyPatcher properly delegates to new Patcher.
    """

    def setUp(self):
        """Create mock modules for testing."""
        self.mock_module = types.ModuleType('delegation_test')
        self.mock_module.func = lambda x: x
        sys.modules['delegation_test'] = self.mock_module
        LegacyPatchWrapper._migration_warning_shown = False

    def tearDown(self):
        """Clean up mock modules."""
        if 'delegation_test' in sys.modules:
            del sys.modules['delegation_test']

    def test_apply_delegates_to_patcher(self):
        """Test that apply() delegates to the internal Patcher."""
        def my_patch(module, options):
            module.func = lambda x: x * 5

        builder = LegacyPatcherBuilder()
        builder.add_module_patch("delegation_test", Patch(my_patch))
        legacy_patcher = builder.build()

        # Before apply
        self.assertFalse(legacy_patcher.is_applied)

        # Apply
        legacy_patcher.apply()

        # After apply
        self.assertTrue(legacy_patcher.is_applied)
        self.assertEqual(self.mock_module.func(10), 50)

    def test_context_manager_delegates(self):
        """Test that context manager delegates to Patcher."""
        def my_patch(module, options):
            module.func = lambda x: x * 3

        builder = LegacyPatcherBuilder()
        builder.add_module_patch("delegation_test", Patch(my_patch))

        with builder.build() as patcher:
            self.assertTrue(patcher.is_applied)
            self.assertEqual(self.mock_module.func(10), 30)


class TestMixedUsage(unittest.TestCase):
    """
    Test mixed usage of old and new patcher APIs.

    Scenarios:
    - Using old Patch(func) with new Patcher
    - Using new Patch classes with old PatcherBuilder
    - Combining both in the same application
    """

    def setUp(self):
        """Create mock modules for testing."""
        self.mock_module1 = types.ModuleType('mixed_test_module1')
        self.mock_module1.func = lambda x: x
        sys.modules['mixed_test_module1'] = self.mock_module1

        self.mock_module2 = types.ModuleType('mixed_test_module2')
        self.mock_module2.func = lambda x: x
        sys.modules['mixed_test_module2'] = self.mock_module2

        LegacyPatchWrapper._migration_warning_shown = False

    def tearDown(self):
        """Clean up mock modules."""
        for name in ['mixed_test_module1', 'mixed_test_module2']:
            if name in sys.modules:
                del sys.modules[name]

    def test_old_patch_with_new_patcher(self):
        """Test using old-style Patch(func) with new Patcher via LegacyPatch."""
        def my_patch(module, options):
            module.func = lambda x: x * 5

        # Old-style patch function can be wrapped in LegacyPatch for new Patcher
        patcher = Patcher()
        patcher.add(LegacyPatch(my_patch, target_module="mixed_test_module1"))
        patcher.apply()

        self.assertEqual(self.mock_module1.func(10), 50)

    def test_new_patch_class_with_old_builder(self):
        """Test using new-style Patch class with old PatcherBuilder."""
        # Define a new-style Patch class
        class MyNewPatch(Patch):
            name = "my_new_patch"

            @classmethod
            def patches(cls, options=None):
                return [
                    AtomicPatch("mixed_test_module2.func", lambda x: x * 7)
                ]

        # Create a wrapper function that applies the new patch
        def apply_new_patch(module, options):
            MyNewPatch.apply_all()

        # Use with old builder
        builder = LegacyPatcherBuilder()
        builder.add_module_patch("mixed_test_module2", Patch(apply_new_patch))

        with builder.build():
            self.assertEqual(self.mock_module2.func(10), 70)

    def test_both_apis_in_same_session(self):
        """Test using both old and new APIs in the same session."""
        # First, use new API
        patcher = Patcher()
        patcher.add(AtomicPatch("mixed_test_module1.func", lambda x: x * 2))
        patcher.apply()

        self.assertEqual(self.mock_module1.func(10), 20)

        # Then, use old API for another module
        def old_patch(module, options):
            module.func = lambda x: x * 3

        builder = LegacyPatcherBuilder()
        builder.add_module_patch("mixed_test_module2", Patch(old_patch))

        with builder.build():
            self.assertEqual(self.mock_module2.func(10), 30)


class TestPatchApplyAll(unittest.TestCase):
    """
    Test Patch.apply_all() classmethod.

    This method allows applying a Patch class directly without Patcher.
    """

    def setUp(self):
        """Create mock module for testing."""
        self.mock_module = types.ModuleType('apply_all_test')
        self.mock_module.func1 = lambda x: x
        self.mock_module.func2 = lambda x: x
        sys.modules['apply_all_test'] = self.mock_module

    def tearDown(self):
        """Clean up mock module."""
        if 'apply_all_test' in sys.modules:
            del sys.modules['apply_all_test']

    def test_apply_all_basic(self):
        """Test basic apply_all usage."""
        class MyPatch(Patch):
            name = "my_patch"

            @classmethod
            def patches(cls, options=None):
                return [
                    AtomicPatch("apply_all_test.func1", lambda x: x * 2),
                    AtomicPatch("apply_all_test.func2", lambda x: x * 3),
                ]

        results = MyPatch.apply_all()

        self.assertEqual(len(results), 2)
        self.assertEqual(self.mock_module.func1(10), 20)
        self.assertEqual(self.mock_module.func2(10), 30)

    def test_apply_all_with_options(self):
        """Test apply_all with options."""
        class MyPatch(Patch):
            name = "my_patch"

            @classmethod
            def patches(cls, options=None):
                multiplier = (options or {}).get('multiplier', 10)
                return [
                    AtomicPatch("apply_all_test.func1", lambda x, m=multiplier: x * m),
                ]

        results = MyPatch.apply_all(options={'multiplier': 5})

        self.assertEqual(len(results), 1)
        self.assertEqual(self.mock_module.func1(10), 50)


class TestLegacyPatchFunctions(unittest.TestCase):
    """
    Test old-style patch functions delegation.

    The old-style patch functions (msda, dc, mdc, etc.) should delegate
    to the new Patch class implementations.
    """

    def test_patch_functions_are_callable(self):
        """Test that old-style patch functions are callable."""
        # Import patch functions from __init__.py (where they are now defined)
        msda = _patcher_init_module.msda
        dc = _patcher_init_module.dc
        mdc = _patcher_init_module.mdc
        batch_matmul = _patcher_init_module.batch_matmul
        index = _patcher_init_module.index

        # All should be callable
        self.assertTrue(callable(msda))
        self.assertTrue(callable(dc))
        self.assertTrue(callable(mdc))
        self.assertTrue(callable(batch_matmul))
        self.assertTrue(callable(index))

    def test_patch_functions_signature(self):
        """Test that old-style patch functions have correct signature."""
        import inspect

        msda = _patcher_init_module.msda

        sig = inspect.signature(msda)
        params = list(sig.parameters.keys())

        # Should accept (module, options) - the auto-generated functions have these params
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0], 'module')
        self.assertEqual(params[1], 'options')


class TestDefaultPatcherBuilder(unittest.TestCase):
    """Test default_patcher_builder proxy."""

    def test_default_patcher_builder_exists(self):
        """Test that default_patcher_builder is accessible."""
        default_patcher_builder = _legacy_module.default_patcher_builder

        self.assertIsNotNone(default_patcher_builder)

    def test_default_patcher_builder_has_methods(self):
        """Test that default_patcher_builder has expected methods."""
        default_patcher_builder = _legacy_module.default_patcher_builder

        # Should have builder methods via proxy
        self.assertTrue(hasattr(default_patcher_builder, 'add_module_patch'))
        self.assertTrue(hasattr(default_patcher_builder, 'disable_patches'))
        self.assertTrue(hasattr(default_patcher_builder, 'build'))
        self.assertTrue(hasattr(default_patcher_builder, 'with_profiling'))
        self.assertTrue(hasattr(default_patcher_builder, 'brake_at'))

    def test_default_patch_classes_put_numpy_before_mmdet3d(self):
        """Default patch order should restore NumPy aliases before mmdet3d patches."""
        default_classes = _patcher_init_module._DEFAULT_PATCH_CLASSES

        numpy_idx = default_classes.index(_patcher_init_module.NumpyCompat)
        dataset_idx = default_classes.index(_patcher_init_module.NuScenesDataset)
        metric_idx = default_classes.index(_patcher_init_module.NuScenesMetric)

        self.assertLess(numpy_idx, dataset_idx)
        self.assertLess(numpy_idx, metric_idx)

    def test_default_patcher_builder_mirrors_numpy_before_mmdet3d(self):
        """Legacy default_patcher_builder should mirror the same safe ordering."""
        default_patcher_builder = _legacy_module.default_patcher_builder
        legacy_patcher = default_patcher_builder.build()
        collected = legacy_patcher._patcher._collect_all_patches()
        parent_names = [getattr(patch, "_parent_name", "") for patch, _ in collected]

        numpy_idx = parent_names.index("numpy_compat")
        dataset_idx = parent_names.index("nuscenes_dataset")
        metric_idx = parent_names.index("nuscenes_metric")

        self.assertLess(numpy_idx, dataset_idx)
        self.assertLess(numpy_idx, metric_idx)


class TestPatchFunctionToClassMapping(unittest.TestCase):
    """Test the mapping from old-style patch functions to new Patch classes."""

    def test_legacy_name_to_class_mapping_exists(self):
        """Test _LEGACY_NAME_TO_CLASS mapping exists and contains expected entries."""
        legacy_name_to_class = _patcher_init_module._LEGACY_NAME_TO_CLASS

        # Test a few known mappings
        known_names = ["msda", "dc", "mdc", "batch_matmul", "index"]
        for name in known_names:
            self.assertIn(name, legacy_name_to_class)
            # Result should be a class
            self.assertTrue(isinstance(legacy_name_to_class[name], type))

    def test_legacy_name_to_class_for_unknown_function(self):
        """Test _LEGACY_NAME_TO_CLASS returns KeyError for unknown functions."""
        legacy_name_to_class = _patcher_init_module._LEGACY_NAME_TO_CLASS

        self.assertNotIn("unknown_patch_function", legacy_name_to_class)


class TestLoggingIntegration(unittest.TestCase):
    """Test that logging works through the legacy API."""

    def setUp(self):
        """Create mock module for testing."""
        self.mock_module = types.ModuleType('logging_test')
        self.mock_module.func = lambda x: x
        sys.modules['logging_test'] = self.mock_module
        LegacyPatchWrapper._migration_warning_shown = False

    def tearDown(self):
        """Clean up mock module."""
        if 'logging_test' in sys.modules:
            del sys.modules['logging_test']

    def test_legacy_wrapper_shows_migration_warning(self):
        """Test that LegacyPatchWrapper shows migration warning."""
        # Reset the flag
        LegacyPatchWrapper._migration_warning_shown = False

        def my_patch(module, options):
            pass

        # First wrapper should trigger warning
        wrapper1 = LegacyPatchWrapper(my_patch)
        self.assertTrue(LegacyPatchWrapper._migration_warning_shown)

        # Second wrapper should not trigger warning again
        def another_patch(module, options):
            pass

        wrapper2 = LegacyPatchWrapper(another_patch)
        # Flag should still be True (warning was shown once)
        self.assertTrue(LegacyPatchWrapper._migration_warning_shown)

    def test_legacy_warning_mentions_compatibility_notice(self):
        """Migration warning should explain that it is informational, not a failure."""
        LegacyPatchWrapper._migration_warning_shown = False
        captured = []
        original_info = patcher_logger.info

        def fake_info(message):
            captured.append(message)

        patcher_logger.info = fake_info
        try:
            def my_patch(module, options):
                pass

            LegacyPatchWrapper(my_patch)
        finally:
            patcher_logger.info = original_info

        self.assertEqual(len(captured), 1)
        self.assertIn("compatibility-layer notice", captured[0])
        self.assertIn("not a patch failure", captured[0])


patcher_logger = _patcher_logger.patcher_logger


class TestPatchNameAutoDefault(unittest.TestCase):
    """Test Patch.name auto-default via __init_subclass__."""

    def test_name_defaults_to_class_name(self):
        """Patch subclass without explicit name gets cls.__name__."""
        class MyCustomPatch(Patch):
            @classmethod
            def patches(cls, options=None):
                return []

        self.assertEqual(MyCustomPatch.name, "MyCustomPatch")

    def test_explicit_name_preserved(self):
        """Patch subclass with explicit name keeps it."""
        class MyPatch(Patch):
            name = "my_explicit_name"

            @classmethod
            def patches(cls, options=None):
                return []

        self.assertEqual(MyPatch.name, "my_explicit_name")

    def test_two_subclasses_independent(self):
        """Each subclass gets its own default name."""
        class AlphaPatch(Patch):
            @classmethod
            def patches(cls, options=None):
                return []

        class BetaPatch(Patch):
            @classmethod
            def patches(cls, options=None):
                return []

        self.assertEqual(AlphaPatch.name, "AlphaPatch")
        self.assertEqual(BetaPatch.name, "BetaPatch")


class TestLegacyPatchReadableNames(unittest.TestCase):
    """LegacyPatch should infer readable names for internal helper closures."""

    def test_internal_apply_name_infers_from_builder(self):
        def build_mmcv_epoch_runner_patch():
            def _apply(module, _options):
                return None
            return LegacyPatch(_apply, target_module="mmcv")

        patch = build_mmcv_epoch_runner_patch()
        self.assertEqual(patch.name, "mmcv_epoch_runner")

    def test_explicit_patch_name_override_wins(self):
        def builder():
            def _apply(module, _options):
                return None
            _apply.__patch_name__ = "custom_patch_name"
            return LegacyPatch(_apply, target_module="mmcv")

        patch = builder()
        self.assertEqual(patch.name, "custom_patch_name")


class TestPatcherDisableEnhanced(unittest.TestCase):
    """Test Patcher.disable() with Patch classes and BasePatch instances."""

    def test_disable_by_patch_class(self):
        """patcher.disable(PatchClass) works."""
        mock_mod = types.ModuleType("disable_test_mod")
        mock_mod.func = lambda x: x
        sys.modules["disable_test_mod"] = mock_mod

        try:
            class MyPatch(Patch):
                name = "disable_test"

                @classmethod
                def patches(cls, options=None):
                    return [AtomicPatch("disable_test_mod.func", lambda x: x * 99)]

            p = Patcher()
            p.add(MyPatch)
            p.disable(MyPatch)
            p.apply()

            # func should NOT have been replaced
            self.assertEqual(mock_mod.func(1), 1)
        finally:
            del sys.modules["disable_test_mod"]

    def test_disable_by_patch_instance(self):
        """patcher.disable(atomic_patch_instance) works."""
        mock_mod = types.ModuleType("disable_inst_mod")
        mock_mod.func = lambda x: x
        sys.modules["disable_inst_mod"] = mock_mod

        try:
            ap = AtomicPatch("disable_inst_mod.func", lambda x: x * 99)
            p = Patcher()
            p.add(ap)
            p.disable(ap)
            p.apply()

            self.assertEqual(mock_mod.func(1), 1)
        finally:
            del sys.modules["disable_inst_mod"]


class TestGlobalInsertionOrder(unittest.TestCase):
    """Regression tests for global insertion order (tech debt #02)."""

    def setUp(self):
        self.mock_mod = types.ModuleType("order_global_test")
        self.mock_mod.func = lambda: "original"
        sys.modules["order_global_test"] = self.mock_mod

    def tearDown(self):
        sys.modules.pop("order_global_test", None)

    def test_direct_then_class_order(self):
        """add(direct); add(Class) → Class should win (added last)."""
        class MyPatch(Patch):
            name = "order_class"
            @classmethod
            def patches(cls, options=None):
                return [AtomicPatch("order_global_test.func", lambda: "from-class")]

        direct = AtomicPatch("order_global_test.func", lambda: "from-direct")

        p = Patcher()
        p.add(direct)
        p.add(MyPatch)
        p.apply()

        self.assertEqual(self.mock_mod.func(), "from-class",
                         "Patch class added AFTER direct should win")

    def test_class_then_direct_order(self):
        """add(Class); add(direct) → direct should win (added last)."""
        class MyPatch(Patch):
            name = "order_class"
            @classmethod
            def patches(cls, options=None):
                return [AtomicPatch("order_global_test.func", lambda: "from-class")]

        direct = AtomicPatch("order_global_test.func", lambda: "from-direct")

        p = Patcher()
        p.add(MyPatch)
        p.add(direct)
        p.apply()

        self.assertEqual(self.mock_mod.func(), "from-direct",
                         "Direct patch added AFTER class should win")

    def test_interleaved_order(self):
        """Interleaved add: direct → class → direct → collection order correct."""
        d1 = AtomicPatch("order_global_test.func", lambda: "d1")

        class CP(Patch):
            name = "cp"
            @classmethod
            def patches(cls, options=None):
                return [AtomicPatch("order_global_test.func", lambda: "cp")]

        d2 = AtomicPatch("order_global_test.func", lambda: "d2")

        p = Patcher()
        p.add(d1)
        p.add(CP)
        p.add(d2)

        names = [patch.name for patch, _ in p._collect_all_patches()]
        d1_idx = names.index("order_global_test.func")
        cp_idx = next(i for i, n in enumerate(names) if n == "order_global_test.func" and i > d1_idx)

        # d1 before cp before d2
        self.assertTrue(d1_idx < cp_idx, "d1 should come before class patch")


class TestConflictsWithEnforcement(unittest.TestCase):
    """Regression tests for conflicts_with enforcement (tech debt #03)."""

    def test_conflict_raises_on_apply(self):
        """Two conflicting patches should raise ValueError on apply."""
        class PatchA(Patch):
            name = "patch_a"
            conflicts_with = ["patch_b"]
            @classmethod
            def patches(cls, options=None):
                return []

        class PatchB(Patch):
            name = "patch_b"
            @classmethod
            def patches(cls, options=None):
                return []

        p = Patcher()
        p.add(PatchA, PatchB)
        with self.assertRaises(ValueError) as ctx:
            p.apply()
        self.assertIn("patch_a", str(ctx.exception))
        self.assertIn("patch_b", str(ctx.exception))
        self.assertIn("default_patcher", str(ctx.exception))
        self.assertIn("patcher.disable('patch_b').add(PatchA)", str(ctx.exception))

    def test_conflict_ok_when_one_disabled(self):
        """Conflict is resolved when one patch is disabled."""
        mock_mod = types.ModuleType("conflict_ok_mod")
        mock_mod.func = lambda: "original"
        sys.modules["conflict_ok_mod"] = mock_mod

        try:
            class PatchA(Patch):
                name = "patch_a"
                conflicts_with = ["patch_b"]
                @classmethod
                def patches(cls, options=None):
                    return [AtomicPatch("conflict_ok_mod.func", lambda: "a")]

            class PatchB(Patch):
                name = "patch_b"
                @classmethod
                def patches(cls, options=None):
                    return [AtomicPatch("conflict_ok_mod.func", lambda: "b")]

            p = Patcher()
            p.add(PatchA, PatchB)
            p.disable(PatchA)
            p.apply()  # Should not raise
            self.assertEqual(mock_mod.func(), "b")
        finally:
            sys.modules.pop("conflict_ok_mod", None)

    def test_no_conflict_no_error(self):
        """Patches without conflicts_with should work normally."""
        class PatchX(Patch):
            name = "patch_x"
            @classmethod
            def patches(cls, options=None):
                return []

        class PatchY(Patch):
            name = "patch_y"
            @classmethod
            def patches(cls, options=None):
                return []

        p = Patcher()
        p.add(PatchX, PatchY)
        p.apply()  # Should not raise

    def test_bidirectional_conflict_detected(self):
        """Conflict should be detected regardless of add order."""
        class PatchA(Patch):
            name = "patch_a"
            conflicts_with = ["patch_b"]
            @classmethod
            def patches(cls, options=None):
                return []

        class PatchB(Patch):
            name = "patch_b"
            @classmethod
            def patches(cls, options=None):
                return []

        # A then B
        p1 = Patcher()
        p1.add(PatchA, PatchB)
        with self.assertRaises(ValueError):
            p1.apply()

        # B then A
        p2 = Patcher()
        p2.add(PatchB, PatchA)
        with self.assertRaises(ValueError):
            p2.apply()

    def test_conflict_warning_on_add_includes_default_patcher_guidance(self):
        """Adding a conflicting patch should warn with a simple disable-then-add hint."""
        class PatchA(Patch):
            name = "patch_a"
            conflicts_with = ["patch_b"]

            @classmethod
            def patches(cls, options=None):
                return []

        class PatchB(Patch):
            name = "patch_b"

            @classmethod
            def patches(cls, options=None):
                return []

        p = Patcher()
        p.add(PatchB)

        with self.assertLogs("mx_driving.patcher", level="WARNING") as logs:
            p.add(PatchA)

        output = "\n".join(logs.output)
        self.assertIn("patch_a", output)
        self.assertIn("patch_b", output)
        self.assertIn("default_patcher", output)
        self.assertIn("patcher.disable('patch_b').add(PatchA)", output)


class TestLegacyUnknownLabelFix(unittest.TestCase):
    """
    Regression tests for legacy unknown-label fix.

    Covers plan Section 8 requirements:
    1. Custom legacy patch no longer grouped as 'unknown'
    2. Built-in legacy patch expansion with _parent_name
    3. Mixed built-in + custom order preservation
    """

    def setUp(self):
        LegacyPatchWrapper._migration_warning_shown = True

    def test_custom_legacy_patch_not_unknown(self):
        """Custom legacy patch should use func name as group, not 'unknown'."""
        mock_mod = types.ModuleType("custom_label_test")
        mock_mod.target_func = lambda: "original"
        sys.modules["custom_label_test"] = mock_mod

        try:
            def my_custom_patch(module, options):
                module.target_func = lambda: "patched"

            builder = LegacyPatcherBuilder()
            builder.add_module_patch("custom_label_test", Patch(my_custom_patch))
            legacy_patcher = builder.build()
            inner = legacy_patcher._patcher

            # Check _collect_all_patches parent_name
            all_patches = inner._collect_all_patches()
            parent_names = [pn for _, pn in all_patches]

            self.assertNotIn("", parent_names,
                             "parent_name should not be empty (would become 'unknown')")
            self.assertIn("my_custom_patch", parent_names,
                          "Custom legacy patch should use func name as parent_name")
        finally:
            del sys.modules["custom_label_test"]

    def test_builtin_legacy_patch_expanded_with_parent_name(self):
        """Built-in legacy patch should expand to child patches with _parent_name."""
        _LEGACY_NAME_TO_CLASS = _patcher_init_module._LEGACY_NAME_TO_CLASS

        # Pick a built-in legacy function that has a mapping
        batch_matmul_func = _patcher_init_module.batch_matmul
        self.assertIn(batch_matmul_func.__name__, _LEGACY_NAME_TO_CLASS,
                      "batch_matmul should be in _LEGACY_NAME_TO_CLASS")

        batch_matmul_cls = _LEGACY_NAME_TO_CLASS[batch_matmul_func.__name__]

        # Build via legacy API
        builder = LegacyPatcherBuilder()
        builder.add_module_patch("torch", Patch(batch_matmul_func))
        legacy_patcher = builder.build()

        # Check that the internal patcher has child patches with _parent_name
        inner_patcher = legacy_patcher._patcher
        all_patches = inner_patcher._collect_all_patches()
        child_patches = [p for p, _ in all_patches]

        self.assertTrue(len(child_patches) > 0,
                        "Built-in legacy patch should expand to child patches")

        for cp in child_patches:
            self.assertTrue(hasattr(cp, "_parent_name"),
                            f"Child patch {cp.name} should have _parent_name")
            self.assertEqual(cp._parent_name, batch_matmul_cls.name,
                             f"Child patch _parent_name should be '{batch_matmul_cls.name}'")

    def test_mixed_builtin_custom_order_preserved(self):
        """Mixed built-in + custom legacy patches preserve insertion order per module key."""
        mock_mod_a = types.ModuleType("order_mod_a")
        mock_mod_a.func = lambda: "a"
        mock_mod_b = types.ModuleType("order_mod_b")
        mock_mod_b.func = lambda: "b"
        sys.modules["order_mod_a"] = mock_mod_a
        sys.modules["order_mod_b"] = mock_mod_b

        _LEGACY_NAME_TO_CLASS = _patcher_init_module._LEGACY_NAME_TO_CLASS
        batch_matmul_func = _patcher_init_module.batch_matmul

        try:
            def custom_first(module, options):
                module.func = lambda: "patched_a"

            def custom_last(module, options):
                pass

            # Order: custom_first (mod_a) → batch_matmul (torch) → custom_last (mod_b)
            builder = LegacyPatcherBuilder()
            builder.add_module_patch("order_mod_a", Patch(custom_first))
            builder.add_module_patch("torch", Patch(batch_matmul_func))
            builder.add_module_patch("order_mod_b", Patch(custom_last))

            legacy_patcher = builder.build()
            inner_patcher = legacy_patcher._patcher

            # Collect patches in order
            all_patches = inner_patcher._collect_all_patches()
            names = [p.name for p, _ in all_patches]

            # custom_first should come before batch_matmul children
            custom_first_idx = names.index("custom_first")
            custom_last_idx = names.index("custom_last")

            # batch_matmul children should be between the two custom patches
            batch_matmul_cls = _LEGACY_NAME_TO_CLASS[batch_matmul_func.__name__]
            bm_child_names = {p.name for p in batch_matmul_cls.patches()}
            bm_indices = [i for i, n in enumerate(names) if n in bm_child_names]

            self.assertTrue(len(bm_indices) > 0, "batch_matmul should have child patches")
            self.assertTrue(all(custom_first_idx < i for i in bm_indices),
                            "custom_first should come before batch_matmul children")
            self.assertTrue(all(i < custom_last_idx for i in bm_indices),
                            "batch_matmul children should come before custom_last")
        finally:
            sys.modules.pop("order_mod_a", None)
            sys.modules.pop("order_mod_b", None)


class TestPatchMmcvVersion(unittest.TestCase):
    """Regression test for patch_mmcv_version wrapper bug (tech debt #01)."""

    def test_patch_mmcv_version_passes_argument(self):
        """patch_mmcv_version should forward expected_version, not builtin str."""
        patch_mmcv_version = _patcher_init_module.patch_mmcv_version

        # Inspect the bytecode to confirm it references expected_version, not str
        import dis
        code = patch_mmcv_version.__code__
        # co_names should NOT contain 'str' as a global reference
        self.assertNotIn("str", code.co_names,
                         "patch_mmcv_version should not reference builtin str")

    def test_patch_mmcv_version_calls_ensure_correctly(self):
        """patch_mmcv_version("1.6.0") should pass "1.6.0" to ensure_mmcv_version."""
        captured = []
        original = _patcher_init_module.ensure_mmcv_version

        def mock_ensure(version):
            captured.append(version)

        _patcher_init_module.ensure_mmcv_version = mock_ensure
        try:
            _patcher_init_module.patch_mmcv_version("1.6.0")
            self.assertEqual(captured, ["1.6.0"],
                             "patch_mmcv_version should forward the version string")
        finally:
            _patcher_init_module.ensure_mmcv_version = original

    def test_ensure_mmcv_version_no_mmcv(self):
        """ensure_mmcv_version should not raise when mmcv is not installed."""
        ensure = _patcher_init_module.ensure_mmcv_version
        # Remove mmcv from sys.modules temporarily
        orig = sys.modules.pop("mmcv", None)
        try:
            ensure("1.6.0")  # Should not raise
        finally:
            if orig is not None:
                sys.modules["mmcv"] = orig


if __name__ == "__main__":
    unittest.main()
