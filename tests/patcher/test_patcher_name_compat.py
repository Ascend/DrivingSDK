import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import patch


_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_patcher_dir = os.path.join(_project_root, "mx_driving", "patcher")
_PATCHER_MODULE_NAMES = [
    "mx_driving.patcher",
    "mx_driving.patcher.patcher_logger",
    "mx_driving.patcher.reporting",
    "mx_driving.patcher.version",
    "mx_driving.patcher.patch",
    "mx_driving.patcher.patcher",
    "mx_driving.patcher.legacy",
]


def _load_module_from_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _backup_patcher_modules():
    return {name: sys.modules.get(name) for name in _PATCHER_MODULE_NAMES}


def _restore_patcher_modules(backup):
    for name in _PATCHER_MODULE_NAMES:
        sys.modules.pop(name, None)
    for name, module in backup.items():
        if module is not None:
            sys.modules[name] = module


def _load_patcher_modules():
    patcher_logger_module = _load_module_from_file(
        "mx_driving.patcher.patcher_logger",
        os.path.join(_patcher_dir, "patcher_logger.py"),
    )
    _load_module_from_file(
        "mx_driving.patcher.reporting",
        os.path.join(_patcher_dir, "reporting.py"),
    )
    _load_module_from_file(
        "mx_driving.patcher.version",
        os.path.join(_patcher_dir, "version.py"),
    )
    patch_module = _load_module_from_file(
        "mx_driving.patcher.patch",
        os.path.join(_patcher_dir, "patch.py"),
    )
    patcher_module = _load_module_from_file(
        "mx_driving.patcher.patcher",
        os.path.join(_patcher_dir, "patcher.py"),
    )
    legacy_module = _load_module_from_file(
        "mx_driving.patcher.legacy",
        os.path.join(_patcher_dir, "legacy.py"),
    )
    patcher_init_module = _load_module_from_file(
        "mx_driving.patcher",
        os.path.join(_patcher_dir, "__init__.py"),
    )
    return {
        "AtomicPatch": patch_module.AtomicPatch,
        "Patch": patch_module.Patch,
        "Patcher": patcher_module.Patcher,
        "LegacyPatcherBuilder": legacy_module.LegacyPatcherBuilder,
        "LegacyPatchWrapper": legacy_module.LegacyPatchWrapper,
        "patcher_logger": patcher_logger_module.patcher_logger,
        "patcher_init_module": patcher_init_module,
    }


def _clear_logger_buffers(patcher_logger):
    patcher_logger._applied_patches.clear()
    patcher_logger._skipped_patches.clear()
    patcher_logger._failed_patches.clear()
    patcher_logger._skipped_modules.clear()
    patcher_logger._injected_imports.clear()


class _BaseIsolatedPatcherTest(unittest.TestCase):
    def setUp(self):
        self._module_backup = _backup_patcher_modules()
        modules = _load_patcher_modules()
        self.AtomicPatch = modules["AtomicPatch"]
        self.Patch = modules["Patch"]
        self.Patcher = modules["Patcher"]
        self.LegacyPatcherBuilder = modules["LegacyPatcherBuilder"]
        self.LegacyPatchWrapper = modules["LegacyPatchWrapper"]
        self.patcher_logger = modules["patcher_logger"]
        self._patcher_init_module = modules["patcher_init_module"]
        _clear_logger_buffers(self.patcher_logger)

    def tearDown(self):
        _clear_logger_buffers(self.patcher_logger)
        _restore_patcher_modules(self._module_backup)


class TestPatchNameCompatibility(_BaseIsolatedPatcherTest):
    def setUp(self):
        super().setUp()
        self.mock_module = types.ModuleType("patch_name_compat_test")
        self.mock_module.func1 = lambda x: x
        self.mock_module.func2 = lambda x: x
        sys.modules["patch_name_compat_test"] = self.mock_module

    def tearDown(self):
        sys.modules.pop("patch_name_compat_test", None)
        super().tearDown()

    def test_patch_name_defaults_to_class_name(self):
        class AutoNamedPatch(self.Patch):
            @classmethod
            def patches(cls, options=None):
                return [self.AtomicPatch("patch_name_compat_test.func1", lambda x: x * 2)]

        self.assertEqual(AutoNamedPatch.name, "AutoNamedPatch")

        patcher = self.Patcher()
        patcher.add(AutoNamedPatch).apply()

        self.assertEqual(self.mock_module.func1(5), 10)

    def test_explicit_patch_name_is_preserved(self):
        class ExplicitlyNamedPatch(self.Patch):
            name = "stable_patch_id"

            @classmethod
            def patches(cls, options=None):
                return [self.AtomicPatch("patch_name_compat_test.func1", lambda x: x * 3)]

        self.assertEqual(ExplicitlyNamedPatch.name, "stable_patch_id")

    def test_disable_accepts_patch_classes_and_patch_instances(self):
        class AutoDisabledPatch(self.Patch):
            @classmethod
            def patches(cls, options=None):
                return [self.AtomicPatch("patch_name_compat_test.func1", lambda x: x * 5)]

        direct_patch = self.AtomicPatch("patch_name_compat_test.func2", lambda x: x * 7)

        patcher = self.Patcher()
        patcher.add(AutoDisabledPatch, direct_patch)
        patcher.disable(AutoDisabledPatch, direct_patch)
        patcher.apply()

        self.assertIn("AutoDisabledPatch", patcher._blacklist)
        self.assertIn("patch_name_compat_test.func2", patcher._blacklist)
        self.assertEqual(self.mock_module.func1(5), 5)
        self.assertEqual(self.mock_module.func2(5), 5)


class TestLegacyUnknownLabelCompatibility(_BaseIsolatedPatcherTest):
    def setUp(self):
        super().setUp()
        self.mock_module = types.ModuleType("legacy_label_test")
        self.mock_module.func = lambda x: x
        sys.modules["legacy_label_test"] = self.mock_module
        self._legacy_name_to_class = self._patcher_init_module._LEGACY_NAME_TO_CLASS
        self._legacy_name_to_class_backup = dict(self._legacy_name_to_class)
        self.LegacyPatchWrapper._migration_warning_shown = False

    def tearDown(self):
        self._legacy_name_to_class.clear()
        self._legacy_name_to_class.update(self._legacy_name_to_class_backup)
        sys.modules.pop("legacy_label_test", None)
        super().tearDown()

    def test_custom_legacy_patch_uses_its_own_group_name(self):
        def custom_legacy(module, options):
            module.func = lambda x: x * 3

        builder = self.LegacyPatcherBuilder()
        builder.add_module_patch("legacy_label_test", self.Patch(custom_legacy))

        with patch.object(self.patcher_logger, "flush_summary", return_value=None):
            builder.build().apply()

        self.assertEqual(self.mock_module.func(5), 15)
        self.assertEqual(len(self.patcher_logger._applied_patches), 1)
        info = self.patcher_logger._applied_patches[0]
        self.assertEqual(info.patch_name, "custom_legacy")
        self.assertEqual(info.target, "custom_legacy")
        self.assertEqual(info.package, "legacy_label_test")

    def test_known_legacy_patch_expands_to_detailed_children(self):
        class DetailedPatch(self.Patch):
            name = "detailed_patch"

            @classmethod
            def patches(cls, options=None):
                multiplier = (options or {}).get("multiplier", 4)
                return [
                    self.AtomicPatch("legacy_label_test.func", lambda x, m=multiplier: x * m),
                ]

        def mapped_legacy(module, options):
            raise AssertionError("Known legacy patch should expand via its Patch class")

        self._legacy_name_to_class["mapped_legacy"] = DetailedPatch

        builder = self.LegacyPatcherBuilder()
        builder.add_module_patch(
            "legacy_label_test",
            self.Patch(mapped_legacy, options={"multiplier": 4}),
        )

        with patch.object(self.patcher_logger, "flush_summary", return_value=None):
            builder.build().apply()

        self.assertEqual(self.mock_module.func(5), 20)
        self.assertEqual(len(self.patcher_logger._applied_patches), 1)
        info = self.patcher_logger._applied_patches[0]
        self.assertEqual(info.patch_name, "detailed_patch")
        self.assertEqual(info.target, "legacy_label_test.func")
        self.assertEqual(info.package, "legacy_label_test")

    def test_known_and_custom_legacy_patches_keep_legacy_order(self):
        class MappedSecondPatch(self.Patch):
            name = "mapped_second_patch"

            @classmethod
            def patches(cls, options=None):
                return [self.AtomicPatch("legacy_label_test.func", lambda x: "mapped-second")]

        def custom_first(module, options):
            module.func = lambda x: "custom-first"

        def mapped_second(module, options):
            raise AssertionError("Known legacy patch should expand via its Patch class")

        self._legacy_name_to_class["mapped_second"] = MappedSecondPatch

        builder = self.LegacyPatcherBuilder()
        builder.add_module_patch(
            "legacy_label_test",
            self.Patch(custom_first, priority=1),
            self.Patch(mapped_second, priority=2),
        )

        with patch.object(self.patcher_logger, "flush_summary", return_value=None):
            builder.build().apply()

        self.assertEqual(self.mock_module.func(0), "mapped-second")
        self.assertEqual(
            [info.patch_name for info in self.patcher_logger._applied_patches],
            ["custom_first", "mapped_second_patch"],
        )


if __name__ == "__main__":
    unittest.main()
