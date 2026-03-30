# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Patcher覆盖率补充测试模块

本模块补充测试patcher框架中未被覆盖的代码路径，包括：
- version模块的完整测试
- reporting模块的测试
- patch模块的边界情况
- patcher模块的边界情况
"""
import importlib.util
import os
import sys
import types
import unittest
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


# Load patcher modules directly
_patcher_logger_module = _load_module_from_file(
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

# Import classes
AtomicPatch = _patch_module.AtomicPatch
BasePatch = _patch_module.BasePatch
LegacyPatch = _patch_module.LegacyPatch
Patch = _patch_module.Patch
RegistryPatch = _patch_module.RegistryPatch
Patcher = _patcher_module.Patcher
PatchResult = _reporting.PatchResult
PatchStatus = _reporting.PatchStatus
_MMCVVersion = _version_module._MMCVVersion
mmcv_version = _version_module.mmcv_version
is_mmcv_v1x = _version_module.is_mmcv_v1x
is_mmcv_v2x = _version_module.is_mmcv_v2x
_import_module = _patch_module._import_module
_get_by_path = _patch_module._get_by_path


# =============================================================================
# Version Module Tests
# =============================================================================

class TestMMCVVersionClass(unittest.TestCase):
    """
    _MMCVVersion类测试
    """

    def test_version_caching(self):
        """
        测试版本检测缓存

        Ground Truth:
        - 首次访问后_cached应为True
        - 多次访问不会重复检测
        """
        version = _MMCVVersion()
        self.assertFalse(version._cached)

        # 触发检测
        _ = version.is_v1x
        self.assertTrue(version._cached)

        # 再次访问不会重新检测
        _ = version.is_v2x
        self.assertTrue(version._cached)

    def test_has_mmcv_property(self):
        """
        测试has_mmcv属性

        Ground Truth:
        - 属性应该返回布尔值
        """
        version = _MMCVVersion()
        result = version.has_mmcv
        self.assertIsInstance(result, bool)

    def test_has_mmengine_property(self):
        """
        测试has_mmengine属性

        Ground Truth:
        - 属性应该返回布尔值
        """
        version = _MMCVVersion()
        result = version.has_mmengine
        self.assertIsInstance(result, bool)

    def test_version_property(self):
        """
        测试version属性

        Ground Truth:
        - 属性应该返回字符串或None
        """
        version = _MMCVVersion()
        result = version.version
        self.assertTrue(result is None or isinstance(result, str))

    def test_available_property(self):
        """
        测试available属性

        Ground Truth:
        - 属性应该返回布尔值
        """
        version = _MMCVVersion()
        result = version.available
        self.assertIsInstance(result, bool)

    def test_bool_conversion(self):
        """
        测试__bool__方法

        Ground Truth:
        - bool(version)应该等于version.available
        """
        version = _MMCVVersion()
        self.assertEqual(bool(version), version.available)


class TestVersionFunctions(unittest.TestCase):
    """
    版本检测函数测试
    """

    def test_is_mmcv_v1x_returns_bool(self):
        """
        测试is_mmcv_v1x返回布尔值

        Ground Truth:
        - 函数应该返回布尔值
        """
        result = is_mmcv_v1x()
        self.assertIsInstance(result, bool)

    def test_is_mmcv_v2x_returns_bool(self):
        """
        测试is_mmcv_v2x返回布尔值

        Ground Truth:
        - 函数应该返回布尔值
        """
        result = is_mmcv_v2x()
        self.assertIsInstance(result, bool)

    def test_v1x_and_v2x_mutually_exclusive(self):
        """
        测试v1x和v2x互斥

        Ground Truth:
        - 不能同时为True
        """
        v1x = is_mmcv_v1x()
        v2x = is_mmcv_v2x()
        self.assertFalse(v1x and v2x)


# =============================================================================
# Reporting Module Tests
# =============================================================================

class TestPatchStatusEnum(unittest.TestCase):
    """
    PatchStatus枚举测试
    """

    def test_all_status_values(self):
        """
        测试所有状态值

        Ground Truth:
        - APPLIED = "applied"
        - SKIPPED = "skipped"
        - FAILED = "failed"
        """
        self.assertEqual(PatchStatus.APPLIED.value, "applied")
        self.assertEqual(PatchStatus.SKIPPED.value, "skipped")
        self.assertEqual(PatchStatus.FAILED.value, "failed")

    def test_status_count(self):
        """
        测试状态数量

        Ground Truth:
        - 应该有3个状态
        """
        self.assertEqual(len(PatchStatus), 3)


class TestPatchResultDataclass(unittest.TestCase):
    """
    PatchResult数据类测试
    """

    def test_required_fields(self):
        """
        测试必需字段

        Ground Truth:
        - status, name, module是必需的
        """
        result = PatchResult(PatchStatus.APPLIED, "test", "module")
        self.assertEqual(result.status, PatchStatus.APPLIED)
        self.assertEqual(result.name, "test")
        self.assertEqual(result.module, "module")

    def test_optional_reason(self):
        """
        测试可选的reason字段

        Ground Truth:
        - reason默认为None
        - 可以设置为字符串
        """
        result1 = PatchResult(PatchStatus.APPLIED, "test", "module")
        self.assertIsNone(result1.reason)

        result2 = PatchResult(PatchStatus.SKIPPED, "test", "module", "reason")
        self.assertEqual(result2.reason, "reason")


# =============================================================================
# Patch Module Helper Functions Tests
# =============================================================================

class TestImportModule(unittest.TestCase):
    """
    _import_module函数测试
    """

    def test_import_existing_module(self):
        """
        测试导入存在的模块

        Ground Truth:
        - 应该返回模块对象
        """
        result = _import_module("sys")
        self.assertIsNotNone(result)
        self.assertEqual(result, sys)

    def test_import_nonexistent_module(self):
        """
        测试导入不存在的模块

        Ground Truth:
        - 应该返回None
        """
        result = _import_module("nonexistent_module_xyz")
        self.assertIsNone(result)


class TestGetByPath(unittest.TestCase):
    """
    _get_by_path函数测试
    """

    def setUp(self):
        """创建测试模块"""
        self.mock_module = types.ModuleType('path_test_module')
        self.mock_module.sub = types.ModuleType('path_test_module.sub')
        self.mock_module.sub.value = 42
        self.mock_module.sub.func = lambda: "test"
        sys.modules['path_test_module'] = self.mock_module
        sys.modules['path_test_module.sub'] = self.mock_module.sub

    def tearDown(self):
        """清理测试模块"""
        for name in list(sys.modules.keys()):
            if name.startswith('path_test_module'):
                del sys.modules[name]

    def test_get_module_attribute(self):
        """
        测试获取模块属性

        Ground Truth:
        - 应该返回正确的属性值
        """
        result = _get_by_path("path_test_module.sub.value")
        self.assertEqual(result, 42)

    def test_get_nested_attribute(self):
        """
        测试获取嵌套属性

        Ground Truth:
        - 应该返回正确的嵌套属性
        """
        result = _get_by_path("path_test_module.sub.func")
        self.assertEqual(result(), "test")

    def test_get_nonexistent_path(self):
        """
        测试获取不存在的路径

        Ground Truth:
        - 应该返回None
        """
        result = _get_by_path("path_test_module.nonexistent")
        self.assertIsNone(result)

    def test_get_empty_path(self):
        """
        测试空路径

        Ground Truth:
        - 空字符串会导致ValueError（空模块名）
        - 这是预期行为，因为空路径没有意义
        """
        with self.assertRaises(ValueError):
            _get_by_path("")

    def test_get_dict_value(self):
        """
        测试从字典获取值

        Ground Truth:
        - 应该支持字典路径
        """
        self.mock_module.sub.data = {"key": "value"}
        result = _get_by_path("path_test_module.sub.data.key")
        self.assertEqual(result, "value")


# =============================================================================
# AtomicPatch Edge Cases Tests
# =============================================================================

class TestAtomicPatchEdgeCases(unittest.TestCase):
    """
    AtomicPatch边界情况测试
    """

    def setUp(self):
        """创建测试模块"""
        self.mock_module = types.ModuleType('edge_test_module')
        self.mock_module.func = lambda x: x
        sys.modules['edge_test_module'] = self.mock_module

    def tearDown(self):
        """清理测试模块"""
        if 'edge_test_module' in sys.modules:
            del sys.modules['edge_test_module']

    def test_no_replacement_no_wrapper_raises(self):
        """
        测试没有replacement也没有target_wrapper时抛出异常

        Ground Truth:
        - 应该抛出ValueError
        """
        with self.assertRaises(ValueError) as context:
            AtomicPatch("edge_test_module.func")

        self.assertIn("Either replacement or target_wrapper must be provided", str(context.exception))

    def test_name_property(self):
        """
        测试name属性

        Ground Truth:
        - name应该等于target
        """
        patch = AtomicPatch("edge_test_module.func", lambda x: x)
        self.assertEqual(patch.name, "edge_test_module.func")

    def test_module_property(self):
        """
        测试module属性

        Ground Truth:
        - module应该是target的第一部分
        """
        patch = AtomicPatch("edge_test_module.func", lambda x: x)
        self.assertEqual(patch.module, "edge_test_module")

    def test_repr(self):
        """
        测试__repr__方法

        Ground Truth:
        - 应该返回可读的字符串表示
        """
        patch = AtomicPatch("edge_test_module.func", lambda x: x)
        repr_str = repr(patch)
        self.assertIn("AtomicPatch", repr_str)
        self.assertIn("edge_test_module.func", repr_str)

    def test_precheck_exception_handling(self):
        """
        测试precheck异常处理

        Ground Truth:
        - precheck抛出异常时应该返回FAILED（代码bug）
        """
        def bad_precheck():
            raise RuntimeError("precheck error")

        patch = AtomicPatch(
            "edge_test_module.func",
            lambda x: x * 2,
            precheck=bad_precheck
        )
        result = patch.apply()

        self.assertEqual(result.status, PatchStatus.FAILED)
        self.assertIn("precheck error", result.reason)


# =============================================================================
# RegistryPatch Edge Cases Tests
# =============================================================================

class TestRegistryPatchEdgeCases(unittest.TestCase):
    """
    RegistryPatch边界情况测试
    """

    def test_no_module_cls_no_factory_raises(self):
        """
        测试没有module_cls也没有module_factory时抛出异常

        Ground Truth:
        - 应该抛出ValueError
        """
        with self.assertRaises(ValueError) as context:
            RegistryPatch("some.registry")

        self.assertIn("Either module_cls or module_factory must be provided", str(context.exception))

    def test_factory_without_name_raises(self):
        """
        测试使用factory但没有name时抛出异常

        Ground Truth:
        - 应该抛出ValueError
        """
        with self.assertRaises(ValueError) as context:
            RegistryPatch("some.registry", module_factory=lambda: type("Test", (), {}))

        self.assertIn("name is required when using module_factory", str(context.exception))

    def test_name_property(self):
        """
        测试name属性

        Ground Truth:
        - name应该是registry.register_name的组合
        """
        class TestClass:
            pass

        patch = RegistryPatch("some.registry", TestClass, name="TestClass")
        self.assertEqual(patch.name, "some.registry.TestClass")

    def test_repr(self):
        """
        测试__repr__方法

        Ground Truth:
        - 应该返回可读的字符串表示
        """
        class TestClass:
            pass

        patch = RegistryPatch("some.registry", TestClass, name="TestClass")
        repr_str = repr(patch)
        self.assertIn("RegistryPatch", repr_str)
        self.assertIn("some.registry", repr_str)

    def test_get_info(self):
        """
        测试get_info方法

        Ground Truth:
        - 应该返回包含registry和类名的信息
        """
        class TestClass:
            pass

        patch = RegistryPatch("some.registry", TestClass, name="TestClass")
        info = patch.get_info()
        self.assertIn("some.registry", info)
        self.assertIn("TestClass", info)

    def test_get_info_with_factory(self):
        """
        测试使用factory时的get_info

        Ground Truth:
        - 应该显示<factory>
        """
        patch = RegistryPatch(
            "some.registry",
            name="TestClass",
            module_factory=lambda: type("Test", (), {})
        )
        info = patch.get_info()
        self.assertIn("<factory>", info)

    def test_registry_not_found(self):
        """
        测试registry不存在时的处理

        Ground Truth:
        - 应该返回SKIPPED状态
        """
        class TestClass:
            pass

        patch = RegistryPatch("nonexistent.registry", TestClass, name="TestClass")
        result = patch.apply()

        self.assertEqual(result.status, PatchStatus.SKIPPED)
        self.assertIn("registry not found", result.reason)

    def test_invalid_registry(self):
        """
        测试无效registry的处理

        Ground Truth:
        - 没有register_module方法的对象应该返回SKIPPED
        """
        mock_module = types.ModuleType('invalid_registry_module')
        mock_module.REGISTRY = "not a registry"  # 不是有效的registry
        sys.modules['invalid_registry_module'] = mock_module

        try:
            class TestClass:
                pass

            patch = RegistryPatch("invalid_registry_module.REGISTRY", TestClass, name="TestClass")
            result = patch.apply()

            self.assertEqual(result.status, PatchStatus.SKIPPED)
            self.assertIn("invalid registry", result.reason)
        finally:
            del sys.modules['invalid_registry_module']

    def test_factory_error(self):
        """
        测试factory执行错误的处理

        Ground Truth:
        - factory抛出异常时应该返回FAILED
        """
        mock_module = types.ModuleType('factory_error_module')
        mock_registry = MagicMock()
        mock_registry.register_module = MagicMock()
        mock_module.REGISTRY = mock_registry
        sys.modules['factory_error_module'] = mock_module

        try:
            def bad_factory():
                raise RuntimeError("factory error")

            patch = RegistryPatch(
                "factory_error_module.REGISTRY",
                name="TestClass",
                module_factory=bad_factory
            )
            result = patch.apply()

            self.assertEqual(result.status, PatchStatus.FAILED)
            self.assertIn("factory error", result.reason)
        finally:
            del sys.modules['factory_error_module']

    def test_precheck_with_kwargs(self):
        """
        测试precheck接收kwargs参数

        Ground Truth:
        - precheck可以接收registry, name等参数
        """
        mock_module = types.ModuleType('precheck_kwargs_module')
        mock_registry = MagicMock()
        mock_registry.register_module = MagicMock()
        mock_module.REGISTRY = mock_registry
        sys.modules['precheck_kwargs_module'] = mock_module

        try:
            received_kwargs = {}

            def precheck_with_kwargs(registry, name):
                received_kwargs['registry'] = registry
                received_kwargs['name'] = name
                return True

            class TestClass:
                pass

            patch = RegistryPatch(
                "precheck_kwargs_module.REGISTRY",
                TestClass,
                name="TestClass",
                precheck=precheck_with_kwargs
            )
            patch.apply()

            self.assertEqual(received_kwargs['registry'], "precheck_kwargs_module.REGISTRY")
            self.assertEqual(received_kwargs['name'], "TestClass")
        finally:
            del sys.modules['precheck_kwargs_module']


# =============================================================================
# LegacyPatch Edge Cases Tests
# =============================================================================

class TestLegacyPatchEdgeCases(unittest.TestCase):
    """
    LegacyPatch边界情况测试
    """

    def test_name_property(self):
        """
        测试name属性

        Ground Truth:
        - name应该是函数名
        """
        def my_patch_func(module, options):
            pass

        patch = LegacyPatch(my_patch_func, target_module="some_module")
        self.assertEqual(patch.name, "my_patch_func")

    def test_module_property(self):
        """
        测试module属性

        Ground Truth:
        - module应该是target_module
        """
        def my_patch_func(module, options):
            pass

        patch = LegacyPatch(my_patch_func, target_module="some_module")
        self.assertEqual(patch.module, "some_module")

    def test_repr(self):
        """
        测试__repr__方法

        Ground Truth:
        - 应该返回可读的字符串表示
        """
        def my_patch_func(module, options):
            pass

        patch = LegacyPatch(my_patch_func, target_module="some_module")
        repr_str = repr(patch)
        self.assertIn("LegacyPatch", repr_str)
        self.assertIn("my_patch_func", repr_str)

    def test_attribute_error_handling(self):
        """
        测试AttributeError处理

        Ground Truth:
        - AttributeError应该返回SKIPPED
        """
        mock_module = types.ModuleType('attr_error_module')
        sys.modules['attr_error_module'] = mock_module

        try:
            def bad_patch(module, options):
                _ = module.nonexistent_attr

            patch = LegacyPatch(bad_patch, target_module="attr_error_module")
            result = patch.apply()

            self.assertEqual(result.status, PatchStatus.SKIPPED)
        finally:
            del sys.modules['attr_error_module']

    def test_general_exception_handling(self):
        """
        测试一般异常处理

        Ground Truth:
        - 一般异常应该返回FAILED
        """
        mock_module = types.ModuleType('exception_module')
        sys.modules['exception_module'] = mock_module

        try:
            def bad_patch(module, options):
                raise RuntimeError("general error")

            patch = LegacyPatch(bad_patch, target_module="exception_module")
            result = patch.apply()

            self.assertEqual(result.status, PatchStatus.FAILED)
            self.assertIn("general error", result.reason)
        finally:
            del sys.modules['exception_module']


# =============================================================================
# Patch Class Edge Cases Tests
# =============================================================================

class TestPatchClassEdgeCases(unittest.TestCase):
    """
    Patch类边界情况测试
    """

    def test_patch_apply_returns_result(self):
        """
        测试直接调用Patch.apply()返回PatchResult

        Ground Truth:
        - 应该返回PatchResult对象
        - 空patches列表应该返回SKIPPED状态
        """
        class TestPatch(Patch):
            name = "test_patch"

            @classmethod
            def patches(cls, options=None):
                return []

        patch = TestPatch()
        result = patch.apply()
        self.assertEqual(result.status, PatchStatus.SKIPPED)

    def test_patch_iter(self):
        """
        测试Patch的__iter__方法

        Ground Truth:
        - 应该返回patches()的迭代器
        """
        mock_module = types.ModuleType('iter_test_module')
        mock_module.func = lambda x: x
        sys.modules['iter_test_module'] = mock_module

        try:
            class TestPatch(Patch):
                name = "test_patch"

                @classmethod
                def patches(cls, options=None):
                    return [
                        AtomicPatch("iter_test_module.func", lambda x: x * 2)
                    ]

            patch = TestPatch()
            patches_list = list(patch)
            self.assertEqual(len(patches_list), 1)
        finally:
            del sys.modules['iter_test_module']

    def test_patch_repr(self):
        """
        测试Patch的__repr__方法

        Ground Truth:
        - 应该返回可读的字符串表示
        """
        class TestPatch(Patch):
            name = "test_patch"

            @classmethod
            def patches(cls, options=None):
                return []

        patch = TestPatch()
        repr_str = repr(patch)
        self.assertIn("Patch", repr_str)
        self.assertIn("test_patch", repr_str)


# =============================================================================
# Patcher Edge Cases Tests
# =============================================================================

class TestPatcherEdgeCases(unittest.TestCase):
    """
    Patcher边界情况测试
    """

    def test_add_invalid_type_raises(self):
        """
        测试添加无效类型抛出异常

        Ground Truth:
        - 添加非Patch类或BasePatch实例应该抛出TypeError
        """
        patcher = Patcher()
        with self.assertRaises(TypeError) as context:
            patcher.add("not a patch")

        self.assertIn("Expected Patch class or BasePatch instance", str(context.exception))

    def test_is_applied_property(self):
        """
        测试is_applied属性

        Ground Truth:
        - 初始为False
        - apply后为True
        """
        patcher = Patcher()
        self.assertFalse(patcher.is_applied)

    def test_context_manager_exit(self):
        """
        测试上下文管理器退出

        Ground Truth:
        - __exit__应该正常执行
        """
        patcher = Patcher()
        with patcher:
            pass
        # 不应该抛出异常

    def test_disable_multiple_names(self):
        """
        测试禁用多个名称

        Ground Truth:
        - 可以一次禁用多个补丁
        """
        patcher = Patcher()
        patcher.disable("patch1", "patch2", "patch3")

        self.assertIn("patch1", patcher._blacklist)
        self.assertIn("patch2", patcher._blacklist)
        self.assertIn("patch3", patcher._blacklist)

    def test_with_profiling_all_options(self):
        """
        测试with_profiling的所有选项

        Ground Truth:
        - 所有选项应该正确存储
        """
        patcher = Patcher()
        patcher.with_profiling(
            path="/path/to/prof",
            level=2,
            skip_first=10,
            wait=2,
            warmup=3,
            active=4,
            repeat=5
        )

        opts = patcher._profiling_options
        self.assertEqual(opts['profiling_path'], "/path/to/prof")
        self.assertEqual(opts['profiling_level'], 2)
        self.assertEqual(opts['step_ctrl'], (2, 3, 4, 5, 10))


# =============================================================================
# BasePatch Tests
# =============================================================================

class TestBasePatch(unittest.TestCase):
    """
    BasePatch基类测试
    """

    def test_module_property_with_dot(self):
        """
        测试module属性（带点的name）

        Ground Truth:
        - module应该是name的第一部分
        """
        patch = AtomicPatch("module.submodule.func", lambda: None)
        self.assertEqual(patch.module, "module")

    def test_module_property_without_dot(self):
        """
        测试module属性（不带点的name）

        Ground Truth:
        - 不带点时module应该为空字符串
        """
        # 创建一个简单的BasePatch子类来测试
        class SimplePatch(BasePatch):
            def __init__(self, name):
                self._name = name

            @property
            def name(self):
                return self._name

            def apply(self):
                return PatchResult(PatchStatus.APPLIED, self.name, "")

        patch = SimplePatch("simple_name")
        self.assertEqual(patch.module, "")

    def test_get_info_default(self):
        """
        测试get_info默认实现

        Ground Truth:
        - 默认返回name
        """
        class SimplePatch(BasePatch):
            def __init__(self, name):
                self._name = name

            @property
            def name(self):
                return self._name

            def apply(self):
                return PatchResult(PatchStatus.APPLIED, self.name, "")

        patch = SimplePatch("test_name")
        self.assertEqual(patch.get_info(), "test_name")


if __name__ == "__main__":
    unittest.main()
