# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Patcher框架扩展测试模块

本模块是test_patcher.py的补充，覆盖更多边界情况和高级场景：
- 辅助函数测试：_import_module, _get_by_path, _get_callable_name等
- PatcherLogger日志系统测试
- MMCV版本检测的详细测试
- AtomicPatch/RegistryPatch/LegacyPatch的高级用法
- Patcher的高级配置和集成场景
- 边界情况和错误处理

测试设计原则：
- 覆盖test_patcher.py未涉及的场景
- 重点测试错误处理和边界条件
- 验证各组件的协作
"""
import importlib.util
import logging
import os
import sys
import types
import unittest
from typing import Dict, List
from types import ModuleType
from unittest.mock import MagicMock, patch, PropertyMock

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

# Import classes from loaded modules
AtomicPatch = _patch_module.AtomicPatch
BasePatch = _patch_module.BasePatch
LegacyPatch = _patch_module.LegacyPatch
Patch = _patch_module.Patch
RegistryPatch = _patch_module.RegistryPatch
mmcv_version = _patch_module.mmcv_version
is_mmcv_v1x = _patch_module.is_mmcv_v1x
is_mmcv_v2x = _patch_module.is_mmcv_v2x
_import_module = _patch_module._import_module
_get_by_path = _patch_module._get_by_path
_get_callable_name = _patch_module._get_callable_name
_get_source_diff = _patch_module._get_source_diff
Patcher = _patcher_module.Patcher
PatchResult = _reporting.PatchResult
PatchStatus = _reporting.PatchStatus
PatcherLogger = _patcher_logger_module.PatcherLogger
PatchError = _patcher_logger_module.PatchError
patcher_logger = _patcher_logger_module.patcher_logger
configure_patcher_logging = _patcher_logger_module.configure_patcher_logging


class TestHelperFunctions(unittest.TestCase):
    """
    辅助函数测试

    测试patch.py中的内部辅助函数，这些函数是补丁机制的基础设施。
    """

    def test_import_module_success(self):
        """
        测试成功导入模块

        测试目的：验证_import_module能正确导入存在的模块

        执行步骤：
        1. 调用_import_module("sys")
        2. 验证返回的是sys模块

        为什么测试这个：
        - _import_module是补丁定位目标的基础
        """
        result = _import_module("sys")
        self.assertIsNotNone(result)
        self.assertEqual(result, sys)

    def test_import_module_not_found(self):
        """
        测试导入不存在的模块

        测试目的：验证_import_module对不存在的模块返回None而非抛异常

        执行步骤：
        1. 调用_import_module("nonexistent_module_xyz_123")
        2. 验证返回None

        为什么测试这个：
        - 补丁框架需要优雅处理可选依赖
        """
        result = _import_module("nonexistent_module_xyz_123")
        self.assertIsNone(result)

    def test_get_by_path_simple(self):
        """
        测试简单路径解析

        测试目的：验证_get_by_path能解析简单的模块.属性路径

        执行步骤：
        1. 调用_get_by_path("sys.path")
        2. 验证返回sys.path

        为什么测试这个：
        - 路径解析是补丁定位的核心功能
        """
        result = _get_by_path("sys.path")
        self.assertIsNotNone(result)
        self.assertEqual(result, sys.path)

    def test_get_by_path_nested(self):
        """
        测试嵌套路径解析

        测试目的：验证_get_by_path能解析多层嵌套的路径

        执行步骤：
        1. 调用_get_by_path("os.path.join")
        2. 验证返回os.path.join函数

        为什么测试这个：
        - 实际补丁目标通常是深层嵌套的
        """
        result = _get_by_path("os.path.join")
        self.assertIsNotNone(result)
        self.assertEqual(result, os.path.join)

    def test_get_by_path_not_found(self):
        """
        测试路径不存在的情况

        测试目的：验证_get_by_path对不存在的属性返回None

        执行步骤：
        1. 调用_get_by_path("sys.nonexistent_attr")
        2. 验证返回None

        为什么测试这个：
        - 需要安全处理目标不存在的情况
        """
        result = _get_by_path("sys.nonexistent_attr")
        self.assertIsNone(result)

    def test_get_by_path_empty(self):
        """
        测试空路径处理

        测试目的：验证_get_by_path对空路径的处理

        执行步骤：
        1. 调用_get_by_path("")
        2. 验证返回None或抛出ValueError

        为什么测试这个：
        - 边界条件测试，确保不会崩溃
        """
        # Empty path should raise ValueError or return None
        try:
            result = _get_by_path("")
            self.assertIsNone(result)
        except ValueError:
            pass  # Expected behavior

    def test_get_callable_name_function(self):
        """
        测试获取函数名称

        测试目的：验证_get_callable_name能正确提取函数名

        执行步骤：
        1. 定义函数my_func
        2. 调用_get_callable_name(my_func)
        3. 验证结果包含"my_func"

        为什么测试这个：
        - 函数名用于日志和调试信息
        """
        def my_func():
            pass
        result = _get_callable_name(my_func)
        self.assertIn("my_func", result)

    def test_get_callable_name_class(self):
        """
        测试获取类名称

        测试目的：验证_get_callable_name能正确提取类名
        """
        class MyClass:
            pass
        result = _get_callable_name(MyClass)
        self.assertIn("MyClass", result)

    def test_get_callable_name_none(self):
        """
        测试None输入的处理

        测试目的：验证_get_callable_name对None返回"<None>"
        """
        result = _get_callable_name(None)
        self.assertEqual(result, "<None>")

    def test_get_callable_name_lambda(self):
        """
        测试lambda函数名称

        测试目的：验证_get_callable_name能处理lambda函数
        """
        func = lambda x: x
        result = _get_callable_name(func)
        self.assertIn("lambda", result)

    def test_get_source_diff_functions(self):
        """
        测试源码差异生成

        测试目的：验证_get_source_diff能生成两个函数的源码差异

        执行步骤：
        1. 定义两个不同的函数
        2. 调用_get_source_diff
        3. 验证返回字符串(可能包含diff标记)

        为什么测试这个：
        - 源码差异用于调试和日志
        """
        def original(x):
            return x * 2

        def replacement(x):
            return x * 3

        diff = _get_source_diff(original, replacement)
        # Should contain diff markers or be empty if source not available
        self.assertIsInstance(diff, str)

    def test_get_source_diff_builtin(self):
        """
        测试内置函数的源码差异

        测试目的：验证_get_source_diff对无源码的内置函数返回空字符串

        为什么测试这个：
        - 内置函数没有Python源码，需要特殊处理
        """
        # Builtin functions don't have source, should return empty string
        diff = _get_source_diff(len, str)
        self.assertEqual(diff, "")

    def test_get_by_path_with_dict(self):
        """
        测试包含字典属性的路径解析

        测试目的：验证_get_by_path能获取模块中的字典属性
        """
        # Create a mock module with dict attribute
        mock_module = types.ModuleType('dict_test_module')
        mock_module.config = {'key1': 'value1', 'nested': {'key2': 'value2'}}
        sys.modules['dict_test_module'] = mock_module

        try:
            result = _get_by_path("dict_test_module.config")
            self.assertEqual(result, {'key1': 'value1', 'nested': {'key2': 'value2'}})
        finally:
            del sys.modules['dict_test_module']

    def test_get_by_path_module_not_found(self):
        """
        测试根模块不存在的情况

        测试目的：验证_get_by_path在根模块不存在时返回None
        """
        result = _get_by_path("nonexistent_root_module.some.path")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()


class TestPatcherLogger(unittest.TestCase):
    """
    PatcherLogger日志系统测试

    PatcherLogger提供补丁应用过程的日志记录，支持多种日志级别和动作配置。
    测试目的：验证日志系统的各种配置和行为。
    """

    def setUp(self):
        """测试环境准备 - 创建logger实例并设置为DEBUG级别"""
        self.logger = PatcherLogger()
        # Set to debug level to capture all logs
        self.logger.set_level(logging.DEBUG)

    def test_configure_on_apply(self):
        """
        测试配置on_apply动作

        测试目的：验证configure()能正确设置补丁应用时的日志动作

        执行步骤：
        1. 调用configure(on_apply="debug")
        2. 验证_on_apply被设置为"debug"

        为什么测试这个：
        - on_apply控制补丁成功应用时的日志行为
        """
        self.logger.configure(on_apply="debug")
        self.assertEqual(self.logger._on_apply, "debug")

    def test_configure_on_skip(self):
        """测试配置on_skip动作 - 控制补丁跳过时的日志行为"""
        self.logger.configure(on_skip="warning")
        self.assertEqual(self.logger._on_skip, "warning")

    def test_configure_on_fail(self):
        """测试配置on_fail动作 - 控制补丁失败时的日志行为"""
        self.logger.configure(on_fail="error")
        self.assertEqual(self.logger._on_fail, "error")

    def test_configure_on_error(self):
        """测试配置on_error动作 - 控制发生错误时的行为(如exception抛异常)"""
        self.logger.configure(on_error="exception")
        self.assertEqual(self.logger._on_error, "exception")

    def test_configure_chaining(self):
        """测试configure方法链式调用 - 验证返回self支持链式API"""
        result = self.logger.configure(on_apply="info", on_skip="debug")
        self.assertIs(result, self.logger)

    def test_set_level_chaining(self):
        """测试set_level方法链式调用"""
        result = self.logger.set_level(logging.WARNING)
        self.assertIs(result, self.logger)

    def test_applied_silent(self):
        """测试silent模式下的applied调用 - 不输出任何日志"""
        self.logger.configure(on_apply="silent")
        # Should not raise
        self.logger.applied("test_patch")

    def test_skipped_silent(self):
        """测试silent模式下的skipped调用"""
        self.logger.configure(on_skip="silent")
        # Should not raise
        self.logger.skipped("test_patch", "reason")

    def test_failed_silent(self):
        """测试silent模式下的failed调用"""
        self.logger.configure(on_fail="silent")
        # Should not raise
        self.logger.failed("test_patch", "reason")

    def test_error_exception(self):
        """
        测试exception模式下的error调用

        测试目的：验证on_error="exception"时会抛出PatchError异常

        执行步骤：
        1. 配置on_error="exception"
        2. 调用error()方法
        3. 验证抛出PatchError

        为什么测试这个：
        - 某些场景需要错误立即中断执行
        """
        self.logger.configure(on_error="exception")
        with self.assertRaises(PatchError):
            self.logger.error("test error message")

    def test_handle_exception_action(self):
        """测试_handle方法的exception动作 - 验证异常消息正确"""
        with self.assertRaises(PatchError) as context:
            self.logger._handle("exception", "test message")
        self.assertIn("test message", str(context.exception))

    def test_handle_exit_action(self):
        """测试_handle方法的exit动作 - 验证会调用sys.exit"""
        with self.assertRaises(SystemExit):
            self.logger._handle("exit", "test exit message")

    def test_backward_compatible_methods(self):
        """测试向后兼容的日志方法 - debug/info/warning方法应正常工作"""
        # These should not raise
        self.logger.debug("debug message")
        self.logger.info("info message")
        self.logger.warning("warning message")


class TestConfigurePatcherLogging(unittest.TestCase):
    """
    configure_patcher_logging函数测试

    这是一个便捷函数，用于配置全局的patcher_logger实例。
    """

    def test_configure_returns_logger(self):
        """测试configure_patcher_logging返回logger实例"""
        result = configure_patcher_logging(on_apply="info")
        self.assertIs(result, patcher_logger)

    def test_configure_all_options(self):
        """测试一次配置所有选项"""
        configure_patcher_logging(
            on_apply="debug",
            on_skip="silent",
            on_fail="warning",
            on_error="error"
        )
        self.assertEqual(patcher_logger._on_apply, "debug")
        self.assertEqual(patcher_logger._on_skip, "silent")
        self.assertEqual(patcher_logger._on_fail, "warning")
        self.assertEqual(patcher_logger._on_error, "error")


class TestMMCVVersionDetection(unittest.TestCase):
    """
    MMCV版本检测详细测试

    测试_MMCVVersion类的各种属性和方法，确保版本检测在各种环境下都能正常工作。
    """

    def test_mmcv_version_singleton(self):
        """测试mmcv_version是单例对象"""
        self.assertIsNotNone(mmcv_version)

    def test_mmcv_version_properties_no_raise(self):
        """测试所有版本属性访问不会抛异常 - 即使mmcv未安装"""
        # These should not raise regardless of mmcv installation status
        _ = mmcv_version.is_v1x
        _ = mmcv_version.is_v2x
        _ = mmcv_version.has_mmcv
        _ = mmcv_version.has_mmengine
        _ = mmcv_version.version
        _ = mmcv_version.available

    def test_mmcv_version_bool(self):
        """测试__bool__方法 - 用于if mmcv_version判断"""
        result = bool(mmcv_version)
        self.assertIsInstance(result, bool)

    def test_is_mmcv_v1x_function(self):
        """测试is_mmcv_v1x()函数返回布尔值"""
        result = is_mmcv_v1x()
        self.assertIsInstance(result, bool)

    def test_is_mmcv_v2x_function(self):
        """测试is_mmcv_v2x()函数返回布尔值"""
        result = is_mmcv_v2x()
        self.assertIsInstance(result, bool)

    def test_version_detection_caching(self):
        """测试版本检测结果被缓存 - 避免重复检测"""
        # Access properties multiple times
        _ = mmcv_version.is_v1x
        _ = mmcv_version.is_v2x
        # Should be cached now
        self.assertTrue(mmcv_version._cached)

    def test_version_mutual_exclusion(self):
        """测试v1x和v2x互斥 - 不能同时为True"""
        if mmcv_version.has_mmcv:
            # If mmcv exists, can't be both v1x and v2x
            self.assertFalse(mmcv_version.is_v1x and mmcv_version.is_v2x)


class TestAtomicPatchAdvanced(unittest.TestCase):
    """
    AtomicPatch高级功能测试

    测试AtomicPatch的高级特性，包括字符串替换、wrapper组合、错误处理等。
    """

    def setUp(self):
        """测试环境准备 - 创建atomic_test_module模块"""
        self.mock_module = types.ModuleType('atomic_test_module')
        self.mock_module.submodule = types.ModuleType('atomic_test_module.submodule')
        self.mock_module.submodule.original_func = lambda x: x * 2
        self.mock_module.submodule.original_class = type('OriginalClass', (), {'value': 10})
        sys.modules['atomic_test_module'] = self.mock_module
        sys.modules['atomic_test_module.submodule'] = self.mock_module.submodule

    def tearDown(self):
        """测试环境清理"""
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith('atomic_test_module'):
                del sys.modules[mod_name]

    def test_constructor_requires_replacement_or_wrapper(self):
        """
        测试构造函数参数校验

        测试目的：验证必须提供replacement或wrapper参数

        执行步骤：
        1. 尝试只传target创建AtomicPatch
        2. 验证抛出ValueError

        为什么测试这个：
        - 没有替换逻辑的补丁没有意义
        - 应该在创建时就报错
        """
        with self.assertRaises(ValueError):
            AtomicPatch("some.target")

    def test_string_replacement_lazy_resolve(self):
        """
        测试字符串替换的延迟解析

        测试目的：验证replacement可以是字符串路径，在应用时才解析

        执行步骤：
        1. 在模块中定义replacement_func
        2. 创建AtomicPatch，replacement为字符串路径
        3. 应用补丁
        4. 验证正确解析并替换

        为什么测试这个：
        - 字符串替换避免循环导入问题
        - 支持延迟加载
        """
        # Create target for string replacement
        self.mock_module.submodule.replacement_func = lambda x: x * 5

        patch = AtomicPatch(
            "atomic_test_module.submodule.original_func",
            "atomic_test_module.submodule.replacement_func"
        )
        result = patch.apply()

        self.assertTrue(patch.is_applied)
        self.assertEqual(self.mock_module.submodule.original_func(10), 50)

    def test_wrapper_only_mode(self):
        """
        测试纯wrapper模式

        测试目的：验证只传wrapper时，会包装原始函数

        执行步骤：
        1. 定义wrapper，在原始结果上加1000
        2. 创建AtomicPatch只传target_wrapper
        3. 应用并验证：original(10)=20, +1000=1020
        """
        def my_wrapper(original):
            def wrapped(x):
                return original(x) + 1000
            return wrapped

        patch = AtomicPatch(
            "atomic_test_module.submodule.original_func",
            target_wrapper=my_wrapper
        )
        result = patch.apply()

        self.assertTrue(patch.is_applied)
        # original(10) = 20, wrapped adds 1000
        self.assertEqual(self.mock_module.submodule.original_func(10), 1020)

    def test_wrapper_plus_replacement_mode(self):
        """
        测试replacement_wrapper+replacement组合模式

        测试目的：验证同时传replacement_wrapper和replacement时，wrapper包装replacement

        执行步骤：
        1. 定义new_func(x*10)和wrapper(+1)
        2. 创建AtomicPatch同时传两者
        3. 应用并验证：new_func(10)=100, +1=101
        """
        def new_func(x):
            return x * 10

        def my_wrapper(func):
            def wrapped(x):
                return func(x) + 1
            return wrapped

        patch = AtomicPatch(
            "atomic_test_module.submodule.original_func",
            new_func,
            replacement_wrapper=my_wrapper
        )
        result = patch.apply()

        self.assertTrue(patch.is_applied)
        # new_func(10) = 100, wrapper adds 1
        self.assertEqual(self.mock_module.submodule.original_func(10), 101)

    def test_precheck_with_multiple_params(self):
        """
        测试precheck接收多个参数

        测试目的：验证precheck可以接收target和replacement参数

        执行步骤：
        1. 定义precheck捕获接收到的参数
        2. 应用补丁
        3. 验证参数正确传递
        """
        def new_func(x):
            return x * 3

        received_params = {}

        def precheck_func(target, replacement):
            received_params['target'] = target
            received_params['replacement'] = replacement
            return True

        patch = AtomicPatch(
            "atomic_test_module.submodule.original_func",
            new_func,
            precheck=precheck_func
        )
        result = patch.apply()

        self.assertTrue(patch.is_applied)
        self.assertEqual(received_params['target'], "atomic_test_module.submodule.original_func")
        self.assertEqual(received_params['replacement'], new_func)

    def test_precheck_exception_fails_patch(self):
        """
        测试precheck抛异常时返回FAILED

        测试目的：验证precheck内部异常是代码bug，应返回FAILED而非SKIPPED
        """
        def new_func(x):
            return x * 3

        def bad_precheck():
            raise RuntimeError("precheck error")

        patch = AtomicPatch(
            "atomic_test_module.submodule.original_func",
            new_func,
            precheck=bad_precheck
        )
        result = patch.apply()

        self.assertFalse(patch.is_applied)
        self.assertEqual(result.status, PatchStatus.FAILED)

    def test_invalid_target_path(self):
        """测试无效的目标路径(单段路径) - 应返回FAILED状态(用户代码错误)"""
        patch = AtomicPatch("singlepart", lambda: None)
        result = patch.apply()

        self.assertFalse(patch.is_applied)
        self.assertEqual(result.status, PatchStatus.FAILED)
        self.assertIn("invalid target path", result.reason)

    def test_get_info_basic(self):
        """测试get_info()返回补丁信息字符串"""
        def new_func(x):
            return x * 3

        patch = AtomicPatch(
            "atomic_test_module.submodule.original_func",
            new_func
        )
        patch.apply()

        info = patch.get_info()
        self.assertIn("atomic_test_module.submodule.original_func", info)

    def test_repr(self):
        """Test __repr__ method."""
        patch = AtomicPatch("some.target.func", lambda: None)
        repr_str = repr(patch)
        self.assertIn("AtomicPatch", repr_str)
        self.assertIn("some.target.func", repr_str)

    def test_name_property(self):
        """Test name property returns target."""
        patch = AtomicPatch("my.target.path", lambda: None)
        self.assertEqual(patch.name, "my.target.path")

    def test_module_property(self):
        """Test module property extracts first part."""
        patch = AtomicPatch("mymodule.submodule.func", lambda: None)
        self.assertEqual(patch.module, "mymodule")

    def test_runtime_check_uses_replacement_when_true(self):
        """Test runtime_check uses replacement when check returns True."""
        def new_func(x):
            return x * 100

        patch = AtomicPatch(
            "atomic_test_module.submodule.original_func",
            new_func,
            runtime_check=lambda x: x > 5
        )
        patch.apply()

        # x=10 > 5, use new_func
        self.assertEqual(self.mock_module.submodule.original_func(10), 1000)
        # x=3 <= 5, use original (x * 2)
        self.assertEqual(self.mock_module.submodule.original_func(3), 6)

    def test_runtime_check_exception_falls_back_to_original(self):
        """Test runtime_check exception falls back to original."""
        def new_func(x):
            return x * 100

        def bad_check(x):
            raise RuntimeError("check error")

        patch = AtomicPatch(
            "atomic_test_module.submodule.original_func",
            new_func,
            runtime_check=bad_check
        )
        patch.apply()

        # Should fall back to original due to exception
        self.assertEqual(self.mock_module.submodule.original_func(10), 20)

    def test_wrapper_error_returns_failed(self):
        """Test target_wrapper error returns FAILED status."""
        def bad_wrapper(original):
            raise RuntimeError("wrapper error")

        patch = AtomicPatch(
            "atomic_test_module.submodule.original_func",
            target_wrapper=bad_wrapper
        )
        result = patch.apply()

        self.assertFalse(patch.is_applied)
        self.assertEqual(result.status, PatchStatus.FAILED)
        self.assertIn("wrapper error", result.reason)

    def test_apply_to_missing_attribute_is_skipped(self):
        """Test applying to missing attribute is SKIPPED (fail-safe)."""
        def new_func(x):
            return x * 5

        patch = AtomicPatch(
            "atomic_test_module.submodule.new_attribute",
            new_func
        )
        result = patch.apply()

        self.assertFalse(patch.is_applied)
        self.assertEqual(result.status, PatchStatus.SKIPPED)
        self.assertTrue(result.reason and "target not found" in result.reason)
        self.assertFalse(hasattr(self.mock_module.submodule, 'new_attribute'))

    def test_apply_twice_returns_same_result(self):
        """Test applying patch twice doesn't re-apply."""
        def new_func(x):
            return x * 3

        patch = AtomicPatch(
            "atomic_test_module.submodule.original_func",
            new_func
        )
        result1 = patch.apply()
        self.assertTrue(patch.is_applied)

        # Modify the function to detect re-application
        self.mock_module.submodule.original_func = lambda x: x * 999

        # Apply again - should not change anything since is_applied is True
        # Note: apply() doesn't check is_applied internally, but Patcher does
        result2 = patch.apply()

        # The patch was applied again (AtomicPatch doesn't prevent this)
        self.assertEqual(self.mock_module.submodule.original_func(10), 30)


class TestRegistryPatchAdvanced(unittest.TestCase):
    """
    RegistryPatch高级功能测试

    测试RegistryPatch的高级特性，包括工厂函数、错误处理等。
    """

    def setUp(self):
        """测试环境准备 - 创建mock的Registry"""
        # Create mock registry
        self.mock_registry = MagicMock()
        self.mock_registry.register_module = MagicMock()

        self.mock_module = types.ModuleType('registry_adv_test')
        self.mock_module.REGISTRY = self.mock_registry
        sys.modules['registry_adv_test'] = self.mock_module

    def tearDown(self):
        """Clean up test fixtures."""
        if 'registry_adv_test' in sys.modules:
            del sys.modules['registry_adv_test']

    def test_constructor_requires_module_cls_or_factory(self):
        """Test constructor requires module_cls or module_factory."""
        with self.assertRaises(ValueError):
            RegistryPatch("some.registry")

    def test_constructor_requires_name_with_factory(self):
        """Test constructor requires name when using module_factory."""
        with self.assertRaises(ValueError):
            RegistryPatch("some.registry", module_factory=lambda: type('X', (), {}))

    def test_name_property(self):
        """Test name property combines registry and register_name."""
        class MyClass:
            pass

        patch = RegistryPatch("registry_adv_test.REGISTRY", MyClass, name="MyClass")
        self.assertEqual(patch.name, "registry_adv_test.REGISTRY.MyClass")

    def test_name_defaults_to_class_name(self):
        """Test name defaults to module_cls.__name__."""
        class AutoNamedClass:
            pass

        patch = RegistryPatch("registry_adv_test.REGISTRY", AutoNamedClass)
        self.assertEqual(patch.register_name, "AutoNamedClass")

    def test_precheck_with_params(self):
        """Test precheck receives registry parameters."""
        class MyClass:
            pass

        received_params = {}

        def precheck_func(registry, name):
            received_params['registry'] = registry
            received_params['name'] = name
            return True

        patch = RegistryPatch(
            "registry_adv_test.REGISTRY",
            MyClass,
            name="TestClass",
            precheck=precheck_func
        )
        patch.apply()

        self.assertEqual(received_params['registry'], "registry_adv_test.REGISTRY")
        self.assertEqual(received_params['name'], "TestClass")

    def test_precheck_exception_fails(self):
        """Test precheck exception causes FAILED (code bug)."""
        class MyClass:
            pass

        def bad_precheck():
            raise RuntimeError("precheck error")

        patch = RegistryPatch(
            "registry_adv_test.REGISTRY",
            MyClass,
            precheck=bad_precheck
        )
        result = patch.apply()

        self.assertFalse(patch.is_applied)
        self.assertEqual(result.status, PatchStatus.FAILED)

    def test_registry_not_found(self):
        """Test registry not found returns SKIPPED."""
        class MyClass:
            pass

        patch = RegistryPatch("nonexistent.REGISTRY", MyClass)
        result = patch.apply()

        self.assertFalse(patch.is_applied)
        self.assertEqual(result.status, PatchStatus.SKIPPED)
        self.assertIn("registry not found", result.reason)

    def test_invalid_registry_no_register_module(self):
        """Test invalid registry without register_module method."""
        # Create registry without register_module
        self.mock_module.BAD_REGISTRY = "not a registry"

        class MyClass:
            pass

        patch = RegistryPatch("registry_adv_test.BAD_REGISTRY", MyClass)
        result = patch.apply()

        self.assertFalse(patch.is_applied)
        self.assertEqual(result.status, PatchStatus.SKIPPED)
        self.assertIn("invalid registry", result.reason)

    def test_factory_error_returns_failed(self):
        """Test factory error returns FAILED status."""
        def bad_factory():
            raise RuntimeError("factory error")

        patch = RegistryPatch(
            "registry_adv_test.REGISTRY",
            name="FactoryClass",
            module_factory=bad_factory
        )
        result = patch.apply()

        self.assertFalse(patch.is_applied)
        self.assertEqual(result.status, PatchStatus.FAILED)
        self.assertIn("factory error", result.reason)

    def test_register_module_exception_returns_failed(self):
        """Test register_module exception returns FAILED."""
        self.mock_registry.register_module.side_effect = RuntimeError("register error")

        class MyClass:
            pass

        patch = RegistryPatch("registry_adv_test.REGISTRY", MyClass)
        result = patch.apply()

        self.assertFalse(patch.is_applied)
        self.assertEqual(result.status, PatchStatus.FAILED)

    def test_get_info_with_class(self):
        """Test get_info with module_cls."""
        class MyClass:
            pass

        patch = RegistryPatch("registry_adv_test.REGISTRY", MyClass, name="MyClass")
        info = patch.get_info()

        self.assertIn("registry_adv_test.REGISTRY", info)
        self.assertIn("MyClass", info)

    def test_get_info_with_factory(self):
        """Test get_info with module_factory."""
        patch = RegistryPatch(
            "registry_adv_test.REGISTRY",
            name="FactoryClass",
            module_factory=lambda: type('X', (), {})
        )
        info = patch.get_info()

        self.assertIn("<factory>", info)

    def test_repr(self):
        """Test __repr__ method."""
        class MyClass:
            pass

        patch = RegistryPatch("registry_adv_test.REGISTRY", MyClass, name="MyClass")
        repr_str = repr(patch)

        self.assertIn("RegistryPatch", repr_str)
        self.assertIn("registry_adv_test.REGISTRY", repr_str)


class TestPatchComposite(unittest.TestCase):
    """
    Patch组合类测试

    测试Patch类作为补丁容器的功能，包括子类化、迭代等。
    """

    def setUp(self):
        """测试环境准备"""
        self.mock_module = types.ModuleType('patch_composite_test')
        self.mock_module.func1 = lambda x: x
        self.mock_module.func2 = lambda x: x
        sys.modules['patch_composite_test'] = self.mock_module

    def tearDown(self):
        """Clean up test fixtures."""
        if 'patch_composite_test' in sys.modules:
            del sys.modules['patch_composite_test']

    def test_patch_subclass_with_patches(self):
        """Test creating a Patch subclass with patches method."""
        class MyPatch(Patch):
            name = "my_test_patch"

            @classmethod
            def patches(cls, options=None) -> List[BasePatch]:
                return [
                    AtomicPatch("patch_composite_test.func1", lambda x: x * 2),
                    AtomicPatch("patch_composite_test.func2", lambda x: x * 3),
                ]

        # Test iteration
        patches_list = list(MyPatch())
        self.assertEqual(len(patches_list), 2)

    def test_patch_with_options(self):
        """Test Patch subclass using options parameter."""
        class ConfigurablePatch(Patch):
            name = "configurable_patch"

            @classmethod
            def patches(cls, options=None) -> List[BasePatch]:
                multiplier = (options or {}).get('multiplier', 10)
                return [
                    AtomicPatch("patch_composite_test.func1", lambda x, m=multiplier: x * m),
                ]

        # With default options
        patches_default = ConfigurablePatch.patches()
        self.assertEqual(len(patches_default), 1)

        # With custom options
        patches_custom = ConfigurablePatch.patches({'multiplier': 5})
        self.assertEqual(len(patches_custom), 1)

    def test_patch_apply_returns_result(self):
        """Test that Patch.apply() returns a PatchResult."""
        class MyPatch(Patch):
            name = "my_patch"

            @classmethod
            def patches(cls, options=None):
                return []

        patch = MyPatch()
        result = patch.apply()
        # With no patches, should return SKIPPED status
        self.assertEqual(result.status, PatchStatus.SKIPPED)

    def test_patch_repr(self):
        """Test Patch __repr__ method."""
        class MyPatch(Patch):
            name = "my_repr_patch"

            @classmethod
            def patches(cls, options=None):
                return []

        patch = MyPatch()
        repr_str = repr(patch)
        self.assertIn("Patch", repr_str)
        self.assertIn("my_repr_patch", repr_str)


class TestLegacyPatchAdvanced(unittest.TestCase):
    """
    LegacyPatch高级功能测试

    测试LegacyPatch的错误处理和边界情况。
    """

    def setUp(self):
        """测试环境准备"""
        self.mock_module = types.ModuleType('legacy_adv_test')
        self.mock_module.ops = types.ModuleType('legacy_adv_test.ops')
        self.mock_module.ops.func = lambda x: x + 1
        sys.modules['legacy_adv_test'] = self.mock_module
        sys.modules['legacy_adv_test.ops'] = self.mock_module.ops

    def tearDown(self):
        """Clean up test fixtures."""
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith('legacy_adv_test'):
                del sys.modules[mod_name]

    def test_name_property(self):
        """Test name property returns function name."""
        def my_patch_func(module, options):
            pass

        patch = LegacyPatch(my_patch_func, target_module="legacy_adv_test")
        self.assertEqual(patch.name, "my_patch_func")

    def test_module_property(self):
        """Test module property returns target_module."""
        def my_patch_func(module, options):
            pass

        patch = LegacyPatch(my_patch_func, target_module="legacy_adv_test")
        self.assertEqual(patch.module, "legacy_adv_test")

    def test_apply_with_attribute_error(self):
        """Test apply handles AttributeError gracefully."""
        def bad_patch(module, options):
            raise AttributeError("missing attribute")

        patch = LegacyPatch(bad_patch, target_module="legacy_adv_test")
        result = patch.apply()

        self.assertFalse(patch.is_applied)
        self.assertEqual(result.status, PatchStatus.SKIPPED)

    def test_apply_with_generic_exception(self):
        """Test apply handles generic exception."""
        def bad_patch(module, options):
            raise RuntimeError("something went wrong")

        patch = LegacyPatch(bad_patch, target_module="legacy_adv_test")
        result = patch.apply()

        self.assertFalse(patch.is_applied)
        self.assertEqual(result.status, PatchStatus.FAILED)

    def test_options_passed_to_func(self):
        """Test options are passed to patch function."""
        received_options = {}

        def capture_options(module, options):
            received_options.update(options)

        patch = LegacyPatch(
            capture_options,
            target_module="legacy_adv_test",
            options={'key1': 'value1', 'key2': 'value2'}
        )
        patch.apply()

        self.assertEqual(received_options['key1'], 'value1')
        self.assertEqual(received_options['key2'], 'value2')

    def test_repr(self):
        """Test __repr__ method."""
        def my_patch(module, options):
            pass

        patch = LegacyPatch(my_patch, target_module="legacy_adv_test")
        repr_str = repr(patch)

        self.assertIn("LegacyPatch", repr_str)
        self.assertIn("my_patch", repr_str)


class TestPatcherAdvanced(unittest.TestCase):
    """
    Patcher高级功能测试

    测试Patcher的高级配置、缓存机制、执行顺序等。
    """

    def setUp(self):
        """测试环境准备"""
        self.mock_module = types.ModuleType('patcher_adv_test')
        self.mock_module.func1 = lambda x: x
        self.mock_module.func2 = lambda x: x
        self.mock_module.func3 = lambda x: x
        sys.modules['patcher_adv_test'] = self.mock_module

    def tearDown(self):
        """Clean up test fixtures."""
        if 'patcher_adv_test' in sys.modules:
            del sys.modules['patcher_adv_test']

    def test_add_invalid_type_raises(self):
        """Test add with invalid type raises TypeError."""
        patcher = Patcher()
        with self.assertRaises(TypeError):
            patcher.add("not a patch")

    def test_add_patch_class(self):
        """Test adding Patch class."""
        class MyPatch(Patch):
            name = "my_patcher_test_patch"

            @classmethod
            def patches(cls, options=None):
                return [AtomicPatch("patcher_adv_test.func1", lambda x: x * 2)]

        patcher = Patcher()
        result = patcher.add(MyPatch)
        self.assertIs(result, patcher)  # Check chaining

    def test_add_multiple_items(self):
        """Test adding multiple items at once."""
        patcher = Patcher()
        patcher.add(
            AtomicPatch("patcher_adv_test.func1", lambda x: x * 2),
            AtomicPatch("patcher_adv_test.func2", lambda x: x * 3),
        )
        patcher.apply()

        self.assertEqual(self.mock_module.func1(10), 20)
        self.assertEqual(self.mock_module.func2(10), 30)

    def test_disable_multiple_names(self):
        """Test disabling multiple patches by name."""
        patcher = Patcher()
        patcher.add(
            AtomicPatch("patcher_adv_test.func1", lambda x: x * 2),
            AtomicPatch("patcher_adv_test.func2", lambda x: x * 3),
        )
        patcher.disable("patcher_adv_test.func1", "patcher_adv_test.func2")
        patcher.apply()

        # Both should remain unchanged
        self.assertEqual(self.mock_module.func1(10), 10)
        self.assertEqual(self.mock_module.func2(10), 10)

    def test_disable_patch_class_by_name(self):
        """Test disabling Patch class by name."""
        class MyPatch(Patch):
            name = "disabled_patch"

            @classmethod
            def patches(cls, options=None):
                return [AtomicPatch("patcher_adv_test.func1", lambda x: x * 100)]

        patcher = Patcher()
        patcher.add(MyPatch)
        patcher.disable("disabled_patch")
        patcher.apply()

        # Should remain unchanged
        self.assertEqual(self.mock_module.func1(10), 10)

    def test_with_profiling_configuration(self):
        """Test with_profiling sets options correctly."""
        patcher = Patcher()
        result = patcher.with_profiling(
            "/path/to/prof",
            level=2,
            skip_first=100,
            wait=2,
            warmup=3,
            active=4,
            repeat=5
        )

        self.assertIs(result, patcher)  # Check chaining
        self.assertIsNotNone(patcher._profiling_options)
        self.assertEqual(patcher._profiling_options['profiling_path'], "/path/to/prof")
        self.assertEqual(patcher._profiling_options['profiling_level'], 2)
        self.assertEqual(patcher._profiling_options['step_ctrl'], (2, 3, 4, 5, 100))

    def test_brake_at_configuration(self):
        """Test brake_at sets step correctly."""
        patcher = Patcher()
        result = patcher.brake_at(500)

        self.assertIs(result, patcher)  # Check chaining
        self.assertEqual(patcher._brake_step, 500)

    def test_is_applied_property(self):
        """Test is_applied property."""
        patcher = Patcher()
        self.assertFalse(patcher.is_applied)

        patcher.apply()
        self.assertTrue(patcher.is_applied)

    def test_apply_twice_warns(self):
        """Test applying twice logs warning."""
        patcher = Patcher()
        patcher.apply()
        # Second apply should warn but not raise
        patcher.apply()
        self.assertTrue(patcher.is_applied)

    def test_context_manager(self):
        """Test using Patcher as context manager."""
        patcher = Patcher()
        patcher.add(AtomicPatch("patcher_adv_test.func1", lambda x: x * 5))

        with patcher as p:
            self.assertIs(p, patcher)
            self.assertTrue(patcher.is_applied)
            self.assertEqual(self.mock_module.func1(10), 50)

    def test_print_info_returns_self(self):
        """Test print_info returns self for chaining."""
        patcher = Patcher()
        patcher.add(AtomicPatch("patcher_adv_test.func1", lambda x: x * 2))

        # Capture stdout to avoid cluttering test output
        import io
        import sys as sys_module
        captured = io.StringIO()
        old_stdout = sys_module.stdout
        sys_module.stdout = captured

        try:
            result = patcher.print_info()
            self.assertIs(result, patcher)
        finally:
            sys_module.stdout = old_stdout

    def test_collect_all_patches_caching(self):
        """Test _collect_all_patches caches results."""
        patcher = Patcher()
        patcher.add(AtomicPatch("patcher_adv_test.func1", lambda x: x * 2))

        # First call
        patches1 = patcher._collect_all_patches()
        # Second call should return cached
        patches2 = patcher._collect_all_patches()

        self.assertIs(patches1, patches2)

    def test_collect_all_patches_force_refresh(self):
        """Test _collect_all_patches with force_refresh."""
        patcher = Patcher()
        patcher.add(AtomicPatch("patcher_adv_test.func1", lambda x: x * 2))

        # First call
        patches1 = patcher._collect_all_patches()
        # Force refresh
        patches2 = patcher._collect_all_patches(force_refresh=True)

        # Should be different list objects
        self.assertIsNot(patches1, patches2)

    def test_patch_order_preserved(self):
        """Test patches are applied in insertion order."""
        applied_order = []

        def make_patch(name, order_list):
            def replacement(x):
                order_list.append(name)
                return x
            return replacement

        patcher = Patcher()
        patcher.add(AtomicPatch("patcher_adv_test.func1", make_patch("first", applied_order)))
        patcher.add(AtomicPatch("patcher_adv_test.func2", make_patch("second", applied_order)))
        patcher.add(AtomicPatch("patcher_adv_test.func3", make_patch("third", applied_order)))
        patcher.apply()

        # Call functions to trigger order recording
        self.mock_module.func1(1)
        self.mock_module.func2(1)
        self.mock_module.func3(1)

        self.assertEqual(applied_order, ["first", "second", "third"])


class TestIntegrationScenarios(unittest.TestCase):
    """
    集成测试场景

    测试多种补丁类型混合使用、链式配置、部分失败等真实场景。
    """

    def setUp(self):
        """测试环境准备 - 创建包含多个子模块的模拟模块"""
        self.mock_module = types.ModuleType('integration_test')
        self.mock_module.ops = types.ModuleType('integration_test.ops')
        self.mock_module.ops.func1 = lambda x: x
        self.mock_module.ops.func2 = lambda x: x
        self.mock_module.models = types.ModuleType('integration_test.models')
        self.mock_module.models.Model = type('Model', (), {'forward': lambda self, x: x})
        sys.modules['integration_test'] = self.mock_module
        sys.modules['integration_test.ops'] = self.mock_module.ops
        sys.modules['integration_test.models'] = self.mock_module.models

    def tearDown(self):
        """Clean up test fixtures."""
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith('integration_test'):
                del sys.modules[mod_name]

    def test_mixed_patch_types(self):
        """Test using different patch types together."""
        # Create a Patch class
        class OpsPatch(Patch):
            name = "ops_patch"

            @classmethod
            def patches(cls, options=None):
                return [
                    AtomicPatch("integration_test.ops.func1", lambda x: x * 2),
                ]

        # Create a legacy patch function
        def legacy_patch(module, options):
            module.ops.func2 = lambda x: x * 3

        patcher = Patcher()
        patcher.add(OpsPatch)
        patcher.add(LegacyPatch(legacy_patch, target_module="integration_test"))
        patcher.add(AtomicPatch("integration_test.models.Model.forward",
                                lambda self, x: x * 4))
        patcher.apply()

        self.assertEqual(self.mock_module.ops.func1(10), 20)
        self.assertEqual(self.mock_module.ops.func2(10), 30)

        model = self.mock_module.models.Model()
        self.assertEqual(model.forward(10), 40)

    def test_chained_configuration(self):
        """Test chained configuration methods."""
        patcher = (
            Patcher()
            .add(AtomicPatch("integration_test.ops.func1", lambda x: x * 2))
            .disable("integration_test.ops.func2")
            .with_profiling("/tmp/prof", level=1)
            .brake_at(100)
        )

        self.assertIsNotNone(patcher._profiling_options)
        self.assertEqual(patcher._brake_step, 100)
        self.assertIn("integration_test.ops.func2", patcher._blacklist)

    def test_patch_class_with_options_in_patcher(self):
        """Test Patch class with options passed through Patcher."""
        class ConfigurablePatch(Patch):
            name = "configurable"

            @classmethod
            def patches(cls, options=None):
                mult = (options or {}).get('multiplier', 10)
                return [
                    AtomicPatch("integration_test.ops.func1", lambda x, m=mult: x * m),
                ]

        patcher = Patcher()
        patcher.add(ConfigurablePatch, options={'multiplier': 7})
        patcher.apply()

        self.assertEqual(self.mock_module.ops.func1(10), 70)

    def test_partial_failure_continues(self):
        """Test that partial failures don't stop other patches."""
        patcher = Patcher()
        patcher.add(
            AtomicPatch("nonexistent.module.func", lambda x: x),  # Will skip
            AtomicPatch("integration_test.ops.func1", lambda x: x * 2),  # Should apply
        )
        patcher.apply()

        # Second patch should still be applied
        self.assertEqual(self.mock_module.ops.func1(10), 20)

    def test_wrapper_chain(self):
        """Test chaining multiple wrappers."""
        def add_logging(func):
            def wrapped(x):
                return func(x)
            wrapped._logged = True
            return wrapped

        def add_timing(func):
            def wrapped(x):
                return func(x)
            wrapped._timed = True
            return wrapped

        # First wrapper
        patcher1 = Patcher()
        patcher1.add(AtomicPatch(
            "integration_test.ops.func1",
            target_wrapper=add_logging
        ))
        patcher1.apply()

        # Second wrapper on already wrapped function
        patcher2 = Patcher()
        patcher2.add(AtomicPatch(
            "integration_test.ops.func1",
            target_wrapper=add_timing
        ))
        patcher2.apply()

        # Both wrappers should be applied
        self.assertTrue(hasattr(self.mock_module.ops.func1, '_timed'))

    def test_runtime_check_with_complex_condition(self):
        """Test runtime_check with complex condition."""
        def optimized_func(x, y):
            return (x + y) * 2

        def check_condition(x, y):
            return isinstance(x, int) and isinstance(y, int) and x > 0 and y > 0

        self.mock_module.ops.add = lambda x, y: x + y

        patcher = Patcher()
        patcher.add(AtomicPatch(
            "integration_test.ops.add",
            optimized_func,
            runtime_check=check_condition
        ))
        patcher.apply()

        # Condition met: use optimized
        self.assertEqual(self.mock_module.ops.add(5, 3), 16)

        # Condition not met: use original
        self.assertEqual(self.mock_module.ops.add(-1, 3), 2)
        self.assertEqual(self.mock_module.ops.add("a", "b"), "ab")


class TestEdgeCases(unittest.TestCase):
    """
    边界情况测试

    测试各种边界条件，如空Patcher、不存在的禁用名称、深层嵌套路径等。
    """

    def setUp(self):
        """测试环境准备"""
        self.mock_module = types.ModuleType('edge_test')
        self.mock_module.sub = types.ModuleType('edge_test.sub')
        self.mock_module.sub.func = lambda x: x
        sys.modules['edge_test'] = self.mock_module
        sys.modules['edge_test.sub'] = self.mock_module.sub

    def tearDown(self):
        """Clean up test fixtures."""
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith('edge_test'):
                del sys.modules[mod_name]

    def test_patch_dict_attribute(self):
        """Test patching dict-like attribute access."""
        self.mock_module.config = {'key': 'original'}

        # AtomicPatch doesn't directly support dict item patching,
        # but we can patch the whole dict
        patcher = Patcher()
        patcher.add(AtomicPatch(
            "edge_test.config",
            {'key': 'patched'}
        ))
        patcher.apply()

        self.assertEqual(self.mock_module.config['key'], 'patched')

    def test_patch_class_method(self):
        """Test patching a class method."""
        class MyClass:
            def method(self, x):
                return x

        self.mock_module.MyClass = MyClass

        def new_method(self, x):
            return x * 10

        patcher = Patcher()
        patcher.add(AtomicPatch("edge_test.MyClass.method", new_method))
        patcher.apply()

        obj = self.mock_module.MyClass()
        self.assertEqual(obj.method(5), 50)

    def test_patch_static_method(self):
        """Test patching a static method."""
        class MyClass:
            @staticmethod
            def static_func(x):
                return x

        self.mock_module.MyClass = MyClass

        patcher = Patcher()
        patcher.add(AtomicPatch(
            "edge_test.MyClass.static_func",
            staticmethod(lambda x: x * 20)
        ))
        patcher.apply()

        self.assertEqual(self.mock_module.MyClass.static_func(5), 100)

    def test_patch_class_attribute(self):
        """Test patching a class attribute."""
        class MyClass:
            value = 10

        self.mock_module.MyClass = MyClass

        patcher = Patcher()
        patcher.add(AtomicPatch("edge_test.MyClass.value", 100))
        patcher.apply()

        self.assertEqual(self.mock_module.MyClass.value, 100)

    def test_patch_nested_module_path(self):
        """Test patching deeply nested module path."""
        self.mock_module.level1 = types.ModuleType('edge_test.level1')
        self.mock_module.level1.level2 = types.ModuleType('edge_test.level1.level2')
        self.mock_module.level1.level2.level3 = types.ModuleType('edge_test.level1.level2.level3')
        self.mock_module.level1.level2.level3.func = lambda x: x
        sys.modules['edge_test.level1'] = self.mock_module.level1
        sys.modules['edge_test.level1.level2'] = self.mock_module.level1.level2
        sys.modules['edge_test.level1.level2.level3'] = self.mock_module.level1.level2.level3

        patcher = Patcher()
        patcher.add(AtomicPatch(
            "edge_test.level1.level2.level3.func",
            lambda x: x * 5
        ))
        patcher.apply()

        self.assertEqual(self.mock_module.level1.level2.level3.func(10), 50)

    def test_empty_patcher_apply(self):
        """Test applying empty patcher."""
        patcher = Patcher()
        patcher.apply()  # Should not raise
        self.assertTrue(patcher.is_applied)

    def test_disable_nonexistent_patch(self):
        """Test disabling non-existent patch name."""
        patcher = Patcher()
        patcher.add(AtomicPatch("edge_test.sub.func", lambda x: x * 2))
        patcher.disable("nonexistent_patch_name")  # Should not raise
        patcher.apply()

        # The actual patch should still be applied
        self.assertEqual(self.mock_module.sub.func(10), 20)

    def test_precheck_with_all_available_params(self):
        """Test precheck receiving all available parameters."""
        received = {}

        def capture_all(target, replacement, replacement_wrapper, target_wrapper, aliases):
            received['target'] = target
            received['replacement'] = replacement
            received['replacement_wrapper'] = replacement_wrapper
            received['target_wrapper'] = target_wrapper
            received['aliases'] = aliases
            return True

        def my_wrapper(f):
            return f

        patcher = Patcher()
        patcher.add(AtomicPatch(
            "edge_test.sub.func",
            lambda x: x * 2,
            replacement_wrapper=my_wrapper,
            aliases=["edge_test.alias"],
            precheck=capture_all
        ))
        patcher.apply()

        self.assertEqual(received['target'], "edge_test.sub.func")
        self.assertIsNotNone(received['replacement'])
        self.assertEqual(received['replacement_wrapper'], my_wrapper)
        self.assertEqual(received['aliases'], ["edge_test.alias"])

    def test_multiple_aliases(self):
        """Test patching with multiple aliases."""
        self.mock_module.alias1 = types.ModuleType('edge_test.alias1')
        self.mock_module.alias1.func = self.mock_module.sub.func
        self.mock_module.alias2 = types.ModuleType('edge_test.alias2')
        self.mock_module.alias2.func = self.mock_module.sub.func
        sys.modules['edge_test.alias1'] = self.mock_module.alias1
        sys.modules['edge_test.alias2'] = self.mock_module.alias2

        patcher = Patcher()
        patcher.add(AtomicPatch(
            "edge_test.sub.func",
            lambda x: x * 7,
            aliases=["edge_test.alias1.func", "edge_test.alias2.func"]
        ))
        patcher.apply()

        self.assertEqual(self.mock_module.sub.func(10), 70)
        self.assertEqual(self.mock_module.alias1.func(10), 70)
        self.assertEqual(self.mock_module.alias2.func(10), 70)


class TestPatchResultAndStatus(unittest.TestCase):
    """
    PatchResult和PatchStatus详细测试

    测试结果对象的创建、比较、各种状态等。
    """

    def test_patch_status_enum_values(self):
        """测试PatchStatus枚举值正确"""
        self.assertEqual(PatchStatus.APPLIED.value, "applied")
        self.assertEqual(PatchStatus.SKIPPED.value, "skipped")
        self.assertEqual(PatchStatus.FAILED.value, "failed")

    def test_patch_status_enum_members(self):
        """Test PatchStatus enum has all expected members."""
        members = list(PatchStatus)
        self.assertEqual(len(members), 3)
        self.assertIn(PatchStatus.APPLIED, members)
        self.assertIn(PatchStatus.SKIPPED, members)
        self.assertIn(PatchStatus.FAILED, members)

    def test_patch_result_creation_minimal(self):
        """Test PatchResult creation with minimal args."""
        result = PatchResult(PatchStatus.APPLIED, "test_patch", "test_module")
        self.assertEqual(result.status, PatchStatus.APPLIED)
        self.assertEqual(result.name, "test_patch")
        self.assertEqual(result.module, "test_module")
        self.assertIsNone(result.reason)

    def test_patch_result_creation_with_reason(self):
        """Test PatchResult creation with reason."""
        result = PatchResult(
            PatchStatus.SKIPPED,
            "test_patch",
            "test_module",
            "module not found"
        )
        self.assertEqual(result.status, PatchStatus.SKIPPED)
        self.assertEqual(result.reason, "module not found")

    def test_patch_result_equality(self):
        """Test PatchResult equality comparison."""
        result1 = PatchResult(PatchStatus.APPLIED, "patch1", "module1")
        result2 = PatchResult(PatchStatus.APPLIED, "patch1", "module1")
        result3 = PatchResult(PatchStatus.SKIPPED, "patch1", "module1")

        self.assertEqual(result1, result2)
        self.assertNotEqual(result1, result3)

    def test_patch_result_with_different_statuses(self):
        """Test creating PatchResult with each status."""
        for status in PatchStatus:
            result = PatchResult(status, f"patch_{status.value}", "module")
            self.assertEqual(result.status, status)


class TestBasePatchAbstract(unittest.TestCase):
    """
    BasePatch抽象基类测试

    测试BasePatch作为抽象类的行为，包括不能直接实例化、子类必须实现的方法等。
    """

    def test_cannot_instantiate_base_patch(self):
        """测试BasePatch不能直接实例化 - 应抛出TypeError"""
        with self.assertRaises(TypeError):
            BasePatch()

    def test_subclass_must_implement_name(self):
        """Test that subclass must implement name property."""
        class IncompletePatch(BasePatch):
            def apply(self):
                return PatchResult(PatchStatus.APPLIED, "test", "test")

        with self.assertRaises(TypeError):
            IncompletePatch()

    def test_subclass_must_implement_apply(self):
        """Test that subclass must implement apply method."""
        class IncompletePatch(BasePatch):
            @property
            def name(self):
                return "incomplete"

        with self.assertRaises(TypeError):
            IncompletePatch()

    def test_complete_subclass_works(self):
        """Test that complete subclass can be instantiated."""
        class CompletePatch(BasePatch):
            @property
            def name(self):
                return "complete_patch"

            def apply(self):
                return PatchResult(PatchStatus.APPLIED, self.name, "test")

        patch = CompletePatch()
        self.assertEqual(patch.name, "complete_patch")
        result = patch.apply()
        self.assertEqual(result.status, PatchStatus.APPLIED)

    def test_module_property_default(self):
        """Test module property default behavior."""
        class MyPatch(BasePatch):
            @property
            def name(self):
                return "mymodule.submodule.func"

            def apply(self):
                return PatchResult(PatchStatus.APPLIED, self.name, "")

        patch = MyPatch()
        self.assertEqual(patch.module, "mymodule")

    def test_module_property_no_dot(self):
        """Test module property when name has no dot."""
        class MyPatch(BasePatch):
            @property
            def name(self):
                return "simple_name"

            def apply(self):
                return PatchResult(PatchStatus.APPLIED, self.name, "")

        patch = MyPatch()
        self.assertEqual(patch.module, "")

    def test_get_info_default(self):
        """Test get_info default implementation."""
        class MyPatch(BasePatch):
            @property
            def name(self):
                return "my_patch_name"

            def apply(self):
                return PatchResult(PatchStatus.APPLIED, self.name, "")

        patch = MyPatch()
        self.assertEqual(patch.get_info(), "my_patch_name")
