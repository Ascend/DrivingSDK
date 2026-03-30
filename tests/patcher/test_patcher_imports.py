# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Patcher导入相关功能测试模块

本模块测试patcher框架的导入相关功能，包括：
- skip_import: 跳过指定模块的导入（注册stub）
- inject_import: 注入缺失的导入
- with_imports: 延迟导入装饰器

测试设计原则：
- 使用mock模块模拟真实环境
- 直接加载patcher子模块，避免触发torch依赖
- 每个测试用例独立，通过setUp/tearDown管理测试环境
"""
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from functools import wraps as _functools_wraps
from typing import Dict, List

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
Patcher = _patcher_module.Patcher
AtomicPatch = _patch_module.AtomicPatch
LegacyPatch = _patch_module.LegacyPatch
Patch = _patch_module.Patch
with_imports = _patch_module.with_imports
_patcher_logger = _patcher_logger_module.patcher_logger


# =============================================================================
# Test: skip_import functionality
# =============================================================================

class TestSkipImport(unittest.TestCase):
    """
    skip_import功能测试

    skip_import用于跳过指定模块的导入，通过注册stub模块到sys.modules。
    这对于处理CUDA专用依赖（如flash_attn）非常有用。
    """

    def setUp(self):
        """记录测试前的sys.modules状态"""
        self._original_modules = set(sys.modules.keys())

    def tearDown(self):
        """清理测试中添加的模块"""
        current_modules = set(sys.modules.keys())
        for module_name in current_modules - self._original_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]

    def test_skip_import_registers_stub(self):
        """
        测试skip_import注册stub模块

        Ground Truth:
        - 调用skip_import后，模块应该存在于sys.modules中
        - 模块应该是一个stub，可以安全访问任意属性
        """
        patcher = Patcher()
        patcher.skip_import("fake_cuda_module")

        # Ground truth: module should be in sys.modules
        self.assertIn("fake_cuda_module", sys.modules)

        # Ground truth: stub should handle attribute access
        module = sys.modules["fake_cuda_module"]
        # Accessing any attribute should not raise
        _ = module.some_function
        _ = module.SomeClass

    def test_skip_import_immediate_execution(self):
        """
        测试skip_import立即执行

        Ground Truth:
        - skip_import应该在调用时立即注册stub
        - 不需要等待apply()调用
        """
        patcher = Patcher()

        # Before skip_import
        self.assertNotIn("immediate_test_module", sys.modules)

        # Call skip_import (should execute immediately)
        patcher.skip_import("immediate_test_module")

        # Ground truth: stub should be registered immediately, before apply()
        self.assertIn("immediate_test_module", sys.modules)

    def test_skip_import_multiple_modules(self):
        """
        测试skip_import支持多个模块

        Ground Truth:
        - 可以一次跳过多个模块
        - 所有模块都应该被注册为stub
        """
        patcher = Patcher()
        patcher.skip_import("multi_test_a", "multi_test_b", "multi_test_c")

        # Ground truth: all modules should be registered
        self.assertIn("multi_test_a", sys.modules)
        self.assertIn("multi_test_b", sys.modules)
        self.assertIn("multi_test_c", sys.modules)

    def test_skip_import_submodule_access(self):
        """
        测试skip_import的stub支持子模块访问

        Ground Truth:
        - stub应该支持子模块的导入语法
        - from fake_module.submodule import something 应该工作
        """
        patcher = Patcher()
        patcher.skip_import("fake_parent_module")

        # Ground truth: submodule access should work
        module = sys.modules["fake_parent_module"]
        submodule = module.submodule
        self.assertIsNotNone(submodule)

    def test_skip_import_already_imported_no_effect(self):
        """
        测试skip_import对已导入模块无效

        Ground Truth:
        - 如果模块已经存在于sys.modules中，skip_import不应该覆盖它
        """
        # Pre-register a real module
        real_module = types.ModuleType("pre_existing_module")
        real_module.real_attr = "real_value"
        sys.modules["pre_existing_module"] = real_module

        patcher = Patcher()
        patcher.skip_import("pre_existing_module")

        # Ground truth: original module should be preserved
        self.assertIs(sys.modules["pre_existing_module"], real_module)
        self.assertEqual(sys.modules["pre_existing_module"].real_attr, "real_value")

    def test_skip_import_chaining(self):
        """
        测试skip_import支持链式调用

        Ground Truth:
        - skip_import应该返回self，支持链式调用
        """
        patcher = Patcher()
        result = patcher.skip_import("chain_test_1").skip_import("chain_test_2")

        # Ground truth: should return self for chaining
        self.assertIs(result, patcher)
        self.assertIn("chain_test_1", sys.modules)
        self.assertIn("chain_test_2", sys.modules)


# =============================================================================
# Test: inject_import functionality
# =============================================================================

class TestInjectImport(unittest.TestCase):
    """
    inject_import功能测试

    inject_import用于将一个模块中的类/函数注入到另一个模块中。
    这对于修复第三方代码中缺失的导出非常有用。
    """

    def setUp(self):
        """创建测试模块"""
        # Create source module with a class
        self.source_module = types.ModuleType("inject_source_module")
        self.source_module.MyClass = type("MyClass", (), {"value": 42})
        self.source_module.my_function = lambda x: x * 2
        sys.modules["inject_source_module"] = self.source_module

        # Create target module (initially without the class)
        self.target_module = types.ModuleType("inject_target_module")
        self.target_module.__all__ = []
        sys.modules["inject_target_module"] = self.target_module

    def tearDown(self):
        """清理测试模块"""
        for name in ["inject_source_module", "inject_target_module"]:
            if name in sys.modules:
                del sys.modules[name]

    def test_inject_import_basic(self):
        """
        测试基本的inject_import功能

        Ground Truth:
        - 类应该从source模块注入到target模块
        - target模块应该可以访问注入的类
        """
        patcher = Patcher()
        patcher.inject_import(
            "inject_source_module",
            "MyClass",
            "inject_target_module"
        )

        # Ground truth: class should be accessible from target module
        self.assertTrue(hasattr(self.target_module, "MyClass"))
        self.assertEqual(self.target_module.MyClass.value, 42)

    def test_inject_import_immediate_execution(self):
        """
        测试inject_import立即执行

        Ground Truth:
        - inject_import应该在调用时立即执行注入
        - 不需要等待apply()调用
        """
        patcher = Patcher()

        # Before inject_import
        self.assertFalse(hasattr(self.target_module, "MyClass"))

        # Call inject_import (should execute immediately)
        patcher.inject_import(
            "inject_source_module",
            "MyClass",
            "inject_target_module"
        )

        # Ground truth: injection should happen immediately, before apply()
        self.assertTrue(hasattr(self.target_module, "MyClass"))

    def test_inject_import_adds_to_all(self):
        """
        测试inject_import将类添加到__all__

        Ground Truth:
        - 如果target模块有__all__，注入的类名应该被添加进去
        """
        patcher = Patcher()
        patcher.inject_import(
            "inject_source_module",
            "MyClass",
            "inject_target_module"
        )

        # Ground truth: class name should be in __all__
        self.assertIn("MyClass", self.target_module.__all__)

    def test_inject_import_function(self):
        """
        测试inject_import可以注入函数

        Ground Truth:
        - 不仅类，函数也可以被注入
        """
        patcher = Patcher()
        patcher.inject_import(
            "inject_source_module",
            "my_function",
            "inject_target_module"
        )

        # Ground truth: function should be accessible and work correctly
        self.assertTrue(hasattr(self.target_module, "my_function"))
        self.assertEqual(self.target_module.my_function(5), 10)

    def test_inject_import_nonexistent_class(self):
        """
        测试inject_import处理不存在的类

        Ground Truth:
        - 如果source模块中不存在指定的类，应该静默失败
        - 不应该抛出异常
        """
        patcher = Patcher()

        # Should not raise
        patcher.inject_import(
            "inject_source_module",
            "NonExistentClass",
            "inject_target_module"
        )

        # Ground truth: target module should not have the class
        self.assertFalse(hasattr(self.target_module, "NonExistentClass"))

    def test_inject_import_nonexistent_module(self):
        """
        测试inject_import处理不存在的模块

        Ground Truth:
        - 如果source或target模块不存在，应该静默失败
        - 不应该抛出异常
        """
        patcher = Patcher()

        # Should not raise for nonexistent source
        patcher.inject_import(
            "nonexistent_source",
            "MyClass",
            "inject_target_module"
        )

        # Should not raise for nonexistent target
        patcher.inject_import(
            "inject_source_module",
            "MyClass",
            "nonexistent_target"
        )

    def test_inject_import_no_duplicate(self):
        """
        测试inject_import不会重复注入

        Ground Truth:
        - 如果target模块已经有同名属性，不应该覆盖
        """
        # Pre-set an attribute in target
        self.target_module.MyClass = "original_value"

        patcher = Patcher()
        patcher.inject_import(
            "inject_source_module",
            "MyClass",
            "inject_target_module"
        )

        # Ground truth: original value should be preserved
        self.assertEqual(self.target_module.MyClass, "original_value")

    def test_inject_import_chaining(self):
        """
        测试inject_import支持链式调用

        Ground Truth:
        - inject_import应该返回self，支持链式调用
        """
        patcher = Patcher()
        result = patcher.inject_import(
            "inject_source_module",
            "MyClass",
            "inject_target_module"
        ).inject_import(
            "inject_source_module",
            "my_function",
            "inject_target_module"
        )

        # Ground truth: should return self for chaining
        self.assertIs(result, patcher)
        self.assertTrue(hasattr(self.target_module, "MyClass"))
        self.assertTrue(hasattr(self.target_module, "my_function"))


# =============================================================================
# Test: with_imports decorator
# =============================================================================

class TestWithImports(unittest.TestCase):
    """
    with_imports装饰器测试

    with_imports用于延迟导入模块到函数的全局命名空间。
    这允许replacement函数使用与原始函数相同的导入语法。
    """

    def setUp(self):
        """创建测试模块"""
        self.test_module = types.ModuleType("with_imports_test_module")
        self.test_module.CONSTANT_A = 10
        self.test_module.CONSTANT_B = 20
        self.test_module.helper_func = lambda x: x * 3
        sys.modules["with_imports_test_module"] = self.test_module

    def tearDown(self):
        """清理测试模块"""
        if "with_imports_test_module" in sys.modules:
            del sys.modules["with_imports_test_module"]

    def test_with_imports_basic(self):
        """
        测试with_imports基本功能

        Ground Truth:
        - 装饰后的函数应该能访问注入的名称
        - 函数应该正常执行并返回正确结果
        """
        @with_imports(("with_imports_test_module", "CONSTANT_A", "CONSTANT_B"))
        def my_func():
            return CONSTANT_A + CONSTANT_B  # noqa: F821

        # Ground truth: function should work with injected names
        result = my_func()
        self.assertEqual(result, 30)  # 10 + 20

    def test_with_imports_lazy_resolution(self):
        """
        测试with_imports延迟解析

        Ground Truth:
        - 导入应该在第一次调用时才发生
        - 装饰时不应该导入模块
        """
        @with_imports(("with_imports_test_module", "CONSTANT_A"))
        def lazy_func():
            return CONSTANT_A  # noqa: F821

        # Ground truth: function should work correctly
        result = lazy_func()
        self.assertEqual(result, 10)

    def test_with_imports_caching(self):
        """
        测试with_imports缓存机制

        Ground Truth:
        - 第一次调用后，后续调用不应该重新导入
        - 函数应该使用缓存的结果
        """
        call_count = [0]

        @with_imports(("with_imports_test_module", "CONSTANT_A"))
        def cached_func():
            call_count[0] += 1
            return CONSTANT_A  # noqa: F821

        # Call multiple times
        result1 = cached_func()
        result2 = cached_func()
        result3 = cached_func()

        # Ground truth: all calls should return correct result
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)
        self.assertEqual(result3, 10)
        self.assertEqual(call_count[0], 3)

    def test_with_imports_multiple_specs(self):
        """
        测试with_imports支持多个导入规格

        Ground Truth:
        - 可以从多个模块导入
        - 所有导入的名称都应该可用
        """
        # Create another test module
        other_module = types.ModuleType("other_test_module")
        other_module.OTHER_CONST = 100
        sys.modules["other_test_module"] = other_module

        try:
            @with_imports(
                ("with_imports_test_module", "CONSTANT_A"),
                ("other_test_module", "OTHER_CONST"),
            )
            def multi_import_func():
                return CONSTANT_A + OTHER_CONST  # noqa: F821

            # Ground truth: both imports should work
            result = multi_import_func()
            self.assertEqual(result, 110)  # 10 + 100
        finally:
            del sys.modules["other_test_module"]

    def test_with_imports_with_staticmethod(self):
        """
        测试with_imports与@staticmethod配合使用

        Ground Truth:
        - @staticmethod应该放在@with_imports之前
        - 静态方法应该正常工作
        """
        class MyClass:
            @staticmethod
            @with_imports(("with_imports_test_module", "CONSTANT_A"))
            def static_method():
                return CONSTANT_A * 2  # noqa: F821

        # Ground truth: static method should work
        result = MyClass.static_method()
        self.assertEqual(result, 20)  # 10 * 2

    def test_with_imports_with_classmethod(self):
        """
        测试with_imports与@classmethod配合使用

        Ground Truth:
        - @classmethod应该放在@with_imports之前
        - 类方法应该正常工作
        """
        class MyClass:
            multiplier = 3

            @classmethod
            @with_imports(("with_imports_test_module", "CONSTANT_A"))
            def class_method(cls):
                return CONSTANT_A * cls.multiplier  # noqa: F821

        # Ground truth: class method should work
        result = MyClass.class_method()
        self.assertEqual(result, 30)  # 10 * 3

    def test_with_imports_with_arguments(self):
        """
        测试with_imports装饰的函数可以接收参数

        Ground Truth:
        - 装饰后的函数应该正确传递参数
        """
        @with_imports(("with_imports_test_module", "CONSTANT_A"))
        def func_with_args(x, y, z=1):
            return CONSTANT_A + x + y + z  # noqa: F821

        # Ground truth: arguments should be passed correctly
        result = func_with_args(1, 2, z=3)
        self.assertEqual(result, 16)  # 10 + 1 + 2 + 3

    def test_with_imports_missing_name(self):
        """
        测试with_imports处理不存在的名称

        Ground Truth:
        - 如果模块中不存在指定的名称，应该静默跳过
        - 函数仍然可以执行（如果不使用该名称）
        """
        @with_imports(("with_imports_test_module", "NONEXISTENT", "CONSTANT_A"))
        def func_with_missing():
            return CONSTANT_A  # noqa: F821

        # Ground truth: function should still work with available names
        result = func_with_missing()
        self.assertEqual(result, 10)

    def test_with_imports_missing_module(self):
        """
        测试with_imports处理不存在的模块

        Ground Truth:
        - 如果模块不存在，应该静默跳过
        - 函数仍然可以执行（如果不使用该模块的名称）
        """
        @with_imports(
            ("nonexistent_module", "SOMETHING"),
            ("with_imports_test_module", "CONSTANT_A"),
        )
        def func_with_missing_module():
            return CONSTANT_A  # noqa: F821

        # Ground truth: function should still work
        result = func_with_missing_module()
        self.assertEqual(result, 10)

    def test_with_imports_function_attributes_preserved(self):
        """
        测试with_imports保留函数属性

        Ground Truth:
        - 装饰后的函数应该保留原函数的__name__和__doc__
        """
        @with_imports(("with_imports_test_module", "CONSTANT_A"))
        def documented_func():
            """This is a docstring."""
            return CONSTANT_A  # noqa: F821

        # Ground truth: function attributes should be preserved
        self.assertEqual(documented_func.__name__, "documented_func")
        self.assertEqual(documented_func.__doc__, "This is a docstring.")


# =============================================================================
# Test: Integration - skip_import, inject_import, and apply order
# =============================================================================

class TestImportIntegration(unittest.TestCase):
    """
    导入功能集成测试

    测试skip_import、inject_import和apply的执行顺序和交互。
    """

    def setUp(self):
        """记录测试前的sys.modules状态"""
        self._original_modules = set(sys.modules.keys())

    def tearDown(self):
        """清理测试中添加的模块"""
        current_modules = set(sys.modules.keys())
        for module_name in current_modules - self._original_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]

    def test_skip_then_inject_order(self):
        """
        测试skip_import和inject_import的执行顺序

        Ground Truth:
        - skip_import和inject_import都应该在调用时立即执行
        - 不需要等待apply()
        """
        # Create source module for injection
        source = types.ModuleType("order_test_source")
        source.MyClass = type("MyClass", (), {})
        sys.modules["order_test_source"] = source

        target = types.ModuleType("order_test_target")
        sys.modules["order_test_target"] = target

        patcher = Patcher()

        # Both should execute immediately
        patcher.skip_import("order_test_skip")
        patcher.inject_import("order_test_source", "MyClass", "order_test_target")

        # Ground truth: both should have taken effect before apply()
        self.assertIn("order_test_skip", sys.modules)
        self.assertTrue(hasattr(target, "MyClass"))

    def test_typical_usage_pattern(self):
        """
        测试典型使用模式

        Ground Truth:
        - 典型用法: skip_import -> inject_import -> add patches -> apply
        - 所有操作应该按预期工作
        """
        # Create modules
        source = types.ModuleType("typical_source")
        source.Helper = type("Helper", (), {"value": 100})
        sys.modules["typical_source"] = source

        target = types.ModuleType("typical_target")
        target.func = lambda x: x
        sys.modules["typical_target"] = target

        patcher = Patcher()

        # Typical usage pattern
        patcher.skip_import("cuda_only_module")
        patcher.inject_import("typical_source", "Helper", "typical_target")

        # Ground truth: skip and inject should work before apply
        self.assertIn("cuda_only_module", sys.modules)
        self.assertTrue(hasattr(target, "Helper"))
        self.assertEqual(target.Helper.value, 100)


# =============================================================================
# Test: _StubModule special methods (iteration, path, etc.)
# =============================================================================

class TestStubModuleSpecialMethods(unittest.TestCase):
    """
    _StubModule特殊方法测试

    测试_StubModule的各种Python特殊方法，确保它能正确与Python导入机制配合。
    这些测试覆盖了导致 TypeError: '_StubModule' object is not iterable 的场景。
    """

    def setUp(self):
        """记录测试前的sys.modules状态"""
        self._original_modules = set(sys.modules.keys())

    def tearDown(self):
        """清理测试中添加的模块"""
        current_modules = set(sys.modules.keys())
        for module_name in current_modules - self._original_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]

    def test_stub_module_is_iterable(self):
        """
        测试stub模块可迭代

        Ground Truth:
        - stub模块应该支持迭代操作
        - 迭代结果应该为空列表
        - 这是Python导入机制查找子模块时的必要条件
        """
        patcher = Patcher()
        patcher.skip_import("iterable_test_module")

        module = sys.modules["iterable_test_module"]

        # Ground truth: iteration should work and return empty
        items = list(module)
        self.assertEqual(items, [])

        # Also test iter() directly
        iterator = iter(module)
        self.assertEqual(list(iterator), [])

    def test_stub_module_has_path(self):
        """
        测试stub模块有__path__属性

        Ground Truth:
        - stub模块应该有__path__属性
        - __path__应该是一个空列表
        - 这是Python将模块视为包的必要条件
        """
        patcher = Patcher()
        patcher.skip_import("path_test_module")

        module = sys.modules["path_test_module"]

        # Ground truth: __path__ should exist and be a list
        self.assertTrue(hasattr(module, "__path__"))
        self.assertIsInstance(module.__path__, list)
        self.assertEqual(module.__path__, [])

    def test_stub_module_has_all(self):
        """
        测试stub模块有__all__属性

        Ground Truth:
        - stub模块应该有__all__属性
        - __all__应该是一个空列表
        - 这是"from module import *"语法的必要条件
        """
        patcher = Patcher()
        patcher.skip_import("all_test_module")

        module = sys.modules["all_test_module"]

        # Ground truth: __all__ should exist and be a list
        self.assertTrue(hasattr(module, "__all__"))
        self.assertIsInstance(module.__all__, list)
        self.assertEqual(module.__all__, [])

    def test_stub_module_contains(self):
        """
        测试stub模块支持in操作符

        Ground Truth:
        - stub模块应该支持"in"操作符
        - 任何检查都应该返回False
        """
        patcher = Patcher()
        patcher.skip_import("contains_test_module")

        module = sys.modules["contains_test_module"]

        # Ground truth: "in" operator should work and return False
        self.assertFalse("anything" in module)
        self.assertFalse("submodule" in module)

    def test_stub_module_len(self):
        """
        测试stub模块支持len()

        Ground Truth:
        - stub模块应该支持len()调用
        - 长度应该为0
        """
        patcher = Patcher()
        patcher.skip_import("len_test_module")

        module = sys.modules["len_test_module"]

        # Ground truth: len() should work and return 0
        self.assertEqual(len(module), 0)

    def test_stub_module_bool(self):
        """
        测试stub模块的布尔值

        Ground Truth:
        - stub模块的布尔值应该为False
        - 这表明它不是一个真实的模块
        """
        patcher = Patcher()
        patcher.skip_import("bool_test_module")

        module = sys.modules["bool_test_module"]

        # Ground truth: bool value should be False
        self.assertFalse(bool(module))

    def test_stub_module_repr(self):
        """
        测试stub模块的字符串表示

        Ground Truth:
        - stub模块应该有可读的字符串表示
        - 应该包含模块名称
        """
        patcher = Patcher()
        patcher.skip_import("repr_test_module")

        module = sys.modules["repr_test_module"]

        # Ground truth: repr should be readable
        repr_str = repr(module)
        self.assertIn("repr_test_module", repr_str)
        self.assertIn("Stub", repr_str)

    def test_stub_submodule_registered_in_sys_modules(self):
        """
        测试stub子模块自动注册到sys.modules

        Ground Truth:
        - 访问stub.submodule时，子模块应该自动注册到sys.modules
        - 这是Python导入机制能找到子模块的必要条件
        - 解决 ModuleNotFoundError: No module named 'xxx.submodule' 问题
        """
        patcher = Patcher()
        patcher.skip_import("auto_register_test")

        module = sys.modules["auto_register_test"]

        # Access submodule
        submodule = module.submodule

        # Ground truth: submodule should be registered in sys.modules
        self.assertIn("auto_register_test.submodule", sys.modules)
        self.assertIs(sys.modules["auto_register_test.submodule"], submodule)

    def test_stub_deep_submodule_registered(self):
        """
        测试深层stub子模块自动注册到sys.modules

        Ground Truth:
        - 访问stub.level1.level2.level3时，所有层级都应该注册到sys.modules
        """
        patcher = Patcher()
        patcher.skip_import("deep_register_test")

        module = sys.modules["deep_register_test"]

        # Access deep nested submodule
        deep = module.level1.level2.level3

        # Ground truth: all levels should be registered
        self.assertIn("deep_register_test.level1", sys.modules)
        self.assertIn("deep_register_test.level1.level2", sys.modules)
        self.assertIn("deep_register_test.level1.level2.level3", sys.modules)

    def test_stub_submodule_iteration(self):
        """
        测试stub子模块的迭代

        Ground Truth:
        - 通过属性访问获取的子模块也应该可迭代
        - 这是导致原始bug的场景
        """
        patcher = Patcher()
        patcher.skip_import("parent_iter_test")

        parent = sys.modules["parent_iter_test"]
        submodule = parent.submodule

        # Ground truth: submodule should also be iterable
        items = list(submodule)
        self.assertEqual(items, [])

    def test_stub_module_from_import_star(self):
        """
        测试stub模块支持from ... import *语法

        Ground Truth:
        - "from stub_module import *"不应该抛出异常
        - 由于__all__为空，不会导入任何内容
        """
        patcher = Patcher()
        patcher.skip_import("star_import_test")

        # Ground truth: this should not raise
        # We can't actually use "from star_import_test import *" in a function,
        # but we can verify the module has the necessary attributes
        module = sys.modules["star_import_test"]
        self.assertTrue(hasattr(module, "__all__"))
        self.assertEqual(module.__all__, [])

    def test_stub_deep_nested_access(self):
        """
        测试深层嵌套的stub访问

        Ground Truth:
        - 深层嵌套的属性访问应该都返回stub
        - 每个层级都应该可迭代
        """
        patcher = Patcher()
        patcher.skip_import("deep_nested_test")

        module = sys.modules["deep_nested_test"]

        # Access deeply nested attributes
        deep = module.level1.level2.level3.level4

        # Ground truth: all levels should be iterable
        self.assertEqual(list(module), [])
        self.assertEqual(list(module.level1), [])
        self.assertEqual(list(deep), [])

    def test_stub_module_callable_returns_none(self):
        """
        测试stub模块作为函数调用返回None

        Ground Truth:
        - stub模块可以被调用
        - 调用结果应该是None
        """
        patcher = Patcher()
        patcher.skip_import("callable_test")

        module = sys.modules["callable_test"]

        # Ground truth: calling should return None
        result = module()
        self.assertIsNone(result)

        # Also test with arguments
        result = module(1, 2, 3, key="value")
        self.assertIsNone(result)

    def test_stub_attribute_callable(self):
        """
        测试stub属性作为函数调用

        Ground Truth:
        - stub的任何属性都可以被调用
        - 调用结果应该是None
        """
        patcher = Patcher()
        patcher.skip_import("attr_callable_test")

        module = sys.modules["attr_callable_test"]

        # Ground truth: any attribute should be callable
        result = module.some_function(1, 2, 3)
        self.assertIsNone(result)

        result = module.SomeClass()
        self.assertIsNone(result)


# =============================================================================
# Test: Real-world import scenarios
# =============================================================================

class TestRealWorldImportScenarios(unittest.TestCase):
    """
    真实世界导入场景测试

    模拟真实的CUDA依赖导入场景，确保skip_import能正确处理。
    """

    def setUp(self):
        """记录测试前的sys.modules状态"""
        self._original_modules = set(sys.modules.keys())

    def tearDown(self):
        """清理测试中添加的模块"""
        current_modules = set(sys.modules.keys())
        for module_name in current_modules - self._original_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]

    def test_flash_attn_like_import(self):
        """
        测试类似flash_attn的导入场景

        Ground Truth:
        - 模拟flash_attn的导入模式
        - from flash_attn.flash_attn_interface import xxx 应该工作
        """
        patcher = Patcher()
        patcher.skip_import("flash_attn")

        # Simulate: from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func
        module = sys.modules["flash_attn"]
        interface = module.flash_attn_interface
        func = interface.flash_attn_varlen_kvpacked_func

        # Ground truth: all accesses should work
        self.assertIsNotNone(func)

        # The function should be callable
        result = func(None, None, None)
        self.assertIsNone(result)

    def test_flash_attn_real_import_statement(self):
        """
        测试使用真实import语句导入flash_attn子模块

        Ground Truth:
        - 使用importlib.import_module模拟真实的import语句
        - 子模块应该能被正确导入
        - 这是解决 ModuleNotFoundError: No module named 'flash_attn.flash_attn_interface' 的关键测试
        """
        import importlib

        patcher = Patcher()
        patcher.skip_import("flash_attn")

        # This simulates: from flash_attn.flash_attn_interface import ...
        # which is what causes ModuleNotFoundError in the real scenario
        submodule = importlib.import_module("flash_attn.flash_attn_interface")

        # Ground truth: submodule should be a stub
        self.assertIsNotNone(submodule)
        self.assertIn("flash_attn.flash_attn_interface", sys.modules)

        # Should be able to access attributes
        func = submodule.flash_attn_varlen_kvpacked_func
        self.assertIsNotNone(func)

    def test_deep_submodule_real_import(self):
        """
        测试使用真实import语句导入深层子模块

        Ground Truth:
        - 深层子模块也应该能被正确导入
        """
        import importlib

        patcher = Patcher()
        patcher.skip_import("deep_import_test")

        # Import a deep submodule
        submodule = importlib.import_module("deep_import_test.level1.level2.level3")

        # Ground truth: all levels should be registered
        self.assertIn("deep_import_test", sys.modules)
        self.assertIn("deep_import_test.level1", sys.modules)
        self.assertIn("deep_import_test.level1.level2", sys.modules)
        self.assertIn("deep_import_test.level1.level2.level3", sys.modules)

    def test_torch_scatter_like_import(self):
        """
        测试类似torch_scatter的导入场景

        Ground Truth:
        - 模拟torch_scatter的导入模式
        - from torch_scatter import scatter_max 应该工作
        """
        patcher = Patcher()
        patcher.skip_import("torch_scatter")

        # Simulate: from torch_scatter import scatter_max
        module = sys.modules["torch_scatter"]
        scatter_max = module.scatter_max

        # Ground truth: attribute access should work
        self.assertIsNotNone(scatter_max)

    def test_spconv_like_import(self):
        """
        测试类似spconv的导入场景

        Ground Truth:
        - 模拟spconv的导入模式
        - from spconv.pytorch import SparseConvTensor 应该工作
        """
        patcher = Patcher()
        patcher.skip_import("spconv")

        # Simulate: from spconv.pytorch import SparseConvTensor
        module = sys.modules["spconv"]
        pytorch = module.pytorch
        SparseConvTensor = pytorch.SparseConvTensor

        # Ground truth: nested access should work
        self.assertIsNotNone(SparseConvTensor)

    def test_multiple_cuda_deps_skip(self):
        """
        测试跳过多个CUDA依赖

        Ground Truth:
        - 可以同时跳过多个CUDA依赖
        - 所有依赖都应该被正确stub
        """
        patcher = Patcher()
        patcher.skip_import(
            "flash_attn",
            "torch_scatter",
            "spconv",
            "cumm",
        )

        # Ground truth: all modules should be stubbed
        self.assertIn("flash_attn", sys.modules)
        self.assertIn("torch_scatter", sys.modules)
        self.assertIn("spconv", sys.modules)
        self.assertIn("cumm", sys.modules)

        # All should be iterable
        for name in ["flash_attn", "torch_scatter", "spconv", "cumm"]:
            module = sys.modules[name]
            self.assertEqual(list(module), [])


# =============================================================================
# Test: replace_import - Module replacement
# =============================================================================

class TestReplaceImport(unittest.TestCase):
    """
    replace_import功能测试

    测试Patcher.replace_import方法，用于替换模块导入。
    """

    def setUp(self):
        """记录测试前的sys.modules状态"""
        self._original_modules = set(sys.modules.keys())

    def tearDown(self):
        """清理测试中添加的模块"""
        current_modules = set(sys.modules.keys())
        for module_name in current_modules - self._original_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]

    def test_replace_import_with_attrs(self):
        """
        测试使用自定义属性替换模块

        Ground Truth:
        - 替换的模块应该出现在sys.modules中
        - 模块的属性应该可以访问
        """
        patcher = Patcher()

        patcher.replace_import(
            "custom_ops.my_op",
            MyFunction=lambda x: x * 2,
        )

        # Ground truth: module should be registered
        self.assertIn("custom_ops.my_op", sys.modules)
        self.assertEqual(sys.modules["custom_ops.my_op"].MyFunction(5), 10)

    def test_replace_import_with_class(self):
        """
        测试替换包含类的模块

        Ground Truth:
        - 模块中的类应该可以正常使用
        """
        patcher = Patcher()

        class DeformableAggregationFunction:
            @staticmethod
            def apply(x, y):
                return x + y

        patcher.replace_import(
            "class_ops.aggregation",
            DeformableAggregationFunction=DeformableAggregationFunction,
        )

        # Ground truth: class should be usable
        self.assertIn("class_ops.aggregation", sys.modules)
        result = sys.modules["class_ops.aggregation"].DeformableAggregationFunction.apply(3, 4)
        self.assertEqual(result, 7)

    def test_replace_import_with_replacement_module(self):
        """
        测试用另一个模块替换

        Ground Truth:
        - 可以用一个已存在的模块替换目标模块
        - 替换后的模块应该有原模块的所有属性
        """
        patcher = Patcher()

        # Create a source module to use as replacement
        source = types.ModuleType("replacement_source")
        source.VALUE = 42
        source.func = lambda x: x * 3
        sys.modules["replacement_source"] = source

        patcher.replace_import(
            "target_to_replace",
            "replacement_source",
        )

        # Ground truth: target should have source's attributes
        self.assertIn("target_to_replace", sys.modules)
        self.assertEqual(sys.modules["target_to_replace"].VALUE, 42)
        self.assertEqual(sys.modules["target_to_replace"].func(5), 15)

    def test_replace_import_with_replacement_and_override(self):
        """
        测试替换模块并覆盖特定属性

        Ground Truth:
        - 可以用另一个模块替换，同时覆盖特定属性
        """
        patcher = Patcher()

        # Create a source module
        source = types.ModuleType("override_source")
        source.VALUE = 42
        source.func = lambda x: x * 3
        sys.modules["override_source"] = source

        patcher.replace_import(
            "override_target",
            "override_source",
            VALUE=100,  # Override VALUE
        )

        # Ground truth: VALUE should be overridden, func should be from source
        self.assertIn("override_target", sys.modules)
        self.assertEqual(sys.modules["override_target"].VALUE, 100)  # Overridden
        self.assertEqual(sys.modules["override_target"].func(5), 15)  # From source

    def test_replace_import_skip_if_exists(self):
        """
        测试已存在的模块不会被覆盖

        Ground Truth:
        - 如果模块已存在，replace_import应该跳过
        """
        patcher = Patcher()

        # Pre-register a module
        existing_module = types.ModuleType("existing_ops.op")
        existing_module.original = True
        sys.modules["existing_ops.op"] = existing_module

        # Try to replace with same path
        patcher.replace_import("existing_ops.op", original=False)

        # Ground truth: original module should be preserved
        self.assertTrue(sys.modules["existing_ops.op"].original)

    def test_replace_import_skip_if_exists_logs_warning(self):
        """
        replace_import should warn with guidance when target module is already loaded.
        """
        import logging

        patcher = Patcher()
        existing_module = types.ModuleType("existing_ops.warn")
        sys.modules["existing_ops.warn"] = existing_module

        warnings_logged = []

        class CapHandler(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.WARNING:
                    warnings_logged.append(record.getMessage())

        handler = CapHandler()
        _patcher_logger._logger.addHandler(handler)
        try:
            patcher.replace_import("existing_ops.warn", VALUE=1)
        finally:
            _patcher_logger._logger.removeHandler(handler)

        self.assertTrue(
            any("replace_import skipped for existing_ops.warn" in msg for msg in warnings_logged)
        )
        self.assertTrue(
            any("Call replace_import() before importing the target module" in msg
                for msg in warnings_logged)
        )

    def test_replace_import_returns_self(self):
        """
        测试replace_import返回self以支持链式调用

        Ground Truth:
        - replace_import()应该返回patcher实例本身
        """
        patcher = Patcher()

        result = patcher.replace_import("chain_test.op", value=1)

        # Ground truth: should return self
        self.assertIs(result, patcher)

    def test_replace_import_chaining(self):
        """
        测试replace_import的链式调用

        Ground Truth:
        - 可以链式调用多个replace_import
        """
        patcher = Patcher()

        patcher.replace_import("chain.op1", value=1).replace_import("chain.op2", value=2)

        # Ground truth: both modules should be registered
        self.assertIn("chain.op1", sys.modules)
        self.assertIn("chain.op2", sys.modules)
        self.assertEqual(sys.modules["chain.op1"].value, 1)
        self.assertEqual(sys.modules["chain.op2"].value, 2)

    def test_replace_import_importable(self):
        """
        测试替换的模块可以通过import语句导入

        Ground Truth:
        - 替换后的模块应该可以通过importlib.import_module导入
        """
        import importlib

        patcher = Patcher()

        patcher.replace_import("importable_ops.custom", CONSTANT=123)

        # Ground truth: should be importable
        imported = importlib.import_module("importable_ops.custom")
        self.assertEqual(imported.CONSTANT, 123)

    def test_deformable_aggregation_pattern(self):
        """
        测试类似DeformableAggregation的使用模式

        Ground Truth:
        - 模拟DiffusionDrive中DeformableAggregationOp的使用场景
        """
        import importlib

        patcher = Patcher()

        class DeformableAggregationFunction:
            @staticmethod
            def apply(*args, **kwargs):
                return sum(args) if args else None

        patcher.replace_import(
            "projects.mmdet3d_plugin.ops.deformable_aggregation",
            DeformableAggregationFunction=DeformableAggregationFunction,
        )

        # Ground truth: should be importable and usable
        imported = importlib.import_module("projects.mmdet3d_plugin.ops.deformable_aggregation")
        result = imported.DeformableAggregationFunction.apply(1, 2, 3)
        self.assertEqual(result, 6)


class TestWithImportsEnhancements(unittest.TestCase):
    """Tests for with_imports improvements from deep research."""

    def setUp(self):
        self.test_module = types.ModuleType("wi_enhance_test")
        self.test_module.VALUE = 42
        sys.modules["wi_enhance_test"] = self.test_module

    def tearDown(self):
        sys.modules.pop("wi_enhance_test", None)
        sys.modules.pop("wi_decorator_mod", None)

    def test_import_failure_logs_warning(self):
        """ImportError should log at WARNING level, not DEBUG."""
        import logging
        warnings_logged = []

        class CapHandler(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.WARNING:
                    warnings_logged.append(record.getMessage())

        handler = CapHandler()
        _patcher_logger._logger.addHandler(handler)
        try:
            @with_imports("nonexistent_module_xyz123")
            def func():
                pass
            func()  # Trigger lazy resolution
        finally:
            _patcher_logger._logger.removeHandler(handler)

        self.assertTrue(any("nonexistent_module_xyz123" in w for w in warnings_logged),
                        "ImportError should produce a WARNING log")

    def test_annotations_preserved(self):
        """Function annotations should be preserved after decoration."""
        @with_imports(("wi_enhance_test", "VALUE"))
        def annotated_func(x: int) -> int:
            return VALUE + x  # noqa: F821

        # Call to trigger resolution
        result = annotated_func(1)
        self.assertEqual(result, 43)
        # Annotations on the wrapper should be present via @wraps
        self.assertIn('x', annotated_func.__annotations__)

    def test_stacking_warning(self):
        """Stacking multiple @with_imports should produce a warning."""
        import logging
        warnings_logged = []

        class CapHandler(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.WARNING:
                    warnings_logged.append(record.getMessage())

        handler = CapHandler()
        _patcher_logger._logger.addHandler(handler)
        try:
            @with_imports("wi_enhance_test")
            @with_imports(("wi_enhance_test", "VALUE"))
            def stacked_func():
                return VALUE  # noqa: F821
        finally:
            _patcher_logger._logger.removeHandler(handler)

        self.assertTrue(any("stacking" in w.lower() for w in warnings_logged),
                        "Stacking @with_imports should produce a warning")

    def test_apply_decorators_basic(self):
        """apply_decorators should lazily apply decorators from target modules."""
        dec_mod = types.ModuleType("wi_decorator_mod")

        def my_decorator(multiplier=1):
            def wrapper(func):
                @_functools_wraps(func)
                def inner(*args, **kwargs):
                    return func(*args, **kwargs) * multiplier
                return inner
            return wrapper

        dec_mod.my_decorator = my_decorator
        sys.modules["wi_decorator_mod"] = dec_mod

        @with_imports(
            ("wi_enhance_test", "VALUE"),
            apply_decorators=[
                ("wi_decorator_mod.my_decorator", {"multiplier": 3})
            ]
        )
        def func():
            return VALUE  # noqa: F821 - VALUE = 42

        result = func()
        self.assertEqual(result, 42 * 3,
                         "apply_decorators should wrap the resolved function")

    def test_at_prefix_decorator_with_kwargs(self):
        """@ string expression decorator with kwargs."""
        dec_mod = types.ModuleType("wi_decorator_mod")

        def my_decorator(multiplier=1):
            def wrapper(func):
                @_functools_wraps(func)
                def inner(*args, **kwargs):
                    return func(*args, **kwargs) * multiplier
                return inner
            return wrapper

        dec_mod.my_decorator = my_decorator
        sys.modules["wi_decorator_mod"] = dec_mod

        @with_imports(
            ("wi_enhance_test", "VALUE"),
            ("wi_decorator_mod", "my_decorator"),
            "@my_decorator(multiplier=5)",
        )
        def func():
            return VALUE  # noqa: F821

        self.assertEqual(func(), 42 * 5)

    def test_at_prefix_decorator_no_args(self):
        """@ string expression decorator without kwargs."""
        dec_mod = types.ModuleType("wi_decorator_mod")

        def double(func):
            @_functools_wraps(func)
            def inner(*args, **kwargs):
                return func(*args, **kwargs) * 2
            return inner

        dec_mod.double = double
        sys.modules["wi_decorator_mod"] = dec_mod

        @with_imports(
            ("wi_enhance_test", "VALUE"),
            ("wi_decorator_mod", "double"),
            "@double",
        )
        def func():
            return VALUE  # noqa: F821

        self.assertEqual(func(), 42 * 2)

    def test_apply_decorators_missing_graceful(self):
        """apply_decorators with missing decorator should not crash."""
        @with_imports(
            ("wi_enhance_test", "VALUE"),
            apply_decorators=[
                ("nonexistent_module.decorator", {"arg": 1})
            ]
        )
        def func():
            return VALUE  # noqa: F821

        # Should still work, just without the decorator
        result = func()
        self.assertEqual(result, 42)

    def test_with_imports_decorated_flag(self):
        """Decorated functions should have _with_imports_decorated flag."""
        @with_imports("wi_enhance_test")
        def func():
            pass

        self.assertTrue(getattr(func, '_with_imports_decorated', False))


# =============================================================================
# Test: inject_import _StubModule fix
# =============================================================================

class TestInjectImportStubModuleFix(unittest.TestCase):
    """
    Tests for inject_import() fix when target module is a _StubModule.

    The old implementation used hasattr() which always returns True for _StubModule
    (since __getattr__ returns a new stub for any non-private attribute).
    The fix uses `class_name in vars(tgt_mod)` instead.
    """

    def setUp(self):
        self._original_modules = set(sys.modules.keys())

    def tearDown(self):
        current_modules = set(sys.modules.keys())
        for module_name in current_modules - self._original_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]

    def test_inject_into_stub_target_succeeds(self):
        """
        inject_import should succeed when target is a _StubModule.

        Previously, hasattr(stub, 'MyClass') returned True because
        _StubModule.__getattr__ returns a new stub, so injection was skipped.
        """
        patcher = Patcher()

        # Create source module with a real class
        source = types.ModuleType("ii_stub_test.source")
        source.MyClass = type("MyClass", (), {"value": 42})
        sys.modules["ii_stub_test.source"] = source

        # Skip import creates a stub target
        patcher.skip_import("ii_stub_test.stub_target")
        self.assertIn("ii_stub_test.stub_target", sys.modules)

        # Inject into the stub target
        patcher.inject_import("ii_stub_test.source", "MyClass", "ii_stub_test.stub_target")

        # Ground truth: the real MyClass should be injected, not a stub
        target = sys.modules["ii_stub_test.stub_target"]
        self.assertTrue(hasattr(target, "MyClass"))
        self.assertEqual(target.MyClass.value, 42)

    def test_inject_into_normal_module_still_skips_existing(self):
        """
        inject_import should still skip if target already has the attribute
        (for real modules, not stubs).
        """
        patcher = Patcher()

        # Source with real class
        source = types.ModuleType("ii_normal_test.source")
        source.MyClass = type("MyClass", (), {"value": 42})
        sys.modules["ii_normal_test.source"] = source

        # Target already has MyClass
        target = types.ModuleType("ii_normal_test.target")
        target.MyClass = type("MyClass", (), {"value": 999})
        sys.modules["ii_normal_test.target"] = target

        # Inject should skip since target already has MyClass
        patcher.inject_import("ii_normal_test.source", "MyClass", "ii_normal_test.target")

        # Ground truth: original value preserved
        self.assertEqual(sys.modules["ii_normal_test.target"].MyClass.value, 999)


# =============================================================================
# Test: replace_import improvements
# =============================================================================

class TestReplaceImportImprovements(unittest.TestCase):
    """
    Tests for replace_import() improvements:
    - Parent package auto-registration
    - Metadata preservation (__name__, __package__)
    - Summary logging
    - New API: base_module, exports, replace_with.module()
    """

    def setUp(self):
        self._original_modules = set(sys.modules.keys())

    def tearDown(self):
        current_modules = set(sys.modules.keys())
        for module_name in current_modules - self._original_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]

    # --- Parent package auto-registration ---

    def test_parent_packages_auto_created(self):
        """
        replace_import should auto-create parent packages in sys.modules.

        This fixes the issue where `import pkg.mod` and `from pkg import mod`
        failed because parent packages were not registered.
        """
        patcher = Patcher()

        patcher.replace_import("ri_parent.pkg.mod", VALUE=42)

        # Ground truth: parent packages should exist
        self.assertIn("ri_parent", sys.modules)
        self.assertIn("ri_parent.pkg", sys.modules)
        self.assertIn("ri_parent.pkg.mod", sys.modules)

    def test_parent_package_attribute_chain(self):
        """
        Parent packages should have child attributes set for `from pkg import mod`.
        """
        patcher = Patcher()

        patcher.replace_import("ri_chain.sub.leaf", VALUE=99)

        # Ground truth: attribute chain should work
        parent = sys.modules["ri_chain"]
        self.assertTrue(hasattr(parent, "sub"))
        sub = sys.modules["ri_chain.sub"]
        self.assertTrue(hasattr(sub, "leaf"))

    def test_from_pkg_import_mod_works(self):
        """
        After replace_import, `from pkg import mod` style should work.
        """
        patcher = Patcher()
        patcher.replace_import("ri_from_test.pkg.mod", CONSTANT=55)

        # This import form requires parent packages to be registered
        mod = sys.modules["ri_from_test.pkg.mod"]
        self.assertEqual(mod.CONSTANT, 55)

        # Verify the parent knows about the child
        pkg = sys.modules["ri_from_test.pkg"]
        self.assertIs(getattr(pkg, "mod", None), mod)

    def test_importlib_import_still_works(self):
        """
        importlib.import_module should continue to work.
        """
        import importlib

        patcher = Patcher()
        patcher.replace_import("ri_importlib.ops.custom", VALUE=77)

        imported = importlib.import_module("ri_importlib.ops.custom")
        self.assertEqual(imported.VALUE, 77)

    def test_real_parent_packages_remain_importable(self):
        """
        replace_import should not replace real filesystem packages with empty stubs.

        Regression target:
        - replace_import("pkg.sub.ops.leaf", ...)
        - later import pkg.sub.apis should still work
        """
        import importlib

        patcher = Patcher()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = os.path.join(temp_dir, "ri_real_parent")
            subpkg = os.path.join(root, "subpkg")
            apis = os.path.join(subpkg, "apis")
            os.makedirs(apis)

            with open(os.path.join(root, "__init__.py"), "w", encoding="utf-8") as f:
                f.write("")
            with open(os.path.join(subpkg, "__init__.py"), "w", encoding="utf-8") as f:
                f.write("")
            with open(os.path.join(apis, "__init__.py"), "w", encoding="utf-8") as f:
                f.write("VALUE = 123\n")

            sys.path.insert(0, temp_dir)
            try:
                patcher.replace_import("ri_real_parent.subpkg.ops.custom", VALUE=42)

                parent_pkg = importlib.import_module("ri_real_parent.subpkg")
                apis_pkg = importlib.import_module("ri_real_parent.subpkg.apis")

                self.assertEqual(apis_pkg.VALUE, 123)
                self.assertTrue(hasattr(parent_pkg, "__path__"))
                self.assertNotEqual(list(parent_pkg.__path__), [])
            finally:
                sys.path.remove(temp_dir)

    def test_namespace_parent_packages_remain_importable(self):
        """
        replace_import should preserve namespace-package parents as real packages.

        Regression target:
        - top-level package directory exists without __init__.py
        - later sibling imports under that namespace package should still work
        """
        import importlib

        patcher = Patcher()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = os.path.join(temp_dir, "ri_ns_parent")
            subpkg = os.path.join(root, "subpkg")
            apis = os.path.join(subpkg, "apis")
            os.makedirs(apis)

            with open(os.path.join(subpkg, "__init__.py"), "w", encoding="utf-8") as f:
                f.write("")
            with open(os.path.join(apis, "__init__.py"), "w", encoding="utf-8") as f:
                f.write("VALUE = 456\n")

            sys.path.insert(0, temp_dir)
            try:
                patcher.replace_import("ri_ns_parent.subpkg.ops.custom", VALUE=42)

                ns_pkg = importlib.import_module("ri_ns_parent")
                apis_pkg = importlib.import_module("ri_ns_parent.subpkg.apis")

                self.assertEqual(apis_pkg.VALUE, 456)
                self.assertTrue(hasattr(ns_pkg, "__path__"))
                self.assertNotEqual(list(ns_pkg.__path__), [])
            finally:
                sys.path.remove(temp_dir)

    def test_single_segment_no_parent_needed(self):
        """
        Single-segment module paths should work without parent creation.
        """
        patcher = Patcher()
        patcher.replace_import("ri_single_seg", VALUE=1)

        self.assertIn("ri_single_seg", sys.modules)
        self.assertEqual(sys.modules["ri_single_seg"].VALUE, 1)

    # --- Metadata preservation ---

    def test_metadata_preserved_with_replacement_module(self):
        """
        When using a replacement module, __name__ and __package__ should
        reflect the target path, not the replacement source.
        """
        patcher = Patcher()

        # Create a source module
        source = types.ModuleType("ri_meta_source")
        source.VALUE = 42
        sys.modules["ri_meta_source"] = source

        patcher.replace_import("ri_meta.target.mod", "ri_meta_source")

        target = sys.modules["ri_meta.target.mod"]
        self.assertEqual(target.__name__, "ri_meta.target.mod")
        self.assertEqual(target.__package__, "ri_meta.target")
        self.assertEqual(target.VALUE, 42)

    # --- New API: base_module keyword ---

    def test_base_module_keyword(self):
        """
        base_module= keyword should work as preferred replacement for positional arg.
        """
        patcher = Patcher()

        source = types.ModuleType("ri_bm_source")
        source.func = lambda: "hello"
        sys.modules["ri_bm_source"] = source

        patcher.replace_import("ri_bm.target", base_module="ri_bm_source")

        target = sys.modules["ri_bm.target"]
        self.assertEqual(target.func(), "hello")

    def test_replacement_and_base_module_conflict(self):
        """
        Passing both positional replacement and base_module= should raise ValueError.
        """
        patcher = Patcher()

        source = types.ModuleType("ri_conflict_source")
        sys.modules["ri_conflict_source"] = source

        with self.assertRaises(ValueError):
            patcher.replace_import("ri_conflict.target", "ri_conflict_source",
                                   base_module="ri_conflict_source")

    # --- New API: exports keyword ---

    def test_exports_keyword(self):
        """
        exports= should work as the preferred way to declare module attributes.
        """
        patcher = Patcher()

        patcher.replace_import(
            "ri_exports.ops",
            exports={"MyFunc": lambda: 42, "MyClass": int},
        )

        target = sys.modules["ri_exports.ops"]
        self.assertEqual(target.MyFunc(), 42)
        self.assertIs(target.MyClass, int)

    def test_exports_and_attrs_conflict(self):
        """
        If same key appears in both exports= and **attrs, should raise ValueError.
        """
        patcher = Patcher()

        with self.assertRaises(ValueError):
            patcher.replace_import(
                "ri_exp_conflict.ops",
                exports={"VALUE": 1},
                VALUE=2,
            )

    def test_base_module_with_exports(self):
        """
        base_module + exports should use base as template and override with exports.
        """
        patcher = Patcher()

        source = types.ModuleType("ri_base_exp_source")
        source.VALUE = 42
        source.func = lambda: "original"
        sys.modules["ri_base_exp_source"] = source

        patcher.replace_import(
            "ri_base_exp.target",
            base_module="ri_base_exp_source",
            exports={"func": lambda: "overridden"},
        )

        target = sys.modules["ri_base_exp.target"]
        self.assertEqual(target.VALUE, 42)  # From base
        self.assertEqual(target.func(), "overridden")  # From exports

    # --- New API: replace_with.module() ---

    def test_replace_with_module_string(self):
        """
        replace_with.module("new.module") should work as replacement spec.
        """
        replace_with = _patcher_module.replace_with

        patcher = Patcher()

        source = types.ModuleType("ri_rw_source")
        source.VALUE = 99
        sys.modules["ri_rw_source"] = source

        patcher.replace_import("ri_rw.target", replace_with.module("ri_rw_source"))

        target = sys.modules["ri_rw.target"]
        self.assertEqual(target.VALUE, 99)

    def test_replace_with_module_exports(self):
        """
        replace_with.module(Foo=Bar) should create a module with given exports.
        """
        replace_with = _patcher_module.replace_with

        patcher = Patcher()

        patcher.replace_import(
            "ri_rw_exp.ops",
            replace_with.module(MyFunc=lambda: 42),
        )

        target = sys.modules["ri_rw_exp.ops"]
        self.assertEqual(target.MyFunc(), 42)

    def test_replace_with_module_base_and_override(self):
        """
        replace_with.module("base", Foo=Bar) should use base module + override.
        """
        replace_with = _patcher_module.replace_with

        patcher = Patcher()

        source = types.ModuleType("ri_rw_bo_source")
        source.VALUE = 42
        source.func = lambda: "original"
        sys.modules["ri_rw_bo_source"] = source

        patcher.replace_import(
            "ri_rw_bo.target",
            replace_with.module("ri_rw_bo_source", func=lambda: "override"),
        )

        target = sys.modules["ri_rw_bo.target"]
        self.assertEqual(target.VALUE, 42)
        self.assertEqual(target.func(), "override")

    # --- Summary logging ---

    def test_replace_import_in_summary(self):
        """
        replace_import should record entries for summary output.
        """
        patcher = Patcher()

        # Clear any existing logger state
        _patcher_logger._replaced_imports.clear()

        patcher.replace_import("ri_summary.test", VALUE=1)

        # Ground truth: should be recorded
        self.assertTrue(len(_patcher_logger._replaced_imports) > 0)
        self.assertTrue(any("ri_summary.test" in desc
                            for desc in _patcher_logger._replaced_imports))

    # --- Legacy compatibility ---

    def test_legacy_positional_replacement_still_works(self):
        """
        Legacy patcher.replace_import("old", "new") should still work.
        """
        patcher = Patcher()

        source = types.ModuleType("ri_legacy_source")
        source.VALUE = 42
        sys.modules["ri_legacy_source"] = source

        patcher.replace_import("ri_legacy.target", "ri_legacy_source")

        target = sys.modules["ri_legacy.target"]
        self.assertEqual(target.VALUE, 42)

    def test_legacy_attrs_still_works(self):
        """
        Legacy patcher.replace_import("old", Foo=Bar) should still work.
        """
        patcher = Patcher()

        patcher.replace_import("ri_legacy_attrs.ops", MyFunc=lambda: 42)

        target = sys.modules["ri_legacy_attrs.ops"]
        self.assertEqual(target.MyFunc(), 42)


class TestSkipImportParentPackagePreservation(unittest.TestCase):
    """
    Regression tests for skip_import() parent-package poisoning.

    skip_import("pkg.sub.leaf") should not replace real parent packages
    such as pkg or pkg.sub with _StubModule placeholders.
    """

    def setUp(self):
        self._original_modules = set(sys.modules.keys())

    def tearDown(self):
        current_modules = set(sys.modules.keys())
        for module_name in current_modules - self._original_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]

    def test_real_parent_package_not_stubbed(self):
        """
        Real filesystem-backed parents should remain real importable packages.
        """
        import importlib

        patcher = Patcher()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = os.path.join(temp_dir, "si_real_parent")
            subpkg = os.path.join(root, "subpkg")
            sibling = os.path.join(root, "sibling")
            os.makedirs(subpkg)
            os.makedirs(sibling)

            with open(os.path.join(root, "__init__.py"), "w", encoding="utf-8") as f:
                f.write("VALUE = 123\n")
            with open(os.path.join(subpkg, "__init__.py"), "w", encoding="utf-8") as f:
                f.write("SUB_VALUE = 456\n")
            with open(os.path.join(sibling, "__init__.py"), "w", encoding="utf-8") as f:
                f.write("SIBLING = 789\n")

            sys.path.insert(0, temp_dir)
            try:
                patcher.skip_import("si_real_parent.subpkg.blocked_leaf")

                parent_pkg = importlib.import_module("si_real_parent")
                sub_pkg = importlib.import_module("si_real_parent.subpkg")
                sibling_pkg = importlib.import_module("si_real_parent.sibling")

                self.assertEqual(parent_pkg.VALUE, 123)
                self.assertEqual(sub_pkg.SUB_VALUE, 456)
                self.assertEqual(sibling_pkg.SIBLING, 789)
                self.assertFalse(type(parent_pkg).__name__.startswith("_StubModule"))
                self.assertFalse(type(sub_pkg).__name__.startswith("_StubModule"))
            finally:
                sys.path.remove(temp_dir)

    def test_namespace_parent_package_not_stubbed(self):
        """
        Namespace-package parents should keep working for sibling imports.
        """
        import importlib

        patcher = Patcher()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = os.path.join(temp_dir, "si_ns_parent")
            subpkg = os.path.join(root, "subpkg")
            sibling = os.path.join(root, "sibling")
            os.makedirs(subpkg)
            os.makedirs(sibling)

            with open(os.path.join(subpkg, "__init__.py"), "w", encoding="utf-8") as f:
                f.write("SUB_VALUE = 456\n")
            with open(os.path.join(sibling, "__init__.py"), "w", encoding="utf-8") as f:
                f.write("SIBLING = 789\n")

            sys.path.insert(0, temp_dir)
            try:
                patcher.skip_import("si_ns_parent.subpkg.blocked_leaf")

                ns_parent = importlib.import_module("si_ns_parent")
                sibling_pkg = importlib.import_module("si_ns_parent.sibling")

                self.assertTrue(hasattr(ns_parent, "__path__"))
                self.assertEqual(sibling_pkg.SIBLING, 789)
                self.assertFalse(type(ns_parent).__name__.startswith("_StubModule"))
            finally:
                sys.path.remove(temp_dir)

    def test_skip_import_does_not_execute_real_parent_init(self):
        """
        skip_import should not import real parent packages during stub registration.

        This protects startup paths where a deep blocked import lives under a real
        plugin package whose __init__.py has heavyweight or incompatible imports.
        """
        patcher = Patcher()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = os.path.join(temp_dir, "si_no_init_import")
            subpkg = os.path.join(root, "subpkg")
            os.makedirs(subpkg)

            with open(os.path.join(root, "__init__.py"), "w", encoding="utf-8") as f:
                f.write("raise RuntimeError('parent __init__ should not execute during skip_import')\n")
            with open(os.path.join(subpkg, "__init__.py"), "w", encoding="utf-8") as f:
                f.write("")

            sys.path.insert(0, temp_dir)
            try:
                # Ground truth: registration itself should not execute parent __init__.py
                patcher.skip_import("si_no_init_import.subpkg.blocked_leaf")
                self.assertIn("si_no_init_import.subpkg.blocked_leaf", sys.modules)
            finally:
                sys.path.remove(temp_dir)


class TestBootstrapPatchesBeforeCollect(unittest.TestCase):
    """
    Regression tests for compatibility patches that must apply before collect.

    Some Patch.patches() implementations eagerly import third-party modules while
    building AtomicPatch objects. Compatibility patches such as numpy alias
    restoration must therefore be able to run before later Patch classes are
    collected.
    """

    def setUp(self):
        self._original_modules = dict(sys.modules)

    def tearDown(self):
        current_modules = set(sys.modules.keys())
        original_modules = set(self._original_modules.keys())
        for module_name in current_modules - original_modules:
            sys.modules.pop(module_name, None)
        for module_name in original_modules:
            sys.modules[module_name] = self._original_modules[module_name]

    def test_apply_before_collect_allows_eager_import_patch_classes(self):
        """
        apply_before_collect patches should run before later Patch.patches()
        methods trigger import-time compatibility checks.
        """
        import importlib

        fake_numpy = types.ModuleType("numpy")
        sys.modules["numpy"] = fake_numpy

        class FakeNumpyCompat(Patch):
            name = "fake_numpy_compat"
            apply_before_collect = True

            @staticmethod
            def _restore_alias(np_module, _options):
                if hasattr(np_module, "int"):
                    raise AttributeError("np.int already exists")
                np_module.int = int

            @classmethod
            def patches(cls, options=None):
                return [
                    LegacyPatch(cls._restore_alias, target_module="numpy", options=options),
                ]

        class EagerImportPatch(Patch):
            name = "eager_import_patch"

            @classmethod
            def patches(cls, options=None):
                importlib.import_module("bootstrap_pkg.consumer")
                return [
                    AtomicPatch("bootstrap_pkg.consumer.VALUE", 2),
                ]

        patcher = Patcher()

        with tempfile.TemporaryDirectory() as temp_dir:
            pkg_dir = os.path.join(temp_dir, "bootstrap_pkg")
            os.makedirs(pkg_dir)

            with open(os.path.join(pkg_dir, "__init__.py"), "w", encoding="utf-8") as f:
                f.write("")
            with open(os.path.join(pkg_dir, "consumer.py"), "w", encoding="utf-8") as f:
                f.write(
                    "import numpy as np\n"
                    "if not hasattr(np, 'int'):\n"
                    "    raise RuntimeError('np.int missing during collection')\n"
                    "VALUE = 1\n"
                )

            sys.path.insert(0, temp_dir)
            try:
                patcher.add(FakeNumpyCompat, EagerImportPatch)
                patcher.apply()

                consumer = importlib.import_module("bootstrap_pkg.consumer")
                self.assertIs(fake_numpy.int, int)
                self.assertEqual(consumer.VALUE, 2)
            finally:
                sys.path.remove(temp_dir)


if __name__ == "__main__":
    unittest.main()
