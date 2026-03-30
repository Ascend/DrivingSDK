# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
NumPy兼容性补丁模块测试文件

本文件测试NumPy库的兼容性补丁类。由于NumPy 1.20+版本移除了
numpy.bool和numpy.float等别名，此补丁用于恢复这些别名以保持
向后兼容性。

测试的补丁类：
- NumpyCompat: NumPy兼容性补丁，恢复被移除的类型别名
  - 通过显式补齐别名兼容 numpy.bool / numpy.float

测试目的：
1. 验证NumpyCompat补丁类的属性配置正确
2. 验证补丁返回LegacyPatch实例（2.0框架内的1.0等效写法）
3. 验证补丁函数为缺失别名注入内置类型
4. 验证补丁类与Patcher的集成使用
"""
import importlib.util
import os
import sys
import types
import unittest
from typing import List
from types import ModuleType
from unittest.mock import MagicMock, Mock, patch

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
_numpy_patch_module = _load_module_from_file(
    "mx_driving.patcher.numpy_patch",
    os.path.join(_patcher_dir, "numpy_patch.py")
)

# Import classes from loaded modules
AtomicPatch = _patch_module.AtomicPatch
BasePatch = _patch_module.BasePatch
LegacyPatch = _patch_module.LegacyPatch
Patch = _patch_module.Patch
Patcher = _patcher_module.Patcher
PatchResult = _reporting.PatchResult
PatchStatus = _reporting.PatchStatus

# Import numpy patch classes
NumpyCompat = _numpy_patch_module.NumpyCompat


class TestNumpyCompatPatch(unittest.TestCase):
    """
    NumpyCompat补丁类测试

    NumpyCompat补丁用于解决NumPy版本兼容性问题。NumPy 1.20+版本
    移除了numpy.bool、numpy.float、numpy.int等类型别名，导致依赖这些别名的
    旧代码无法运行。此补丁通过将这些别名指向Python内置类型来恢复兼容性。

    测试内容：
    - 补丁类的基本属性
    - patches()返回1个LegacyPatch（显式注入别名）
    - 补丁目标模块是否正确
    - 补丁函数是否注入正确的内置类型
    """

    def test_patch_class_attributes(self):
        """
        测试目的：验证NumpyCompat补丁类具有必需的属性
        测试步骤：
            1. 检查name属性是否为"numpy_compat"
            2. 检查是否有patches方法
        验证点：确保补丁类符合框架要求的接口规范
        """
        self.assertEqual(NumpyCompat.name, "numpy_compat")
        self.assertTrue(hasattr(NumpyCompat, 'patches'))

    def test_patches_returns_list(self):
        """
        测试目的：验证patches()方法返回LegacyPatch列表
        测试步骤：
            1. 调用NumpyCompat.patches()获取补丁列表
            2. 验证返回值是列表类型
            3. 验证列表长度为1（注入别名）
            4. 验证列表中每个元素都是LegacyPatch实例
        验证点：确保补丁定义符合框架规范
        """
        patches = NumpyCompat.patches()
        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 1)
        for p in patches:
            self.assertIsInstance(p, LegacyPatch)

    def test_patches_target_paths(self):
        """
        测试目的：验证补丁目标模块正确
        测试步骤：
            1. 获取补丁对象
            2. 验证target_module为numpy
        验证点：确保LegacyPatch指向正确的目标模块
        """
        patch_obj = NumpyCompat.patches()[0]
        self.assertEqual(patch_obj.target_module, "numpy")

    def test_patch_function_injects_builtin_aliases(self):
        """
        测试目的：验证补丁函数会为缺失别名注入内置类型
        验证点：等效于1.0写法（setattr补齐别名）
        """
        dummy = types.ModuleType("dummy_numpy")

        self.assertFalse(hasattr(dummy, "bool"))
        self.assertFalse(hasattr(dummy, "float"))
        self.assertFalse(hasattr(dummy, "int"))

        NumpyCompat._patch_deprecated_aliases(dummy, {})

        self.assertTrue(hasattr(dummy, "bool"))
        self.assertTrue(hasattr(dummy, "float"))
        self.assertTrue(hasattr(dummy, "int"))
        self.assertIs(dummy.bool, bool)
        self.assertIs(dummy.float, float)
        self.assertIs(dummy.int, int)

    def test_patch_function_noop_raises_attributeerror(self):
        """当别名已存在时，应抛出AttributeError以触发SKIPPED语义。"""
        dummy = types.ModuleType("dummy_numpy")
        dummy.bool = bool
        dummy.float = float
        dummy.int = int

        with self.assertRaises(AttributeError):
            NumpyCompat._patch_deprecated_aliases(dummy, {})


class TestPatcherIntegration(unittest.TestCase):
    """
    Patcher集成测试

    测试numpy补丁类与Patcher管理器的集成使用场景，
    验证补丁的添加、禁用、收集等操作是否正常工作。

    测试内容：
    - 向Patcher添加NumpyCompat补丁类
    - 通过名称禁用补丁
    - 验证补丁正确收集
    """

    def test_add_patch_class(self):
        """
        测试目的：验证可以向Patcher添加补丁类
        测试步骤：
            1. 创建Patcher实例
            2. 调用add方法添加NumpyCompat补丁类
            3. 验证add方法返回patcher实例（支持链式调用）
        验证点：确保补丁类可以正确添加到Patcher
        """
        patcher = Patcher()
        result = patcher.add(NumpyCompat)
        self.assertIs(result, patcher)

    def test_disable_patch_by_name(self):
        """
        测试目的：验证可以通过名称禁用补丁
        测试步骤：
            1. 创建Patcher实例
            2. 添加NumpyCompat补丁类
            3. 调用disable方法禁用该补丁
            4. 验证补丁名称被添加到黑名单
        验证点：确保禁用机制正常工作
        """
        patcher = Patcher()
        patcher.add(NumpyCompat)
        patcher.disable(NumpyCompat.name)
        self.assertIn(NumpyCompat.name, patcher._blacklist)

    def test_patches_collected(self):
        """
        测试目的：验证补丁正确收集
        测试步骤：
            1. 创建Patcher实例
            2. 添加NumpyCompat补丁类
            3. 调用_collect_all_patches收集所有补丁
            4. 验证收集到1个补丁
        验证点：确保NumpyCompat补丁点被正确收集
        """
        patcher = Patcher()
        patcher.add(NumpyCompat)
        patches = patcher._collect_all_patches()
        self.assertEqual(len(patches), 1)


if __name__ == "__main__":
    unittest.main()
