# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
torch_scatter补丁模块测试文件

本文件测试torch_scatter库的补丁类。torch_scatter是PyTorch的扩展库，
提供了高效的scatter操作实现，常用于图神经网络等场景。

测试的补丁类：
- TorchScatter: torch_scatter补丁，优化scatter操作在NPU上的性能
  - scatter_sum: 按索引求和操作
  - scatter_mean: 按索引求均值操作
  - scatter_max: 按索引求最大值操作

  每个操作都有两个补丁点：
  - 子模块路径：torch_scatter.scatter.scatter_xxx
  - 顶层导出路径：torch_scatter.scatter_xxx

测试目的：
1. 验证TorchScatter补丁类的属性配置正确
2. 验证补丁目标路径指向正确的torch_scatter模块
3. 验证补丁替换方法存在且可调用
4. 验证补丁同时覆盖子模块和顶层导出
5. 验证补丁类与Patcher的集成使用
"""
import importlib.util
import os
import sys
import types
import unittest
from typing import List
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
_torch_scatter_patch_module = _load_module_from_file(
    "mx_driving.patcher.torch_scatter_patch",
    os.path.join(_patcher_dir, "torch_scatter_patch.py")
)

# Import classes from loaded modules
AtomicPatch = _patch_module.AtomicPatch
BasePatch = _patch_module.BasePatch
Patch = _patch_module.Patch
Patcher = _patcher_module.Patcher
PatchResult = _reporting.PatchResult
PatchStatus = _reporting.PatchStatus

# Import torch_scatter patch classes
TorchScatter = _torch_scatter_patch_module.TorchScatter


class TestTorchScatterPatch(unittest.TestCase):
    """
    TorchScatter补丁类测试

    TorchScatter补丁优化torch_scatter库的scatter操作，
    包括scatter_sum、scatter_mean、scatter_max三种操作。
    每种操作都需要同时补丁子模块和顶层导出两个位置。

    测试内容：
    - 补丁类的基本属性
    - patches()返回6个AtomicPatch（3种操作×2个位置）
    - 补丁目标路径是否正确
    - 三个替换方法是否存在
    - 补丁是否同时覆盖子模块和顶层导出
    """

    def test_patch_class_attributes(self):
        """
        测试目的：验证TorchScatter补丁类具有必需的属性
        测试步骤：
            1. 检查name属性是否为"torch_scatter"
            2. 检查是否有patches方法
        验证点：确保补丁类符合框架要求的接口规范
        """
        self.assertEqual(TorchScatter.name, "torch_scatter")
        self.assertTrue(hasattr(TorchScatter, 'patches'))

    def test_patches_returns_list(self):
        """
        测试目的：验证patches()方法返回AtomicPatch列表
        测试步骤：
            1. 调用TorchScatter.patches()获取补丁列表
            2. 验证返回值是列表类型
            3. 验证列表长度为6（3种操作×2个位置）
            4. 验证列表中每个元素都是AtomicPatch实例
        验证点：确保补丁定义符合框架规范，包含所有6个补丁点
        """
        patches = TorchScatter.patches()
        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 6)
        for p in patches:
            self.assertIsInstance(p, AtomicPatch)

    def test_patches_target_paths(self):
        """
        测试目的：验证补丁目标路径正确
        测试步骤：
            1. 获取所有补丁的target属性
            2. 验证子模块路径：scatter.scatter_sum/mean/max
            3. 验证顶层导出路径：scatter_sum/mean/max
        验证点：确保补丁指向正确的torch_scatter模块路径
        为什么要测：torch_scatter同时在子模块和顶层导出这些函数，
                   必须同时补丁两个位置才能完全覆盖
        """
        patches = TorchScatter.patches()
        targets = [p.target for p in patches]
        # Submodule patches
        self.assertIn("torch_scatter.scatter.scatter_sum", targets)
        self.assertIn("torch_scatter.scatter.scatter_mean", targets)
        self.assertIn("torch_scatter.scatter.scatter_max", targets)
        # Top-level re-exports
        self.assertIn("torch_scatter.scatter_sum", targets)
        self.assertIn("torch_scatter.scatter_mean", targets)
        self.assertIn("torch_scatter.scatter_max", targets)

    def test_patches_use_replacement_wrapper(self):
        """
        测试目的：验证补丁使用replacement_wrapper进行类型转换
        测试步骤：
            1. 获取所有补丁
            2. 检查每个补丁的_replacement_wrapper属性不为None
        验证点：确保补丁使用replacement_wrapper进行参数类型转换
        """
        patches = TorchScatter.patches()
        for p in patches:
            self.assertIsNotNone(p._replacement_wrapper)

    def test_patches_cover_both_submodule_and_toplevel(self):
        """
        测试目的：验证补丁同时覆盖子模块和顶层导出
        测试步骤：
            1. 获取所有补丁的target属性
            2. 对于每种操作（sum、mean、max）：
               - 验证存在子模块路径补丁
               - 验证存在顶层导出路径补丁
        验证点：确保每种操作都有完整的补丁覆盖
        为什么要测：用户可能通过不同方式导入这些函数，
                   必须同时补丁两个位置才能保证功能完整
        """
        patches = TorchScatter.patches()
        targets = [p.target for p in patches]

        # Each operation should have both submodule and top-level patch
        for op in ['scatter_sum', 'scatter_mean', 'scatter_max']:
            submodule_target = f"torch_scatter.scatter.{op}"
            toplevel_target = f"torch_scatter.{op}"
            self.assertIn(submodule_target, targets)
            self.assertIn(toplevel_target, targets)


class TestPatcherIntegration(unittest.TestCase):
    """
    Patcher集成测试

    测试torch_scatter补丁类与Patcher管理器的集成使用场景，
    验证补丁的添加、禁用、收集等操作是否正常工作。

    测试内容：
    - 向Patcher添加TorchScatter补丁类
    - 通过名称禁用补丁
    - 验证补丁正确收集
    """

    def test_add_patch_class(self):
        """
        测试目的：验证可以向Patcher添加补丁类
        测试步骤：
            1. 创建Patcher实例
            2. 调用add方法添加TorchScatter补丁类
            3. 验证add方法返回patcher实例（支持链式调用）
        验证点：确保补丁类可以正确添加到Patcher
        """
        patcher = Patcher()
        result = patcher.add(TorchScatter)
        self.assertIs(result, patcher)

    def test_disable_patch_by_name(self):
        """
        测试目的：验证可以通过名称禁用补丁
        测试步骤：
            1. 创建Patcher实例
            2. 添加TorchScatter补丁类
            3. 调用disable方法禁用该补丁
            4. 验证补丁名称被添加到黑名单
        验证点：确保禁用机制正常工作
        """
        patcher = Patcher()
        patcher.add(TorchScatter)
        patcher.disable(TorchScatter.name)
        self.assertIn(TorchScatter.name, patcher._blacklist)

    def test_patches_collected(self):
        """
        测试目的：验证补丁正确收集
        测试步骤：
            1. 创建Patcher实例
            2. 添加TorchScatter补丁类
            3. 调用_collect_all_patches收集所有补丁
            4. 验证收集到6个补丁
        验证点：确保TorchScatter的所有6个补丁点都被正确收集
        """
        patcher = Patcher()
        patcher.add(TorchScatter)
        patches = patcher._collect_all_patches()
        self.assertEqual(len(patches), 6)


if __name__ == "__main__":
    unittest.main()
