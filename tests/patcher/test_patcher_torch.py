# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
PyTorch补丁模块测试文件

本文件测试PyTorch库的补丁类。这些补丁用于优化PyTorch在NPU上的
运行性能，主要针对张量索引和批量矩阵乘法操作。

测试的补丁类：
- TensorIndex: 张量索引补丁，优化torch.Tensor.__getitem__操作
- BatchMatmul: 批量矩阵乘法补丁，优化torch.matmul及相关操作
  - torch.matmul
  - torch.Tensor.matmul
  - torch.Tensor.__matmul__

测试目的：
1. 验证各补丁类的属性配置正确
2. 验证补丁目标路径指向正确的torch模块
3. 验证运行时检查方法存在且可调用
4. 验证补丁替换方法存在且可调用
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
_torch_patch_module = _load_module_from_file(
    "mx_driving.patcher.torch_patch",
    os.path.join(_patcher_dir, "torch_patch.py")
)

# Import classes from loaded modules
AtomicPatch = _patch_module.AtomicPatch
BasePatch = _patch_module.BasePatch
Patch = _patch_module.Patch
Patcher = _patcher_module.Patcher
PatchResult = _reporting.PatchResult
PatchStatus = _reporting.PatchStatus

# Import torch patch classes
TensorIndex = _torch_patch_module.TensorIndex
BatchMatmul = _torch_patch_module.BatchMatmul


class TestTensorIndexPatch(unittest.TestCase):
    """
    TensorIndex补丁类测试

    TensorIndex补丁优化PyTorch张量的索引操作（__getitem__），
    通过运行时检查决定是否使用优化的实现。

    测试内容：
    - 补丁类的基本属性
    - patches()返回1个AtomicPatch
    - 补丁目标路径是否正确
    - 运行时检查方法是否存在
    - 替换方法是否存在
    - 补丁是否配置了运行时检查
    """

    def test_patch_class_attributes(self):
        """
        测试目的：验证TensorIndex补丁类具有必需的属性
        测试步骤：
            1. 检查name属性是否为"tensor_index"
            2. 检查是否有patches方法
        验证点：确保补丁类符合框架要求的接口规范
        """
        self.assertEqual(TensorIndex.name, "tensor_index")
        self.assertTrue(hasattr(TensorIndex, 'patches'))

    def test_patches_returns_list(self):
        """
        测试目的：验证patches()方法返回AtomicPatch列表
        测试步骤：
            1. 调用TensorIndex.patches()获取补丁列表
            2. 验证返回值是列表类型
            3. 验证列表长度为1
            4. 验证列表中每个元素都是AtomicPatch实例
        验证点：确保补丁定义符合框架规范
        """
        patches = TensorIndex.patches()
        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 1)
        for p in patches:
            self.assertIsInstance(p, AtomicPatch)

    def test_patches_target_paths(self):
        """
        测试目的：验证补丁目标路径正确
        测试步骤：
            1. 获取所有补丁的target属性
            2. 验证目标路径包含torch.Tensor.__getitem__
        验证点：确保补丁指向正确的torch模块路径
        """
        patches = TensorIndex.patches()
        targets = [p.target for p in patches]
        self.assertIn("torch.Tensor.__getitem__", targets)

    def test_runtime_check_method_exists(self):
        """
        测试目的：验证运行时检查方法存在且可调用
        测试步骤：
            1. 检查TensorIndex类是否有_runtime_check属性
            2. 检查该属性是否可调用
        验证点：确保补丁类实现了运行时检查方法
        为什么要测：运行时检查用于在每次调用时决定是否使用优化实现
        """
        self.assertTrue(hasattr(TensorIndex, '_runtime_check'))
        self.assertTrue(callable(TensorIndex._runtime_check))

    def test_replacement_method_exists(self):
        """
        测试目的：验证替换方法存在且可调用
        测试步骤：
            1. 检查TensorIndex类是否有_replacement属性
            2. 检查该属性是否可调用
        验证点：确保补丁类实现了必要的替换方法
        """
        self.assertTrue(hasattr(TensorIndex, '_replacement'))
        self.assertTrue(callable(TensorIndex._replacement))

    def test_patches_have_runtime_check(self):
        """
        测试目的：验证补丁配置了运行时检查函数
        测试步骤：
            1. 获取所有补丁
            2. 遍历每个补丁，检查runtime_check属性不为None
        验证点：确保每个补丁都有运行时检查函数
        为什么要测：运行时检查允许在每次调用时动态决定是否使用优化实现
        """
        patches = TensorIndex.patches()
        for p in patches:
            self.assertIsNotNone(p.runtime_check)


class TestBatchMatmulPatch(unittest.TestCase):
    """
    BatchMatmul补丁类测试

    BatchMatmul补丁优化PyTorch的批量矩阵乘法操作，
    通过运行时检查张量形状决定是否使用优化的BMM实现。
    补丁同时修改torch.matmul、Tensor.matmul和Tensor.__matmul__。

    测试内容：
    - 补丁类的基本属性
    - patches()返回3个AtomicPatch
    - 补丁目标路径是否正确
    - 形状检查方法是否存在
    - 运行时检查方法是否存在
    - 替换方法是否存在
    - 形状检查方法对无效输入的处理
    """

    def test_patch_class_attributes(self):
        """
        测试目的：验证BatchMatmul补丁类具有必需的属性
        测试步骤：
            1. 检查name属性是否为"batch_matmul"
            2. 检查是否有patches方法
        验证点：确保补丁类符合框架要求的接口规范
        """
        self.assertEqual(BatchMatmul.name, "batch_matmul")
        self.assertTrue(hasattr(BatchMatmul, 'patches'))

    def test_patches_returns_list(self):
        """
        测试目的：验证patches()方法返回AtomicPatch列表
        测试步骤：
            1. 调用BatchMatmul.patches()获取补丁列表
            2. 验证返回值是列表类型
            3. 验证列表长度为3（matmul、Tensor.matmul、__matmul__各一个）
            4. 验证列表中每个元素都是AtomicPatch实例
        验证点：确保补丁定义符合框架规范，包含三个补丁点
        """
        patches = BatchMatmul.patches()
        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 3)
        for p in patches:
            self.assertIsInstance(p, AtomicPatch)

    def test_patches_target_paths(self):
        """
        测试目的：验证补丁目标路径正确
        测试步骤：
            1. 获取所有补丁的target属性
            2. 验证目标路径包含torch.matmul
            3. 验证目标路径包含torch.Tensor.matmul
            4. 验证目标路径包含torch.Tensor.__matmul__
        验证点：确保补丁指向正确的torch模块路径
        """
        patches = BatchMatmul.patches()
        targets = [p.target for p in patches]
        self.assertIn("torch.matmul", targets)
        self.assertIn("torch.Tensor.matmul", targets)
        self.assertIn("torch.Tensor.__matmul__", targets)

    def test_check_shape_bmm_method_exists(self):
        """
        测试目的：验证形状检查方法存在且可调用
        测试步骤：
            1. 检查BatchMatmul类是否有_check_shape_bmm属性
            2. 检查该属性是否可调用
        验证点：确保补丁类实现了形状检查方法
        为什么要测：形状检查用于判断是否可以使用BMM优化
        """
        self.assertTrue(hasattr(BatchMatmul, '_check_shape_bmm'))
        self.assertTrue(callable(BatchMatmul._check_shape_bmm))

    def test_runtime_check_method_exists(self):
        """
        测试目的：验证运行时检查方法存在且可调用
        测试步骤：
            1. 检查BatchMatmul类是否有_runtime_check属性
            2. 检查该属性是否可调用
        验证点：确保补丁类实现了运行时检查方法
        """
        self.assertTrue(hasattr(BatchMatmul, '_runtime_check'))
        self.assertTrue(callable(BatchMatmul._runtime_check))

    def test_replacement_method_exists(self):
        """
        测试目的：验证替换方法存在且可调用
        测试步骤：
            1. 检查BatchMatmul类是否有_replacement属性
            2. 检查该属性是否可调用
        验证点：确保补丁类实现了必要的替换方法
        """
        self.assertTrue(hasattr(BatchMatmul, '_replacement'))
        self.assertTrue(callable(BatchMatmul._replacement))

    def test_patches_have_runtime_check(self):
        """
        测试目的：验证补丁配置了运行时检查函数
        测试步骤：
            1. 获取所有补丁
            2. 遍历每个补丁，检查runtime_check属性不为None
        验证点：确保每个补丁都有运行时检查函数
        """
        patches = BatchMatmul.patches()
        for p in patches:
            self.assertIsNotNone(p.runtime_check)

    def test_check_shape_bmm_with_invalid_input(self):
        """
        测试目的：验证形状检查方法对无效输入的处理
        测试步骤：
            1. 使用None作为输入调用_check_shape_bmm
            2. 验证返回False
            3. 使用没有dim属性的对象调用_check_shape_bmm
            4. 验证返回False
        验证点：确保形状检查方法能正确处理无效输入
        为什么要测：运行时可能传入各种类型的参数，
                   形状检查必须能安全处理无效输入
        """
        # Test with non-tensor input
        result = BatchMatmul._check_shape_bmm(None, None)
        self.assertFalse(result)

        # Test with object without dim attribute
        class FakeObj:
            pass
        result = BatchMatmul._check_shape_bmm(FakeObj(), FakeObj())
        self.assertFalse(result)


class TestPatcherIntegration(unittest.TestCase):
    """
    Patcher集成测试

    测试torch补丁类与Patcher管理器的集成使用场景，
    验证补丁的添加、禁用、收集等操作是否正常工作。

    测试内容：
    - 向Patcher添加TensorIndex补丁类
    - 向Patcher添加BatchMatmul补丁类
    - 向Patcher添加多个补丁类
    - 通过名称禁用补丁
    - 验证补丁正确收集
    """

    def test_add_tensor_index_patch(self):
        """
        测试目的：验证可以向Patcher添加TensorIndex补丁类
        测试步骤：
            1. 创建Patcher实例
            2. 调用add方法添加TensorIndex补丁类
            3. 验证add方法返回patcher实例（支持链式调用）
        验证点：确保补丁类可以正确添加到Patcher
        """
        patcher = Patcher()
        result = patcher.add(TensorIndex)
        self.assertIs(result, patcher)

    def test_add_batch_matmul_patch(self):
        """
        测试目的：验证可以向Patcher添加BatchMatmul补丁类
        测试步骤：
            1. 创建Patcher实例
            2. 调用add方法添加BatchMatmul补丁类
            3. 验证add方法返回patcher实例
        验证点：确保补丁类可以正确添加到Patcher
        """
        patcher = Patcher()
        result = patcher.add(BatchMatmul)
        self.assertIs(result, patcher)

    def test_add_multiple_patch_classes(self):
        """
        测试目的：验证可以向Patcher添加多个补丁类
        测试步骤：
            1. 创建Patcher实例
            2. 一次性添加TensorIndex和BatchMatmul两个补丁类
            3. 调用_collect_all_patches收集所有补丁
            4. 验证收集到4个补丁（1+3）
        验证点：确保多个补丁类可以同时添加并正确收集
        """
        patcher = Patcher()
        patcher.add(TensorIndex, BatchMatmul)
        patches = patcher._collect_all_patches()
        self.assertEqual(len(patches), 4)  # 1 + 3

    def test_disable_patch_by_name(self):
        """
        测试目的：验证可以通过名称禁用补丁
        测试步骤：
            1. 创建Patcher实例
            2. 添加TensorIndex补丁类
            3. 调用disable方法禁用该补丁
            4. 验证补丁名称被添加到黑名单
        验证点：确保禁用机制正常工作
        """
        patcher = Patcher()
        patcher.add(TensorIndex)
        patcher.disable(TensorIndex.name)
        self.assertIn(TensorIndex.name, patcher._blacklist)

    def test_patches_collected_tensor_index(self):
        """
        测试目的：验证TensorIndex补丁正确收集
        测试步骤：
            1. 创建Patcher实例
            2. 添加TensorIndex补丁类
            3. 调用_collect_all_patches收集所有补丁
            4. 验证收集到1个补丁
        验证点：确保TensorIndex的补丁点被正确收集
        """
        patcher = Patcher()
        patcher.add(TensorIndex)
        patches = patcher._collect_all_patches()
        self.assertEqual(len(patches), 1)

    def test_patches_collected_batch_matmul(self):
        """
        测试目的：验证BatchMatmul补丁正确收集
        测试步骤：
            1. 创建Patcher实例
            2. 添加BatchMatmul补丁类
            3. 调用_collect_all_patches收集所有补丁
            4. 验证收集到3个补丁
        验证点：确保BatchMatmul的三个补丁点都被正确收集
        """
        patcher = Patcher()
        patcher.add(BatchMatmul)
        patches = patcher._collect_all_patches()
        self.assertEqual(len(patches), 3)


if __name__ == "__main__":
    unittest.main()
