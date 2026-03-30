# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
MMDet3D（OpenMMLab 3D Detection）补丁模块测试文件

本文件测试MMDet3D三维目标检测库的补丁类。MMDet3D是OpenMMLab生态系统中的
三维目标检测框架，支持点云、多视角图像等多种3D感知任务。

测试的补丁类：
- NuScenesDataset: nuScenes数据集补丁，用于替换datasets模块的output_to_nusc_box函数
- NuScenesMetric: nuScenes评估指标补丁，用于替换evaluation模块的output_to_nusc_box函数
- NuScenes: 向后兼容别名，等同于NuScenesDataset

测试目的：
1. 验证NuScenesDataset和NuScenesMetric补丁类的属性配置正确
2. 验证补丁目标路径指向正确的mmdet3d模块
3. 验证补丁替换方法存在且可调用
4. 验证依赖检查方法正常工作
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
_mmdet3d_patch_module = _load_module_from_file(
    "mx_driving.patcher.mmdet3d_patch",
    os.path.join(_patcher_dir, "mmdet3d_patch.py")
)

# Import classes from loaded modules
AtomicPatch = _patch_module.AtomicPatch
BasePatch = _patch_module.BasePatch
Patch = _patch_module.Patch
Patcher = _patcher_module.Patcher
PatchResult = _reporting.PatchResult
PatchStatus = _reporting.PatchStatus

# Import mmdet3d patch classes
NuScenesDataset = _mmdet3d_patch_module.NuScenesDataset
NuScenesMetric = _mmdet3d_patch_module.NuScenesMetric
NuScenes = _mmdet3d_patch_module.NuScenes
_nuscenes_deps_available = _mmdet3d_patch_module._nuscenes_deps_available


class TestNuScenesDatasetPatch(unittest.TestCase):
    """
    NuScenesDataset补丁类测试

    NuScenesDataset补丁替换mmdet3d.datasets.nuscenes_dataset.output_to_nusc_box函数，
    优化NPU上的数据转换操作。

    测试内容：
    - 补丁类的基本属性（名称、patches方法）
    - patches()返回1个AtomicPatch
    - 补丁目标路径是否正确
    - 替换方法是否存在
    - 补丁是否配置了预检查函数
    """

    def test_patch_class_attributes(self):
        """
        测试目的：验证NuScenesDataset补丁类具有必需的属性
        测试步骤：
            1. 检查name属性是否为"nuscenes_dataset"
            2. 检查是否有patches方法
        验证点：确保补丁类符合框架要求的接口规范
        """
        self.assertEqual(NuScenesDataset.name, "nuscenes_dataset")
        self.assertTrue(hasattr(NuScenesDataset, 'patches'))

    def test_patches_returns_list(self):
        """
        测试目的：验证patches()方法返回AtomicPatch列表
        测试步骤：
            1. 调用NuScenesDataset.patches()获取补丁列表
            2. 验证返回值是列表类型
            3. 验证列表长度为1
            4. 验证列表中元素是AtomicPatch实例
        验证点：确保补丁定义符合框架规范
        """
        patches = NuScenesDataset.patches()
        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 1)
        self.assertIsInstance(patches[0], AtomicPatch)

    def test_patches_target_path(self):
        """
        测试目的：验证补丁目标路径正确
        测试步骤：
            1. 获取补丁的target属性
            2. 验证目标路径为mmdet3d.datasets.nuscenes_dataset.output_to_nusc_box
        验证点：确保补丁指向正确的mmdet3d模块路径
        """
        patches = NuScenesDataset.patches()
        self.assertEqual(
            patches[0].target,
            "mmdet3d.datasets.nuscenes_dataset.output_to_nusc_box"
        )

    def test_replacement_exists(self):
        """
        测试目的：验证替换方法存在
        测试步骤：
            1. 检查NuScenesDataset类是否有_replacement属性
            2. 检查该属性是否可调用
        验证点：确保补丁类实现了替换方法
        """
        self.assertTrue(hasattr(NuScenesDataset, '_replacement'))
        self.assertTrue(callable(NuScenesDataset._replacement))

    def test_patches_have_precheck(self):
        """
        测试目的：验证补丁配置了预检查函数
        测试步骤：
            1. 获取补丁
            2. 检查precheck属性不为None
        验证点：确保补丁有预检查函数
        """
        patches = NuScenesDataset.patches()
        self.assertIsNotNone(patches[0].precheck)


class TestNuScenesMetricPatch(unittest.TestCase):
    """
    NuScenesMetric补丁类测试

    NuScenesMetric补丁替换mmdet3d.evaluation.metrics.output_to_nusc_box函数，
    优化NPU上的评估指标计算。

    测试内容：
    - 补丁类的基本属性（名称、patches方法）
    - patches()返回1个AtomicPatch
    - 补丁目标路径是否正确
    - 替换方法是否存在
    - 补丁是否配置了预检查函数
    """

    def test_patch_class_attributes(self):
        """
        测试目的：验证NuScenesMetric补丁类具有必需的属性
        测试步骤：
            1. 检查name属性是否为"nuscenes_metric"
            2. 检查是否有patches方法
        验证点：确保补丁类符合框架要求的接口规范
        """
        self.assertEqual(NuScenesMetric.name, "nuscenes_metric")
        self.assertTrue(hasattr(NuScenesMetric, 'patches'))

    def test_patches_returns_list(self):
        """
        测试目的：验证patches()方法返回AtomicPatch列表
        测试步骤：
            1. 调用NuScenesMetric.patches()获取补丁列表
            2. 验证返回值是列表类型
            3. 验证列表长度为1
            4. 验证列表中元素是AtomicPatch实例
        验证点：确保补丁定义符合框架规范
        """
        patches = NuScenesMetric.patches()
        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 1)
        self.assertIsInstance(patches[0], AtomicPatch)

    def test_patches_target_path(self):
        """
        测试目的：验证补丁目标路径正确
        测试步骤：
            1. 获取补丁的target属性
            2. 验证目标路径为mmdet3d.evaluation.metrics.output_to_nusc_box
        验证点：确保补丁指向正确的mmdet3d模块路径
        """
        patches = NuScenesMetric.patches()
        self.assertEqual(
            patches[0].target,
            "mmdet3d.evaluation.metrics.output_to_nusc_box"
        )

    def test_replacement_exists(self):
        """
        测试目的：验证替换方法存在
        测试步骤：
            1. 检查NuScenesMetric类是否有_replacement属性
            2. 检查该属性是否可调用
        验证点：确保补丁类实现了替换方法
        """
        self.assertTrue(hasattr(NuScenesMetric, '_replacement'))
        self.assertTrue(callable(NuScenesMetric._replacement))

    def test_patches_have_precheck(self):
        """
        测试目的：验证补丁配置了预检查函数
        测试步骤：
            1. 获取补丁
            2. 检查precheck属性不为None
        验证点：确保补丁有预检查函数
        """
        patches = NuScenesMetric.patches()
        self.assertIsNotNone(patches[0].precheck)


class TestNuScenesAlias(unittest.TestCase):
    """
    NuScenes别名测试

    验证NuScenes是NuScenesDataset的别名，保持向后兼容性。
    """

    def test_nuscenes_is_alias(self):
        """
        测试目的：验证NuScenes是NuScenesDataset的别名
        测试步骤：
            1. 检查NuScenes是否与NuScenesDataset相同
        验证点：确保向后兼容性
        """
        self.assertIs(NuScenes, NuScenesDataset)


class TestDepsAvailable(unittest.TestCase):
    """
    依赖检查函数测试

    验证_nuscenes_deps_available函数正常工作。
    """

    def test_deps_available_returns_bool(self):
        """
        测试目的：验证依赖检查函数返回布尔值
        测试步骤：
            1. 调用_nuscenes_deps_available()
            2. 验证返回值是布尔类型
        验证点：确保依赖检查函数返回正确的类型
        """
        result = _nuscenes_deps_available()
        self.assertIsInstance(result, bool)


class TestPatcherIntegration(unittest.TestCase):
    """
    Patcher集成测试

    测试mmdet3d补丁类与Patcher管理器的集成使用场景，
    验证补丁的添加、禁用、收集等操作是否正常工作。

    测试内容：
    - 向Patcher添加NuScenesDataset和NuScenesMetric补丁类
    - 通过名称禁用补丁
    - 验证补丁正确收集
    """

    def test_add_nuscenes_dataset_patch(self):
        """
        测试目的：验证可以向Patcher添加NuScenesDataset补丁类
        测试步骤：
            1. 创建Patcher实例
            2. 调用add方法添加NuScenesDataset补丁类
            3. 验证add方法返回patcher实例（支持链式调用）
        验证点：确保补丁类可以正确添加到Patcher
        """
        patcher = Patcher()
        result = patcher.add(NuScenesDataset)
        self.assertIs(result, patcher)

    def test_add_nuscenes_metric_patch(self):
        """
        测试目的：验证可以向Patcher添加NuScenesMetric补丁类
        测试步骤：
            1. 创建Patcher实例
            2. 调用add方法添加NuScenesMetric补丁类
            3. 验证add方法返回patcher实例（支持链式调用）
        验证点：确保补丁类可以正确添加到Patcher
        """
        patcher = Patcher()
        result = patcher.add(NuScenesMetric)
        self.assertIs(result, patcher)

    def test_add_both_patches(self):
        """
        测试目的：验证可以同时添加两个补丁类
        测试步骤：
            1. 创建Patcher实例
            2. 添加NuScenesDataset和NuScenesMetric
            3. 收集所有补丁
            4. 验证收集到2个补丁
        验证点：确保两个补丁都被正确收集
        """
        patcher = Patcher()
        patcher.add(NuScenesDataset, NuScenesMetric)
        patches = patcher._collect_all_patches()
        self.assertEqual(len(patches), 2)

    def test_disable_nuscenes_dataset_by_name(self):
        """
        测试目的：验证可以通过名称禁用NuScenesDataset补丁
        测试步骤：
            1. 创建Patcher实例
            2. 添加NuScenesDataset补丁类
            3. 调用disable方法禁用该补丁
            4. 验证补丁名称被添加到黑名单
        验证点：确保禁用机制正常工作
        """
        patcher = Patcher()
        patcher.add(NuScenesDataset)
        patcher.disable(NuScenesDataset.name)
        self.assertIn(NuScenesDataset.name, patcher._blacklist)

    def test_disable_nuscenes_metric_by_name(self):
        """
        测试目的：验证可以通过名称禁用NuScenesMetric补丁
        测试步骤：
            1. 创建Patcher实例
            2. 添加NuScenesMetric补丁类
            3. 调用disable方法禁用该补丁
            4. 验证补丁名称被添加到黑名单
        验证点：确保禁用机制正常工作
        """
        patcher = Patcher()
        patcher.add(NuScenesMetric)
        patcher.disable(NuScenesMetric.name)
        self.assertIn(NuScenesMetric.name, patcher._blacklist)

    def test_disable_one_keep_other(self):
        """
        测试目的：验证可以单独禁用一个补丁而保留另一个
        测试步骤：
            1. 创建Patcher实例
            2. 添加两个补丁类
            3. 只禁用NuScenesDataset
            4. 验证只有NuScenesDataset在黑名单中
        验证点：确保可以独立控制每个补丁
        """
        patcher = Patcher()
        patcher.add(NuScenesDataset, NuScenesMetric)
        patcher.disable(NuScenesDataset.name)
        self.assertIn(NuScenesDataset.name, patcher._blacklist)
        self.assertNotIn(NuScenesMetric.name, patcher._blacklist)

    def test_nuscenes_dataset_patches_collected(self):
        """
        测试目的：验证NuScenesDataset补丁正确收集
        测试步骤：
            1. 创建Patcher实例
            2. 添加NuScenesDataset补丁类
            3. 调用_collect_all_patches收集所有补丁
            4. 验证收集到1个补丁
        验证点：确保NuScenesDataset的补丁点被正确收集
        """
        patcher = Patcher()
        patcher.add(NuScenesDataset)
        patches = patcher._collect_all_patches()
        self.assertEqual(len(patches), 1)

    def test_nuscenes_metric_patches_collected(self):
        """
        测试目的：验证NuScenesMetric补丁正确收集
        测试步骤：
            1. 创建Patcher实例
            2. 添加NuScenesMetric补丁类
            3. 调用_collect_all_patches收集所有补丁
            4. 验证收集到1个补丁
        验证点：确保NuScenesMetric的补丁点被正确收集
        """
        patcher = Patcher()
        patcher.add(NuScenesMetric)
        patches = patcher._collect_all_patches()
        self.assertEqual(len(patches), 1)


if __name__ == "__main__":
    unittest.main()
