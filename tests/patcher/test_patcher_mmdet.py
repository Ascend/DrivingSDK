# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
MMDet（OpenMMLab Detection）补丁模块测试文件

本文件测试MMDet目标检测库的补丁类。MMDet是OpenMMLab生态系统中的目标检测框架，
提供了丰富的检测模型实现，包括ResNet骨干网络、采样器等组件。

测试的补丁类：
- PseudoSampler: 伪采样器补丁，用于替换mmdet的PseudoSampler.sample方法
- ResNetAddRelu: ResNet添加ReLU补丁，优化BasicBlock和Bottleneck的前向传播
- ResNetMaxPool: ResNet最大池化补丁，优化ResNet的前向传播
- ResNetFP16: ResNet半精度补丁，支持FP16推理，与ResNetMaxPool互斥

测试目的：
1. 验证各补丁类的属性配置正确（名称、patches方法、冲突声明等）
2. 验证补丁目标路径指向正确的mmdet模块
3. 验证补丁替换方法存在且可调用
4. 验证补丁类与Patcher的集成使用
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
_mmdet_patch_module = _load_module_from_file(
    "mx_driving.patcher.mmdet_patch",
    os.path.join(_patcher_dir, "mmdet_patch.py")
)

# Import classes from loaded modules
AtomicPatch = _patch_module.AtomicPatch
BasePatch = _patch_module.BasePatch
Patch = _patch_module.Patch
Patcher = _patcher_module.Patcher
PatchResult = _reporting.PatchResult
PatchStatus = _reporting.PatchStatus

# Import mmdet patch classes
PseudoSampler = _mmdet_patch_module.PseudoSampler
ResNetAddRelu = _mmdet_patch_module.ResNetAddRelu
ResNetMaxPool = _mmdet_patch_module.ResNetMaxPool
ResNetFP16 = _mmdet_patch_module.ResNetFP16


class TestPseudoSamplerPatch(unittest.TestCase):
    """
    PseudoSampler补丁类测试

    PseudoSampler是mmdet中的伪采样器，用于不需要真正采样的场景。
    此补丁替换其sample方法，优化NPU上的采样行为。

    测试内容：
    - 补丁类的基本属性（名称、patches方法、预检查方法）
    - patches()返回的AtomicPatch列表
    - 补丁目标路径是否正确指向mmdet的PseudoSampler.sample
    - 替换方法_sample_replacement是否存在
    """

    def test_patch_class_attributes(self):
        """
        测试目的：验证PseudoSampler补丁类具有必需的属性
        测试步骤：
            1. 检查name属性是否为"pseudo_sampler"
            2. 检查是否有patches方法
            3. 检查是否有_precheck方法
        验证点：确保补丁类符合框架要求的接口规范
        """
        self.assertEqual(PseudoSampler.name, "pseudo_sampler")
        self.assertTrue(hasattr(PseudoSampler, 'patches'))
        self.assertTrue(hasattr(PseudoSampler, '_precheck'))

    def test_patches_returns_list(self):
        """
        测试目的：验证patches()方法返回AtomicPatch列表
        测试步骤：
            1. 调用PseudoSampler.patches()获取补丁列表
            2. 验证返回值是列表类型
            3. 验证列表长度为1（只有一个补丁点）
            4. 验证列表中每个元素都是AtomicPatch实例
        验证点：确保补丁定义符合框架规范，返回正确的补丁对象
        """
        patches = PseudoSampler.patches()
        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 1)
        for p in patches:
            self.assertIsInstance(p, AtomicPatch)

    def test_patches_target_paths(self):
        """
        测试目的：验证补丁目标路径正确
        测试步骤：
            1. 获取所有补丁的target属性
            2. 验证目标路径包含mmdet.core.bbox.samplers.pseudo_sampler.PseudoSampler.sample
        验证点：确保补丁指向正确的mmdet模块路径，避免补丁应用到错误位置
        """
        patches = PseudoSampler.patches()
        targets = [p.target for p in patches]
        self.assertIn("mmdet.core.bbox.samplers.pseudo_sampler.PseudoSampler.sample", targets)

    def test_sample_replacement_method_exists(self):
        """
        测试目的：验证替换方法_sample_replacement存在且可调用
        测试步骤：
            1. 检查PseudoSampler类是否有_sample_replacement属性
            2. 检查该属性是否可调用
        验证点：确保补丁类实现了必要的替换方法
        """
        self.assertTrue(hasattr(PseudoSampler, '_sample_replacement'))
        self.assertTrue(callable(PseudoSampler._sample_replacement))


class TestResNetAddReluPatch(unittest.TestCase):
    """
    ResNetAddRelu补丁类测试

    ResNetAddRelu补丁优化ResNet骨干网络中BasicBlock和Bottleneck的前向传播，
    通过融合Add和ReLU操作来提升NPU上的推理性能。

    测试内容：
    - 补丁类的基本属性
    - patches()返回2个AtomicPatch（分别针对BasicBlock和Bottleneck）
    - 补丁目标路径是否正确指向mmdet的ResNet模块
    - 两个替换方法是否存在
    """

    def test_patch_class_attributes(self):
        """
        测试目的：验证ResNetAddRelu补丁类具有必需的属性
        测试步骤：
            1. 检查name属性是否为"resnet_add_relu"
            2. 检查是否有patches方法
        验证点：确保补丁类符合框架要求的接口规范
        """
        self.assertEqual(ResNetAddRelu.name, "resnet_add_relu")
        self.assertTrue(hasattr(ResNetAddRelu, 'patches'))

    def test_patches_returns_list(self):
        """
        测试目的：验证patches()方法返回AtomicPatch列表
        测试步骤：
            1. 调用ResNetAddRelu.patches()获取补丁列表
            2. 验证返回值是列表类型
            3. 验证列表长度为2（BasicBlock和Bottleneck各一个）
            4. 验证列表中每个元素都是AtomicPatch实例
        验证点：确保补丁定义符合框架规范，包含两个补丁点
        """
        patches = ResNetAddRelu.patches()
        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 2)
        for p in patches:
            self.assertIsInstance(p, AtomicPatch)

    def test_patches_target_paths(self):
        """
        测试目的：验证补丁目标路径正确
        测试步骤：
            1. 获取所有补丁的target属性
            2. 验证目标路径包含BasicBlock.forward
            3. 验证目标路径包含Bottleneck.forward
        验证点：确保补丁指向正确的ResNet模块路径
        """
        patches = ResNetAddRelu.patches()
        targets = [p.target for p in patches]
        self.assertIn("mmdet.models.backbones.resnet.BasicBlock.forward", targets)
        self.assertIn("mmdet.models.backbones.resnet.Bottleneck.forward", targets)

    def test_basicblock_forward_replacement_exists(self):
        """
        测试目的：验证BasicBlock前向传播替换方法存在
        测试步骤：
            1. 检查ResNetAddRelu类是否有_basicblock_forward_replacement属性
            2. 检查该属性是否可调用
        验证点：确保补丁类实现了BasicBlock的替换方法
        """
        self.assertTrue(hasattr(ResNetAddRelu, '_basicblock_forward_replacement'))
        self.assertTrue(callable(ResNetAddRelu._basicblock_forward_replacement))

    def test_bottleneck_forward_replacement_exists(self):
        """
        测试目的：验证Bottleneck前向传播替换方法存在
        测试步骤：
            1. 检查ResNetAddRelu类是否有_bottleneck_forward_replacement属性
            2. 检查该属性是否可调用
        验证点：确保补丁类实现了Bottleneck的替换方法
        """
        self.assertTrue(hasattr(ResNetAddRelu, '_bottleneck_forward_replacement'))
        self.assertTrue(callable(ResNetAddRelu._bottleneck_forward_replacement))


class TestResNetMaxPoolPatch(unittest.TestCase):
    """
    ResNetMaxPool补丁类测试

    ResNetMaxPool补丁优化ResNet骨干网络的前向传播，
    主要针对最大池化操作进行NPU适配优化。

    测试内容：
    - 补丁类的基本属性
    - patches()返回1个AtomicPatch
    - 补丁目标路径是否正确指向ResNet.forward
    - 替换方法是否存在
    """

    def test_patch_class_attributes(self):
        """
        测试目的：验证ResNetMaxPool补丁类具有必需的属性
        测试步骤：
            1. 检查name属性是否为"resnet_maxpool"
            2. 检查是否有patches方法
        验证点：确保补丁类符合框架要求的接口规范
        """
        self.assertEqual(ResNetMaxPool.name, "resnet_maxpool")
        self.assertTrue(hasattr(ResNetMaxPool, 'patches'))

    def test_patches_returns_list(self):
        """
        测试目的：验证patches()方法返回AtomicPatch列表
        测试步骤：
            1. 调用ResNetMaxPool.patches()获取补丁列表
            2. 验证返回值是列表类型
            3. 验证列表长度为1
            4. 验证列表中每个元素都是AtomicPatch实例
        验证点：确保补丁定义符合框架规范
        """
        patches = ResNetMaxPool.patches()
        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 1)
        for p in patches:
            self.assertIsInstance(p, AtomicPatch)

    def test_patches_target_paths(self):
        """
        测试目的：验证补丁目标路径正确
        测试步骤：
            1. 获取所有补丁的target属性
            2. 验证目标路径包含ResNet.forward
        验证点：确保补丁指向正确的ResNet模块路径
        """
        patches = ResNetMaxPool.patches()
        targets = [p.target for p in patches]
        self.assertIn("mmdet.models.backbones.resnet.ResNet.forward", targets)

    def test_forward_replacement_exists(self):
        """
        测试目的：验证前向传播替换方法存在
        测试步骤：
            1. 检查ResNetMaxPool类是否有_forward_replacement属性
            2. 检查该属性是否可调用
        验证点：确保补丁类实现了必要的替换方法
        """
        self.assertTrue(hasattr(ResNetMaxPool, '_forward_replacement'))
        self.assertTrue(callable(ResNetMaxPool._forward_replacement))


class TestResNetFP16Patch(unittest.TestCase):
    """
    ResNetFP16补丁类测试

    ResNetFP16补丁支持ResNet骨干网络的半精度（FP16）推理，
    与ResNetMaxPool补丁互斥，因为两者都修改ResNet.forward方法。

    测试内容：
    - 补丁类的基本属性（包括冲突声明）
    - 验证与ResNetMaxPool的冲突关系
    - patches()返回的AtomicPatch列表
    - 补丁目标路径与ResNetMaxPool相同
    """

    def test_patch_class_attributes(self):
        """
        测试目的：验证ResNetFP16补丁类具有必需的属性
        测试步骤：
            1. 检查name属性是否为"resnet_fp16"
            2. 检查是否有patches方法
            3. 检查是否有conflicts_with属性（声明冲突的补丁）
        验证点：确保补丁类符合框架要求的接口规范，并正确声明冲突
        """
        self.assertEqual(ResNetFP16.name, "resnet_fp16")
        self.assertTrue(hasattr(ResNetFP16, 'patches'))
        self.assertTrue(hasattr(ResNetFP16, 'conflicts_with'))

    def test_conflicts_with_resnet_maxpool(self):
        """
        测试目的：验证ResNetFP16声明与ResNetMaxPool的冲突
        测试步骤：
            1. 检查conflicts_with属性是否包含"resnet_maxpool"
        验证点：确保补丁正确声明了互斥关系，防止两个补丁同时应用
        为什么要测：ResNetFP16和ResNetMaxPool都修改ResNet.forward，
                   同时应用会导致冲突，必须声明互斥关系
        """
        self.assertIn("resnet_maxpool", ResNetFP16.conflicts_with)

    def test_patches_returns_list(self):
        """
        测试目的：验证patches()方法返回AtomicPatch列表
        测试步骤：
            1. 调用ResNetFP16.patches()获取补丁列表
            2. 验证返回值是列表类型
            3. 验证列表长度为1
            4. 验证列表中每个元素都是AtomicPatch实例
        验证点：确保补丁定义符合框架规范
        """
        patches = ResNetFP16.patches()
        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 1)
        for p in patches:
            self.assertIsInstance(p, AtomicPatch)

    def test_patches_target_same_as_maxpool(self):
        """
        测试目的：验证ResNetFP16与ResNetMaxPool目标路径相同
        测试步骤：
            1. 分别获取ResNetFP16和ResNetMaxPool的补丁列表
            2. 提取两者的target属性
            3. 比较两者的目标路径是否相同
        验证点：确认两个补丁确实修改同一位置，验证冲突声明的必要性
        为什么要测：这个测试证明了为什么需要conflicts_with声明，
                   因为两个补丁都修改ResNet.forward
        """
        fp16_patches = ResNetFP16.patches()
        maxpool_patches = ResNetMaxPool.patches()
        fp16_targets = [p.target for p in fp16_patches]
        maxpool_targets = [p.target for p in maxpool_patches]
        # Both should target ResNet.forward
        self.assertEqual(fp16_targets, maxpool_targets)


class TestPatcherIntegration(unittest.TestCase):
    """
    Patcher集成测试

    测试mmdet补丁类与Patcher管理器的集成使用场景，
    验证补丁的添加、禁用、替换等操作是否正常工作。

    测试内容：
    - 向Patcher添加单个补丁类
    - 向Patcher添加多个补丁类
    - 通过名称禁用补丁
    - 用ResNetFP16替换ResNetMaxPool的场景
    """

    def test_add_patch_class(self):
        """
        测试目的：验证可以向Patcher添加补丁类
        测试步骤：
            1. 创建Patcher实例
            2. 调用add方法添加PseudoSampler补丁类
            3. 验证add方法返回patcher实例（支持链式调用）
        验证点：确保补丁类可以正确添加到Patcher
        """
        patcher = Patcher()
        result = patcher.add(PseudoSampler)
        self.assertIs(result, patcher)

    def test_add_multiple_patch_classes(self):
        """
        测试目的：验证可以向Patcher添加多个补丁类
        测试步骤：
            1. 创建Patcher实例
            2. 一次性添加PseudoSampler、ResNetAddRelu、ResNetMaxPool三个补丁类
            3. 调用_collect_all_patches收集所有补丁
            4. 验证收集到的补丁数量大于0
        验证点：确保多个补丁类可以同时添加并正确收集
        """
        patcher = Patcher()
        patcher.add(
            PseudoSampler,
            ResNetAddRelu,
            ResNetMaxPool,
        )
        # Verify patches are collected
        patches = patcher._collect_all_patches()
        self.assertGreater(len(patches), 0)

    def test_disable_patch_by_name(self):
        """
        测试目的：验证可以通过名称禁用补丁
        测试步骤：
            1. 创建Patcher实例
            2. 添加ResNetMaxPool补丁类
            3. 调用disable方法禁用该补丁
            4. 验证补丁名称被添加到黑名单
        验证点：确保禁用机制正常工作，补丁可以被排除
        """
        patcher = Patcher()
        patcher.add(ResNetMaxPool)
        patcher.disable(ResNetMaxPool.name)
        self.assertIn(ResNetMaxPool.name, patcher._blacklist)

    def test_replace_maxpool_with_fp16(self):
        """
        测试目的：验证可以用ResNetFP16替换ResNetMaxPool
        测试步骤：
            1. 创建Patcher实例
            2. 添加ResNetMaxPool补丁类
            3. 禁用ResNetMaxPool
            4. 添加ResNetFP16补丁类
            5. 收集所有补丁并验证目标路径
        验证点：确保补丁替换场景正常工作
        为什么要测：这是一个常见的使用场景，用户可能需要在不同的补丁
                   实现之间切换（如从普通推理切换到FP16推理）
        """
        patcher = Patcher()
        patcher.add(ResNetMaxPool)
        patcher.disable(ResNetMaxPool.name)
        patcher.add(ResNetFP16)

        patches = patcher._collect_all_patches()
        # Should have patches from ResNetFP16 but not ResNetMaxPool
        # _collect_all_patches returns List[Tuple[BasePatch, str]]
        targets = [p[0].target for p in patches]
        self.assertIn("mmdet.models.backbones.resnet.ResNet.forward", targets)


if __name__ == "__main__":
    unittest.main()
