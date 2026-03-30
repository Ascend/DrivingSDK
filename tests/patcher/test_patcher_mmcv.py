# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
MMCV补丁模块测试

本模块测试mmcv_patch.py中定义的各种MMCV相关补丁类：
- MultiScaleDeformableAttention: 多尺度可变形注意力算子补丁
- DeformConv: 可变形卷积算子补丁
- ModulatedDeformConv: 调制可变形卷积算子补丁
- SparseConv3D: 3D稀疏卷积算子补丁
- Stream: CUDA流相关补丁
- DDP: 分布式数据并行补丁

测试目的：
- 验证各补丁类的属性和方法定义正确
- 验证patches()返回正确的AtomicPatch列表
- 验证补丁目标路径正确
- 验证与Patcher的集成
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
_mmcv_patch_module = _load_module_from_file(
    "mx_driving.patcher.mmcv_patch",
    os.path.join(_patcher_dir, "mmcv_patch.py")
)

# Import classes from loaded modules
AtomicPatch = _patch_module.AtomicPatch
BasePatch = _patch_module.BasePatch
Patch = _patch_module.Patch
Patcher = _patcher_module.Patcher
PatchResult = _reporting.PatchResult
PatchStatus = _reporting.PatchStatus

# Import mmcv patch classes
MultiScaleDeformableAttention = _mmcv_patch_module.MultiScaleDeformableAttention
DeformConv = _mmcv_patch_module.DeformConv
ModulatedDeformConv = _mmcv_patch_module.ModulatedDeformConv
SparseConv3D = _mmcv_patch_module.SparseConv3D
Stream = _mmcv_patch_module.Stream
DDP = _mmcv_patch_module.DDP


class TestMultiScaleDeformableAttentionPatch(unittest.TestCase):
    """
    MultiScaleDeformableAttention补丁类测试

    多尺度可变形注意力是BEVFormer等模型的核心算子。
    此补丁将CUDA实现替换为NPU兼容实现。
    """

    def test_patch_class_attributes(self):
        """
        测试补丁类必需属性

        验证内容：
        - name属性为"multi_scale_deformable_attention"
        - 存在patches方法

        为什么测试：确保补丁类符合框架规范
        """
        self.assertEqual(MultiScaleDeformableAttention.name, "multi_scale_deformable_attention")
        self.assertTrue(hasattr(MultiScaleDeformableAttention, 'patches'))

    def test_patches_returns_list(self):
        """
        测试patches()返回AtomicPatch列表

        验证内容：
        - 返回值是列表
        - 包含2个补丁(forward和backward)
        - 每个元素都是AtomicPatch实例
        """
        patches = MultiScaleDeformableAttention.patches()
        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 2)
        for p in patches:
            self.assertIsInstance(p, AtomicPatch)

    def test_patches_target_paths(self):
        """
        测试补丁目标路径

        验证内容：
        - 包含forward方法的补丁路径
        - 包含backward方法的补丁路径

        为什么测试：确保补丁应用到正确的MMCV函数
        """
        patches = MultiScaleDeformableAttention.patches()
        targets = [p.target for p in patches]
        self.assertIn("mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction.forward", targets)
        self.assertIn("mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction.backward", targets)

    def test_no_precheck_defined(self):
        """Test that MSDA relies on framework path validation, not a custom precheck."""
        self.assertFalse(hasattr(MultiScaleDeformableAttention, 'precheck'))

    def test_patches_use_replacement_wrapper(self):
        """Test that patches use replacement_wrapper for parameter adaptation."""
        patches = MultiScaleDeformableAttention.patches()
        for p in patches:
            # Should have replacement_wrapper for parameter/return value adaptation
            self.assertIsNotNone(p._replacement_wrapper)


class TestDeformConvPatch(unittest.TestCase):
    """Tests for DeformConv patch class."""

    def test_patch_class_attributes(self):
        """Test that patch class has required attributes."""
        self.assertEqual(DeformConv.name, "deform_conv")
        self.assertTrue(hasattr(DeformConv, 'patches'))

    def test_patches_returns_list(self):
        """Test that patches() returns a list of AtomicPatch."""
        patches = DeformConv.patches()
        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 2)
        for p in patches:
            self.assertIsInstance(p, AtomicPatch)

    def test_patches_target_paths(self):
        """Test that patches target correct paths."""
        patches = DeformConv.patches()
        targets = [p.target for p in patches]
        self.assertIn("mmcv.ops.deform_conv.DeformConv2dFunction", targets)
        self.assertIn("mmcv.ops.deform_conv.deform_conv2d", targets)

    def test_patches_use_string_replacement(self):
        """Test that patches use string replacement for lazy resolution."""
        patches = DeformConv.patches()
        for p in patches:
            # String replacement is stored in _replacement
            self.assertIsInstance(p._replacement, str)
            self.assertTrue(p._replacement.startswith("mx_driving."))


class TestModulatedDeformConvPatch(unittest.TestCase):
    """Tests for ModulatedDeformConv patch class."""

    def test_patch_class_attributes(self):
        """Test that patch class has required attributes."""
        self.assertEqual(ModulatedDeformConv.name, "modulated_deform_conv")
        self.assertTrue(hasattr(ModulatedDeformConv, 'patches'))

    def test_patches_returns_list(self):
        """Test that patches() returns a list of AtomicPatch."""
        patches = ModulatedDeformConv.patches()
        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 2)
        for p in patches:
            self.assertIsInstance(p, AtomicPatch)

    def test_patches_target_paths(self):
        """Test that patches target correct paths."""
        patches = ModulatedDeformConv.patches()
        targets = [p.target for p in patches]
        self.assertIn("mmcv.ops.modulated_deform_conv.ModulatedDeformConv2dFunction", targets)
        self.assertIn("mmcv.ops.modulated_deform_conv.modulated_deform_conv2d", targets)


class TestSparseConv3DPatch(unittest.TestCase):
    """Tests for SparseConv3D patch class."""

    def test_patch_class_attributes(self):
        """Test that patch class has required attributes."""
        self.assertEqual(SparseConv3D.name, "spconv3d")
        self.assertTrue(hasattr(SparseConv3D, 'patches'))

    def test_patches_returns_list(self):
        """Test that patches() returns a list of AtomicPatch."""
        patches = SparseConv3D.patches()
        self.assertIsInstance(patches, list)
        self.assertGreater(len(patches), 0)
        for p in patches:
            self.assertIsInstance(p, AtomicPatch)

    def test_patches_include_core_targets(self):
        """Test that patches include core sparse conv targets."""
        patches = SparseConv3D.patches()
        targets = [p.target for p in patches]
        # Check for core targets
        self.assertIn("mmcv.ops.SparseConvTensor", targets)
        self.assertIn("mmcv.ops.SparseSequential", targets)
        self.assertIn("mmcv.ops.SparseModule", targets)
        self.assertIn("mmcv.ops.SparseConvolution", targets)
        self.assertIn("mmcv.ops.SubMConv3d", targets)
        self.assertIn("mmcv.ops.SparseConv3d", targets)
        self.assertIn("mmcv.ops.SparseInverseConv3d", targets)


class TestStreamPatch(unittest.TestCase):
    """Tests for Stream patch class."""

    def test_patch_class_attributes(self):
        """Test that patch class has required attributes."""
        self.assertEqual(Stream.name, "stream")
        self.assertTrue(hasattr(Stream, 'patches'))
        self.assertTrue(hasattr(Stream, 'precheck'))

    def test_patches_returns_list(self):
        """Test that patches() returns a list of AtomicPatch."""
        patches = Stream.patches()
        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 1)

    def test_patches_target_scatter_forward(self):
        """Test that patches target Scatter.forward."""
        patches = Stream.patches()
        self.assertEqual(patches[0].target, "mmcv.parallel._functions.Scatter.forward")

    def test_scatter_forward_method_exists(self):
        """Test that scatter_forward method exists."""
        self.assertTrue(hasattr(Stream, 'scatter_forward'))
        self.assertTrue(callable(Stream.scatter_forward))


class TestDDPPatch(unittest.TestCase):
    """Tests for DDP patch class."""

    def test_patch_class_attributes(self):
        """Test that patch class has required attributes."""
        self.assertEqual(DDP.name, "ddp")
        self.assertTrue(hasattr(DDP, 'patches'))
        self.assertTrue(hasattr(DDP, 'precheck'))

    def test_patches_returns_list(self):
        """Test that patches() returns a list of AtomicPatch."""
        patches = DDP.patches()
        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 2)

    def test_patches_target_ddp(self):
        """Test that patches target DDP related paths."""
        patches = DDP.patches()
        targets = [p.target for p in patches]
        self.assertIn("mmcv.parallel.distributed.MMDistributedDataParallel._run_ddp_forward", targets)
        self.assertIn("mmcv.parallel.distributed.MMDistributedDataParallel", targets)

    def test_ddp_forward_method_exists(self):
        """Test that ddp_forward method exists."""
        self.assertTrue(hasattr(DDP, 'ddp_forward'))
        self.assertTrue(callable(DDP.ddp_forward))


class TestPatcherIntegration(unittest.TestCase):
    """Integration tests for using mmcv patches with Patcher."""

    def test_add_patch_class(self):
        """Test adding patch class to Patcher."""
        patcher = Patcher()
        result = patcher.add(MultiScaleDeformableAttention)
        self.assertIs(result, patcher)

    def test_add_multiple_patch_classes(self):
        """Test adding multiple patch classes to Patcher."""
        patcher = Patcher()
        patcher.add(
            MultiScaleDeformableAttention,
            DeformConv,
            ModulatedDeformConv,
        )
        # Verify patches are collected
        patches = patcher._collect_all_patches()
        self.assertGreater(len(patches), 0)

    def test_disable_patch_by_name(self):
        """Test disabling patch by name."""
        patcher = Patcher()
        patcher.add(MultiScaleDeformableAttention)
        patcher.disable(MultiScaleDeformableAttention.name)
        self.assertIn(MultiScaleDeformableAttention.name, patcher._blacklist)


if __name__ == "__main__":
    unittest.main()
