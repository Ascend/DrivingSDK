# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
MMEngine和MMCV训练引擎补丁模块测试文件

本文件测试MMEngine（OpenMMLab 2.x训练引擎）和MMCV（1.x版本）的补丁类。
这些补丁主要用于优化训练循环、优化器、分布式训练等核心功能。

测试的补丁类：
MMEngine补丁：
- OptimizerWrapper: 优化器包装器补丁，优化OptimWrapper的初始化
- build_mmengine_epoch_train_loop_patch: 构建基于epoch的训练循环补丁
- build_mmengine_iter_train_loop_patch: 构建基于iteration的训练循环补丁

MMCV 1.x补丁：
- OptimizerHooks: 优化器钩子补丁，替换4种优化器钩子实现
- Stream: 流处理补丁，优化Scatter.forward方法
- DDP: 分布式数据并行补丁，优化MMDistributedDataParallel
- build_mmcv_epoch_runner_patch: 构建基于epoch的运行器补丁
- build_mmcv_iter_runner_patch: 构建基于iteration的运行器补丁

测试目的：
1. 验证各补丁类的属性配置正确
2. 验证补丁目标路径指向正确的模块
3. 验证补丁替换方法和工厂方法存在
4. 验证训练循环补丁构建器正确工作
5. 验证性能分析和断点配置功能
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
_mmengine_patch_module = _load_module_from_file(
    "mx_driving.patcher.mmengine_patch",
    os.path.join(_patcher_dir, "mmengine_patch.py")
)
_mmcv_patch_module = _load_module_from_file(
    "mx_driving.patcher.mmcv_patch",
    os.path.join(_patcher_dir, "mmcv_patch.py")
)

# Import classes from loaded modules
AtomicPatch = _patch_module.AtomicPatch
BasePatch = _patch_module.BasePatch
LegacyPatch = _patch_module.LegacyPatch
Patch = _patch_module.Patch
Patcher = _patcher_module.Patcher
PatchResult = _reporting.PatchResult
PatchStatus = _reporting.PatchStatus

# Import mmengine patch classes
OptimizerWrapper = _mmengine_patch_module.OptimizerWrapper
build_mmengine_epoch_train_loop_patch = _mmengine_patch_module.build_mmengine_epoch_train_loop_patch
build_mmengine_iter_train_loop_patch = _mmengine_patch_module.build_mmengine_iter_train_loop_patch

# Import mmcv patch classes (for mmcv 1.x patches)
OptimizerHooks = _mmcv_patch_module.OptimizerHooks
Stream = _mmcv_patch_module.Stream
DDP = _mmcv_patch_module.DDP
build_mmcv_epoch_runner_patch = _mmcv_patch_module.build_mmcv_epoch_runner_patch
build_mmcv_iter_runner_patch = _mmcv_patch_module.build_mmcv_iter_runner_patch


class TestOptimizerWrapperPatch(unittest.TestCase):
    """
    OptimizerWrapper补丁类测试（MMEngine）

    OptimizerWrapper补丁优化MMEngine的OptimWrapper初始化过程，
    通过包装器模式增强优化器的功能。

    测试内容：
    - 补丁类的基本属性
    - patches()返回1个AtomicPatch
    - 补丁目标路径是否正确
    - 补丁是否使用包装器模式
    - _wrap_init方法是否存在
    """

    def test_patch_class_attributes(self):
        """
        测试目的：验证OptimizerWrapper补丁类具有必需的属性
        测试步骤：
            1. 检查name属性是否为"optimizer_wrapper"
            2. 检查是否有patches方法
            3. 检查是否有_wrap_init方法
        验证点：确保补丁类符合框架要求的接口规范
        """
        self.assertEqual(OptimizerWrapper.name, "optimizer_wrapper")
        self.assertTrue(hasattr(OptimizerWrapper, 'patches'))
        self.assertTrue(hasattr(OptimizerWrapper, '_wrap_init'))

    def test_patches_returns_list(self):
        """
        测试目的：验证patches()方法返回AtomicPatch列表
        测试步骤：
            1. 调用OptimizerWrapper.patches()获取补丁列表
            2. 验证返回值是列表类型
            3. 验证列表长度为1
            4. 验证列表中每个元素都是AtomicPatch实例
        验证点：确保补丁定义符合框架规范
        """
        patches = OptimizerWrapper.patches()
        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 1)
        for p in patches:
            self.assertIsInstance(p, AtomicPatch)

    def test_patches_target_paths(self):
        """
        测试目的：验证补丁目标路径正确
        测试步骤：
            1. 获取所有补丁的target属性
            2. 验证目标路径包含OptimWrapper.__init__
        验证点：确保补丁指向正确的mmengine模块路径
        """
        patches = OptimizerWrapper.patches()
        targets = [p.target for p in patches]
        self.assertIn("mmengine.optim.optimizer.optimizer_wrapper.OptimWrapper.__init__", targets)

    def test_patches_use_wrapper_mode(self):
        """
        测试目的：验证补丁使用包装器模式
        测试步骤：
            1. 获取所有补丁
            2. 检查每个补丁的_target_wrapper属性不为None
        验证点：确保补丁使用包装器模式而非直接替换
        为什么要测：包装器模式允许在原方法前后添加逻辑，
                   而不是完全替换原方法
        """
        patches = OptimizerWrapper.patches()
        for p in patches:
            self.assertIsNotNone(p._target_wrapper)

    def test_wrap_init_method_exists(self):
        """
        测试目的：验证_wrap_init方法存在且可调用
        测试步骤：
            1. 检查OptimizerWrapper类是否有_wrap_init属性
            2. 检查该属性是否可调用
        验证点：确保补丁类实现了必要的包装方法
        """
        self.assertTrue(hasattr(OptimizerWrapper, '_wrap_init'))
        self.assertTrue(callable(OptimizerWrapper._wrap_init))


class TestOptimizerHooksPatch(unittest.TestCase):
    """
    OptimizerHooks补丁类测试（MMCV 1.x）

    OptimizerHooks补丁替换MMCV的4种优化器钩子实现，
    通过注册表补丁方式替换HOOKS注册表中的钩子类。

    测试内容：
    - 补丁类的基本属性
    - patches()返回4个RegistryPatch
    - 补丁目标注册表是否正确
    - 注册的钩子名称是否正确
    - 工厂方法是否存在
    """

    def test_patch_class_attributes(self):
        """
        测试目的：验证OptimizerHooks补丁类具有必需的属性
        测试步骤：
            1. 检查name属性是否为"optimizer_hooks"
            2. 检查是否有patches方法
        验证点：确保补丁类符合框架要求的接口规范
        """
        self.assertEqual(OptimizerHooks.name, "optimizer_hooks")
        self.assertTrue(hasattr(OptimizerHooks, 'patches'))

    def test_patches_returns_list(self):
        """
        测试目的：验证patches()方法返回补丁列表
        测试步骤：
            1. 调用OptimizerHooks.patches()获取补丁列表
            2. 验证返回值是列表类型
            3. 验证列表长度为4（4种优化器钩子）
        验证点：确保补丁定义包含所有4种优化器钩子
        """
        patches = OptimizerHooks.patches()
        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 4)  # 4 optimizer hooks

    def test_patches_are_registry_patches(self):
        """
        测试目的：验证补丁是RegistryPatch实例
        测试步骤：
            1. 获取所有补丁
            2. 验证每个补丁都是RegistryPatch实例
        验证点：确保使用正确的补丁类型（注册表补丁）
        为什么要测：OptimizerHooks通过注册表机制替换钩子，
                   必须使用RegistryPatch而非AtomicPatch
        """
        RegistryPatch = _patch_module.RegistryPatch
        patches = OptimizerHooks.patches()
        for p in patches:
            self.assertIsInstance(p, RegistryPatch)

    def test_patches_target_hooks_registry(self):
        """
        测试目的：验证补丁目标注册表正确
        测试步骤：
            1. 获取所有补丁
            2. 验证每个补丁的registry属性指向HOOKS注册表
        验证点：确保补丁指向正确的MMCV钩子注册表
        """
        patches = OptimizerHooks.patches()
        for p in patches:
            self.assertEqual(p.registry, "mmcv.runner.hooks.optimizer.HOOKS")

    def test_patches_register_correct_names(self):
        """
        测试目的：验证补丁注册正确的钩子名称
        测试步骤：
            1. 获取所有补丁的register_name属性
            2. 验证包含OptimizerHook
            3. 验证包含GradientCumulativeOptimizerHook
            4. 验证包含Fp16OptimizerHook
            5. 验证包含GradientCumulativeFp16OptimizerHook
        验证点：确保所有4种优化器钩子都被正确注册
        """
        patches = OptimizerHooks.patches()
        names = [p.register_name for p in patches]
        self.assertIn("OptimizerHook", names)
        self.assertIn("GradientCumulativeOptimizerHook", names)
        self.assertIn("Fp16OptimizerHook", names)
        self.assertIn("GradientCumulativeFp16OptimizerHook", names)

    def test_factory_methods_exist(self):
        """
        测试目的：验证工厂方法存在
        测试步骤：
            1. 检查是否有_create_optimizer_hook方法
            2. 检查是否有_create_gradient_cumulative_optimizer_hook方法
            3. 检查是否有_create_fp16_optimizer_hook方法
            4. 检查是否有_create_gradient_cumulative_fp16_optimizer_hook方法
        验证点：确保补丁类实现了所有必要的工厂方法
        为什么要测：工厂方法用于创建替换的钩子类，是补丁功能的核心
        """
        self.assertTrue(hasattr(OptimizerHooks, '_create_optimizer_hook'))
        self.assertTrue(hasattr(OptimizerHooks, '_create_gradient_cumulative_optimizer_hook'))
        self.assertTrue(hasattr(OptimizerHooks, '_create_fp16_optimizer_hook'))
        self.assertTrue(hasattr(OptimizerHooks, '_create_gradient_cumulative_fp16_optimizer_hook'))


class TestStreamPatch(unittest.TestCase):
    """
    Stream补丁类测试（MMCV 1.x）

    Stream补丁优化MMCV的Scatter.forward方法，
    用于改进数据分发到多GPU的流处理性能。

    测试内容：
    - 补丁类的基本属性（包括预检查方法）
    - patches()返回1个AtomicPatch
    - 补丁目标路径是否正确
    - scatter_forward方法是否存在
    """

    def test_patch_class_attributes(self):
        """
        测试目的：验证Stream补丁类具有必需的属性
        测试步骤：
            1. 检查name属性是否为"stream"
            2. 检查是否有patches方法
            3. 检查是否有precheck方法
        验证点：确保补丁类符合框架要求的接口规范
        """
        self.assertEqual(Stream.name, "stream")
        self.assertTrue(hasattr(Stream, 'patches'))
        self.assertTrue(hasattr(Stream, 'precheck'))

    def test_patches_returns_list(self):
        """
        测试目的：验证patches()方法返回AtomicPatch列表
        测试步骤：
            1. 调用Stream.patches()获取补丁列表
            2. 验证返回值是列表类型
            3. 验证列表长度为1
            4. 验证列表中每个元素都是AtomicPatch实例
        验证点：确保补丁定义符合框架规范
        """
        patches = Stream.patches()
        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 1)
        for p in patches:
            self.assertIsInstance(p, AtomicPatch)

    def test_patches_target_scatter_forward(self):
        """
        测试目的：验证补丁目标路径正确
        测试步骤：
            1. 获取第一个补丁的target属性
            2. 验证目标路径为Scatter.forward
        验证点：确保补丁指向正确的MMCV模块路径
        """
        patches = Stream.patches()
        self.assertEqual(patches[0].target, "mmcv.parallel._functions.Scatter.forward")

    def test_scatter_forward_method_exists(self):
        """
        测试目的：验证scatter_forward方法存在且可调用
        测试步骤：
            1. 检查Stream类是否有scatter_forward属性
            2. 检查该属性是否可调用
        验证点：确保补丁类实现了必要的替换方法
        """
        self.assertTrue(hasattr(Stream, 'scatter_forward'))
        self.assertTrue(callable(Stream.scatter_forward))


class TestDDPPatch(unittest.TestCase):
    """
    DDP补丁类测试（MMCV 1.x）

    DDP补丁优化MMCV的MMDistributedDataParallel实现，
    用于改进分布式数据并行训练的性能。

    测试内容：
    - 补丁类的基本属性（包括预检查方法）
    - patches()返回2个AtomicPatch
    - 补丁目标路径是否正确
    - ddp_forward方法是否存在
    """

    def test_patch_class_attributes(self):
        """
        测试目的：验证DDP补丁类具有必需的属性
        测试步骤：
            1. 检查name属性是否为"ddp"
            2. 检查是否有patches方法
            3. 检查是否有precheck方法
        验证点：确保补丁类符合框架要求的接口规范
        """
        self.assertEqual(DDP.name, "ddp")
        self.assertTrue(hasattr(DDP, 'patches'))
        self.assertTrue(hasattr(DDP, 'precheck'))

    def test_patches_returns_list(self):
        """
        测试目的：验证patches()方法返回AtomicPatch列表
        测试步骤：
            1. 调用DDP.patches()获取补丁列表
            2. 验证返回值是列表类型
            3. 验证列表长度为2
        验证点：确保补丁定义包含两个补丁点
        """
        patches = DDP.patches()
        self.assertIsInstance(patches, list)
        self.assertEqual(len(patches), 2)

    def test_patches_target_ddp(self):
        """
        测试目的：验证补丁目标路径正确
        测试步骤：
            1. 获取所有补丁的target属性
            2. 验证目标路径包含_run_ddp_forward方法
            3. 验证目标路径包含MMDistributedDataParallel类
        验证点：确保补丁指向正确的MMCV分布式模块路径
        """
        patches = DDP.patches()
        targets = [p.target for p in patches]
        self.assertIn("mmcv.parallel.distributed.MMDistributedDataParallel._run_ddp_forward", targets)
        self.assertIn("mmcv.parallel.distributed.MMDistributedDataParallel", targets)

    def test_ddp_forward_method_exists(self):
        """
        测试目的：验证ddp_forward方法存在且可调用
        测试步骤：
            1. 检查DDP类是否有ddp_forward属性
            2. 检查该属性是否可调用
        验证点：确保补丁类实现了必要的替换方法
        """
        self.assertTrue(hasattr(DDP, 'ddp_forward'))
        self.assertTrue(callable(DDP.ddp_forward))


class TestTrainingLoopPatchBuilders(unittest.TestCase):
    """
    训练循环补丁构建器测试

    测试用于构建训练循环补丁的工厂函数，这些函数根据配置选项
    创建LegacyPatch实例，支持性能分析和断点功能。

    测试内容：
    - build_mmcv_epoch_runner_patch构建器
    - build_mmcv_iter_runner_patch构建器
    - build_mmengine_epoch_train_loop_patch构建器
    - build_mmengine_iter_train_loop_patch构建器
    - 性能分析选项传递
    """

    def test_build_mmcv_epoch_runner_patch(self):
        """
        测试目的：验证build_mmcv_epoch_runner_patch返回LegacyPatch
        测试步骤：
            1. 创建配置选项字典
            2. 调用build_mmcv_epoch_runner_patch构建补丁
            3. 验证返回值是LegacyPatch实例
            4. 验证目标模块是mmcv
        验证点：确保构建器正确创建MMCV epoch运行器补丁
        """
        options = {
            'enable_profiler': False,
            'enable_brake': False,
        }
        patch = build_mmcv_epoch_runner_patch(options)
        self.assertIsInstance(patch, LegacyPatch)
        self.assertEqual(patch.target_module, "mmcv")

    def test_build_mmcv_iter_runner_patch(self):
        """
        测试目的：验证build_mmcv_iter_runner_patch返回LegacyPatch
        测试步骤：
            1. 创建配置选项字典
            2. 调用build_mmcv_iter_runner_patch构建补丁
            3. 验证返回值是LegacyPatch实例
            4. 验证目标模块是mmcv
        验证点：确保构建器正确创建MMCV iter运行器补丁
        """
        options = {
            'enable_profiler': False,
            'enable_brake': False,
        }
        patch = build_mmcv_iter_runner_patch(options)
        self.assertIsInstance(patch, LegacyPatch)
        self.assertEqual(patch.target_module, "mmcv")

    def test_build_mmengine_epoch_train_loop_patch(self):
        """
        测试目的：验证build_mmengine_epoch_train_loop_patch返回LegacyPatch
        测试步骤：
            1. 创建配置选项字典
            2. 调用build_mmengine_epoch_train_loop_patch构建补丁
            3. 验证返回值是LegacyPatch实例
            4. 验证目标模块是mmengine
        验证点：确保构建器正确创建MMEngine epoch训练循环补丁
        """
        options = {
            'enable_profiler': False,
            'enable_brake': False,
        }
        patch = build_mmengine_epoch_train_loop_patch(options)
        self.assertIsInstance(patch, LegacyPatch)
        self.assertEqual(patch.target_module, "mmengine")

    def test_build_mmengine_iter_train_loop_patch(self):
        """
        测试目的：验证build_mmengine_iter_train_loop_patch返回LegacyPatch
        测试步骤：
            1. 创建配置选项字典
            2. 调用build_mmengine_iter_train_loop_patch构建补丁
            3. 验证返回值是LegacyPatch实例
            4. 验证目标模块是mmengine
        验证点：确保构建器正确创建MMEngine iter训练循环补丁
        """
        options = {
            'enable_profiler': False,
            'enable_brake': False,
        }
        patch = build_mmengine_iter_train_loop_patch(options)
        self.assertIsInstance(patch, LegacyPatch)
        self.assertEqual(patch.target_module, "mmengine")

    def test_profiling_options_passed_to_patch(self):
        """
        测试目的：验证性能分析选项正确传递给补丁
        测试步骤：
            1. 创建包含性能分析选项的配置字典
            2. 调用build_mmcv_epoch_runner_patch构建补丁
            3. 验证返回值是LegacyPatch实例
        验证点：确保性能分析配置可以正确传递
        为什么要测：性能分析是训练调优的重要功能，
                   需要确保配置选项能正确传递到补丁中
        """
        options = {
            'enable_profiler': True,
            'enable_brake': True,
            'brake_step': 100,
            'profiling_path': '/tmp/prof',
            'profiling_level': 1,
        }
        patch = build_mmcv_epoch_runner_patch(options)
        self.assertIsInstance(patch, LegacyPatch)


class TestPatcherIntegration(unittest.TestCase):
    """
    Patcher集成测试

    测试mmengine和mmcv补丁类与Patcher管理器的集成使用场景，
    验证补丁的添加、禁用、性能分析配置、断点配置等操作。

    测试内容：
    - 向Patcher添加单个补丁类
    - 向Patcher添加多个补丁类
    - 通过名称禁用补丁
    - 配置性能分析功能
    - 配置训练断点功能
    """

    def test_add_patch_class(self):
        """
        测试目的：验证可以向Patcher添加补丁类
        测试步骤：
            1. 创建Patcher实例
            2. 调用add方法添加OptimizerWrapper补丁类
            3. 验证add方法返回patcher实例（支持链式调用）
        验证点：确保补丁类可以正确添加到Patcher
        """
        patcher = Patcher()
        result = patcher.add(OptimizerWrapper)
        self.assertIs(result, patcher)

    def test_add_multiple_patch_classes(self):
        """
        测试目的：验证可以向Patcher添加多个补丁类
        测试步骤：
            1. 创建Patcher实例
            2. 一次性添加OptimizerWrapper、OptimizerHooks、Stream、DDP四个补丁类
            3. 调用_collect_all_patches收集所有补丁
            4. 验证收集到的补丁数量大于0
        验证点：确保多个补丁类可以同时添加并正确收集
        """
        patcher = Patcher()
        patcher.add(
            OptimizerWrapper,
            OptimizerHooks,
            Stream,
            DDP,
        )
        # Verify patches are collected
        patches = patcher._collect_all_patches()
        self.assertGreater(len(patches), 0)

    def test_disable_patch_by_name(self):
        """
        测试目的：验证可以通过名称禁用补丁
        测试步骤：
            1. 创建Patcher实例
            2. 添加OptimizerWrapper补丁类
            3. 调用disable方法禁用该补丁
            4. 验证补丁名称被添加到黑名单
        验证点：确保禁用机制正常工作
        """
        patcher = Patcher()
        patcher.add(OptimizerWrapper)
        patcher.disable(OptimizerWrapper.name)
        self.assertIn(OptimizerWrapper.name, patcher._blacklist)

    def test_with_profiling_configuration(self):
        """
        测试目的：验证性能分析配置功能
        测试步骤：
            1. 创建Patcher实例
            2. 调用with_profiling配置性能分析
            3. 验证_profiling_options不为None
            4. 验证配置的路径和级别正确
        验证点：确保性能分析配置正确存储
        为什么要测：性能分析是训练调优的重要功能，
                   需要确保配置能正确设置
        """
        patcher = Patcher()
        patcher.with_profiling("/tmp/prof", level=1, skip_first=50)
        self.assertIsNotNone(patcher._profiling_options)
        self.assertEqual(patcher._profiling_options['profiling_path'], "/tmp/prof")
        self.assertEqual(patcher._profiling_options['profiling_level'], 1)

    def test_brake_at_configuration(self):
        """
        测试目的：验证训练断点配置功能
        测试步骤：
            1. 创建Patcher实例
            2. 调用brake_at配置断点步数
            3. 验证_brake_step设置正确
        验证点：确保断点配置正确存储
        为什么要测：断点功能允许在指定步数停止训练，
                   用于调试和分析，需要确保配置正确
        """
        patcher = Patcher()
        patcher.brake_at(100)
        self.assertEqual(patcher._brake_step, 100)

    def test_add_training_loop_patches_imports(self):
        """
        测试目的：验证_add_training_loop_patches中的导入正确性
        测试步骤：
            1. 验证build_mmcv_epoch_runner_patch从mmcv_patch导入
            2. 验证build_mmcv_iter_runner_patch从mmcv_patch导入
            3. 验证build_mmengine_epoch_train_loop_patch从mmengine_patch导入
            4. 验证build_mmengine_iter_train_loop_patch从mmengine_patch导入
        验证点：确保导入路径正确，避免ImportError
        为什么要测：之前存在导入错误，build_mmcv_*函数错误地从mmengine_patch导入
        """
        # 验证mmcv_patch中的函数（使用已加载的模块）
        self.assertTrue(callable(build_mmcv_epoch_runner_patch))
        self.assertTrue(callable(build_mmcv_iter_runner_patch))
        self.assertTrue(hasattr(_mmcv_patch_module, 'build_mmcv_epoch_runner_patch'))
        self.assertTrue(hasattr(_mmcv_patch_module, 'build_mmcv_iter_runner_patch'))

        # 验证mmengine_patch中的函数（使用已加载的模块）
        self.assertTrue(callable(build_mmengine_epoch_train_loop_patch))
        self.assertTrue(callable(build_mmengine_iter_train_loop_patch))
        self.assertTrue(hasattr(_mmengine_patch_module, 'build_mmengine_epoch_train_loop_patch'))
        self.assertTrue(hasattr(_mmengine_patch_module, 'build_mmengine_iter_train_loop_patch'))

        # 验证这些函数不在错误的模块中
        self.assertFalse(hasattr(_mmengine_patch_module, 'build_mmcv_epoch_runner_patch'))
        self.assertFalse(hasattr(_mmengine_patch_module, 'build_mmcv_iter_runner_patch'))


if __name__ == "__main__":
    unittest.main()
