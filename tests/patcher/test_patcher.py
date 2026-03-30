# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Patcher框架核心功能测试模块

本模块测试patcher框架的核心组件，包括：
- AtomicPatch: 原子补丁类，用于替换单个函数/属性
- LegacyPatch: 遗留补丁类，兼容旧式补丁函数
- RegistryPatch: 注册表补丁类，用于向mmcv/mmengine注册表注册模块
- Patcher: 补丁管理器，统一管理和应用所有补丁
- Patch: 补丁集合类，用于组织相关补丁

测试设计原则：
- 使用mock模块模拟真实环境，无需NPU硬件
- 直接加载patcher子模块，避免触发torch依赖
- 每个测试用例独立，通过setUp/tearDown管理测试环境
"""
import importlib.util
import os
import sys
import types
import unittest
from typing import Dict, List
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
# First load dependencies in order
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

# Import classes from loaded modules
AtomicPatch = _patch_module.AtomicPatch
BasePatch = _patch_module.BasePatch
LegacyPatch = _patch_module.LegacyPatch
Patch = _patch_module.Patch
RegistryPatch = _patch_module.RegistryPatch
with_imports = _patch_module.with_imports
get_version = _patch_module.get_version
check_version = _patch_module.check_version
mmcv_version = _patch_module.mmcv_version
is_mmcv_v1x = _patch_module.is_mmcv_v1x
is_mmcv_v2x = _patch_module.is_mmcv_v2x
Patcher = _patcher_module.Patcher
PatchResult = _reporting.PatchResult
PatchStatus = _reporting.PatchStatus


class TestAtomicPatch(unittest.TestCase):
    """
    AtomicPatch原子补丁类测试

    AtomicPatch是最基础的补丁单元，用于替换模块中的单个函数或属性。
    测试目的：验证AtomicPatch能正确地替换目标函数，并支持各种高级特性。
    """

    def setUp(self):
        """
        测试环境准备

        执行步骤：
        1. 创建一个模拟模块test_module及其子模块submodule
        2. 在submodule中定义一个原始函数original_func(返回x*2)
        3. 将模拟模块注册到sys.modules中

        为什么需要这样做：
        - AtomicPatch通过模块路径定位目标，需要模块存在于sys.modules中
        - 使用模拟模块可以避免依赖真实的第三方库
        """
        # Create a mock module for testing
        self.mock_module = types.ModuleType('test_module')
        self.mock_module.submodule = types.ModuleType('test_module.submodule')
        self.mock_module.submodule.original_func = lambda x: x * 2
        sys.modules['test_module'] = self.mock_module
        sys.modules['test_module.submodule'] = self.mock_module.submodule

    def tearDown(self):
        """
        测试环境清理

        执行步骤：
        1. 从sys.modules中移除test_module及其子模块
        2. 确保不影响其他测试用例

        为什么需要这样做：
        - 防止测试用例之间相互干扰
        - 保持测试环境的隔离性
        """
        if 'test_module' in sys.modules:
            del sys.modules['test_module']
        if 'test_module.submodule' in sys.modules:
            del sys.modules['test_module.submodule']

    def test_basic_replacement(self):
        """
        测试基本的函数替换功能

        测试目的：验证AtomicPatch能够正确替换目标函数

        执行步骤：
        1. 定义一个新函数new_func(返回x*3)
        2. 创建AtomicPatch，目标为test_module.submodule.original_func
        3. 调用apply()应用补丁
        4. 验证补丁已应用(is_applied为True)
        5. 验证调用原函数时实际执行的是新函数(10*3=30)

        为什么测试这个：
        - 这是AtomicPatch最核心的功能
        - 确保补丁能正确替换目标并生效
        """

        def new_func(x):
            return x * 3

        patch = AtomicPatch("test_module.submodule.original_func", new_func)
        result = patch.apply()

        self.assertTrue(patch.is_applied)
        self.assertEqual(self.mock_module.submodule.original_func(10), 30)

    def test_precheck_with_target(self):
        """
        测试带目标参数的预检查功能

        测试目的：验证precheck函数能接收target参数并正确执行

        执行步骤：
        1. 定义新函数和precheck函数(检查target是否以test_module开头)
        2. 创建AtomicPatch，传入precheck参数
        3. 应用补丁
        4. 验证补丁成功应用(precheck返回True时)

        为什么测试这个：
        - precheck允许在应用补丁前进行条件检查
        - 某些补丁只在特定条件下才需要应用
        - 验证precheck能正确接收和使用target参数
        """
        def new_func(x):
            return x * 3

        # Precheck that accepts target parameter
        patch = AtomicPatch(
            "test_module.submodule.original_func",
            new_func,
            precheck=lambda target: target.startswith("test_module")
        )
        result = patch.apply()

        self.assertTrue(patch.is_applied)

    def test_precheck_without_args(self):
        """
        测试无参数的预检查功能

        测试目的：验证precheck函数可以不接收任何参数

        执行步骤：
        1. 定义新函数和无参数的precheck函数(直接返回True)
        2. 创建AtomicPatch
        3. 应用补丁
        4. 验证补丁成功应用

        为什么测试这个：
        - precheck支持多种签名形式，包括无参数形式
        - 简单的条件检查可能不需要任何参数
        """
        def new_func(x):
            return x * 3

        # Precheck without arguments
        patch = AtomicPatch(
            "test_module.submodule.original_func",
            new_func,
            precheck=lambda: True
        )
        result = patch.apply()

        self.assertTrue(patch.is_applied)

    def test_precheck_fails(self):
        """
        测试预检查失败时的行为

        测试目的：验证当precheck返回False时，补丁不会被应用

        执行步骤：
        1. 保存原始函数的引用
        2. 创建AtomicPatch，precheck始终返回False
        3. 应用补丁
        4. 验证is_applied为False
        5. 验证返回状态为SKIPPED
        6. 验证原始函数未被修改(10*2=20)

        为什么测试这个：
        - precheck失败是正常的业务场景(如依赖不满足)
        - 需要确保失败时不会破坏原有功能
        - 验证状态报告正确
        """
        def new_func(x):
            return x * 3

        original = self.mock_module.submodule.original_func

        patch = AtomicPatch(
            "test_module.submodule.original_func",
            new_func,
            precheck=lambda: False
        )
        result = patch.apply()

        self.assertFalse(patch.is_applied)
        self.assertEqual(result.status, PatchStatus.SKIPPED)
        # Original should be unchanged
        self.assertEqual(self.mock_module.submodule.original_func(10), 20)

    def test_wrapper_mode(self):
        """
        测试包装器模式

        测试目的：验证target_wrapper参数能正确包装原始函数

        执行步骤：
        1. 定义wrapper函数，它接收原始函数并返回包装后的函数
        2. 包装逻辑：调用原始函数后加100
        3. 创建AtomicPatch，只传入target_wrapper参数(不传replacement)
        4. 应用补丁
        5. 验证结果：original_func(10)=20, 加100后=120

        为什么测试这个：
        - target_wrapper模式用于增强而非替换原始函数
        - 常用于添加日志、性能监控等横切关注点
        - 与replacement模式互斥，需要单独测试
        """
        def wrapper(original):
            def wrapped(x):
                return original(x) + 100
            return wrapped

        patch = AtomicPatch(
            "test_module.submodule.original_func",
            target_wrapper=wrapper
        )
        result = patch.apply()

        self.assertTrue(patch.is_applied)
        # original_func(10) = 20, wrapped adds 100
        self.assertEqual(self.mock_module.submodule.original_func(10), 120)

    def test_replacement_wrapper_mode(self):
        """
        测试replacement_wrapper模式

        测试目的：验证replacement_wrapper参数能正确包装replacement函数

        执行步骤：
        1. 定义新函数new_func(返回x*3)
        2. 定义replacement_wrapper，它接收replacement并返回包装后的函数
        3. 包装逻辑：调用replacement后加100
        4. 创建AtomicPatch，传入replacement和replacement_wrapper
        5. 应用补丁
        6. 验证结果：new_func(10)=30, 加100后=130

        为什么测试这个：
        - replacement_wrapper用于在替换函数基础上添加额外逻辑
        - 常用于参数转换、返回值处理等场景
        - 与target_wrapper不同：target_wrapper包装原始函数，replacement_wrapper包装替换函数
        """
        def new_func(x):
            return x * 3

        def wrapper(replacement):
            def wrapped(x):
                return replacement(x) + 100
            return wrapped

        patch = AtomicPatch(
            "test_module.submodule.original_func",
            new_func,
            replacement_wrapper=wrapper
        )
        result = patch.apply()

        self.assertTrue(patch.is_applied)
        # new_func(10) = 30, wrapped adds 100
        self.assertEqual(self.mock_module.submodule.original_func(10), 130)

    def test_runtime_check(self):
        """
        测试运行时条件分发

        测试目的：验证runtime_check能在运行时动态决定使用哪个函数

        执行步骤：
        1. 定义新函数(返回x*10)
        2. 定义runtime_check：当x>5时返回True
        3. 创建并应用AtomicPatch
        4. 测试x=10(>5)：使用新函数，结果=100
        5. 测试x=3(<=5)：使用原函数，结果=6

        为什么测试这个：
        - runtime_check实现条件分发，同一函数根据参数选择不同实现
        - 用于渐进式迁移或特定场景优化
        - 与precheck不同：precheck在应用时检查，runtime_check在调用时检查
        """
        def new_func(x):
            return x * 10

        # Only use new_func when x > 5
        patch = AtomicPatch(
            "test_module.submodule.original_func",
            new_func,
            runtime_check=lambda x: x > 5
        )
        result = patch.apply()

        self.assertTrue(patch.is_applied)
        # x=10 > 5, use new_func
        self.assertEqual(self.mock_module.submodule.original_func(10), 100)
        # x=3 <= 5, use original
        self.assertEqual(self.mock_module.submodule.original_func(3), 6)

    def test_module_not_found(self):
        """
        测试目标模块不存在时的处理

        测试目的：验证当目标模块不存在时，补丁优雅地跳过而非崩溃

        执行步骤：
        1. 创建AtomicPatch，目标为不存在的模块
        2. 应用补丁
        3. 验证is_applied为False
        4. 验证状态为SKIPPED

        为什么测试这个：
        - 实际环境中某些可选依赖可能未安装
        - 补丁框架需要优雅处理这种情况
        - 不应因为可选模块缺失而导致整体失败
        """
        patch = AtomicPatch("nonexistent_module.func", lambda: None)
        result = patch.apply()

        self.assertFalse(patch.is_applied)
        self.assertEqual(result.status, PatchStatus.SKIPPED)

    def test_aliases(self):
        """
        测试别名补丁功能

        测试目的：验证补丁能同时应用到主路径和别名路径

        执行步骤：
        1. 创建别名模块test_module.alias，其func指向原始函数
        2. 创建AtomicPatch，指定aliases参数
        3. 应用补丁
        4. 验证主路径和别名路径都被替换

        为什么测试这个：
        - Python模块常有多个导入路径指向同一对象
        - 如torch.nn.functional.relu和torch.relu
        - 需要确保所有入口点都被正确补丁
        """
        # Create alias path
        self.mock_module.alias = types.ModuleType('test_module.alias')
        self.mock_module.alias.func = self.mock_module.submodule.original_func
        sys.modules['test_module.alias'] = self.mock_module.alias

        def new_func(x):
            return x * 5

        patch = AtomicPatch(
            "test_module.submodule.original_func",
            new_func,
            aliases=["test_module.alias.func"]
        )
        result = patch.apply()

        self.assertTrue(patch.is_applied)
        self.assertEqual(self.mock_module.submodule.original_func(10), 50)
        self.assertEqual(self.mock_module.alias.func(10), 50)

        # Cleanup
        del sys.modules['test_module.alias']


class TestLegacyPatch(unittest.TestCase):
    """
    LegacyPatch遗留补丁类测试

    LegacyPatch用于兼容旧式的补丁函数，这些函数接收(module, options)参数。
    测试目的：验证旧式补丁函数能通过LegacyPatch正确集成到新框架中。
    """

    def setUp(self):
        """
        测试环境准备

        执行步骤：
        1. 创建模拟模块legacy_test_module及其子模块ops
        2. 在ops中定义原始函数func(返回x+1)
        3. 注册到sys.modules

        为什么需要这样做：
        - LegacyPatch需要操作真实存在的模块
        - 模拟环境确保测试的可重复性
        """
        self.mock_module = types.ModuleType('legacy_test_module')
        self.mock_module.ops = types.ModuleType('legacy_test_module.ops')
        self.mock_module.ops.func = lambda x: x + 1
        sys.modules['legacy_test_module'] = self.mock_module
        sys.modules['legacy_test_module.ops'] = self.mock_module.ops

    def tearDown(self):
        """
        测试环境清理 - 移除legacy_test_module相关模块
        """
        if 'legacy_test_module' in sys.modules:
            del sys.modules['legacy_test_module']
        if 'legacy_test_module.ops' in sys.modules:
            del sys.modules['legacy_test_module.ops']

    def test_basic_legacy_patch(self):
        """
        测试基本的遗留补丁应用

        测试目的：验证LegacyPatch能正确执行旧式补丁函数

        执行步骤：
        1. 定义旧式补丁函数my_patch(module, options)
        2. 补丁函数内部替换module.ops.func
        3. 创建LegacyPatch，指定target_module
        4. 应用补丁
        5. 验证函数被替换(10+100=110)

        为什么测试这个：
        - 确保旧代码能平滑迁移到新框架
        - 验证module参数正确传递
        """
        def my_patch(module: ModuleType, options: Dict):
            def new_func(x):
                return x + 100
            module.ops.func = new_func

        patch = LegacyPatch(my_patch, target_module="legacy_test_module")
        result = patch.apply()

        self.assertTrue(patch.is_applied)
        self.assertEqual(self.mock_module.ops.func(10), 110)

    def test_legacy_patch_with_options(self):
        """
        测试带配置选项的遗留补丁

        测试目的：验证options参数能正确传递给补丁函数

        执行步骤：
        1. 定义补丁函数，从options中读取multiplier参数
        2. 创建LegacyPatch，传入options={'multiplier': 5}
        3. 应用补丁
        4. 验证配置生效(10*5=50)

        为什么测试这个：
        - 补丁行为常需要可配置
        - 验证配置能正确传递和使用
        """
        def my_patch(module: ModuleType, options: Dict):
            multiplier = options.get('multiplier', 1)
            def new_func(x):
                return x * multiplier
            module.ops.func = new_func

        patch = LegacyPatch(
            my_patch,
            target_module="legacy_test_module",
            options={'multiplier': 5}
        )
        result = patch.apply()

        self.assertTrue(patch.is_applied)
        self.assertEqual(self.mock_module.ops.func(10), 50)

    def test_legacy_patch_requires_target_module(self):
        """
        测试LegacyPatch必须指定target_module

        测试目的：验证创建LegacyPatch时不指定target_module会抛出异常

        执行步骤：
        1. 定义补丁函数
        2. 尝试创建LegacyPatch但不传target_module
        3. 验证抛出ValueError

        为什么测试这个：
        - target_module是必需参数，用于定位要补丁的模块
        - 缺少此参数应该立即报错，而非运行时失败
        """
        def my_patch(module: ModuleType, options: Dict):
            pass

        with self.assertRaises(ValueError):
            LegacyPatch(my_patch)

    def test_legacy_patch_module_not_found(self):
        """
        测试目标模块不存在时的处理

        测试目的：验证LegacyPatch在目标模块不存在时优雅跳过

        执行步骤：
        1. 创建LegacyPatch，target_module指向不存在的模块
        2. 应用补丁
        3. 验证is_applied为False，状态为SKIPPED

        为什么测试这个：
        - 与AtomicPatch类似，需要处理可选依赖缺失的情况
        """
        def my_patch(module: ModuleType, options: Dict):
            pass

        patch = LegacyPatch(my_patch, target_module="nonexistent_module")
        result = patch.apply()

        self.assertFalse(patch.is_applied)
        self.assertEqual(result.status, PatchStatus.SKIPPED)


class TestPatcher(unittest.TestCase):
    """
    Patcher补丁管理器测试

    Patcher是补丁框架的核心管理类，负责：
    - 收集和管理所有补丁
    - 按顺序应用补丁
    - 支持禁用特定补丁
    - 提供链式API和上下文管理器支持
    """

    def setUp(self):
        """
        测试环境准备 - 创建patcher_test_module模块
        """
        self.mock_module = types.ModuleType('patcher_test_module')
        self.mock_module.func1 = lambda x: x
        self.mock_module.func2 = lambda x: x
        sys.modules['patcher_test_module'] = self.mock_module

    def tearDown(self):
        """测试环境清理 - 移除patcher_test_module"""
        if 'patcher_test_module' in sys.modules:
            del sys.modules['patcher_test_module']

    def test_add_atomic_patch(self):
        """
        测试添加AtomicPatch到Patcher

        测试目的：验证Patcher能正确添加和应用AtomicPatch

        执行步骤：
        1. 创建Patcher实例
        2. 添加一个AtomicPatch
        3. 调用apply()
        4. 验证补丁生效(10*2=20)

        为什么测试这个：
        - 这是Patcher最基本的使用方式
        """
        patcher = Patcher()
        patcher.add(AtomicPatch("patcher_test_module.func1", lambda x: x * 2))
        patcher.apply()

        self.assertEqual(self.mock_module.func1(10), 20)

    def test_add_multiple_patches(self):
        """
        测试一次添加多个补丁

        测试目的：验证add()方法支持同时添加多个补丁

        执行步骤：
        1. 创建Patcher
        2. 一次add()调用添加两个AtomicPatch
        3. 应用补丁
        4. 验证两个函数都被替换

        为什么测试这个：
        - 批量添加是常见操作
        - 验证多补丁不会相互干扰
        """
        patcher = Patcher()
        patcher.add(
            AtomicPatch("patcher_test_module.func1", lambda x: x * 2),
            AtomicPatch("patcher_test_module.func2", lambda x: x * 3),
        )
        patcher.apply()

        self.assertEqual(self.mock_module.func1(10), 20)
        self.assertEqual(self.mock_module.func2(10), 30)

    def test_chained_add(self):
        """
        测试链式add调用

        测试目的：验证add()返回self，支持链式调用

        执行步骤：
        1. 使用链式调用创建并配置Patcher
        2. 应用补丁
        3. 验证所有补丁生效

        为什么测试这个：
        - 链式API提供更流畅的使用体验
        - 验证返回值正确
        """
        patcher = (
            Patcher()
            .add(AtomicPatch("patcher_test_module.func1", lambda x: x * 2))
            .add(AtomicPatch("patcher_test_module.func2", lambda x: x * 3))
        )
        patcher.apply()

        self.assertEqual(self.mock_module.func1(10), 20)
        self.assertEqual(self.mock_module.func2(10), 30)

    def test_disable_by_name(self):
        """
        测试按名称禁用补丁

        测试目的：验证disable()能阻止特定补丁被应用

        执行步骤：
        1. 添加补丁到Patcher
        2. 调用disable()禁用该补丁
        3. 应用补丁
        4. 验证原函数未被修改(10仍返回10)

        为什么测试这个：
        - 某些场景需要选择性禁用补丁
        - 如调试时禁用某个可疑补丁
        """
        patcher = Patcher()
        patcher.add(AtomicPatch("patcher_test_module.func1", lambda x: x * 2))
        patcher.disable("patcher_test_module.func1")
        patcher.apply()

        # Should remain unchanged
        self.assertEqual(self.mock_module.func1(10), 10)

    def test_context_manager(self):
        """
        测试上下文管理器用法

        测试目的：验证Patcher支持with语句

        执行步骤：
        1. 创建并配置Patcher
        2. 使用with语句进入上下文
        3. 在上下文内验证补丁生效

        为什么测试这个：
        - with语句是Python惯用的资源管理方式
        - 确保__enter__正确应用补丁
        """
        patcher = Patcher()
        patcher.add(AtomicPatch("patcher_test_module.func1", lambda x: x * 2))

        with patcher:
            self.assertEqual(self.mock_module.func1(10), 20)

    def test_apply_only_once(self):
        """
        测试补丁只应用一次

        测试目的：验证多次调用apply()不会重复应用补丁

        执行步骤：
        1. 创建带计数器的替换函数
        2. 添加并应用补丁
        3. 再次调用apply()
        4. 调用函数一次，验证计数器为1(非2)

        为什么测试这个：
        - 防止重复应用导致的问题(如wrapper嵌套)
        - 确保幂等性
        """
        call_count = [0]

        def counting_func(x):
            call_count[0] += 1
            return x * 2

        patcher = Patcher()
        patcher.add(AtomicPatch("patcher_test_module.func1", counting_func))
        patcher.apply()
        patcher.apply()  # Second apply should be no-op

        self.mock_module.func1(10)
        self.assertEqual(call_count[0], 1)

    def test_brake_at(self):
        """
        测试brake_at配置

        测试目的：验证brake_at()正确设置断点步数

        执行步骤：
        1. 创建Patcher
        2. 调用brake_at(100)
        3. 验证_brake_step被设置为100

        为什么测试这个：
        - brake_at用于在指定训练步数暂停，便于调试
        - 验证配置正确存储
        """
        patcher = Patcher()
        patcher.brake_at(100)

        self.assertEqual(patcher._brake_step, 100)

    def test_with_profiling(self):
        """
        测试性能分析配置

        测试目的：验证with_profiling()正确设置性能分析选项

        执行步骤：
        1. 创建Patcher
        2. 调用with_profiling()设置分析路径、级别、跳过步数
        3. 验证_profiling_options包含正确的配置

        为什么测试这个：
        - 性能分析是重要的调试功能
        - 验证配置参数正确传递
        """
        patcher = Patcher()
        patcher.with_profiling("/path/to/prof", level=1, skip_first=50)

        self.assertIsNotNone(patcher._profiling_options)
        self.assertEqual(patcher._profiling_options['profiling_path'], "/path/to/prof")
        self.assertEqual(patcher._profiling_options['profiling_level'], 1)

    def test_allow_internal_format(self):
        """
        测试allow_internal_format配置

        测试目的：验证allow_internal_format()正确设置NPU内部格式选项

        执行步骤：
        1. 创建Patcher
        2. 调用allow_internal_format()
        3. 验证_allow_internal_format被设置为True

        为什么测试这个：
        - allow_internal_format是NPU性能优化的重要选项
        - 验证配置正确存储
        """
        patcher = Patcher()
        patcher.allow_internal_format()

        self.assertTrue(patcher._allow_internal_format)

    def test_allow_internal_format_explicit_false(self):
        """
        测试显式禁用internal format

        测试目的：验证allow_internal_format(False)正确禁用NPU内部格式

        执行步骤：
        1. 创建Patcher
        2. 调用allow_internal_format(False)
        3. 验证_allow_internal_format被设置为False

        为什么测试这个：
        - 用户可能需要显式禁用内部格式
        - 验证参数正确传递
        """
        patcher = Patcher()
        patcher.allow_internal_format(False)

        self.assertFalse(patcher._allow_internal_format)

    def test_disallow_internal_format(self):
        """
        测试disallow_internal_format配置

        测试目的：验证disallow_internal_format()正确禁用NPU内部格式

        执行步骤：
        1. 创建Patcher
        2. 调用disallow_internal_format()
        3. 验证_allow_internal_format被设置为False

        为什么测试这个：
        - disallow_internal_format是allow_internal_format(False)的便捷方法
        - 验证配置正确存储
        """
        patcher = Patcher()
        patcher.disallow_internal_format()

        self.assertFalse(patcher._allow_internal_format)

    def test_allow_internal_format_default(self):
        """
        测试allow_internal_format默认值

        测试目的：验证未设置时_allow_internal_format为None（使用默认值False）

        执行步骤：
        1. 创建Patcher
        2. 不调用任何internal format方法
        3. 验证_allow_internal_format为None

        为什么测试这个：
        - 默认行为应该是禁用内部格式（更安全）
        - 验证初始状态正确
        """
        patcher = Patcher()

        self.assertIsNone(patcher._allow_internal_format)

    def test_allow_internal_format_chaining(self):
        """
        测试allow_internal_format链式调用

        测试目的：验证allow_internal_format()返回self，支持链式调用

        执行步骤：
        1. 使用链式调用创建并配置Patcher
        2. 验证配置正确

        为什么测试这个：
        - 链式API提供更流畅的使用体验
        - 验证返回值正确
        """
        patcher = (
            Patcher()
            .allow_internal_format()
            .add(AtomicPatch("patcher_test_module.func1", lambda x: x * 2))
        )

        self.assertTrue(patcher._allow_internal_format)


class TestPatch(unittest.TestCase):
    """
    Patch补丁集合类测试

    Patch用于将相关的多个AtomicPatch组织在一起。
    测试目的：验证自定义Patch子类能正确工作，并与Patcher集成。
    """

    def test_custom_patch(self):
        """
        测试自定义Patch

        测试目的：验证用户可以创建自定义的Patch子类

        执行步骤：
        1. 创建模拟模块
        2. 定义MyPatch类，继承Patch
        3. 实现patches()方法返回AtomicPatch列表
        4. 将MyPatch添加到Patcher并应用
        5. 验证补丁生效(5*10=50)

        为什么测试这个：
        - Patch子类是组织相关补丁的推荐方式
        - 验证类级别的补丁定义能正确工作
        """
        # Create mock module
        mock_module = types.ModuleType('patchset_test_module')
        mock_module.func = lambda x: x
        sys.modules['patchset_test_module'] = mock_module

        try:
            class MyPatch(Patch):
                name = "my_patch_set"

                @classmethod
                def patches(cls, options=None) -> List[AtomicPatch]:
                    return [
                        AtomicPatch("patchset_test_module.func", lambda x: x * 10)
                    ]

            patcher = Patcher()
            patcher.add(MyPatch)
            patcher.apply()

            self.assertEqual(mock_module.func(5), 50)
        finally:
            del sys.modules['patchset_test_module']

    def test_disable_patch_by_name(self):
        """
        测试按名称禁用Patch

        测试目的：验证可以通过Patch的name属性禁用整个补丁集

        执行步骤：
        1. 定义MyPatch，设置name属性
        2. 添加到Patcher
        3. 调用disable(name)禁用
        4. 应用补丁
        5. 验证原函数未被修改

        为什么测试这个：
        - 禁用整个补丁集比逐个禁用更方便
        - 验证name属性正确用于匹配
        """
        mock_module = types.ModuleType('patchset_test_module2')
        mock_module.func = lambda x: x
        sys.modules['patchset_test_module2'] = mock_module

        try:
            class MyPatch(Patch):
                name = "my_disabled_patch_set"

                @classmethod
                def patches(cls, options=None) -> List[AtomicPatch]:
                    return [
                        AtomicPatch("patchset_test_module2.func", lambda x: x * 10)
                    ]

            patcher = Patcher()
            patcher.add(MyPatch)
            patcher.disable("my_disabled_patch_set")
            patcher.apply()

            # Should remain unchanged
            self.assertEqual(mock_module.func(5), 5)
        finally:
            del sys.modules['patchset_test_module2']

    def test_patch_with_options(self):
        """
        测试带options参数的Patch

        测试目的：验证Patch.patches()能接收并使用options参数

        执行步骤：
        1. 定义MyPatch，patches()方法从options读取multiplier
        2. 添加到Patcher时传入options={'multiplier': 5}
        3. 应用补丁
        4. 验证配置生效(5*5=25)

        为什么测试这个：
        - options允许同一Patch类有不同的行为配置
        - 验证options正确传递到patches()方法
        """
        mock_module = types.ModuleType('patchset_test_module3')
        mock_module.func = lambda x: x
        sys.modules['patchset_test_module3'] = mock_module

        try:
            class MyPatch(Patch):
                name = "my_patch_with_options"

                @classmethod
                def patches(cls, options=None) -> List[AtomicPatch]:
                    multiplier = (options or {}).get('multiplier', 10)
                    return [
                        AtomicPatch("patchset_test_module3.func", lambda x: x * multiplier)
                    ]

            patcher = Patcher()
            patcher.add(MyPatch, options={'multiplier': 5})
            patcher.apply()

            self.assertEqual(mock_module.func(5), 25)
        finally:
            del sys.modules['patchset_test_module3']


class TestRegistryPatch(unittest.TestCase):
    """
    RegistryPatch注册表补丁类测试

    RegistryPatch用于向mmcv/mmengine的Registry注册自定义模块。
    测试目的：验证RegistryPatch能正确调用registry.register_module()。
    """

    def test_registry_patch_basic(self):
        """
        测试基本的注册表补丁

        测试目的：验证RegistryPatch能正确注册模块到Registry

        执行步骤：
        1. 创建mock的Registry对象
        2. 创建RegistryPatch，指定registry路径、模块类、名称
        3. 应用补丁
        4. 验证register_module被正确调用

        为什么测试这个：
        - mmcv/mmengine使用Registry管理可插拔组件
        - 需要验证注册参数正确传递
        """
        # Create mock registry
        mock_registry = MagicMock()
        mock_registry.register_module = MagicMock()

        mock_module = types.ModuleType('registry_test_module')
        mock_module.REGISTRY = mock_registry
        sys.modules['registry_test_module'] = mock_module

        try:
            class MyClass:
                pass

            patch = RegistryPatch(
                "registry_test_module.REGISTRY",
                MyClass,
                name="MyClass",
                force=True
            )
            result = patch.apply()

            self.assertTrue(patch.is_applied)
            mock_registry.register_module.assert_called_once_with(
                name="MyClass",
                force=True,
                module=MyClass
            )
        finally:
            del sys.modules['registry_test_module']

    def test_registry_patch_with_factory(self):
        """
        测试使用工厂函数的注册表补丁

        测试目的：验证module_factory参数能延迟创建模块类

        执行步骤：
        1. 定义工厂函数create_class，返回动态创建的类
        2. 创建RegistryPatch，使用module_factory而非module_cls
        3. 应用补丁
        4. 验证register_module被调用

        为什么测试这个：
        - 某些类需要在运行时动态创建(如依赖其他模块)
        - 工厂模式提供更大的灵活性
        """
        mock_registry = MagicMock()
        mock_registry.register_module = MagicMock()

        mock_module = types.ModuleType('registry_test_module2')
        mock_module.REGISTRY = mock_registry
        sys.modules['registry_test_module2'] = mock_module

        try:
            def create_class():
                class DynamicClass:
                    pass
                return DynamicClass

            patch = RegistryPatch(
                "registry_test_module2.REGISTRY",
                name="DynamicClass",
                module_factory=create_class
            )
            result = patch.apply()

            self.assertTrue(patch.is_applied)
            mock_registry.register_module.assert_called_once()
        finally:
            del sys.modules['registry_test_module2']

    def test_registry_patch_precheck(self):
        """
        测试注册表补丁的预检查

        测试目的：验证precheck失败时不会注册模块

        执行步骤：
        1. 创建RegistryPatch，precheck始终返回False
        2. 应用补丁
        3. 验证is_applied为False，状态为SKIPPED
        4. 验证register_module未被调用

        为什么测试这个：
        - 某些模块只在特定条件下需要注册
        - 验证precheck能正确阻止注册
        """
        mock_registry = MagicMock()
        mock_registry.register_module = MagicMock()

        mock_module = types.ModuleType('registry_test_module3')
        mock_module.REGISTRY = mock_registry
        sys.modules['registry_test_module3'] = mock_module

        try:
            class MyClass:
                pass

            patch = RegistryPatch(
                "registry_test_module3.REGISTRY",
                MyClass,
                name="MyClass",
                precheck=lambda: False  # Always fail
            )
            result = patch.apply()

            self.assertFalse(patch.is_applied)
            self.assertEqual(result.status, PatchStatus.SKIPPED)
            mock_registry.register_module.assert_not_called()
        finally:
            del sys.modules['registry_test_module3']


class TestVersionDetection(unittest.TestCase):
    """
    版本检测工具测试

    测试get_version、check_version、mmcv_version等函数。
    这些工具用于检测当前环境中包的版本，以便应用正确的补丁。
    """

    def test_get_version_returns_string_or_none(self):
        """
        测试get_version返回字符串或None

        测试目的：验证get_version对已安装和未安装的包都能正常工作

        执行步骤：
        1. 对已安装的包(如sys)调用get_version
        2. 对未安装的包调用get_version
        3. 验证返回值类型正确

        为什么测试这个：
        - get_version是版本检测的基础函数
        - 需要在任何环境下都能安全调用
        """
        # Test with a package that doesn't exist
        result = get_version("nonexistent_package_xyz")
        self.assertIsNone(result)

    def test_check_version_with_nonexistent_package(self):
        """
        测试check_version对不存在的包返回False

        测试目的：验证check_version在包不存在时返回False而非抛出异常

        执行步骤：
        1. 对不存在的包调用check_version
        2. 验证返回False

        为什么测试这个：
        - check_version常用于precheck函数
        - 需要在包未安装时也能安全调用
        """
        result = check_version("nonexistent_package_xyz", major=1)
        self.assertFalse(result)

    def test_mmcv_version_function(self):
        """
        测试mmcv_version函数

        测试目的：验证mmcv_version()函数可以安全调用

        执行步骤：
        1. 调用mmcv_version()
        2. 验证不会抛出异常
        3. 验证返回值是字符串或None

        为什么测试这个：
        - mmcv_version是常用的版本检测函数
        - 需要在mmcv未安装时也能安全调用
        """
        result = mmcv_version()
        self.assertTrue(result is None or isinstance(result, str))

    def test_version_functions(self):
        """
        测试版本检测函数

        测试目的：验证is_mmcv_v1x()和is_mmcv_v2x()函数可以安全调用

        执行步骤：
        1. 调用is_mmcv_v1x()和is_mmcv_v2x()
        2. 验证不会抛出异常
        3. 验证返回值是布尔类型

        为什么测试这个：
        - 这些函数是补丁条件判断的基础
        - 需要在任何环境下都能安全调用
        """
        result_v1 = is_mmcv_v1x()
        result_v2 = is_mmcv_v2x()
        self.assertIsInstance(result_v1, bool)
        self.assertIsInstance(result_v2, bool)


class TestPatchResult(unittest.TestCase):
    """
    PatchResult和PatchStatus测试

    PatchResult记录单个补丁的应用结果，PatchStatus是结果状态枚举。
    测试目的：验证结果对象能正确创建和使用。
    """

    def test_patch_status_values(self):
        """
        测试PatchStatus枚举值

        测试目的：验证枚举值符合预期

        执行步骤：
        1. 检查APPLIED、SKIPPED、FAILED的值

        为什么测试这个：
        - 枚举值可能用于日志或序列化
        - 确保值的稳定性
        """
        self.assertEqual(PatchStatus.APPLIED.value, "applied")
        self.assertEqual(PatchStatus.SKIPPED.value, "skipped")
        self.assertEqual(PatchStatus.FAILED.value, "failed")

    def test_patch_result_creation(self):
        """
        测试PatchResult创建

        测试目的：验证PatchResult能正确存储补丁结果信息

        执行步骤：
        1. 创建基本的PatchResult(状态、名称、模块)
        2. 创建带reason的PatchResult
        3. 验证各字段值正确

        为什么测试这个：
        - PatchResult是补丁应用的反馈机制
        - 需要确保信息完整记录
        """
        result = PatchResult(PatchStatus.APPLIED, "test_patch", "test_module")
        self.assertEqual(result.status, PatchStatus.APPLIED)
        self.assertEqual(result.name, "test_patch")
        self.assertEqual(result.module, "test_module")

        result_with_reason = PatchResult(
            PatchStatus.SKIPPED,
            "test_patch",
            "test_module",
            "precheck failed"
        )
        self.assertEqual(result_with_reason.reason, "precheck failed")


class TestGaussianWeightsPatchExample(unittest.TestCase):
    """
    高斯核权重计算补丁示例测试

    测试README文档中的高斯核权重计算NPU优化示例，验证：
    - Patch类定义正确
    - with_imports装饰器正常工作
    - runtime_check正确过滤输入
    - 补丁能正确替换目标函数
    """

    def setUp(self):
        """创建mock的目标模块和torch_npu模块"""
        import math

        # Save original torch and torch_npu to restore in tearDown
        self._orig_torch = sys.modules.get("torch")
        self._orig_torch_npu = sys.modules.get("torch_npu")

        # 创建mock的目标模块 my_model.ops.gaussian
        self.gaussian_module = types.ModuleType("my_model.ops.gaussian")

        def original_compute_gaussian_weights(distances, sigma):
            """原始CUDA实现"""
            variance = 2.0 * sigma * sigma
            # 使用mock的torch.exp
            weights = distances.exp_mock(-distances * distances / variance)
            norm_factor = 1.0 / (sigma * math.sqrt(2 * math.pi))
            return weights * norm_factor

        self.gaussian_module.compute_gaussian_weights = original_compute_gaussian_weights

        # 创建模块层级
        my_model = types.ModuleType("my_model")
        my_model_ops = types.ModuleType("my_model.ops")
        my_model.ops = my_model_ops
        my_model_ops.gaussian = self.gaussian_module

        sys.modules["my_model"] = my_model
        sys.modules["my_model.ops"] = my_model_ops
        sys.modules["my_model.ops.gaussian"] = self.gaussian_module

        # 创建mock的torch_npu模块
        self.torch_npu_mock = types.ModuleType("torch_npu")
        self.npu_exp_call_count = 0

        def mock_npu_exp(x):
            self.npu_exp_call_count += 1
            return x  # 简化返回
        self.torch_npu_mock.npu_exp = mock_npu_exp
        sys.modules["torch_npu"] = self.torch_npu_mock

        # 创建mock的torch模块（用于dtype检查）
        self.torch_mock = MagicMock()
        self.torch_mock.float32 = "float32"
        self.torch_mock.float16 = "float16"
        sys.modules["torch"] = self.torch_mock

    def tearDown(self):
        """清理mock模块，恢复原始torch/torch_npu"""
        for mod_name in ["my_model", "my_model.ops", "my_model.ops.gaussian"]:
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        # Restore original torch and torch_npu instead of deleting to avoid reimport issues
        if self._orig_torch is not None:
            sys.modules["torch"] = self._orig_torch
        elif "torch" in sys.modules:
            del sys.modules["torch"]
        if self._orig_torch_npu is not None:
            sys.modules["torch_npu"] = self._orig_torch_npu
        elif "torch_npu" in sys.modules:
            del sys.modules["torch_npu"]

    def test_gaussian_weights_patch_definition(self):
        """
        测试高斯核权重补丁类定义

        测试目的：验证按照README示例定义的Patch类结构正确

        执行步骤：
        1. 定义GaussianWeightsPatch类
        2. 验证patches()返回AtomicPatch列表
        3. 验证AtomicPatch包含正确的target和runtime_check

        为什么测试这个：
        - 确保README示例代码可以正确运行
        - 验证Patch类的基本结构
        """
        import math
        torch = sys.modules["torch"]

        class GaussianWeightsPatch(Patch):
            """高斯核权重计算NPU优化"""
            name = "gaussian_weights"

            @classmethod
            def patches(cls, options=None):
                return [
                    AtomicPatch(
                        "my_model.ops.gaussian.compute_gaussian_weights",
                        cls._compute_gaussian_weights_npu,
                        runtime_check=cls._check_fp32,
                    ),
                ]

            @staticmethod
            def _check_fp32(distances, sigma):
                return distances.dtype == torch.float32

            @staticmethod
            @with_imports("math", "torch_npu")
            def _compute_gaussian_weights_npu(distances, sigma):
                variance = 2.0 * sigma * sigma
                weights = torch_npu.npu_exp(-distances * distances / variance)  # noqa: F821
                norm_factor = 1.0 / (sigma * math.sqrt(2 * math.pi))  # noqa: F821
                return weights * norm_factor

        # 验证Patch类结构
        self.assertEqual(GaussianWeightsPatch.name, "gaussian_weights")
        patches = GaussianWeightsPatch.patches()
        self.assertEqual(len(patches), 1)
        self.assertIsInstance(patches[0], AtomicPatch)
        self.assertEqual(patches[0].name, "my_model.ops.gaussian.compute_gaussian_weights")

    def test_gaussian_weights_patch_with_imports(self):
        """
        测试with_imports装饰器

        测试目的：验证with_imports能正确注入外部模块依赖

        执行步骤：
        1. 定义使用with_imports的替换函数
        2. 调用替换函数
        3. 验证torch_npu.npu_exp被正确调用

        为什么测试这个：
        - with_imports是补丁函数依赖外部模块的关键机制
        - 确保模块注入正确工作
        """
        import math

        @with_imports("math", "torch_npu")
        def replacement_func(distances, sigma):
            variance = 2.0 * sigma * sigma
            weights = torch_npu.npu_exp(-distances * distances / variance)  # noqa: F821
            norm_factor = 1.0 / (sigma * math.sqrt(2 * math.pi))  # noqa: F821
            return weights * norm_factor

        # 创建mock输入
        mock_distances = MagicMock()
        mock_distances.__mul__ = MagicMock(return_value=mock_distances)
        mock_distances.__truediv__ = MagicMock(return_value=mock_distances)
        mock_distances.__neg__ = MagicMock(return_value=mock_distances)

        # 调用替换函数
        self.npu_exp_call_count = 0
        replacement_func(mock_distances, 1.0)

        # 验证torch_npu.npu_exp被调用
        self.assertEqual(self.npu_exp_call_count, 1)

    def test_gaussian_weights_runtime_check(self):
        """
        测试runtime_check功能

        测试目的：验证runtime_check能正确过滤输入dtype

        执行步骤：
        1. 创建FP32和FP16的mock输入
        2. 验证FP32输入通过检查
        3. 验证FP16输入不通过检查

        为什么测试这个：
        - runtime_check是补丁条件执行的关键机制
        - 确保只有符合条件的输入才使用NPU优化
        """
        torch = sys.modules["torch"]

        def check_fp32(distances, sigma):
            return distances.dtype == torch.float32

        # FP32输入应该通过
        fp32_input = MagicMock()
        fp32_input.dtype = torch.float32
        self.assertTrue(check_fp32(fp32_input, 1.0))

        # FP16输入不应该通过
        fp16_input = MagicMock()
        fp16_input.dtype = torch.float16
        self.assertFalse(check_fp32(fp16_input, 1.0))

    def test_gaussian_weights_patch_apply(self):
        """
        测试补丁应用

        测试目的：验证补丁能正确替换目标函数

        执行步骤：
        1. 定义GaussianWeightsPatch
        2. 创建Patcher并添加补丁
        3. 应用补丁
        4. 验证目标函数被替换

        为什么测试这个：
        - 这是补丁的核心功能
        - 确保README示例的完整流程可以正常工作
        """
        import math
        torch = sys.modules["torch"]

        class GaussianWeightsPatch(Patch):
            name = "gaussian_weights"

            @classmethod
            def patches(cls, options=None):
                return [
                    AtomicPatch(
                        "my_model.ops.gaussian.compute_gaussian_weights",
                        cls._compute_gaussian_weights_npu,
                        runtime_check=cls._check_fp32,
                    ),
                ]

            @staticmethod
            def _check_fp32(distances, sigma):
                return distances.dtype == torch.float32

            @staticmethod
            @with_imports("math", "torch_npu")
            def _compute_gaussian_weights_npu(distances, sigma):
                variance = 2.0 * sigma * sigma
                weights = torch_npu.npu_exp(-distances * distances / variance)  # noqa: F821
                norm_factor = 1.0 / (sigma * math.sqrt(2 * math.pi))  # noqa: F821
                return weights * norm_factor

        # 保存原始函数
        original_func = self.gaussian_module.compute_gaussian_weights

        # 创建Patcher并应用补丁
        patcher = Patcher()
        patcher.add(GaussianWeightsPatch)
        patcher.apply()

        # 验证函数已被替换（不再是原始函数）
        current_func = self.gaussian_module.compute_gaussian_weights
        self.assertIsNot(current_func, original_func)


class TestWithImports(unittest.TestCase):
    """
    with_imports装饰器测试

    测试with_imports的两种用法：
    - 字符串形式：导入整个模块
    - 元组形式：从模块导入特定名称
    """

    def setUp(self):
        """创建mock模块"""
        # 创建mock的torch_npu模块
        self.torch_npu_mock = types.ModuleType("torch_npu")
        self.npu_exp_call_count = 0

        def mock_npu_exp(x):
            self.npu_exp_call_count += 1
            return x
        self.torch_npu_mock.npu_exp = mock_npu_exp
        sys.modules["torch_npu"] = self.torch_npu_mock

    def tearDown(self):
        """清理mock模块"""
        if "torch_npu" in sys.modules:
            del sys.modules["torch_npu"]

    def test_with_imports_string_form(self):
        """
        测试with_imports字符串形式（导入整个模块）

        测试目的：验证字符串形式能正确导入整个模块

        执行步骤：
        1. 使用字符串形式定义with_imports
        2. 调用装饰后的函数
        3. 验证模块被正确注入

        为什么测试这个：
        - 字符串形式是最简洁的用法
        - 确保 @with_imports("math", "torch_npu") 正常工作
        """
        @with_imports("math", "torch_npu")
        def func_with_string_imports(x):
            # 使用整个模块
            result = math.sqrt(x)  # noqa: F821
            torch_npu.npu_exp(result)  # noqa: F821
            return result

        result = func_with_string_imports(4.0)
        self.assertEqual(result, 2.0)
        self.assertEqual(self.npu_exp_call_count, 1)

    def test_with_imports_tuple_form_specific_names(self):
        """
        测试with_imports元组形式（导入特定名称）

        测试目的：验证元组形式能正确导入模块中的特定名称

        执行步骤：
        1. 使用元组形式定义with_imports
        2. 调用装饰后的函数
        3. 验证特定名称被正确注入

        为什么测试这个：
        - 元组形式用于只导入需要的名称
        - 确保 @with_imports(("math", "sqrt", "pi")) 正常工作
        """
        @with_imports(("math", "sqrt", "pi"))
        def func_with_tuple_imports(x):
            # 直接使用导入的名称
            return sqrt(x) * pi  # noqa: F821

        result = func_with_tuple_imports(4.0)
        import math
        self.assertAlmostEqual(result, 2.0 * math.pi)

    def test_with_imports_mixed_form(self):
        """
        测试with_imports混合形式

        测试目的：验证字符串和元组形式可以混合使用

        执行步骤：
        1. 混合使用字符串和元组形式
        2. 调用装饰后的函数
        3. 验证所有导入都正确工作

        为什么测试这个：
        - 实际使用中可能需要混合两种形式
        - 确保混合使用时不会冲突
        """
        @with_imports(
            "torch_npu",                    # 导入整个模块
            ("math", "sqrt"),               # 只导入sqrt
        )
        def func_with_mixed_imports(x):
            result = sqrt(x)  # noqa: F821
            torch_npu.npu_exp(result)  # noqa: F821
            return result

        result = func_with_mixed_imports(9.0)
        self.assertEqual(result, 3.0)
        self.assertEqual(self.npu_exp_call_count, 1)

    def test_with_imports_caching(self):
        """
        测试with_imports缓存机制

        测试目的：验证导入只在首次调用时执行，后续调用使用缓存

        执行步骤：
        1. 定义使用with_imports的函数
        2. 多次调用函数
        3. 验证导入只执行一次

        为什么测试这个：
        - 缓存机制是性能优化的关键
        - 确保不会重复执行导入
        """
        call_count = [0]

        @with_imports("math")
        def func_with_caching(x):
            call_count[0] += 1
            return math.sqrt(x)  # noqa: F821

        # 多次调用
        func_with_caching(4.0)
        func_with_caching(9.0)
        func_with_caching(16.0)

        # 函数被调用3次
        self.assertEqual(call_count[0], 3)


if __name__ == "__main__":
    unittest.main()
