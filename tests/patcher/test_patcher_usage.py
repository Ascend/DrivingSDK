# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Patcher框架用法测试模块

本模块专注于测试patcher的各种实际使用场景，包括：
- diff生成的ground truth对比
- 各种补丁模式的实际效果验证
- mock target和mock replacement的完整测试
- 边界情况和错误处理

测试设计原则：
- 每个测试都有明确的期望值(ground truth)进行对比
- 使用mock模块模拟真实环境
- 覆盖常见用法和边界情况
"""
import importlib.util
import os
import sys
import types
import unittest
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
AtomicPatch = _patch_module.AtomicPatch
BasePatch = _patch_module.BasePatch
LegacyPatch = _patch_module.LegacyPatch
Patch = _patch_module.Patch
RegistryPatch = _patch_module.RegistryPatch
Patcher = _patcher_module.Patcher
PatchResult = _reporting.PatchResult
PatchStatus = _reporting.PatchStatus
_get_source_diff = _patch_module._get_source_diff
_get_callable_name = _patch_module._get_callable_name


# =============================================================================
# Mock Target Functions - 用于测试的目标函数
# =============================================================================

def mock_original_add(a, b):
    """原始加法函数"""
    return a + b


def mock_original_multiply(a, b):
    """原始乘法函数"""
    return a * b


def mock_original_process(data):
    """原始数据处理函数"""
    return [x * 2 for x in data]


class MockOriginalClass:
    """原始类，用于测试类方法补丁"""

    def __init__(self, value):
        self.value = value

    def compute(self, x):
        """原始计算方法"""
        return self.value + x

    @staticmethod
    def static_compute(x):
        """原始静态方法"""
        return x * 2

    @classmethod
    def class_compute(cls, x):
        """原始类方法"""
        return x * 3


# =============================================================================
# Mock Replacement Functions - 用于测试的替换函数
# =============================================================================

def mock_replacement_add(a, b):
    """替换加法函数 - 返回a + b + 100"""
    return a + b + 100


def mock_replacement_multiply(a, b):
    """替换乘法函数 - 返回a * b * 10"""
    return a * b * 10


def mock_replacement_process(data):
    """替换数据处理函数 - 返回每个元素的平方"""
    return [x ** 2 for x in data]


class MockReplacementClass:
    """替换类"""

    def __init__(self, value):
        self.value = value * 10

    def compute(self, x):
        """替换计算方法"""
        return self.value * x

    @staticmethod
    def static_compute(x):
        """替换静态方法"""
        return x ** 2

    @classmethod
    def class_compute(cls, x):
        """替换类方法"""
        return x ** 3


# =============================================================================
# Test: Diff Generation with Ground Truth
# =============================================================================

class TestDiffGenerationGroundTruth(unittest.TestCase):
    """
    Diff生成测试 - 使用ground truth进行对比

    测试目的：验证_get_source_diff生成的diff内容正确，
    包含预期的差异标记和代码变化。
    """

    def test_diff_contains_function_names(self):
        """
        测试diff包含函数名称

        Ground Truth:
        - diff应包含"original:"和"replacement:"标记
        - diff应包含原始函数名和替换函数名
        """
        diff = _get_source_diff(mock_original_add, mock_replacement_add)

        # Ground truth assertions
        self.assertIn("original:", diff)
        self.assertIn("replacement:", diff)
        self.assertIn("mock_original_add", diff)
        self.assertIn("mock_replacement_add", diff)

    def test_diff_contains_code_changes(self):
        """
        测试diff包含代码变化

        Ground Truth:
        - diff应包含"-"标记表示删除的行
        - diff应包含"+"标记表示添加的行
        - diff应包含实际的代码差异
        """
        diff = _get_source_diff(mock_original_add, mock_replacement_add)

        # Ground truth: diff should show the return statement change
        # original: return a + b
        # replacement: return a + b + 100
        self.assertIn("-", diff)  # 删除标记
        self.assertIn("+", diff)  # 添加标记
        self.assertIn("return", diff)  # 包含return语句

    def test_diff_unified_format(self):
        """
        测试diff使用unified格式

        Ground Truth:
        - diff应以"---"开头表示原始文件
        - diff应包含"+++"表示新文件
        - diff应包含"@@"表示变化位置
        """
        diff = _get_source_diff(mock_original_add, mock_replacement_add)

        # Ground truth: unified diff format markers
        self.assertIn("---", diff)
        self.assertIn("+++", diff)
        self.assertIn("@@", diff)

    def test_diff_empty_for_identical_functions(self):
        """
        测试相同函数的diff为空或只有头部

        Ground Truth:
        - 当两个函数完全相同时，diff应该没有实际的代码变化
        """
        # 使用同一个函数
        diff = _get_source_diff(mock_original_add, mock_original_add)

        # Ground truth: no actual code changes (no +/- lines except headers)
        lines = diff.split('\n')
        change_lines = [l for l in lines if l.startswith('+') or l.startswith('-')]
        # 只有头部的+++ 和 ---，没有实际代码变化
        actual_changes = [l for l in change_lines if not l.startswith('+++') and not l.startswith('---')]
        self.assertEqual(len(actual_changes), 0)

    def test_diff_for_class_methods(self):
        """
        测试类方法的diff生成

        Ground Truth:
        - diff应正确显示类方法的变化
        - 应包含方法签名和实现的差异
        """
        diff = _get_source_diff(
            MockOriginalClass.compute,
            MockReplacementClass.compute
        )

        # Ground truth assertions
        self.assertIn("compute", diff)
        self.assertIn("self", diff)

    def test_diff_builtin_returns_empty(self):
        """
        测试内置函数的diff返回空字符串

        Ground Truth:
        - 内置函数没有Python源码
        - _get_source_diff应返回空字符串
        """
        diff = _get_source_diff(len, str)

        # Ground truth: builtin functions have no source
        self.assertEqual(diff, "")


# =============================================================================
# Test: Basic Replacement with Ground Truth
# =============================================================================

class TestBasicReplacementGroundTruth(unittest.TestCase):
    """
    基本替换功能测试 - 使用ground truth验证替换效果
    """

    def setUp(self):
        """创建测试模块"""
        self.mock_module = types.ModuleType('usage_test_module')
        self.mock_module.add = mock_original_add
        self.mock_module.multiply = mock_original_multiply
        self.mock_module.process = mock_original_process
        sys.modules['usage_test_module'] = self.mock_module

    def tearDown(self):
        """清理测试模块"""
        if 'usage_test_module' in sys.modules:
            del sys.modules['usage_test_module']

    def test_function_replacement_ground_truth(self):
        """
        测试函数替换的ground truth

        Ground Truth:
        - 原始: add(3, 5) = 8
        - 替换后: add(3, 5) = 108 (3 + 5 + 100)
        """
        # Ground truth: before patch
        self.assertEqual(self.mock_module.add(3, 5), 8)

        patch = AtomicPatch("usage_test_module.add", mock_replacement_add)
        patch.apply()

        # Ground truth: after patch
        self.assertEqual(self.mock_module.add(3, 5), 108)

    def test_multiple_replacements_ground_truth(self):
        """
        测试多个函数替换的ground truth

        Ground Truth:
        - add(2, 3) = 5 -> 105
        - multiply(2, 3) = 6 -> 60
        - process([1,2,3]) = [2,4,6] -> [1,4,9]
        """
        # Ground truth: before patches
        self.assertEqual(self.mock_module.add(2, 3), 5)
        self.assertEqual(self.mock_module.multiply(2, 3), 6)
        self.assertEqual(self.mock_module.process([1, 2, 3]), [2, 4, 6])

        patcher = Patcher()
        patcher.add(
            AtomicPatch("usage_test_module.add", mock_replacement_add),
            AtomicPatch("usage_test_module.multiply", mock_replacement_multiply),
            AtomicPatch("usage_test_module.process", mock_replacement_process),
        )
        patcher.apply()

        # Ground truth: after patches
        self.assertEqual(self.mock_module.add(2, 3), 105)
        self.assertEqual(self.mock_module.multiply(2, 3), 60)
        self.assertEqual(self.mock_module.process([1, 2, 3]), [1, 4, 9])

    def test_string_path_replacement_ground_truth(self):
        """
        测试字符串路径替换的ground truth

        Ground Truth:
        - 使用字符串路径指定replacement
        - 替换效果与直接传入函数相同
        """
        # 在模块中添加replacement函数
        self.mock_module.replacement_add = mock_replacement_add

        # Ground truth: before patch
        self.assertEqual(self.mock_module.add(10, 20), 30)

        patch = AtomicPatch(
            "usage_test_module.add",
            "usage_test_module.replacement_add"
        )
        patch.apply()

        # Ground truth: after patch (10 + 20 + 100 = 130)
        self.assertEqual(self.mock_module.add(10, 20), 130)


# =============================================================================
# Test: Wrapper Modes with Ground Truth
# =============================================================================

class TestWrapperModesGroundTruth(unittest.TestCase):
    """
    Wrapper模式测试 - 验证target_wrapper和replacement_wrapper的效果
    """

    def setUp(self):
        """创建测试模块"""
        self.mock_module = types.ModuleType('wrapper_test_module')
        self.mock_module.add = mock_original_add
        self.mock_module.multiply = mock_original_multiply
        sys.modules['wrapper_test_module'] = self.mock_module

    def tearDown(self):
        """清理测试模块"""
        if 'wrapper_test_module' in sys.modules:
            del sys.modules['wrapper_test_module']

    def test_target_wrapper_ground_truth(self):
        """
        测试target_wrapper的ground truth

        Ground Truth:
        - 原始: add(3, 5) = 8
        - wrapper: 在原始结果上乘以10
        - 包装后: add(3, 5) = 80
        """
        def multiply_result_wrapper(original_func):
            def wrapped(a, b):
                result = original_func(a, b)
                return result * 10
            return wrapped

        # Ground truth: before patch
        self.assertEqual(self.mock_module.add(3, 5), 8)

        patch = AtomicPatch(
            "wrapper_test_module.add",
            target_wrapper=multiply_result_wrapper
        )
        patch.apply()

        # Ground truth: after patch (8 * 10 = 80)
        self.assertEqual(self.mock_module.add(3, 5), 80)

    def test_replacement_wrapper_ground_truth(self):
        """
        测试replacement_wrapper的ground truth

        Ground Truth:
        - replacement: add(a, b) = a + b + 100
        - wrapper: 在replacement结果上加1000
        - 最终: add(3, 5) = 108 + 1000 = 1108
        """
        def add_thousand_wrapper(replacement_func):
            def wrapped(a, b):
                result = replacement_func(a, b)
                return result + 1000
            return wrapped

        # Ground truth: before patch
        self.assertEqual(self.mock_module.add(3, 5), 8)

        patch = AtomicPatch(
            "wrapper_test_module.add",
            mock_replacement_add,
            replacement_wrapper=add_thousand_wrapper
        )
        patch.apply()

        # Ground truth: after patch
        # mock_replacement_add(3, 5) = 3 + 5 + 100 = 108
        # wrapper adds 1000: 108 + 1000 = 1108
        self.assertEqual(self.mock_module.add(3, 5), 1108)

    def test_wrapper_with_logging_ground_truth(self):
        """
        测试带日志功能的wrapper

        Ground Truth:
        - wrapper记录调用参数和结果
        - 原始功能不变
        """
        call_log = []

        def logging_wrapper(original_func):
            def wrapped(a, b):
                result = original_func(a, b)
                call_log.append({'args': (a, b), 'result': result})
                return result
            return wrapped

        patch = AtomicPatch(
            "wrapper_test_module.add",
            target_wrapper=logging_wrapper
        )
        patch.apply()

        # 调用函数
        result1 = self.mock_module.add(1, 2)
        result2 = self.mock_module.add(10, 20)

        # Ground truth: 结果不变
        self.assertEqual(result1, 3)
        self.assertEqual(result2, 30)

        # Ground truth: 日志记录正确
        self.assertEqual(len(call_log), 2)
        self.assertEqual(call_log[0], {'args': (1, 2), 'result': 3})
        self.assertEqual(call_log[1], {'args': (10, 20), 'result': 30})


# =============================================================================
# Test: Runtime Check with Ground Truth
# =============================================================================

class TestRuntimeCheckGroundTruth(unittest.TestCase):
    """
    Runtime Check测试 - 验证运行时条件分发
    """

    def setUp(self):
        """创建测试模块"""
        self.mock_module = types.ModuleType('runtime_test_module')
        self.mock_module.compute = lambda x: x * 2
        sys.modules['runtime_test_module'] = self.mock_module

    def tearDown(self):
        """清理测试模块"""
        if 'runtime_test_module' in sys.modules:
            del sys.modules['runtime_test_module']

    def test_runtime_check_condition_met_ground_truth(self):
        """
        测试runtime_check条件满足时使用replacement

        Ground Truth:
        - 条件: x > 10
        - 原始: compute(x) = x * 2
        - 替换: compute(x) = x * 100
        - x=20 (>10): 使用替换, 结果=2000
        - x=5 (<=10): 使用原始, 结果=10
        """
        def optimized_compute(x):
            return x * 100

        patch = AtomicPatch(
            "runtime_test_module.compute",
            optimized_compute,
            runtime_check=lambda x: x > 10
        )
        patch.apply()

        # Ground truth: x > 10, use replacement
        self.assertEqual(self.mock_module.compute(20), 2000)
        self.assertEqual(self.mock_module.compute(15), 1500)

        # Ground truth: x <= 10, use original
        self.assertEqual(self.mock_module.compute(5), 10)
        self.assertEqual(self.mock_module.compute(10), 20)

    def test_runtime_check_type_based_ground_truth(self):
        """
        测试基于类型的runtime_check

        Ground Truth:
        - 条件: isinstance(x, int)
        - int类型使用优化版本
        - 其他类型使用原始版本
        """
        self.mock_module.process = lambda x: str(x)

        def optimized_process(x):
            return f"INT:{x}"

        patch = AtomicPatch(
            "runtime_test_module.process",
            optimized_process,
            runtime_check=lambda x: isinstance(x, int)
        )
        patch.apply()

        # Ground truth: int type uses replacement
        self.assertEqual(self.mock_module.process(42), "INT:42")

        # Ground truth: other types use original
        self.assertEqual(self.mock_module.process("hello"), "hello")
        self.assertEqual(self.mock_module.process(3.14), "3.14")

    def test_runtime_check_exception_fallback_ground_truth(self):
        """
        测试runtime_check异常时回退到原始函数

        Ground Truth:
        - runtime_check抛出异常时，使用原始函数
        - 不会导致程序崩溃
        """
        def bad_check(x):
            if x == 0:
                raise ValueError("Cannot check zero")
            return x > 5

        def replacement(x):
            return x * 100

        patch = AtomicPatch(
            "runtime_test_module.compute",
            replacement,
            runtime_check=bad_check
        )
        patch.apply()

        # Ground truth: check passes, use replacement
        self.assertEqual(self.mock_module.compute(10), 1000)

        # Ground truth: check raises exception, fallback to original
        self.assertEqual(self.mock_module.compute(0), 0)  # 0 * 2 = 0


# =============================================================================
# Test: Class and Method Patching with Ground Truth
# =============================================================================

class TestClassMethodPatchingGroundTruth(unittest.TestCase):
    """
    类和方法补丁测试 - 验证对类方法的补丁效果
    """

    def setUp(self):
        """创建测试模块"""
        self.mock_module = types.ModuleType('class_test_module')
        self.mock_module.MyClass = MockOriginalClass
        sys.modules['class_test_module'] = self.mock_module

    def tearDown(self):
        """清理测试模块"""
        if 'class_test_module' in sys.modules:
            del sys.modules['class_test_module']

    def test_instance_method_patch_ground_truth(self):
        """
        测试实例方法补丁的ground truth

        Ground Truth:
        - 原始: obj.compute(5) = value + 5
        - 替换: obj.compute(5) = value * 5
        - obj.value=10时: 原始=15, 替换=50
        """
        def new_compute(self, x):
            return self.value * x

        # Ground truth: before patch
        obj = self.mock_module.MyClass(10)
        self.assertEqual(obj.compute(5), 15)  # 10 + 5

        patch = AtomicPatch(
            "class_test_module.MyClass.compute",
            new_compute
        )
        patch.apply()

        # Ground truth: after patch
        obj2 = self.mock_module.MyClass(10)
        self.assertEqual(obj2.compute(5), 50)  # 10 * 5

    def test_static_method_patch_ground_truth(self):
        """
        测试静态方法补丁的ground truth

        Ground Truth:
        - 原始: static_compute(5) = 5 * 2 = 10
        - 替换: static_compute(5) = 5 ** 2 = 25
        """
        # Ground truth: before patch
        self.assertEqual(self.mock_module.MyClass.static_compute(5), 10)

        patch = AtomicPatch(
            "class_test_module.MyClass.static_compute",
            staticmethod(lambda x: x ** 2)
        )
        patch.apply()

        # Ground truth: after patch
        self.assertEqual(self.mock_module.MyClass.static_compute(5), 25)

    def test_class_method_patch_ground_truth(self):
        """
        测试类方法补丁的ground truth

        Ground Truth:
        - 原始: class_compute(5) = 5 * 3 = 15
        - 替换: class_compute(5) = 5 ** 3 = 125
        """
        # Ground truth: before patch
        self.assertEqual(self.mock_module.MyClass.class_compute(5), 15)

        @classmethod
        def new_class_compute(cls, x):
            return x ** 3

        patch = AtomicPatch(
            "class_test_module.MyClass.class_compute",
            new_class_compute
        )
        patch.apply()

        # Ground truth: after patch
        self.assertEqual(self.mock_module.MyClass.class_compute(5), 125)


# =============================================================================
# Test: Precheck with Ground Truth
# =============================================================================

class TestPrecheckGroundTruth(unittest.TestCase):
    """
    Precheck测试 - 验证预检查功能
    """

    def setUp(self):
        """创建测试模块"""
        self.mock_module = types.ModuleType('precheck_test_module')
        self.mock_module.func = lambda x: x
        sys.modules['precheck_test_module'] = self.mock_module

    def tearDown(self):
        """清理测试模块"""
        if 'precheck_test_module' in sys.modules:
            del sys.modules['precheck_test_module']

    def test_precheck_pass_ground_truth(self):
        """
        测试precheck通过时补丁被应用

        Ground Truth:
        - precheck返回True时，补丁应用
        - func(5) = 5 -> 50
        """
        patch = AtomicPatch(
            "precheck_test_module.func",
            lambda x: x * 10,
            precheck=lambda: True
        )
        result = patch.apply()

        # Ground truth
        self.assertEqual(result.status, PatchStatus.APPLIED)
        self.assertEqual(self.mock_module.func(5), 50)

    def test_precheck_fail_ground_truth(self):
        """
        测试precheck失败时补丁被跳过

        Ground Truth:
        - precheck返回False时，补丁跳过
        - func(5) = 5 (保持原样)
        """
        patch = AtomicPatch(
            "precheck_test_module.func",
            lambda x: x * 10,
            precheck=lambda: False
        )
        result = patch.apply()

        # Ground truth
        self.assertEqual(result.status, PatchStatus.SKIPPED)
        self.assertEqual(self.mock_module.func(5), 5)

    def test_precheck_with_target_param_ground_truth(self):
        """
        测试precheck接收target参数

        Ground Truth:
        - precheck可以根据target路径决定是否应用
        """
        received_target = []

        def precheck_with_target(target):
            received_target.append(target)
            return target.endswith('.func')

        patch = AtomicPatch(
            "precheck_test_module.func",
            lambda x: x * 10,
            precheck=precheck_with_target
        )
        result = patch.apply()

        # Ground truth
        self.assertEqual(result.status, PatchStatus.APPLIED)
        self.assertEqual(received_target[0], "precheck_test_module.func")
        self.assertEqual(self.mock_module.func(5), 50)


# =============================================================================
# Test: Aliases with Ground Truth
# =============================================================================

class TestAliasesGroundTruth(unittest.TestCase):
    """
    Aliases测试 - 验证别名补丁功能
    """

    def setUp(self):
        """创建测试模块"""
        self.mock_module = types.ModuleType('alias_test_module')
        self.mock_module.main = types.ModuleType('alias_test_module.main')
        self.mock_module.alias1 = types.ModuleType('alias_test_module.alias1')
        self.mock_module.alias2 = types.ModuleType('alias_test_module.alias2')

        # 原始函数
        original_func = lambda x: x * 2
        self.mock_module.main.func = original_func
        self.mock_module.alias1.func = original_func
        self.mock_module.alias2.func = original_func

        sys.modules['alias_test_module'] = self.mock_module
        sys.modules['alias_test_module.main'] = self.mock_module.main
        sys.modules['alias_test_module.alias1'] = self.mock_module.alias1
        sys.modules['alias_test_module.alias2'] = self.mock_module.alias2

    def tearDown(self):
        """清理测试模块"""
        for name in list(sys.modules.keys()):
            if name.startswith('alias_test_module'):
                del sys.modules[name]

    def test_aliases_all_patched_ground_truth(self):
        """
        测试所有别名都被补丁

        Ground Truth:
        - 主路径和所有别名都应用相同的补丁
        - func(5) = 10 -> 50 (所有路径)
        """
        # Ground truth: before patch
        self.assertEqual(self.mock_module.main.func(5), 10)
        self.assertEqual(self.mock_module.alias1.func(5), 10)
        self.assertEqual(self.mock_module.alias2.func(5), 10)

        patch = AtomicPatch(
            "alias_test_module.main.func",
            lambda x: x * 10,
            aliases=[
                "alias_test_module.alias1.func",
                "alias_test_module.alias2.func"
            ]
        )
        patch.apply()

        # Ground truth: after patch - all paths patched
        self.assertEqual(self.mock_module.main.func(5), 50)
        self.assertEqual(self.mock_module.alias1.func(5), 50)
        self.assertEqual(self.mock_module.alias2.func(5), 50)


# =============================================================================
# Test: Patcher Integration with Ground Truth
# =============================================================================

class TestPatcherIntegrationGroundTruth(unittest.TestCase):
    """
    Patcher集成测试 - 验证Patcher的完整工作流程
    """

    def setUp(self):
        """创建测试模块"""
        self.mock_module = types.ModuleType('patcher_int_test')
        self.mock_module.func1 = lambda x: x + 1
        self.mock_module.func2 = lambda x: x + 2
        self.mock_module.func3 = lambda x: x + 3
        sys.modules['patcher_int_test'] = self.mock_module

    def tearDown(self):
        """清理测试模块"""
        if 'patcher_int_test' in sys.modules:
            del sys.modules['patcher_int_test']

    def test_patcher_apply_order_ground_truth(self):
        """
        测试Patcher按顺序应用补丁

        Ground Truth:
        - 补丁按添加顺序应用
        - 每个补丁独立生效
        """
        patcher = Patcher()
        patcher.add(AtomicPatch("patcher_int_test.func1", lambda x: x * 10))
        patcher.add(AtomicPatch("patcher_int_test.func2", lambda x: x * 20))
        patcher.add(AtomicPatch("patcher_int_test.func3", lambda x: x * 30))
        patcher.apply()

        # Ground truth
        self.assertEqual(self.mock_module.func1(5), 50)
        self.assertEqual(self.mock_module.func2(5), 100)
        self.assertEqual(self.mock_module.func3(5), 150)

    def test_patcher_disable_ground_truth(self):
        """
        测试Patcher禁用特定补丁

        Ground Truth:
        - 被禁用的补丁不会应用
        - 其他补丁正常应用
        """
        patcher = Patcher()
        patcher.add(AtomicPatch("patcher_int_test.func1", lambda x: x * 10))
        patcher.add(AtomicPatch("patcher_int_test.func2", lambda x: x * 20))
        patcher.disable("patcher_int_test.func2")
        patcher.apply()

        # Ground truth
        self.assertEqual(self.mock_module.func1(5), 50)  # patched
        self.assertEqual(self.mock_module.func2(5), 7)   # not patched (5 + 2)

    def test_patcher_context_manager_ground_truth(self):
        """
        测试Patcher作为上下文管理器

        Ground Truth:
        - with块内补丁生效
        """
        patcher = Patcher()
        patcher.add(AtomicPatch("patcher_int_test.func1", lambda x: x * 100))

        with patcher:
            # Ground truth: inside context, patch is applied
            self.assertEqual(self.mock_module.func1(5), 500)


# =============================================================================
# Test: Patch Class with Ground Truth
# =============================================================================

class TestPatchClassGroundTruth(unittest.TestCase):
    """
    Patch类测试 - 验证自定义Patch子类的功能
    """

    def setUp(self):
        """创建测试模块"""
        self.mock_module = types.ModuleType('patch_class_test')
        self.mock_module.add = lambda a, b: a + b
        self.mock_module.sub = lambda a, b: a - b
        sys.modules['patch_class_test'] = self.mock_module

    def tearDown(self):
        """清理测试模块"""
        if 'patch_class_test' in sys.modules:
            del sys.modules['patch_class_test']

    def test_custom_patch_class_ground_truth(self):
        """
        测试自定义Patch类

        Ground Truth:
        - Patch子类的patches()方法返回的补丁都被应用
        """
        class MathPatch(Patch):
            name = "math_patch"

            @classmethod
            def patches(cls, options=None):
                return [
                    AtomicPatch("patch_class_test.add", lambda a, b: a + b + 100),
                    AtomicPatch("patch_class_test.sub", lambda a, b: a - b - 100),
                ]

        patcher = Patcher()
        patcher.add(MathPatch)
        patcher.apply()

        # Ground truth
        self.assertEqual(self.mock_module.add(3, 5), 108)   # 3 + 5 + 100
        self.assertEqual(self.mock_module.sub(10, 3), -93)  # 10 - 3 - 100

    def test_patch_class_with_options_ground_truth(self):
        """
        测试带options的Patch类

        Ground Truth:
        - options参数传递到patches()方法
        - 可以根据options动态生成补丁
        """
        class ConfigurablePatch(Patch):
            name = "configurable_patch"

            @classmethod
            def patches(cls, options=None):
                multiplier = (options or {}).get('multiplier', 1)
                return [
                    AtomicPatch(
                        "patch_class_test.add",
                        lambda a, b, m=multiplier: (a + b) * m
                    ),
                ]

        patcher = Patcher()
        patcher.add(ConfigurablePatch, options={'multiplier': 10})
        patcher.apply()

        # Ground truth: (3 + 5) * 10 = 80
        self.assertEqual(self.mock_module.add(3, 5), 80)


# =============================================================================
# Test: Error Handling with Ground Truth
# =============================================================================

class TestErrorHandlingGroundTruth(unittest.TestCase):
    """
    错误处理测试 - 验证各种错误情况的处理
    """

    def setUp(self):
        """创建测试模块"""
        self.mock_module = types.ModuleType('error_test_module')
        self.mock_module.func = lambda x: x
        sys.modules['error_test_module'] = self.mock_module

    def tearDown(self):
        """清理测试模块"""
        if 'error_test_module' in sys.modules:
            del sys.modules['error_test_module']

    def test_module_not_found_ground_truth(self):
        """
        测试模块不存在时的处理

        Ground Truth:
        - 返回SKIPPED状态
        - reason包含"module not found"
        """
        patch = AtomicPatch("nonexistent_module.func", lambda x: x)
        result = patch.apply()

        # Ground truth
        self.assertEqual(result.status, PatchStatus.SKIPPED)
        self.assertIn("module not found", result.reason)

    def test_invalid_target_path_ground_truth(self):
        """
        测试无效目标路径的处理

        Ground Truth:
        - 单段路径返回FAILED（用户代码错误）
        - reason包含"invalid target path"
        """
        patch = AtomicPatch("singlepart", lambda x: x)
        result = patch.apply()

        # Ground truth
        self.assertEqual(result.status, PatchStatus.FAILED)
        self.assertIn("invalid target path", result.reason)

    def test_target_wrapper_error_ground_truth(self):
        """
        测试target_wrapper错误的处理

        Ground Truth:
        - wrapper抛出异常时返回FAILED
        - reason包含错误信息
        """
        def bad_wrapper(original):
            raise RuntimeError("Wrapper failed!")

        patch = AtomicPatch(
            "error_test_module.func",
            target_wrapper=bad_wrapper
        )
        result = patch.apply()

        # Ground truth
        self.assertEqual(result.status, PatchStatus.FAILED)
        self.assertIn("Wrapper failed", result.reason)

    def test_replacement_wrapper_error_ground_truth(self):
        """
        测试replacement_wrapper错误的处理

        Ground Truth:
        - wrapper抛出异常时返回FAILED
        """
        def bad_wrapper(replacement):
            raise RuntimeError("Replacement wrapper failed!")

        patch = AtomicPatch(
            "error_test_module.func",
            lambda x: x * 2,
            replacement_wrapper=bad_wrapper
        )
        result = patch.apply()

        # Ground truth
        self.assertEqual(result.status, PatchStatus.FAILED)
        self.assertIn("replacement_wrapper error", result.reason)


# =============================================================================
# Test: Get Info with Ground Truth
# =============================================================================

class TestGetInfoGroundTruth(unittest.TestCase):
    """
    get_info测试 - 验证补丁信息获取功能
    """

    def setUp(self):
        """创建测试模块"""
        self.mock_module = types.ModuleType('info_test_module')
        self.mock_module.original_func = mock_original_add
        sys.modules['info_test_module'] = self.mock_module

    def tearDown(self):
        """清理测试模块"""
        if 'info_test_module' in sys.modules:
            del sys.modules['info_test_module']

    def test_get_info_contains_target_ground_truth(self):
        """
        测试get_info包含目标路径

        Ground Truth:
        - info字符串包含完整的目标路径
        """
        patch = AtomicPatch(
            "info_test_module.original_func",
            mock_replacement_add
        )
        patch.apply()

        info = patch.get_info()

        # Ground truth
        self.assertIn("info_test_module.original_func", info)

    def test_get_info_contains_function_names_ground_truth(self):
        """
        测试get_info包含函数名称

        Ground Truth:
        - info包含原始函数名和替换函数名
        """
        patch = AtomicPatch(
            "info_test_module.original_func",
            mock_replacement_add
        )
        patch.apply()

        info = patch.get_info()

        # Ground truth
        self.assertIn("mock_original_add", info)
        self.assertIn("mock_replacement_add", info)

    def test_get_info_with_diff_ground_truth(self):
        """
        测试get_info(show_diff=True)包含diff

        Ground Truth:
        - show_diff=True时，info包含源码差异
        - diff包含unified格式标记
        """
        patch = AtomicPatch(
            "info_test_module.original_func",
            mock_replacement_add
        )
        patch.apply()

        info = patch.get_info(show_diff=True)

        # Ground truth: contains diff markers
        self.assertIn("---", info)
        self.assertIn("+++", info)
        self.assertIn("@@", info)


# =============================================================================
# Test: Callable Name with Ground Truth
# =============================================================================

class TestCallableNameGroundTruth(unittest.TestCase):
    """
    _get_callable_name测试 - 验证可调用对象名称获取
    """

    def test_function_name_ground_truth(self):
        """
        测试函数名称获取

        Ground Truth:
        - 返回函数的__qualname__或__name__
        """
        name = _get_callable_name(mock_original_add)
        self.assertEqual(name, "mock_original_add")

    def test_lambda_name_ground_truth(self):
        """
        测试lambda名称获取

        Ground Truth:
        - lambda函数名称包含"<lambda>"
        """
        func = lambda x: x
        name = _get_callable_name(func)
        self.assertIn("lambda", name)

    def test_class_name_ground_truth(self):
        """
        测试类名称获取

        Ground Truth:
        - 返回类的__qualname__
        """
        name = _get_callable_name(MockOriginalClass)
        self.assertEqual(name, "MockOriginalClass")

    def test_method_name_ground_truth(self):
        """
        测试方法名称获取

        Ground Truth:
        - 返回方法的完整限定名
        """
        name = _get_callable_name(MockOriginalClass.compute)
        self.assertIn("compute", name)

    def test_none_name_ground_truth(self):
        """
        测试None的名称获取

        Ground Truth:
        - None返回"<None>"
        """
        name = _get_callable_name(None)
        self.assertEqual(name, "<None>")


if __name__ == "__main__":
    unittest.main()
