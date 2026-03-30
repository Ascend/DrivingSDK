# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Patcher Fail-Safe机制测试模块

本模块测试patcher的fail-safe机制，确保：
1. default_patcher应用时，部分patch失败不会打断程序
2. 用户能清晰看到哪些patch应用了，哪些没应用
3. 错误使用patcher时能得到明确的错误提示
4. runtime check失败时使用debug级别logging
"""
import importlib.util
import logging
import os
import sys
import types
import unittest
from io import StringIO

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


# Load patcher modules directly
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

# Import classes
AtomicPatch = _patch_module.AtomicPatch
Patch = _patch_module.Patch
Patcher = _patcher_module.Patcher
PatchResult = _reporting.PatchResult
PatchStatus = _reporting.PatchStatus
patcher_logger = _patcher_logger_module.patcher_logger
PatcherLogger = _patcher_logger_module.PatcherLogger


class TestFailSafeMechanism(unittest.TestCase):
    """
    Fail-Safe机制测试

    验证default_patcher的fail-safe特性：
    - 部分patch失败不会打断程序
    - 用户能清晰看到哪些patch应用了，哪些没应用
    """

    def setUp(self):
        """创建测试模块"""
        self.mock_module = types.ModuleType('failsafe_test_module')
        self.mock_module.func1 = lambda x: x + 1
        self.mock_module.func2 = lambda x: x + 2
        self.mock_module.func3 = lambda x: x + 3
        sys.modules['failsafe_test_module'] = self.mock_module

    def tearDown(self):
        """清理测试模块"""
        if 'failsafe_test_module' in sys.modules:
            del sys.modules['failsafe_test_module']

    def test_partial_failure_does_not_break_program(self):
        """
        测试部分patch失败不会打断程序

        Ground Truth:
        - 即使有patch失败，其他patch仍然应用
        - 程序不会抛出异常
        """
        patcher = Patcher()
        patcher.add(
            # 这个会成功
            AtomicPatch("failsafe_test_module.func1", lambda x: x * 10),
            # 这个会失败（模块不存在）
            AtomicPatch("nonexistent_module.func", lambda x: x),
            # 这个会成功
            AtomicPatch("failsafe_test_module.func2", lambda x: x * 20),
        )

        # 不应该抛出异常
        patcher.apply()

        # 成功的patch应该生效
        self.assertEqual(self.mock_module.func1(5), 50)
        self.assertEqual(self.mock_module.func2(5), 100)

    def test_precheck_failure_does_not_break_program(self):
        """
        测试precheck失败不会打断程序

        Ground Truth:
        - precheck返回False时，patch被跳过
        - 其他patch正常应用
        """
        patcher = Patcher()
        patcher.add(
            AtomicPatch("failsafe_test_module.func1", lambda x: x * 10),
            AtomicPatch(
                "failsafe_test_module.func2",
                lambda x: x * 20,
                precheck=lambda: False  # 总是失败
            ),
            AtomicPatch("failsafe_test_module.func3", lambda x: x * 30),
        )

        patcher.apply()

        # func1和func3应该被patch
        self.assertEqual(self.mock_module.func1(5), 50)
        self.assertEqual(self.mock_module.func3(5), 150)
        # func2应该保持原样
        self.assertEqual(self.mock_module.func2(5), 7)  # 5 + 2

    def test_mixed_success_and_failure(self):
        """
        测试混合成功和失败的场景

        Ground Truth:
        - 成功的patch应用
        - 失败的patch被跳过
        - 程序继续运行
        """
        class MixedPatch(Patch):
            name = "mixed_patch"

            @classmethod
            def patches(cls, options=None):
                return [
                    AtomicPatch("failsafe_test_module.func1", lambda x: x * 100),
                    AtomicPatch("nonexistent.func", lambda x: x),  # 会失败
                    AtomicPatch(
                        "failsafe_test_module.func2",
                        lambda x: x * 200,
                        precheck=lambda: False  # 会跳过
                    ),
                    AtomicPatch("failsafe_test_module.func3", lambda x: x * 300),
                ]

        patcher = Patcher()
        patcher.add(MixedPatch)
        patcher.apply()

        # 验证结果
        self.assertEqual(self.mock_module.func1(1), 100)  # patched
        self.assertEqual(self.mock_module.func2(1), 3)    # not patched (1 + 2)
        self.assertEqual(self.mock_module.func3(1), 300)  # patched


class TestLoggingVisibility(unittest.TestCase):
    """
    Logging可见性测试

    验证用户能清晰看到patch状态
    """

    def setUp(self):
        """创建测试模块和logger"""
        self.mock_module = types.ModuleType('logging_test_module')
        self.mock_module.func = lambda x: x
        sys.modules['logging_test_module'] = self.mock_module

        # 创建新的logger实例用于测试
        self.logger = PatcherLogger()
        self.logger.set_rank(0)
        self.logger.set_buffer_enabled(False)  # Disable buffering for immediate output

        # 捕获日志输出
        self.log_capture = StringIO()
        handler = logging.StreamHandler(self.log_capture)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        self.logger._logger.handlers = [handler]
        self.logger._logger.setLevel(logging.DEBUG)

    def tearDown(self):
        """清理测试模块"""
        if 'logging_test_module' in sys.modules:
            del sys.modules['logging_test_module']

    def test_applied_logged_at_info(self):
        """
        测试成功应用的patch在INFO级别记录

        Ground Truth:
        - on_apply默认为"info"
        - 成功的patch应该在INFO级别记录
        """
        self.logger.applied("test_patch")
        output = self.log_capture.getvalue()
        self.assertIn("INFO", output)
        self.assertIn("Applied: test_patch", output)

    def test_skipped_logged_at_info(self):
        """
        测试跳过的patch在INFO级别记录

        Ground Truth:
        - on_skip默认为"info"（不是debug）
        - 跳过的patch应该在INFO级别记录，让用户清楚看到
        """
        self.logger.skipped("test_patch", "precheck failed")
        output = self.log_capture.getvalue()
        self.assertIn("INFO", output)
        self.assertIn("Skipped: test_patch (precheck failed)", output)

    def test_failed_logged_at_warning(self):
        """
        测试失败的patch在WARNING级别记录

        Ground Truth:
        - on_fail默认为"warning"
        - 失败的patch应该在WARNING级别记录
        """
        self.logger.failed("test_patch", "error occurred")
        output = self.log_capture.getvalue()
        self.assertIn("WARNING", output)
        self.assertIn("Failed: test_patch (error occurred)", output)

    def test_default_level_is_info(self):
        """
        测试默认logging级别是INFO

        Ground Truth:
        - 默认级别应该是INFO，让用户看到所有patch状态
        """
        # Reset the shared logger to test default initialization behavior
        shared_logger = logging.getLogger("mx_driving.patcher")
        shared_logger.handlers.clear()
        shared_logger.setLevel(logging.NOTSET)

        # Create a new PatcherLogger which should set level to INFO
        logger = PatcherLogger()
        self.assertEqual(logger._logger.level, logging.INFO)


class TestPrecheckErrorHandling(unittest.TestCase):
    """
    Precheck错误处理测试

    验证precheck异常时的行为
    """

    def setUp(self):
        """创建测试模块"""
        self.mock_module = types.ModuleType('precheck_error_module')
        self.mock_module.func = lambda x: x
        sys.modules['precheck_error_module'] = self.mock_module

    def tearDown(self):
        """清理测试模块"""
        if 'precheck_error_module' in sys.modules:
            del sys.modules['precheck_error_module']

    def test_precheck_exception_returns_failed(self):
        """
        测试precheck抛出异常时返回FAILED

        Ground Truth:
        - precheck异常是代码bug，应该返回FAILED而不是SKIPPED
        - 错误信息应该包含异常详情
        """
        def bad_precheck():
            raise RuntimeError("precheck bug")

        patch = AtomicPatch(
            "precheck_error_module.func",
            lambda x: x * 2,
            precheck=bad_precheck
        )
        result = patch.apply()

        self.assertEqual(result.status, PatchStatus.FAILED)
        self.assertIn("precheck error", result.reason)
        self.assertIn("precheck bug", result.reason)

    def test_precheck_false_returns_skipped(self):
        """
        测试precheck返回False时返回SKIPPED

        Ground Truth:
        - precheck返回False是正常的跳过，应该返回SKIPPED
        """
        patch = AtomicPatch(
            "precheck_error_module.func",
            lambda x: x * 2,
            precheck=lambda: False
        )
        result = patch.apply()

        self.assertEqual(result.status, PatchStatus.SKIPPED)
        self.assertEqual(result.reason, "precheck failed")


class TestRuntimeCheckLogging(unittest.TestCase):
    """
    Runtime Check Logging测试

    验证runtime check失败时的logging行为
    """

    def setUp(self):
        """创建测试模块"""
        self.mock_module = types.ModuleType('runtime_check_module')
        self.mock_module.func = lambda x: x * 2
        sys.modules['runtime_check_module'] = self.mock_module

    def tearDown(self):
        """清理测试模块"""
        if 'runtime_check_module' in sys.modules:
            del sys.modules['runtime_check_module']

    def test_runtime_check_exception_uses_original(self):
        """
        测试runtime check异常时使用原始函数

        Ground Truth:
        - runtime check异常时，应该回退到原始函数
        - 不应该抛出异常
        """
        def bad_check(x):
            raise RuntimeError("check error")

        patch = AtomicPatch(
            "runtime_check_module.func",
            lambda x: x * 100,
            runtime_check=bad_check
        )
        patch.apply()

        # 应该使用原始函数
        result = self.mock_module.func(5)
        self.assertEqual(result, 10)  # 5 * 2

    def test_runtime_check_false_uses_original(self):
        """
        测试runtime check返回False时使用原始函数

        Ground Truth:
        - runtime check返回False时，使用原始函数
        """
        patch = AtomicPatch(
            "runtime_check_module.func",
            lambda x: x * 100,
            runtime_check=lambda x: False
        )
        patch.apply()

        result = self.mock_module.func(5)
        self.assertEqual(result, 10)  # 5 * 2

    def test_runtime_check_true_uses_replacement(self):
        """
        测试runtime check返回True时使用替换函数

        Ground Truth:
        - runtime check返回True时，使用替换函数
        """
        patch = AtomicPatch(
            "runtime_check_module.func",
            lambda x: x * 100,
            runtime_check=lambda x: True
        )
        patch.apply()

        result = self.mock_module.func(5)
        self.assertEqual(result, 500)  # 5 * 100


class TestPrintInfoUsesLogger(unittest.TestCase):
    """
    print_info使用logger测试

    验证print_info方法使用logger而不是print
    """

    def setUp(self):
        """创建测试模块"""
        self.mock_module = types.ModuleType('print_info_module')
        self.mock_module.func = lambda x: x
        sys.modules['print_info_module'] = self.mock_module

    def tearDown(self):
        """清理测试模块"""
        if 'print_info_module' in sys.modules:
            del sys.modules['print_info_module']

    def test_print_info_uses_logger(self):
        """
        测试print_info使用logger

        Ground Truth:
        - print_info应该通过logger输出，而不是print
        """
        # 捕获logger输出
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setFormatter(logging.Formatter("%(message)s"))

        # 获取patcher使用的logger
        patcher_log = logging.getLogger("mx_driving.patcher")
        original_handlers = patcher_log.handlers[:]
        patcher_log.handlers = [handler]
        patcher_log.setLevel(logging.INFO)

        try:
            patcher = Patcher()
            patcher.add(AtomicPatch("print_info_module.func", lambda x: x * 2))
            patcher.print_info()

            output = log_capture.getvalue()
            self.assertIn("Patcher Info", output)
            self.assertIn("print_info_module", output)
        finally:
            patcher_log.handlers = original_handlers


class TestSkippedVsFailedSemantics(unittest.TestCase):
    """
    SKIPPED vs FAILED语义测试

    验证两种状态的正确使用场景
    """

    def setUp(self):
        """创建测试模块"""
        self.mock_module = types.ModuleType('semantics_test_module')
        self.mock_module.func = lambda x: x
        sys.modules['semantics_test_module'] = self.mock_module

    def tearDown(self):
        """清理测试模块"""
        if 'semantics_test_module' in sys.modules:
            del sys.modules['semantics_test_module']

    def test_module_not_found_is_skipped(self):
        """
        测试模块不存在返回SKIPPED

        Ground Truth:
        - 模块不存在是预期内的情况（如可选依赖）
        - 应该返回SKIPPED
        """
        patch = AtomicPatch("nonexistent_module.func", lambda x: x)
        result = patch.apply()

        self.assertEqual(result.status, PatchStatus.SKIPPED)
        self.assertIn("module not found", result.reason)

    def test_precheck_false_is_skipped(self):
        """
        测试precheck返回False是SKIPPED

        Ground Truth:
        - precheck返回False是正常的条件跳过
        - 应该返回SKIPPED
        """
        patch = AtomicPatch(
            "semantics_test_module.func",
            lambda x: x * 2,
            precheck=lambda: False
        )
        result = patch.apply()

        self.assertEqual(result.status, PatchStatus.SKIPPED)

    def test_wrapper_error_is_failed(self):
        """
        测试wrapper错误返回FAILED

        Ground Truth:
        - wrapper执行错误是代码bug
        - 应该返回FAILED
        """
        def bad_wrapper(original):
            raise RuntimeError("wrapper bug")

        patch = AtomicPatch(
            "semantics_test_module.func",
            target_wrapper=bad_wrapper
        )
        result = patch.apply()

        self.assertEqual(result.status, PatchStatus.FAILED)
        self.assertIn("target_wrapper error", result.reason)

    def test_precheck_exception_is_failed(self):
        """
        测试precheck异常返回FAILED

        Ground Truth:
        - precheck抛出异常是代码bug
        - 应该返回FAILED
        """
        def bad_precheck():
            raise RuntimeError("precheck bug")

        patch = AtomicPatch(
            "semantics_test_module.func",
            lambda x: x * 2,
            precheck=bad_precheck
        )
        result = patch.apply()

        self.assertEqual(result.status, PatchStatus.FAILED)
        self.assertIn("precheck error", result.reason)


if __name__ == "__main__":
    unittest.main()
