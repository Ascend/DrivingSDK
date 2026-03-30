# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Patcher Logger测试模块

本模块测试patcher_logger的各种功能，包括：
- 基本日志功能
- 分布式训练场景下的rank控制
- 日志级别配置
- 各种LogAction的处理
"""
import importlib.util
import logging
import os
import sys
import types
import unittest
from io import StringIO
from unittest.mock import patch, MagicMock

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


# Load patcher_logger module directly
_patcher_logger_module = _load_module_from_file(
    "mx_driving.patcher.patcher_logger",
    os.path.join(_patcher_dir, "patcher_logger.py")
)

PatcherLogger = _patcher_logger_module.PatcherLogger
PatchError = _patcher_logger_module.PatchError
configure_patcher_logging = _patcher_logger_module.configure_patcher_logging
set_patcher_log_level = _patcher_logger_module.set_patcher_log_level
_get_rank_from_env = _patcher_logger_module._get_rank_from_env


class TestPatcherLoggerBasic(unittest.TestCase):
    """
    PatcherLogger基本功能测试
    """

    def setUp(self):
        """创建新的logger实例用于测试"""
        self.logger = PatcherLogger()
        # 设置为INFO级别以便测试
        self.logger.set_level(logging.INFO)
        # 确保是rank 0
        self.logger.set_rank(0)

    def test_default_configuration(self):
        """
        测试默认配置

        Ground Truth:
        - on_apply默认为"info"
        - on_skip默认为"info" (为了fail-safe可见性)
        - on_fail默认为"warning"
        - on_error默认为"warning"
        """
        logger = PatcherLogger()
        self.assertEqual(logger._on_apply, "info")
        self.assertEqual(logger._on_skip, "info")  # Changed from "debug" to "info" for fail-safe visibility
        self.assertEqual(logger._on_fail, "warning")
        self.assertEqual(logger._on_error, "warning")

    def test_configure_returns_self(self):
        """
        测试configure返回self以支持链式调用

        Ground Truth:
        - configure()返回logger实例本身
        """
        result = self.logger.configure(on_apply="debug")
        self.assertIs(result, self.logger)

    def test_configure_changes_behavior(self):
        """
        测试configure修改行为配置

        Ground Truth:
        - 配置后行为应该改变
        """
        self.logger.configure(
            on_apply="warning",
            on_skip="error",
            on_fail="silent",
            on_error="debug"
        )
        self.assertEqual(self.logger._on_apply, "warning")
        self.assertEqual(self.logger._on_skip, "error")
        self.assertEqual(self.logger._on_fail, "silent")
        self.assertEqual(self.logger._on_error, "debug")

    def test_configure_partial_update(self):
        """
        测试configure只更新指定的配置

        Ground Truth:
        - 只传入部分参数时，其他参数保持不变
        """
        original_skip = self.logger._on_skip
        original_fail = self.logger._on_fail

        self.logger.configure(on_apply="error")

        self.assertEqual(self.logger._on_apply, "error")
        self.assertEqual(self.logger._on_skip, original_skip)
        self.assertEqual(self.logger._on_fail, original_fail)

    def test_set_level_returns_self(self):
        """
        测试set_level返回self以支持链式调用

        Ground Truth:
        - set_level()返回logger实例本身
        """
        result = self.logger.set_level(logging.DEBUG)
        self.assertIs(result, self.logger)


class TestPatcherLoggerDistributed(unittest.TestCase):
    """
    PatcherLogger分布式训练支持测试
    """

    def setUp(self):
        """创建新的logger实例"""
        self.logger = PatcherLogger()
        self.logger.set_level(logging.INFO)

    def tearDown(self):
        """清理环境变量"""
        for var in ['RANK', 'LOCAL_RANK', 'OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'SLURM_PROCID']:
            if var in os.environ:
                del os.environ[var]

    def test_set_rank_manual(self):
        """
        测试手动设置rank

        Ground Truth:
        - set_rank(0)后is_main_process为True
        - set_rank(1)后is_main_process为False
        """
        self.logger.set_rank(0)
        self.assertTrue(self.logger.is_main_process)
        self.assertEqual(self.logger.rank, 0)

        self.logger.set_rank(1)
        self.assertFalse(self.logger.is_main_process)
        self.assertEqual(self.logger.rank, 1)

    def test_set_rank_returns_self(self):
        """
        测试set_rank返回self以支持链式调用

        Ground Truth:
        - set_rank()返回logger实例本身
        """
        result = self.logger.set_rank(0)
        self.assertIs(result, self.logger)

    def test_set_rank_from_env_pytorch(self):
        """
        测试从PyTorch环境变量检测rank

        Ground Truth:
        - 设置RANK环境变量后，rank应该正确检测
        """
        os.environ['RANK'] = '3'
        self.logger.set_rank_from_env()
        self.assertEqual(self.logger.rank, 3)
        self.assertFalse(self.logger.is_main_process)

    def test_set_rank_from_env_local_rank(self):
        """
        测试从LOCAL_RANK环境变量检测rank

        Ground Truth:
        - 设置LOCAL_RANK环境变量后，rank应该正确检测
        """
        os.environ['LOCAL_RANK'] = '2'
        self.logger.set_rank_from_env()
        self.assertEqual(self.logger.rank, 2)

    def test_set_rank_from_env_openmpi(self):
        """
        测试从OpenMPI环境变量检测rank

        Ground Truth:
        - 设置OMPI_COMM_WORLD_RANK环境变量后，rank应该正确检测
        """
        os.environ['OMPI_COMM_WORLD_RANK'] = '5'
        self.logger.set_rank_from_env()
        self.assertEqual(self.logger.rank, 5)

    def test_set_rank_from_env_mpich(self):
        """
        测试从MPICH环境变量检测rank

        Ground Truth:
        - 设置PMI_RANK环境变量后，rank应该正确检测
        """
        os.environ['PMI_RANK'] = '4'
        self.logger.set_rank_from_env()
        self.assertEqual(self.logger.rank, 4)

    def test_set_rank_from_env_slurm(self):
        """
        测试从SLURM环境变量检测rank

        Ground Truth:
        - 设置SLURM_PROCID环境变量后，rank应该正确检测
        """
        os.environ['SLURM_PROCID'] = '7'
        self.logger.set_rank_from_env()
        self.assertEqual(self.logger.rank, 7)

    def test_set_rank_from_env_priority(self):
        """
        测试环境变量优先级

        Ground Truth:
        - RANK优先于LOCAL_RANK
        """
        os.environ['RANK'] = '1'
        os.environ['LOCAL_RANK'] = '2'
        self.logger.set_rank_from_env()
        self.assertEqual(self.logger.rank, 1)

    def test_set_rank_from_env_invalid_value(self):
        """
        测试无效环境变量值

        Ground Truth:
        - 无效值应该被跳过，继续检查下一个变量
        """
        os.environ['RANK'] = 'invalid'
        os.environ['LOCAL_RANK'] = '3'
        self.logger.set_rank_from_env()
        self.assertEqual(self.logger.rank, 3)

    def test_set_rank_from_env_no_vars(self):
        """
        测试无环境变量时默认为0

        Ground Truth:
        - 没有设置任何rank环境变量时，默认为0
        """
        self.logger.set_rank_from_env()
        self.assertEqual(self.logger.rank, 0)
        self.assertTrue(self.logger.is_main_process)

    def test_set_rank_from_env_returns_self(self):
        """
        测试set_rank_from_env返回self以支持链式调用

        Ground Truth:
        - set_rank_from_env()返回logger实例本身
        """
        result = self.logger.set_rank_from_env()
        self.assertIs(result, self.logger)

    def test_lazy_rank_initialization(self):
        """
        测试rank的延迟初始化

        Ground Truth:
        - 在首次访问rank相关属性前，rank未初始化
        - 首次访问后，自动从环境变量初始化
        """
        logger = PatcherLogger()
        self.assertFalse(logger._rank_initialized)

        os.environ['RANK'] = '5'
        # 访问rank属性触发初始化
        _ = logger.rank
        self.assertTrue(logger._rank_initialized)
        self.assertEqual(logger._rank, 5)


class TestPatcherLoggerRankFiltering(unittest.TestCase):
    """
    测试分布式场景下的日志过滤
    """

    def setUp(self):
        """创建logger并捕获输出"""
        self.logger = PatcherLogger()
        self.logger.set_level(logging.DEBUG)

        # 捕获日志输出
        self.log_capture = StringIO()
        handler = logging.StreamHandler(self.log_capture)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger._logger.handlers = [handler]

    def test_rank0_logs_messages(self):
        """
        测试rank 0打印日志

        Ground Truth:
        - rank 0时，日志应该被打印
        """
        self.logger.set_rank(0)
        self.logger.configure(on_apply="info")
        self.logger.set_buffer_enabled(False)  # Disable buffering for immediate output
        self.logger.applied("test_patch")

        output = self.log_capture.getvalue()
        self.assertIn("Applied: test_patch", output)

    def test_rank_nonzero_silent(self):
        """
        测试非rank 0不打印日志

        Ground Truth:
        - rank > 0时，日志不应该被打印
        """
        self.logger.set_rank(1)
        self.logger.configure(on_apply="info")
        self.logger.applied("test_patch")

        output = self.log_capture.getvalue()
        self.assertEqual(output, "")

    def test_rank_nonzero_still_raises(self):
        """
        测试非rank 0仍然抛出异常

        Ground Truth:
        - rank > 0时，exception action仍然应该抛出异常
        """
        self.logger.set_rank(1)
        self.logger.configure(on_fail="exception")

        with self.assertRaises(PatchError):
            self.logger.failed("test_patch", "error")

    def test_backward_compat_methods_respect_rank(self):
        """
        测试向后兼容方法也遵守rank过滤

        Ground Truth:
        - debug/info/warning方法在非rank 0时不打印
        """
        self.logger.set_rank(1)

        self.logger.debug("debug message")
        self.logger.info("info message")
        self.logger.warning("warning message")

        output = self.log_capture.getvalue()
        self.assertEqual(output, "")


class TestPatcherLoggerActions(unittest.TestCase):
    """
    测试各种LogAction的处理
    """

    def setUp(self):
        """创建logger并捕获输出"""
        self.logger = PatcherLogger()
        self.logger.set_level(logging.DEBUG)
        self.logger.set_rank(0)
        self.logger.set_buffer_enabled(False)  # Disable buffering for immediate output

        # 捕获日志输出
        self.log_capture = StringIO()
        handler = logging.StreamHandler(self.log_capture)
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        self.logger._logger.handlers = [handler]

    def test_action_silent(self):
        """
        测试silent action

        Ground Truth:
        - silent action不产生任何输出
        """
        self.logger.configure(on_apply="silent")
        self.logger.applied("test_patch")

        output = self.log_capture.getvalue()
        self.assertEqual(output, "")

    def test_action_debug(self):
        """
        测试debug action

        Ground Truth:
        - debug action产生DEBUG级别日志
        """
        self.logger.configure(on_apply="debug")
        self.logger.applied("test_patch")

        output = self.log_capture.getvalue()
        self.assertIn("DEBUG", output)
        self.assertIn("Applied: test_patch", output)

    def test_action_info(self):
        """
        测试info action

        Ground Truth:
        - info action产生INFO级别日志
        """
        self.logger.configure(on_apply="info")
        self.logger.applied("test_patch")

        output = self.log_capture.getvalue()
        self.assertIn("INFO", output)
        self.assertIn("Applied: test_patch", output)

    def test_action_warning(self):
        """
        测试warning action

        Ground Truth:
        - warning action产生WARNING级别日志
        """
        self.logger.configure(on_fail="warning")
        self.logger.failed("test_patch", "reason")

        output = self.log_capture.getvalue()
        self.assertIn("WARNING", output)
        self.assertIn("Failed: test_patch (reason)", output)

    def test_action_error(self):
        """
        测试error action

        Ground Truth:
        - error action产生ERROR级别日志
        """
        self.logger.configure(on_error="error")
        self.logger.error("error message")

        output = self.log_capture.getvalue()
        self.assertIn("ERROR", output)
        self.assertIn("error message", output)

    def test_action_exception(self):
        """
        测试exception action

        Ground Truth:
        - exception action抛出PatchError异常
        """
        self.logger.configure(on_fail="exception")

        with self.assertRaises(PatchError) as context:
            self.logger.failed("test_patch", "error reason")

        self.assertIn("Failed: test_patch (error reason)", str(context.exception))

    def test_action_exit(self):
        """
        测试exit action

        Ground Truth:
        - exit action调用sys.exit(1)
        """
        self.logger.configure(on_error="exit")

        with self.assertRaises(SystemExit) as context:
            self.logger.error("fatal error")

        self.assertEqual(context.exception.code, 1)


class TestPatcherLoggerMethods(unittest.TestCase):
    """
    测试各种日志方法
    """

    def setUp(self):
        """创建logger并捕获输出"""
        self.logger = PatcherLogger()
        self.logger.set_level(logging.DEBUG)
        self.logger.set_rank(0)
        self.logger.set_buffer_enabled(False)  # Disable buffering for immediate output

        # 捕获日志输出
        self.log_capture = StringIO()
        handler = logging.StreamHandler(self.log_capture)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger._logger.handlers = [handler]

    def test_applied_message_format(self):
        """
        测试applied方法的消息格式

        Ground Truth:
        - 消息格式为"Applied: {name}"
        """
        self.logger.configure(on_apply="info")
        self.logger.applied("mmcv.ops.msda")

        output = self.log_capture.getvalue()
        self.assertIn("Applied: mmcv.ops.msda", output)

    def test_skipped_message_format_with_reason(self):
        """
        测试skipped方法带reason的消息格式

        Ground Truth:
        - 消息格式为"Skipped: {name} ({reason})"
        """
        self.logger.configure(on_skip="info")
        self.logger.skipped("torch.matmul", "precheck failed")

        output = self.log_capture.getvalue()
        self.assertIn("Skipped: torch.matmul (precheck failed)", output)

    def test_skipped_message_format_without_reason(self):
        """
        测试skipped方法不带reason的消息格式

        Ground Truth:
        - 消息格式为"Skipped: {name}"
        """
        self.logger.configure(on_skip="info")
        self.logger.skipped("torch.matmul")

        output = self.log_capture.getvalue()
        self.assertIn("Skipped: torch.matmul", output)
        self.assertNotIn("()", output)

    def test_failed_message_format_with_reason(self):
        """
        测试failed方法带reason的消息格式

        Ground Truth:
        - 消息格式为"Failed: {name} ({reason})"
        """
        self.logger.configure(on_fail="info")
        self.logger.failed("mmdet.model", "import error")

        output = self.log_capture.getvalue()
        self.assertIn("Failed: mmdet.model (import error)", output)

    def test_failed_message_format_without_reason(self):
        """
        测试failed方法不带reason的消息格式

        Ground Truth:
        - 消息格式为"Failed: {name}"
        """
        self.logger.configure(on_fail="info")
        self.logger.failed("mmdet.model")

        output = self.log_capture.getvalue()
        self.assertIn("Failed: mmdet.model", output)
        self.assertNotIn("()", output)


class TestGlobalFunctions(unittest.TestCase):
    """
    测试全局函数
    """

    def test_configure_patcher_logging(self):
        """
        测试configure_patcher_logging函数

        Ground Truth:
        - 函数应该配置全局patcher_logger
        - 返回patcher_logger实例
        """
        # 使用已加载的模块中的patcher_logger
        patcher_logger = _patcher_logger_module.patcher_logger

        # 保存原始值
        original_on_apply = patcher_logger._on_apply

        result = configure_patcher_logging(on_apply="warning")

        self.assertIs(result, patcher_logger)
        self.assertEqual(patcher_logger._on_apply, "warning")

        # 恢复默认
        configure_patcher_logging(on_apply=original_on_apply)

    def test_set_patcher_log_level(self):
        """
        测试set_patcher_log_level函数

        Ground Truth:
        - 函数应该设置全局patcher_logger的级别
        """
        # 使用已加载的模块中的patcher_logger
        patcher_logger = _patcher_logger_module.patcher_logger

        original_level = patcher_logger._logger.level

        set_patcher_log_level(logging.DEBUG)
        self.assertEqual(patcher_logger._logger.level, logging.DEBUG)

        # 恢复原始级别
        set_patcher_log_level(original_level)

    def test_get_rank_from_env_function(self):
        """
        测试_get_rank_from_env函数

        Ground Truth:
        - 函数应该从环境变量获取rank
        """
        # 清理环境变量
        for var in ['RANK', 'LOCAL_RANK', 'OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'SLURM_PROCID']:
            if var in os.environ:
                del os.environ[var]

        # 无环境变量时返回0
        self.assertEqual(_get_rank_from_env(), 0)

        # 设置环境变量
        os.environ['RANK'] = '5'
        self.assertEqual(_get_rank_from_env(), 5)

        # 清理
        del os.environ['RANK']


class TestPatchError(unittest.TestCase):
    """
    测试PatchError异常类
    """

    def test_patch_error_is_exception(self):
        """
        测试PatchError是Exception的子类

        Ground Truth:
        - PatchError应该是Exception的子类
        """
        self.assertTrue(issubclass(PatchError, Exception))

    def test_patch_error_message(self):
        """
        测试PatchError的消息

        Ground Truth:
        - PatchError应该正确存储和返回消息
        """
        error = PatchError("test error message")
        self.assertEqual(str(error), "test error message")


class TestPatcherLoggerSummaryFormatting(unittest.TestCase):
    """Test summary formatting details that affect readability."""

    def setUp(self):
        self.logger = PatcherLogger()
        self.logger.set_rank(0)
        self.logger.set_level(logging.INFO)
        self.log_capture = StringIO()
        handler = logging.StreamHandler(self.log_capture)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger._logger.handlers = [handler]

    def test_summary_uses_visible_banner(self):
        """Summary should include a stronger banner to stand out in noisy logs."""
        self.logger.applied("mmcv.ops.foo", package="mmcv", patch_name="foo_patch")
        self.logger.flush_summary()

        output = self.log_capture.getvalue()
        self.assertIn("=== MX-DRIVING PATCHER SUMMARY ===", output)
        self.assertIn("Applied Patches:", output)


class TestChainedCalls(unittest.TestCase):
    """
    测试链式调用
    """

    def test_full_chain(self):
        """
        测试完整的链式调用

        Ground Truth:
        - 所有方法都应该返回self，支持链式调用
        """
        logger = PatcherLogger()

        result = (
            logger
            .set_rank(0)
            .set_level(logging.DEBUG)
            .configure(on_apply="info", on_skip="debug")
        )

        self.assertIs(result, logger)
        self.assertEqual(logger._rank, 0)
        self.assertEqual(logger._logger.level, logging.DEBUG)
        self.assertEqual(logger._on_apply, "info")
        self.assertEqual(logger._on_skip, "debug")


if __name__ == "__main__":
    unittest.main()
