# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
Patcher logging module with configurable behavior for different event types.

This module controls patcher output through four independent layers:

1. **Python logger level** - minimum severity threshold for logger messages
   (set via ``set_patcher_log_level()`` / ``patcher_logger.set_level()``)
2. **Event actions** - what to do when a patch event occurs (log, raise, exit, silent)
   (set via ``configure_patcher_logging(on_apply=..., ...)``)
3. **Summary configuration** - controls the post-apply summary output
   (set via ``patcher_logger.configure_summary(...)``)
4. **Distributed rank filtering** - only rank 0 emits log/summary output;
   exception/exit actions execute on all ranks.

Event Actions (``LogAction``):
    - "debug": Log at DEBUG level
    - "info": Log at INFO level
    - "warning": Log at WARNING level
    - "error": Log at ERROR level
    - "exception": Raise ``PatchError``
    - "exit": Log error and terminate the program
    - "silent": No logging, no action

Summary:
    By default, ``apply()`` buffers patch results and emits a hierarchical
    summary through the logger (not ``print()``).  The summary respects:
    - ``summary_level``: the logging level used to emit the summary (default INFO)
    - ``show_applied / show_skipped / show_failed``: per-section visibility
    - event action config: ``on_apply="silent"`` automatically hides the
      Applied section in the summary (likewise for skip/fail)

Usage:
    from mx_driving.patcher import patcher_logger, configure_patcher_logging

    configure_patcher_logging(
        on_apply="info",
        on_skip="info",
        on_fail="warning",
        on_error="exception",
    )
"""
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

LogAction = Literal["debug", "info", "warning", "error", "exception", "exit", "silent"]

# ─── Internal: EventPolicy ──────────────────────────────────────────────────

_LOG_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


@dataclass
class EventPolicy:
    """Internal representation that separates mode from log level."""
    mode: Literal["log", "silent", "exception", "exit"] = "log"
    level: int = logging.INFO


def _parse_action(action: LogAction) -> EventPolicy:
    """Convert a user-facing LogAction string to an EventPolicy."""
    if action == "silent":
        return EventPolicy(mode="silent")
    if action == "exception":
        return EventPolicy(mode="exception")
    if action == "exit":
        return EventPolicy(mode="exit")
    return EventPolicy(mode="log", level=_LOG_LEVEL_MAP.get(action, logging.INFO))


# ─── Data structures ────────────────────────────────────────────────────────

@dataclass
class PatchInfo:
    """Information about a patch for hierarchical logging."""
    target: str           # Full target path (e.g., "mmcv.ops.msda.forward")
    package: str          # Package name (e.g., "mmcv")
    patch_name: str       # Patch name (e.g., "multi_scale_deformable_attention")
    reason: str = ""      # Reason for skip/fail
    extra_info: str = ""  # Additional info (e.g., version)


@dataclass
class SummaryConfig:
    """Configuration for summary output."""
    enabled: bool = True
    level: int = logging.INFO
    show_applied: Optional[bool] = None   # None = auto (follow on_apply action)
    show_skipped: Optional[bool] = None   # None = auto (follow on_skip action)
    show_failed: Optional[bool] = None    # None = auto (follow on_fail action)
    show_import_events: bool = True


def _get_rank_from_env() -> int:
    """
    Get the distributed rank from environment variables.

    Checks common environment variables used by different distributed frameworks:
    - RANK: PyTorch distributed
    - LOCAL_RANK: PyTorch distributed (local rank within node)
    - OMPI_COMM_WORLD_RANK: OpenMPI
    - PMI_RANK: MPICH
    - SLURM_PROCID: SLURM

    Returns:
        The rank (0 for main process), or 0 if not in distributed mode.
    """
    rank_vars = ['RANK', 'LOCAL_RANK', 'OMPI_COMM_WORLD_RANK', 'PMI_RANK', 'SLURM_PROCID']
    for var in rank_vars:
        rank_str = os.environ.get(var)
        if rank_str is not None:
            try:
                return int(rank_str)
            except ValueError:
                continue
    return 0


class PatchError(Exception):
    """Exception raised when a patch operation fails."""
    pass


class PatcherLogger:
    """
    Logger for patcher operations with configurable behavior per event type.

    Supports distributed training by only logging on rank 0 to avoid
    duplicate messages from multiple processes.

    The logging system has four independent layers:

    1. **Logger level** (``set_level``): Python logger threshold.
    2. **Event actions** (``configure``): per-event behavior (log/silent/exception/exit).
    3. **Summary** (``configure_summary``): post-apply hierarchical summary.
    4. **Rank filtering**: only rank 0 outputs logs and summary.

    Example:
        logger = PatcherLogger()

        logger.configure(
            on_apply="info",
            on_skip="debug",
            on_fail="warning",
            on_error="exception",
        )

        logger.set_rank_from_env()
    """

    def __init__(self):
        self._logger = logging.getLogger("mx_driving.patcher")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[Patcher] %(levelname)s: %(message)s"))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        # Event action policies (stored as LogAction strings for backward compat)
        self._on_apply: LogAction = "info"
        self._on_skip: LogAction = "info"
        self._on_fail: LogAction = "warning"
        self._on_error: LogAction = "warning"

        # Distributed training support
        self._rank: int = 0
        self._rank_initialized: bool = False

        # Summary configuration
        self._summary = SummaryConfig()

        # Buffered logging for summary output
        self._buffer_enabled: bool = True
        self._applied_patches: List[PatchInfo] = []
        self._skipped_patches: List[PatchInfo] = []
        self._failed_patches: List[PatchInfo] = []
        self._skipped_modules: List[str] = []
        self._injected_imports: List[str] = []
        self._replaced_imports: List[str] = []

    # ─── Rank ────────────────────────────────────────────────────────────

    def _ensure_rank_initialized(self):
        """Lazily initialize rank from environment on first log call."""
        if not self._rank_initialized:
            self._rank = _get_rank_from_env()
            self._rank_initialized = True

    def set_rank(self, rank: int) -> "PatcherLogger":
        """
        Set the distributed rank manually.

        Only rank 0 will produce log output. Other ranks will be silent
        to avoid duplicate messages in distributed training.

        Args:
            rank: The process rank (0 for main process).

        Returns:
            self for method chaining.
        """
        self._rank = rank
        self._rank_initialized = True
        return self

    def set_rank_from_env(self) -> "PatcherLogger":
        """
        Auto-detect and set rank from environment variables.

        Checks: RANK, LOCAL_RANK (PyTorch), OMPI_COMM_WORLD_RANK (OpenMPI),
        PMI_RANK (MPICH), SLURM_PROCID (SLURM).

        Returns:
            self for method chaining.
        """
        self._rank = _get_rank_from_env()
        self._rank_initialized = True
        return self

    @property
    def rank(self) -> int:
        """Get the current rank."""
        self._ensure_rank_initialized()
        return self._rank

    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        self._ensure_rank_initialized()
        return self._rank == 0

    # ─── Configuration ───────────────────────────────────────────────────

    def configure(
        self,
        on_apply: Optional[LogAction] = None,
        on_skip: Optional[LogAction] = None,
        on_fail: Optional[LogAction] = None,
        on_error: Optional[LogAction] = None,
    ) -> "PatcherLogger":
        """
        Configure event actions for different patch event types.

        Each parameter accepts a ``LogAction`` string that determines what
        happens when the corresponding event fires:

        - ``"debug"/"info"/"warning"/"error"``: emit a log message at that level
        - ``"silent"``: suppress the event entirely (also hides it in summary)
        - ``"exception"``: raise ``PatchError``
        - ``"exit"``: log an error and call ``sys.exit(1)``

        Args:
            on_apply: Action when a patch is successfully applied.
            on_skip: Action when a patch is skipped.
            on_fail: Action when a patch application fails.
            on_error: Action when an unexpected error occurs.

        Returns:
            self for method chaining.
        """
        if on_apply is not None:
            self._on_apply = on_apply
        if on_skip is not None:
            self._on_skip = on_skip
        if on_fail is not None:
            self._on_fail = on_fail
        if on_error is not None:
            self._on_error = on_error
        return self

    def configure_summary(
        self,
        enabled: Optional[bool] = None,
        level: Optional[int] = None,
        show_applied: Optional[bool] = None,
        show_skipped: Optional[bool] = None,
        show_failed: Optional[bool] = None,
        show_import_events: Optional[bool] = None,
    ) -> "PatcherLogger":
        """
        Configure summary output independently from event actions.

        Args:
            enabled: Whether to emit the summary at all (default True).
            level: Logging level for summary output (default logging.INFO).
                   The summary is only shown if the logger level allows it.
            show_applied: Show the Applied Patches section.
                          None (default) = auto-hide when on_apply is "silent".
            show_skipped: Show the Skipped Patches section.
                          None (default) = auto-hide when on_skip is "silent".
            show_failed: Show the Failed Patches section.
                         None (default) = auto-hide when on_fail is "silent".
            show_import_events: Show Skipped Modules / Injected / Replaced
                                Imports sections (default True).

        Returns:
            self for method chaining.
        """
        if enabled is not None:
            self._summary.enabled = enabled
        if level is not None:
            self._summary.level = level
        if show_applied is not None:
            self._summary.show_applied = show_applied
        if show_skipped is not None:
            self._summary.show_skipped = show_skipped
        if show_failed is not None:
            self._summary.show_failed = show_failed
        if show_import_events is not None:
            self._summary.show_import_events = show_import_events
        return self

    def set_level(self, level: int) -> "PatcherLogger":
        """
        Set the minimum logging level.

        This controls the Python logger threshold.  Both immediate event
        messages and the summary are emitted through the logger, so this
        level affects all output.

        Args:
            level: logging.DEBUG, logging.INFO, logging.WARNING, etc.
        """
        self._logger.setLevel(level)
        return self

    def set_buffer_enabled(self, enabled: bool) -> "PatcherLogger":
        """Enable or disable buffered logging."""
        self._buffer_enabled = enabled
        return self

    # ─── Internal dispatch ───────────────────────────────────────────────

    def _should_log(self) -> bool:
        """Check if logging should occur (only on rank 0)."""
        self._ensure_rank_initialized()
        return self._rank == 0

    def _handle(self, action: LogAction, message: str):
        """Execute the configured action for a log event."""
        policy = _parse_action(action)

        if policy.mode == "silent":
            return

        # exception/exit execute on ALL ranks
        if policy.mode == "exception":
            raise PatchError(message)
        if policy.mode == "exit":
            if self._should_log():
                self._logger.error(message)
            sys.exit(1)

        # log mode: only on rank 0
        if not self._should_log():
            return
        self._logger.log(policy.level, message)

    # ─── Event methods ───────────────────────────────────────────────────

    def applied(self, target: str, package: str = "", patch_name: str = "", extra_info: str = ""):
        """Log a successfully applied patch (buffered)."""
        if self._buffer_enabled:
            info = PatchInfo(target=target, package=package, patch_name=patch_name, extra_info=extra_info)
            self._applied_patches.append(info)
        else:
            self._handle(self._on_apply, f"Applied: {target}")

    def skipped(self, target: str, reason: str = "", package: str = "", patch_name: str = "", extra_info: str = ""):
        """Log a skipped patch (buffered)."""
        if self._buffer_enabled:
            info = PatchInfo(target=target, package=package, patch_name=patch_name, reason=reason, extra_info=extra_info)
            self._skipped_patches.append(info)
        else:
            msg = f"Skipped: {target}" + (f" ({reason})" if reason else "")
            self._handle(self._on_skip, msg)

    def failed(self, target: str, reason: str = "", package: str = "", patch_name: str = "", extra_info: str = ""):
        """Log a failed patch (immediately for exceptions, otherwise buffered)."""
        if self._on_fail in ("exception", "exit"):
            msg = f"Failed: {target}" + (f" ({reason})" if reason else "")
            self._handle(self._on_fail, msg)
        elif self._buffer_enabled:
            info = PatchInfo(target=target, package=package, patch_name=patch_name, reason=reason, extra_info=extra_info)
            self._failed_patches.append(info)
        else:
            msg = f"Failed: {target}" + (f" ({reason})" if reason else "")
            self._handle(self._on_fail, msg)

    def add_skipped_module(self, module_name: str):
        """Record a skipped module for summary output."""
        self._skipped_modules.append(module_name)

    def add_injected_import(self, import_path: str):
        """Record an injected import for summary output."""
        self._injected_imports.append(import_path)

    def add_replaced_import(self, desc: str):
        """Record a replaced import for summary output."""
        self._replaced_imports.append(desc)

    # ─── Summary ─────────────────────────────────────────────────────────

    def _should_show_section(self, action: LogAction, explicit_flag: Optional[bool]) -> bool:
        """Determine whether a summary section should be shown.

        If the user set an explicit flag (True/False) via configure_summary,
        honour it.  Otherwise, auto-hide the section when the corresponding
        event action is "silent".
        """
        if explicit_flag is not None:
            return explicit_flag
        return action != "silent"

    def _group_patches_hierarchically(self, patches: List[PatchInfo]) -> Dict[str, Dict[str, List[PatchInfo]]]:
        """Group patches by package -> patch_name -> list of PatchInfo."""
        grouped: Dict[str, Dict[str, List[PatchInfo]]] = defaultdict(lambda: defaultdict(list))
        for info in patches:
            package = info.package or "other"
            patch_name = info.patch_name or "unknown"
            grouped[package][patch_name].append(info)
        return grouped

    def _format_hierarchical_section(self, title: str, patches: List[PatchInfo], prefix: str = "+") -> List[str]:
        """Format a section with hierarchical grouping."""
        lines = []
        grouped = self._group_patches_hierarchically(patches)

        lines.append(f"  {title}:")

        for package in sorted(grouped.keys()):
            patch_names = grouped[package]
            first_patch = next(iter(next(iter(patch_names.values()))))
            version_str = f" ({first_patch.extra_info})" if first_patch.extra_info else ""
            lines.append(f"    [{package}]{version_str}")

            for patch_name in sorted(patch_names.keys()):
                targets = patch_names[patch_name]
                lines.append(f"      {patch_name}:")
                for info in targets:
                    reason_str = f" ({info.reason})" if info.reason else ""
                    lines.append(f"        {prefix} {info.target}{reason_str}")

        return lines

    @staticmethod
    def _build_summary_banner(title: str) -> List[str]:
        """Build a more visible summary banner so patcher output stands out in noisy logs."""
        line = "=" * 78
        inner = f"=== {title} ==="
        return [line, inner, line]

    def flush_summary(self):
        """Emit buffered patch summary through the logger.

        The summary is emitted at ``self._summary.level`` (default INFO)
        through the Python logger, so it respects the logger level and
        handler configuration.  Sections are shown/hidden based on
        ``configure_summary()`` settings and event action configuration.
        """
        if not self._should_log():
            self._clear_buffers()
            return

        if not self._summary.enabled:
            self._clear_buffers()
            return

        # Snapshot and clear buffers
        applied = self._applied_patches[:]
        skipped = self._skipped_patches[:]
        failed = self._failed_patches[:]
        skipped_modules = self._skipped_modules[:]
        injected = self._injected_imports[:]
        replaced = self._replaced_imports[:]
        self._clear_buffers()

        # Determine section visibility
        show_applied = self._should_show_section(self._on_apply, self._summary.show_applied) and bool(applied)
        show_skipped = self._should_show_section(self._on_skip, self._summary.show_skipped) and bool(skipped)
        show_failed = self._should_show_section(self._on_fail, self._summary.show_failed) and bool(failed)
        show_imports = self._summary.show_import_events
        has_import_events = bool(skipped_modules) or bool(injected) or bool(replaced)

        if not show_applied and not show_skipped and not show_failed and not (show_imports and has_import_events):
            return

        # Build summary text
        lines = []
        separator = "=" * 78
        thin_sep = "-" * 78

        lines.append("")
        lines.extend(self._build_summary_banner("MX-DRIVING PATCHER SUMMARY"))

        if show_imports and skipped_modules:
            lines.append("  Skipped Modules:")
            for mod in skipped_modules:
                lines.append(f"    - {mod}")
            lines.append(thin_sep)

        if show_imports and injected:
            lines.append("  Injected Imports:")
            for imp in injected:
                lines.append(f"    + {imp}")
            lines.append(thin_sep)

        if show_imports and replaced:
            lines.append("  Replaced Imports:")
            for desc in replaced:
                lines.append(f"    * {desc}")
            lines.append(thin_sep)

        if show_applied:
            lines.extend(self._format_hierarchical_section("Applied Patches", applied, "+"))
            lines.append(thin_sep)

        if show_skipped:
            lines.extend(self._format_hierarchical_section("Skipped Patches", skipped, "~"))
            lines.append(thin_sep)

        if show_failed:
            lines.extend(self._format_hierarchical_section("Failed Patches", failed, "!"))
            lines.append(thin_sep)

        # Remove trailing separator if present
        if lines and lines[-1] == thin_sep:
            lines.pop()

        lines.append(separator)
        lines.append("")

        summary_text = "\n".join(lines)
        self._logger.log(self._summary.level, summary_text)

    def _clear_buffers(self):
        """Clear all summary buffers."""
        self._applied_patches.clear()
        self._skipped_patches.clear()
        self._failed_patches.clear()
        self._skipped_modules.clear()
        self._injected_imports.clear()
        self._replaced_imports.clear()

    # ─── Direct logging methods (backward compatible) ────────────────────

    def error(self, message: str):
        """Log an unexpected error using the on_error action."""
        self._handle(self._on_error, message)

    def debug(self, message: str):
        """Log a debug message (only on rank 0)."""
        if self._should_log():
            self._logger.debug(message)

    def info(self, message: str):
        """Log an info message (only on rank 0)."""
        if self._should_log():
            self._logger.info(message)

    def warning(self, message: str):
        """Log a warning message (only on rank 0)."""
        if self._should_log():
            self._logger.warning(message)


# Global singleton instance
patcher_logger = PatcherLogger()


def configure_patcher_logging(
    on_apply: Optional[LogAction] = None,
    on_skip: Optional[LogAction] = None,
    on_fail: Optional[LogAction] = None,
    on_error: Optional[LogAction] = None,
) -> PatcherLogger:
    """
    Configure the global patcher logger event actions.

    This is a convenience function for ``patcher_logger.configure(...)``.

    Args:
        on_apply: Action when patch is successfully applied.
        on_skip: Action when patch is skipped.
        on_fail: Action when patch application fails.
        on_error: Action when an unexpected error occurs.

    Returns:
        The configured patcher_logger instance.

    Example:
        from mx_driving.patcher import configure_patcher_logging

        # Production: only log important events
        configure_patcher_logging(
            on_apply="info",
            on_skip="silent",
            on_fail="warning",
            on_error="error",
        )

        # Development: verbose with strict error handling
        configure_patcher_logging(
            on_apply="info",
            on_skip="info",
            on_fail="exception",
            on_error="exception",
        )
    """
    return patcher_logger.configure(
        on_apply=on_apply,
        on_skip=on_skip,
        on_fail=on_fail,
        on_error=on_error,
    )


def set_patcher_log_level(level: int):
    """
    Set the minimum logging level for the patcher logger.

    This affects both immediate event messages and the summary output,
    since both go through the Python logger.

    Args:
        level: logging.DEBUG, logging.INFO, logging.WARNING, etc.
    """
    patcher_logger.set_level(level)
