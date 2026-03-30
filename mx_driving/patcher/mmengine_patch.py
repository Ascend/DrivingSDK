# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
MMEngine patches for NPU adaptation.

Provides:
- OptimizerWrapper: mmengine optimizer wrapper with fused gradient clipping
- Training loop patches with profiling and early stopping support
"""
import importlib
import re
from typing import Dict, List

from mx_driving.patcher.patcher_logger import patcher_logger
from mx_driving.patcher.patch import (
    AtomicPatch,
    BasePatch,
    LegacyPatch,
    Patch,
    mmcv_version,
)


# =============================================================================
# Optimizer Wrapper (mmengine)
# =============================================================================

class OptimizerWrapper(Patch):
    """
    Optimizer wrapper patch for mmengine with gradient clipping support.

    Only applies when mmcv 2.x is detected (mmengine exists).
    """

    name = "optimizer_wrapper"
    legacy_name = "optimizer_wrapper"
    target_module = "mmengine"

    @staticmethod
    def _wrap_init(original):
        """Wrap OptimWrapper.__init__ to add fused gradient clipping."""
        def _get_clip_func(optimizer):
            def clip_func(params, **kwargs):
                return optimizer.clip_grad_norm_fused_(**kwargs)
            return clip_func

        def new_init(self, *args, **kwargs):
            original(self, *args, **kwargs)
            self.clip_grads = _get_clip_func(self.optimizer)

        return new_init

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "mmengine.optim.optimizer.optimizer_wrapper.OptimWrapper.__init__",
                target_wrapper=cls._wrap_init,
                precheck=lambda: mmcv_version.is_v2x,
            ),
        ]


# =============================================================================
# Training loop patches (mmengine, for profiling/brake)
# =============================================================================

def _parse_profiler_options(options: Dict):
    import torch_npu

    path = options["profiling_path"]
    level = options["profiling_level"]

    if bool(re.search(r'[ +#%&{}\<>*?/$!\'":@`|;=]', path)):
        patcher_logger.warning("profiling path contains illegal character")

    if level < 0 or level > 2:
        raise ValueError("valid profiling levels are integers within range [0, 2]")

    step_ctrl = options.get('step_ctrl', (1, 1, 1, 1, 20))

    activities = (
        [torch_npu.profiler.ProfilerActivity.NPU]
        if level == 0
        else [torch_npu.profiler.ProfilerActivity.NPU, torch_npu.profiler.ProfilerActivity.CPU]
    )
    profiler_level = torch_npu.profiler.ProfilerLevel.Level0 if level == 0 else torch_npu.profiler.ProfilerLevel.Level1
    return path, level, activities, profiler_level, step_ctrl


def build_mmengine_epoch_train_loop_patch(options: Dict) -> LegacyPatch:
    def _apply(module, _options):
        import sys
        import torch_npu

        enable_profiler = bool(options.get("enable_profiler"))
        enable_brake = bool(options.get("enable_brake"))
        if enable_profiler:
            path, level, activities, profiler_level, step_ctrl = _parse_profiler_options(options)
            wait, warmup, active, repeat, skip_first = step_ctrl
        if enable_brake:
            brake_step = options.get("brake_step")

        def run_epoch(self):
            self.runner.call_hook("before_train_epoch")
            self.runner.model.train()

            if enable_profiler:
                with torch_npu.profiler.profile(
                    activities=activities,
                    with_stack=level == 2,
                    record_shapes=level > 0,
                    profile_memory=level == 2,
                    schedule=torch_npu.profiler.schedule(wait, warmup, active, repeat, skip_first),
                    experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=profiler_level),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(path),
                ) as prof:
                    for idx, data_batch in enumerate(self.dataloader):
                        self.run_iter(idx, data_batch)
                        prof.step()
                        if enable_brake and self._iter == brake_step:
                            sys.exit(0)
            else:
                for idx, data_batch in enumerate(self.dataloader):
                    self.run_iter(idx, data_batch)
                    if enable_brake and self._iter == brake_step:
                        sys.exit(0)

            self.runner.call_hook("after_train_epoch")
            self._epoch += 1

        importlib.import_module(f"{module.__name__}.runner")
        module.runner.EpochBasedTrainLoop.run_epoch = run_epoch

    return LegacyPatch(_apply, target_module="mmengine")


def build_mmengine_iter_train_loop_patch(options: Dict) -> LegacyPatch:
    def _apply(module, _options):
        import sys
        import torch_npu

        enable_profiler = bool(options.get("enable_profiler"))
        enable_brake = bool(options.get("enable_brake"))
        if enable_profiler:
            path, level, activities, profiler_level, step_ctrl = _parse_profiler_options(options)
            wait, warmup, active, repeat, skip_first = step_ctrl
        if enable_brake:
            brake_step = options.get("brake_step")

        print_log = module.logging.print_log
        logging = module.logging

        def run(self):
            self.runner.call_hook("before_train")
            self.runner.call_hook("before_train_epoch")
            if self._iter > 0:
                print_log(
                    f"Advance dataloader {self._iter} steps to skip data that has already been trained",
                    logger="current",
                    level=logging.WARNING,
                )
                for _ in range(self._iter):
                    next(self.dataloader_iterator)

            if enable_profiler:
                with torch_npu.profiler.profile(
                    activities=activities,
                    with_stack=level == 2,
                    record_shapes=level > 0,
                    profile_memory=level == 2,
                    schedule=torch_npu.profiler.schedule(wait, warmup, active, repeat, skip_first),
                    experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=profiler_level),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(path),
                ) as prof:
                    while self._iter < self._max_iters and not self.stop_training:
                        self.runner.model.train()
                        data_batch = next(self.dataloader_iterator)
                        self.run_iter(data_batch)
                        prof.step()
                        if enable_brake and self._iter == brake_step:
                            sys.exit(0)
                        self._decide_current_val_interval()
                        if (
                            self.runner.val_loop is not None
                            and self._iter >= self.val_begin
                            and (self._iter % self.val_interval == 0 or self._iter == self._max_iters)
                        ):
                            self.runner.val_loop.run()
            else:
                while self._iter < self._max_iters and not self.stop_training:
                    self.runner.model.train()
                    data_batch = next(self.dataloader_iterator)
                    self.run_iter(data_batch)
                    if enable_brake and self._iter == brake_step:
                        sys.exit(0)
                    self._decide_current_val_interval()
                    if (
                        self.runner.val_loop is not None
                        and self._iter >= self.val_begin
                        and (self._iter % self.val_interval == 0 or self._iter == self._max_iters)
                    ):
                        self.runner.val_loop.run()

            self.runner.call_hook("after_train_epoch")
            self.runner.call_hook("after_train")
            return self.runner.model

        importlib.import_module(f"{module.__name__}.runner")
        module.runner.IterBasedTrainLoop.run = run

    return LegacyPatch(_apply, target_module="mmengine")
