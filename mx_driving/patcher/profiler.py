# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import time
from types import ModuleType
from typing import Dict


def profiler(runner: ModuleType, options: Dict):
    import torch_npu

    path = options["profiling_path"]
    level = options["profiling_level"]

    activities = (
        [torch_npu.profiler.ProfilerActivity.NPU]
        if level == 0
        else [
            torch_npu.profiler.ProfilerActivity.NPU,
            torch_npu.profiler.ProfilerActivity.CPU,
        ]
    )
    profiler_level = torch_npu.profiler.ProfilerLevel.Level0 if level == 0 else torch_npu.profiler.ProfilerLevel.Level1

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(data_loader)
        self.call_hook("before_train_epoch")
        time.sleep(2)
        with torch_npu.profiler.profile(
            activities=activities,
            with_stack=level == 2,
            record_shapes=level > 0,
            profile_memory=level == 2,
            schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=1, repeat=1, skip_first=20),
            experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=profiler_level),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(path),
        ) as prof:
            for i, data_batch in enumerate(data_loader):
                self.data_batch = data_batch
                self._inner_iter = i
                self.call_hook("before_train_iter")
                self.run_iter(data_batch, train_mode=True, **kwargs)
                self.call_hook("after_train_iter")
                del self.data_batch
                self._iter += 1
                prof.step()
        self.call_hook("after_train_epoch")
        self._epoch += 1

    def run_epoch(self) -> None:
        self.runner.call_hook("before_train_epoch")
        self.runner.model.train()
        with torch_npu.profiler.profile(
            activities=activities,
            with_stack=level == 2,
            record_shapes=level > 0,
            profile_memory=level == 2,
            schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=1, repeat=1, skip_first=20),
            experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=profiler_level),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(path),
        ) as prof:
            for idx, data_batch in enumerate(self.data_loader):
                self.run_iter(idx, data_batch)
                prof.step()

        self.runner.call_hook("after_train_epoch")
        self.runner._epoch += 1
        
    def run(self) -> None:
        self.runner.call_hook('before_train')
        self.runner.call_hook('before_train_epoch')
        if self._iter > 0:
            print_log(
                f'Advance dataloader {self._iter} steps to skip data '
                'that has already been trained',
                logger='current',
                level=logging.WARNING)
            for _ in range(self._iter):
                next(self.dataloader_iterator)
        with torch_npu.profiler.profile(
            activities=activities,
            with_stack=level == 2,
            record_shapes=level > 0,
            profile_memory=level == 2,
            schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=1, repeat=1, skip_first=20),
            experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=profiler_level),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(path),
        ) as prof:
            while self._iter < self._max_iters and not self.stop_training:
                self.runner.model.train()

                data_batch = next(self.dataloader_iterator)
                self.run_iter(data_batch)
                prof.step()

                self._decide_current_val_interval()
                if (self.runner.val_loop is not None
                        and self._iter >= self.val_begin
                        and (self._iter % self.val_interval == 0
                            or self._iter == self._max_iters)):
                    self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
        return self.runner.model

    if hasattr(runner, "EpochBasedRunner"):
        runner.EpochBasedRunner.train = train
    if hasattr(runner, "EpochBasedTrainLoop"):
        runner.EpochBasedTrainLoop.run_epoch = run_epoch
    if hasattr(runner, "IterBasedTrainLoop"):
        runner.IterBasedTrainLoop.run = run
