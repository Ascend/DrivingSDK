# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Callable, List, Optional, Union

import torch
import torch_npu

from mmcv.runner import master_only
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class ProfilerHookNPU(Hook):
    """
    Profiling Hook NPU Version
    Example:
        dict(
            type='ProfilerHookNPU',
            by_epoch=True if runner['type'] == "EpochBasedRunner" else False,
            activities=['cpu', 'npu'],
            with_stack=True, # 采集torch op的函数调用栈的开关,会占用较多空间
            record_shapes=False, # 采集torch op的input shape和input type的开关
            profile_memory=True, # 采集memory相关数据的开关
            on_trace_ready="/your/path/",
            schedule={'wait': 10, 'warmup': 1, 'active': 1, 'repeat': 1, 'skip_first': 10},
            priority='NORMAL',
        )
    """

    def __init__(self,
                 by_epoch: bool = True,
                 profile_iters: int = 1,
                 activities: tuple[str] = ('cpu', 'npu'),
                 schedule: Optional[dict] = None,
                 on_trace_ready: Optional[Union[str, dict]] = None,
                 record_shapes: bool = False,
                 profile_memory: bool = False,
                 with_stack: bool = False,
                 with_flops: bool = False,
                 json_trace_path: Optional[str] = None) -> None:
        try:
            from torch_npu import profiler  # torch version >= 1.8.1
        except ImportError as e:
            raise ImportError('Import torch_npu profiler error!') from e

        if not isinstance(by_epoch, bool):
            raise ValueError(
                '``by_epoch`` should be a boolean.')
        self.by_epoch = by_epoch

        if profile_iters < 1:
            raise ValueError('profile_iters should be greater than 0, but got '
                             f'{profile_iters}')
        self.profile_iters = profile_iters

        if not isinstance(activities, tuple):
            raise ValueError(
                f'activities should be tuple, but got {type(activities)}')
        self.activities = []
        for activity in activities:
            activity = activity.lower()
            if activity == 'cpu':
                self.activities.append(profiler.ProfilerActivity.CPU)
            elif activity == 'npu':
                self.activities.append(profiler.ProfilerActivity.NPU)
            else:
                raise ValueError(
                    f'activity should be "cpu" or "npu", but got {activity}')

        if schedule is not None:
            self.schedule = profiler.schedule(**schedule)
        else:
            self.schedule = None

        self.on_trace_ready = torch_npu.profiler.tensorboard_trace_handler(on_trace_ready)
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.json_trace_path = json_trace_path

    @master_only
    def before_run(self, runner):
        if self.by_epoch and runner.max_epochs < self.profile_iters:
            raise ValueError('self.profile_iters should not be greater than '
                             f'{runner.max_epochs}')

        if not self.by_epoch and runner.max_iters < self.profile_iters:
            raise ValueError('self.profile_iters should not be greater than '
                             f'{runner.max_iters}')

        if callable(self.on_trace_ready):  # handler
            _on_trace_ready = self.on_trace_ready
        elif isinstance(self.on_trace_ready, dict):  # config of handler
            trace_cfg = self.on_trace_ready.copy()
            trace_type = trace_cfg.pop('type')  # log_trace handler
            if trace_type == 'log_trace':

                def _log_handler(prof):
                    print(prof.key_averages().table(**trace_cfg))

                _on_trace_ready = _log_handler
            elif trace_type == 'tb_trace':  # tensorboard_trace handler
                try:
                    import torch_tb_profiler  # noqa: F401
                except ImportError as e:
                    raise ImportError('please run "pip install '
                                      'torch-tb-profiler" to install '
                                      'torch_tb_profiler') from e
                if 'dir_name' not in trace_cfg:
                    trace_cfg['dir_name'] = osp.join(runner.work_dir,
                                                     'tf_tracing_logs')
                elif not osp.isabs(trace_cfg['dir_name']):
                    trace_cfg['dir_name'] = osp.join(runner.work_dir,
                                                     trace_cfg['dir_name'])
                runner.logger.info(
                    'tracing files of ProfilerHook will be saved to '
                    f"{trace_cfg['dir_name']}.")
                _on_trace_ready = torch.profiler.tensorboard_trace_handler(
                    **trace_cfg)
            else:
                raise ValueError('trace_type should be "log_trace" or '
                                 f'"tb_trace", but got {trace_type}')
        elif self.on_trace_ready is None:
            _on_trace_ready = None  # type: ignore
        else:
            raise ValueError('on_trace_ready should be handler, dict or None, '
                             f'but got {type(self.on_trace_ready)}')

        if self.by_epoch and runner.max_epochs > 1:
            warnings.warn(f'profiler will profile {runner.max_epochs} epochs '
                          'instead of 1 epoch. Since profiler will slow down '
                          'the training, it is recommended to train 1 epoch '
                          'with ProfilerHook and adjust your setting according'
                          ' to the profiler summary. During normal training '
                          '(epoch > 1), you may disable the ProfilerHook.')

        experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2)
        
        self.profiler = torch_npu.profiler.profile(
            activities=self.activities,
            schedule=self.schedule,
            on_trace_ready=_on_trace_ready,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            experimental_config=experimental_config,
            with_flops=self.with_flops)

        self.profiler.__enter__()
        runner.logger.info('npu profiler is profiling...')

    @master_only
    def after_train_epoch(self, runner):
        if self.by_epoch and runner.epoch == self.profile_iters - 1:
            runner.logger.info('profiler may take a few minutes...')
            self.profiler.__exit__(None, None, None)
            if self.json_trace_path is not None:
                self.profiler.export_chrome_trace(self.json_trace_path)

    @master_only
    def after_train_iter(self, runner):
        self.profiler.step()
        if not self.by_epoch and runner.iter == self.profile_iters - 1:
            runner.logger.info('profiler may take a few minutes...')
            self.profiler.__exit__(None, None, None)
            if self.json_trace_path is not None:
                self.profiler.export_chrome_trace(self.json_trace_path)
