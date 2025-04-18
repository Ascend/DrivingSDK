# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import sys
import time
import warnings
from types import ModuleType
from typing import Dict, Optional, Union, List, Tuple


def brake(runner: ModuleType, options: Dict):
    when_iter = options["when_iter"]
    if not isinstance(when_iter, int):
        raise ValueError(f"when_iter must be an integer, but got {type(when_iter)}")

    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = "train"
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(data_loader)
        self.call_hook("before_train_epoch")
        time.sleep(2)

        for i, data_batch in enumerate(data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            self.call_hook("before_train_iter")
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook("after_train_iter")
            del self.data_batch
            self._iter += 1
            if self._iter == when_iter:
                # pylint: disable=avoid-using-exit
                sys.exit(0)
        self.call_hook("after_train_epoch")
        self._epoch += 1

    def run_epoch(self) -> None:
        self.runner.call_hook("before_train_epoch")
        self.runner.model.train()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)
            if self._iter == when_iter:
                # pylint: disable=avoid-using-exit
                sys.exit(0)

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
        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()

            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)
            if self._iter == when_iter:
                # pylint: disable=avoid-using-exit
                sys.exit(0)

            self._decide_current_val_interval()
            # pylint: disable=too-many-boolean-expressions
            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and (self._iter % self.val_interval == 0
                        or self._iter == self._max_iters)):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
        return self.runner.model

    try:
        IterLoader = runner.iter_based_runner.IterLoader
        get_host_info = runner.iter_based_runner.get_host_info
        DataLoader = runner.iter_based_runner.DataLoader
    except AttributeError:
        DataLoader = None

    def run_iter(self,
                 data_loaders: List[DataLoader],
                 workflow: List[Tuple[str, int]],
                 max_iters: Optional[int] = None,
                 **kwargs) -> None:
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
        """
        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        self.call_hook('before_run')

        iter_loaders = [IterLoader(x) for x in data_loaders]

        self.call_hook('before_epoch')

        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    iter_runner(iter_loaders[i], **kwargs)
                    if self._iter == when_iter:
                        # pylint: disable=avoid-using-exit
                        sys.exit(0)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')

    if hasattr(runner, "EpochBasedRunner"):
        runner.EpochBasedRunner.train = train
    if hasattr(runner, "EpochBasedTrainLoop"):
        runner.EpochBasedTrainLoop.run_epoch = run_epoch
    if hasattr(runner, "IterBasedTrainLoop"):
        runner.IterBasedTrainLoop.run = run
    if hasattr(runner, "IterBasedRunner"):
        runner.IterBasedRunner.run = run_iter
