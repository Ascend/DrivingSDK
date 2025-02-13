# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import sys
import time
from types import ModuleType
from typing import Dict


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
                sys.exit(0)
        self.call_hook("after_train_epoch")
        self._epoch += 1

    def run_epoch(self) -> None:
        self.runner.call_hook("before_train_epoch")
        self.runner.model.train()
        for idx, data_batch in enumerate(self.data_loader):
            self.run_iter(idx, data_batch)
            if self._iter == when_iter:
                sys.exit(0)

        self.runner.call_hook("after_train_epoch")
        self.runner._epoch += 1

    if hasattr(runner, "EpochBasedRunner"):
        runner.EpochBasedRunner.train = train
    elif hasattr(runner, "EpochBasedTrainLoop"):
        runner.EpochBasedTrainLoop.run_epoch = run_epoch
