# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from types import ModuleType
from typing import Dict


def ddp(mmcvparallel: ModuleType, options: Dict):
    if hasattr(mmcvparallel, "distributed"):
        import mmcv
        mmcvparallel.distributed.MMDistributedDataParallel = mmcv.device.npu.NPUDistributedDataParallel


def ddp_forward(mmcvparallel: ModuleType, options: Dict):
    def new_forward(self, *inputs, **kwargs):
        module_to_run = self.module

        if self.device_ids:
            inputs, kwargs = self.to_kwargs(  # type: ignore
                inputs, kwargs, self.device_ids[0])
            return module_to_run(*inputs[0], **kwargs[0])  # type: ignore
        else:
            return module_to_run(*inputs, **kwargs)
    
    if hasattr(mmcvparallel, "distributed"):
        mmcvparallel.distributed.MMDistributedDataParallel._run_ddp_forward = new_forward