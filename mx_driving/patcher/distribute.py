# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from types import ModuleType
from typing import Dict


def ddp(module: ModuleType, options: Dict): 
    # For mmcv 1.x: module path is mmcv.parallel.distributed
    
    def _run_ddp_forward(self, *inputs, **kwargs):
        module_to_run = self.module

        if self.device_ids:
            inputs, kwargs = self.to_kwargs(  # type: ignore
                inputs, kwargs, self.device_ids[0])
            return module_to_run(*inputs[0], **kwargs[0])  # type: ignore
        else:
            return module_to_run(*inputs, **kwargs)
    
    
    if hasattr(module, "MMDistributedDataParallel"):
        import mmcv.device
        module.MMDistributedDataParallel._run_ddp_forward = _run_ddp_forward
        module.MMDistributedDataParallel = mmcv.device.npu.NPUDistributedDataParallel
    else:
        raise AttributeError("MMDistributedDataParallel not found")