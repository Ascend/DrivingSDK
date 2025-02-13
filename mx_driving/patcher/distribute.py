# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from types import ModuleType
from typing import Dict


def ddp(mmcvparraller: ModuleType, options: Dict):
    if hasattr(mmcvparraller, "distributed"):
        import mmcv
        mmcvparraller.distributed.MMDistributedDataParallel = mmcv.device.npu.NPUDistributedDataParallel