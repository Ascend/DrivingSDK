# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from types import ModuleType
from typing import Dict


def ddp(mmcvparallel: ModuleType, options: Dict):
    if hasattr(mmcvparallel, "distributed"):
        import mmcv
        mmcvparallel.distributed.MMDistributedDataParallel = mmcv.device.npu.NPUDistributedDataParallel
