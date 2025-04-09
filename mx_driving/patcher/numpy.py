# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from types import ModuleType
from typing import Dict


def numpy_type(np: ModuleType, options: Dict):
    if not hasattr(np, "bool"):
        np.bool = bool

    if not hasattr(np, "float"):
        np.float = float
