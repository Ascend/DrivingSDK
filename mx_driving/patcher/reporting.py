# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PatchStatus(Enum):
    APPLIED = "applied"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class PatchResult:
    status: PatchStatus
    name: str
    module: str
    reason: Optional[str] = None
