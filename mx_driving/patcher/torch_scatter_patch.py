# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""
torch scatter patch
"""

from types import ModuleType
from typing import Dict, Optional, Tuple
import importlib


def scatter(ts: ModuleType, _: Dict):
    """
    patch scatter_sum, scatter_mean and scatter_max to torch-scatter
    """
    torch = importlib.import_module("torch")
    mx_driving = importlib.import_module("mx_driving")
    sc = importlib.import_module(f"{ts.__name__}.scatter")

    def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                    out: Optional[torch.Tensor] = None,
                    dim_size: Optional[int] = None) -> torch.Tensor:
        return mx_driving.scatter_add(src.float(), index.to(torch.int32),
                                      out, dim, dim_size)

    def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                     out: Optional[torch.Tensor] = None,
                     dim_size: Optional[int] = None) -> torch.Tensor:
        return mx_driving.scatter_mean(src.float(), index.to(torch.int32),
                                       out, dim, dim_size)

    def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                    out: Optional[torch.Tensor] = None,
                    dim_size: Optional[int] = None,
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return mx_driving.scatter_max(src.float(), index.to(torch.int32), out)

    sc.scatter_sum = scatter_sum
    sc.scatter_mean = scatter_mean
    sc.scatter_max = scatter_max

    ts.scatter_sum = scatter_sum
    ts.scatter_mean = scatter_mean
    ts.scatter_max = scatter_max
