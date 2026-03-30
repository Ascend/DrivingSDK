# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""Torch scatter operations patches for NPU."""
from typing import List

from mx_driving.patcher.patch import AtomicPatch, BasePatch, Patch


class TorchScatter(Patch):
    """Torch scatter operations patch (scatter_sum, scatter_mean, scatter_max)."""

    name = "torch_scatter"
    legacy_name = "scatter"
    target_module = "torch_scatter"

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        # replacement_wrapper: convert src to float and index to int32
        def scatter_wrapper(npu_func):
            def wrapper(src, index, dim=-1, out=None, dim_size=None):
                import torch
                return npu_func(src.float(), index.to(torch.int32), out, dim, dim_size)
            return wrapper

        return [
            # Patch torch_scatter.scatter submodule
            AtomicPatch(
                "torch_scatter.scatter.scatter_sum",
                "mx_driving.scatter_add",
                replacement_wrapper=scatter_wrapper,
            ),
            AtomicPatch(
                "torch_scatter.scatter.scatter_mean",
                "mx_driving.scatter_mean",
                replacement_wrapper=scatter_wrapper,
            ),
            AtomicPatch(
                "torch_scatter.scatter.scatter_max",
                "mx_driving.scatter_max",
                replacement_wrapper=scatter_wrapper,
            ),
            # Patch torch_scatter top-level module (re-exports)
            AtomicPatch(
                "torch_scatter.scatter_sum",
                "mx_driving.scatter_add",
                replacement_wrapper=scatter_wrapper,
            ),
            AtomicPatch(
                "torch_scatter.scatter_mean",
                "mx_driving.scatter_mean",
                replacement_wrapper=scatter_wrapper,
            ),
            AtomicPatch(
                "torch_scatter.scatter_max",
                "mx_driving.scatter_max",
                replacement_wrapper=scatter_wrapper,
            ),
        ]
