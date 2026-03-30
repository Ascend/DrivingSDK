# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""Torch tensor operation patches for NPU optimization."""
from typing import List

from mx_driving.patcher.patch import AtomicPatch, BasePatch, Patch


class TensorIndex(Patch):
    """Tensor indexing optimization patch for torch."""

    name = "tensor_index"
    legacy_name = "index"
    target_module = "torch"

    @staticmethod
    def _runtime_check(self, indices, *_, **__) -> bool:
        import torch
        if not isinstance(indices, torch.Tensor) or indices.dtype != torch.bool or indices.dim() != 1:
            return False
        if self.dim() == 1:
            return True
        if self.dim() == 2 and self.shape[0] == indices.shape[0]:
            return True
        return False

    @staticmethod
    def _replacement(self, indices):
        import torch
        if self.dim() == 1:
            return torch.masked_select(self, indices)
        if self.dim() == 2 and self.shape[0] == indices.shape[0]:
            indices = indices.unsqueeze(1).expand(self.shape)
            return torch.masked_select(self, indices).view(-1, self.shape[1])
        return torch.masked_select(self, indices)

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "torch.Tensor.__getitem__",
                cls._replacement,
                runtime_check=cls._runtime_check,
            ),
        ]


class BatchMatmul(Patch):
    """Batch matmul optimization patch for torch."""

    name = "batch_matmul"
    legacy_name = "batch_matmul"
    target_module = "torch"

    @staticmethod
    def _check_shape_bmm(a, b) -> bool:
        if not hasattr(b, 'dim'):
            return False
        if not (a.dim() == b.dim() and 4 <= a.dim() <= 7):
            return False
        if not all(ad == bd or ad == 1 or bd == 1 for ad, bd in zip(a.shape[:-2], b.shape[:-2])):
            return False
        return a.shape[-2] == a.shape[-1] and a.shape[-2] == b.shape[-2] and b.shape[-1] == 1

    @staticmethod
    def _runtime_check(a, b, *_, **__) -> bool:
        return BatchMatmul._check_shape_bmm(a, b)

    @staticmethod
    def _replacement(a, b):
        from mx_driving import npu_batch_matmul
        return npu_batch_matmul(a, b)

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "torch.matmul",
                cls._replacement,
                runtime_check=cls._runtime_check,
            ),
            AtomicPatch(
                "torch.Tensor.matmul",
                cls._replacement,
                runtime_check=cls._runtime_check,
            ),
            AtomicPatch(
                "torch.Tensor.__matmul__",
                cls._replacement,
                runtime_check=cls._runtime_check,
            ),
        ]
