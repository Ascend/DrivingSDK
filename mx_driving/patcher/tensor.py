# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from types import ModuleType
from typing import Dict


# rewrite torch.Tensor.__getitem__ to support boolean indexing
def index(torch: ModuleType, options: Dict):
    fn = torch.Tensor.__getitem__

    def new_fn(self, indices):
        # check if indices is a boolean tensor
        if not isinstance(indices, torch.Tensor) or indices.dtype != torch.bool or indices.dim() != 1:
            return fn(self, indices)
        if self.dim() == 1:
            return torch.masked_select(self, indices)
        if self.dim() == 2 and self.shape[0] == indices.shape[0]:
            indices = indices.unsqueeze(1).expand(self.shape)
            return torch.masked_select(self, indices).view(-1, self.shape[1])
        return fn(self, indices)  # fallback to the original function

    torch.Tensor.__getitem__ = new_fn


def check_shape_bmm(a, b):
    if not (a.dim() == b.dim() and 4 <= a.dim() <= 7):
        return False

    if not all(ad == bd or ad == 1 or bd == 1 
               for ad, bd in zip(a.shape[:-2], b.shape[:-2])):
        return False

    return (a.shape[-2] == a.shape[-1] and 
            a.shape[-2] == b.shape[-2] and 
            b.shape[-1] == 1)


def batch_matmul(torch: ModuleType, options: Dict):
    from mx_driving import npu_batch_matmul

    def adjust_matrix(b):
        return b.view(*b.shape[:-2], 1, b.shape[-2]).contiguous()

    def create_wrapper(original_fn):
        def wrapper(a, b):
            if check_shape_bmm(a, b):
                return npu_batch_matmul(a, adjust_matrix(b))
            return original_fn(a, b)
        return wrapper

    torch.matmul = create_wrapper(torch.matmul)
    torch.Tensor.matmul = create_wrapper(torch.Tensor.matmul)
    torch.Tensor.__matmul__ = create_wrapper(torch.Tensor.__matmul__)
