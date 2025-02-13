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
