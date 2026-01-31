# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
from types import ModuleType
from typing import Dict


# rewrite torch.Tensor.__getitem__ to support boolean indexing
def index(torch_mod: ModuleType, options: Dict):
    original_torch_function = torch_mod.Tensor.__torch_function__

    def new_torch_function(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func is torch_mod.Tensor.__getitem__:
            self, indices = args[0], args[1]

            if torch_mod.is_tensor(indices) and indices.dtype == torch_mod.bool and indices.dim() == 1:
                self_dim = self.dim()
                if self_dim == 1:
                    return torch_mod.masked_select(self, indices)
                if self_dim == 2 and self.shape[0] == indices.shape[0]:
                    expanded_indices = indices.unsqueeze(1).expand_as(self)
                    return torch_mod.masked_select(self, expanded_indices).view(-1, self.shape[1])

        return original_torch_function(func, types, args, kwargs)

    torch_mod.Tensor.__torch_function__ = classmethod(new_torch_function)


def check_shape_bmm(a, b):
    # skip a @ b which b in type numpy.ndarray
    if not hasattr(b, 'dim'):
        return False

    if not (a.dim() == b.dim() and 4 <= a.dim() <= 7):
        return False

    if not all(ad == bd or ad == 1 or bd == 1 
               for ad, bd in zip(a.shape[:-2], b.shape[:-2])):
        return False

    return (a.shape[-2] == a.shape[-1] and 
            a.shape[-2] == b.shape[-2] and 
            b.shape[-1] == 1)


def batch_matmul(torch: ModuleType, options: Dict):
    
    def create_wrapper(original_fn):
        def wrapper(a, b):
            if check_shape_bmm(a, b):
                return npu_batch_matmul(a, b)
            return original_fn(a, b)
        return wrapper
    
    matmul_not_found = True
    if hasattr(torch, "matmul"):
        from mx_driving import npu_batch_matmul

        torch.matmul = create_wrapper(torch.matmul)
        matmul_not_found = False
    
    if hasattr(torch.Tensor, "matmul") and hasattr(torch.Tensor, "__matmul__"):
        from mx_driving import npu_batch_matmul
        
        torch.Tensor.matmul = create_wrapper(torch.Tensor.matmul)
        torch.Tensor.__matmul__ = create_wrapper(torch.Tensor.__matmul__)
        matmul_not_found = False    
    
    if matmul_not_found:
        raise AttributeError("In torch, matmul or Tensor.matmul or Tensor.__matmul__ not found")
