import torch

import mx_driving._C


class SortPairs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, keys_in, values_in, dim, descending=False):
        res = mx_driving._C.npu_sort_pairs(keys_in, values_in, dim, descending)
        return res


sort_pairs = SortPairs.apply
