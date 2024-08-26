import ads_c
import torch


class SortPairs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, keys_in, values_in, dim, descending=False):
        res = ads_c.npu_sort_pairs(keys_in, values_in, dim, descending)
        return res

sort_pairs = SortPairs.apply
