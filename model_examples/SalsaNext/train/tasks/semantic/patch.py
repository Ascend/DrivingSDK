"""

MIT License

Copyright (c) 2018 Maxim Berman
Copyright (c) 2020 Tiago Cortinhal, George Tzelepis and Eren Erdal Aksoy


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

"""
from types import ModuleType
from typing import Dict

import torch
import torch_npu
from torch.autograd.function import Function

import mx_driving
from mx_driving.patcher import PatcherBuilder, Patch, numpy_type


class SparseSort(Function):
    @staticmethod
    def forward(ctx, inputs, dim=0, descending=False):
        outputs, index = torch.sort(inputs, dim=dim, descending=descending)
        ctx.save_for_backward(index)
        return outputs, index

    @staticmethod
    def backward(ctx, grad_outputs, grad_index):
        index = ctx.saved_tensors[0]
        mask = (grad_outputs != 0).nonzero().squeeze()
        index_nonzero = torch.index_select(index, 0, mask)
        grad_nonzero = torch.index_select(grad_outputs, 0, mask)
        grad_inputs = torch.zeros_like(grad_outputs).scatter(0, index_nonzero, grad_nonzero)

        return grad_inputs, None, None

sparse_sort = SparseSort.apply


def lovasz_softmax_loss(losses: ModuleType, options: Dict):
    Variable = losses.Variable

    def lovasz_softmax_flat(probas, labels, classes='present'):
        """
        Multi-class Lovasz-Softmax loss
        probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
        labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.
        C = probas.size(1)
        errors_sorted_list = []
        fg_sorted_list = []
        arange = torch.arange(labels.size(0), device=labels.device) + 1
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
        for c in class_to_sum:
            fg = (labels == c).float()  # foreground for class c
            if (classes == 'present' and fg.sum() == 0):
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (Variable(fg) - class_pred).abs()
            errors_sorted, perm = sparse_sort(errors, 0, True)
            fg_sorted = torch.index_select(fg, 0, perm.long())
            errors_sorted_list.append(errors_sorted)
            fg_sorted_list.append(fg_sorted)

        errors_sorted = torch.stack(errors_sorted_list, dim=0)
        fg_sorted = torch.stack(fg_sorted_list, dim=0)

        fg_cumsum = torch.cumsum(fg_sorted, dim=1)
        fg_sum = fg_cumsum[:, -1].unsqueeze(1)

        intersection = fg_sum - fg_cumsum
        union = intersection + arange.unsqueeze(0)
        weights = 1. - intersection / union
        if fg_cumsum.shape[1] > 1:  # cover 1-pixel case
            weights[:, 1:] = weights[:, 1:] - weights[:, 0:-1]

        loss = (errors_sorted * weights).sum(dim=1).mean(dim=0)

        return loss

    def flatten_probas(probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.dim() == 3:
            # assumes output of a sigmoid layer
            B, H, W = probas.size()
            probas = probas.view(B, 1, H, W)
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = (labels != ignore).nonzero().squeeze().long()
        vprobas = torch.index_select(probas, 0, valid)
        vlabels = torch.index_select(labels, 0, valid)
        return vprobas, vlabels

    if hasattr(losses, "lovasz_softmax_flat"):
        losses.lovasz_softmax_flat = lovasz_softmax_flat

    if hasattr(losses, "flatten_probas"):
        losses.flatten_probas = flatten_probas

salsa_next_patcher_builder = (
    PatcherBuilder()
    .add_module_patch("numpy", Patch(numpy_type))
    .add_module_patch("tasks.semantic.modules.Lovasz_Softmax", Patch(lovasz_softmax_loss))
)