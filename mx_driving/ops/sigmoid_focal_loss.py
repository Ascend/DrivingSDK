# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import mx_driving._C

device_name = torch.npu.get_device_name()


class SigmoidFocalLossFunction(Function):
    @staticmethod
    def forward(
        ctx,
        logit: torch.Tensor,
        target: torch.Tensor,
        gamma: float = 2.0,
        alpha: float = 0.25,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        if target.dtype != torch.long:
            raise Exception("Tensor target's dtype should be torch.long")  # pylint: disable=broad-exception-raised
        if logit.dim() != 2:
            raise Exception("Tensor logit's dimension should be 2")  # pylint: disable=broad-exception-raised
        if target.dim() != 1:
            raise Exception("Tensor target's dimension should be 1")  # pylint: disable=broad-exception-raised
        if logit.size(0) != target.size(0):
            raise Exception("logit.size(0) should equal to target.size(0)")  # pylint: disable=broad-exception-raised

        if weight is None:
            weight = logit.new_empty(0)
        else:
            if weight.dim() != 1:
                raise Exception("Tensor weight's dimension should be 1")  # pylint: disable=broad-exception-raised
            if logit.size(1) != weight.size(0):
                raise Exception("logit.size(1) should equal to weight.size(0)")  # pylint: disable=broad-exception-raised

        ctx.reduction_dict = {'none': 0, 'mean': 1, 'sum': 2}
        if reduction not in ctx.reduction_dict.keys():
            raise Exception("reduction should be 'none', 'mean', or 'sum'")  # pylint: disable=broad-exception-raised

        ctx.gamma = float(gamma)
        ctx.alpha = float(alpha)
        ctx.reduction = ctx.reduction_dict[reduction]

        output = logit.new_zeros(logit.size())
        if 'Ascend950' in device_name:
            mx_driving._C.sigmoid_focal_loss(logit, target, weight, output, ctx.gamma, ctx.alpha)
        else:
            mx_driving._C.sigmoid_focal_loss_cann(logit, target, weight, output, ctx.gamma, ctx.alpha)
        if ctx.reduction == ctx.reduction_dict['mean']:
            output = output.sum() / logit.size(0)
        elif ctx.reduction == ctx.reduction_dict['sum']:
            output = output.sum()
        ctx.save_for_backward(logit, target, weight)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        logit, target, weight = ctx.saved_tensors
        grad_logit = logit.new_zeros(logit.size())
        if 'Ascend950' in device_name:
            mx_driving._C.sigmoid_focal_loss_backward(logit, target, weight, grad_logit, ctx.gamma, ctx.alpha)
        else:
            mx_driving._C.sigmoid_focal_loss_backward_cann(logit, target, weight, grad_logit, ctx.gamma, ctx.alpha)
        grad_logit *= grad_output
        if ctx.reduction == ctx.reduction_dict['mean']:
            grad_logit /= logit.size(0)
        return grad_logit, None, None, None, None, None


sigmoid_focal_loss = SigmoidFocalLossFunction.apply
