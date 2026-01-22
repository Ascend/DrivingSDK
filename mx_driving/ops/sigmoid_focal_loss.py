from typing import Optional
import torch
import torch.nn.functional as F
import torch_npu
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import mx_driving._C

class SigmoidFocalLossFunction(Function):
    @staticmethod
    def forward(ctx,
                input: torch.Tensor,
                target: torch.Tensor,
                gamma: float = 2.0,
                alpha: float = 0.25,
                weight: Optional[torch.Tensor] = None,
                reduction: str = 'mean') -> torch.Tensor:

        if target.dtype != torch.long:
            raise Exception("Tensor target's dtype should be torch.long")
        if input.dim() != 2:
            raise Exception("Tensor input's dimension should be 2")
        if target.dim() != 1:
            raise Exception("Tensor target's dimension should be 1")
        if input.size(0) != target.size(0):
            raise Exception("input.size(0) should equal to target.size(0)")

        if weight is None:
            weight = input.new_empty(0)
        else:
            if weight.dim() != 1:
                raise Exception("Tensor weight's dimension should be 1")
            if input.size(1) != weight.size(0):
                raise Exception("input.size(1) should equal to weight.size(0)")

        ctx.reduction_dict = {'none': 0, 'mean': 1, 'sum': 2}
        if reduction not in ctx.reduction_dict.keys():
            raise Exception("reduction should be 'none', 'mean', or 'sum'")

        ctx.gamma = float(gamma)
        ctx.alpha = float(alpha)
        ctx.reduction = ctx.reduction_dict[reduction]

        output = input.new_zeros(input.size())
        mx_driving._C.sigmoid_focal_loss(
            input, target, weight, output, ctx.gamma, ctx.alpha)
        if ctx.reduction == ctx.reduction_dict['mean']:
            output = output.sum() / input.size(0)
        elif ctx.reduction == ctx.reduction_dict['sum']:
            output = output.sum()
        ctx.save_for_backward(input, target, weight)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        input, target, weight = ctx.saved_tensors

        grad_input = input.new_zeros(input.size())

        mx_driving._C.sigmoid_focal_loss_backward(
            input, target, weight, grad_input, ctx.gamma, ctx.alpha)

        grad_input *= grad_output
        if ctx.reduction == ctx.reduction_dict['mean']:
            grad_input /= input.size(0)
        return grad_input, None, None, None, None, None

sigmoid_focal_loss = SigmoidFocalLossFunction.apply