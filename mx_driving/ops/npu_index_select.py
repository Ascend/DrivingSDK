import torch
import torch_npu
from torch.autograd import Function

import mx_driving._C


class IndexSelectFunction(Function):
    """Returns a new tensor which indexes the input tensor along dimension dim using the entries in index"""

    @staticmethod
    def forward(ctx, feature: torch.Tensor, dim: int, index: torch.Tensor):
        """
        Args:
            feature (Tensor): the input tensor.
            dim (int): the dimension in which we index
            index (IntTensor or LongTensor): the 1-D tensor containing the indices to index

        Returns:
            out (Tensor): the output tensor.
        """
        input_dim = feature.size()[0]
        output = mx_driving._C.index_select(feature, dim, index)

        ctx.for_backwards = (input_dim, dim, index)
        return output

    @staticmethod
    def backward(ctx, source: torch.Tensor):
        """
        Args:
            source (Tensor): tensor of the gradients of the output from forward.

        Returns:
            Tensor: gradient of the input.
        """
        if torch.numel(source) == 0:
            raise Exception("Error! tensor of the gradients can not be a empty Tensor.\n")

        input_dim, dim, index = ctx.for_backwards
        grad_input = mx_driving._C.index_select_backward(input_dim, dim, index, source)

        return grad_input, None, None


def npu_index_select(feature: torch.Tensor, dim: int, index: torch.Tensor):

    return IndexSelectFunction.apply(feature, dim, index)
