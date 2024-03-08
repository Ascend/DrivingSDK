from typing import Optional, List
import torch
from torch import Tensor
import torch.onnx.symbolic_helper as sym_help
import ads.common


class NPUAddCustomOP(torch.autograd.Function):


    @staticmethod
    def forward(ctx, *args, **kwargs):
        return ads.common.npu_ads_add(*args, **kwargs)

    @staticmethod
    def symbolic(g, tensor1: Tensor, tensor2: Tensor):
        return g.op("npu::AddCustom", tensor1, tensor2)