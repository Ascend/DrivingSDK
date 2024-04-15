import numpy as np
import torch
from torch.autograd import Function
from torch.nn import Module

import torch_npu
import ads_c


class AdsGroupPoints(Function):
    """Group feature with given index."""

    @staticmethod
    def forward(
            ctx,
            features: torch.Tensor,
            indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (Tensor): Tensor of features to group, input shape is
                (B, C, N).
            indices (Tensor):  The indices of features to group with, input
                shape is (B, npoint, nsample).

        Returns:
            Tensor: Grouped features, the shape is (B, C, npoint, nsample)
        """
        features = features.contiguous()
        indices = indices.contiguous()

        B, nfeatures, nsample = indices.size()
        _, C, N = features.size()

        if features.device.type != "npu":
            raise ValueError('The device is not npu!')
        output = ads_c.group_points(
            features,
            indices,
            B,
            C,
            N,
            nfeatures,
            nsample)

        ctx.for_backwards = (indices, N)
        return output


npu_group_points = AdsGroupPoints.apply