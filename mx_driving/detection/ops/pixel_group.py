import torch
import torch_npu
from torch.autograd import Function

import mx_driving._C


class PixelGroupFunction(Function):
    @staticmethod
    # pylint: disable=huawei-too-many-arguments
    def forward(
        ctx,
        score,
        mask,
        embedding,
        kernel_label,
        kernel_contour,
        kernel_region_num,
        distance_threshold,
    ):
        output = mx_driving._C.pixel_group(
            score,
            mask,
            embedding,
            kernel_label,
            kernel_contour,
            kernel_region_num,
            distance_threshold,
        )

        return output


pixel_group = PixelGroupFunction.apply
