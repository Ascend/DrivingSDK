# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.

import os
import sys
from types import ModuleType
from typing import Dict
import torch
import mx_driving
from mx_driving.patcher import PatcherBuilder, Patch
from mx_driving.patcher import numpy_type
import nuscenes

sys.path.append("..")


def data_classess(models: ModuleType, options: Dict):
    from typing import List, Dict
    from nuscenes.eval.detection.constants import DETECTION_NAMES

    def __init__(self, class_range: Dict[str, int], dist_fcn: str,
                 dist_ths: List[float], dist_th_tp: float, min_recall: float,
                 min_precision: float, max_boxes_per_sample: int,
                 mean_ap_weight: int):

        assert set(class_range.keys()) == set(
            DETECTION_NAMES), "Class count mismatch."
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

        self.class_names = list(self.class_range.keys())

    if hasattr(models, "DetectionConfig"):
        models.DetectionConfig.__init__ = __init__


def data_parallel(target: ModuleType, options: dict):
    from typing import Any
    from itertools import chain

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        with torch.autograd.profiler.record_function("DataParallel.forward"):
            if not self.device_ids:
                return self.module(*inputs, **kwargs)

            for t in chain(self.module.parameters(), self.module.buffers()):
                if t.device != t.device:
                    raise RuntimeError(
                        "module must have its parameters and buffers "
                        f"on device {self.src_device_obj} (device_ids[0]) but found one of "
                        f"them on device: {t.device}")

            inputs, module_kwargs = self.scatter(inputs, kwargs,
                                                 self.device_ids)
            # for forward function without any inputs, empty list and dict will be created
            # so the module can be executed on one device which is the first one in device_ids
            if not inputs and not module_kwargs:
                inputs = ((), )
                module_kwargs = ({}, )

            if len(self.device_ids) == 1:
                return self.module(*inputs[0], **module_kwargs[0])
            replicas = self.replicate(self.module,
                                      self.device_ids[:len(inputs)])
            outputs = self.parallel_apply(replicas, inputs, module_kwargs)
            return self.gather(outputs, self.output_device)

    if hasattr(target, "DataParallel"):
        target.DataParallel.forward = forward


def focal_loss(target: ModuleType, options: dict):
    from torch.autograd.function import once_differentiable
    from typing import Optional, Union
    import mx_driving._C

    @staticmethod
    def forward(ctx,
                input: torch.Tensor,
                target: Union[torch.LongTensor, torch.cuda.LongTensor],
                gamma: float = 2.0,
                alpha: float = 0.25,
                weight: Optional[torch.Tensor] = None,
                reduction: str = 'mean') -> torch.Tensor:

        assert target.dtype == torch.long
        assert input.dim() == 2
        assert target.dim() == 1
        assert input.size(0) == target.size(0)
        if weight is None:
            weight = input.new_empty(0)
        else:
            assert weight.dim() == 1
            assert input.size(1) == weight.size(0)
        ctx.reduction_dict = {'none': 0, 'mean': 1, 'sum': 2}
        assert reduction in ctx.reduction_dict.keys()

        ctx.gamma = float(gamma)
        ctx.alpha = float(alpha)
        ctx.reduction = ctx.reduction_dict[reduction]

        output = input.new_zeros(input.size())

        mx_driving._C.sigmoid_focal_loss(input, target, weight, output,
                                         ctx.gamma, ctx.alpha)
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

        mx_driving._C.sigmoid_focal_loss_backward(input, target, weight,
                                                  grad_input, ctx.gamma,
                                                  ctx.alpha)

        grad_input *= grad_output
        if ctx.reduction == ctx.reduction_dict['mean']:
            grad_input /= input.size(0)
        return grad_input, None, None, None, None, None

    if hasattr(target, "SigmoidFocalLossFunction"):
        target.SigmoidFocalLossFunction.forward = forward
        target.SigmoidFocalLossFunction.backward = backward


def modulated_deform_conv2d_(target: ModuleType, options: dict):
    from mmcv.ops.modulated_deform_conv import modulated_deform_conv2d

    def forward_1(self, x: torch.Tensor, offset: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
        return modulated_deform_conv2d(x, offset, mask, self.weight.half(),
                                       self.bias, self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

    def forward_2(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return modulated_deform_conv2d(x, offset, mask, self.weight.half(),
                                       self.bias, self.stride, self.padding,
                                       self.dilation, self.groups,
                                       self.deform_groups)

    if hasattr(target, "ModulatedDeformConv2d"):
        target.ModulatedDeformConv2d.forward = forward_1
    if hasattr(target, "ModulatedDeformConv2dPack"):
        target.ModulatedDeformConv2dPack.forward = forward_2


def generate_patcher_builder():
    bevformer_patcher_builder = (
        PatcherBuilder()
        .add_module_patch("numpy", Patch(numpy_type))
        .add_module_patch("nuscenes.eval.detection.data_classes", Patch(data_classess))
        .add_module_patch("torch.nn.parallel.data_parallel", Patch(data_parallel))
        .add_module_patch("mmcv.ops.focal_loss", Patch(focal_loss))
        .add_module_patch("mmcv.ops.modulated_deform_conv", Patch(modulated_deform_conv2d_))
    )
    if os.environ.get("BEVFORMER_PERFORMANCE_FLAG"):
        bevformer_patcher_builder.brake_at(500)
    return bevformer_patcher_builder
