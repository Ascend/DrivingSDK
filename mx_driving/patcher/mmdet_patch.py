# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""MMDetection patches for NPU optimization."""
from typing import List

from mx_driving.patcher.patch import AtomicPatch, BasePatch, Patch, mmcv_version


class PseudoSampler(Patch):
    """Pseudo sampler patch for mmdet (mmdet 2.x / mmcv 1.x only)."""

    name = "pseudo_sampler"
    legacy_name = "pseudo_sampler"
    target_module = "mmdet"

    @staticmethod
    def _precheck() -> bool:
        """Only apply for mmdet 2.x (mmcv 1.x)."""
        return mmcv_version.is_v1x

    @staticmethod
    def _sample_replacement(self, assign_result, bboxes, gt_bboxes, *args, **kwargs):
        import torch
        from mmdet.core.bbox.samplers.sampling_result import SamplingResult
        pos_inds = torch.squeeze(assign_result.gt_inds > 0, -1)
        neg_inds = torch.squeeze(assign_result.gt_inds == 0, -1)
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        return SamplingResult(
            pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags
        )

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "mmdet.core.bbox.samplers.pseudo_sampler.PseudoSampler.sample",
                cls._sample_replacement,
                precheck=cls._precheck,
            ),
        ]


class ResNetAddRelu(Patch):
    """ResNet add+relu fusion patch for mmdet."""

    name = "resnet_add_relu"
    legacy_name = "resnet_add_relu"
    target_module = "mmdet"

    @staticmethod
    def _basicblock_forward_replacement(self, x):
        from mx_driving import npu_add_relu
        import torch.utils.checkpoint as cp

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.norm2(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            return npu_add_relu(out, identity)

        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner_forward, x)
        return _inner_forward(x)

    @staticmethod
    def _bottleneck_forward_replacement(self, x):
        from mx_driving import npu_add_relu
        import torch.utils.checkpoint as cp

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)
            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)
            out = self.conv3(out)
            out = self.norm3(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)
            if self.downsample is not None:
                identity = self.downsample(x)
            return npu_add_relu(out, identity)

        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner_forward, x)
        return _inner_forward(x)

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "mmdet.models.backbones.resnet.BasicBlock.forward",
                cls._basicblock_forward_replacement,
            ),
            AtomicPatch(
                "mmdet.models.backbones.resnet.Bottleneck.forward",
                cls._bottleneck_forward_replacement,
            ),
        ]


class ResNetMaxPool(Patch):
    """ResNet maxpool optimization patch for mmdet."""

    name = "resnet_maxpool"
    legacy_name = "resnet_maxpool"
    target_module = "mmdet"

    @staticmethod
    def _forward_replacement(self, x):
        from mx_driving import npu_max_pool2d
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x) if x.requires_grad else npu_max_pool2d(x, 3, 2, 1)
        out = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                out.append(x)
        return tuple(out)

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "mmdet.models.backbones.resnet.ResNet.forward",
                cls._forward_replacement,
            ),
        ]


class ResNetFP16(Patch):
    """
    ResNet FP16 autocast patch for mmdet.

    WARNING: This patch conflicts with ResNetMaxPool as both patch the same method
    (mmdet.models.backbones.resnet.ResNet.forward). Only one should be enabled at a time.
    This patch is NOT included in default_patcher - use it explicitly if needed:

        from mx_driving.patcher import default_patcher, ResNetFP16
        default_patcher.disable(ResNetMaxPool.name).add(ResNetFP16).apply()
    """

    name = "resnet_fp16"
    legacy_name = "resnet_fp16"
    target_module = "mmdet"
    # Conflicts with ResNetMaxPool - both patch ResNet.forward
    conflicts_with = ["resnet_maxpool"]

    @staticmethod
    def _forward_replacement(self, x):
        import torch
        with torch.autocast(device_type="npu", dtype=torch.float16):
            if self.deep_stem:
                x = self.stem(x)
            else:
                x = self.conv1(x)
                x = self.norm1(x)
                x = self.relu(x)
            x = self.maxpool(x)
            outs = []
            for i, layer_name in enumerate(self.res_layers):
                res_layer = getattr(self, layer_name)
                x = res_layer(x)
                if i in self.out_indices:
                    outs.append(x)
        return tuple([out.float() for out in outs])

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "mmdet.models.backbones.resnet.ResNet.forward",
                cls._forward_replacement,
            ),
        ]
