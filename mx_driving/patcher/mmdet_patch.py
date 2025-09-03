# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from types import ModuleType
from typing import Dict


def pseudo_sampler(mmdet: ModuleType, options: Dict):
    if hasattr(mmdet.core.bbox.samplers.pseudo_sampler.PseudoSampler, "sample"):

        def sample(self, assign_result, bboxes, gt_bboxes, *args, **kwargs):
            import torch

            pos_inds = torch.squeeze(assign_result.gt_inds > 0, -1)
            neg_inds = torch.squeeze(assign_result.gt_inds == 0, -1)
            gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
            sampling_result = mmdet.core.bbox.samplers.sampling_result.SamplingResult(
                pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags
            )
            return sampling_result

        mmdet.core.bbox.samplers.pseudo_sampler.PseudoSampler.sample = sample
    else:
        raise AttributeError("mmdet.core.bbox.samplers.pseudo_sampler.PseudoSampler.sample not found")


def resnet_add_relu(mmdet: ModuleType, options: Dict):    
    basic_block_not_found = True
    bottle_neck_not_found = True
    if hasattr(mmdet.models.backbones.resnet.BasicBlock, "forward"):
        from mx_driving import npu_add_relu
        import torch.utils.checkpoint as cp

        def forward(self, x):
            def _inner_forward(x):
                identity = x
                out = self.conv1(x)
                out = self.norm1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.norm2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)
                out = npu_add_relu(out, identity)

                return out

            if self.with_cp and x.requires_grad:
                out = cp.checkpoint(_inner_forward, x)
            else:
                out = _inner_forward(x)

            return out

        mmdet.models.backbones.resnet.BasicBlock.forward = forward
        basic_block_not_found = False

    if hasattr(mmdet.models.backbones.resnet.Bottleneck, "forward"):
        from mx_driving import npu_add_relu
        import torch.utils.checkpoint as cp
        
        def forward(self, x):
            """Forward function."""

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

                out = npu_add_relu(out, identity)

                return out

            if self.with_cp and x.requires_grad:
                out = cp.checkpoint(_inner_forward, x)
            else:
                out = _inner_forward(x)

            return out

        mmdet.models.backbones.resnet.Bottleneck.forward = forward
        bottle_neck_not_found = False
    
    if basic_block_not_found or bottle_neck_not_found:
        raise AttributeError("In mmdet.models.backbones.resnet, BasicBlock.forward or Bottleneck.forward not found")



def resnet_maxpool(mmdet: ModuleType, options: Dict):
    if hasattr(mmdet.models.backbones.resnet.ResNet, "forward"):
        from mx_driving import npu_max_pool2d

        def forward(self, x):
            if self.deep_stem:
                x = self.stem(x)
            else:
                x = self.conv1(x)
                x = self.norm1(x)
                x = self.relu(x)
            if x.requires_grad:
                x = self.maxpool(x)
            else:
                x = npu_max_pool2d(x, 3, 2, 1)
            out = []
            for i, layer_name in enumerate(self.res_layers):
                res_layer = getattr(self, layer_name)
                x = res_layer(x)
                if i in self.out_indices:
                    out.append(x)
            return tuple(out)

        mmdet.models.backbones.resnet.ResNet.forward = forward
    else:
        raise AttributeError("mmdet.models.backbones.resnet.ResNet.forward not found")


def resnet_fp16(mmdet: ModuleType, options: Dict):
    if hasattr(mmdet.models.backbones.resnet.ResNet, "forward"):

        def forward(self, x):
            import torch
            with torch.autocast(device_type="npu", dtype=torch.float16):
                """Forward function."""
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
            return tuple([out.float() for out in tuple(outs)])

        mmdet.models.backbones.resnet.ResNet.forward = forward
    else:
        raise AttributeError("mmdet.models.backbones.resnet.ResNet.forward not found")