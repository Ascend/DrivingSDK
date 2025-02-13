# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from types import ModuleType
from typing import Dict


def pseudo_sampler(mmdetsamplers: ModuleType, options: Dict):
    if hasattr(mmdetsamplers, "pseudo_sampler"):

        def sample(self, assign_result, bboxes, gt_bboxes, *args, **kwargs):
            import torch

            pos_inds = torch.squeeze(assign_result.gt_inds > 0, -1)
            neg_inds = torch.squeeze(assign_result.gt_inds == 0, -1)
            gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
            sampling_result = mmdetsamplers.sampling_result.SamplingResult(
                pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags
            )
            return sampling_result

        mmdetsamplers.pseudo_sampler.PseudoSampler.sample = sample


def resnet_add_relu(mmdetresnet: ModuleType, options: Dict):
    if hasattr(mmdetresnet, "BasicBlock"):
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

        mmdetresnet.BasicBlock.forward = forward

    if hasattr(mmdetresnet, "Bottleneck"):

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

        mmdetresnet.Bottleneck.forward = forward


def resnet_maxpool(mmdetresnet: ModuleType, options: Dict):
    if hasattr(mmdetresnet, "ResNet"):
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

        mmdetresnet.ResNet.forward = forward
