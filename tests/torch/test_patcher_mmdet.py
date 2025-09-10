import types
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from torch_npu.testing.testcase import TestCase, run_tests

from mx_driving.patcher import pseudo_sampler, resnet_add_relu, resnet_maxpool


def assertIsNotInstance(obj, cls):
    assert not isinstance(obj, cls), f"Expected {repr(obj)} to NOT be an instance of {cls.__name__}"


class EmptyAttribute:
    pass


class TestPseudoSamplerPatch(TestCase):

    def setUp(self):
        # mmdet.core.bbox.samplers.pseudo_sampler
        # Create mock mmdetsamplers module
        self.mock_mmdet = types.ModuleType('mmdet')
        self.mock_mmdet.core = types.ModuleType('core')
        self.mock_mmdet.core.bbox = types.ModuleType('bbox')
        self.mock_mmdet.core.bbox.samplers = types.ModuleType('samplers')
        self.mock_mmdet.core.bbox.samplers.pseudo_sampler = types.ModuleType('pseudo_sampler')
        self.mock_mmdet.core.bbox.samplers.sampling_result = types.ModuleType('sampling_result')

        # Mock PseudoSampler class and sample method
        self.mock_pseudo_sampler_cls = MagicMock()
        self.mock_mmdet.core.bbox.samplers.pseudo_sampler.PseudoSampler = self.mock_pseudo_sampler_cls

        # Mock SamplingResult class
        self.mock_sampling_result = MagicMock()
        self.mock_mmdet.core.bbox.samplers.sampling_result.SamplingResult = self.mock_sampling_result

    def test_patching_with_pseudo_sampler(self):
        """Test successful patching when PseudoSampler exists."""
        # Apply patching
        pseudo_sampler(self.mock_mmdet, {})

        # Verify sample method was replaced
        assertIsNotInstance(
            self.mock_mmdet.core.bbox.samplers.pseudo_sampler.PseudoSampler.sample,
            MagicMock
        )

    def test_patched_behavior(self):
        """Test behavior of patched sample method."""
        # Apply patching
        pseudo_sampler(self.mock_mmdet, {})

        # Create mock inputs
        mock_assign_result = MagicMock()
        mock_assign_result.gt_inds = torch.tensor([1, 0, 2]).unsqueeze(1)
        mock_bboxes = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        mock_gt_bboxes = torch.tensor([[0.1, 0.1, 0.2], [0.3, 0.3, 0.4]])

        # Call patched method
        patched_sample = self.mock_mmdet.core.bbox.samplers.pseudo_sampler.PseudoSampler.sample
        sampling_result = patched_sample(
            None, mock_assign_result, mock_bboxes, mock_gt_bboxes
        )

        # Verify SamplingResult construction
        self.mock_mmdet.core.bbox.samplers.sampling_result.SamplingResult.assert_called_once()
        pos_inds, neg_inds, _, _, _, _ = self.mock_sampling_result.call_args[0]
        self.assertRtolEqual(pos_inds, torch.tensor([True, False, True]))
        self.assertRtolEqual(neg_inds, torch.tensor([False, True, False]))

    def test_patch_failure(self):
        mock_mmdet = MagicMock()
        with self.assertRaises(AttributeError):
            mock_mmdet.core.bbox.samplers.pseudo_sampler.PseudoSampler = EmptyAttribute
            pseudo_sampler(mock_mmdet, {})


class TestResNetAddReLUPatch(TestCase):
    def setUp(self):
        # mmdet.models.backbones.resnet
        self.mock_mmdet = types.ModuleType('mmdet')
        self.mock_mmdet.models = types.ModuleType('models')
        self.mock_mmdet.models.backbones = types.ModuleType('backbones')
        self.mock_mmdet.models.backbones.resnet = types.ModuleType('resnet')

        class BasicBlock(nn.Module):
            def __init__(self, downsample=None):
                super().__init__()
                self.conv1 = nn.Identity()
                self.norm1 = nn.Identity()
                self.relu = nn.Identity()
                self.conv2 = nn.Identity()
                self.norm2 = nn.Identity()
                self.downsample = downsample
                self.with_cp = False

            def forward(self, x):
                return x  # just a mock, original implementation will be replaced

        class Bottleneck(nn.Module):
            def __init__(self, downsample=None):
                super().__init__()
                self.conv1 = nn.Identity()
                self.norm1 = nn.Identity()
                self.relu = nn.Identity()
                self.conv2 = nn.Identity()
                self.norm2 = nn.Identity()
                self.conv3 = nn.Identity()
                self.norm3 = nn.Identity()
                self.downsample = downsample
                self.with_cp = False
                self.with_plugins = False
                self.after_conv1_plugin_names = []
                self.after_conv2_plugin_names = []
                self.after_conv3_plugin_names = []

                def forward_plugin_func(x, _):
                    return x

                self.forward_plugin = forward_plugin_func

            def forward(self, x):
                return x  # just a mock, original implementation will be replaced

        self.mock_mmdet.models.backbones.resnet.BasicBlock = BasicBlock
        self.mock_mmdet.models.backbones.resnet.Bottleneck = Bottleneck

    def test_basic_block_forward(self):
        # Apply patch
        resnet_add_relu(self.mock_mmdet, {})

        block = self.mock_mmdet.models.backbones.resnet.BasicBlock()
        x = torch.tensor([1.0]).npu()

        # execute forward and verify
        result = block(x)
        self.assertRtolEqual(result, torch.tensor([2.0]).npu())

    def test_basic_block_with_downsample(self):
        # Apply patch
        resnet_add_relu(self.mock_mmdet, {})

        # with downsample
        downsample = nn.Identity()
        block = self.mock_mmdet.models.backbones.resnet.BasicBlock(downsample=downsample)
        x = torch.tensor([1.0]).npu()

        # execute forward and verify
        result = block(x)
        self.assertRtolEqual(result, torch.tensor([2.0]).npu())

    def test_basic_block_with_cp(self):
        # Apply patch
        resnet_add_relu(self.mock_mmdet, {})

        # with checkpoint
        block = self.mock_mmdet.models.backbones.resnet.BasicBlock()
        block.with_cp = True
        x = torch.tensor([1.0], requires_grad=True).npu()

        # execute forward and verify
        result = block(x)
        self.assertRtolEqual(result, torch.tensor([2.0]).npu())

    def test_bottleneck_forward(self):
        # Apply patch
        resnet_add_relu(self.mock_mmdet, {})

        block = self.mock_mmdet.models.backbones.resnet.Bottleneck()
        x = torch.tensor([1.0]).npu()

        # execute forward and verify
        result = block(x)
        self.assertRtolEqual(result, torch.tensor([2.0]).npu())

    def test_bottleneck_with_plugins(self):
        # Apply patch
        resnet_add_relu(self.mock_mmdet, {})

        # with plugins
        block = self.mock_mmdet.models.backbones.resnet.Bottleneck()
        block.with_plugins = True
        x = torch.tensor([1.0]).npu()

        # execute and verify
        result = block(x)
        self.assertRtolEqual(result, torch.tensor([2.0]).npu())

    def test_patch_failure(self):
        mock_mmdet = MagicMock()
        with self.assertRaises(AttributeError):
            mock_mmdet.models.backbones.resnet.BasicBlock = EmptyAttribute
            resnet_add_relu(mock_mmdet, {})


class TestResNetMaxPoolPatch(TestCase):
    def setUp(self):
        # mmdet.models.backbones.resnet
        self.mock_mmdet = types.ModuleType('mmdet')
        self.mock_mmdet.models = types.ModuleType('models')
        self.mock_mmdet.models.backbones = types.ModuleType('backbones')
        self.mock_mmdet.models.backbones.resnet = types.ModuleType('resnet')

        class ResNet(nn.Module):
            def __init__(self, deep_stem=False):
                super().__init__()
                self.deep_stem = deep_stem
                self.stem = nn.Identity() if deep_stem else None
                self.conv1 = nn.Identity()
                self.norm1 = nn.Identity()
                self.relu = nn.Identity()
                self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
                self.res_layers = ['layer1', 'layer2']
                self.out_indices = [0, 1]
                self.layer1 = nn.Identity()
                self.layer2 = nn.Identity()

            def forward(self, x):
                return x  # just a mock, original implementation will be replaced

        self.mock_mmdet.models.backbones.resnet.ResNet = ResNet

    def test_forward_with_grad(self):
        resnet_maxpool(self.mock_mmdet, {})

        model = self.mock_mmdet.models.backbones.resnet.ResNet()
        x = torch.ones(1, 3, 32, 32, requires_grad=True).npu()

        result = model(x)
        self.assertEqual(len(result), 2)  # verify output layer number

    def test_forward_without_grad(self):
        resnet_maxpool(self.mock_mmdet, {})

        model = self.mock_mmdet.models.backbones.resnet.ResNet()
        x = torch.ones(1, 3, 32, 32, requires_grad=False).npu()

        result = model(x)
        self.assertEqual(len(result), 2)

    def test_deep_stem_path(self):
        resnet_maxpool(self.mock_mmdet, {})

        model = self.mock_mmdet.models.backbones.resnet.ResNet(deep_stem=True)
        x = torch.ones(1, 3, 32, 32).npu()

        result = model(x)
        self.assertEqual(len(result), 2)

    def test_out_indices_handling(self):
        resnet_maxpool(self.mock_mmdet, {})

        model = self.mock_mmdet.models.backbones.resnet.ResNet()
        model.out_indices = [1]  # only output the second layer

        x = torch.ones(1, 3, 32, 32).npu()

        result = model(x)
        self.assertEqual(len(result), 1)  # verify only output 1 layer

    def test_patch_failure(self):
        mock_mmdet = MagicMock()
        with self.assertRaises(AttributeError):
            mock_mmdet.models.backbones.resnet.ResNet = EmptyAttribute
            resnet_maxpool(mock_mmdet, {})


if __name__ == "__main__":
    run_tests()