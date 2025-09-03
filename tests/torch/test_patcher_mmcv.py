import importlib
import types
import random
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List, Union, Dict
from types import ModuleType

import torch
from torch_npu.testing.testcase import TestCase, run_tests

from mx_driving.patcher import mmcv_patch, mmengine_patch 



def assertIsNotInstance(obj, cls):
    assert not isinstance(obj, cls), f"Expected {repr(obj)} to NOT be an instance of {cls.__name__}"


class TestPatcherMMCV(TestCase):
    def setUp(self):
        pass
    
    def test_patch_mmcv_version_found(self):
        """Test successful import of mmcv and version patching"""
        with patch('importlib.import_module') as mock_import:
            # Mock mmcv module
            mock_mmcv = MagicMock()
            mock_mmcv.__version__ = "1.7.2"
            mock_import.return_value = mock_mmcv
            
            # Call patching function
            mmcv_patch.patch_mmcv_version("2.1.0")
            
            # Assert version restoration
            self.assertEqual(mock_mmcv.__version__, "1.7.2", "Version should be restored to original")
            
            # Assert import attempts
            mock_import.assert_any_call("mmdet")
            mock_import.assert_any_call("mmdet3d")

    def test_patch_mmcv_version_not_found(self):
        """Test handling when mmcv cannot be imported"""
        with patch('importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError
            # Assert no exception raised
            mmcv_patch.patch_mmcv_version("666.888.2333")
            mock_import.assert_called_once_with("mmcv")

    def test_dc(self):
        """Test monkeypatching for deform_conv2d"""
        mock_mmcv = MagicMock()
        
        # Call dc function
        mmcv_patch.dc(mock_mmcv, {})
        
        # Assert function replacements
        assertIsNotInstance(mock_mmcv.ops.deform_conv.DeformConv2dFunction, MagicMock)
        assertIsNotInstance(mock_mmcv.ops.deform_conv.deform_conv2d, MagicMock)

    def test_mdc(self):
        """Test monkeypatching for modulated_deform_conv2d"""
        mock_mmcv = MagicMock()
        
        # Call mdc function
        mmcv_patch.mdc(mock_mmcv, {})
        
        # Assert function replacements
        assertIsNotInstance(mock_mmcv.ops.modulated_deform_conv.ModulatedDeformConv2dFunction, MagicMock)
        assertIsNotInstance(mock_mmcv.ops.modulated_deform_conv.modulated_deform_conv2d, MagicMock)

    def test_msda(self):
        """Test monkeypatching for multi_scale_deformable_attn"""
        mock_mmcv = MagicMock()

        # Call msda function
        mmcv_patch.msda(mock_mmcv, {})
        
        # Assert function replacements
        assertIsNotInstance(mock_mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction.forward, MagicMock)
        assertIsNotInstance(mock_mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction.backward, MagicMock)
    
    def test_ddp_patch(self):
        mock_mmcv = MagicMock()
        mock_mmcv.device.npu.NPUDistributedDataParallel = types.ModuleType('npuddp')
        mmddp_b4replacement = mock_mmcv.parallel.distributed.MMDistributedDataParallel
        
        # Apply monkey patch
        mmengine_patch.ddp(mock_mmcv, {})
        assertIsNotInstance(mmddp_b4replacement._run_ddp_forward, MagicMock)
        assertIsNotInstance(mock_mmcv.parallel.distributed.MMDistributedDataParallel, MagicMock)


class TestPatcherStream(TestCase):
    def setUp(self):
        # Create mock mmcvparallel module
        self.mock_mmcv = types.ModuleType('mmcv')
        self.mock_mmcv.parallel = types.ModuleType('mmcvparallel')
        self.mock_mmcv.parallel._functions = types.ModuleType('_functions')
        self.mock_mmcv.parallel._functions.Scatter = MagicMock()
        
        # Add the missing attributes for torch
        self.mock_mmcv.parallel._functions.torch = types.ModuleType('torch')
        self.mock_mmcv.parallel._functions.torch.device = torch.device

        # Set up necessary functions and types
        self.mock_mmcv.parallel._functions.get_input_device = MagicMock()
        self.mock_mmcv.parallel._functions.scatter = MagicMock()
        self.mock_mmcv.parallel._functions.synchronize_stream = MagicMock()
        self.mock_mmcv.parallel._functions._get_stream = MagicMock()
        self.mock_mmcv.parallel._functions.Tensor = torch.Tensor
        
        # Set default return values
        self.mock_mmcv.parallel._functions.get_input_device.return_value = -1
        self.mock_mmcv.parallel._functions.scatter.return_value = ["scatter_output"]
        
        # Dynamically return target # of gpu 
        def scatter_mock(input_, target_gpus, streams=None):
            return [f"output_{i}" for i in range(len(target_gpus))]
    
        self.mock_mmcv.parallel._functions.scatter = MagicMock(side_effect=scatter_mock)
    
    def test_monkeypatch(self):
        """Verify forward method is correctly replaced"""
        options = {}
        
        # Apply monkeypatch using stream function
        mmengine_patch.stream(self.mock_mmcv, options)
        
        # Verify Scatter.forward has been replaced with new_forward
        assertIsNotInstance(self.mock_mmcv.parallel._functions.Scatter.forward, MagicMock)
        
    def test_new_forward_input_device_neg_one(self):
        """Test stream behavior when input device is -1 and target GPUs are not [-1]"""
        mmengine_patch.stream(self.mock_mmcv, {})
        
        # Create mock input
        test_input = MagicMock(spec=torch.Tensor)
        target_gpus = [0, 1]
        
        # Execute new forward method
        result = self.mock_mmcv.parallel._functions.Scatter.forward.__func__(target_gpus, test_input)
        
        # Verify stream handling logic
        self.mock_mmcv.parallel._functions._get_stream.assert_called()
        self.mock_mmcv.parallel._functions.scatter.assert_called_once()
        self.mock_mmcv.parallel._functions.synchronize_stream.assert_called_once()
        
        # Verify output format
        self.assertEqual(len(result), len(target_gpus))
        self.assertIsInstance(result, tuple)

    def test_new_forward_non_neg_input_device(self):
        """Test behavior when input device is not -1"""
        mmengine_patch.stream(self.mock_mmcv, {})
        
        # Set input device to non-negative value
        self.mock_mmcv.parallel._functions.get_input_device.return_value = 0
        test_input = MagicMock(spec=torch.Tensor)
        target_gpus = [0, 1]
        
        # Execute new forward method
        result = self.mock_mmcv.parallel._functions.Scatter.forward.__func__(target_gpus, test_input)
        
        # Verify no stream handling occurs
        self.mock_mmcv.parallel._functions._get_stream.assert_not_called()
        self.mock_mmcv.parallel._functions.scatter.assert_called_once()
        self.mock_mmcv.parallel._functions.synchronize_stream.assert_not_called()
        self.assertIsInstance(result, tuple)

    def test_new_forward_list_input(self):
        """Test handling of list input"""
        mmengine_patch.stream(self.mock_mmcv, {})
        
        # Create list input
        test_input = [torch.tensor([1]), torch.tensor([2])]
        target_gpus = [0, 1]
        
        # Execute new forward method
        result = self.mock_mmcv.parallel._functions.Scatter.forward.__func__(target_gpus, test_input)
        
        # Verify processing logic
        self.mock_mmcv.parallel._functions.get_input_device.assert_called_once()
        self.mock_mmcv.parallel._functions.scatter.assert_called_once()
        self.assertIsInstance(result, tuple)


if __name__ == '__main__':
    run_tests()