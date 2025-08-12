import random
import types
import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from typing import List, Union, Dict
from types import ModuleType

import torch
from torch_npu.testing.testcase import TestCase, run_tests


def assertIsNotInstance(obj, cls):
    assert not isinstance(obj, cls), f"Expected {repr(obj)} to NOT be an instance of {cls.__name__}"


class TestPatcherStream(TestCase):
    def setUp(self):
        # Create mock mmcvparallel module
        self.mock_mmcvparallel = types.ModuleType('mmcvparallel')
        self.mock_mmcvparallel._functions = types.ModuleType('_functions')
        self.mock_mmcvparallel._functions.Scatter = MagicMock()
        
        # Add the missing attributes for torch
        self.mock_mmcvparallel._functions.torch = types.ModuleType('torch')
        self.mock_mmcvparallel._functions.torch.device = torch.device

        # Set up necessary functions and types
        self.mock_mmcvparallel._functions.get_input_device = MagicMock()
        self.mock_mmcvparallel._functions.scatter = MagicMock()
        self.mock_mmcvparallel._functions.synchronize_stream = MagicMock()
        self.mock_mmcvparallel._functions._get_stream = MagicMock()
        self.mock_mmcvparallel._functions.Tensor = torch.Tensor
        
        # Set default return values
        self.mock_mmcvparallel._functions.get_input_device.return_value = -1
        self.mock_mmcvparallel._functions.scatter.return_value = ["scatter_output"]
        
        # Dynamically return target # of gpu 
        def scatter_mock(input_, target_gpus, streams=None):
            return [f"output_{i}" for i in range(len(target_gpus))]
    
        self.mock_mmcvparallel._functions.scatter = MagicMock(side_effect=scatter_mock)
    
    def test_monkeypatch(self):
        """Verify forward method is correctly replaced"""
        from mx_driving.patcher import stream
        options = {}
        
        # Apply monkeypatch using stream function
        stream(self.mock_mmcvparallel, options)
        
        # Verify Scatter.forward has been replaced with new_forward
        assertIsNotInstance(self.mock_mmcvparallel._functions.Scatter.forward, MagicMock)
        
    def test_new_forward_input_device_neg_one(self):
        """Test stream behavior when input device is -1 and target GPUs are not [-1]"""
        from mx_driving.patcher import stream
        stream(self.mock_mmcvparallel, {})
        
        # Create mock input
        test_input = MagicMock(spec=torch.Tensor)
        target_gpus = [0, 1]
        
        # Execute new forward method
        result = self.mock_mmcvparallel._functions.Scatter.forward.__func__(target_gpus, test_input)
        
        # Verify stream handling logic
        self.mock_mmcvparallel._functions._get_stream.assert_called()
        self.mock_mmcvparallel._functions.scatter.assert_called_once()
        self.mock_mmcvparallel._functions.synchronize_stream.assert_called_once()
        
        # Verify output format
        self.assertEqual(len(result), len(target_gpus))
        self.assertIsInstance(result, tuple)

    def test_new_forward_non_neg_input_device(self):
        """Test behavior when input device is not -1"""
        from mx_driving.patcher import stream
        stream(self.mock_mmcvparallel, {})
        
        # Set input device to non-negative value
        self.mock_mmcvparallel._functions.get_input_device.return_value = 0
        test_input = MagicMock(spec=torch.Tensor)
        target_gpus = [0, 1]
        
        # Execute new forward method
        result = self.mock_mmcvparallel._functions.Scatter.forward.__func__(target_gpus, test_input)
        
        # Verify no stream handling occurs
        self.mock_mmcvparallel._functions._get_stream.assert_not_called()
        self.mock_mmcvparallel._functions.scatter.assert_called_once()
        self.mock_mmcvparallel._functions.synchronize_stream.assert_not_called()
        self.assertIsInstance(result, tuple)

    def test_new_forward_list_input(self):
        """Test handling of list input"""
        from mx_driving.patcher import stream
        stream(self.mock_mmcvparallel, {})
        
        # Create list input
        test_input = [torch.tensor([1]), torch.tensor([2])]
        target_gpus = [0, 1]
        
        # Execute new forward method
        result = self.mock_mmcvparallel._functions.Scatter.forward.__func__(target_gpus, test_input)
        
        # Verify processing logic
        self.mock_mmcvparallel._functions.get_input_device.assert_called_once()
        self.mock_mmcvparallel._functions.scatter.assert_called_once()
        self.assertIsInstance(result, tuple)


if __name__ == "__main__":
    run_tests()