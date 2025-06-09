import random
import types
import unittest
from unittest.mock import ANY, patch, MagicMock, PropertyMock

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from mx_driving.patcher import ddp, ddp_forward


def assertIsNotInstance(obj, cls):
    assert not isinstance(obj, cls), f"Expected {repr(obj)} to NOT be an instance of {cls.__name__}"


class TestDistribute(TestCase):
    def setUp(self):
        # Create mock objects for testing
        self.mock_mmcvparallel = MagicMock()
        self.mock_mmcvparallel.distributed = MagicMock()
        self.mock_mmcvparallel.distributed.MMDistributedDataParallel = MagicMock()
    
    def test_ddp_patch(self):
        # Apply monkey patch
        ddp(self.mock_mmcvparallel, {})
        
        assertIsNotInstance(self.mock_mmcvparallel.distributed.MMDistributedDataParallel, MagicMock)    
        
    def test_ddp_forward_patch(self):
        # Apply the ddp_forward patch
        ddp_forward(self.mock_mmcvparallel, {})
        
        # Get the patched _run_ddp_forward method
        new_forward = self.mock_mmcvparallel.distributed.MMDistributedDataParallel._run_ddp_forward
        
        # Verify _run_ddp_forward is correctly replaced 
        assertIsNotInstance(
            new_forward, 
            MagicMock
        )
        
        # Create mock instance and inputs
        mock_self = MagicMock()
        mock_self.device_ids = [0]  # Simulate device IDs present
        mock_self.module = MagicMock(return_value="module_output")
        
        # Mock the to_kwargs method
        mock_self.to_kwargs = MagicMock(return_value=(
            [("processed_input",)], 
            [{"processed_kwarg": "value"}]
        ))
        
        # Call the patched forward method
        result = new_forward(mock_self, "input1", "input2", kwarg1="value1")
        
        # Check to_kwargs is called correctly
        mock_self.to_kwargs.assert_called_once_with(
            ("input1", "input2"), 
            {"kwarg1": "value1"}, 
            0
        )
        
        # Check module is called correctly
        mock_self.module.assert_called_once_with(
            "processed_input", 
            processed_kwarg="value"
        )
        
        # Verify return value
        self.assertEqual(result, "module_output")
        
        # Test case with no device_ids
        mock_self.reset_mock()
        mock_self.device_ids = []
        result = new_forward(mock_self, "input3", kwarg2="value2")
        mock_self.module.assert_called_once_with("input3", kwarg2="value2")

if __name__ == '__main__':
    run_tests()