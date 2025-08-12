import random
import types
import unittest
from unittest.mock import ANY, patch, MagicMock, PropertyMock

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from mx_driving.patcher import ddp


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
        ddp(self.mock_mmcvparallel.distributed, {})
        
        assertIsNotInstance(self.mock_mmcvparallel.distributed.MMDistributedDataParallel, MagicMock)    
        
if __name__ == '__main__':
    run_tests()