import importlib
import types
import unittest
from unittest.mock import Mock
from torch_npu.testing.testcase import TestCase, run_tests

from mx_driving.patcher import numpy_type


class TestNumpyTypePatch(TestCase):
    
    def setUp(self):
        self.mock_np = Mock(spec=[]) # restrictd attr, hasattr will be false
    
    def test_numpy_type_patch_replacement(self):
        numpy_type(self.mock_np, {})
        self.assertEqual(self.mock_np.bool, bool)
        self.assertEqual(self.mock_np.float, float)
        
        
if __name__ == "__main__":
    run_tests()