import importlib
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving.patcher.mmcv as mx_mmcv


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
            mx_mmcv.patch_mmcv_version("2.1.0")
            
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
            mx_mmcv.patch_mmcv_version("666.888.2333")
            mock_import.assert_called_once_with("mmcv")

    def test_dc(self):
        """Test monkeypatching for deform_conv2d"""
        mock_mmcvops = MagicMock()
        
        # Call dc function
        mx_mmcv.dc(mock_mmcvops, {})
        
        # Assert function replacements
        assertIsNotInstance(mock_mmcvops.deform_conv.DeformConv2dFunction, MagicMock)
        assertIsNotInstance(mock_mmcvops.deform_conv.deform_conv2d, MagicMock)

    def test_mdc(self):
        """Test monkeypatching for modulated_deform_conv2d"""
        mock_mmcvops = MagicMock()
        mock_mmcvops.modulated_deform_conv = MagicMock()
        
        # Call mdc function
        mx_mmcv.mdc(mock_mmcvops, {})
        
        # Assert function replacements
        assertIsNotInstance(mock_mmcvops.modulated_deform_conv.ModulatedDeformConv2dFunction, MagicMock)
        assertIsNotInstance(mock_mmcvops.modulated_deform_conv.modulated_deform_conv2d, MagicMock)

    def test_msda(self):
        """Test monkeypatching for multi_scale_deformable_attn"""
        mock_mmcvops = MagicMock()
        mock_mmcvops.multi_scale_deformable_attn = MagicMock()
        
        # Call msda function
        mx_mmcv.msda(mock_mmcvops, {})
        
        # Assert function replacements
        assertIsNotInstance(mock_mmcvops.multi_scale_deformable_attn.MultiScaleDeformableAttnFunction, MagicMock)
        assertIsNotInstance(mock_mmcvops.multi_scale_deformable_attn.multi_scale_deformable_attn, MagicMock)


if __name__ == '__main__':
    run_tests()