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


class EmptyAttribute:
    pass


class TestMultiScaleDeformableAttnPatch(TestCase):
    def test_monkey_patch(self):
        """Test monkeypatching for multi_scale_deformable_attn"""
        mock_mmcv = MagicMock()

        # Call msda function
        mmcv_patch.msda(mock_mmcv, {})
        
        # Assert function replacements
        assertIsNotInstance(mock_mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction.forward, MagicMock)
        assertIsNotInstance(mock_mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction.backward, MagicMock)
    
    def test_patch_failure(self):
        mock_mmcv = MagicMock()
        with self.assertRaises(AttributeError):
            mock_mmcv.ops = EmptyAttribute
            mmcv_patch.msda(mock_mmcv, {})
        
        mock_mmcv = MagicMock()
        with self.assertRaises(AttributeError):
            mock_mmcv.ops = EmptyAttribute
            mmcv_patch.msda(mock_mmcv, {})


class TestDeformConv2dPatch(TestCase):
    def test_monkey_patch(self):
        """Test monkeypatching for deform_conv2d"""
        mock_mmcv = MagicMock()
        
        # Call dc function
        mmcv_patch.dc(mock_mmcv, {})
        
        # Assert function replacements
        assertIsNotInstance(mock_mmcv.ops.deform_conv.DeformConv2dFunction, MagicMock)
        assertIsNotInstance(mock_mmcv.ops.deform_conv.deform_conv2d, MagicMock)
        
    def test_patch_failure(self):
        mock_mmcv = MagicMock()
        with self.assertRaises(AttributeError):
            mock_mmcv.ops = EmptyAttribute
            mmcv_patch.dc(mock_mmcv, {})
        
        mock_mmcv = MagicMock()
        with self.assertRaises(AttributeError):
            mock_mmcv.ops = EmptyAttribute
            mmcv_patch.dc(mock_mmcv, {})


class TestModulatedDeformConv2dPatch(TestCase):
    def test_monkey_patch(self):
        """Test monkeypatching for modulated_deform_conv2d"""
        mock_mmcv = MagicMock()
        
        # Call mdc function
        mmcv_patch.mdc(mock_mmcv, {})
        
        # Assert function replacements
        assertIsNotInstance(mock_mmcv.ops.modulated_deform_conv.ModulatedDeformConv2dFunction, MagicMock)
        assertIsNotInstance(mock_mmcv.ops.modulated_deform_conv.modulated_deform_conv2d, MagicMock)

    def test_patch_failure(self):
        mock_mmcv = MagicMock()
        with self.assertRaises(AttributeError):
            mock_mmcv.ops = EmptyAttribute
            mmcv_patch.mdc(mock_mmcv, {})
        
        mock_mmcv = MagicMock()
        with self.assertRaises(AttributeError):
            mock_mmcv.ops = EmptyAttribute
            mmcv_patch.mdc(mock_mmcv, {})


class TestResolveMMCV_VersionConflict(TestCase):
    def test_mmcv_version_found(self):
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

    def test_mmcv_version_not_found(self):
        """Test handling when mmcv cannot be imported"""
        with patch('importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError
            # Assert no exception raised
            mmcv_patch.patch_mmcv_version("666.888.2333")
            mock_import.assert_called_once_with("mmcv")

    def test_no_version_conflict(self):
        with patch('importlib.import_module') as mock_import:
            # Mock mmcv module
            mock_mmcv = MagicMock()
            mock_mmcv.__version__ = "1.7.2"
            mock_import.return_value = mock_mmcv
            
            # Call patching function
            mmcv_patch.patch_mmcv_version("1.7.2")
            
            # Assert version restoration
            self.assertEqual(mock_mmcv.__version__, "1.7.2", "Version should be restored to original")


if __name__ == '__main__':
    run_tests()