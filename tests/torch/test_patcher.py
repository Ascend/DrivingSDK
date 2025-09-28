import importlib
import types
import random
import unittest
import sys
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List, Union, Dict
from types import ModuleType

from torch_npu.testing.testcase import TestCase, run_tests

from mx_driving.patcher import Patch, PatcherBuilder


def assertIsNotInstance(obj, cls):
    assert not isinstance(obj, cls), f"Expected {repr(obj)} to NOT be an instance of {cls.__name__}"


class TestPatcher(TestCase):
    def test_basic_monkey_patch(self): 
        def original_func():
            return 0

        sys.modules['mmcv.runner'] = ModuleType('mmcv.runner')
        sys.modules['mmengine.runner'] = ModuleType('mmengine.runner')
        
        mock_module = MagicMock()
        mock_module.__name__ = 'mock_module'
        sys.modules['mock_module'] = mock_module
        mock_module.xxx.yyy.func = original_func
        
        
        _fake_mmcv = MagicMock()
        _fake_mmcv.__spec__ = importlib.util.spec_from_file_location(
        'mmcv', location='/dev/null/fake_mmcv.py')   # location 随便写
        _fake_mmcv.__name__ = 'mmcv'
        sys.modules['mmcv'] = _fake_mmcv
        
        _fake_mmengine = MagicMock()
        _fake_mmengine.__spec__ = importlib.util.spec_from_file_location(
        'mmengine', location='/dev/null/fake_mmengine.py')   # location 随便写
        _fake_mmengine.__name__ = 'mmengine'
        sys.modules['mmengine'] = _fake_mmengine
        
        
        
        def my_patch(module: ModuleType, options: Dict):
            def new_func():
                return 1
            
            if hasattr(module.xxx.yyy, 'func'):
                module.xxx.yyy.func = new_func 
            else:
                raise AttributeError("func not found")
        
        
        my_patch_builder = (
            PatcherBuilder()
            .add_module_patch('mock_module', Patch(my_patch, {}))
            .with_profiling("profiling/path/", 1)
            .brake_at(1000)
        )
        with my_patch_builder.build() as patcher:
            self.assertEqual(mock_module.xxx.yyy.func(), 1)
        
    def test_blacklist(self):
        def original_func():
            return 0
        
        sys.modules['mmcv.runner'] = ModuleType('mmcv.runner')
        sys.modules['mmengine.runner'] = ModuleType('mmengine.runner')

        mock_module = MagicMock()
        mock_module.__name__ = 'mock_module'
        sys.modules['mock_module'] = mock_module
        mock_module.xxx.yyy.func = original_func
        
        def my_patch(module: ModuleType, options: Dict):
            def new_func():
                return 1
            
            if hasattr(module.xxx.yyy, 'func'):
                module.xxx.yyy.func = new_func 
            else:
                raise AttributeError("func not found")
            
        my_patch_builder = (
            PatcherBuilder()
            .add_module_patch('mock_module', Patch(my_patch, {}))
            .disable_patches('my_patch')
        )
        with my_patch_builder.build() as patcher:
            self.assertEqual(mock_module.xxx.yyy.func(), 0)
        
        
if __name__ == "__main__":
    run_tests()