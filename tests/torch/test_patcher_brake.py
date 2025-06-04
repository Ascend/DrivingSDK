import random
import types

import unittest
from unittest.mock import ANY, patch, MagicMock, PropertyMock

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import mx_driving.patcher.brake as mx_brake


def assertIsNotInstance(obj, cls):
    assert not isinstance(obj, cls), f"Expected {repr(obj)} to NOT be an instance of {cls.__name__}"



class TestPatcherBrake(TestCase):
    def setUp(self):
        pass
    
    def test_monkeypatch(self):
        mock_runner = MagicMock()
        
        # mx_driving's brake patcher: brake(runner: ModuleType, options: Dict)
        # apply monkeypatch
        mx_brake.brake(mock_runner, {'when_iter': 1000})
        
        # check replacement took place
        assertIsNotInstance(mock_runner.EpochBasedRunner.train, MagicMock) 
        assertIsNotInstance(mock_runner.EpochBasedTrainLoop.run_epoch, MagicMock)
        assertIsNotInstance(mock_runner.IterBasedTrainLoop.run, MagicMock)
        assertIsNotInstance(mock_runner.IterBasedRunner.run, MagicMock)
    
    def test_when_iter_not_an_int_exception(self):
        mock_runner = MagicMock()
        
        # apply monkeypatch
        with unittest.TestCase.assertRaises(self, ValueError):
            mx_brake.brake(mock_runner, {'when_iter': '1000'})
    
    def test_train_brake(self):
        mock_runner = MagicMock()
        
        # apply monkeypatch, brake at 1000th step
        mx_brake.brake(mock_runner, {'when_iter': 1000})
        
        dataloader_len = 2000
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__.return_value = (torch.randn(3, 256, 256) for _ in range(dataloader_len))
        mock_dataloader.__len__.return_value = dataloader_len

        mock_runner.EpochBasedRunner._iter = 0

        # execute EpochBasedRunner.train and catch brake exit(0) 
        with unittest.TestCase.assertRaises(self, SystemExit) as se:
            mock_runner.EpochBasedRunner.train(mock_runner.EpochBasedRunner, mock_dataloader)
        self.assertEqual(se.exception.code, 0)
        self.assertEqual(mock_runner.EpochBasedRunner._iter, 1000)
        
    def test_run_epoch_brake(self):
        mock_runner = MagicMock()
        
        # apply monkeypatch, brake at 1000th step
        mx_brake.brake(mock_runner, {'when_iter': 1000})
        
        dataloader_len = 2000
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__.return_value = (torch.randn(3, 256, 256) for _ in range(dataloader_len))
        mock_dataloader.__len__.return_value = dataloader_len

        mock_runner.EpochBasedTrainLoop.dataloader = mock_dataloader
        mock_runner.EpochBasedTrainLoop._iter = 0
        
        def mock_run_iter(self, idx, data_batch):
            self._iter += 1
        
        mock_runner.EpochBasedTrainLoop.run_iter = types.MethodType(mock_run_iter, 
                                                                        mock_runner.EpochBasedTrainLoop)
        
        # execute EpochBasedTrainLoop.run_epoch
        with unittest.TestCase.assertRaises(self, SystemExit) as se:
            mock_runner.EpochBasedTrainLoop.run_epoch(mock_runner.EpochBasedTrainLoop)
        self.assertEqual(se.exception.code, 0)
        self.assertEqual(mock_runner.EpochBasedTrainLoop._iter, 1000)
        
    def test_run_brake(self):
        mock_runner = MagicMock()
        
        # apply monkeypatch, brake at 1000th step
        mx_brake.brake(mock_runner, {'when_iter': 1000})

        mock_runner.IterBasedTrainLoop._iter = 0
        mock_runner.IterBasedTrainLoop._max_iters = 2000
        mock_runner.IterBasedTrainLoop.stop_training = False
        mock_runner.IterBasedTrainLoop.val_loop = MagicMock()
        mock_runner.IterBasedTrainLoop.val_begin = 100
        mock_runner.IterBasedTrainLoop.val_interval = 500
        
        def mock_run_iter(self, data_batch):
            self._iter += 1
        
        mock_runner.IterBasedTrainLoop.run_iter = types.MethodType(mock_run_iter, 
                                                                        mock_runner.IterBasedTrainLoop)
        
        
        # execute IterBasedTrainLoop.run (corresponds to "run" function in patcher's brake.py)
        with unittest.TestCase.assertRaises(self, SystemExit) as se:
            mock_runner.IterBasedTrainLoop.run(mock_runner.IterBasedTrainLoop)
        self.assertEqual(se.exception.code, 0)
        self.assertEqual(mock_runner.IterBasedTrainLoop._iter, 1000)
        
    def test_run_iter_brake(self):
        mock_runner = MagicMock()
        
        # apply monkeypatch, brake at 1000th step
        mx_brake.brake(mock_runner, {'when_iter': 1000})

        workflow = [('train', 1000), ('val', 10), ('train', 1000)]
        dataloader_len = 1000
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__.return_value = (torch.randn(3, 256, 256) for _ in range(dataloader_len))
        mock_dataloader.__len__.return_value = dataloader_len
        
        mock_runner.IterBasedRunner.iter = 0
        mock_runner.IterBasedRunner._iter = 0
        mock_runner.IterBasedRunner._max_iters = 2010
        
        def mock_train(self, data_loader, **kwargs):
            self.iter += 1
            self._iter += 1
        
        mock_runner.IterBasedRunner.train = types.MethodType(mock_train, mock_runner.IterBasedRunner)
        
        def mock_val(self, data_loader, **kwargs):
            self.iter += 1
            self._iter += 1
        
        mock_runner.IterBasedRunner.val = types.MethodType(mock_val, mock_runner.IterBasedRunner)
        
        # execute IterBasedRunner.run (corresponds to "run_iter" function in patcher's brake.py)
        with unittest.TestCase.assertRaises(self, SystemExit) as se:
            mock_runner.IterBasedRunner.run(mock_runner.IterBasedRunner,
                                                [mock_dataloader] * len(workflow),
                                                workflow)
        self.assertEqual(se.exception.code, 0)
        self.assertEqual(mock_runner.IterBasedRunner._iter, 1000)
        
    def test_workflow_exception(self):
        mock_runner = MagicMock()
        
        # apply monkeypatch, brake at 1000th step
        mx_brake.brake(mock_runner, {'when_iter': 1000})
        
        workflow = [('train', 1), (666, 888)]
        dataloader_len = 1
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__.return_value = (torch.randn(3, 256, 256) for _ in range(dataloader_len))
        mock_dataloader.__len__.return_value = dataloader_len
        
        mock_runner.IterBasedRunner.iter = 0
        mock_runner.IterBasedRunner._iter = 0
        mock_runner.IterBasedRunner._max_iters = 10
        
        def mock_train(self, data_loader, **kwargs):
            self.iter += 1
            self._iter += 1
        
        mock_runner.IterBasedRunner.train = types.MethodType(mock_train, mock_runner.IterBasedRunner)
        
        def mock_val(self, data_loader, **kwargs):
            self.iter += 1
            self._iter += 1
        
        mock_runner.IterBasedRunner.val = types.MethodType(mock_val, mock_runner.IterBasedRunner)

        with unittest.TestCase.assertRaises(self, ValueError):
            mock_runner.IterBasedRunner.run(mock_runner.IterBasedRunner,
                                                [mock_dataloader] * len(workflow),
                                                workflow)
            
    def test_max_iters_warning(self):
        mock_runner = MagicMock()
        
        # apply monkeypatch, brake at 1000th step
        mx_brake.brake(mock_runner, {'when_iter': 1000})
        
        mock_runner.IterBasedRunner.iter = 0
        mock_runner.IterBasedRunner._iter = 0
        mock_runner.IterBasedRunner._max_iters = 2010
        
        def mock_train(self, data_loader, **kwargs):
            self.iter += 1
            self._iter += 1
        
        mock_runner.IterBasedRunner.train = types.MethodType(mock_train, mock_runner.IterBasedRunner)
        
        def mock_val(self, data_loader, **kwargs):
            self.iter += 1
            self._iter += 1
        
        mock_runner.IterBasedRunner.val = types.MethodType(mock_val, mock_runner.IterBasedRunner)
        
        workflow = [('train', 1)]
        dataloader_len = 1
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__.return_value = (torch.randn(3, 256, 256) for _ in range(dataloader_len))
        mock_dataloader.__len__.return_value = dataloader_len
        
        with unittest.TestCase.assertWarns(self, DeprecationWarning):
            mock_runner.IterBasedRunner.run(mock_runner.IterBasedRunner,
                                                [mock_dataloader], [('train', 5)], 666)
    
    def test_trigger_AttributeError(self):
        mock_runner = MagicMock()
        mock_runner.iter_based_runner = MagicMock()
        prop_mock = PropertyMock(side_effect=AttributeError("IterLoader not found"))
        type(mock_runner.iter_based_runner).IterLoader = prop_mock
        mx_brake.brake(mock_runner, {'when_iter': 1000})
        prop_mock.assert_called_once()
        
    def test_setting_brake_after_training_completion(self):
        mock_runner = MagicMock()
        
        mx_brake.brake(mock_runner, {'when_iter': 9999})
        
        # EpochBasedRunner.train 
        dataloader_len = 2000
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__.return_value = (torch.randn(3, 256, 256) for _ in range(dataloader_len))
        mock_dataloader.__len__.return_value = dataloader_len
        mock_runner.EpochBasedRunner._iter = 0
        try:
            mock_runner.EpochBasedRunner.train(mock_runner.EpochBasedRunner, mock_dataloader)
        except Exception as e:
            self.fail(f"Unexpected exception: {str(e)}")
            
        # EpochBasedTrainLoop.run_epoch
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__.return_value = (torch.randn(3, 256, 256) for _ in range(dataloader_len))
        mock_dataloader.__len__.return_value = dataloader_len
        mock_runner.EpochBasedTrainLoop.dataloader = mock_dataloader
        mock_runner.EpochBasedTrainLoop._iter = 0
        
        def mock_run_iter(self, idx, data_batch):
            self._iter += 1
        
        mock_runner.EpochBasedTrainLoop.run_iter = types.MethodType(mock_run_iter, 
                                                                        mock_runner.EpochBasedTrainLoop)
        try:
            mock_runner.EpochBasedTrainLoop.run_epoch(mock_runner.EpochBasedTrainLoop)
        except Exception as e:
            self.fail(f"Unexpected exception: {str(e)}")
        
        # IterBasedTrainLoop.run (corresponds to "run" function in patcher's brake.py)
        mock_runner.IterBasedTrainLoop._iter = 0
        mock_runner.IterBasedTrainLoop._max_iters = 2000
        mock_runner.IterBasedTrainLoop.stop_training = False
        mock_runner.IterBasedTrainLoop.val_loop = MagicMock()
        mock_runner.IterBasedTrainLoop.val_begin = 100
        mock_runner.IterBasedTrainLoop.val_interval = 500
        
        def mock_run_iter(self, data_batch):
            self._iter += 1
        
        mock_runner.IterBasedTrainLoop.run_iter = types.MethodType(mock_run_iter, 
                                                                        mock_runner.IterBasedTrainLoop)
        try:
            mock_runner.IterBasedTrainLoop.run(mock_runner.IterBasedTrainLoop)  
        except Exception as e:
            self.fail(f"Unexpected exception: {str(e)}")  
        
        # IterBasedRunner.run (corresponds to "run_iter" function in patcher's brake.py) 
        workflow = [('train', 1000), ('val', 10), ('train', 1000)]
        dataloader_len = 1000
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__.return_value = (torch.randn(3, 256, 256) for _ in range(dataloader_len))
        mock_dataloader.__len__.return_value = dataloader_len
        
        mock_runner.IterBasedRunner.iter = 0
        mock_runner.IterBasedRunner._iter = 0
        mock_runner.IterBasedRunner._max_iters = 2010
        
        def mock_train(self, data_loader, **kwargs):
            self.iter += 1
            self._iter += 1
        
        mock_runner.IterBasedRunner.train = types.MethodType(mock_train, mock_runner.IterBasedRunner)
        
        def mock_val(self, data_loader, **kwargs):
            self.iter += 1
            self._iter += 1
        
        mock_runner.IterBasedRunner.val = types.MethodType(mock_val, mock_runner.IterBasedRunner)
        try:
            mock_runner.IterBasedRunner.run(mock_runner.IterBasedRunner,
                                                [mock_dataloader] * len(workflow),
                                                workflow)    
        except Exception as e:
            self.fail(f"Unexpected exception: {str(e)}") 
        
if __name__ == '__main__':
    run_tests()