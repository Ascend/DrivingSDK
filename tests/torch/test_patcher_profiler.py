import random
import types
import tempfile

import unittest
from unittest.mock import ANY, patch, MagicMock, PropertyMock

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import mx_driving.patcher.profiler as mx_profiler


def assertIsNotInstance(obj, cls):
    assert not isinstance(obj, cls), f"Expected {repr(obj)} to NOT be an instance of {cls.__name__}"


class TestPatcherProfiler(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.profiling_path = self.temp_dir.name
          
    def test_monkeypatch(self):
        mock_runner = MagicMock()
        
        # mx_driving's profiler patcher: profiler(runner: ModuleType, options: Dict)
        # apply monkeypatch
        mx_profiler.profiler(mock_runner, 
                    {"profiling_path": self.profiling_path, 
                     "profiling_level": 0})
        
        # check replacement took place
        assertIsNotInstance(mock_runner.EpochBasedRunner.train, MagicMock) 
        assertIsNotInstance(mock_runner.EpochBasedTrainLoop.run_epoch, MagicMock)
        assertIsNotInstance(mock_runner.IterBasedTrainLoop.run, MagicMock)
        assertIsNotInstance(mock_runner.IterBasedRunner.run, MagicMock)
    
    def execute_mock_profiler(self, profiler_level, dataloader_len, max_iter, extra_args=None):
        if extra_args is None:
            val_begin = 0
            val_interval = 1
            workflow = [('train', 1)]
            batch_size = 1
        else:
            val_begin, val_interval, workflow, batch_size = extra_args
            
        mock_runner = MagicMock()
        # apply monkey patch
        mx_profiler.profiler(mock_runner, 
                    {"profiling_path": self.profiling_path, 
                    "profiling_level": profiler_level})
        '''
        Using another monkeypatch to replace torch_npu.profiler by a mock to 
        avoid intricated calling of the actual profiler, only the code logics
        within the mx_driving/.../patcher/profiler.py is examined through this 
        unit test, not the torch_npu.profiler it depends.
        '''
        with patch('torch_npu.profiler.profile') as mock_profiler, \
                patch('torch_npu.profiler.schedule') as mock_schedule, \
                patch('torch_npu.profiler._ExperimentalConfig') as mock_exp_config, \
                patch('torch_npu.profiler.tensorboard_trace_handler') as mock_trace_handler:
            # __enter__ invoked by "with torch_npu.profiler.profile(...) as prof" needs to return itself as "prof"
            mock_profiler.return_value.__enter__.return_value = mock_profiler
            
            mock_dataloader = MagicMock()
            mock_dataloader.__iter__.return_value = (torch.randn(3, 256, 256) for _ in range(dataloader_len))
            mock_dataloader.__len__.return_value = dataloader_len
            
            # execute EpochBasedRunner.train 
            mock_runner.EpochBasedRunner.train(mock_runner.EpochBasedRunner, mock_dataloader)
            
            epoch_runner_profiler = mock_profiler
            
        mock_runner = MagicMock()
        mx_profiler.profiler(mock_runner, 
                    {"profiling_path": self.profiling_path, 
                    "profiling_level": profiler_level})
        with patch('torch_npu.profiler.profile') as mock_profiler, \
                patch('torch_npu.profiler.schedule') as mock_schedule, \
                patch('torch_npu.profiler._ExperimentalConfig') as mock_exp_config, \
                patch('torch_npu.profiler.tensorboard_trace_handler') as mock_trace_handler:
            # __enter__ invoked by "with torch_npu.profiler.profile(...) as prof" needs to return itself as "prof"
            mock_profiler.return_value.__enter__.return_value = mock_profiler
            
            mock_dataloader = MagicMock()
            mock_dataloader.__iter__.return_value = (torch.randn(3, 256, 256) for _ in range(dataloader_len))
            mock_dataloader.__len__.return_value = dataloader_len
            
            mock_runner.EpochBasedTrainLoop.dataloader = mock_dataloader
            
            # execute EpochBasedTrainLoop.run_epoch
            mock_runner.EpochBasedTrainLoop.run_epoch(mock_runner.EpochBasedTrainLoop)
            
            epoch_trainloop_profiler = mock_profiler
        
        mock_runner = MagicMock()
        mx_profiler.profiler(mock_runner, 
                    {"profiling_path": self.profiling_path, 
                    "profiling_level": profiler_level})    
        with patch('torch_npu.profiler.profile') as mock_profiler, \
                patch('torch_npu.profiler.schedule') as mock_schedule, \
                patch('torch_npu.profiler._ExperimentalConfig') as mock_exp_config, \
                patch('torch_npu.profiler.tensorboard_trace_handler') as mock_trace_handler:
            # __enter__ invoked by "with torch_npu.profiler.profile(...) as prof" needs to return itself as "prof"
            mock_profiler.return_value.__enter__.return_value = mock_profiler
            
            # execute IterBasedTrainLoop.run (corresponds to "run" function in patcher's profiler.py)
            mock_runner.IterBasedTrainLoop._iter = 0
            mock_runner.IterBasedTrainLoop._max_iters = max_iter
            mock_runner.IterBasedTrainLoop.stop_training = False
            mock_runner.IterBasedTrainLoop.val_loop = MagicMock()
            mock_runner.IterBasedTrainLoop.val_begin = max_iter
            mock_runner.IterBasedTrainLoop.val_interval = max_iter
            
            def mock_run_iter(self, data_batch):
                self._iter += 1
            
            mock_runner.IterBasedTrainLoop.run_iter = types.MethodType(mock_run_iter, 
                                                                            mock_runner.IterBasedTrainLoop)
            mock_runner.IterBasedTrainLoop.run(mock_runner.IterBasedTrainLoop)
            
            iter_trainloop_profiler = mock_profiler

        mock_runner = MagicMock()
        mx_profiler.profiler(mock_runner, 
                    {"profiling_path": self.profiling_path, 
                    "profiling_level": profiler_level})
        with patch('torch_npu.profiler.profile') as mock_profiler, \
                patch('torch_npu.profiler.schedule') as mock_schedule, \
                patch('torch_npu.profiler._ExperimentalConfig') as mock_exp_config, \
                patch('torch_npu.profiler.tensorboard_trace_handler') as mock_trace_handler:
            # __enter__ invoked by "with torch_npu.profiler.profile(...) as prof" needs to return itself as "prof"
            mock_profiler.return_value.__enter__.return_value = mock_profiler
            
            mock_dataloader = MagicMock()
            mock_dataloader.__iter__.return_value = (torch.randn(3, 256, 256) for _ in range(dataloader_len))
            mock_dataloader.__len__.return_value = dataloader_len
            
            # execute IterBasedRunner.run (corresponds to "run_iter" function in patcher's profiler.py)
            mock_runner.IterBasedRunner.iter = 0
            mock_runner.IterBasedRunner._max_iters = max_iter
            
            def mock_train(self, data_loader, **kwargs):
                self.iter += 1
            
            mock_runner.IterBasedRunner.train = types.MethodType(mock_train, mock_runner.IterBasedRunner)
            
            def mock_val(self, data_loader, **kwargs):
                self.iter += 1
            
            mock_runner.IterBasedRunner.val = types.MethodType(mock_val, mock_runner.IterBasedRunner)
            mock_runner.IterBasedRunner.run(mock_runner.IterBasedRunner,
                                                [mock_dataloader] * len(workflow),
                                                workflow)
            
            iter_runner_profiler = mock_profiler
        
        return (epoch_runner_profiler, epoch_trainloop_profiler, iter_trainloop_profiler, iter_runner_profiler)

    def test_level0_configs(self):
        level = 0
        golden_level0_configs = {
            'activities': [torch_npu.profiler.ProfilerActivity.NPU],
            'with_stack': False,
            'record_shapes': False,
            'profile_memory': False,
            'schedule': ANY,
            'experimental_config': ANY,
            'on_trace_ready': ANY
        }
        
        epoch_runner_profiler, epoch_trainloop_profiler, iter_trainloop_profiler, iter_runner_profiler = \
            self.execute_mock_profiler(profiler_level=level, dataloader_len=1, max_iter=1)

        # check profiler configs
        golden_configs = golden_level0_configs
        epoch_runner_profiler.assert_called_with(**golden_configs)
        epoch_trainloop_profiler.assert_called_with(**golden_configs)
        iter_trainloop_profiler.assert_called_with(**golden_configs)
        iter_runner_profiler.assert_called_with(**golden_configs)
        # check step() getting called
        epoch_runner_profiler.step.assert_called_once()
        epoch_trainloop_profiler.step.assert_called_once()
        iter_trainloop_profiler.step.assert_called_once()
        iter_runner_profiler.step.assert_called_once()

    def test_level1_configs(self):
        level = 1
        golden_level1_configs = {
            'activities': [torch_npu.profiler.ProfilerActivity.NPU, 
                           torch_npu.profiler.ProfilerActivity.CPU],
            'with_stack': False,
            'record_shapes': True,
            'profile_memory': False,
            'schedule': ANY,
            'experimental_config': ANY,
            'on_trace_ready': ANY
        }
        
        epoch_runner_profiler, epoch_trainloop_profiler, iter_trainloop_profiler, iter_runner_profiler = \
            self.execute_mock_profiler(profiler_level=level, dataloader_len=1, max_iter=1)

        # check profiler configs
        golden_configs = golden_level1_configs
        epoch_runner_profiler.assert_called_with(**golden_configs)
        epoch_trainloop_profiler.assert_called_with(**golden_configs)
        iter_trainloop_profiler.assert_called_with(**golden_configs)
        iter_runner_profiler.assert_called_with(**golden_configs)
        # check step() getting called
        epoch_runner_profiler.step.assert_called_once()
        epoch_trainloop_profiler.step.assert_called_once()
        iter_trainloop_profiler.step.assert_called_once()
        iter_runner_profiler.step.assert_called_once()
        
    def test_level2_configs(self):
        level = 2
        golden_level2_configs = {
            'activities': [torch_npu.profiler.ProfilerActivity.NPU, 
                           torch_npu.profiler.ProfilerActivity.CPU],
            'with_stack': True,
            'record_shapes': True,
            'profile_memory': True,
            'schedule': ANY,
            'experimental_config': ANY,
            'on_trace_ready': ANY
        }
        
        epoch_runner_profiler, epoch_trainloop_profiler, iter_trainloop_profiler, iter_runner_profiler = \
            self.execute_mock_profiler(profiler_level=level, dataloader_len=1, max_iter=1)

        # check profiler configs
        golden_configs = golden_level2_configs
        epoch_runner_profiler.assert_called_with(**golden_configs)
        epoch_trainloop_profiler.assert_called_with(**golden_configs)
        iter_trainloop_profiler.assert_called_with(**golden_configs)
        iter_runner_profiler.assert_called_with(**golden_configs)
        # check step() getting called
        epoch_runner_profiler.step.assert_called_once()
        epoch_trainloop_profiler.step.assert_called_once()
        iter_trainloop_profiler.step.assert_called_once()
        iter_runner_profiler.step.assert_called_once() 

    def test_rand_len_profiling(self):
        num_databatch = random.randint(1, 100)
        num_iters = random.randint(1, 100)
        level = random.randint(0, 2)
        
        epoch_runner_profiler, epoch_trainloop_profiler, iter_trainloop_profiler, iter_runner_profiler = \
            self.execute_mock_profiler(profiler_level=level, dataloader_len=num_databatch, max_iter=num_iters)
        
        assert epoch_runner_profiler.step.call_count == num_databatch
        assert epoch_trainloop_profiler.step.call_count == num_databatch
        assert iter_trainloop_profiler.step.call_count == num_iters
        assert iter_runner_profiler.step.call_count == num_iters 
           
    def test_iter_runner_val_loop(self):
        num_databatch = 50
        num_iters = 100
        level = random.randint(0, 2)
        
        val_size = 10
        val_begin = 20
        val_interval = 50
        workflow = [('train', 50), ('val', val_size), ('train', 50)]
        batch_size = 1
        extra_args = (val_begin, val_interval, workflow, batch_size)
        
        epoch_runner_profiler, epoch_trainloop_profiler, iter_trainloop_profiler, iter_runner_profiler = \
            self.execute_mock_profiler(profiler_level=level, dataloader_len=num_databatch, max_iter=num_iters, 
                                       extra_args=extra_args)

        assert iter_trainloop_profiler.step.call_count == num_iters 
        assert iter_runner_profiler.step.call_count == num_iters
    
    def test_workflow_exception(self):
        num_databatch = 1
        num_iters = 10
        level = random.randint(0, 2)

        val_begin = num_iters
        val_interval = num_iters 
        workflow = [('train', 5), (666, 5)]
        batch_size = 1
        extra_args = (val_begin, val_interval, workflow, batch_size)

        with unittest.TestCase.assertRaises(self, ValueError):
            self.execute_mock_profiler(profiler_level=level, dataloader_len=num_databatch, max_iter=num_iters, 
                                       extra_args=extra_args)
            
    def test_max_iters_warning(self):
        mock_runner = MagicMock()
        
        # apply monkey patch
        mx_profiler.profiler(mock_runner, 
                    {"profiling_path": self.profiling_path, 
                    "profiling_level": random.randint(0, 2)})
        with patch('torch_npu.profiler.profile') as mock_profiler, \
                patch('torch_npu.profiler.schedule') as mock_schedule, \
                patch('torch_npu.profiler._ExperimentalConfig') as mock_exp_config, \
                patch('torch_npu.profiler.tensorboard_trace_handler') as mock_trace_handler:
            
            mock_runner.IterBasedRunner.iter = 0
            mock_runner.IterBasedRunner._max_iters = 0
            
            def mock_train(self, data_loader, **kwargs):
                self.iter += 1
            
            mock_runner.IterBasedRunner.train = types.MethodType(mock_train, mock_runner.IterBasedRunner)
            
            def mock_val(self, data_loader, **kwargs):
                self.iter += 1
            
            mock_runner.IterBasedRunner.val = types.MethodType(mock_val, mock_runner.IterBasedRunner)
            
            dataloader_len = 5
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
        mx_profiler.profiler(mock_runner, 
                    {"profiling_path": self.profiling_path, 
                    "profiling_level": random.randint(0, 2)})
        prop_mock.assert_called_once()
if __name__ == '__main__':
    run_tests()