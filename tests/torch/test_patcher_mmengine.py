import random
import types
import sys
from types import ModuleType

import unittest
from unittest.mock import ANY, patch, MagicMock, PropertyMock, Mock

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from mx_driving.patcher import optimizer_hooks, optimizer_wrapper
from mx_driving.patcher import mmcv_patch, mmengine_patch


''' 
    MMCV 1.x modules = MMCV 2.x moudles + MMEngine modules, 
    therefore, some module tested here correponds to mmcv instead of mmengine 
'''


def assertIsNotInstance(obj, cls):
    assert not isinstance(obj, cls), f"Expected {repr(obj)} to NOT be an instance of {cls.__name__}"


class EmptyAttribute:
    pass


# For mmcv 1.x
class TestOptimizerHooks(TestCase):
    def setUp(self):
        class UnifiedMeta(type):
            pass
        
        class MockHook(metaclass=UnifiedMeta):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
            
            def every_n_iters(self, runner, n):
                return (runner.iter % n == 0) if n > 0 else False
    
            def is_last_iter(self, runner):
                return runner.iter == runner.max_iters
        
        # Create mock simulated registry which supports save and retrieval
        class MockRegistry:
            def __init__(self):
                self._registry = {}
            
            def register_module(self, name=None, force=False):
                def decorator(cls):
                    key = name or cls.__name__
                    self._registry[key] = cls
                    return cls
                return decorator
            
            def get(self, name):    
                return self._registry.get(name)
        
        self.mock_registry = MockRegistry()
        
        self.mmcv = ModuleType('mmcv')
        self.mmcv.__name__ = 'mmcv'
        self.mmcv.runner = ModuleType('runner')
        self.mmcv.runner.hooks = ModuleType('hooks')
        self.mmcvhooks = self.mmcv.runner.hooks
        self.mmcvhooks.optimizer = ModuleType('optimizer')
        self.mmcvhooks.optimizer.HOOKS = self.mock_registry
        self.mmcvhooks.optimizer.Hook = MockHook
        
        class DummyBatchNorm:
            pass
        
        self.mmcvhooks.optimizer._BatchNorm = DummyBatchNorm
        
        # Mock dependencies
        self.mmcvhooks.optimizer.logging = MagicMock()
        self.mmcvhooks.optimizer.GradScaler = MagicMock()
        self.mmcvhooks.optimizer.wrap_fp16_model = MagicMock()
        self.mmcvhooks.optimizer.Tensor = MagicMock()
        self.mmcvhooks.optimizer.nn = MagicMock()
        
        # Apply patch and verify
        optimizer_hooks(self.mmcv, {})
        self.assertEqual(len(self.mock_registry._registry), 4, "4 hook classes should be registered")
        
        # Fetch classes registered by patcher's hook decorator
        self.OptimizerHook = self.mock_registry.get('OptimizerHook')
        self.GradientCumulativeHook = self.mock_registry.get('GradientCumulativeOptimizerHook')
        self.Fp16Hook = self.mock_registry.get('Fp16OptimizerHook')
        self.GradientCumulativeFp16Hook = self.mock_registry.get('GradientCumulativeFp16OptimizerHook')  
        
        # Verify registry
        self.assertIsNotNone(self.OptimizerHook)
        self.assertIsNotNone(self.GradientCumulativeHook)
        self.assertIsNotNone(self.Fp16Hook)
        self.assertIsNotNone(self.GradientCumulativeFp16Hook)       

    # -------------------Test for OptimizerHook(Hook)--------------------#
    def test_oh_clip_grads(self):
        hook = self.OptimizerHook(grad_clip={'max_norm': 10})
        runner = MagicMock()
        runner.model.parameters.return_value = [MagicMock(grad=MagicMock())]
        
        # Test with gradients present
        hook.clip_grads(runner.model.parameters(), runner)
        runner.optimizer.clip_grad_norm_fused_.assert_called_once()

        # Test without gradients
        runner.reset_mock()
        runner.model.parameters.return_value = []
        result = hook.clip_grads(runner.model.parameters(), runner)
        self.assertIsNone(result)

    def test_oh_after_train_iter(self):
        hook = self.OptimizerHook(grad_clip={'max_norm': 10})
        runner = MagicMock()
        runner.model.parameters.return_value = [MagicMock(grad=MagicMock())]
        runner.outputs = {'loss': MagicMock(), 'num_samples': 1}
        
        hook.after_train_iter(runner)
        
        # Verify call sequence
        runner.optimizer.zero_grad.assert_called_once()
        runner.outputs['loss'].backward.assert_called_once()
        runner.optimizer.step.assert_called_once()

    def test_oh_anomaly_detection(self):
        hook = self.OptimizerHook(detect_anomalous_params=True)
        runner = MagicMock()
        
        # Create mock loss function
        mock_grad_fn = MagicMock()
        mock_grad_fn.variable = 'valid_param'
        mock_grad_fn.next_functions = [(None, None)]
        
        mock_loss = MagicMock()
        mock_loss.grad_fn = mock_grad_fn
        
        runner.outputs = {'loss': mock_loss, 'num_samples': 1}
        
        # Create model parameters
        mock_param = MagicMock()
        mock_param.requires_grad = True
        
        runner.model.named_parameters.return_value = [
            ('valid_param', 'valid_param'),
            ('anomalous_param', mock_param)
        ]
        
        # Execute detection
        hook.detect_anomalous_parameters(mock_loss, runner)
        runner.logger.log.assert_called_once()
        
        # Cover dectection called by after_train_iter
        hook.after_train_iter(runner)
        runner.logger.log.assert_called()

    # ------Test for GradientCumulativeOptimizerHook(OptimizerHook)--------#            
    def test_gch_constructor_exceptions(self):
        
        with unittest.TestCase.assertRaises(self, ValueError):
            self.GradientCumulativeHook(cumulative_iters='Not Int')
            
        with unittest.TestCase.assertRaises(self, ValueError):
            self.GradientCumulativeHook(cumulative_iters=-1)

    def test_gch_has_batch(self):
        hook = self.GradientCumulativeHook(cumulative_iters=1)
        
        bn_instance = self.mmcvhooks.optimizer._BatchNorm()
        
        # Test directly having BatchNorm
        mock_module = MagicMock()
        mock_module.children.return_value = [bn_instance]
        self.assertTrue(hook.has_batch_norm(mock_module))
        
        # Test child module has BatchNorm
        child_module = MagicMock()
        child_module.children.return_value = [bn_instance]
        parent_module = MagicMock()
        parent_module.children.return_value = [child_module]
        self.assertTrue(hook.has_batch_norm(parent_module))
        
        # Test no BatchNorm
        no_bn_module = MagicMock()
        no_bn_module.children.return_value = [MagicMock(), MagicMock()]
        self.assertFalse(hook.has_batch_norm(no_bn_module))

    def test_gch_init(self):
        hook = self.GradientCumulativeHook(cumulative_iters=5)
        
        runner = MagicMock()
        runner.iter = 6
        runner.max_iters = 8

        bn_instance = self.mmcvhooks.optimizer._BatchNorm()
        runner.model.children.return_value = [bn_instance]
                
        hook._init(runner)
        self.assertEqual(hook.divisible_iters, 5)
        self.assertEqual(hook.remainder_iters, 3)

    def test_gch_loss_factor(self):
        hook = self.GradientCumulativeHook(cumulative_iters=5)
        runner = MagicMock()
        runner.max_iters = 18
        hook._init(runner)
        
        # Test normal iterations
        runner.iter = 10
        self.assertEqual(hook._get_loss_factor(runner), 5)
        
        # Test remainder iterations
        runner.iter = 17
        self.assertEqual(hook._get_loss_factor(runner), 3)
        
        # Test exception
        runner.max_iters = 1
        runner.iter = 100
        hook.remainder_iters = -1
        with unittest.TestCase.assertRaises(self, ValueError):
            hook._get_loss_factor(runner)

    def test_gch_after_train_iter(self):
        hook = self.GradientCumulativeHook(cumulative_iters=4, grad_clip={'max_norm': 10})
        runner = MagicMock()
        runner.iter = 4
        runner.max_iters = 20
        runner.outputs = {'loss': MagicMock(), 'num_samples': 1}
        
        # Test with gradients present
        runner.model.parameters.return_value = [MagicMock(grad=MagicMock())]
        
        hook.initialized = False
        hook.after_train_iter(runner)
        
        # Validate accumulation logic
        runner.outputs['loss'].__truediv__.assert_called_once()
        runner.optimizer.step.assert_called_once()
        runner.optimizer.zero_grad.assert_called_once()

    # ---------------Test for Fp16OptimizerHook(OptimizerHook)-----------------#            
    def test_f16h_constructor(self):
        # Test dynamic loss scale
        hook = self.Fp16Hook(loss_scale='dynamic')
        self.assertIsNotNone(hook.loss_scaler)
        
        # Test fixed loss scale
        hook = self.Fp16Hook(loss_scale=512.0)
        self.assertEqual(hook._scale_update_param, 512.0)
        
        # Test dictionary configuration
        hook = self.Fp16Hook(loss_scale={'init_scale': 256})
        self.assertIsNotNone(hook.loss_scaler)
        
        # Test exception
        with unittest.TestCase.assertRaises(self, ValueError):
            self.Fp16Hook(loss_scale='InvalidValue')

    def test_f16h_before_run(self):
        hook = self.Fp16Hook(loss_scale=512.0)
        runner = MagicMock()
        state_dict = MagicMock()
        runner.meta = {'fp16': {'loss_scaler': state_dict}}
        
        hook.before_run(runner)
        self.mmcvhooks.optimizer.wrap_fp16_model.assert_called_once_with(runner.model)
        hook.loss_scaler.load_state_dict.assert_called_once_with(state_dict)

    def test_f16h_after_train_iter(self):
        hook = self.Fp16Hook(grad_clip={'max_norm': 10})
        runner = MagicMock()
        runner.outputs = {'loss': MagicMock(), 'num_samples': 1}
        runner.meta = {}
        
        # Test with gradients present
        runner.model.parameters.return_value = [MagicMock(grad=MagicMock())]
        
        # Mock loss scalar's behavior
        hook.loss_scaler.scale.return_value = runner.outputs['loss']
        
        hook.after_train_iter(runner)
        
        # Verify FP16-specific calls
        hook.loss_scaler.scale.assert_called_once()
        hook.loss_scaler.unscale_.assert_called_once()
        hook.loss_scaler.step.assert_called_once()
        hook.loss_scaler.update.assert_called_once()

    def test_f16h_state_saving(self):
        # Test FP16 grad scaler state correctly saved to runner.meta
        hook = self.Fp16Hook(loss_scale=512.0)
        runner = MagicMock()
        runner.meta = {}
        runner.outputs = {'loss': MagicMock(), 'num_samples': 1}
        
        hook.after_train_iter(runner)
        
        # Verify scaler state being saved
        self.assertIn('fp16', runner.meta)
        self.assertIn('loss_scaler', runner.meta['fp16'])
        hook.loss_scaler.state_dict.assert_called_once()

    def test_f16h_grad_clipping(self):
        # Test gradient clipping in FP16 mix precision optimizier hook
        hook = self.Fp16Hook(grad_clip={'max_norm': 10})
        runner = MagicMock()
        runner.outputs = {'loss': MagicMock(), 'num_samples': 1}
        
        # Mock grad existence
        runner.model.parameters.return_value = [MagicMock(grad=MagicMock())]
        
        hook.after_train_iter(runner)
        
        hook.loss_scaler.unscale_.assert_called_once()
        runner.optimizer.clip_grad_norm_fused_.assert_called_once()

    def test_f16h_copy_grads_to_fp32(self, device="npu"):
        hook = self.Fp16Hook(loss_scale=512.0)
        
        fp16_grad = torch.tensor([1.0], dtype=torch.float16, device=device)
        fp16_param = MagicMock(grad=fp16_grad)
        
        fp32_weight = torch.zeros(1, dtype=torch.float32, device=device)
        fp32_weight.grad = None  
        
        hook.copy_grads_to_fp32(
            fp16_net=MagicMock(parameters=MagicMock(return_value=[fp16_param])),
            fp32_weights=[fp32_weight]
        )
        
        self.assertTrue(torch.allclose(fp32_weight.grad, fp16_grad.float()))

    def test_f16h_copy_params_to_fp16(self, device="npu"):
        hook = self.Fp16Hook(loss_scale=512.0)
        
        fp32_weight = torch.tensor([2.0], dtype=torch.float32, device=device)
        
        fp16_param = torch.zeros(1, dtype=torch.float16, device=device)
        fp16_net = MagicMock(parameters=MagicMock(return_value=[fp16_param]))
        
        hook.copy_params_to_fp16(fp16_net, [fp32_weight])
        
        self.assertRtolEqual(fp16_param, fp32_weight.half())

    # ---Test for GradientCumulativeFp16OptimizerHook(GradientCumulativeOptimizerHook, Fp16OptimizerHook)---#
    def test_gcf16h_after_train_iter(self):
        hook = self.GradientCumulativeFp16Hook(cumulative_iters=3)
        runner = MagicMock()
        runner.iter = 9
        runner.max_iters = 15
        runner.outputs = {'loss': MagicMock(), 'num_samples': 1}
        hook.initialized = True
        
        hook.after_train_iter(runner)
        
        hook.loss_scaler.step.assert_called_once()
        runner.model.zero_grad.assert_called_once()
        runner.optimizer.zero_grad.assert_called_once()   


class TestOptimizerWrapperPatch(TestCase):
    def test_optimizer_wrapper_patch(self):
        # Create mock
        mmengine = ModuleType('mmengine')
        
        mmengine.__name__ = 'mmengine'
        sys.modules['mmengine.optim'] = ModuleType('mmengine.optim')
        
        mmengine.optim = ModuleType('optim')
        mmengine.optim.optimizer = ModuleType('optimizer')
        mmengine.optim.optimizer.optimizer_wrapper = ModuleType('optimizer_wrapper')
        
        class OptimWrapper:
            def __init__(self, optimizer):
                self.optimizer = optimizer
        mmengine.optim.optimizer.optimizer_wrapper.OptimWrapper = OptimWrapper
        
        # Keep original __init__
        orig_init = OptimWrapper.__init__
        
        # Apply patch
        optimizer_wrapper(mmengine, {})
        
        # Create mock optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.clip_grad_norm_fused_ = MagicMock()
        
        # Instantiate to trigger calling of new_init
        wrapper = mmengine.optim.optimizer.optimizer_wrapper.OptimWrapper(mock_optimizer)
        
        # Validate clip_grads existence
        self.assertTrue(hasattr(wrapper, 'clip_grads'))
        
        # Call and Verify
        wrapper.clip_grads(params='params', max_norm=10)
        mock_optimizer.clip_grad_norm_fused_.assert_called_once_with(max_norm=10)

    def test_patch_failure(self):
        mock_mmengine = MagicMock()
        mock_mmengine.__name__ = 'mmengine'
        sys.modules['mmengine.optim'] = ModuleType('mmengine.optim')
        with self.assertRaises(AttributeError):
            mock_mmengine.optim.optimizer.optimizer_wrapper = EmptyAttribute
            mmengine_patch.optimizer_wrapper(mock_mmengine, {})


class TestStreamPatch(TestCase):
    def setUp(self):
        # Create mock mmcvparallel module
        self.mock_mmcv = types.ModuleType('mmcv')
        self.mock_mmcv.__name__ = 'mmcv'
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

    def test_patch_failure(self):
        _mock_mmcv = MagicMock()
        _mock_mmcv.__name__ = 'mmcv'
        with self.assertRaises(AttributeError):
            _mock_mmcv.parallel._functions.Scatter = EmptyAttribute
            mmengine_patch.stream(_mock_mmcv, {})

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


class TestDdpPatch(TestCase):
    def test_monkey_patch(self):
        mock_mmcv = MagicMock()
        mock_mmcv.__name__ = 'mmcv'
        mock_mmcv.device.npu.NPUDistributedDataParallel = types.ModuleType('npuddp')
        mmddp_b4replacement = mock_mmcv.parallel.distributed.MMDistributedDataParallel
        
        # Apply monkey patch
        mmengine_patch.ddp(mock_mmcv, {})
        assertIsNotInstance(mmddp_b4replacement._run_ddp_forward, MagicMock)
        assertIsNotInstance(mock_mmcv.parallel.distributed.MMDistributedDataParallel, MagicMock)

    def test_patch_failure(self):
        mock_mmcv = MagicMock()
        mock_mmcv.__name__ = 'mmcv'
        
        with self.assertRaises(AttributeError):
            mock_mmcv.parallel.distributed.MMDistributedDataParallel = EmptyAttribute
            mmengine_patch.ddp(mock_mmcv, {})

    def test_with_device_ids(self):
        mock_self = MagicMock()
        
        mock_self.device_ids = [0]
        
        mock_inputs = [('input1', 'input2')]
        mock_kwargs = [{'key1': 'value1'}]
        mock_self.to_kwargs.return_value = (mock_inputs, mock_kwargs)
        
        expected_result = "mock_result"
        mock_self.module.return_value = expected_result
        
        test_inputs = ('arg1', 'arg2')
        test_kwargs = {'kwarg1': 'value1'}
        
        mock_mmcv = MagicMock()
        mock_mmcv.__name__ = 'mmcv'
        
        # In actual MMCV library, npuddp extends from the mmddp class, here setting them as the same
        mock_mmcv.device.npu.NPUDistributedDataParallel = mock_mmcv.parallel.distributed.MMDistributedDataParallel
        
        mmengine_patch.ddp(mock_mmcv, {})
        result = mock_mmcv.parallel.distributed.MMDistributedDataParallel._run_ddp_forward(mock_self, *test_inputs, **test_kwargs)
        
        mock_self.to_kwargs.assert_called_once_with(test_inputs, test_kwargs, 0)
        
        self.assertEqual(result, expected_result)

    def test_without_device_ids(self):
        mock_self = MagicMock()
        
        mock_self.device_ids = None
        
        expected_result = "mock_result_no_devices"
        mock_self.module.return_value = expected_result
        
        test_inputs = ('arg1', 'arg2')
        test_kwargs = {'kwarg1': 'value1'}
        
        mock_mmcv = MagicMock()
        mock_mmcv.__name__ = 'mmcv'
        
        # In actual MMCV library, npuddp extends from the mmddp class, here setting them as the same
        mock_mmcv.device.npu.NPUDistributedDataParallel = mock_mmcv.parallel.distributed.MMDistributedDataParallel
        
        
        mmengine_patch.ddp(mock_mmcv, {})
        result = mock_mmcv.parallel.distributed.MMDistributedDataParallel._run_ddp_forward(mock_self, *test_inputs, **test_kwargs)
        
        mock_self.to_kwargs.assert_not_called()
        
        mock_self.module.assert_called_once_with(*test_inputs, **test_kwargs)
        
        self.assertEqual(result, expected_result)

# pylint: disable=protected-access
class TestProfiler(TestCase):
    def test_parse_profiler_options_level0(self):
        options = {}
        options['profiling_path'] = "/xxx/yyy/level0"
        options['profiling_level'] = 0
        path, level, activities, profiler_level, step_ctrl = mmengine_patch._parse_profiler_options(options)
        self.assertEqual(path, options['profiling_path'])
        self.assertEqual(level, options['profiling_level'])
        self.assertIn(torch_npu.profiler.ProfilerActivity.NPU, activities)
        self.assertNotIn(torch_npu.profiler.ProfilerActivity.CPU, activities)
        self.assertEqual(profiler_level, torch_npu.profiler.ProfilerLevel.Level0)

    def test_parse_profiler_options_level1(self):
        options = {}
        options['profiling_path'] = "/xxx/yyy/level1"
        options['profiling_level'] = 1
        path, level, activities, profiler_level, step_ctrl = mmengine_patch._parse_profiler_options(options)
        self.assertEqual(path, options['profiling_path'])
        self.assertEqual(level, options['profiling_level'])
        self.assertIn(torch_npu.profiler.ProfilerActivity.NPU, activities)
        self.assertIn(torch_npu.profiler.ProfilerActivity.CPU, activities)
        self.assertEqual(profiler_level, torch_npu.profiler.ProfilerLevel.Level1)

    def test_parse_profiler_options_level2(self):
        options = {}
        options['profiling_path'] = "/xxx/yyy/level2"
        options['profiling_level'] = 2
        path, level, activities, profiler_level, step_ctrl = mmengine_patch._parse_profiler_options(options)
        self.assertEqual(path, options['profiling_path'])
        self.assertEqual(level, options['profiling_level'])
        self.assertIn(torch_npu.profiler.ProfilerActivity.NPU, activities)
        self.assertIn(torch_npu.profiler.ProfilerActivity.CPU, activities)
        self.assertEqual(profiler_level, torch_npu.profiler.ProfilerLevel.Level1)
        
    def test_parse_profiler_options_invalid_level(self):
        options = {}
        options['profiling_path'] = "/xxx/+++/level2"
        options['profiling_level'] = -1
        with self.assertRaises(ValueError):
            path, level, activities, profiler_level, step_ctrl = mmengine_patch._parse_profiler_options(options)
        
        options['profiling_level'] = 3
        with self.assertRaises(ValueError):
            path, level, activities, profiler_level, step_ctrl = mmengine_patch._parse_profiler_options(options)
    
    def test_parse_default_step_ctrl(self):
        options = {}
        options['profiling_path'] = "/xxx/yyy/levelx"
        options['profiling_level'] = random.randint(0, 2)
        
        path, level, activities, profiler_level, step_ctrl = mmengine_patch._parse_profiler_options(options)
        self.assertEqual(path, options['profiling_path'])
        self.assertEqual(level, options['profiling_level'])
        wait, warmup, active, repeat, skip_first = step_ctrl
        self.assertEqual(wait, 1)
        self.assertEqual(warmup, 1)
        self.assertEqual(active, 1)
        self.assertEqual(repeat, 1)
        self.assertEqual(skip_first, 20)
        
    def test_parse_find_grained_step_ctrl(self):
        options = {}
        options['profiling_path'] = "/xxx/yyy/levelx"
        options['profiling_level'] = random.randint(0, 2)
        options['step_ctrl'] = (1, 2, 3, 4, 5)
        
        path, level, activities, profiler_level, step_ctrl = mmengine_patch._parse_profiler_options(options)
        self.assertEqual(path, options['profiling_path'])
        self.assertEqual(level, options['profiling_level'])
        wait, warmup, active, repeat, skip_first = step_ctrl
        self.assertEqual(wait, 1)
        self.assertEqual(warmup, 2)
        self.assertEqual(active, 3)
        self.assertEqual(repeat, 4)
        self.assertEqual(skip_first, 5)
        
    def _execute_mock_profiler(self, profiler_level, dataloader_len, max_iter, extra_args=None, initial_iter=0):
        if extra_args is None:
            val_begin = 0
            val_interval = 1
            workflow = [('train', 1)]
            batch_size = 1
        else:
            val_begin, val_interval, workflow, batch_size = extra_args
            
        mock_mmcv = MagicMock()
        mock_mmcv.__name__ = 'mmcv'
        mock_mmcv.runner.EpochBasedRunner._iter = initial_iter
        mmengine_patch.epoch_runner(mock_mmcv, {'enable_profiler': True, 
                                                'enable_brake': True,
                                                'brake_step': 1000,
                                                'profiling_path': "/.../profiling_path/",
                                                'profiling_level': profiler_level})    
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
            mock_mmcv.runner.EpochBasedRunner.train(mock_mmcv.runner.EpochBasedRunner, mock_dataloader)
            
            epoch_runner_profiler = mock_profiler
  
  
        mock_mmcv = MagicMock()
        mock_mmcv.__name__ = 'mmcv'
        mock_mmcv.runner.EpochBasedTrainLoop._iter = initial_iter
        mmengine_patch.epoch_train_loop(mock_mmcv, {'enable_profiler': True,
                                                    'enable_brake': True,
                                                    'brake_step': 1000,
                                                    'profiling_path': "/.../profiling_path/",
                                                    'profiling_level': profiler_level})   
        with patch('torch_npu.profiler.profile') as mock_profiler, \
                patch('torch_npu.profiler.schedule') as mock_schedule, \
                patch('torch_npu.profiler._ExperimentalConfig') as mock_exp_config, \
                patch('torch_npu.profiler.tensorboard_trace_handler') as mock_trace_handler:
            # __enter__ invoked by "with torch_npu.profiler.profile(...) as prof" needs to return itself as "prof"
            mock_profiler.return_value.__enter__.return_value = mock_profiler
            
            mock_dataloader = MagicMock()
            mock_dataloader.__iter__.return_value = (torch.randn(3, 256, 256) for _ in range(dataloader_len))
            mock_dataloader.__len__.return_value = dataloader_len
            
            mock_mmcv.runner.EpochBasedTrainLoop.dataloader = mock_dataloader
            
            # execute EpochBasedTrainLoop.run_epoch
            mock_mmcv.runner.EpochBasedTrainLoop.run_epoch(mock_mmcv.runner.EpochBasedTrainLoop)
            
            epoch_trainloop_profiler = mock_profiler
        

        mock_mmcv = MagicMock()
        mock_mmcv.__name__ = 'mmcv'
        mmengine_patch.iter_train_loop(mock_mmcv, {'enable_profiler': True,
                                                   'enable_brake': True,
                                                   'brake_step': 1001,
                                                   'profiling_path': "/.../profiling_path/",
                                                   'profiling_level': profiler_level})
        with patch('torch_npu.profiler.profile') as mock_profiler, \
                patch('torch_npu.profiler.schedule') as mock_schedule, \
                patch('torch_npu.profiler._ExperimentalConfig') as mock_exp_config, \
                patch('torch_npu.profiler.tensorboard_trace_handler') as mock_trace_handler:
            # __enter__ invoked by "with torch_npu.profiler.profile(...) as prof" needs to return itself as "prof"
            mock_profiler.return_value.__enter__.return_value = mock_profiler
            
            # execute IterBasedTrainLoop.run (corresponds to "run" function in patcher's profiler.py)
            mock_mmcv.runner.IterBasedTrainLoop._iter = initial_iter
            mock_mmcv.runner.IterBasedTrainLoop._max_iters = max_iter
            mock_mmcv.runner.IterBasedTrainLoop.stop_training = False
            mock_mmcv.runner.IterBasedTrainLoop.val_loop = MagicMock()
            mock_mmcv.runner.IterBasedTrainLoop.val_begin = max_iter
            mock_mmcv.runner.IterBasedTrainLoop.val_interval = max_iter
            
            def mock_run_iter(self, data_batch):
                self._iter += 1
            
            mock_mmcv.runner.IterBasedTrainLoop.run_iter = types.MethodType(mock_run_iter, 
                                                                            mock_mmcv.runner.IterBasedTrainLoop)
            mock_mmcv.runner.IterBasedTrainLoop.run(mock_mmcv.runner.IterBasedTrainLoop)
            
            iter_trainloop_profiler = mock_profiler


        mock_mmcv = MagicMock()
        mock_mmcv.__name__ = 'mmcv'
        mmengine_patch.iter_runner(mock_mmcv, {'enable_profiler': True,
                                               'enable_brake': True,
                                               'brake_step': 1001,
                                               'profiling_path': "/.../profiling_path/",
                                               'profiling_level': profiler_level})
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
            mock_mmcv.runner.IterBasedRunner.iter = initial_iter
            mock_mmcv.runner.IterBasedRunner._max_iters = max_iter
            
            # run actually calls iter_runner, which came from getattr of mode param input as either train or val
            def mock_train(self, data_loader, **kwargs):
                self.iter += 1
            
            mock_mmcv.runner.IterBasedRunner.train = types.MethodType(mock_train, mock_mmcv.runner.IterBasedRunner)
            
            def mock_val(self, data_loader, **kwargs):
                self.iter += 1
            
            mock_mmcv.runner.IterBasedRunner.val = types.MethodType(mock_val, mock_mmcv.runner.IterBasedRunner)
            mock_mmcv.runner.IterBasedRunner.run(mock_mmcv.runner.IterBasedRunner,
                                                [mock_dataloader] * len(workflow),
                                                workflow)
            
            iter_runner_profiler = mock_profiler
        
        return (epoch_runner_profiler, epoch_trainloop_profiler, iter_trainloop_profiler, iter_runner_profiler)

    def test_profiler_and_brake(self):
        with unittest.TestCase.assertRaises(self, SystemExit) as se:
            self._execute_mock_profiler(profiler_level=1, dataloader_len=1001, max_iter=1001, initial_iter=1000)

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
            self._execute_mock_profiler(profiler_level=level, dataloader_len=1, max_iter=1)

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
            self._execute_mock_profiler(profiler_level=level, dataloader_len=1, max_iter=1)

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
            self._execute_mock_profiler(profiler_level=level, dataloader_len=1, max_iter=1)

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
        num_databatch = random.randint(1, 10)
        num_iters = random.randint(1, 10)
        level = random.randint(0, 2)
        
        epoch_runner_profiler, epoch_trainloop_profiler, iter_trainloop_profiler, iter_runner_profiler = \
            self._execute_mock_profiler(profiler_level=level, dataloader_len=num_databatch, max_iter=num_iters)
        
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
            self._execute_mock_profiler(profiler_level=level, dataloader_len=num_databatch, max_iter=num_iters, 
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
            self._execute_mock_profiler(profiler_level=level, dataloader_len=num_databatch, max_iter=num_iters, 
                                       extra_args=extra_args)
         
    def test_max_iters_warning(self):
        mock_mmcv = MagicMock()
        mock_mmcv.__name__ = 'mmcv'
        mmengine_patch.iter_runner(mock_mmcv, {'enable_profiler': True, 'enable_brake': False,
                                                'profiling_path': "/.../profiling_path/",
                                                'profiling_level': random.randint(0, 2)})
        with patch('torch_npu.profiler.profile') as mock_profiler, \
                patch('torch_npu.profiler.schedule') as mock_schedule, \
                patch('torch_npu.profiler._ExperimentalConfig') as mock_exp_config, \
                patch('torch_npu.profiler.tensorboard_trace_handler') as mock_trace_handler:
            
            mock_mmcv.runner.IterBasedRunner.iter = 0
            mock_mmcv.runner.IterBasedRunner._max_iters = 0
            
            def mock_train(self, data_loader, **kwargs):
                self.iter += 1
            
            mock_mmcv.runner.IterBasedRunner.train = types.MethodType(mock_train, mock_mmcv.runner.IterBasedRunner)
            
            def mock_val(self, data_loader, **kwargs):
                self.iter += 1
            
            mock_mmcv.runner.IterBasedRunner.val = types.MethodType(mock_val, mock_mmcv.runner.IterBasedRunner)
            
            dataloader_len = 5
            mock_dataloader = MagicMock()
            mock_dataloader.__iter__.return_value = (torch.randn(3, 256, 256) for _ in range(dataloader_len))
            mock_dataloader.__len__.return_value = dataloader_len
            
            with unittest.TestCase.assertWarns(self, DeprecationWarning):
                mock_mmcv.runner.IterBasedRunner.run(mock_mmcv.runner.IterBasedRunner,
                                                    [mock_dataloader], [('train', 5)], 666)

    def test_patch_failure(self):        
        mock_mmcv = MagicMock()
        mock_mmcv.__name__ = 'mmcv'
        with self.assertRaises(AttributeError):
            mock_mmcv.runner.EpochBasedRunner = EmptyAttribute
            mmengine_patch.epoch_runner(mock_mmcv, {'enable_profiler': True, 'enable_brake': False,
                                                'profiling_path': "/.../profiling_path/",
                                                'profiling_level': random.randint(0, 2)})
            
        
        mock_mmcv = MagicMock()
        mock_mmcv.__name__ = 'mmcv'
        with self.assertRaises(AttributeError):
            mock_mmcv.runner.IterBasedRunner = EmptyAttribute
            mmengine_patch.iter_runner(mock_mmcv, {'enable_profiler': True, 'enable_brake': False,
                                                'profiling_path': "/.../profiling_path/",
                                                'profiling_level': random.randint(0, 2)})

        
        mock_mmengine = MagicMock()
        mock_mmengine.__name__ = 'mmengine'
        sys.modules['mmengine.runner'] = ModuleType('mmengine.runner')
        with self.assertRaises(AttributeError):
            mock_mmengine.runner.EpochBasedTrainLoop = EmptyAttribute
            mmengine_patch.epoch_train_loop(mock_mmengine, {'enable_profiler': True, 'enable_brake': False,
                                                'profiling_path': "/.../profiling_path/",
                                                'profiling_level': random.randint(0, 2)})
            
            
        mock_mmengine = MagicMock()
        mock_mmengine.__name__ = 'mmengine'
        sys.modules['mmengine.runner'] = ModuleType('mmengine.runner')
        with self.assertRaises(AttributeError):
            mock_mmengine.runner.IterBasedTrainLoop = EmptyAttribute
            mmengine_patch.iter_train_loop(mock_mmengine, {'enable_profiler': True, 'enable_brake': False,
                                                'profiling_path': "/.../profiling_path/",
                                                'profiling_level': random.randint(0, 2)})


class TestBrake(TestCase):
    def test_epoch_runner_brake(self):
        mock_mmcv = MagicMock()
        mock_mmcv.__name__ = 'mmcv'
        
        # apply monkeypatch, brake at 1000th step
        mmengine_patch.epoch_runner(mock_mmcv, {'enable_profiler': False, 
                                                'enable_brake': True,
                                                'brake_step': 1000})
        
        dataloader_len = 2000
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__.return_value = (torch.randn(3, 256, 256) for _ in range(dataloader_len))
        mock_dataloader.__len__.return_value = dataloader_len

        mock_mmcv.runner.EpochBasedRunner._iter = 0

        # execute EpochBasedRunner.train and catch brake exit(0) 
        with unittest.TestCase.assertRaises(self, SystemExit) as se:
            mock_mmcv.runner.EpochBasedRunner.train(mock_mmcv.runner.EpochBasedRunner, mock_dataloader)
        self.assertEqual(se.exception.code, 0)
        self.assertEqual(mock_mmcv.runner.EpochBasedRunner._iter, 1000)

    def test_epoch_train_loop_brake(self):
        mock_mmcv = MagicMock()
        mock_mmcv.__name__ = 'mmcv'
        
        # apply monkeypatch, brake at 1000th step
        mmengine_patch.epoch_train_loop(mock_mmcv, {'enable_profiler': False, 
                                                    'enable_brake': True,
                                                    'brake_step': 1000})
        
        dataloader_len = 2000
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__.return_value = (torch.randn(3, 256, 256) for _ in range(dataloader_len))
        mock_dataloader.__len__.return_value = dataloader_len

        mock_mmcv.runner.EpochBasedTrainLoop.dataloader = mock_dataloader
        mock_mmcv.runner.EpochBasedTrainLoop._iter = 0
        
        def mock_run_iter(self, idx, data_batch):
            self._iter += 1
        
        mock_mmcv.runner.EpochBasedTrainLoop.run_iter = types.MethodType(mock_run_iter, 
                                                                        mock_mmcv.runner.EpochBasedTrainLoop)
        
        # execute EpochBasedTrainLoop.run_epoch
        with unittest.TestCase.assertRaises(self, SystemExit) as se:
            mock_mmcv.runner.EpochBasedTrainLoop.run_epoch(mock_mmcv.runner.EpochBasedTrainLoop)
        self.assertEqual(se.exception.code, 0)
        self.assertEqual(mock_mmcv.runner.EpochBasedTrainLoop._iter, 1000)

    def test_iter_train_loop_brake(self):
        mock_engine = MagicMock()
        mock_engine.__name__ = 'mmengine'
        sys.modules['mmengine.runner'] = ModuleType('mmengine.runner')
        
        # apply monkeypatch, brake at 1000th step
        mmengine_patch.iter_train_loop(mock_engine, {'enable_profiler': False, 
                                                     'enable_brake': True,
                                                     'brake_step': 1000})

        mock_engine.runner.IterBasedTrainLoop._iter = 0
        mock_engine.runner.IterBasedTrainLoop._max_iters = 2000
        mock_engine.runner.IterBasedTrainLoop.stop_training = False
        mock_engine.runner.IterBasedTrainLoop.val_loop = MagicMock()
        mock_engine.runner.IterBasedTrainLoop.val_begin = 100
        mock_engine.runner.IterBasedTrainLoop.val_interval = 500
        
        def mock_run_iter(self, data_batch):
            self._iter += 1
        
        mock_engine.runner.IterBasedTrainLoop.run_iter = types.MethodType(mock_run_iter, 
                                                                        mock_engine.runner.IterBasedTrainLoop)
        
        
        # execute IterBasedTrainLoop.run (corresponds to "run" function in patcher's brake.py)
        with unittest.TestCase.assertRaises(self, SystemExit) as se:
            mock_engine.runner.IterBasedTrainLoop.run(mock_engine.runner.IterBasedTrainLoop)
        self.assertEqual(se.exception.code, 0)
        self.assertEqual(mock_engine.runner.IterBasedTrainLoop._iter, 1000)

    def test_iter_runner_brake(self):
        mock_engine = MagicMock()
        mock_engine.__name__ = 'mmengine'
        sys.modules['mmengine.runner'] = ModuleType('mmengine.runner')
        
        # apply monkeypatch, brake at 1000th step
        mmengine_patch.iter_runner(mock_engine, {'enable_profiler': False, 
                                                 'enable_brake': True,
                                                 'brake_step': 1000})

        workflow = [('train', 1000), ('val', 10), ('train', 1000)]
        dataloader_len = 1000
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__.return_value = (torch.randn(3, 256, 256) for _ in range(dataloader_len))
        mock_dataloader.__len__.return_value = dataloader_len
        
        mock_engine.runner.IterBasedRunner.iter = 0
        mock_engine.runner.IterBasedRunner._iter = 0
        mock_engine.runner.IterBasedRunner._max_iters = 2010
        
        def mock_train(self, data_loader, **kwargs):
            self.iter += 1
            self._iter += 1
        
        mock_engine.runner.IterBasedRunner.train = types.MethodType(mock_train, mock_engine.runner.IterBasedRunner)
        
        def mock_val(self, data_loader, **kwargs):
            self.iter += 1
            self._iter += 1
        
        mock_engine.runner.IterBasedRunner.val = types.MethodType(mock_val, mock_engine.runner.IterBasedRunner)
        
        # execute IterBasedRunner.run (corresponds to "run_iter" function in patcher's brake.py)
        with unittest.TestCase.assertRaises(self, SystemExit) as se:
            mock_engine.runner.IterBasedRunner.run(mock_engine.runner.IterBasedRunner,
                                                [mock_dataloader] * len(workflow),
                                                workflow)
        self.assertEqual(se.exception.code, 0)
        self.assertEqual(mock_engine.runner.IterBasedRunner._iter, 1000)


if __name__ == '__main__':
    run_tests()