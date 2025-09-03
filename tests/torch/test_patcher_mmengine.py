import random
import types
from types import ModuleType

import unittest
from unittest.mock import ANY, patch, MagicMock, PropertyMock, Mock

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from mx_driving.patcher import optimizer_hooks, optimizer_wrapper

# MMCV 1.x modules = MMCV 2.x moudles + MMEngine modules, therefore, some module tested here correponds to mmcv instead of mmengine


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

if __name__ == '__main__':
    run_tests()