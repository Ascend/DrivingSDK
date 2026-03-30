# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
MMCV patches for NPU adaptation.

Provides NPU-compatible replacements for mmcv operators including:
- Multi-Scale Deformable Attention (MSDA)
- Deformable Convolution
- Modulated Deformable Convolution
- Sparse Convolution 3D
- Scatter/Stream operations
- Distributed Data Parallel (DDP)
- Optimizer Hooks (mmcv 1.x)
- Training loop patches (mmcv 1.x)
"""
import importlib
import re
import warnings
from typing import Dict, List, Optional, Union

from mx_driving.patcher.patcher_logger import patcher_logger
from mx_driving.patcher.patch import (
    AtomicPatch,
    BasePatch,
    LegacyPatch,
    Patch,
    RegistryPatch,
    mmcv_version,
)


# =============================================================================
# MultiScaleDeformableAttention
# =============================================================================

class MultiScaleDeformableAttention(Patch):
    """Multi-Scale Deformable Attention patch for mmcv."""

    name = "multi_scale_deformable_attention"
    legacy_name = "msda"
    target_module = "mmcv"

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        base = "mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction"
        npu_base = "mx_driving.MultiScaleDeformableAttnFunction"

        # forward wrapper: ignore im2col_step parameter
        def forward_wrapper(npu_forward):
            def forward(ctx, value, spatial_shapes, level_start_index,
                        sampling_locations, attention_weights, im2col_step=None):
                return npu_forward(ctx, value, spatial_shapes, level_start_index,
                                   sampling_locations, attention_weights)
            return forward

        # backward wrapper: append None to return value (for im2col_step gradient)
        def backward_wrapper(npu_backward):
            def backward(ctx, grad_output):
                return (*npu_backward(ctx, grad_output), None)
            return backward

        return [
            AtomicPatch(f"{base}.forward", f"{npu_base}.forward",
                        replacement_wrapper=forward_wrapper),
            AtomicPatch(f"{base}.backward", f"{npu_base}.backward",
                        replacement_wrapper=backward_wrapper),
        ]


# =============================================================================
# DeformConv
# =============================================================================

class DeformConv(Patch):
    """Deformable Convolution patch for mmcv."""

    name = "deform_conv"
    legacy_name = "dc"
    target_module = "mmcv"

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "mmcv.ops.deform_conv.DeformConv2dFunction",
                "mx_driving.DeformConv2dFunction",
            ),
            AtomicPatch(
                "mmcv.ops.deform_conv.deform_conv2d",
                "mx_driving.deform_conv2d",
            ),
        ]


# =============================================================================
# ModulatedDeformConv
# =============================================================================

class ModulatedDeformConv(Patch):
    """Modulated Deformable Convolution patch for mmcv."""

    name = "modulated_deform_conv"
    legacy_name = "mdc"
    target_module = "mmcv"

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "mmcv.ops.modulated_deform_conv.ModulatedDeformConv2dFunction",
                "mx_driving.ModulatedDeformConv2dFunction",
            ),
            AtomicPatch(
                "mmcv.ops.modulated_deform_conv.modulated_deform_conv2d",
                "mx_driving.modulated_deform_conv2d",
            ),
        ]


# =============================================================================
# SparseConv3D
# =============================================================================

class SparseConv3D(Patch):
    """Sparse Convolution 3D patch for mmcv."""

    name = "spconv3d"
    legacy_name = "spconv3d"
    target_module = "mmcv"

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        patches = [
            # mmcv.ops
            AtomicPatch("mmcv.ops.SparseConvTensor", "mx_driving.SparseConvTensor"),
            AtomicPatch("mmcv.ops.sparse_structure.SparseConvTensor", "mx_driving.SparseConvTensor"),
            AtomicPatch("mmcv.ops.SparseSequential", "mx_driving.SparseSequential"),
            AtomicPatch("mmcv.ops.sparse_modules.SparseSequential", "mx_driving.SparseSequential"),
            AtomicPatch("mmcv.ops.SparseModule", "mx_driving.SparseModule"),
            AtomicPatch("mmcv.ops.sparse_modules.SparseModule", "mx_driving.SparseModule"),
            AtomicPatch("mmcv.ops.SparseConvolution", "mx_driving.SparseConvolution"),
            AtomicPatch("mmcv.ops.sparse_conv.SparseConvolution", "mx_driving.SparseConvolution"),
            AtomicPatch("mmcv.ops.SubMConv3d", "mx_driving.SubMConv3d"),
            AtomicPatch("mmcv.ops.sparse_conv.SubMConv3d", "mx_driving.SubMConv3d"),
            AtomicPatch("mmcv.ops.SparseConv3d", "mx_driving.SparseConv3d"),
            AtomicPatch("mmcv.ops.sparse_conv.SparseConv3d", "mx_driving.SparseConv3d"),
            AtomicPatch("mmcv.ops.SparseInverseConv3d", "mx_driving.SparseInverseConv3d"),
            AtomicPatch("mmcv.ops.sparse_conv.SparseInverseConv3d", "mx_driving.SparseInverseConv3d"),
        ]

        # mmcv 1.x cnn.CONV_LAYERS registry
        if mmcv_version.is_v1x:
            patches.extend([
                AtomicPatch(
                    "mmcv.cnn.CONV_LAYERS._module_dict.SubMConv3d",
                    "mx_driving.SubMConv3d",
                ),
                AtomicPatch(
                    "mmcv.cnn.CONV_LAYERS._module_dict.SparseConv3d",
                    "mx_driving.SparseConv3d",
                ),
                AtomicPatch(
                    "mmcv.cnn.CONV_LAYERS._module_dict.SparseInverseConv3d",
                    "mx_driving.SparseInverseConv3d",
                ),
            ])

        # mmcv 2.x mmengine.registry.MODELS
        if mmcv_version.is_v2x:
            patches.extend([
                AtomicPatch(
                    "mmengine.registry.MODELS._module_dict.SubMConv3d",
                    "mx_driving.SubMConv3d",
                ),
                AtomicPatch(
                    "mmengine.registry.MODELS._module_dict.SparseConv3d",
                    "mx_driving.SparseConv3d",
                ),
                AtomicPatch(
                    "mmengine.registry.MODELS._module_dict.SparseInverseConv3d",
                    "mx_driving.SparseInverseConv3d",
                ),
            ])

        return patches


# =============================================================================
# Stream
# =============================================================================

class Stream(Patch):
    """Scatter stream patch for mmcv 1.x."""

    name = "stream"
    legacy_name = "stream"
    target_module = "mmcv"

    @staticmethod
    def precheck() -> bool:
        return mmcv_version.is_v1x

    @staticmethod
    def scatter_forward(target_gpus, input_):
        import mmcv.parallel._functions as F

        input_device = F.get_input_device(input_)
        streams = None
        if input_device == -1 and target_gpus != [-1]:
            import torch
            streams = [F._get_stream(torch.device("cuda", d)) for d in target_gpus]

        outputs = F.scatter(input_, target_gpus, streams)
        if streams is not None:
            F.synchronize_stream(outputs, target_gpus, streams)

        return tuple(outputs) if isinstance(outputs, list) else (outputs,)

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "mmcv.parallel._functions.Scatter.forward",
                staticmethod(cls.scatter_forward),
                precheck=cls.precheck,
            ),
        ]


# =============================================================================
# DDP
# =============================================================================

class DDP(Patch):
    """Distributed Data Parallel (DDP) patch for mmcv 1.x."""

    name = "ddp"
    legacy_name = "ddp"
    target_module = "mmcv"

    @staticmethod
    def precheck() -> bool:
        return mmcv_version.is_v1x

    @staticmethod
    def ddp_forward(self, *inputs, **kwargs):
        module_to_run = self.module
        if self.device_ids:
            inputs, kwargs = self.to_kwargs(inputs, kwargs, self.device_ids[0])
            return module_to_run(*inputs[0], **kwargs[0])
        return module_to_run(*inputs, **kwargs)

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "mmcv.parallel.distributed.MMDistributedDataParallel._run_ddp_forward",
                cls.ddp_forward,
                precheck=cls.precheck,
            ),
            AtomicPatch(
                "mmcv.parallel.distributed.MMDistributedDataParallel",
                "mmcv.device.npu.NPUDistributedDataParallel",
                precheck=cls.precheck,
            ),
        ]


# =============================================================================
# Optimizer Hooks (mmcv 1.x)
# =============================================================================

class OptimizerHooks(Patch):
    """
    Optimizer hooks patch for mmcv 1.x with gradient clipping support.

    Only applies when mmcv 1.x is detected (no mmengine).
    """

    name = "optimizer_hooks"
    legacy_name = "optimizer_hooks"
    target_module = "mmcv"

    @staticmethod
    def _create_optimizer_hook():
        """Factory for OptimizerHook class."""
        import mmcv
        importlib.import_module("mmcv.runner.hooks")

        logging = mmcv.runner.hooks.optimizer.logging
        Hook = mmcv.runner.hooks.optimizer.Hook
        Tensor = mmcv.runner.hooks.optimizer.Tensor

        class OptimizerHook(Hook):
            def __init__(self, grad_clip: Optional[dict] = None, detect_anomalous_params: bool = False):
                self.grad_clip = grad_clip
                self.detect_anomalous_params = detect_anomalous_params

            def clip_grads(self, params, runner):
                params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
                if len(params) > 0:
                    return runner.optimizer.clip_grad_norm_fused_(**self.grad_clip)
                return None

            def after_train_iter(self, runner):
                runner.optimizer.zero_grad()
                if self.detect_anomalous_params:
                    self.detect_anomalous_parameters(runner.outputs["loss"], runner)
                runner.outputs["loss"].backward()

                if self.grad_clip is not None:
                    grad_norm = self.clip_grads(runner.model.parameters(), runner)
                    if grad_norm is not None:
                        runner.log_buffer.update({"grad_norm": float(grad_norm)}, runner.outputs["num_samples"])
                runner.optimizer.step()

            def detect_anomalous_parameters(self, loss: Tensor, runner) -> None:
                logger = runner.logger
                parameters_in_graph = set()
                visited = set()

                def traverse(grad_fn):
                    if grad_fn is None:
                        return
                    if grad_fn not in visited:
                        visited.add(grad_fn)
                        if hasattr(grad_fn, "variable"):
                            parameters_in_graph.add(grad_fn.variable)
                        parents = grad_fn.next_functions
                        if parents is not None:
                            for parent in parents:
                                grad_fn = parent[0]
                                traverse(grad_fn)

                traverse(loss.grad_fn)
                for n, p in runner.model.named_parameters():
                    if p not in parameters_in_graph and p.requires_grad:
                        logger.log(level=logging.ERROR, msg=f"{n} with shape {p.size()} is not in the computational graph\n")

        return OptimizerHook

    @staticmethod
    def _create_gradient_cumulative_optimizer_hook():
        """Factory for GradientCumulativeOptimizerHook class."""
        import mmcv
        importlib.import_module("mmcv.runner.hooks")

        _BatchNorm = mmcv.runner.hooks.optimizer._BatchNorm
        HOOKS = mmcv.runner.hooks.optimizer.HOOKS
        OptimizerHook = HOOKS.module_dict.get("OptimizerHook")

        if OptimizerHook is None:
            raise RuntimeError("OptimizerHook must be registered before GradientCumulativeOptimizerHook")

        class GradientCumulativeOptimizerHook(OptimizerHook):
            def __init__(self, cumulative_iters: int = 1, **kwargs):
                super().__init__(**kwargs)
                if not isinstance(cumulative_iters, int) or cumulative_iters <= 0:
                    raise ValueError(f"cumulative_iters only accepts positive int, but got {type(cumulative_iters)} instead.")
                self.cumulative_iters = cumulative_iters
                self.divisible_iters = 0
                self.remainder_iters = 0
                self.initialized = False

            def has_batch_norm(self, m) -> bool:
                if isinstance(m, _BatchNorm):
                    return True
                for child in m.children():
                    if self.has_batch_norm(child):
                        return True
                return False

            def _init(self, runner):
                if runner.iter % self.cumulative_iters != 0:
                    runner.logger.warning("Resume iter number is not divisible by cumulative_iters")
                if self.has_batch_norm(runner.model) and self.cumulative_iters > 1:
                    runner.logger.warning("GradientCumulativeOptimizerHook may slightly decrease performance with BatchNorm layers")
                self.divisible_iters = runner.max_iters // self.cumulative_iters * self.cumulative_iters
                self.remainder_iters = runner.max_iters - self.divisible_iters
                self.initialized = True

            def _get_loss_factor(self, runner):
                if runner.iter < runner.max_iters - self.remainder_iters:
                    return self.cumulative_iters
                loss_factor = self.remainder_iters
                runner.logger.warning(f"Loss will be divided by {loss_factor} in the last {self.remainder_iters} iterations")
                if loss_factor <= 0:
                    raise ValueError("loss_factor should be larger than 0.")
                return loss_factor

            def after_train_iter(self, runner):
                if not self.initialized:
                    self._init(runner)
                loss = runner.outputs["loss"] / self._get_loss_factor(runner)
                loss.backward()
                if self.every_n_iters(runner, self.cumulative_iters) or self.is_last_iter(runner):
                    if self.grad_clip is not None:
                        grad_norm = self.clip_grads(runner.model.parameters(), runner)
                        if grad_norm is not None:
                            runner.log_buffer.update({"grad_norm": float(grad_norm)}, runner.outputs["num_samples"])
                    runner.optimizer.step()
                    runner.optimizer.zero_grad()

        return GradientCumulativeOptimizerHook

    @staticmethod
    def _create_fp16_optimizer_hook():
        """Factory for Fp16OptimizerHook class."""
        import mmcv
        importlib.import_module("mmcv.runner.hooks")

        GradScaler = mmcv.runner.hooks.optimizer.GradScaler
        wrap_fp16_model = mmcv.runner.hooks.optimizer.wrap_fp16_model
        HOOKS = mmcv.runner.hooks.optimizer.HOOKS
        OptimizerHook = HOOKS.module_dict.get("OptimizerHook")

        if OptimizerHook is None:
            raise RuntimeError("OptimizerHook must be registered before Fp16OptimizerHook")

        class Fp16OptimizerHook(OptimizerHook):
            def __init__(self, grad_clip: Optional[dict] = None, coalesce: bool = True, bucket_size_mb: int = -1,
                         loss_scale: Union[float, str, dict] = 512.0, distributed: bool = True):
                self.grad_clip = grad_clip
                self.coalesce = coalesce
                self.bucket_size_mb = bucket_size_mb
                self.distributed = distributed
                self._scale_update_param = None
                if loss_scale == "dynamic":
                    self.loss_scaler = GradScaler()
                elif isinstance(loss_scale, float):
                    self._scale_update_param = loss_scale
                    self.loss_scaler = GradScaler(init_scale=loss_scale)
                elif isinstance(loss_scale, dict):
                    self.loss_scaler = GradScaler(**loss_scale)
                else:
                    raise ValueError(f'loss_scale must be of type float, dict, or "dynamic", got {loss_scale}')

            def before_run(self, runner) -> None:
                wrap_fp16_model(runner.model)
                if "fp16" in runner.meta and "loss_scaler" in runner.meta["fp16"]:
                    self.loss_scaler.load_state_dict(runner.meta["fp16"]["loss_scaler"])

            def after_train_iter(self, runner) -> None:
                runner.model.zero_grad()
                runner.optimizer.zero_grad()
                self.loss_scaler.scale(runner.outputs["loss"]).backward()
                self.loss_scaler.unscale_(runner.optimizer)
                if self.grad_clip is not None:
                    grad_norm = self.clip_grads(runner.model.parameters(), runner)
                    if grad_norm is not None:
                        runner.log_buffer.update({"grad_norm": float(grad_norm)}, runner.outputs["num_samples"])
                self.loss_scaler.step(runner.optimizer)
                self.loss_scaler.update(self._scale_update_param)
                runner.meta.setdefault("fp16", {})["loss_scaler"] = self.loss_scaler.state_dict()

        return Fp16OptimizerHook

    @staticmethod
    def _create_gradient_cumulative_fp16_optimizer_hook():
        """Factory for GradientCumulativeFp16OptimizerHook class."""
        import mmcv
        importlib.import_module("mmcv.runner.hooks")

        HOOKS = mmcv.runner.hooks.optimizer.HOOKS
        GradientCumulativeOptimizerHook = HOOKS.module_dict.get("GradientCumulativeOptimizerHook")
        Fp16OptimizerHook = HOOKS.module_dict.get("Fp16OptimizerHook")

        if GradientCumulativeOptimizerHook is None:
            raise RuntimeError("GradientCumulativeOptimizerHook must be registered before GradientCumulativeFp16OptimizerHook")
        if Fp16OptimizerHook is None:
            raise RuntimeError("Fp16OptimizerHook must be registered before GradientCumulativeFp16OptimizerHook")

        class GradientCumulativeFp16OptimizerHook(GradientCumulativeOptimizerHook, Fp16OptimizerHook):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def after_train_iter(self, runner) -> None:
                if not self.initialized:
                    self._init(runner)
                loss = runner.outputs["loss"] / self._get_loss_factor(runner)
                self.loss_scaler.scale(loss).backward()
                if self.every_n_iters(runner, self.cumulative_iters) or self.is_last_iter(runner):
                    self.loss_scaler.unscale_(runner.optimizer)
                    if self.grad_clip is not None:
                        grad_norm = self.clip_grads(runner.model.parameters(), runner)
                        if grad_norm is not None:
                            runner.log_buffer.update({"grad_norm": float(grad_norm)}, runner.outputs["num_samples"])
                    self.loss_scaler.step(runner.optimizer)
                    self.loss_scaler.update(self._scale_update_param)
                    runner.meta.setdefault("fp16", {})["loss_scaler"] = self.loss_scaler.state_dict()
                    runner.model.zero_grad()
                    runner.optimizer.zero_grad()

        return GradientCumulativeFp16OptimizerHook

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        precheck = lambda: mmcv_version.is_v1x
        return [
            # Note: Order matters! Later hooks depend on earlier ones being registered.
            RegistryPatch(
                "mmcv.runner.hooks.optimizer.HOOKS",
                name="OptimizerHook",
                module_factory=cls._create_optimizer_hook,
                precheck=precheck,
            ),
            RegistryPatch(
                "mmcv.runner.hooks.optimizer.HOOKS",
                name="GradientCumulativeOptimizerHook",
                module_factory=cls._create_gradient_cumulative_optimizer_hook,
                precheck=precheck,
            ),
            RegistryPatch(
                "mmcv.runner.hooks.optimizer.HOOKS",
                name="Fp16OptimizerHook",
                module_factory=cls._create_fp16_optimizer_hook,
                precheck=precheck,
            ),
            RegistryPatch(
                "mmcv.runner.hooks.optimizer.HOOKS",
                name="GradientCumulativeFp16OptimizerHook",
                module_factory=cls._create_gradient_cumulative_fp16_optimizer_hook,
                precheck=precheck,
            ),
        ]


# =============================================================================
# Training loop patches (mmcv 1.x, for profiling/brake)
# =============================================================================

def _parse_profiler_options(options: Dict):
    import torch_npu

    path = options["profiling_path"]
    level = options["profiling_level"]

    if bool(re.search(r'[ +#%&{}\<>*?/$!\'":@`|;=]', path)):
        patcher_logger.warning("profiling path contains illegal character")

    if level < 0 or level > 2:
        raise ValueError("valid profiling levels are integers within range [0, 2]")

    step_ctrl = options.get('step_ctrl', (1, 1, 1, 1, 20))

    activities = (
        [torch_npu.profiler.ProfilerActivity.NPU]
        if level == 0
        else [torch_npu.profiler.ProfilerActivity.NPU, torch_npu.profiler.ProfilerActivity.CPU]
    )
    profiler_level = torch_npu.profiler.ProfilerLevel.Level0 if level == 0 else torch_npu.profiler.ProfilerLevel.Level1
    return path, level, activities, profiler_level, step_ctrl


def build_mmcv_epoch_runner_patch(options: Dict) -> LegacyPatch:
    def _apply(module, _options):
        import time
        import sys
        import torch_npu

        enable_profiler = bool(options.get("enable_profiler"))
        enable_brake = bool(options.get("enable_brake"))
        if enable_profiler:
            path, level, activities, profiler_level, step_ctrl = _parse_profiler_options(options)
            wait, warmup, active, repeat, skip_first = step_ctrl
        if enable_brake:
            brake_step = options.get("brake_step")

        def train(self, data_loader, **kwargs):
            self.model.train()
            self.mode = "train"
            self.data_loader = data_loader
            self._max_iters = self._max_epochs * len(data_loader)
            self.call_hook("before_train_epoch")
            time.sleep(2)

            if enable_profiler:
                with torch_npu.profiler.profile(
                    activities=activities,
                    with_stack=level == 2,
                    record_shapes=level > 0,
                    profile_memory=level == 2,
                    schedule=torch_npu.profiler.schedule(wait, warmup, active, repeat, skip_first),
                    experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=profiler_level),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(path),
                ) as prof:
                    for i, data_batch in enumerate(data_loader):
                        self.data_batch = data_batch
                        self._inner_iter = i
                        self.call_hook("before_train_iter")
                        self.run_iter(data_batch, train_mode=True, **kwargs)
                        self.call_hook("after_train_iter")
                        del self.data_batch
                        self._iter += 1
                        prof.step()
                        if enable_brake and self._iter == brake_step:
                            sys.exit(0)
            else:
                for i, data_batch in enumerate(data_loader):
                    self.data_batch = data_batch
                    self._inner_iter = i
                    self.call_hook("before_train_iter")
                    self.run_iter(data_batch, train_mode=True, **kwargs)
                    self.call_hook("after_train_iter")
                    del self.data_batch
                    self._iter += 1
                    if enable_brake and self._iter == brake_step:
                        sys.exit(0)

            self.call_hook("after_train_epoch")
            self._epoch += 1

        importlib.import_module(f"{module.__name__}.runner")
        module.runner.EpochBasedRunner.train = train

    return LegacyPatch(_apply, target_module="mmcv")


def build_mmcv_iter_runner_patch(options: Dict) -> LegacyPatch:
    def _apply(module, _options):
        import time
        import sys
        import torch_npu

        enable_profiler = bool(options.get("enable_profiler"))
        enable_brake = bool(options.get("enable_brake"))
        if enable_profiler:
            path, level, activities, profiler_level, step_ctrl = _parse_profiler_options(options)
            wait, warmup, active, repeat, skip_first = step_ctrl
        if enable_brake:
            brake_step = options.get("brake_step")

        importlib.import_module(f"{module.__name__}.runner")
        IterLoader = module.runner.iter_based_runner.IterLoader
        get_host_info = module.runner.iter_based_runner.get_host_info

        def run(self, data_loaders, workflow, max_iters=None, **kwargs):
            if max_iters is not None:
                warnings.warn('setting max_iters in run is deprecated', DeprecationWarning)
                self._max_iters = max_iters

            work_dir = self.work_dir if self.work_dir is not None else 'NONE'
            self.logger.info('Start running, host: %s, work_dir: %s', get_host_info(), work_dir)
            self.logger.info('Hooks will be executed in the following order:\n%s', self.get_hook_info())
            self.logger.info('workflow: %s, max: %d iters', workflow, self._max_iters)
            self.call_hook('before_run')

            iter_loaders = [IterLoader(x) for x in data_loaders]
            self.call_hook('before_epoch')

            if enable_profiler:
                with torch_npu.profiler.profile(
                    activities=activities,
                    with_stack=level == 2,
                    record_shapes=level > 0,
                    profile_memory=level == 2,
                    schedule=torch_npu.profiler.schedule(wait, warmup, active, repeat, skip_first),
                    experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=profiler_level),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(path),
                ) as prof:
                    while self.iter < self._max_iters:
                        for i, flow in enumerate(workflow):
                            self._inner_iter = 0
                            mode, iters = flow
                            if not isinstance(mode, str) or not hasattr(self, mode):
                                raise ValueError(f'runner has no method named "{mode}" to run a workflow')
                            iter_runner = getattr(self, mode)
                            for _ in range(iters):
                                if mode == 'train' and self.iter >= self._max_iters:
                                    break
                                iter_runner(iter_loaders[i], **kwargs)
                                prof.step()
                                if enable_brake and self._iter == brake_step:
                                    sys.exit(0)
            else:
                while self.iter < self._max_iters:
                    for i, flow in enumerate(workflow):
                        self._inner_iter = 0
                        mode, iters = flow
                        if not isinstance(mode, str) or not hasattr(self, mode):
                            raise ValueError(f'runner has no method named "{mode}" to run a workflow')
                        iter_runner = getattr(self, mode)
                        for _ in range(iters):
                            if mode == 'train' and self.iter >= self._max_iters:
                                break
                            iter_runner(iter_loaders[i], **kwargs)
                            if enable_brake and self._iter == brake_step:
                                sys.exit(0)

            time.sleep(1)
            self.call_hook('after_epoch')
            self.call_hook('after_run')

        module.runner.IterBasedRunner.run = run

    return LegacyPatch(_apply, target_module="mmcv")
