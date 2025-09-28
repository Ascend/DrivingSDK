# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import importlib
from types import ModuleType
from typing import List, Optional, Union, Dict, Tuple
import warnings
import re

# Since MMCV 2.0 era, modules such as Runner, Hook, Parallel, Config, FileIO, have been moved to MMEngine
# Many patches were moved from mmcv_patch.py to comply with MMCV 2.x style
# Patches commented with For mmcv 1.x still takes mmcv as input instead of mmengine


# For mmcv 1.x
def ddp(mmcv: ModuleType, options: Dict): 
    importlib.import_module(f"{mmcv.__name__}.parallel")
    
    # For mmcv 1.x: module path is mmcv.parallel.distributed
    
    def _run_ddp_forward(self, *inputs, **kwargs):
        module_to_run = self.module

        if self.device_ids:
            inputs, kwargs = self.to_kwargs(  # type: ignore
                inputs, kwargs, self.device_ids[0])
            return module_to_run(*inputs[0], **kwargs[0])  # type: ignore
        else:
            return module_to_run(*inputs, **kwargs)
    
    
    if hasattr(mmcv.parallel.distributed.MMDistributedDataParallel, "_run_ddp_forward"):
        mmcv.parallel.distributed.MMDistributedDataParallel._run_ddp_forward = _run_ddp_forward
        mmcv.parallel.distributed.MMDistributedDataParallel = mmcv.device.npu.NPUDistributedDataParallel
    else:
        raise AttributeError("mmcv.parallel.distributed.MMDistributedDataParallel._run_ddp_forward not found")


# For mmcv 1.x
def stream(mmcv: ModuleType, options: Dict):
    importlib.import_module(f"{mmcv.__name__}.parallel")
    
    get_input_device = mmcv.parallel._functions.get_input_device
    scatter = mmcv.parallel._functions.scatter
    synchronize_stream = mmcv.parallel._functions.synchronize_stream
    _get_stream = mmcv.parallel._functions._get_stream
    Tensor = mmcv.parallel._functions.Tensor
    

    @staticmethod
    def new_forward(target_gpus: List[int], input_: Union[List, Tensor]) -> tuple:
        input_device = get_input_device(input_)
        streams = None
        if input_device == -1 and target_gpus != [-1]:
            # Perform CPU to GPU copies in a background stream
            streams = [
                _get_stream(mmcv.parallel._functions.torch.device("cuda", device))
                for device in target_gpus
            ]

        outputs = scatter(input_, target_gpus, streams)
        # Synchronize with the copy stream
        if streams is not None:
            synchronize_stream(outputs, target_gpus, streams)

        return tuple(outputs) if isinstance(outputs, list) else (outputs, )

    if hasattr(mmcv.parallel._functions.Scatter, "forward"):
        mmcv.parallel._functions.Scatter.forward = new_forward
    else:
        raise AttributeError("mmcv.parallel._functions.Scatter.forward not found")
    
    
# For mmcv 1.x
def optimizer_hooks(mmcv: ModuleType, options: Dict):
    """
    Patch mmcv hooks to support gradient accumulation and fp16 training.
    mmcv 1.x required.
    patch module: "mmcv.runner.hooks"
    """

    logging = mmcv.runner.hooks.optimizer.logging
    HOOKS = mmcv.runner.hooks.optimizer.HOOKS
    Hook = mmcv.runner.hooks.optimizer.Hook
    _BatchNorm = mmcv.runner.hooks.optimizer._BatchNorm
    GradScaler = mmcv.runner.hooks.optimizer.GradScaler
    wrap_fp16_model = mmcv.runner.hooks.optimizer.wrap_fp16_model
    Tensor = mmcv.runner.hooks.optimizer.Tensor

    @HOOKS.register_module(force=True)
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
                    logger.log(
                        level=logging.ERROR,
                        msg=f"{n} with shape {p.size()} is not " f"in the computational graph \n",
                    )

    @HOOKS.register_module(force=True)
    class GradientCumulativeOptimizerHook(OptimizerHook):
        def __init__(self, cumulative_iters: int = 1, **kwargs):
            super().__init__(**kwargs)

            if not isinstance(cumulative_iters, int) or cumulative_iters <= 0:
                raise ValueError(
                    f"cumulative_iters only accepts positive int, but got " f"{type(cumulative_iters)} instead."
                )

            self.cumulative_iters = cumulative_iters
            self.divisible_iters = 0
            self.remainder_iters = 0
            self.initialized = False

        def has_batch_norm(self, module: mmcv.runner.hooks.optimizer.nn.Module) -> bool:
            if isinstance(module, _BatchNorm):
                return True
            for m in module.children():
                if self.has_batch_norm(m):
                    return True
            return False

        def _init(self, runner):
            if runner.iter % self.cumulative_iters != 0:
                runner.logger.warning(
                    "Resume iter number is not divisible by cumulative_iters in "
                    "GradientCumulativeOptimizerHook, which means the gradient of "
                    "some iters is lost and the result may be influenced slightly."
                )

            if self.has_batch_norm(runner.model) and self.cumulative_iters > 1:
                runner.logger.warning(
                    "GradientCumulativeOptimizerHook may slightly decrease "
                    "performance if the model has BatchNorm layers."
                )

            self.divisible_iters = runner.max_iters // self.cumulative_iters * self.cumulative_iters
            self.remainder_iters = runner.max_iters - self.divisible_iters

            self.initialized = True

        def _get_loss_factor(self, runner):
            """Get loss division factor for the current iteration."""
            if runner.iter < runner.max_iters - self.remainder_iters:
                loss_factor = self.cumulative_iters
            else:
                loss_factor = self.remainder_iters
                runner.logger.warning(
                    f"Loss will be divided by {loss_factor} in the last "
                    f"{self.remainder_iters} iterations because they are not "
                    f"enough for {self.cumulative_iters} cumulative_iters."
                )
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
                        # Add grad norm to the logger
                        runner.log_buffer.update({"grad_norm": float(grad_norm)}, runner.outputs["num_samples"])
                runner.optimizer.step()
                runner.optimizer.zero_grad()

    @HOOKS.register_module(force=True)
    class Fp16OptimizerHook(OptimizerHook):
        # pylint: disable=huawei-super-init-not-called
        def __init__(
            self,
            grad_clip: Optional[dict] = None,
            coalesce: bool = True,
            bucket_size_mb: int = -1,
            loss_scale: Union[float, str, dict] = 512.0,
            distributed: bool = True,
        ):
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
                raise ValueError("loss_scale must be of type float, dict, or " f'"dynamic", got {loss_scale}')

        def before_run(self, runner) -> None:
            """Preparing steps before Mixed Precision Training."""
            # wrap model mode to fp16
            wrap_fp16_model(runner.model)
            # resume from state dict
            if "fp16" in runner.meta and "loss_scaler" in runner.meta["fp16"]:
                scaler_state_dict = runner.meta["fp16"]["loss_scaler"]
                self.loss_scaler.load_state_dict(scaler_state_dict)

        def copy_grads_to_fp32(self, fp16_net: mmcv.runner.hooks.optimizer.nn.Module, fp32_weights: Tensor) -> None:
            """Copy gradients from fp16 model to fp32 weight copy."""
            for fp32_param, fp16_param in zip(fp32_weights, fp16_net.parameters()):
                if fp16_param.grad is not None:
                    if fp32_param.grad is None:
                        fp32_param.grad = fp32_param.data.new(fp32_param.size())
                    fp32_param.grad.copy_(fp16_param.grad)

        def copy_params_to_fp16(self, fp16_net: mmcv.runner.hooks.optimizer.nn.Module, fp32_weights: Tensor) -> None:
            """Copy updated params from fp32 weight copy to fp16 model."""
            for fp16_param, fp32_param in zip(fp16_net.parameters(), fp32_weights):
                fp16_param.data.copy_(fp32_param.data)

        def after_train_iter(self, runner) -> None:
            runner.model.zero_grad()
            runner.optimizer.zero_grad()

            self.loss_scaler.scale(runner.outputs["loss"]).backward()
            self.loss_scaler.unscale_(runner.optimizer)
            # grad clip
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters(), runner)
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({"grad_norm": float(grad_norm)}, runner.outputs["num_samples"])
            # backward and update scaler
            self.loss_scaler.step(runner.optimizer)
            self.loss_scaler.update(self._scale_update_param)

            # save state_dict of loss_scaler
            runner.meta.setdefault("fp16", {})["loss_scaler"] = self.loss_scaler.state_dict()

    @HOOKS.register_module(force=True)
    class GradientCumulativeFp16OptimizerHook(GradientCumulativeOptimizerHook, Fp16OptimizerHook):
        """Fp16 optimizer Hook (using PyTorch's implementation) implements
        multi-iters gradient cumulating.

        If you are using PyTorch >= 1.6, torch.cuda.amp is used as the backend,
        to take care of the optimization procedure.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def after_train_iter(self, runner) -> None:
            if not self.initialized:
                self._init(runner)

            loss = runner.outputs["loss"] / self._get_loss_factor(runner)
            self.loss_scaler.scale(loss).backward()

            if self.every_n_iters(runner, self.cumulative_iters) or self.is_last_iter(runner):

                # copy fp16 grads in the model to fp32 params in the optimizer
                self.loss_scaler.unscale_(runner.optimizer)

                if self.grad_clip is not None:
                    grad_norm = self.clip_grads(runner.model.parameters(), runner)
                    if grad_norm is not None:
                        # Add grad norm to the logger
                        runner.log_buffer.update({"grad_norm": float(grad_norm)}, runner.outputs["num_samples"])

                # backward and update scaler
                self.loss_scaler.step(runner.optimizer)
                self.loss_scaler.update(self._scale_update_param)

                # save state_dict of loss_scaler
                runner.meta.setdefault("fp16", {})["loss_scaler"] = self.loss_scaler.state_dict()

                # clear grads
                runner.model.zero_grad()
                runner.optimizer.zero_grad()


# For mmcv 2.x
def optimizer_wrapper(mmengine: ModuleType, options: Dict):
    importlib.import_module(f"{mmengine.__name__}.optim")
    
    """
    Patch mmengine optimizer wrapper to support gradient clipping.
    mmcv 2.x required.
    patch module: "mmengine.optim.optimizer.optimizer_wrapper"
    """
    if hasattr(mmengine.optim.optimizer.optimizer_wrapper.OptimWrapper, "__init__"):
        OptimWrapper = mmengine.optim.optimizer.optimizer_wrapper.OptimWrapper
        orig_init = OptimWrapper.__init__

        def _get_clip_func(optimizer):
            def clip_func(params, **kwargs):
                return optimizer.clip_grad_norm_fused_(**kwargs)

            return clip_func

        def new_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            self.clip_grads = _get_clip_func(self.optimizer)

        OptimWrapper.__init__ = new_init
    else:
        raise AttributeError("mmengine.optim.optimizer.optimizer_wrapper.OptimWrapper not found")


def _parse_profiler_options(options: Dict):
    import torch_npu
    
    path = options["profiling_path"]
    level = options["profiling_level"]
    
    if bool(re.search(r'[ +#%&{}\<>*?/$!\'":@`|;=]', path)):
        warnings.warn("profiling path contains illegal character, profiler might not generate output properly")
    
    if level < 0 or level > 2:
        raise ValueError("valid profiling levels are integers within range [0, 2]")
    
    if 'step_ctrl' in options:
        fine_grained_step_control = options['step_ctrl']
    else:
        # default step control configs
        wait = 1
        warmup = 1
        active = 1
        repeat = 1
        skip_first = 20
        fine_grained_step_control = (wait, warmup, active, repeat, skip_first)

    activities = (
        [torch_npu.profiler.ProfilerActivity.NPU]
        if level == 0
        else [
            torch_npu.profiler.ProfilerActivity.NPU,
            torch_npu.profiler.ProfilerActivity.CPU,
        ]
    )
    profiler_level = torch_npu.profiler.ProfilerLevel.Level0 if level == 0 else torch_npu.profiler.ProfilerLevel.Level1
    return path, level, activities, profiler_level, fine_grained_step_control


# For MMCV 1.x
def epoch_runner(mmcv: ModuleType, options: Dict):
    importlib.import_module(f"{mmcv.__name__}.runner")
    
    import time
    import sys
    import torch_npu
    
    enable_profiler = False
    enable_brake = False
    if options['enable_profiler']:
        enable_profiler = True
        path, level, activities, profiler_level, step_ctrl = _parse_profiler_options(options)
        wait, warmup, active, repeat, skip_first = step_ctrl
    if options['enable_brake']:
        enable_brake = True
        brake_step = options['brake_step']
        
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
                        # pylint: disable=avoid-using-exit
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
                    # pylint: disable=avoid-using-exit
                    sys.exit(0)
            
        self.call_hook("after_train_epoch")
        self._epoch += 1
        
    if hasattr(mmcv.runner.EpochBasedRunner, "train"):
        mmcv.runner.EpochBasedRunner.train = train
    else:
        raise AttributeError('mmcv.runner.EpochBasedRunner.train not found')


# For MMCV 1.x
def iter_runner(mmcv: ModuleType, options: Dict):
    importlib.import_module(f"{mmcv.__name__}.runner")
    
    import time
    import sys
    import torch_npu
    
    enable_profiler = False
    enable_brake = False
    if options['enable_profiler']:
        enable_profiler = True
        path, level, activities, profiler_level, step_ctrl = _parse_profiler_options(options)
        wait, warmup, active, repeat, skip_first = step_ctrl
    if options['enable_brake']:
        enable_brake = True
        brake_step = options['brake_step']


    IterLoader = mmcv.runner.iter_based_runner.IterLoader
    get_host_info = mmcv.runner.iter_based_runner.get_host_info
    DataLoader = mmcv.runner.iter_based_runner.DataLoader


    def run(self,
                 data_loaders: List[DataLoader],
                 workflow: List[Tuple[str, int]],
                 max_iters: Optional[int] = None,
                 **kwargs) -> None:
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
        """
        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
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
                            raise ValueError(
                                'runner has no method named "{}" to run a workflow'.
                                format(mode))
                        iter_runner = getattr(self, mode)
                        for _ in range(iters):
                            if mode == 'train' and self.iter >= self._max_iters:
                                break
                            iter_runner(iter_loaders[i], **kwargs)
                            prof.step()
                            if enable_brake and self._iter == brake_step:
                                # pylint: disable=avoid-using-exit
                                sys.exit(0)
        else:
            while self.iter < self._max_iters:
                for i, flow in enumerate(workflow):
                    self._inner_iter = 0
                    mode, iters = flow
                    if not isinstance(mode, str) or not hasattr(self, mode):
                        raise ValueError(
                            'runner has no method named "{}" to run a workflow'.
                            format(mode))
                    iter_runner = getattr(self, mode)
                    for _ in range(iters):
                        if mode == 'train' and self.iter >= self._max_iters:
                            break
                        iter_runner(iter_loaders[i], **kwargs)
                        if enable_brake and self._iter == brake_step:
                            # pylint: disable=avoid-using-exit
                            sys.exit(0)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')
    
    if hasattr(mmcv.runner.IterBasedRunner, "run"):
        mmcv.runner.IterBasedRunner.run = run
    else:
        raise AttributeError('mmcv.runner.IterBasedRunner.run not found')


# For MMCV 2.x
def epoch_train_loop(mmengine: ModuleType, options: Dict):
    importlib.import_module(f"{mmengine.__name__}.runner")
    
    import time
    import sys
    import torch_npu
    
    enable_profiler = False
    enable_brake = False
    if options['enable_profiler']:
        enable_profiler = True
        path, level, activities, profiler_level, step_ctrl = _parse_profiler_options(options)
        wait, warmup, active, repeat, skip_first = step_ctrl
    if options['enable_brake']:
        enable_brake = True
        brake_step = options['brake_step']

    def run_epoch(self) -> None:
        self.runner.call_hook("before_train_epoch")
        self.runner.model.train()
        
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
                for idx, data_batch in enumerate(self.dataloader):
                    self.run_iter(idx, data_batch)
                    prof.step()
                    if enable_brake and self._iter == brake_step:
                        # pylint: disable=avoid-using-exit
                        sys.exit(0)
        else:
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch)
                if enable_brake and self._iter == brake_step:
                    # pylint: disable=avoid-using-exit
                    sys.exit(0)

        self.runner.call_hook("after_train_epoch")
        self._epoch += 1
    
    if hasattr(mmengine.runner.EpochBasedTrainLoop, "run_epoch"):
        mmengine.runner.EpochBasedTrainLoop.run_epoch = run_epoch
    else:
        raise AttributeError('mmengine.runner.EpochBasedTrainLoop.run_epoch not found')


# For MMCV 2.x
def iter_train_loop(mmengine: ModuleType, options: Dict):
    importlib.import_module(f"{mmengine.__name__}.runner")
    
    import time
    import sys
    import torch_npu
    
    enable_profiler = False
    enable_brake = False
    if options['enable_profiler']:
        enable_profiler = True
        path, level, activities, profiler_level, step_ctrl = _parse_profiler_options(options)
        wait, warmup, active, repeat, skip_first = step_ctrl
    if options['enable_brake']:
        enable_brake = True
        brake_step = options['brake_step']
    

    print_log = mmengine.logging.print_log
    logging = mmengine.logging

    def run(self) -> None:
        self.runner.call_hook("before_train")
        self.runner.call_hook("before_train_epoch")
        if self._iter > 0:
            print_log(
                f"Advance dataloader {self._iter} steps to skip data " "that has already been trained",
                logger="current",
                level=logging.WARNING,
            )
            for _ in range(self._iter):
                next(self.dataloader_iterator)
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
                while self._iter < self._max_iters and not self.stop_training:
                    self.runner.model.train()

                    data_batch = next(self.dataloader_iterator)
                    self.run_iter(data_batch)
                    prof.step()
                    if enable_brake and self._iter == brake_step:
                        # pylint: disable=avoid-using-exit
                        sys.exit(0)

                    self._decide_current_val_interval()
                    # pylint: disable=too-many-boolean-expressions
                    if (
                        self.runner.val_loop is not None
                        and self._iter >= self.val_begin
                        and (self._iter % self.val_interval == 0 or self._iter == self._max_iters)
                    ):
                        self.runner.val_loop.run()
        else:
            while self._iter < self._max_iters and not self.stop_training:
                self.runner.model.train()

                data_batch = next(self.dataloader_iterator)
                self.run_iter(data_batch)
                if enable_brake and self._iter == brake_step:
                    # pylint: disable=avoid-using-exit
                    sys.exit(0)

                self._decide_current_val_interval()
                # pylint: disable=too-many-boolean-expressions
                if (
                    self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and (self._iter % self.val_interval == 0 or self._iter == self._max_iters)
                ):
                    self.runner.val_loop.run()

        self.runner.call_hook("after_train_epoch")
        self.runner.call_hook("after_train")
        return self.runner.model
    
    if hasattr(mmengine.runner.IterBasedTrainLoop, "run"):
        mmengine.runner.IterBasedTrainLoop.run = run
    else:
        raise AttributeError('mmengine.runner.IterBasedTrainLoop.run not found')
    
