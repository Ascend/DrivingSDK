# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from types import ModuleType
from typing import Dict, Optional, Union


def optimizer_hooks(mmcvhooks: ModuleType, options: Dict):
    """
    Patch mmcv hooks to support gradient accumulation and fp16 training.
    mmcv 1.x required.
    patch module: "mmcv.runner.hooks"
    """
    if hasattr(mmcvhooks, "optimizer"):
        logging = mmcvhooks.optimizer.logging
        HOOKS = mmcvhooks.optimizer.HOOKS
        Hook = mmcvhooks.optimizer.Hook
        _BatchNorm = mmcvhooks.optimizer._BatchNorm
        GradScaler = mmcvhooks.optimizer.GradScaler
        wrap_fp16_model = mmcvhooks.optimizer.wrap_fp16_model
        Tensor = mmcvhooks.optimizer.Tensor

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

            def has_batch_norm(self, module: mmcvhooks.optimizer.nn.Module) -> bool:
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

            def copy_grads_to_fp32(self, fp16_net: mmcvhooks.optimizer.nn.Module, fp32_weights: Tensor) -> None:
                """Copy gradients from fp16 model to fp32 weight copy."""
                for fp32_param, fp16_param in zip(fp32_weights, fp16_net.parameters()):
                    if fp16_param.grad is not None:
                        if fp32_param.grad is None:
                            fp32_param.grad = fp32_param.data.new(fp32_param.size())
                        fp32_param.grad.copy_(fp16_param.grad)

            def copy_params_to_fp16(self, fp16_net: mmcvhooks.optimizer.nn.Module, fp32_weights: Tensor) -> None:
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


def optimizer_wrapper(mmcvoptwrapper: ModuleType, options: Dict):
    """
    Patch mmcv optimizer wrapper to support gradient clipping.
    mmcv 2.x required.
    patch module: "mmcv.optim.optimizer.optimizer_wrapper"
    """
    if hasattr(mmcvoptwrapper, "OptimWrapper"):
        OptimWrapper = mmcvoptwrapper.OptimWrapper
        orig_init = OptimWrapper.__init__

        def _get_clip_func(optimizer):
            def clip_func(params, **kwargs):
                return optimizer.clip_grad_norm_fused_(**kwargs)

            return clip_func

        def new_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            self.clip_grads = _get_clip_func(self.optimizer)

        OptimWrapper.__init__ = new_init
