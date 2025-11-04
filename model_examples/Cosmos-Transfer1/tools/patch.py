
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
import importlib
import os
import math
from types import ModuleType
from typing import Dict
from typing import Tuple
from typing import Optional
import torch
import torch_npu
import mx_driving
import importlib

import megatron.core.utils
import megatron.core.tensor_parallel.layers

from mx_driving.patcher import PatcherBuilder, Patch

def get_prepare_input_tensors_for_wgrad_compute():
    def prepare_input_tensors_for_wgrad_compute(grad_output, all_gathered_input):
        """Ensure grad_output is stored in a contiguous buffer."""
        # Doing gather + slicing during the NeMo forward pass can make this tensor
        # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
        # clones it if it's not contiguous:
        # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
        grad_output = grad_output.contiguous()
        all_gathered_input = all_gathered_input.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        if grad_output.dim() == 3:
            grad_output = grad_output.view(
                grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
            )
            all_gathered_input = all_gathered_input.view(
                all_gathered_input.shape[0] * all_gathered_input.shape[1], all_gathered_input.shape[2]
            )
        if grad_output.dim() == 5:
            grad_output = grad_output.view(
                grad_output.shape[0] * grad_output.shape[1] * grad_output.shape[2] * grad_output.shape[3], grad_output.shape[4]
            )
            all_gathered_input = all_gathered_input.view(
                all_gathered_input.shape[0] * all_gathered_input.shape[1] * all_gathered_input.shape[2] * all_gathered_input.shape[3], all_gathered_input.shape[4]
            )

        return grad_output, all_gathered_input
    
    return prepare_input_tensors_for_wgrad_compute


def fixed_set_cuda_rng_state(random_module: ModuleType, options: Dict):
    def _set_cuda_rng_state(new_state: torch.Tensor, device: int = -1, graph_safe: bool = False):
        """Sets the random number generator state of the current GPU.

        Arguments:
            new_state (torch.ByteTensor): The desired state
            device (int): The gpu to retrieve the rng state
            graph_safe (bool): Set the rng state in a graph safe manner.

        This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
        with a single change: the input state is not cloned. Cloning caused
        major performance issues for +4 GPU cases.
        """
        if hasattr(random_module, '_cuda_setRNGState') and callable(random_module._cuda_setRNGState):
            # older PyTorch
            def cb():
                with device_ctx_manager(device):
                    random_module._cuda_setRNGState(new_state)

        else:
            # newer PyTorch
            if device == -1:
                device = torch.device('cuda')
            elif isinstance(device, str):
                device = torch.device(device)
            elif isinstance(device, int):
                device = torch.device('cuda', device)

            def cb():
                idx = device.index
                if idx is None:
                    idx = torch.cuda.current_device()
                default_generator = torch.npu.default_generators[idx]

                # if graph capturing, set the rng state in a cudagraphable way
                if graph_safe:
                    default_generator.graphsafe_set_state(new_state)
                else:
                    default_generator.set_state(new_state)
        _lazy_call = random_module._lazy_call
        _lazy_call(cb)
    if hasattr(random_module, "_set_cuda_rng_state"):
        random_module._set_cuda_rng_state = _set_cuda_rng_state



# get the patch for OpenDWM
def generate_patcher_builder():
    cosmos_transfer1_patcher_builder = (
        PatcherBuilder()
        
        # .add_module_patch("megatron.core.utils", Patch(fixed_prepare_input_tensors_for_wgrad_compute))
        .add_module_patch("megatron.core.tensor_parallel.random", Patch(fixed_set_cuda_rng_state))
    )
    return cosmos_transfer1_patcher_builder

def _init():
    megatron.core.utils.prepare_input_tensors_for_wgrad_compute = get_prepare_input_tensors_for_wgrad_compute()
    megatron.core.tensor_parallel.layers.prepare_input_tensors_for_wgrad_compute = get_prepare_input_tensors_for_wgrad_compute()

_init()
