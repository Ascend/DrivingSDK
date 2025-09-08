# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from __future__ import annotations
import importlib
import warnings
from typing import Callable, Dict, List, Optional, Set

from mx_driving.patcher.mmengine_patch import epoch_runner, iter_runner
from mx_driving.patcher.mmengine_patch import epoch_train_loop, iter_train_loop


class Patch:
    def __init__(self, func: Callable, options: Optional[Dict] = None, 
                 priority: int = 0, patch_failure_warning: bool = True):
        self.func = func
        self.name = func.__name__
        self.options = {} if options is None else options
        self.priority = priority
        self.is_applied = False
        self.patch_failure_warning = patch_failure_warning

    def __lt__(self, other):
        return self.priority < other.priority


class Patcher:
    def __init__(self, module_patches: Dict[str, List[Patch]], blacklist: Set[str], \
            allow_internal_format: bool = False):
        self.modules = []
        self.module_patches = module_patches
        self.blacklist = blacklist
        self.allow_internal_format = allow_internal_format
        for module_name in module_patches:
            try:
                module = importlib.import_module(module_name)
                self.modules.append(module)
            except ModuleNotFoundError:
                warnings.warn(f"Module {module_name} not found")
                continue

    def apply(self):
        for module in self.modules:
            for patch in self.module_patches[module.__name__]:
                if patch.name in self.blacklist or patch.is_applied:
                    continue
                try:
                    patch.func(module, patch.options)
                    patch.is_applied = True
                    print(f"Applied patch {patch.name} to module {module.__name__}")
                except Exception as e:
                    if patch.patch_failure_warning:
                        warnings.warn(f"Failed to apply patch {patch.name} to module {module.__name__}: {e}")

    # pylint: disable=add-staticmethod-or-classmethod-decorator
    def transfer_to_npu(self):
        import torch
        import torch_npu
        from torch_npu.contrib import transfer_to_npu

        if self.allow_internal_format:
            torch.npu.config.allow_internal_format = True
        else:
            torch.npu.config.allow_internal_format = False

    def __enter__(self):
        self.transfer_to_npu()
        self.apply()

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class PatcherBuilder:
    def __init__(self):
        self.module_patches = {}
        self.blacklist: Set[str] = set()
        self.training_runner_loop_options = {}
        self.training_runner_loop_options['enable_profiler'] = False
        self.training_runner_loop_options['enable_brake'] = False


    def add_module_patch(self, module_name: str, *patches: Patch) -> PatcherBuilder:
        if module_name not in self.module_patches:
            self.module_patches[module_name] = []
        self.module_patches[module_name].extend(patches)
        self.module_patches[module_name].sort()
        return self


    def disable_patches(self, *patch_names: str) -> PatcherBuilder:
        self.blacklist.update(patch_names)
        return self


    def with_profiling(self, path: str, level: int = 0) -> PatcherBuilder:
        self.training_runner_loop_options['enable_profiler'] = True
        self.training_runner_loop_options['profiling_path'] = path
        self.training_runner_loop_options['profiling_level'] = level
        return self


    def brake_at(self, brake_step: int) -> PatcherBuilder:
        self.training_runner_loop_options['enable_brake'] = True
        self.training_runner_loop_options['brake_step'] = brake_step
        return self


    def build(self, allow_internal_format: bool = False):
        if self.training_runner_loop_options['enable_profiler'] or self.training_runner_loop_options['enable_brake']:
            # Before building, append util patch for profiler and braker
            # Try MMCV 1.x
            if importlib.util.find_spec("mmcv") is not None:
                self.add_module_patch("mmcv", Patch(epoch_runner, 
                                                    self.training_runner_loop_options, 
                                                    patch_failure_warning=True))
                self.add_module_patch("mmcv", Patch(iter_runner, 
                                                    self.training_runner_loop_options,
                                                    patch_failure_warning=True))
                
            # Try MMCV 2.x
            if importlib.util.find_spec("mmengine") is not None:
                self.add_module_patch("mmengine", Patch(epoch_train_loop, 
                                                        self.training_runner_loop_options,
                                                        patch_failure_warning=True))
                self.add_module_patch("mmengine", Patch(iter_train_loop, 
                                                        self.training_runner_loop_options,
                                                        patch_failure_warning=True))
            
        return Patcher(self.module_patches, self.blacklist, allow_internal_format)

