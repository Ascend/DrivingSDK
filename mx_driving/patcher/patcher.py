# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
import importlib
import warnings
from typing import Callable, Dict, List, Optional, Set

from mx_driving.patcher.profiler import profiler
from mx_driving.patcher.brake import brake


class Patch:
    def __init__(self, func: Callable, options: Optional[Dict] = None, priority: int = 0):
        self.func = func
        self.name = func.__name__
        self.options = {} if options is None else options
        self.priority = priority
        self.is_applied = False

    def __lt__(self, other):
        return self.priority < other.priority


class Patcher:
    def __init__(self, module_patches: Dict[str, List[Patch]], blacklist: Set[str]):
        self.modules = []
        self.module_patches = module_patches
        self.blacklist = blacklist
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
                    warnings.warn(f"Failed to apply patch {patch.name} to module {module.__name__}: {e}")

    # pylint: disable=add-staticmethod-or-classmethod-decorator
    def transfer_to_npu(self):
        import torch
        import torch_npu
        from torch_npu.contrib import transfer_to_npu

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

    def add_module_patch(self, module_name: str, *patches: Patch) -> "PatcherBuilder":
        if module_name not in self.module_patches:
            self.module_patches[module_name] = []
        self.module_patches[module_name].extend(patches)
        self.module_patches[module_name].sort()
        return self

    def disable_patches(self, *patch_names: str) -> "PatcherBuilder":
        self.blacklist.update(patch_names)
        return self

    def with_profiling(self, path: str, level: int = 0) -> "PatcherBuilder":
        return self.add_module_patch(
            "mmcv.runner", Patch(profiler, {"profiling_path": path, "profiling_level": level})
        ).add_module_patch("mmengine.runner", Patch(profiler, {"profiling_path": path, "profiling_level": level}))

    def brake_at(self, when_iter: int) -> "PatcherBuilder":
        return self.add_module_patch("mmcv.runner", Patch(brake, {"when_iter": when_iter})).add_module_patch(
            "mmengine.runner", Patch(brake, {"when_iter": when_iter})
        )

    def build(self):
        return Patcher(self.module_patches, self.blacklist)
