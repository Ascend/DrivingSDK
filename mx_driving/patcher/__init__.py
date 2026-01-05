# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
The Patcher module provides non-invasive python monkey patch (a.k.a patcher) for quick migration and adaptation 
from GPU implementation to Ascend NPU implementation with appropriate optimization. Each monkey patch will replace 
the specific python function/class originally implemented for GPU/CUDA ecosystem by a corresponding NPU substitution 
implemented within this module, which takes effect throughout all occurance of that function/class in a model.
Currently the majority of public patches implemented within this Patcher module are designed for PyTorch or MMCV suites
Usage:
    - A universal default patcher is predefined for easy and quick migration to NPU with just a few lines of code:
    ```python
    from mx_driving.patcher import default_patcher_builder
    with default_patcher_builder.build() as patcher:
        # train model here
    ```
    
    - You may also customize your own patcher to select the specific patches applicable for your own models and frameworks, 
      check the following as an example:
    ```python
    from mx_driving.patcher import PatcherBuilder, Patcher, Patch
    from mx_driving.patcher.mmcv import msda
    from mx_driving.patcher.torch import index
    from mx_driving.patcher.mmcv import patch_mmcv_version
    if __name__ == "__main__":
        patcher_builder = PatcherBuilder()
        patcher_builder.add_module_patch("mmcv", Patch(msda))
        patcher_builder.add_module_patch("torch", Patch(index))

        with patcher_builder.build() as patcher:
            # train model here
    ```

"""

__all__ = [
    "default_patcher_builder",
    "msda",
    "dc",
    "mdc",
    "index",
    "batch_matmul",
    "PatcherBuilder",
    "Patcher",
    "Patch",
    "patch_mmcv_version",
    "pseudo_sampler",
    "numpy_type",
    "ddp",
    "stream",
    "resnet_add_relu",
    "resnet_maxpool",
    "resnet_fp16",
    "nuscenes_dataset",
    "nuscenes_metric",
    "optimizer_wrapper",
    "optimizer_hooks"
]

# Some patches in mmengine_patch are applied on mmcv module but organized in mmengine_patch
from mx_driving.patcher.mmengine_patch import stream, ddp, optimizer_hooks, optimizer_wrapper 

from mx_driving.patcher.mmcv_patch import dc, mdc, msda, patch_mmcv_version, spconv3d
from mx_driving.patcher.mmdet_patch import pseudo_sampler, resnet_add_relu, resnet_maxpool, resnet_fp16
from mx_driving.patcher.mmdet3d_patch import nuscenes_dataset, nuscenes_metric
from mx_driving.patcher.numpy_patch import numpy_type
from mx_driving.patcher.torch_patch import index, batch_matmul

from mx_driving.patcher.patcher import Patch, Patcher, PatcherBuilder


default_patcher_builder = (
    PatcherBuilder()
    .add_module_patch("mmcv", Patch(msda), Patch(dc), Patch(mdc), Patch(stream), Patch(ddp), Patch(spconv3d))
    .add_module_patch("torch", Patch(index), Patch(batch_matmul))
    .add_module_patch("numpy", Patch(numpy_type))
    .add_module_patch("mmdet", Patch(resnet_add_relu), Patch(resnet_maxpool))
    .add_module_patch("mmdet3d", Patch(nuscenes_dataset), Patch(nuscenes_metric))
)
