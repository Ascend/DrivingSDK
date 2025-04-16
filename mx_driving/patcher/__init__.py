# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
This module is used to patch the code to support mx_driving.
Usage:
    - with default config, mx_driving will be applied to all the code that using mmcv and torch.
    ```python
    from mx_driving.patcher import default_patcher_builder
    with default_patcher_builder.build() as patcher:
        # train model here
    ```
    - if you want to use mx_driving with other frameworks, you can use the following code to customize the patcher.

    ```python
    from mx_driving.patcher import PatcherBuilder, Patcher, Patch
    from mx_driving.patcher.mmcv import msda
    from mx_driving.patcher.tensor import index
    from mx_driving.patcher.mmcv import patch_mmcv_version
    if __name__ == "__main__":
        patcher_builder = PatcherBuilder()
        patcher_builder.add_module_patch("mmcv.ops", Patch(msda))
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
    "ddp_forward",
    "stream",
    "resnet_add_relu",
    "resnet_maxpool",
    "nuscenes_dataset",
    "nuscenes_metric",
    "nuscenes_mot_metric",
    "optimizer",
]

from mx_driving.patcher.distribute import ddp, ddp_forward
from mx_driving.patcher.functions import stream
from mx_driving.patcher.mmcv import dc, mdc, msda, patch_mmcv_version
from mx_driving.patcher.mmdet import pseudo_sampler, resnet_add_relu, resnet_maxpool
from mx_driving.patcher.mmdet3d import nuscenes_dataset, nuscenes_metric
from mx_driving.patcher.numpy import numpy_type
from mx_driving.patcher.optimizer import optimizer_hooks, optimizer_wrapper
from mx_driving.patcher.patcher import Patch, Patcher, PatcherBuilder
from mx_driving.patcher.tensor import index, batch_matmul
from mx_driving.patcher.nuscenes import nuscenes_mot_metric


default_patcher_builder = (
    PatcherBuilder()
    .add_module_patch("mmcv.ops", Patch(msda), Patch(dc), Patch(mdc))
    .add_module_patch("torch", Patch(index), Patch(batch_matmul))
    .add_module_patch("numpy", Patch(numpy_type))
    .add_module_patch("mmdet.core.bbox.samplers", Patch(pseudo_sampler))
    .add_module_patch("mmcv.parallel", Patch(ddp), Patch(stream), Patch(ddp_forward))
    .add_module_patch("mmdet.models.backbones.resnet", Patch(resnet_add_relu), Patch(resnet_maxpool))
    .add_module_patch("mmdet3d.datasets.nuscenes_dataset", Patch(nuscenes_dataset))
    .add_module_patch("mmdet3d.evaluation.metrics", Patch(nuscenes_metric))
)
