# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""MMDetection3D patches for NPU."""
import importlib.util
from typing import List

from mx_driving.patcher.patch import AtomicPatch, BasePatch, Patch


def _nuscenes_deps_available():
    return (
        importlib.util.find_spec("numpy") is not None
        and importlib.util.find_spec("pyquaternion") is not None
        and importlib.util.find_spec("nuscenes") is not None
    )


class NuScenesDataset(Patch):
    """NuScenes dataset output conversion patch for mmdet3d.datasets."""

    name = "nuscenes_dataset"
    legacy_name = "nuscenes_dataset"
    target_module = "mmdet3d"

    @staticmethod
    def _replacement(detection, with_velocity=True):
        import numpy as np
        import pyquaternion
        from nuscenes.utils.data_classes import Box as NuScenesBox

        box3d = detection["boxes_3d"]
        scores = detection["scores_3d"].numpy()
        labels = detection["labels_3d"].numpy()
        box_gravity_center = box3d.gravity_center.numpy()
        box_dims = box3d.dims.numpy()
        box_yaw = -box3d.yaw.numpy() - np.pi / 2

        box_list = []
        for i in range(len(box3d)):
            quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
            velocity = (*box3d.tensor[i, 7:9], 0.0) if with_velocity else (0, 0, 0)
            box_list.append(NuScenesBox(
                box_gravity_center[i], box_dims[i], quat,
                label=labels[i], score=scores[i], velocity=velocity
            ))
        return box_list

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "mmdet3d.datasets.nuscenes_dataset.output_to_nusc_box",
                cls._replacement,
                precheck=_nuscenes_deps_available,
            ),
        ]


class NuScenesMetric(Patch):
    """NuScenes metric output conversion patch for mmdet3d.evaluation."""

    name = "nuscenes_metric"
    legacy_name = "nuscenes_metric"
    target_module = "mmdet3d"

    @staticmethod
    def _replacement(detection):
        import numpy as np
        import pyquaternion
        from nuscenes.utils.data_classes import Box as NuScenesBox
        from mmdet3d.structures import CameraInstance3DBoxes, LiDARInstance3DBoxes

        bbox3d = detection["bboxes_3d"]
        scores = detection["scores_3d"].numpy()
        labels = detection["labels_3d"].numpy()
        attrs = detection["attr_labels"].numpy() if "attr_labels" in detection else None

        box_gravity_center = bbox3d.gravity_center.numpy()
        box_dims = bbox3d.dims.numpy()
        box_yaw = bbox3d.yaw.numpy()
        box_list = []

        if isinstance(bbox3d, LiDARInstance3DBoxes):
            box_yaw = -box_yaw - np.pi / 2
            for i in range(len(bbox3d)):
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
                velocity = (*bbox3d.tensor[i, 7:9], 0.0)
                box_list.append(NuScenesBox(
                    box_gravity_center[i], box_dims[i], quat,
                    label=labels[i], score=scores[i], velocity=velocity,
                ))
        elif isinstance(bbox3d, CameraInstance3DBoxes):
            nus_box_dims = box_dims[:, [2, 0, 1]]
            nus_box_yaw = -box_yaw
            for i in range(len(bbox3d)):
                q1 = pyquaternion.Quaternion(axis=[0, 0, 1], radians=nus_box_yaw[i])
                q2 = pyquaternion.Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
                quat = q2 * q1
                velocity = (bbox3d.tensor[i, 7], 0.0, bbox3d.tensor[i, 8])
                box_list.append(NuScenesBox(
                    box_gravity_center[i], nus_box_dims[i], quat,
                    label=labels[i], score=scores[i], velocity=velocity,
                ))
        else:
            raise NotImplementedError(f"Do not support convert {type(bbox3d)} bboxes")
        return box_list, attrs

    @classmethod
    def patches(cls, options=None) -> List[BasePatch]:
        return [
            AtomicPatch(
                "mmdet3d.evaluation.metrics.output_to_nusc_box",
                cls._replacement,
                precheck=_nuscenes_deps_available,
            ),
        ]


# Backward compatibility alias
NuScenes = NuScenesDataset
