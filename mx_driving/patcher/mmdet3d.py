# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from types import ModuleType
from typing import Dict, List, Tuple, Union


def nuscenes_dataset(mmdet3ddatasets: ModuleType, options: Dict):
    if hasattr(mmdet3ddatasets, "output_to_nusc_box"):
        import numpy as np
        import pyquaternion
        from nuscenes.utils.data_classes import Box as NuScenesBox

        def output_to_nusc_box(detection, with_velocity=True):
            box3d = detection["boxes_3d"]
            scores = detection["scores_3d"].numpy()
            labels = detection["labels_3d"].numpy()

            box_gravity_center = box3d.gravity_center.numpy()
            box_dims = box3d.dims.numpy()
            box_yaw = box3d.yaw.numpy()
            box_yaw = -box_yaw - np.pi / 2

            box_list = []
            for i in range(len(box3d)):
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
                if with_velocity:
                    velocity = (*box3d.tensor[i, 7:9], 0.0)
                else:
                    velocity = (0, 0, 0)
                box = NuScenesBox(
                    box_gravity_center[i], box_dims[i], quat, label=labels[i], score=scores[i], velocity=velocity
                )
                box_list.append(box)
            return box_list

        mmdet3ddatasets.output_to_nusc_box = output_to_nusc_box


def nuscenes_metric(mmdet3dmetrics: ModuleType, options: Dict):
    if hasattr(mmdet3dmetrics, "output_to_nusc_box"):
        import numpy as np
        import pyquaternion
        from nuscenes.utils.data_classes import Box as NuScenesBox
        from mmdet3d.structures import CameraInstance3DBoxes, LiDARInstance3DBoxes

        def output_to_nusc_box(detection: dict) -> Tuple[List[NuScenesBox], Union[np.ndarray, None]]:
            bbox3d = detection["bboxes_3d"]
            scores = detection["scores_3d"].numpy()
            labels = detection["labels_3d"].numpy()
            attrs = None
            if "attr_labels" in detection:
                attrs = detection["attr_labels"].numpy()

            box_gravity_center = bbox3d.gravity_center.numpy()
            box_dims = bbox3d.dims.numpy()
            box_yaw = bbox3d.yaw.numpy()

            box_list = []

            if isinstance(bbox3d, LiDARInstance3DBoxes):
                box_yaw = -box_yaw - np.pi / 2
                for i in range(len(bbox3d)):
                    quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
                    velocity = (*bbox3d.tensor[i, 7:9], 0.0)
                    box = NuScenesBox(
                        box_gravity_center[i],
                        box_dims[i],
                        quat,
                        label=labels[i],
                        score=scores[i],
                        velocity=velocity,
                    )
                    box_list.append(box)
            elif isinstance(bbox3d, CameraInstance3DBoxes):
                # our Camera coordinate system -> nuScenes box coordinate system
                # convert the dim/rot to nuscbox convention
                nus_box_dims = box_dims[:, [2, 0, 1]]
                nus_box_yaw = -box_yaw
                for i in range(len(bbox3d)):
                    q1 = pyquaternion.Quaternion(axis=[0, 0, 1], radians=nus_box_yaw[i])
                    q2 = pyquaternion.Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
                    quat = q2 * q1
                    velocity = (bbox3d.tensor[i, 7], 0.0, bbox3d.tensor[i, 8])
                    box = NuScenesBox(
                        box_gravity_center[i],
                        nus_box_dims[i],
                        quat,
                        label=labels[i],
                        score=scores[i],
                        velocity=velocity,
                    )
                    box_list.append(box)
            else:
                raise NotImplementedError(f"Do not support convert {type(bbox3d)} bboxes " "to standard NuScenesBoxes.")

            return box_list, attrs
        
        mmdet3dmetrics.output_to_nusc_box = output_to_nusc_box
