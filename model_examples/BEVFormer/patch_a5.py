# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.

import os
import sys
from types import ModuleType
from typing import Dict
import mx_driving
from mx_driving.patcher import PatcherBuilder, Patch
from mx_driving.patcher import numpy_type
import nuscenes

sys.path.append("..")



def data_classess(models: ModuleType, options: Dict):
    from typing import List, Dict
    from nuscenes.eval.detection.constants import DETECTION_NAMES
    def __init__(self,
                 class_range: Dict[str, int],
                 dist_fcn: str,
                 dist_ths: List[float],
                 dist_th_tp: float,
                 min_recall: float,
                 min_precision: float,
                 max_boxes_per_sample: int,
                 mean_ap_weight: int):

        assert set(class_range.keys()) == set(DETECTION_NAMES), "Class count mismatch."
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

        self.class_names = list(self.class_range.keys())

    if hasattr(models, "DetectionConfig"):
        models.DetectionConfig.__init__ = __init__


def generate_patcher_builder():
    bevformer_patcher_builder = (
        PatcherBuilder()
        .add_module_patch("numpy", Patch(numpy_type))
        .add_module_patch("nuscenes.eval.detection.data_classes", Patch(data_classess))
    )
    return bevformer_patcher_builder