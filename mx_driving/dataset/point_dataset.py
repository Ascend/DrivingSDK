# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from ..dataset.utils.dynamic_dataset import CapacityBucketingDynamicDataset


class PointCloudDynamicDataset(CapacityBucketingDynamicDataset):
    def sorting(self):
        self.infos.sort(key=lambda x: x['voxel_num'], reverse=True)
        self.sorted_ids = list(range(len(self.infos)))