# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from mmseg.utils import get_root_logger
import torch
import torch.distributed as dist

class MeanIoU:

    def __init__(self,
                 class_indices,
                 ignore_label: int,
                 label_str,
                 name
                 # empty_class: int
        ):
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.ignore_label = ignore_label
        self.label_str = label_str
        self.name = name

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.num_classes).cuda()
        self.total_correct = torch.zeros(self.num_classes).cuda()
        self.total_positive = torch.zeros(self.num_classes).cuda()

    def _after_step(self, outputs, targets):
        outputs = outputs[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]

        for i, c in enumerate(self.class_indices):
            self.total_seen[i] += torch.sum(targets == c).item()
            self.total_correct[i] += torch.sum((targets == c)
                                               & (outputs == c)).item()
            self.total_positive[i] += torch.sum(outputs == c).item()

    def _after_epoch(self):
        dist.all_reduce(self.total_seen)
        dist.all_reduce(self.total_correct)
        dist.all_reduce(self.total_positive)

        ious = []

        for i in range(self.num_classes):
            cur_iou = self.total_correct[i] / (self.total_seen[i]
                                               + self.total_positive[i]
                                               - self.total_correct[i])
            ious.append(cur_iou.item())

        miou = np.mean(ious)
        logger = get_root_logger()
        logger.info(f'Validation per class iou {self.name}:')
        for iou, label_str in zip(ious, self.label_str):
            logger.info('%s : %.2f%%' % (label_str, iou * 100))
        
        return miou * 100
        
