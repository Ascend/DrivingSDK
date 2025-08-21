# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from OpenPCDet
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]

import os
import random

import PIL
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F

from PIL import Image
from mx_driving.modules.voxelization import Voxelization


def balanced_resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        raise RuntimeError("COCO dynamic transform methond only support detection tasks!")

    if w < h and w / h < 0.75:
        rescaled_image = rescaled_image.transpose(Image.ROTATE_90)
        rw, rh = rescaled_image.size

        ratio_width = rw / w
        ratio_height = rh / h

        target = target.copy()
        if "boxes" in target and target['boxes'].numel() > 0:
            boxes = target["boxes"]
            transformed_boxes = []
            for box in target["boxes"]:
                xmin, ymin, xmax, ymax = box

                new_xmin = ymin
                new_xmax = ymax
                new_ymin = w - xmax
                new_ymax = w - xmin
                transformed_boxes.append([new_xmin, new_ymin, new_xmax, new_ymax])

            target["boxes"] = torch.tensor(transformed_boxes)

        target["size"] = torch.tensor([rh, rw])

        if "masks" in target:
            raise RuntimeError("COCO dynamic transform methond only support detection tasks!")

    return rescaled_image, target


class BalancedRandomResize(object):
    def __init__(self, sizes, max_size=None, seed=None):
        if not isinstance(sizes, (list, tuple)):
            raise TypeError("sizes must be type of `list` or `tuple`.")

        self.sizes = sizes
        self.max_size = max_size
        self.idx = 0
        self.length = len(self.sizes)

        if seed:
            torch.manual_seed(seed)
            random.seed(seed)
        self.size = self.sizes[random.randint(0, self.length - 1)]

    def __call__(self, img, target=None):
        if self.idx % 8 == 0:
            self.size = self.sizes[random.randint(0, self.length - 1)]

        self.idx += 1
        return balanced_resize(img, target, self.size, self.max_size)


def get_voxel_number_from_mean_vfe(data_path, filename, sweeps, max_sweeps):
    seed = 666
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    voxel_size = [0.075, 0.075, 0.2]
    max_num_points = 10
    max_voxels = 120000
    point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
    voxel_generator = Voxelization(voxel_size, point_cloud_range, max_num_points, max_voxels)

    points = np.fromfile(str(os.path.join(data_path, filename)), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
    sweep_points_list = [points]
    sweep_times_list = [np.zeros((points.shape[0], 1))]

    for k in np.random.choice(len(sweeps), max_sweeps - 1, replace=False):
        lidar_path = os.path.join(data_path, sweeps[k]['lidar_path'])
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        mask = ~((np.abs(points_sweep[:, 0]) < 1.0) & (np.abs(points_sweep[:, 1]) < 1.0))
        points_sweep = points_sweep[mask].T
        if sweeps[k]['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweeps[k]['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweeps[k]['time_lag'] * np.ones((1, points_sweep.shape[1]))
        sweep_points_list.append(points_sweep.T)
        sweep_times_list.append(cur_times.T)

    points = np.concatenate(sweep_points_list, axis=0)
    times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

    points = np.concatenate((points, times), axis=1)

    _, _, coordinates, _ = voxel_generator(torch.tensor(points).npu())
    return coordinates.shape[0]
