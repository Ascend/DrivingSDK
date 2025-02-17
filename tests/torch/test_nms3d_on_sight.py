"""
Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
"""
import copy
from math import cos, sin, fabs, atan2
import unittest
from functools import reduce
from typing import List

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving
from mx_driving import nms3d_on_sight


torch.npu.config.allow_internal_format = False
torch_npu.npu.set_compile_mode(jit_compile=False)
DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]
EPS = 1e-8

def in_front_120fov(m, n):
    """
    Determines whether a given point (m, n) falls within a 120-degree field of view (FOV) in front

    Parameters:
    m (float): The distance of the point from the reference point along one axis (e.g., z-axis).
    n (float): The distance of the point from the reference point along another axis (e.g., x-axis or y-axis).
    
    return:
    bool: True if the point is within the 120-degree FOV; False otherwise.
    """
    return m > (1.73205 * fabs(n) / 3)

def dist_bev(box_a: List[float], box_b: List[float]):
    '''
    Calculate the dist_bev (BEV distance) between box_a and box_b.

    params box_a: [x, y, z, dx, dy, dz, heading]
    params box_b: [x, y, z, dx, dy, dz, heading]
    '''
    # 5.0 meter
    max_merge_dist = 5.0
    # 30° degree
    max_ry_diff = 0.523598
    # flag value, means 10m ** 2
    very_far = -100.0

    # the angle between (ego, cente_a) and (ego, center_b) should be less than 90°
    if box_a[0] * box_b[0] + box_a[1] * box_b[1] <= 0:
        return very_far

    # only merge box not in lidar area
    if (in_front_120fov(box_a[0], box_a[1]) or in_front_120fov(box_b[0], box_b[1])):
        return very_far

    # If the squared distance is greater than the maximum allowed merge distance squared(5.0 meter), consider the boxes as 'very_far' apart.
    diff_x = box_a[0] - box_b[0]
    diff_y = box_a[1] - box_b[1]
    dist_diff = -(diff_x * diff_x + diff_y * diff_y)

    if dist_diff <= -(max_merge_dist * max_merge_dist):
        return very_far
    
    # Check if the difference in rotation around the Y-axis between the two boxes exceeds the maximum allowed rotation difference(30 degree).
    if fabs(box_a[6] - box_b[6]) >= max_ry_diff:
        return very_far
    
    # bev_dist
    up = box_a[0] * box_b[1] - box_a[1] * box_b[0]
    down = max(box_a[0] * box_a[0] + box_a[1] * box_a[1], box_b[0] * box_b[0] + box_b[1] * box_b[1]) + 0.0001
    return -(up * up) / down

def nms3d_forward(boxes: List[List[float]], nms3d_thresh=0.0):
    mask = np.ones(boxes.shape[0], dtype=int)
    keep = -np.ones(boxes.shape[0])
    out_num = 0
    bev_dist = 0
    for i in range(0, boxes.shape[0]):
        if mask[i] == 0:
            continue
        keep[out_num] = i
        out_num += 1
        for j in range(i + 1, boxes.shape[0]):
            bev_dist = dist_bev(boxes[i], boxes[j])
            if bev_dist > nms3d_thresh:
                mask[j] = 0
    return keep, out_num

def nms3d_cpu(boxes, scores, threshold=0.0):
    boxes = torch.from_numpy(boxes)
    scores = torch.from_numpy(scores)
    order = scores.sort(0, descending=True)[1]
    boxes = boxes[order].contiguous()
    boxes = np.array(boxes, dtype=np.float32)
    keep, num_out = nms3d_forward(boxes, threshold)
    keep = order[keep[:num_out].astype(int)]
    return np.array(keep)

def generate_boxes(boxes_shape):
    boxes_np = np.round(np.random.uniform(0, 100, boxes_shape), 2).astype(np.float32)
    angle_np = np.round(np.random.uniform(0, 1, boxes_shape[0]), 2).astype(np.float32)
    boxes_np[:, 6] = angle_np
    boxes = torch.from_numpy(boxes_np)
    return boxes

def generate_unique_random_scores(scores_count, precision=4):
    step = 10 ** -precision
    numbers = np.arange(0, 1, step)
    if len(numbers) < scores_count:
        raise ValueError("Requested number of unique values is larger than possible unique values at this precision.")
    scores_np = np.random.choice(numbers, size=scores_count, replace=False).astype(np.float32)
    scores = torch.from_numpy(scores_np)
    return scores


class TestNms3dOnSight(TestCase):
    def cpu_to_exec(self, boxes, scores, threshold):
        boxes_cpu = boxes.cpu()
        scores_cpu = scores.cpu()
        cpu_outputs = nms3d_cpu(boxes_cpu.numpy(), scores_cpu.numpy(), -threshold**2)
        return cpu_outputs


    def npu_to_exec(self, boxes, scores, threshold):
        npu_outputs = nms3d_on_sight(boxes.npu(), scores.npu(), threshold)
        return npu_outputs

    @unittest.skipIf(DEVICE_NAME not in ['Ascend910B'], "OP `BorderAlign` is not supported, skip this ut!")
    def test_nms3d_on_sight(self):
        shape_format = [
            [57, 7],
            [150, 7],
            [400, 7],
            [1000, 7],
            [1624, 7],
            [2324, 7],
            [2500, 7]
        ]
        for item in shape_format:
            boxes_shape = item
            scores_count = item[0]

            boxes = generate_boxes(boxes_shape).npu()
            scores = generate_unique_random_scores(scores_count).npu()
            threshold = np.random.uniform(-100, 100)
            
            out_npu = npu_to_exec(boxes, scores, threshold)
            out_cpu = cpu_to_exec(boxes, scores, threshold)
            self.assertRtolEqual(out_cpu, out_npu.cpu())
