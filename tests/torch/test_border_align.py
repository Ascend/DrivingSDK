"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""
import unittest
import math
from typing import List
from functools import reduce

import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import mx_driving

from mx_driving.detection import border_align

torch.npu.config.allow_internal_format = False
torch_npu.npu.set_compile_mode(jit_compile=False)
DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]
EPS = 1e-8


def generate_features(feature_shape):
    features = torch.rand(feature_shape)
    return features


def generate_rois(inputs):
    num_boxes = inputs.shape[0] * inputs.shape[2] * inputs.shape[3]
    xyxy = torch.rand(num_boxes, 4)
    xyxy[:, 0::2] = xyxy[:, 0::2] * inputs.size(3)
    xyxy[:, 1::2] = xyxy[:, 1::2] * inputs.size(2)
    xyxy[:, 2:] = xyxy[:, 0:2] + xyxy[:, 2:]
    rois = xyxy.view(inputs.shape[0], -1, 4).contiguous()
    return rois


def border_align_cpu_golden(inputs, rois, pooled_size_):
    n, c4, h, w = inputs.shape
    c = c4 // 4
    assert rois.size(1) == h * w
    inputs = inputs.view(n, 4, c, h, w).permute(0, 2, 3, 4, 1).contiguous()
    outputs_features = torch.zeros(n, c, h * w, 4)
    outputs_index = torch.zeros(n, c, h * w, 4).int()
    for index in (range(n * c * h * w)):
        pn = index // (c * h * w)
        pc = (index // (h * w)) % c
        ph = (index // w) % h
        pw = index % w

        features = inputs[pn, pc]
        x1, y1 = rois[pn, ph * w + pw, 0], rois[pn, ph * w + pw, 1]
        x2, y2 = rois[pn, ph * w + pw, 2], rois[pn, ph * w + pw, 3]
        width, height = x2 - x1, y2 - y1
        dx, dy = width / pooled_size_, height / pooled_size_
        ops = [[dx, 0], [0, dy], [-dx, 0], [0, -dy]]
        start_points = [[x1, y1], [x1, y1], [x2, y2], [x2, y2]]
        for i in range(4):
            x, y = start_points[i][0], start_points[i][1]
            offset_features = features[:, :, i].view(-1).contiguous()
            val = bilinear_interpolate(offset_features, h, w, y, x)
            idx = 0
            for j in range(1, pooled_size_ + 1):
                x, y = x + ops[i][0], y + ops[i][1]
                tmp = bilinear_interpolate(offset_features, h, w, y, x)
                if tmp > val:
                    val = tmp
                    idx = j
            outputs_features[pn, pc, ph * w + pw, i] = val
            outputs_index[pn, pc, ph * w + pw, i] = idx

    return outputs_features


def bilinear_interpolate(offset_input, height, width, y, x):
    if y < -1 or y > height:
        return 0
    if x < -1 or x > width:
        return 0
    y = y if y > 0 else 0
    x = x if x > 0 else 0
    y_low = int(y)
    x_low = int(x)
    if y_low >= height - 1:
        y_high = y_low = height - 1
        y = y_low
    else:
        y_high = y_low + 1
    if x_low >= width - 1:
        x_high = x_low = width - 1
        x = x_low
    else:
        x_high = x_low + 1

    ly = float(y - y_low)
    lx = float(x - x_low)
    hy = 1 - ly
    hx = 1 - lx
    v1 = offset_input[y_low * width + x_low]
    v2 = offset_input[y_low * width + x_high]
    v3 = offset_input[y_high * width + x_low]
    v4 = offset_input[y_high * width + x_high]
    w1 = (hy * hx)
    w2 = (hy * lx)
    w3 = (ly * hx)
    w4 = (ly * lx)
    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return val


class TestBorderAlign(TestCase):
    def cpu_to_exec(self, features, rois, pooled_size):
        output = border_align_cpu_golden(features.detach().cpu(), rois.detach().cpu(), pooled_size)
        return output

    def npu_to_exec(self, features, rois, pooled_size):
        output = border_align(features.npu(), rois.npu(), pooled_size)
        return output

    @unittest.skipIf(DEVICE_NAME not in ['Ascend910B'], "OP `BorderAlign` is not supported, skip this ut!")
    def test_border_align(self):
        shape_format = [
            # Aligned Case
            [1, 16, 16, 16, 5],
            [2, 8, 8, 24, 5],
            [1, 32, 16, 8, 7],
            [2, 16, 4, 12, 5],
            # Not Aligned Case
            [2, 36, 5, 13, 5],
            [3, 20, 29, 3, 6],
            [2, 28, 11, 17, 3],
            [1, 12, 7, 33, 2],
        ]
        for item in shape_format:
            batch_size = item[0]
            input_channels = item[1]
            input_height = item[2]
            input_width = item[3]
            pooled_size = item[4]

            features = generate_features([batch_size, input_channels, input_height, input_width])
            rois = generate_rois(features)
            
            out_cpu = self.cpu_to_exec(features, rois, pooled_size)
            out_npu = self.npu_to_exec(features, rois, pooled_size)
            error = out_cpu - out_npu.cpu()
            self.assertRtolEqual(out_cpu, out_npu)


if __name__ == '__main__':
    run_tests()