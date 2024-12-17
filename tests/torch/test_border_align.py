"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""
import copy
import math
import unittest
from functools import reduce
from typing import List

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving
from mx_driving import border_align


torch.npu.config.allow_internal_format = False
torch_npu.npu.set_compile_mode(jit_compile=False)
DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]
EPS = 1e-8


@golden_data_cache(__file__)
def generate_features(feature_shape):
    features = torch.rand(feature_shape)
    return features


@golden_data_cache(__file__)
def generate_grad_outputs(output_shape):
    grad_outputs = torch.rand(output_shape)
    return grad_outputs


@golden_data_cache(__file__)
def generate_rois(inputs):
    num_boxes = inputs.shape[0] * inputs.shape[2] * inputs.shape[3]
    xyxy = torch.rand(num_boxes, 4)
    xyxy[:, 0::2] = xyxy[:, 0::2] * inputs.size(3)
    xyxy[:, 1::2] = xyxy[:, 1::2] * inputs.size(2)
    xyxy[:, 2:] = xyxy[:, 0:2] + xyxy[:, 2:]
    rois = xyxy.view(inputs.shape[0], -1, 4).contiguous()
    return rois


@golden_data_cache(__file__)
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

    return outputs_features, outputs_index


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


def bilinear_interpolate_backward(inputs, args_dict):
    """
    双线性插值
    input--> [4C, H, W]
    x or y [pool_size + 1] 个
    output [C, 1]  argmax_idx [C, 1]
    """
    x, y, c_start, output, argmax_idx = args_dict.values()
    C = output.shape[0]
    x_floor, y_floor = np.floor(x).astype(int), np.floor(y).astype(int)
    x_ceil, y_ceil = np.ceil(x).astype(int), np.ceil(y).astype(int)

    x_floor = np.clip(x_floor, 0, inputs.shape[2] - 1)
    y_floor = np.clip(y_floor, 0, inputs.shape[1] - 1)
    x_ceil = np.clip(x_ceil, 0, inputs.shape[2] - 1)
    y_ceil = np.clip(y_ceil, 0, inputs.shape[1] - 1)

    u = x - x_floor
    v = y - y_floor

    w1 = (1 - u) * (1 - v)
    w3 = (1 - u) * v
    w2 = u * (1 - v)
    w4 = u * v

    for i in range(C):
        x_ = x[argmax_idx[i]]
        y_ = y[argmax_idx[i]]
        if y_ < -1 or y_ > inputs.shape[1]:
            continue
        if x_ < -1 or x_ > inputs.shape[2]:
            continue
        
        w1_ = w1[argmax_idx[i]]
        w2_ = w2[argmax_idx[i]]
        w3_ = w3[argmax_idx[i]]
        w4_ = w4[argmax_idx[i]]

        x_floor_ = x_floor[argmax_idx[i]]
        y_floor_ = y_floor[argmax_idx[i]]
        x_ceil_ = x_ceil[argmax_idx[i]]
        y_ceil_ = y_ceil[argmax_idx[i]]

        inputs[c_start + i, y_floor_, x_floor_] += output[i] * w1_
        inputs[c_start + i, y_floor_, x_ceil_] += output[i] * w2_
        inputs[c_start + i, y_ceil_, x_floor_] += output[i] * w3_
        inputs[c_start + i, y_ceil_, x_ceil_] += output[i] * w4_

    return inputs


def border_align_box(box, pool_size, inputs, output, argmax_idx):
    ## box为[4], input为[4C, H, W]
    ## output [C, 4],  argmax_idx [C, 4]
    # 解析 box
    x1, y1, x2, y2 = box
    
    #计算对应channel
    c_idx = inputs.shape[0] // 4

    # 遍历四个边缘
    #      shape (N, 4C, h, w) for input.
    #  [0,C) for top feature, [C,2C) for left feature,
    #  [2C,3C) for bottom feature, [3C,4C) for right feature
    for i in range(4):
        if i == 0:  # 上边边缘
            x = np.linspace(x1, x2, num=pool_size + 1)
            y = np.full_like(x, y1)

        elif i == 1:  # 左边边缘
            y = np.linspace(y1, y2, num=pool_size + 1)
            x = np.full_like(y, x1)

        elif i == 2:  # 下边边缘 --->生成x序列反向-->ascend C实现可以参照cuda的stride方式
            x = np.linspace(x2, x1, num=pool_size + 1)
            y = np.full_like(x, y2)

        elif i == 3:  # 右边边缘---->生成y序列反向
            y = np.linspace(y2, y1, num=pool_size + 1)
            x = np.full_like(y, x2)  
        
        # 双线性插值并找到最大值   
        args_dict = dict(x=x, 
                         y=y, 
                         c_start=c_idx * i, 
                         output=output[:, i], 
                         argmax_idx=argmax_idx[:, i])
        
        bilinear_interpolate_backward(inputs, args_dict) 

    return inputs


@golden_data_cache(__file__)
def border_align_grad_cpu_golden(boxes, pool_size, inputs, grad_output, argmax_idx):
    grad_inputs = torch.zeros_like(inputs).detach().cpu().numpy()
    grad_output = grad_output.transpose(1, 2).contiguous().detach().cpu().numpy()
    argmax_idx = argmax_idx.transpose(1, 2).contiguous().detach().cpu().numpy()
    boxes = boxes.detach().cpu().numpy()
    inputs = inputs.detach().cpu().numpy()
    B, C, H, W = inputs.shape
    C = int(C / 4)

    #对每个batch的每个box进行border_align
    for b in range(B):
        input_each_b = copy.deepcopy(grad_inputs[b])
        output_each_b = grad_output[b] # [HW, C, 4]
        argmax_idx_b = argmax_idx[b] # [HW, C, 4]
        temp = np.zeros((C * 4, H, W)) # [4C, H, W]

        for i, box in enumerate(boxes[b]):
            temp = temp + border_align_box(box, pool_size, input_each_b, output_each_b[i], argmax_idx_b[i]) # [4C, H, W]
            input_each_b.fill(0)

        grad_inputs[b] = copy.deepcopy(temp)

    return grad_inputs


class TestBorderAlign(TestCase):
    def cpu_to_exec(self, features, rois, grad_output, pooled_size):
        output, index = border_align_cpu_golden(features.detach().cpu(), rois.detach().cpu(), pooled_size)
        grad_features = border_align_grad_cpu_golden(rois, pooled_size, features, grad_output, index)
        grad_features = torch.tensor(grad_features)
        return output, grad_features
   

    def npu_to_exec(self, features, rois, grad_output, pooled_size):
        npu_outputs = border_align(features.npu(), rois.npu(), pooled_size)
        npu_outputs.backward(grad_output.npu())
        return npu_outputs, features.grad

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

            features = generate_features([batch_size, input_channels, input_height, input_width]).npu()
            features.requires_grad = True
            rois = generate_rois(features)
            grad_output = generate_grad_outputs([batch_size, input_channels // 4, input_height * input_width, 4])

            out_cpu, grad_cpu = self.cpu_to_exec(features, rois, grad_output, pooled_size)
            out_npu, grad_npu = self.npu_to_exec(features, rois, grad_output, pooled_size)

            self.assertRtolEqual(out_cpu, out_npu.cpu())
            self.assertRtolEqual(grad_cpu, grad_npu.cpu())


if __name__ == '__main__':
    run_tests()