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
import mx_driving.detection

torch.npu.config.allow_internal_format = False
torch_npu.npu.set_compile_mode(jit_compile=False)
DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]
EPS = 1e-8


def cpu_roi_align_rotated(input_array, rois, args_dict):
    spatial_scale, sampling_ratio, pooled_height, pooled_width, aligned, clockwise = args_dict.values()
    N, C, H, W = input_array.shape
    output_shape = [rois.shape[0], C, pooled_height, pooled_width]
    number = reduce(lambda x1, x2: x1 * x2, output_shape)
    output = np.zeros(number).astype(np.float32)
    feature_map = input_array.reshape(-1)

    roi_offset = 0.5 if aligned else 0
    roi_batch_idx = rois[:, 0]
    roi_center_w = rois[:, 1] * spatial_scale - roi_offset
    roi_center_h = rois[:, 2] * spatial_scale - roi_offset
    roi_width = rois[:, 3] * spatial_scale
    roi_height = rois[:, 4] * spatial_scale
    theta = rois[:, 5]
    theta = -theta if clockwise else theta

    if not aligned:
        roi_width = np.maximum(roi_width, 1)
        roi_height = np.maximum(roi_height, 1)

    bin_size_h = roi_height / pooled_height
    bin_size_w = roi_width / pooled_width

    if sampling_ratio > 0:
        roi_bin_grid_h = np.ones(bin_size_h.shape).astype("int32")
        roi_bin_grid_w = np.ones(bin_size_w.shape).astype("int32")
        roi_bin_grid_h = roi_bin_grid_h * sampling_ratio
        roi_bin_grid_w = roi_bin_grid_w * sampling_ratio
    else:
        roi_bin_grid_h = np.ceil(bin_size_h).astype("int32")
        roi_bin_grid_w = np.ceil(bin_size_w).astype("int32")
    
    roi_start_h = -roi_height / 2
    roi_start_w = -roi_width / 2
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    count = np.maximum(roi_bin_grid_h * roi_bin_grid_w, 1)

    for index in range(number):
        pw = index % pooled_width
        ph = (index // pooled_width) % pooled_height
        c = (index // pooled_width // pooled_height) % C
        n = index // pooled_width // pooled_height // C

        start_h = roi_start_h[n]
        start_w = roi_start_w[n]
        grid_h = roi_bin_grid_h[n]
        grid_w = roi_bin_grid_w[n]
        center_h = roi_center_h[n]
        center_w = roi_center_w[n]
        size_h = bin_size_h[n]
        size_w = bin_size_w[n]

        fm_batch = int(roi_batch_idx[n])

        if 0 <= fm_batch < N:
            val_n = count[n]
            sin_theta_n = sin_theta[n]
            cos_theta_n = cos_theta[n]
            output_val = 0

            for iy in range(grid_h):
                yy = start_h + ph * size_h + (iy + 0.5) * size_h / grid_h

                for ix in range(grid_w):
                    xx = start_w + pw * size_w + (ix + 0.5) * size_w / grid_w

                    x_val = yy * sin_theta_n + xx * cos_theta_n + center_w
                    y_val = yy * cos_theta_n - xx * sin_theta_n + center_h
                    
                    bilinear_dict = dict(C=C,
                                         H=H,
                                         W=W,
                                         y_val=y_val,
                                         x_val=x_val,
                                         c=c)
                    val = bilinear_interpolate(feature_map, fm_batch, bilinear_dict)

                    output_val += val
            
            output_val = output_val / val_n
            output[index] = output_val

    output = output.reshape(output_shape)

    return output


def bilinear_interpolate(feature_map, fm_batch, bilinear_args):
    channels, height, width, y_val, x_val, c = bilinear_args.values()

    if y_val < -1.0 or y_val > height:
        return 0
    if x_val < -1.0 or x_val > width:
        return 0
    if y_val <= 0:
        y_val = 0
    if x_val <= 0:
        x_val = 0
    
    y_low = int(y_val)
    x_low = int(x_val)

    if y_low >= height - 1:
        y_high = y_low = height - 1
        y_val = y_low
    else:
        y_high = y_low + 1
    
    if x_low >= width - 1:
        x_high = x_low = width - 1
        x_val = x_low
    else:
        x_high = x_low + 1
    
    ly = y_val - y_low
    lx = x_val - x_low
    hy = 1 - ly
    hx = 1 - lx

    fm_idx = (fm_batch * channels + c) * height * width

    v1 = feature_map[fm_idx + y_low * width + x_low]
    v2 = feature_map[fm_idx + y_low * width + x_high]
    v3 = feature_map[fm_idx + y_high * width + x_low]
    v4 = feature_map[fm_idx + y_high * width + x_high]

    w1 = hy * hx
    w2 = hy * lx
    w3 = ly * hx
    w4 = ly * lx

    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return val


class TestRoiAlignedRotatedV2(TestCase):
    def cpu_to_exec(self, features, rois, args_dict):
        features = features.numpy()
        rois = rois.numpy()
        output = cpu_roi_align_rotated(features, rois, args_dict)
        return torch.from_numpy(output)

    def npu_to_exec(self, features, rois, args_dict):
        spatial_scale, sampling_ratio, ph, pw, aligned, clockwise = args_dict.values()
        output = mx_driving.detection.roi_align_rotated_v2(features.npu(), rois.npu(), spatial_scale, sampling_ratio, ph, pw, aligned, clockwise)
        return output.cpu()

    def generate_features(self, feature_shape):
        features = torch.rand(feature_shape)
        return features

    def generate_rois(self, roi_shape, feature_shape, spatial_scale):
        num_boxes = roi_shape[0]
        rois = torch.Tensor(6, num_boxes)
        rois[0] = torch.randint(0, feature_shape[0], (num_boxes,))
        rois[1].uniform_(0, feature_shape[2]) / spatial_scale
        rois[2].uniform_(0, feature_shape[3]) / spatial_scale
        rois[3].uniform_(0, feature_shape[2]) / spatial_scale
        rois[4].uniform_(0, feature_shape[3]) / spatial_scale
        rois[5].uniform_(0, math.pi)

        return rois.transpose(0, 1).contiguous()
    
    @unittest.skipIf(DEVICE_NAME not in ['Ascend910B', 'Ascend910C'], "OP `RoiAlignedRotatedV2` is only supported on 910B and 910C, skip this ut!")
    def test_RoiAlignedRotatedV2_Aligned(self):
        shape_format = [
            [[8, 32, 64, 64], [16, 6], 0.5, 2, 7, 7, True, False],
            [[8, 32, 64, 64], [32, 6], 0.5, 2, 7, 7, True, True],
            [[8, 128, 64, 64], [32, 6], 0.5, 2, 7, 7, True, False],
            [[8, 128, 64, 64], [48, 6], 0.5, 2, 7, 7, True, False],
            [[8, 128, 64, 64], [48, 6], 0.5, 2, 9, 9, True, False],
        ]
        for item in shape_format:
            features = self.generate_features(item[0])
            rois = self.generate_rois(item[1], item[0], item[2])
            spatial_scale = item[2]
            sampling_ratio = item[3]
            ph = item[4]
            pw = item[5]
            aligned = item[6]
            clockwise = item[7]
            args_dict = dict(spatial_scale=spatial_scale, 
                             sampling_ratio=sampling_ratio,
                             ph=ph,
                             pw=pw, 
                             aligned=aligned, 
                             clockwise=clockwise)
            out_cpu = self.cpu_to_exec(features, rois, args_dict)
            out_npu = self.npu_to_exec(features, rois, args_dict)
            self.assertRtolEqual(out_cpu, out_npu)

    @unittest.skipIf(DEVICE_NAME not in ['Ascend910B', 'Ascend910C'], "OP `RoiAlignedRotatedV2` is only supported on 910B and 910C, skip this ut!")
    def test_RoiAlignedRotatedV2_NonAligned(self):
        shape_format = [
            [[8, 3, 64, 64], [16, 6], 0.5, 2, 7, 7, True, True],
            [[1, 32, 64, 64], [32, 6], 0.5, 2, 7, 7, True, True],
            [[3, 13, 64, 64], [32, 6], 0.5, 2, 7, 7, True, True],
            [[3, 7, 18, 19], [23, 6], 0.5, 2, 1, 1, True, False],
            [[8, 128, 64, 64], [39, 6], 0.5, 2, 9, 9, True, False],
        ]
        for item in shape_format:
            features = self.generate_features(item[0])
            rois = self.generate_rois(item[1], item[0], item[2])
            spatial_scale = item[2]
            sampling_ratio = item[3]
            ph = item[4]
            pw = item[5]
            aligned = item[6]
            clockwise = item[7]
            args_dict = dict(spatial_scale=spatial_scale, 
                             sampling_ratio=sampling_ratio,
                             ph=ph,
                             pw=pw, 
                             aligned=aligned, 
                             clockwise=clockwise)
            out_cpu = self.cpu_to_exec(features, rois, args_dict)
            out_npu = self.npu_to_exec(features, rois, args_dict)
            self.assertRtolEqual(out_cpu, out_npu)

if __name__ == '__main__':
    run_tests()