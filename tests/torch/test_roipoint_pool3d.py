# Copyright (c) 2024, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import unittest

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

from mx_driving import roipoint_pool3d
from mx_driving.preprocess import RoIPointPool3d, roipoint_pool3d


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


# float16[-14,16], float32[-126,128], float64[-1022,1024], int16[0,15], int32[0,31], int64[0,63]
# random_value(-7, 8, (1, 2, 3), np.float32, True, True, False, False)
# pylint: disable=too-many-arguments,huawei-too-many-arguments
@golden_data_cache(__file__)
def random_value(
    min_log, max_log, size, dtype=np.float32, nega_flag=True, zero_flag=True, inf_flag=False, nan_flag=False
):
    matrix_log = np.random.uniform(low=min_log, high=max_log, size=size).astype(np.float32)
    matrix = np.exp2(matrix_log).astype(dtype)
    flag_value = int(zero_flag) + int(inf_flag) + int(nan_flag)
    size_value = np.prod(size)
    p0 = 0.1
    if (flag_value > 0) and (size_value > 0):
        p0 = 0.1 / flag_value / size_value  # 10%
    if nega_flag:
        matrix *= np.random.choice(a=[1, -1], size=size, p=[0.5, 0.5])
    if zero_flag:
        matrix *= np.random.choice(a=[1, 0], size=size, p=[1 - p0, p0])
    if inf_flag:
        np_inf = np.array([np.inf]).astype(dtype)[0]
        matrix += np.random.choice(a=[0, np_inf], size=size, p=[1 - p0, p0])
    if nan_flag:
        np_nan = np.array([np.nan]).astype(dtype)[0]
        matrix += np.random.choice(a=[0, np_nan], size=size, p=[1 - p0, p0])
    return matrix


# points: (B, N, 3) 输入点
# point_features: (B, N, C) 输入点特征
# boxes3d: (B, M, 7) 边界框
# pooled_features: (B, M, num, 3+C) 特征汇聚
# pooled_empty_flag: (B, M) 空标志
def check_point_in_box3d(point, box3d):
    x = point[0]
    y = point[1]
    z = point[2]
    cx = box3d[0]
    cy = box3d[1]
    cz = box3d[2]
    x_size = box3d[3]
    y_size = box3d[4]
    z_size = box3d[5]
    rz = box3d[6]
    if (z_size < 0) or ((z - cz) < 0) or ((z - cz) > z_size):
        return 0
    cosa = np.cos(-rz)
    sina = np.sin(-rz)
    local_x = (x - cx) * cosa + (y - cy) * (-sina)
    local_y = (x - cx) * sina + (y - cy) * cosa
    in_flag = (abs(local_x) < (x_size / 2)) and (abs(local_y) < (y_size / 2))
    return in_flag


@golden_data_cache(__file__)
def roipoint_pool3d_forward(num_sampled_points, points, point_features, boxes3d, pooled_features):
    point_num = points.shape[0]  # N
    feature_len = point_features.shape[1]  # C
    point_flag = np.zeros((point_num), dtype=np.int32)  # (N)
    point_idx = np.zeros((num_sampled_points), dtype=np.int32)  # (num)

    for pt_idx in range(point_num):
        point_flag[pt_idx] = check_point_in_box3d(points[pt_idx], boxes3d)

    cnt = 0
    for pt_idx in range(point_num):
        if point_flag[pt_idx] == 0:
            continue
        point_idx[cnt] = pt_idx
        cnt += 1
        if cnt == num_sampled_points:
            break

    if cnt == 0:
        return 1
    if cnt < num_sampled_points:
        for spn_idx in range(cnt, num_sampled_points):
            point_idx[spn_idx] = point_idx[spn_idx % cnt]

    for sample_point_idx in range(num_sampled_points):
        src_point_idx = point_idx[sample_point_idx]
        pooled_features[sample_point_idx, 0:3] = points[src_point_idx, 0:3]
        pooled_features[sample_point_idx, 3 : 3 + feature_len] = point_features[src_point_idx, 0:feature_len]
    return 0


@golden_data_cache(__file__)
def cpu_roipoint_pool3d(num_sampled_points, points, point_features, boxes3d):
    # B=batch_size; N=point_num; M=boxes_num; C=feature_len; num = num_sampled_points
    batch_size = points.shape[0]  # B
    feature_len = point_features.shape[2]  # C
    boxes_num = boxes3d.shape[1]  # M
    pooled_features = np.zeros_like(points, shape=(batch_size, boxes_num, num_sampled_points, 3 + feature_len))
    pooled_empty_flag = np.zeros((batch_size, boxes_num), dtype=np.int32)
    for bs_idx in range(batch_size):
        for boxes_idx in range(boxes_num):
            pooled_empty_flag[bs_idx][boxes_idx] = roipoint_pool3d_forward(
                num_sampled_points,
                points[bs_idx],
                point_features[bs_idx],
                boxes3d[bs_idx][boxes_idx],
                pooled_features[bs_idx][boxes_idx],
            )
    return pooled_features, pooled_empty_flag


class TestRoipointPool3d(TestCase):
    @unittest.skipIf(DEVICE_NAME != "Ascend910B", "OP `RoipointPool3d` is only supported on 910B, skip this ut!")
    def test_roipoint_pool3d_float(self):
        random.seed()
        batch_size = random.randint(1, 4)  # B
        num_sampled_points = random.randint(1, 48)  # num
        boxes_num = random.randint(1, 48)  # M
        point_num = random.randint(max(num_sampled_points, boxes_num), 105)  # N
        points = random_value(-15.5, 16, (batch_size, point_num, 3), np.float32)  # (B, N, 3)
        point_features = points.copy()  # (B, N, C)
        boxes3d = np.zeros((batch_size, boxes_num, 7), dtype=np.float32)  # (B, M, 7)
        boxes3d[0:, 0:, 0:3] = random_value(-15.5, 16, (batch_size, boxes_num, 3))
        boxes3d[0:, 0:, 3:6] = random_value(-63, 64, (batch_size, boxes_num, 3), nega_flag=False)
        boxes3d[0:, 0:, 6:] = np.random.uniform(low=0, high=3.141592654, size=(batch_size, boxes_num, 1)).astype(
            np.float32
        )

        cpu_pooled_features, cpu_pooled_empty_flag = cpu_roipoint_pool3d(
            num_sampled_points, points, point_features, boxes3d
        )

        roipoint_pool3d_tmp = RoIPointPool3d(num_sampled_points)
        pooled_features, pooled_empty_flag = roipoint_pool3d_tmp(
            torch.from_numpy(points).npu(), torch.from_numpy(point_features).npu(), torch.from_numpy(boxes3d).npu()
        )
        
        pooled_features, pooled_empty_flag = roipoint_pool3d(
            num_sampled_points, torch.from_numpy(points).npu(), torch.from_numpy(point_features).npu(), torch.from_numpy(boxes3d).npu()
        )

        float_pooled_features = pooled_features.cpu().numpy()
        float_pooled_empty_flag = pooled_empty_flag.cpu().numpy()
        self.assertRtolEqual(float_pooled_features, cpu_pooled_features, prec=0.00005)  # (B, M, num, 3+C)
        self.assertRtolEqual(float_pooled_empty_flag, cpu_pooled_empty_flag, prec=0.00005)  # (B, M)

    @unittest.skipIf(DEVICE_NAME != "Ascend910B", "OP `RoipointPool3d` is only supported on 910B, skip this ut!")
    def test_roipoint_pool3d_half(self):
        random.seed()
        batch_size = random.randint(1, 4)  # B
        num_sampled_points = random.randint(1, 60)  # num
        boxes_num = random.randint(1, 60)  # M
        point_num = random.randint(max(num_sampled_points, boxes_num), 105)  # N
        points = random_value(-3.5, 4, (batch_size, point_num, 3), np.float16)  # (B, N, 3)
        point_features = points.copy()  # (B, N, C)
        boxes3d = np.zeros((batch_size, boxes_num, 7), dtype=np.float16)  # (B, M, 7)
        boxes3d[0:, 0:, 0:3] = random_value(-3.5, 4, (batch_size, boxes_num, 3), np.float16)
        boxes3d[0:, 0:, 3:6] = random_value(-14, 16, (batch_size, boxes_num, 3), np.float16, nega_flag=False)
        boxes3d[0:, 0:, 6:] = np.random.uniform(low=0, high=3.142, size=(batch_size, boxes_num, 1)).astype(np.float16)

        cpu_pooled_features, cpu_pooled_empty_flag = cpu_roipoint_pool3d(
            num_sampled_points, points.astype(np.float32), point_features.astype(np.float32), boxes3d.astype(np.float32)
        )

        roipoint_pool3d_tmp = RoIPointPool3d(num_sampled_points)
        pooled_features, pooled_empty_flag = roipoint_pool3d_tmp(
            torch.from_numpy(points).npu(), torch.from_numpy(point_features).npu(), torch.from_numpy(boxes3d).npu()
        )
        
        pooled_features, pooled_empty_flag = roipoint_pool3d(
            num_sampled_points, torch.from_numpy(points).npu(), torch.from_numpy(point_features).npu(), torch.from_numpy(boxes3d).npu()
        )

        half_pooled_features = pooled_features.cpu().numpy().astype(np.float32)
        half_pooled_empty_flag = pooled_empty_flag.cpu().numpy()
        self.assertRtolEqual(half_pooled_features, cpu_pooled_features, prec=0.0005)  # (B, M, num, 3+C)
        self.assertRtolEqual(half_pooled_empty_flag, cpu_pooled_empty_flag, prec=0.0005)  # (B, M)


if __name__ == "__main__":
    run_tests()
