# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
import unittest

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving.preprocess


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


@golden_data_cache(__file__)
def cpu_gen_inputs(range_boxes, range_points, shape_boxes, shape_points):
    boxes = np.random.uniform(range_boxes[0], range_boxes[1], shape_boxes).astype(np.float32)
    boxes = torch.from_numpy(boxes)
    points = np.random.uniform(range_points[0], range_points[1], shape_points).astype(np.float32)
    points = torch.from_numpy(points)

    return boxes, points


def lidar_to_local_coords_cpu(shift_x, shift_y, rz):
    cosa = torch.cos(-rz) 
    sina = torch.sin(-rz)
    
    local_x = shift_x * cosa + shift_y * (-sina)
    local_y = shift_x * sina + shift_y * cosa
    return local_x, local_y


def check_pt_in_box3d_cpu(pt, box3d, idx):
    temp = box3d.detach()
    x = pt[0]
    y = pt[1]
    z = pt[2]
    cx = temp[0]
    cy = temp[1]
    cz = temp[2]
    x_size = temp[3]
    y_size = temp[4]
    z_size = temp[5]
    rz = temp[6]
    cz += z_size / 2.0
    box3d[2] -= z_size / 2.0
    #  shift to the center since cz in box3d is the bottom center
    if (torch.abs(z - cz) > z_size / 2.0):
        return 0
    local_x, local_y = lidar_to_local_coords_cpu(x - cx, y - cy, rz)
    in_flag = torch.abs(local_x) < (x_size / 2.0 + 1e-5) and torch.abs(local_y) < (y_size / 2.0 + 1e-5)
    return in_flag


@golden_data_cache(__file__)
def points_in_boxes_cpu_forward(boxes_tensor, pts_tensor, pts_indices_tensor):
    boxes_num = boxes_tensor.size(0)
    pts_num = pts_tensor.size(0)
    for j in range(pts_num):
        for i in range(boxes_num):
            cur_in_flag = check_pt_in_box3d_cpu(pts_tensor[j], boxes_tensor[i], i)
            if (cur_in_flag):
                pts_indices_tensor[j] = i
                break
    return pts_indices_tensor


class TestPointsInBox(TestCase):
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `PointsInBox` is only supported on 910B, skip this ut!")
    def test_points_in_box_shape_format_fp32(self, device="npu"):
        boxes = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 20.0, 1.0, 0.523598]]],
                             dtype=torch.float32).npu()  # 30 degrees
        pts = torch.tensor(
            [[[4, 6.928, 0], [6.928, 4, 0], [4, -6.928, 0], [6.928, -4, 0],
            [-4, 6.928, 0], [-6.928, 4, 0], [-4, -6.928, 0], [-6.928, -4, 0]]],
            dtype=torch.float32).npu()
        point_indices = mx_driving.preprocess.npu_points_in_box(boxes, pts).cpu().numpy()
        point_indices2 = mx_driving.points_in_box(boxes, pts).cpu().numpy()
        expected_point_indices = torch.tensor([[-1, -1, 0, -1, 0, -1, -1, -1]],
                                            dtype=torch.int32).cpu().numpy()
        
        self.assertRtolEqual(point_indices, expected_point_indices)
        self.assertRtolEqual(point_indices2, expected_point_indices)
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `PointsInBox` is only supported on 910B, skip this ut!")
    def test_points_in_box_shape_randn(self, device="npu"):
        boxes, points = cpu_gen_inputs([0, 1], [0, 2.0], [1, 200, 7], [1, 100, 3])
        shape1 = points.shape
        batch_size = shape1[0]
        num_points = shape1[1]
        num_boxes = boxes.shape[1]
        point_indices = points.new_zeros((batch_size, num_points), dtype=torch.int).fill_(-1)
        for b in range(batch_size):
            point_indices[b] = points_in_boxes_cpu_forward(boxes[b].float(),
                                        points[b].float(),
                                        point_indices[b])

        point_indices_npu = mx_driving.preprocess.npu_points_in_box(boxes.npu(), points.npu())
        self.assertRtolEqual(point_indices.numpy(), point_indices_npu.cpu().numpy())
        point_indices_npu2 = mx_driving.points_in_box(boxes.npu(), points.npu())
        self.assertRtolEqual(point_indices.numpy(), point_indices_npu2.cpu().numpy())
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `PointsInBox` is only supported on 910B, skip this ut!")
    def test_points_in_box_shape_large_boxes(self, device="npu"):
        boxes, points = cpu_gen_inputs([0, 1], [0, 2.0], [1, 2000, 7], [1, 100, 3])
        shape1 = points.shape
        batch_size = shape1[0]
        num_points = shape1[1]
        num_boxes = boxes.shape[1]
        point_indices = points.new_zeros((batch_size, num_points), dtype=torch.int).fill_(-1)
        for b in range(batch_size):
            point_indices[b] = points_in_boxes_cpu_forward(boxes[b].float(),
                                        points[b].float(),
                                        point_indices[b])

        with self.assertRaisesRegex(RuntimeError, "boxes is larger than 200"):
            point_indices_npu = mx_driving.preprocess.npu_points_in_box(boxes.npu(), points.npu())
        with self.assertRaisesRegex(RuntimeError, "boxes is larger than 200"):
            point_indices_npu2 = mx_driving.points_in_box(boxes.npu(), points.npu())
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `PointsInBox` is only supported on 910B, skip this ut!")
    def test_points_in_box_shape_large_points(self, device="npu"):
        boxes, points = cpu_gen_inputs([0, 1], [0, 2.0], [1, 200, 7], [1, 1500, 3])
        shape1 = points.shape
        batch_size = shape1[0]
        num_points = shape1[1]
        num_boxes = boxes.shape[1]
        point_indices = points.new_zeros((batch_size, num_points), dtype=torch.int).fill_(-1)
        for b in range(batch_size):
            point_indices[b] = points_in_boxes_cpu_forward(boxes[b].float(),
                                        points[b].float(),
                                        point_indices[b])

        point_indices_npu = mx_driving.preprocess.npu_points_in_box(boxes.npu(), points.npu())
        self.assertRtolEqual(point_indices.numpy(), point_indices_npu.cpu().numpy())
        point_indices_npu2 = mx_driving.points_in_box(boxes.npu(), points.npu())
        self.assertRtolEqual(point_indices.numpy(), point_indices_npu2.cpu().numpy())
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `PointsInBox` is only supported on 910B, skip this ut!")
    def test_points_in_box_shape_large_batch(self, device="npu"):
        boxes, points = cpu_gen_inputs([0, 1], [0, 2.0], [2, 200, 7], [2, 1500, 3])
        shape1 = points.shape
        batch_size = shape1[0]
        num_points = shape1[1]
        num_boxes = boxes.shape[1]
        point_indices = points.new_zeros((batch_size, num_points), dtype=torch.int).fill_(-1)
        for b in range(batch_size):
            point_indices[b] = points_in_boxes_cpu_forward(boxes[b].float(),
                                        points[b].float(),
                                        point_indices[b])

        with self.assertRaisesRegex(RuntimeError, "points_in_box npu only support batch size = 1"):
            point_indices_npu = mx_driving.preprocess.npu_points_in_box(boxes.npu(), points.npu())
        with self.assertRaisesRegex(RuntimeError, "points_in_box npu only support batch size = 1"):
            point_indices_npu2 = mx_driving.points_in_box(boxes.npu(), points.npu())


if __name__ == "__main__":
    run_tests()
