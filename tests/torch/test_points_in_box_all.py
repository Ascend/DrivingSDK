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


@golden_data_cache(__file__)
def lidar_to_local_coords_cpu(shift_x, shift_y, rz):
    cosa = torch.cos(-rz) 
    sina = torch.sin(-rz)
    
    local_x = shift_x * cosa + shift_y * (-sina)
    local_y = shift_x * sina + shift_y * cosa
    return local_x, local_y


@golden_data_cache(__file__)
def points_in_boxes_all_cpu_forward(boxes, pts):
    cx, cy, cz, x_size, y_size, z_size, rz = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5], boxes[:, 6]
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    
    cz = cz + z_size / 2.0
    
    cx = cx.unsqueeze(0)
    cy = cy.unsqueeze(0)
    cz = cz.unsqueeze(0)
    x_size = x_size.unsqueeze(0)
    y_size = y_size.unsqueeze(0)
    z_size = z_size.unsqueeze(0)
    rz = rz.unsqueeze(0)
    
    shift_x = x.unsqueeze(1) - cx
    shift_y = y.unsqueeze(1) - cy
    shift_z = z.unsqueeze(1) - cz
    
    z_flag = (shift_z.abs() <= z_size / 2.0)
    local_x, local_y = lidar_to_local_coords_cpu(shift_x, shift_y, rz)
    x_flag = (local_x > -x_size / 2.0) & (local_x < x_size / 2.0)
    y_flag = (local_y > -y_size / 2.0) & (local_y < y_size / 2.0)

    in_flag = z_flag & x_flag & y_flag
    return in_flag.int()


class TestPointsInBoxAll(TestCase):
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `PointsInBoxAll` is only supported on 910B, skip this ut!")
    def test_points_in_box_shape_randn(self, device="npu"):
        boxes, points = cpu_gen_inputs([0, 1], [0, 2.0], [1, 100, 7], [1, 100, 3])
        shape1 = points.shape
        batch_size = shape1[0]
        num_points = shape1[1]
        num_boxes = boxes.shape[1]
        point_indices = points.new_zeros((batch_size, num_points, num_boxes), dtype=torch.int).fill_(0)
        for b in range(batch_size):
            point_indices[b] = points_in_boxes_all_cpu_forward(boxes[b].float(),
                                        points[b].float())
        point_indices_npu = mx_driving.preprocess.npu_points_in_box_all(boxes.npu(), points.npu())
        self.assertRtolEqual(point_indices.numpy(), point_indices_npu.cpu().numpy())
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `PointsInBoxAll` is only supported on 910B, skip this ut!")
    def test_points_in_box_shape_large_boxes(self, device="npu"):
        boxes, points = cpu_gen_inputs([0, 1], [0, 2.0], [1, 10000, 7], [1, 100, 3])
        shape1 = points.shape
        batch_size = shape1[0]
        num_points = shape1[1]
        num_boxes = boxes.shape[1]
        point_indices = points.new_zeros((batch_size, num_points, num_boxes), dtype=torch.int).fill_(0)
        for b in range(batch_size):
            point_indices[b] = points_in_boxes_all_cpu_forward(boxes[b].float(),
                                        points[b].float())
        point_indices_npu = mx_driving.preprocess.npu_points_in_box_all(boxes.npu(), points.npu())
        self.assertRtolEqual(point_indices.numpy(), point_indices_npu.cpu().numpy())
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `PointsInBoxAll` is only supported on 910B, skip this ut!")
    def test_points_in_box_shape_large_points(self, device="npu"):
        boxes, points = cpu_gen_inputs([0, 1], [0, 2.0], [1, 100, 7], [1, 10000, 3])
        shape1 = points.shape
        batch_size = shape1[0]
        num_points = shape1[1]
        num_boxes = boxes.shape[1]
        point_indices = points.new_zeros((batch_size, num_points, num_boxes), dtype=torch.int).fill_(0)
        for b in range(batch_size):
            point_indices[b] = points_in_boxes_all_cpu_forward(boxes[b].float(),
                                        points[b].float())
        point_indices_npu = mx_driving.preprocess.npu_points_in_box_all(boxes.npu(), points.npu())
        self.assertRtolEqual(point_indices.numpy(), point_indices_npu.cpu().numpy())
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `PointsInBoxAll` is only supported on 910B, skip this ut!")
    def test_points_in_box_shape_large_boxes_points(self, device="npu"):
        boxes, points = cpu_gen_inputs([0, 1], [0, 2.0], [1, 10000, 7], [1, 10000, 3])
        shape1 = points.shape
        batch_size = shape1[0]
        num_points = shape1[1]
        num_boxes = boxes.shape[1]
        point_indices = points.new_zeros((batch_size, num_points, num_boxes), dtype=torch.int).fill_(0)
        for b in range(batch_size):
            point_indices[b] = points_in_boxes_all_cpu_forward(boxes[b].float(),
                                        points[b].float())
        point_indices_npu = mx_driving.preprocess.npu_points_in_box_all(boxes.npu(), points.npu())
        self.assertRtolEqual(point_indices.numpy(), point_indices_npu.cpu().numpy())
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `PointsInBoxAll` is only supported on 910B, skip this ut!")
    def test_points_in_box_shape_large_batch(self, device="npu"):
        boxes, points = cpu_gen_inputs([0, 1], [0, 2.0], [100, 100, 7], [100, 100, 3])
        shape1 = points.shape
        batch_size = shape1[0]
        num_points = shape1[1]
        num_boxes = boxes.shape[1]
        point_indices = points.new_zeros((batch_size, num_points, num_boxes), dtype=torch.int).fill_(0)
        for b in range(batch_size):
            point_indices[b] = points_in_boxes_all_cpu_forward(boxes[b].float(),
                                        points[b].float())
        point_indices_npu = mx_driving.preprocess.npu_points_in_box_all(boxes.npu(), points.npu())
        self.assertRtolEqual(point_indices.numpy(), point_indices_npu.cpu().numpy())

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `PointsInBoxAll` is only supported on 910B, skip this ut!")
    def test_points_in_box_shape_random_shape(self, device="npu"):
        boxes, points = cpu_gen_inputs([-5, 5], [-5, 5], [214, 192, 7], [214, 371, 3])
        shape1 = points.shape
        batch_size = shape1[0]
        num_points = shape1[1]
        num_boxes = boxes.shape[1]
        point_indices = points.new_zeros((batch_size, num_points, num_boxes), dtype=torch.int).fill_(0)
        for b in range(batch_size):
            point_indices[b] = points_in_boxes_all_cpu_forward(boxes[b].float(),
                                                               points[b].float())
        point_indices_npu = mx_driving.preprocess.npu_points_in_box_all(boxes.npu(), points.npu())
        self.assertRtolEqual(point_indices.numpy(), point_indices_npu.cpu().numpy())
if __name__ == "__main__":
    run_tests()