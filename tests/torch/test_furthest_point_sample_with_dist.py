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

import mx_driving
import mx_driving.point


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestFurthestPointSampleWithDist(TestCase):
    @golden_data_cache(__file__)
    def create_input_data(self, shape):
        b, n = shape
        point_xyz = np.random.uniform(0, 10, [b, n, 3]).astype(np.float32)
        point_dist = np.zeros([b, n, n]).astype(np.float32)
        for batch_id in range(b):
            for src_id in range(n):
                x1, y1, z1 = point_xyz[batch_id, src_id]
                for dst_id in range(n):
                    x2, y2, z2 = point_xyz[batch_id, dst_id]
                    point_dist[batch_id, src_id, dst_id] = point_dist[batch_id, src_id, dst_id] =\
                    (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2)
        return point_dist

    def compare_min(self, a, b):
        if a > b:
            return b
        else:
            return a

    @golden_data_cache(__file__)
    def supported_op_exec(self, point_dist, point_num):
        b, n, _ = point_dist.shape
        tmp = np.zeros([b, n]).astype(np.float32)
        result_cpu = np.zeros([b, point_num]).astype(np.int32)
        for batch in range(b):
            for i in range(n):
                tmp[batch, i] = point_dist[batch, 0, i]
            for idx in range(1, point_num):
                best = 0
                best_i = 0
                last_time_idx = result_cpu[batch, idx - 1]
                for i in range(n):
                    tmp[batch, i] = self.compare_min(point_dist[batch, last_time_idx, i], tmp[batch, i])
                    if(best < tmp[batch, i]):
                        best = tmp[batch, i]
                        best_i = i
                result_cpu[batch, idx] = best_i
        return result_cpu

    def custom_op_exec(self, point_dist, point_num, input_dtype):
        point_dist_npu = torch.tensor(point_dist, dtype=input_dtype).npu()
        output = mx_driving.furthest_point_sample_with_dist(point_dist_npu, point_num)

        output_verify = mx_driving.point.furthest_point_sample_with_dist(point_dist_npu, point_num)
        self.assertRtolEqual(output, output_verify)

        return output.cpu().numpy()

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `FurthestPointSampleWithDist` is only supported on 910B, skip this ut!")
    def test_FurthestPointSampleWithDist(self):
        shape_list = [[4, 100], [30, 1000], [3, 2567], [454, 6]]
        point_num_list = [32, 1000, 1400, 3]
        dtype_list = [torch.float32, torch.float32, torch.float32, torch.float32]
        for idx in range(2):
            shape = shape_list[idx]
            point_num = point_num_list[idx]
            input_dtype = dtype_list[idx]
            point_dist = self.create_input_data(shape)

            exoutput = self.supported_op_exec(point_dist, point_num)

            output = self.custom_op_exec(point_dist, point_num, input_dtype)

            self.assertRtolEqual(exoutput, output)

if __name__ == "__main__":
    run_tests()