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
import torch
import numpy as np
import torch_npu

from torch_npu.testing.testcase import TestCase, run_tests
import ads.common


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestPointsInBox(TestCase):
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `PointsInBox` is only supported on 910B, skip this ut!")
    def test_points_in_box_shape_format_fp16(self, device="npu"):
        boxes = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 20.0, 1.0, 0.523598]]],
                             dtype=torch.float32).npu()  # 30 degrees
        pts = torch.tensor(
            [[[4, 6.928, 0], [6.928, 4, 0], [4, -6.928, 0], [6.928, -4, 0],
            [-4, 6.928, 0], [-6.928, 4, 0], [-4, -6.928, 0], [-6.928, -4, 0]]],
            dtype=torch.float32).npu()
        point_indices = ads.common.npu_points_in_box(points=pts, boxes=boxes).cpu().numpy()
        expected_point_indices = torch.tensor([[-1, -1, 0, -1, 0, -1, -1, -1]],
                                            dtype=torch.int32).cpu().numpy()
        self.assertRtolEqual(point_indices, expected_point_indices)


if __name__ == "__main__":
    run_tests()
