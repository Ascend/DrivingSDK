
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
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving
from data_cache import golden_data_cache


@golden_data_cache(__file__)
# pylint: disable=huawei-too-many-arguments
def radius_golden_python(x, y, ptr_x, ptr_y, r, max_num_neighbors):
    batch_size = ptr_x.shape[0] - 1
    core_num = 40
    
    batch_size_per_core_head = batch_size // core_num + 1
    batch_size_per_core_tail = batch_size // core_num
    
    head_core_num = batch_size - (core_num * batch_size_per_core_tail) 
    num_points_output = 0
    index_x_list, index_y_list = [], []
    for block_idx in range(core_num):
        if block_idx < head_core_num:
            batch_size_per_core = batch_size_per_core_head
            ptr_offset = block_idx * batch_size_per_core
        else:
            batch_size_per_core = batch_size_per_core_tail
            ptr_offset = block_idx * batch_size_per_core + head_core_num
        ptr_x_this_batch = ptr_x[ptr_offset: ptr_offset + batch_size_per_core + 1]
        ptr_y_this_batch = ptr_y[ptr_offset: ptr_offset + batch_size_per_core + 1]
        
        for batch_idx in range(batch_size_per_core):
            ptr_x_left = ptr_x_this_batch[batch_idx].item()
            ptr_x_right = ptr_x_this_batch[batch_idx + 1].item()
            ptr_y_left = ptr_y_this_batch[batch_idx].item()
            ptr_y_right = ptr_y_this_batch[batch_idx + 1].item()
            points_x = x[ptr_x_left: ptr_x_right, :]
            points_y = y[ptr_y_left: ptr_y_right, :]
            for point_idx, point_y in enumerate(points_y):
                distance = (point_y[0] - points_x[:, 0])**2 + (point_y[1] - points_x[:, 1])**2
                index_x = torch.arange(ptr_x_left, ptr_x_right)
                index_x = index_x[distance < r * r][:max_num_neighbors]
                index_y = torch.ones_like(index_x) * (ptr_y_left + point_idx)
                distance = distance[distance < r * r][:max_num_neighbors]
                num_points_output += distance.shape[0]
                index_x_list.append(index_x)
                index_y_list.append(index_y)
    index_x_total = torch.cat(index_x_list).reshape([1, -1])
    index_y_total = torch.cat(index_y_list).reshape([1, -1])
    results = torch.cat([index_y_total, index_x_total], dim=0)
    return results


@golden_data_cache(__file__)
def gen_points(num_points, data_range):
    points = 2 * data_range * (torch.rand([num_points, 2]) - 0.5)
    return points


@golden_data_cache(__file__)
def gen_batch_ptr(batch_size, max_points_per_batch):
    batch_list = torch.randint(0, max_points_per_batch, [batch_size]).int()
    batch_ptr = torch.cumsum(batch_list, dim=0).int()
    batch_ptr = torch.cat([torch.zeros([1]).int(), batch_ptr])
    return batch_ptr


@golden_data_cache(__file__)
def gen_inputs(data_range, batch_size, max_points_per_batch):
    ptr_x = gen_batch_ptr(batch_size, max_points_per_batch)
    ptr_y = gen_batch_ptr(batch_size, max_points_per_batch)
    num_points_x = ptr_x[-1]
    num_points_y = ptr_y[-1]
    x = gen_points(num_points_x, data_range)
    y = gen_points(num_points_y, data_range)
    return x, y, ptr_x, ptr_y


class TestRadius(TestCase):
    def test_radius(self):
        #test for max range
        #test for all point distances smaller than r
        #test for all point distances larger than r
        #test for the least value of max_num_neighbors
        #test for the least value of r and max_num_neighbors
        #test for batch_size equal to core_num
        para_lists = [[100, 1024, 512, 50, 300],
                      [10, 1024, 512, 50, 300],
                      [100, 100, 256, 0, 30],
                      [10, 100, 64, 50, 1],
                      [10, 10, 20, 0, 1],
                      [100, 40, 10, 50, 300]]
        
        #iterate through many values of batch_size
        for x in range(1, 1025, 31):
            para_lists.append([100, x, 50, 50, 300])
            
        #iterate through many values of max_points_per_batch
        for x in range(2, 513, 30):
            para_lists.append([100, 50, x, 50, 300])
        
        for para_list in para_lists:
            data_range, batch_size, max_points_per_batch, r, max_num_neighbors = para_list
            x, y, ptr_x, ptr_y = gen_inputs(data_range, batch_size, max_points_per_batch)
            out_cpu = radius_golden_python(x, y, ptr_x, ptr_y, r, max_num_neighbors).int()
            out_npu = mx_driving.radius(x.npu(), y.npu(), ptr_x.npu(), ptr_y.npu(), r, max_num_neighbors)
            self.assertRtolEqual(out_cpu, out_npu) 
    
if __name__ == "__main__":
    run_tests()