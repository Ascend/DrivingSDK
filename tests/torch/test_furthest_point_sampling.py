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
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving
import mx_driving.point


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class CreateBenchMarkTest(ABC):
    def __init__(self):
        self.batch = None
        self.N = None
        self.numPoints = None

        self.point = None
        self.nearestDist = None

    @abstractmethod
    def createData(self):
        pass

    def compare_min(self, a):
        if a[0] > a[1]:
            return a[1]
        else :
            return a[0]

    @golden_data_cache(__file__)
    def getCpuRes(self):
        cpuRes = np.zeros([self.batch, self.numPoints], dtype=np.int32)
        nearestDistCopy = self.nearestDist.copy()

        for i in range(self.batch):
            sampled = 1
            index = 0
            while sampled < self.numPoints:
                deltaX = self.point[i][0] - self.point[i][0][index]
                deltaY = self.point[i][1] - self.point[i][1][index]
                deltaZ = self.point[i][2] - self.point[i][2][index]
                deltaX2 = deltaX * deltaX
                deltaY2 = deltaY * deltaY
                deltaZ2 = deltaZ * deltaZ
                currentDist = deltaX2 + deltaY2 + deltaZ2

                nearestDistCopy[i] = np.apply_along_axis(self.compare_min, 0, np.stack((currentDist, nearestDistCopy[i]), axis=0))
                index = np.argmax(nearestDistCopy[i])
                cpuRes[i][sampled] = index
                sampled = sampled + 1
        return cpuRes


class Test1(CreateBenchMarkTest):
    def createData(self):
        self.batch = 47
        self.N = 717
        self.numPoints = 580

        self.point = np.zeros([self.batch, 3, self.N], dtype=np.float32)
        for i in range(self.batch):
            for j in range(self.N):
                self.point[i, 0, j] = j

        self.nearestDist = np.ones([self.batch, self.N], dtype=np.float32) * 1e10
        self.point = torch.from_numpy(self.point)


class Test2(CreateBenchMarkTest):
    def createData(self):
        self.batch = 193
        self.N = 579
        self.numPoints = 123

        self.point = np.zeros([self.batch, 3, self.N], dtype=np.float32)
        for i in range(self.batch):
            for j in range(self.N):
                self.point[i, 0, j] = j
                self.point[i, 1, j] = j + 1
                self.point[i, 2, j] = j + 3

        self.nearestDist = np.ones([self.batch, self.N], dtype=np.float32) * 1e10
        self.point = torch.from_numpy(self.point)


class Test3(CreateBenchMarkTest):
    def createData(self):
        self.batch = 21
        self.N = 3901
        self.numPoints = 671

        self.point = np.zeros([self.batch, 3, self.N], dtype=np.float32)
        for i in range(self.batch):
            for j in range(self.N):
                self.point[i, 0, j] = j

        self.nearestDist = np.ones([self.batch, self.N], dtype=np.float32) * 1e10
        self.point = torch.from_numpy(self.point)


class Test4(CreateBenchMarkTest):
    def createData(self):
        self.batch = 151
        self.N = 3901
        self.numPoints = 671

        self.point = np.zeros([self.batch, 3, self.N], dtype=np.float32)
        for i in range(self.batch):
            for j in range(self.N):
                self.point[i, 0, j] = j
                self.point[i, 1, j] = j + 1
                self.point[i, 2, j] = j + 3

        self.nearestDist = np.ones([self.batch, self.N], dtype=np.float32) * 1e10
        self.point = torch.from_numpy(self.point)


test1 = Test1()
test2 = Test2()
test3 = Test3()
test4 = Test4()


class TestFurthestPointSample(TestCase):
    def cpu_op_exec(self, myTest):
        return myTest.getCpuRes()

    def npu_op_exec(self, myTest):
        res1 = mx_driving.point.furthest_point_sampling(myTest.point.clone().permute(0, 2, 1).npu(), myTest.numPoints)
        res2 = mx_driving.point.npu_furthest_point_sampling(myTest.point.clone().permute(0, 2, 1).npu(), myTest.numPoints)
        res3 = mx_driving.furthest_point_sampling(myTest.point.clone().permute(0, 2, 1).npu(), myTest.numPoints)
        return res1, res2, res3

    def compare_res(self, myTest):
        myTest.createData()
        cpuOutput = torch.from_numpy(self.cpu_op_exec(myTest))
        npuOutput1, npuOutput2, npuOutput3 = self.npu_op_exec(myTest)
        self.assertRtolEqual(cpuOutput, npuOutput1)
        self.assertRtolEqual(cpuOutput, npuOutput2)
        self.assertRtolEqual(cpuOutput, npuOutput3)

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `FurthestPointSampling` is only for 910B, skip it.")
    def test_FurthestPointSample(self):
        self.compare_res(test1)
        self.compare_res(test2)
        self.compare_res(test3)


if __name__ == "__main__":
    run_tests()
