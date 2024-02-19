# Copyright (c) OpenMMLab. All rights reserved.
import unittest
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import ads.common

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestNms3dNormal(TestCase):
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `RotaryMul` is only supported on 910B, skip this ut!")
    def test_nms3d_normal(self):
        # test for 5 boxes
        np_boxes = np.asarray([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0],
                            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0],
                            [3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 0.3],
                            [3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 0.0],
                            [3.0, 3.2, 3.2, 3.0, 2.0, 2.0, 0.3]],
                            dtype=np.float32)
        np_scores = np.array([0.6, 0.9, 0.1, 0.2, 0.15], dtype=np.float32)
        np_inds = np.array([1, 0, 3])
        boxes = torch.from_numpy(np_boxes)
        scores = torch.from_numpy(np_scores)
        inds = ads.common.npu_nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        
        self.assertRtolEqual(inds.cpu().numpy(), np_inds)

        # test for many boxes
        np.random.seed(42)
        np_boxes = np.random.rand(555, 7).astype(np.float32)
        np_scores = np.random.rand(555).astype(np.float32)
        boxes = torch.from_numpy(np_boxes)
        scores = torch.from_numpy(np_scores)
        inds = ads.common.npu_nms3d_normal(boxes.npu(), scores.npu(), 0.3)

        self.assertRtolEqual(len(inds.cpu().numpy()), 148)

if __name__ == "__main__":
    run_tests()
