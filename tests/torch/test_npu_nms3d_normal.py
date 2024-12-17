# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving
import mx_driving.detection


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestNms3dNormal(TestCase):
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `Nms3d_Normal` is only supported on 910B, skip this ut!")
    def test_nms3d_normal_for_5_boxes(self):
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
        inds_1 = mx_driving.nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(inds_1.cpu().numpy(), np_inds)
        inds_2 = mx_driving.detection.nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(inds_2.cpu().numpy(), np_inds)
        inds_3 = mx_driving.detection.npu_nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(inds_3.cpu().numpy(), np_inds)
    
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `Nms3d_Normal` is only supported on 910B, skip this ut!")
    def test_nms3d_normal_for_15_boxes(self):
        # test for many boxes
        # 10
        np.random.seed(15)
        np_boxes = np.random.rand(10, 7).astype(np.float32)
        np_scores = np.random.rand(10).astype(np.float32)
        boxes = torch.from_numpy(np_boxes)
        scores = torch.from_numpy(np_scores)
        inds_1 = mx_driving.nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(len(inds_1.cpu().numpy()), 9)
        inds_2 = mx_driving.detection.nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(len(inds_2.cpu().numpy()), 9)
        inds_3 = mx_driving.detection.npu_nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(len(inds_3.cpu().numpy()), 9)

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `Nms3d_Normal` is only supported on 910B, skip this ut!")
    def test_nms3d_normal_for_200_boxes(self):
        # 200
        np.random.seed(29)
        np_boxes = np.random.rand(200, 7).astype(np.float32)
        np_scores = np.random.rand(200).astype(np.float32)
        boxes = torch.from_numpy(np_boxes)
        scores = torch.from_numpy(np_scores)
        inds_1 = mx_driving.nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(len(inds_1.cpu().numpy()), 79)
        inds_2 = mx_driving.detection.nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(len(inds_2.cpu().numpy()), 79)
        inds_3 = mx_driving.detection.npu_nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(len(inds_3.cpu().numpy()), 79)

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `Nms3d_Normal` is only supported on 910B, skip this ut!")
    def test_nms3d_normal_for_369_boxes(self):
        # 369
        np.random.seed(8)
        np_boxes = np.random.rand(369, 7).astype(np.float32)
        np_scores = np.random.rand(369).astype(np.float32)
        boxes = torch.from_numpy(np_boxes)
        scores = torch.from_numpy(np_scores)
        inds_1 = mx_driving.nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(len(inds_1.cpu().numpy()), 109)
        inds_2 = mx_driving.detection.nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(len(inds_2.cpu().numpy()), 109)
        inds_3 = mx_driving.detection.npu_nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(len(inds_3.cpu().numpy()), 109)

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `Nms3d_Normal` is only supported on 910B, skip this ut!")
    def test_nms3d_normal_for_555_boxes(self):
        # 555
        np.random.seed(42)
        np_boxes = np.random.rand(555, 7).astype(np.float32)
        np_scores = np.random.rand(555).astype(np.float32)
        boxes = torch.from_numpy(np_boxes)
        scores = torch.from_numpy(np_scores)
        inds_1 = mx_driving.nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(len(inds_1.cpu().numpy()), 148)
        inds_2 = mx_driving.detection.nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(len(inds_2.cpu().numpy()), 148)
        inds_3 = mx_driving.detection.npu_nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(len(inds_3.cpu().numpy()), 148)

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `Nms3d_Normal` is only supported on 910B, skip this ut!")
    def test_nms3d_normal_for_300_boxes(self):
        # 300
        np.random.seed(16)
        np_boxes = np.random.rand(300, 7).astype(np.float32)
        np_scores = np.random.rand(300).astype(np.float32)
        boxes = torch.from_numpy(np_boxes)
        scores = torch.from_numpy(np_scores)
        inds_1 = mx_driving.nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(len(inds_1.cpu().numpy()), 102)
        inds_2 = mx_driving.detection.nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(len(inds_2.cpu().numpy()), 102)
        inds_3 = mx_driving.detection.npu_nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(len(inds_3.cpu().numpy()), 102)

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `Nms3d_Normal` is only supported on 910B, skip this ut!")
    def test_nms3d_normal_for_600_boxes(self):
        # 600
        np.random.seed(31)
        np_boxes = np.random.rand(600, 7).astype(np.float32)
        np_scores = np.random.rand(600).astype(np.float32)
        boxes = torch.from_numpy(np_boxes)
        scores = torch.from_numpy(np_scores)
        inds_1 = mx_driving.nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(len(inds_1.cpu().numpy()), 161)
        inds_2 = mx_driving.detection.nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(len(inds_2.cpu().numpy()), 161)
        inds_3 = mx_driving.detection.npu_nms3d_normal(boxes.npu(), scores.npu(), 0.3)
        self.assertRtolEqual(len(inds_3.cpu().numpy()), 161)

if __name__ == "__main__":
    run_tests()

