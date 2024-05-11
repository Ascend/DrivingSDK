import unittest
import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import ads_c


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestGroupPointsGrad(TestCase):
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def golden_group_points_grad(self, np_grad_out, np_indices, np_grad_features, B, npoints, nsample):

        np_grad_out = np_grad_out.transpose(0, 2, 3, 1)
        np_grad_features = np_grad_features.transpose(0, 2, 1)

        for b in range(B):
            for npo in range(npoints):
                for nsa in range(nsample):
                    idx_offset = np_indices[b, npo, nsa]
                    np_grad_features[b, idx_offset, :] += np_grad_out[b, npo, nsa, :]
        
        np_grad_features = np_grad_features.transpose(0, 2, 1)
        return np_grad_features

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `GroupPointsGrad` is only supported on 910B, skip this ut!")
    def test_group_points_grad(self):
        B = 16
        C = 512
        N = 64
        npoints = 16
        nsample = 32

        np_grad_out = np.random.rand(B, C, npoints, nsample).astype(np.float32)
        np_indices = np.random.randint(0, N, (B, npoints, nsample)).astype(np.int32)
        np_grad_features = np.zeros((B, C, N)).astype(np.float32)

        torch_grad_out = torch.from_numpy(np_grad_out).npu()
        torch_indices = torch.from_numpy(np_indices).npu()

        golden_grad_features = self.golden_group_points_grad(
            np_grad_out, np_indices, np_grad_features, B, npoints, nsample)
        npu_grad_features = ads_c.group_points_backward(torch_grad_out, torch_indices, B, C, N, npoints, nsample)

        self.assertRtolEqual(golden_grad_features, npu_grad_features.cpu().numpy())


if __name__ == "__main__":
    run_tests()
