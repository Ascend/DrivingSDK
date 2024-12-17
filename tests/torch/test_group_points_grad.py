import unittest

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving._C


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


@golden_data_cache(__file__)
def cpu_gen_inputs(B, C, N, npoints, nsample):
    np_grad_out = np.random.rand(B, C, npoints, nsample).astype(np.float32)
    np_indices = np.random.randint(0, N, (B, npoints, nsample)).astype(np.int32)
    np_grad_features = np.zeros((B, C, N)).astype(np.float32)

    return np_grad_out, np_indices, np_grad_features


class TestGroupPointsGrad(TestCase):
    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    @golden_data_cache(__file__)
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
        np.random.seed(50051)
        B_list = [16, 32, 64]
        C_list = [16, 31, 32, 35, 64, 512]
        N_list = [64]
        npoints_list = [16, 100]
        nsample_list = [32, 50]

        for B in B_list:
            for C in C_list:
                for N in N_list:
                    for npoints in npoints_list:
                        for nsample in nsample_list:
                            np_grad_out, np_indices, np_grad_features = cpu_gen_inputs(B, C, N, npoints, nsample)

                            torch_grad_out = torch.from_numpy(np_grad_out).npu()
                            torch_indices = torch.from_numpy(np_indices).npu()

                            golden_grad_features = self.golden_group_points_grad(
                                np_grad_out, np_indices, np_grad_features, B, npoints, nsample)
                            npu_grad_features = mx_driving._C.group_points_backward(torch_grad_out, torch_indices, B, C, N,
                                                                            npoints, nsample)

                            self.assertRtolEqual(golden_grad_features, npu_grad_features.cpu().numpy())


if __name__ == "__main__":
    run_tests()
