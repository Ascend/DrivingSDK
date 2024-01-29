import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor
import ads.common

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]
reduce_type_mapping = {"mean": 1, "max": 2}


class TestDynamicScatter(TestCase):

    def cpu_op_exec(self, feats, coors, reduce_type):
        clean_coors = coors.masked_fill(coors.lt(0).any(-1, True), -1)
        out_coors, coors_map, reduce_count = clean_coors.unique(dim=0, sorted=True, return_inverse=True,
                                                                return_counts=True)
        out_coors = out_coors[out_coors.min(dim=-1).values >= 0]

        if out_coors[0][0].lt(0):
            out_coors = out_coors.slice(0, 1)
            reduce_count = reduce_count.slice(0, 1)
            coors_map = coors_map - 1

        output_feats = []
        if reduce_type == 1:
            for ref_voxel_coors in out_coors:
                voxel_mask = (coors == ref_voxel_coors).all(dim=-1)
                output_feats.append(feats[voxel_mask].mean(dim=0))
        else:
            for ref_voxel_coors in out_coors:
                voxel_mask = (coors == ref_voxel_coors).all(dim=-1)
                output_feats.append(feats[voxel_mask].max(dim=0).values)
        output_feats = torch.stack(output_feats)
        return output_feats.numpy(), out_coors.numpy()

    def npu_op_exec(self, feats, coors, reduce_type):
        output_feats, output_coors = ads.common.npu_dynamic_scatter(feats, coors, reduce_type)
        return output_feats.cpu().numpy(), output_coors.cpu().numpy()

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `DynamicScatter` is only supported on 910B, skip this ut!")
    def test_dynamic_scatter_max_fp32(self):
        shape_feats = (2000, 3)
        shape_coors = (2000, 3)
        cpu_feats, npu_feats = create_common_tensor(["float32", 2, shape_feats], -50, 50)
        cpu_coors, npu_coors = create_common_tensor(["int32", 2, shape_coors], -1, 20)
        reduce_type = reduce_type_mapping["max"]
        cpu_output = self.cpu_op_exec(cpu_feats, cpu_coors, reduce_type)
        npu_output = self.npu_op_exec(npu_feats, npu_coors, reduce_type)
        self.assertRtolEqual(cpu_output[0], npu_output[0])
        self.assertRtolEqual(cpu_output[1], npu_output[1])

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `DynamicScatter` is only supported on 910B, skip this ut!")
    def test_dynamic_scatter_mean_fp32(self):
        shape_feats = (2000, 3)
        shape_coors = (2000, 3)
        cpu_feats, npu_feats = create_common_tensor(["float32", 2, shape_feats], -50, 50)
        cpu_coors, npu_coors = create_common_tensor(["int32", 2, shape_coors], -1, 20)
        reduce_type = reduce_type_mapping["mean"]
        cpu_output = self.cpu_op_exec(cpu_feats, cpu_coors, reduce_type)
        npu_output = self.npu_op_exec(npu_feats, npu_coors, reduce_type)
        self.assertRtolEqual(cpu_output[0], npu_output[0])
        self.assertRtolEqual(cpu_output[1], npu_output[1])

if __name__ == "__main__":
    run_tests()
