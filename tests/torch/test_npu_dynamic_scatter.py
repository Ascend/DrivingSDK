import unittest

import numpy as np
import torch
import torch_npu
from data_cache import golden_data_cache
from torch_npu.testing.common_utils import create_common_tensor
from torch_npu.testing.testcase import TestCase, run_tests

import mx_driving.point, mx_driving._C
from mx_driving.ops.npu_dynamic_scatter import DynamicScatterFunction


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


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
        if reduce_type == "mean":
            for ref_voxel_coors in out_coors:
                voxel_mask = (coors == ref_voxel_coors).all(dim=-1)
                output_feats.append(feats[voxel_mask].mean(dim=0))
        elif reduce_type == "sum":
            for ref_voxel_coors in out_coors:
                voxel_mask = (coors == ref_voxel_coors).all(dim=-1)
                output_feats.append(feats[voxel_mask].sum(dim=0))
        else:
            for ref_voxel_coors in out_coors:
                voxel_mask = (coors == ref_voxel_coors).all(dim=-1)
                output_feats.append(feats[voxel_mask].max(dim=0).values)
        output_feats = torch.stack(output_feats)
        return output_feats.numpy(), out_coors.numpy()

    def npu_op_exec(self, feats, coors, reduce_type):
        output_feats, output_coors = mx_driving.point.npu_dynamic_scatter(feats, coors, reduce_type)
        output_feats2, output_coors2 = mx_driving.dynamic_scatter(feats, coors, reduce_type)
        return (output_feats.cpu().numpy(), output_coors.cpu().numpy()), (output_feats2.cpu().numpy(), output_coors2.cpu().numpy())

    def grad_npu_op_exec(self, feats, coors, reduce_type):
        class _mockCtx:
            def __init__(self, feats, coors, reduce_type):
                voxel_idx = mx_driving._C.point_to_voxel(coors, [], [], "XYZ")
                num_voxels, _, prefix_sum_point_per_voxel, argsort_coor, _ = mx_driving._C.unique_voxel(voxel_idx)
                _, compare_mask = mx_driving._C.npu_dynamic_scatter(
                    feats, coors, prefix_sum_point_per_voxel, argsort_coor, num_voxels, reduce_type
                )
                self.feats_shape = feats.shape
                self.reduce_type = reduce_type
                self.saved_tensors = (prefix_sum_point_per_voxel, argsort_coor, compare_mask)

        ctx = _mockCtx(feats, coors, reduce_type)
        feats.requires_grad_()
        output_feats, output_coors = mx_driving.dynamic_scatter(feats, coors, reduce_type)
        output_feats, _, _ = DynamicScatterFunction.backward(ctx, torch.ones_like(output_feats), torch.ones_like(output_coors))
        return output_feats.detach().cpu().numpy(), output_coors.detach().cpu().numpy()

    def test_dynamic_scatter_max_fp32(self):
        shape_feats = (2000, 3)
        shape_coors = (2000, 3)
        cpu_feats, npu_feats = create_common_tensor(["float32", 2, shape_feats], -50, 50)
        cpu_coors, npu_coors = create_common_tensor(["int32", 2, shape_coors], -1, 20)
        reduce_type = "max"
        cpu_output = self.cpu_op_exec(cpu_feats, cpu_coors, reduce_type)
        npu_output, npu_output2 = self.npu_op_exec(npu_feats, npu_coors, reduce_type)
        self.assertRtolEqual(cpu_output[0], npu_output[0])
        self.assertRtolEqual(cpu_output[1], npu_output[1])
        self.assertRtolEqual(cpu_output[0], npu_output2[0])
        self.assertRtolEqual(cpu_output[1], npu_output2[1])

    def test_dynamic_scatter_mean_fp32(self):
        shape_feats = (2000, 3)
        shape_coors = (2000, 3)
        cpu_feats, npu_feats = create_common_tensor(["float32", 2, shape_feats], -50, 50)
        cpu_coors, npu_coors = create_common_tensor(["int32", 2, shape_coors], -1, 20)
        reduce_type = "mean"
        cpu_output = self.cpu_op_exec(cpu_feats, cpu_coors, reduce_type)
        npu_output, npu_output2 = self.npu_op_exec(npu_feats, npu_coors, reduce_type)
        self.assertRtolEqual(cpu_output[0], npu_output[0])
        self.assertRtolEqual(cpu_output[1], npu_output[1])
        self.assertRtolEqual(cpu_output[0], npu_output2[0])
        self.assertRtolEqual(cpu_output[1], npu_output2[1])
    
    def test_dynamic_scatter_sum_fp32(self):
        shape_feats = (2000, 3)
        shape_coors = (2000, 3)
        cpu_feats, npu_feats = create_common_tensor(["float32", 2, shape_feats], -50, 50)
        cpu_coors, npu_coors = create_common_tensor(["int32", 2, shape_coors], -1, 20)
        reduce_type = "sum"
        cpu_output = self.cpu_op_exec(cpu_feats, cpu_coors, reduce_type)
        npu_output, npu_output2 = self.npu_op_exec(npu_feats, npu_coors, reduce_type)
        self.assertRtolEqual(cpu_output[0], npu_output[0])
        self.assertRtolEqual(cpu_output[1], npu_output[1])
        self.assertRtolEqual(cpu_output[0], npu_output2[0])
        self.assertRtolEqual(cpu_output[1], npu_output2[1])

    def test_dynamic_scatter_empty_tensor(self):
        try:
            _ = self.npu_op_exec(torch.empty(0), torch.empty(0), 'max')
            assert False, "Expected Exception for empty tensor, but no exception was raised."
        except Exception as e:
            assert "empty tensor" in str(e), f"Expected 'empty tensor' in error message, but got: {e}"

    def test_dynamic_scatter_invalid_reduce(self):
        shape_feats = (2000, 3)
        shape_coors = (2000, 3)
        feats = ((torch.rand(shape_feats, dtype=torch.float32) * 100) - 50).npu()
        coors = torch.randint(-1, 20, shape_coors, dtype=torch.int32).npu()
        reduce_type = 'invalid reduce'

        try:
            _ = self.npu_op_exec(feats, coors, reduce_type)
            assert False, "Expected Exception for invalid reduce, but no exception was raised."
        except Exception as e:
            assert "reduce_type should be" in str(e), f"Expected 'reduce_type should be' in error message, but got: {e}"

    def test_dynamic_scatter_grad_sum_fp32(self):
        shape_feats = (2000, 3)
        shape_coors = (2000, 3)
        feats = ((torch.rand(shape_feats, dtype=torch.float32) * 100) - 50).npu()
        coors = torch.randint(-1, 20, shape_coors, dtype=torch.int32).npu()
        reduce_type = 'sum'
        npu_output = self.grad_npu_op_exec(feats, coors, reduce_type)
        self.assertIsNotNone(npu_output[0])
        self.assertIsNotNone(npu_output[1])

    def test_dynamic_scatter_grad_mean_fp32(self):
        shape_feats = (2000, 3)
        shape_coors = (2000, 3)
        feats = ((torch.rand(shape_feats, dtype=torch.float32) * 100) - 50).npu()
        coors = torch.randint(-1, 20, shape_coors, dtype=torch.int32).npu()
        reduce_type = 'mean'
        npu_output = self.grad_npu_op_exec(feats, coors, reduce_type)
        self.assertIsNotNone(npu_output[0])
        self.assertIsNotNone(npu_output[1])

    def test_dynamic_scatter_grad_max_fp32(self):
        shape_feats = (2000, 3)
        shape_coors = (2000, 3)
        feats = ((torch.rand(shape_feats, dtype=torch.float32) * 100) - 50).npu()
        coors = torch.randint(-1, 20, shape_coors, dtype=torch.int32).npu()
        reduce_type = 'max'
        npu_output = self.grad_npu_op_exec(feats, coors, reduce_type)
        self.assertIsNotNone(npu_output[0])
        self.assertIsNotNone(npu_output[1])

    def test_dynamic_scatter_grad_empty_tensor(self):
        try:
            _, _, _ = DynamicScatterFunction.backward(None, torch.empty(0), torch.empty(0))
            assert False, "Expected Exception for invalid reduce, but no exception was raised."
        except Exception as e:
            assert "empty tensor" in str(e), f"Expected 'empty tensor' in error message, but got: {e}"


if __name__ == "__main__":
    run_tests()
