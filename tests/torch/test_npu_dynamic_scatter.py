import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

import ads_c
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

    def grad_cpu_op_exec(self, input_tensors, reduce_type):
        grad_point_feats, grad_voxel_feats, prefix_sum_point_per_voxel, argsort_coor, compare_mask = input_tensors
        total_point_num, feats_dim = grad_point_feats.shape
        total_voxel_num, _ = grad_voxel_feats.shape
        zero_tensor = torch.zeros_like(grad_voxel_feats[0])
        for voxel_idx in range(total_voxel_num):
            if voxel_idx == 0:
                start_point = 0
                point_num = prefix_sum_point_per_voxel[0]
            elif voxel_idx == total_voxel_num - 1:
                start_point = prefix_sum_point_per_voxel[voxel_idx - 1]
                point_num = total_point_num - start_point
            else:
                start_point = prefix_sum_point_per_voxel[voxel_idx - 1]
                point_num = prefix_sum_point_per_voxel[voxel_idx] - start_point
            point_idx_list = argsort_coor[start_point: start_point + point_num]
            for point_idx in point_idx_list:
                if reduce_type == "sum":
                    grad_point_feats[point_idx, :] = grad_voxel_feats[voxel_idx, :]
                elif reduce_type == "mean":
                    if point_num > 0:
                        grad_point_feats[point_idx, :] = grad_voxel_feats[voxel_idx, :] / point_num
                    else:
                        raise ValueError
                elif reduce_type == "max":
                    mask_t = compare_mask[point_idx, :]
                    mask_list = [''.join(list(reversed(bin(mask_t[i])[2:].zfill(8)))) for i in range(len(mask_t))]
                    mask_bit_list = list(''.join(mask_list))[:feats_dim]
                    mask_bit = torch.tensor([int(i) for i in mask_bit_list], dtype=torch.uint8)
                    grad_point_feats[point_idx, :] = torch.where(mask_bit, grad_voxel_feats[voxel_idx, :], zero_tensor)

    def npu_op_exec(self, feats, coors, reduce_type):
        output_feats, output_coors = ads.common.npu_dynamic_scatter(feats, coors, reduce_type)
        return output_feats.cpu().numpy(), output_coors.cpu().numpy()

    def grad_npu_op_exec(self, feats, coors, reduce_type):
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

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `DynamicScatterGrad` is only supported on 910B, skip this ut!")
    def test_dynamic_scatter_grad_sum_fp32(self):
        point_num, voxel_num, feats_dim = 1000, 500, 2048
        reduce_type = "sum"
        feats_shape = (point_num, feats_dim)
        voxel_shape = (voxel_num, feats_dim)
        mask_dim = (feats_dim + 8 - 1) // 8
        mask_shape = (point_num, mask_dim)

        golden_result = torch.zeros(feats_shape, dtype=torch.float)
        grad_point_feats = torch.zeros(feats_shape, dtype=torch.float).npu()
        grad_voxel_feats = torch.rand(voxel_shape, dtype=torch.float).npu()
        prefix_sum_point_per_voxel = torch.tensor([i * 2 for i in range(1, voxel_num)], dtype=torch.int32).npu()
        argsort_coor = torch.tensor(list(range(0, point_num)), dtype=torch.int32).npu()
        compare_mask = torch.randint(0, 255, mask_shape).to(torch.uint8).npu()

        self.grad_cpu_op_exec([golden_result, grad_voxel_feats.contiguous().cpu(), prefix_sum_point_per_voxel.cpu(),
                              argsort_coor.cpu(), compare_mask.cpu()], reduce_type)
        ads_c.npu_dynamic_scatter_grad(grad_point_feats, grad_voxel_feats.contiguous(),
                                       prefix_sum_point_per_voxel, argsort_coor, compare_mask, reduce_type)
        self.assertRtolEqual(golden_result.cpu().numpy(), grad_point_feats.cpu().numpy())

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `DynamicScatterGrad` is only supported on 910B, skip this ut!")
    def test_dynamic_scatter_grad_mean_fp32(self):
        point_num, voxel_num, feats_dim = 1000, 500, 2048
        reduce_type = "mean"
        feats_shape = (point_num, feats_dim)
        voxel_shape = (voxel_num, feats_dim)
        mask_dim = (feats_dim + 8 - 1) // 8
        mask_shape = (point_num, mask_dim)

        golden_result = torch.zeros(feats_shape, dtype=torch.float)
        grad_point_feats = torch.zeros(feats_shape, dtype=torch.float).npu()
        grad_voxel_feats = torch.rand(voxel_shape, dtype=torch.float).npu()
        prefix_sum_point_per_voxel = torch.tensor([i * 2 for i in range(1, voxel_num)], dtype=torch.int32).npu()
        argsort_coor = torch.tensor(list(range(0, point_num)), dtype=torch.int32).npu()
        compare_mask = torch.randint(0, 255, mask_shape).to(torch.uint8).npu()

        self.grad_cpu_op_exec([golden_result, grad_voxel_feats.contiguous().cpu(), prefix_sum_point_per_voxel.cpu(),
                              argsort_coor.cpu(), compare_mask.cpu()], reduce_type)
        ads_c.npu_dynamic_scatter_grad(grad_point_feats, grad_voxel_feats.contiguous(),
                                       prefix_sum_point_per_voxel, argsort_coor, compare_mask, reduce_type)
        self.assertRtolEqual(golden_result.cpu().numpy(), grad_point_feats.cpu().numpy())

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `DynamicScatterGrad` is only supported on 910B, skip this ut!")
    def test_dynamic_scatter_grad_max_fp32(self):
        point_num, voxel_num, feats_dim = 1000, 500, 2048
        reduce_type = "max"
        feats_shape = (point_num, feats_dim)
        voxel_shape = (voxel_num, feats_dim)
        mask_dim = (feats_dim + 8 - 1) // 8
        mask_shape = (point_num, mask_dim)

        golden_result = torch.zeros(feats_shape, dtype=torch.float)
        grad_point_feats = torch.zeros(feats_shape, dtype=torch.float).npu()
        grad_voxel_feats = torch.rand(voxel_shape, dtype=torch.float).npu()
        prefix_sum_point_per_voxel = torch.tensor([i * 2 for i in range(1, voxel_num)], dtype=torch.int32).npu()
        argsort_coor = torch.tensor(list(range(0, point_num)), dtype=torch.int32).npu()
        compare_mask = torch.randint(0, 255, mask_shape).to(torch.uint8).npu()

        self.grad_cpu_op_exec([golden_result, grad_voxel_feats.contiguous().cpu(), prefix_sum_point_per_voxel.cpu(),
                              argsort_coor.cpu(), compare_mask.cpu()], reduce_type)
        ads_c.npu_dynamic_scatter_grad(grad_point_feats, grad_voxel_feats.contiguous(),
                                       prefix_sum_point_per_voxel, argsort_coor, compare_mask, reduce_type)
        self.assertRtolEqual(golden_result.cpu().numpy(), grad_point_feats.cpu().numpy())


if __name__ == "__main__":
    run_tests()
