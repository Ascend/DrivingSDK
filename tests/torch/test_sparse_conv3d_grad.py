# Copyright (c) 2025, Huawei Technologies.All rights reserved.
import os
from pathlib import Path

import torch
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from data_cache import golden_data_cache
from cv_fused_double_benchmark_compare import CvFusedDoubleBenchmarkAccuracyCompare


class TestSparseConv3dGrad(TestCase):

    def setUp(self):
        self.stride = [[2, 2, 2], [2, 2, 2], [1, 1, 2]]
        self.padding = [[1, 1, 1], [1, 1, 0], [0, 0, 0]]
        self.dilation = [1, 1, 1]
        self.batch_size = [1, 2, 3, 4]

    def random_input_generator(self, num_points, spatial_shape, in_channels, out_channels, kernel_size, dtype, seed=42):

        np.random.seed(seed)
        torch.manual_seed(seed)
        rand_idx = np.random.randint(0, len(self.stride))

        stride = self.stride[rand_idx]
        padding = self.padding[rand_idx]
        dilation = self.dilation
        batch_size = self.batch_size[np.random.randint(0, len(self.batch_size))]

        features = torch.rand(num_points, in_channels, dtype=dtype) * 10 - 5
        weight = torch.rand(*kernel_size, in_channels, out_channels, dtype=dtype) * 10 - 5
        indices = self.generate_sparse_indices(spatial_shape, num_points, batch_size)

        return features, indices, weight, spatial_shape, batch_size, kernel_size, stride, padding, dilation

    @golden_data_cache(__file__)
    def cal_sparse_conv3d_indices(self, indices, spatial_shape, kernel_size, stride, padding):
        indices = indices.cpu()
        k0, k1, k2 = kernel_size
        out_spatial_shape = [(spatial_shape[i] + 2 * padding[i] - kernel_size[i]) // stride[i] + 1 for i in range(3)]

        k0_offset = torch.arange(k0, device=indices.device)
        k1_offset = torch.arange(k1, device=indices.device)
        k2_offset = torch.arange(k2, device=indices.device)

        k_offset = torch.cartesian_prod(k0_offset - k0 // 2, k1_offset - k1 // 2, k2_offset - k2 // 2)
        zeros = torch.zeros((k_offset.shape[0], 1), device=indices.device)
        k_offset = torch.cat([zeros, k_offset], dim=1)

        indices_offset = (k_offset + indices[:, None, :]).double()
        indices_offset[..., 1] = (indices_offset[..., 1] + padding[0] - kernel_size[0] // 2) / stride[0]
        indices_offset[..., 2] = (indices_offset[..., 2] + padding[1] - kernel_size[1] // 2) / stride[1]
        indices_offset[..., 3] = (indices_offset[..., 3] + padding[2] - kernel_size[2] // 2) / stride[2]

        valid_mask1 = (
            (indices_offset[..., 1] >= 0)
            & (indices_offset[..., 1] < out_spatial_shape[0])
            & (indices_offset[..., 2] >= 0)
            & (indices_offset[..., 2] < out_spatial_shape[1])
            & (indices_offset[..., 3] >= 0)
            & (indices_offset[..., 3] < out_spatial_shape[2])
        )
        valid_mask2 = (indices_offset.frac() == 0).all(dim=-1)
        valid_mask = valid_mask1 & valid_mask2

        indices_offset = (
            indices_offset[..., 0] * (out_spatial_shape[0] * out_spatial_shape[1] * out_spatial_shape[2])
            + indices_offset[..., 1] * (out_spatial_shape[1] * out_spatial_shape[2])
            + indices_offset[..., 2] * out_spatial_shape[2]
            + indices_offset[..., 3]
        )

        indices_offset[~valid_mask] = -1
        output_offset = torch.flip(indices_offset, (-1,)).to(torch.int64).flatten()

        to_insert = torch.tensor(-1, device=indices.device)
        sorted_idx, sorted_idx_to_former_indices = torch.sort(output_offset)

        new_sorted_idx = torch.cat((to_insert.view(1), sorted_idx), 0)
        new_sorted_idx_2 = torch.cat((sorted_idx, to_insert.view(1)), 0)

        sub_result = new_sorted_idx - new_sorted_idx_2
        unique_indices_offset = torch.nonzero(sub_result != 0).flatten()
        if len(unique_indices_offset) == 0:
            raise ValueError(
                f"'unique_indices_offset' equal to {unique_indices_offset}, All the input points cannot be convolved to the valid output point."
            )

        return sorted_idx_to_former_indices.int(), unique_indices_offset.int()

    @golden_data_cache(__file__)
    def sparse_conv3d_grad_cpu(self, sorted_indices, indices_offset, features, weight, benchmark="single_benchmark"):
        ori_dtype = features.dtype
        if ori_dtype == torch.float16:
            features = features.float().cpu()
            weight = weight.float().cpu()
        elif ori_dtype == torch.float32:
            if benchmark == "double_benchmark":
                features = features.double().cpu()
                weight = weight.double().cpu()
        else:
            raise TypeError(f"Only support dtype in 'torch.float16', 'torch.float32', but got {features.dtype}")

        in_channels = features.shape[1]
        out_channels = weight.shape[4]
        k0, k1, k2 = weight.shape[0], weight.shape[1], weight.shape[2]
        k_size = k0 * k1 * k2

        out_length = indices_offset.shape[0] - 1
        arange_idx = (
            torch.arange(out_length, device=features.device)
            .repeat_interleave(indices_offset[1:] - indices_offset[:-1])
            .int()
        )

        k_pos = (sorted_indices[indices_offset[0] : indices_offset[-1]] % k_size).int()
        input_idx = (sorted_indices[indices_offset[0] : indices_offset[-1]] / k_size).int()

        img2col_mat = torch.zeros(
            (out_length, k_size, features.shape[-1]), device=features.device, dtype=features.dtype
        )
        img2col_mat[arange_idx, k_pos] = features[input_idx]

        grad = torch.ones((out_length, out_channels), device=features.device, dtype=features.dtype)
        # (out_num, k*k*k*cIn).T @ (out_num, cOut) = (k*k*k*cIn, cOut)
        weight_grad = img2col_mat.reshape(out_length, -1).T @ grad
        weight_grad = weight_grad.reshape(k0, k1, k2, in_channels, out_channels)
        # (out_num, cOut) @ (k*k*k*cIn, cOut).T = (out_num, k*k*k*cIn)
        img2col_feature_grad = grad @ weight.reshape(-1, out_channels).T

        flat_index = arange_idx * k_size + k_pos
        selected_grad = img2col_feature_grad.reshape(-1, in_channels)[flat_index]
        feature_grad = torch.zeros_like(features, device=features.device, dtype=features.dtype)
        target_indices = input_idx.long()
        feature_grad.scatter_add_(dim=0, index=target_indices.unsqueeze(1).expand(-1, in_channels), src=selected_grad)
        return feature_grad.to(ori_dtype), weight_grad.to(ori_dtype)

    @golden_data_cache(__file__)
    def generate_sparse_indices(self, spatial_shape, total_points, batch_size):
        if batch_size == 1:
            num_points = [total_points]
        else:
            num_points = [total_points // batch_size for _ in range(batch_size - 1)]
            num_points.append(total_points - sum(num_points))
        indices = []
        batch_idx = 0
        for num_point in num_points:
            batch_indices = []
            batch_indices.append(np.ones((2 * num_point, 1)) * batch_idx)
            for spatial_size in spatial_shape:
                idx = np.random.uniform(0, spatial_size, (2 * num_point, 1)).astype(np.int32)
                batch_indices.append(idx)

            batch_indices = np.concatenate(batch_indices, axis=1)
            idx_unique = np.unique(batch_indices, axis=0)
            indices.append(idx_unique[:num_point])
            batch_idx += 1

        indices = np.concatenate(indices, axis=0)
        return torch.from_numpy(indices).int()

    def get_golden_output(self, sorted_indices, indices_offset, features, weight, benchmark="single_benchmark"):
        feature_grad, weight_grad = self.sparse_conv3d_grad_cpu(
            sorted_indices, indices_offset, features, weight, benchmark="single_benchmark"
        )
        return feature_grad, weight_grad

    def get_npu_output(self, sorted_indices, indices_offset, features, weight):
        import mx_driving._C

        sorted_indices_npu = sorted_indices.npu()
        indices_offset_npu = indices_offset.npu()
        features_npu = features.npu()
        weight_npu = weight.npu()

        dout = torch.ones(len(indices_offset) - 1, weight_npu.shape[-1], dtype=features.dtype).npu()
        feature_grad, weight_grad = mx_driving._C.npu_sparse_conv3d_grad_v2(
            sorted_indices_npu, indices_offset_npu, features_npu, weight_npu, dout
        )
        return feature_grad, weight_grad

    def get_gpu_output(self, path):
        gpu_out_data = torch.load(path / "output.pt", map_location="cpu")
        return gpu_out_data["feature_grad"], gpu_out_data["weight_grad"]

    def cpu_single_benchmark_compare(self, input_data):

        features, indices, weight, spatial_shape, batch_size, kernel_size, stride, padding, dilation = input_data
        sorted_indices, indices_offset = self.cal_sparse_conv3d_indices(
            indices, spatial_shape, kernel_size, stride, padding
        )

        feature_grad_npu, weight_grad_npu = self.get_npu_output(sorted_indices, indices_offset, features, weight)
        feature_grad_golden, weight_grad_golden = self.get_golden_output(
            sorted_indices, indices_offset, features, weight, benchmark="single_benchmark"
        )

        feature_grad_npu = feature_grad_npu.detach().cpu()
        weight_grad_npu = weight_grad_npu.detach().cpu()

        self.assertRtolEqual(feature_grad_npu, feature_grad_golden, 1e-3)
        self.assertRtolEqual(weight_grad_npu, weight_grad_golden, 1e-3)

    def cv_fused_double_benchmark_compare(self, case_path):

        input_data = torch.load(case_path / "input.pt", map_location="cpu")
        features, indices, weight, spatial_shape, batch_size, kernel_size, stride, padding, dilation = (
            input_data.values()
        )
        sorted_indices, indices_offset = self.cal_sparse_conv3d_indices(
            indices, spatial_shape, kernel_size, stride, padding
        )

        feature_grad_npu, weight_grad_npu = self.get_npu_output(sorted_indices, indices_offset, features, weight)
        feature_grad_gpu, weight_grad_gpu = self.get_gpu_output(case_path)
        feature_grad_golden, weight_grad_golden = self.get_golden_output(
            sorted_indices, indices_offset, features, weight, "double_benchmark"
        )
        
        feature_grad_npu = feature_grad_npu.detach().cpu()
        weight_grad_npu = weight_grad_npu.detach().cpu()
        compare = CvFusedDoubleBenchmarkAccuracyCompare(
            [feature_grad_npu, weight_grad_npu],
            [feature_grad_gpu, weight_grad_gpu],
            [feature_grad_golden, weight_grad_golden],
        )
        ret = compare.run()
        assert "False" not in ret, f"Accuracy check failed for case: {case_path}"

    def case_test_iterator(self, case_dict):

        for i, (case_key, case_value) in enumerate(case_dict.items()):
            num_points, spatial_shape, in_channels, out_channels, kernel_size, dtype = case_value.values()

            gpu_out_path = os.getenv("GPU_OUT_PATH", None)
            case_dir_name = (
                f"{spatial_shape[0]}_{spatial_shape[1]}_{spatial_shape[2]}_{in_channels}_{out_channels}_"
                f"{num_points}_{kernel_size[0]}_{kernel_size[1]}_{kernel_size[2]}_{dtype}"
            )
            case_path = Path(gpu_out_path) / "sparse_conv3d_grad_gpu" / case_dir_name if gpu_out_path else None
            
            double_benchmark_flag = False # (case_path is not None) and os.path.exists(case_path)
            if double_benchmark_flag:
                self.cv_fused_double_benchmark_compare(case_path)
            else:
                input_data = self.random_input_generator(
                    num_points, spatial_shape, in_channels, out_channels, kernel_size, dtype, i
                )
                self.cpu_single_benchmark_compare(input_data)

    def test_bevfusion_cases_fp32(self):
        bevfusion_cases = dict(
            case1=dict(
                num_points=56535,
                spatial_shape=[180, 180, 5],
                in_channels=128,
                out_channels=128,
                kernel_size=[1, 1, 3],
                dtype=torch.float32,
            ),
            case2=dict(
                num_points=256723,
                spatial_shape=[719, 719, 21],
                in_channels=32,
                out_channels=64,
                kernel_size=[3, 3, 3],
                dtype=torch.float32,
            ),
            case3=dict(
                num_points=390558,
                spatial_shape=[1439, 1439, 41],
                in_channels=16,
                out_channels=32,
                kernel_size=[3, 3, 3],
                dtype=torch.float32,
            ),
        )
        self.case_test_iterator(bevfusion_cases)

    def test_bevfusion_cases_fp16(self):
        bevfusion_cases = dict(
            case1=dict(
                num_points=56535,
                spatial_shape=[180, 180, 5],
                in_channels=128,
                out_channels=128,
                kernel_size=[1, 1, 3],
                dtype=torch.float16,
            ),
            case2=dict(
                num_points=390558,
                spatial_shape=[1439, 1439, 41],
                in_channels=16,
                out_channels=32,
                kernel_size=[3, 3, 3],
                dtype=torch.float16,
            ),
        )
        self.case_test_iterator(bevfusion_cases)

    def test_large_channels_cases(self):
        case_list = dict(
            case1=dict(
                num_points=56535,
                spatial_shape=[180, 180, 5],
                in_channels=1024,
                out_channels=1024,
                kernel_size=[1, 1, 3],
                dtype=torch.float32,
            ),
            case2=dict(
                num_points=56535,
                spatial_shape=[180, 180, 5],
                in_channels=1024,
                out_channels=1024,
                kernel_size=[1, 1, 3],
                dtype=torch.float16,
            ),
        )
        self.case_test_iterator(case_list)

    def test_kernel_size_5_cases(self):
        case_list = dict(
            case1=dict(
                num_points=56535,
                spatial_shape=[180, 180, 5],
                in_channels=16,
                out_channels=32,
                kernel_size=[5, 5, 5],
                dtype=torch.float32,
            ),
            case2=dict(
                num_points=56535,
                spatial_shape=[180, 180, 5],
                in_channels=16,
                out_channels=32,
                kernel_size=[5, 5, 5],
                dtype=torch.float16,
            ),
        )
        self.case_test_iterator(case_list)


if __name__ == "__main__":
    run_tests()
