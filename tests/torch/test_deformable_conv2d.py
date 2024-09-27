import unittest
import torch
import numpy as np
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import mx_driving.fused


class TestDeformableConv2d(TestCase):
    def create_golden_offset(self, offset):
        B, N3, H, W = offset.shape
        N = N3 // 3
        even_idx = np.arange(N3) % 2 == 0
        odd_idx = np.arange(N3) % 2 == 1

        even_idx[(2 * N) :] = 0
        odd_idx[(2 * N) :] = 0

        even_elements = offset[:, even_idx, :, :]
        odd_elements = offset[:, odd_idx, :, :]
        mask_elements = offset[:, (2 * N) :, :, :]
        out_offset = np.concatenate((odd_elements, even_elements, mask_elements), axis=1)
        return out_offset

    def create_single_npu_tensor(self, item, minvalue, maxvalue, rsv=False):
        dtype = item[0]
        format1 = item[1]
        shape = item[2]
        input1 = np.random.uniform(minvalue, maxvalue, shape).astype(dtype)
        if rsv:
            input2 = self.create_golden_offset(input1)
        npu_input = torch.from_numpy(input1).to("npu")
        if format1 != -1:
            npu_input = torch_npu._npu_format_cast(npu_input, format1)
            if rsv:
                npu_input2 = torch.from_numpy(input2).to("npu")
                npu_input2 = torch_npu._npu_format_cast(npu_input2, format1)
                return npu_input, npu_input2
        return npu_input

    def helper_gen(self, offsets_shape, kernel_sizes, strides, pads, dialation):
        H_OUT = offsets_shape[1]
        W_OUT = offsets_shape[2]
        K_H, K_W = kernel_sizes
        STRIDED_H, STRIDED_W = strides[1], strides[2]
        dialation_h, dialation_w = dialation[1], dialation[2]
        try:
            group = offsets_shape[3] / 3 / kernel_sizes[0] / kernel_sizes[1]
        except ZeroDivisionError as e:
            print("kernel_sizes can not be 0.")
        group = int(group)

        pad_top, pad_left = pads[0], pads[2]
        helper_tensor = np.zeros((H_OUT, W_OUT, 3 * group * K_H * K_W), np.float32)
        for h in range(H_OUT):
            for w in range(W_OUT):
                for k_h in range(K_H):
                    for k_w in range(K_W):
                        for g in range(group):
                            helper_tensor[h][w][0 * group * K_H * K_W + g * K_H * K_W + k_h * K_W + k_w] = (
                                w * STRIDED_W - pad_left + k_w * dialation_w
                            )
                            helper_tensor[h][w][1 * group * K_H * K_W + g * K_H * K_W + k_h * K_W + k_w] = (
                                h * STRIDED_H - pad_top + k_h * dialation_h
                            )

        return helper_tensor

    def deformable_offsets(self, x, offsets, args):
        kernel_size, strides, pads, dilations = args
        dtype = x.dtype
        if dtype == np.float16:
            x = x.astype(np.float32)
            offsets = offsets.astype(np.float32)
        N, H_OUT, W_OUT, _ = offsets.shape
        H_IN = x.shape[1]
        W_IN = x.shape[2]
        C = x.shape[-1]
        K_H, K_W = kernel_size
        GROUP = offsets.shape[-1] // K_H // K_W // 3
        GROUP_C = C // GROUP
        helper = self.helper_gen(offsets.shape, kernel_size, strides, pads, dilations)

        x = x.reshape((N, H_IN, W_IN, GROUP, GROUP_C))
        offsets = offsets.reshape((N, H_OUT, W_OUT, 3, GROUP, K_H, K_W))
        helper = helper.reshape((H_OUT, W_OUT, 3, GROUP, K_H, K_W))
        index_offsets = offsets + helper
        floor_index = np.floor(index_offsets)
        ceil_index = floor_index + 1
        int32_ceil_index = ceil_index.astype(np.int32)
        int32_floor_index = floor_index.astype(np.int32)
        l_t_tensor = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)
        r_t_tensor = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)
        l_b_tensor = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)
        r_b_tensor = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)
        for n in range(N):
            for h_out in range(H_OUT):
                for k_h in range(K_H):
                    for w_out in range(W_OUT):
                        for k_w in range(K_W):
                            for g in range(GROUP):
                                l_t_h = int32_floor_index[n][h_out][w_out][1][g][k_h][k_w]
                                l_t_w = int32_floor_index[n][h_out][w_out][0][g][k_h][k_w]

                                if 0 <= l_t_h < H_IN and 0 <= l_t_w < W_IN:
                                    l_t_tensor[n][h_out][k_h][w_out][k_w] = x[n][l_t_h][l_t_w][g]
                                else:
                                    l_t_tensor[n][h_out][k_h][w_out][k_w] = 0

                                l_b_h = int32_ceil_index[n][h_out][w_out][1][g][k_h][k_w]
                                l_b_w = int32_floor_index[n][h_out][w_out][0][g][k_h][k_w]

                                if 0 <= l_b_h < H_IN and 0 <= l_b_w < W_IN:
                                    l_b_tensor[n][h_out][k_h][w_out][k_w] = x[n][l_b_h][l_b_w][g]
                                else:
                                    l_b_tensor[n][h_out][k_h][w_out][k_w] = 0

                                r_t_h = int32_floor_index[n][h_out][w_out][1][g][k_h][k_w]
                                r_t_w = int32_ceil_index[n][h_out][w_out][0][g][k_h][k_w]

                                if 0 <= r_t_h < H_IN and 0 <= r_t_w < W_IN:
                                    r_t_tensor[n][h_out][k_h][w_out][k_w] = x[n][r_t_h][r_t_w][g]
                                else:
                                    r_t_tensor[n][h_out][k_h][w_out][k_w] = 0

                                r_b_h = int32_ceil_index[n][h_out][w_out][1][g][k_h][k_w]
                                r_b_w = int32_ceil_index[n][h_out][w_out][0][g][k_h][k_w]

                                if 0 <= r_b_h < H_IN and 0 <= r_b_w < W_IN:
                                    r_b_tensor[n][h_out][k_h][w_out][k_w] = x[n][r_b_h][r_b_w][g]
                                else:
                                    r_b_tensor[n][h_out][k_h][w_out][k_w] = 0

        ceil_sub_value = ceil_index - index_offsets
        ceil_sub_value = 1 - ceil_sub_value
        sub_floor_value = index_offsets - floor_index
        sub_floor_value = 1 - sub_floor_value

        l_t_weight = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)
        r_t_weight = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)
        r_b_weight = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)
        l_b_weight = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)

        scale_weight = np.zeros((N, H_OUT, K_H, W_OUT, K_W, GROUP, GROUP_C), np.float32)
        for n in range(N):
            for h_out in range(H_OUT):
                for k_h in range(K_H):
                    for w_out in range(W_OUT):
                        for k_w in range(K_W):
                            for g in range(GROUP):
                                l_t_h = sub_floor_value[n][h_out][w_out][1][g][k_h][k_w]
                                l_t_w = sub_floor_value[n][h_out][w_out][0][g][k_h][k_w]
                                l_t_weight[n][h_out][k_h][w_out][k_w][g] = l_t_h * l_t_w

                                l_b_h = ceil_sub_value[n][h_out][w_out][1][g][k_h][k_w]
                                l_b_w = sub_floor_value[n][h_out][w_out][0][g][k_h][k_w]
                                l_b_weight[n][h_out][k_h][w_out][k_w][g] = l_b_h * l_b_w

                                r_t_h = sub_floor_value[n][h_out][w_out][1][g][k_h][k_w]
                                r_t_w = ceil_sub_value[n][h_out][w_out][0][g][k_h][k_w]
                                r_t_weight[n][h_out][k_h][w_out][k_w][g] = r_t_h * r_t_w

                                r_b_h = ceil_sub_value[n][h_out][w_out][1][g][k_h][k_w]
                                r_b_w = ceil_sub_value[n][h_out][w_out][0][g][k_h][k_w]
                                r_b_weight[n][h_out][k_h][w_out][k_w][g] = r_b_h * r_b_w

                                scale_weight[n][h_out][k_h][w_out][k_w][g] = offsets[n][h_out][w_out][2][g][k_h][k_w]
        out_tensor = (
            l_t_tensor * l_t_weight + l_b_tensor * l_b_weight + r_t_tensor * r_t_weight + r_b_tensor * r_b_weight
        )
        out_tensor = out_tensor * scale_weight
        if dtype == np.float16:
            out_tensor = out_tensor.astype(np.float16)
        return out_tensor.reshape((N, H_OUT * K_H, W_OUT * K_W, C))

    def get_fwd_golden(self, x, weight, offset, args):
        ksize, strides, pads, dilations, groups = args
        x_nhwc = torch_npu.npu_transpose(x, (0, 2, 3, 1), True).cpu().numpy()
        o_nhwc = torch_npu.npu_transpose(offset, (0, 2, 3, 1), True).cpu().numpy()
        deformable_offsets_args = (ksize, strides, pads, dilations)
        deformable_offsets_out = self.deformable_offsets(x_nhwc, o_nhwc, deformable_offsets_args)
        deformable_offsets_out_nchw = torch_npu.npu_transpose(
            torch.from_numpy(deformable_offsets_out).npu(), (0, 3, 1, 2), True
        )
        conv2d_out = torch_npu.npu_conv2d(
            deformable_offsets_out_nchw, weight, None, ksize, (0, 0, 0, 0), (1, 1), groups
        )
        return conv2d_out, deformable_offsets_out_nchw

    def test_deformable_conv2d(self):
        N, cIn, cOut, K, hIn, wIn, hOut, wOut = 18, 512, 512, 3, 29, 50, 29, 50

        npu_x = self.create_single_npu_tensor([np.float32, 0, (N, cIn, hIn, wIn)], -5, 5)
        npu_w = self.create_single_npu_tensor([np.float32, 0, (cOut, cIn, K, K)], -5, 5)
        npu_o = self.create_single_npu_tensor([np.float32, 0, (N, 2 * K * K, hOut, wOut)], -5, 5)

        dcn_out = mx_driving.fused.deformable_conv2d(npu_x, npu_o, npu_w, 1, 1, 1)


if __name__ == "__main__":
    torch.npu.conv.allow_hf32 = False
    run_tests()
