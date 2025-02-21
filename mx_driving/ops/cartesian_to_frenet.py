"""
Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""

import torch
import torch_npu
from torch.autograd import Function

import mx_driving._C


class CartesianToFrenet(Function):
    @staticmethod
    def forward(ctx, pt, poly_line):

        pt_shape_ori = pt.shape
        poly_line_shape_ori = poly_line.shape

        if (pt_shape_ori[:-2] != poly_line_shape_ori[:-2]):
            raise Exception("Point and poly_line must share the same batch size.")

        if (len(pt_shape_ori) > 3):
            pt = pt.reshape(-1, pt_shape_ori[-2], pt_shape_ori[-1])
            poly_line = poly_line.reshape(-1, poly_line_shape_ori[-2], poly_line_shape_ori[-1])

        poly_line_shape = list(poly_line.shape)
        poly_line_shape[-2] = 1
        diff_tensor = torch.zeros(poly_line_shape, dtype=poly_line.dtype, device=poly_line.device)
        diff_tensor = torch.cat([diff_tensor, poly_line[..., 1:, :] - poly_line[..., :-1, :]], dim=-2)
        diff_dist = torch.norm(diff_tensor, dim=-1)  # (*shape, k2)
        s_cum = torch.cumsum(diff_dist, dim=-1)  # (*shape, k2)

        dist_vec = pt.unsqueeze(dim=-2) - poly_line.unsqueeze(dim=-3)

        min_idx, back_idx = mx_driving._C.cartesian_to_frenet1(dist_vec)
        min_idx = mx_driving._C.select_idx_with_mask(poly_line, min_idx, pt, back_idx)
        poly_start, poly_end, sl = mx_driving._C.calc_poly_start_end_sl(min_idx, poly_line, pt, s_cum)

        if (len(pt_shape_ori) > 3):
            poly_start = poly_start.reshape(pt_shape_ori[:-2] + (-1, 2))
            poly_end = poly_end.reshape(poly_line_shape_ori[:-2] + (-1, 2))
            sl = sl.reshape(pt_shape_ori[:-2] + (-1, 2))

        return poly_start, poly_end, sl

cartesian_to_frenet = CartesianToFrenet.apply
