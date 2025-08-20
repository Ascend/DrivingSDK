# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
from types import ModuleType
from typing import Dict
import torch
import torch_npu

import mx_driving
from mx_driving.patcher import PatcherBuilder, Patch
from mx_driving.patcher import batch_matmul

sys.path.append("..")



def lssview(view_transform: ModuleType, options: Dict):

    def get_lidar_coor_(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                    bda):
        B, N, _, _ = sensor2ego.shape

        # post-transformation
        # B x N x D x H x W x 3
        points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
        B, N, D, H, W, _ = points.shape
        points = points.view(B, N, D * H * W, 3, 1)
        points = torch.inverse(post_rots).view(B, N, 1, 3, 3).matmul(points)

        # cam_to_ego
        points = torch.cat((points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 3)
        combine = sensor2ego[:, :, :3, :3].matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 3, 3).matmul(points).squeeze(-1)
        points += sensor2ego[:, :, :3, 3].view(B, N, 1, 3)
        points = bda[:, :3, :3].view(B, 1, 1, 3, 3).matmul(
            points.unsqueeze(-1)).squeeze(-1)
        points += bda[:, :3, 3].view(B, 1, 1, 3)
        return points.view(B, N, D, H, W, 3)

    def gen_grid_(self, metas, B, N, D, H, W, hi, wi):
        frustum = metas['frustum']
        points = frustum - metas['post_trans'].view(B, N, 1, 1, 1, 3)
        ori_shape = points.shape
        points = points.view(B, N, -1, 3)
        points = torch.inverse(metas['post_rots']).view(B, N, 1, 3, 3) \
            .matmul(points.unsqueeze(-1))
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 3)

        rots = metas['k2s_sensor'][:, :, :3, :3].contiguous()
        trans = metas['k2s_sensor'][:, :, :3, 3].contiguous()
        combine = rots.matmul(torch.inverse(metas['intrins']))

        points = combine.view(B, N, 1, 3, 3).matmul(points)
        points += trans.view(B, N, 1, 3, 1)
        neg_mask = (points.view(ori_shape))[..., 2, 0] < 1e-3
        points = metas['intrins'].view(B, N, 1, 3, 3).matmul(points)
        points = points[..., :2, :] / points[..., 2:3, :]

        points = metas['post_rots'][..., :2, :2].view(B, N, 1, 2, 2).matmul(
            points).squeeze(-1)
        points += metas['post_trans'][..., :2].view(B, N, 1, 2)

        new_shape = list(ori_shape)
        new_shape[-1] = 2
        points = points.view(new_shape)
        px = points[..., 0] / (wi - 1.0) * 2.0 - 1.0
        py = points[..., 1] / (hi - 1.0) * 2.0 - 1.0
        px[neg_mask] = -2
        py[neg_mask] = -2
        grid = torch.stack([px, py], dim=-1)
        grid = grid.view(B * N, D * H, W, 2)
        return grid

    if hasattr(view_transform, "LSSViewTransformer"):
        view_transform.LSSViewTransformer.get_lidar_coor = get_lidar_coor_
    if hasattr(view_transform, "DepthNet"):
        view_transform.DepthNet.gen_grid = gen_grid_


def swin_fp16(swin: ModuleType, options: Dict):

    def forward_fp16(self, x):
        # change the code to support fp16
        with torch.autocast(device_type="npu", dtype=torch.float16):
            x = self.patch_embed(x)

            hw_shape = (self.patch_embed.DH, self.patch_embed.DW)
            if self.use_abs_pos_embed:
                x = x + self.absolute_pos_embed
            x = self.drop_after_pos(x)

            outs = []
            for i, stage in enumerate(self.stages):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
                if i == 0 and self.return_stereo_feat:
                    out = out.view(-1, *out_hw_shape,
                                    self.num_features[i]).permute(0, 3, 1,
                                                                    2).contiguous()
                    outs.append(out)
                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                    out = norm_layer(out)
                    out = out.view(-1, *out_hw_shape,
                                    self.num_features[i]).permute(0, 3, 1,
                                                                    2).contiguous()
                    outs.append(out)
                elif self.output_missing_index_as_none:
                    outs.append(None)
            return [out.float() for out in outs]

    if hasattr(swin, "SwinTransformer"):
        swin.SwinTransformer.forward = forward_fp16


def generate_patcher_builder():
    bevdet4d_patcher_builder = (
        PatcherBuilder()
        .add_module_patch("torch", Patch(batch_matmul))
        .add_module_patch("mmdet3d.models.necks.view_transformer", Patch(lssview))
    )
    if os.environ.get("BEVDET4D_FP16"):
        bevdet4d_patcher_builder.add_module_patch("mmdet3d.models.backbones.swin", Patch(swin_fp16))
    if os.environ.get("BEVDET4D_PERFORMANCE_FLAG"):
        bevdet4d_patcher_builder.brake_at(250)
    return bevdet4d_patcher_builder
