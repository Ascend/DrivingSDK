# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Robertwyq. All rights reserved.
# Copyright (c) Alibaba; Inc. and its affiliates. All rights reserved.

# pylint: disable=huawei-wrong-import-position, wrong-import-order
import importlib
import collections
import sys
import os
import math
import types
import warnings
import random
from types import ModuleType
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
import mmcv
import mmcv.runner

from .patch_panoseg_occ_head import panoseg_occ_head_patch

from mx_driving.patcher import PatcherBuilder, Patch
from mx_driving.patcher.distribute import ddp
from mx_driving.patcher.functions import stream
from mx_driving.patcher.tensor import index, batch_matmul
from mx_driving.patcher.numpy import numpy_type
from mx_driving.patcher.mmcv import mdc, msda, dc
from mx_driving.patcher.optimizer import optimizer_hooks
from mx_driving.patcher.mmdet import resnet_add_relu, resnet_maxpool
from mx_driving import multi_scale_deformable_attn


# pylint: disable=huawei-redefined-outer-name, inconsistent-return-statements
def panoseg_transformer_occ_patch(panoseg_transformer_occ_module: ModuleType, options: Dict):
    
    def align_prev_bev(self, prev_bev, bev_h, bev_w, bev_z, **kwargs):
        if prev_bev is not None:
            pc_range = self.cam_encoder.pc_range
            ref_y, ref_x, ref_z = torch.meshgrid(
                    torch.linspace(0.5, bev_h - 0.5, bev_h, dtype=prev_bev.dtype, device=prev_bev.device),
                    torch.linspace(0.5, bev_w - 0.5, bev_w, dtype=prev_bev.dtype, device=prev_bev.device),
                    torch.linspace(0.5, bev_z - 0.5, bev_z, dtype=prev_bev.dtype, device=prev_bev.device),
                )
            ref_y = ref_y / bev_h
            ref_x = ref_x / bev_w
            ref_z = ref_z / bev_z

            grid = torch.stack(
                    (ref_x,
                    ref_y,
                    ref_z,
                    ref_x.new_ones(ref_x.shape)), dim=-1)

            min_x, min_y, min_z, max_x, max_y, max_z = pc_range
            grid[..., 0] = grid[..., 0] * (max_x - min_x) + min_x
            grid[..., 1] = grid[..., 1] * (max_y - min_y) + min_y
            grid[..., 2] = grid[..., 2] * (max_z - min_z) + min_z
            grid = grid.reshape(-1, 4)

            bs = prev_bev.shape[0]
            len_queue = prev_bev.shape[1]
            for i in range(bs):
                lidar_to_ego = kwargs['img_metas'][i]['lidar2ego_transformation']
                curr_ego_to_global = kwargs['img_metas'][i]['ego2global_transform_lst'][-1]

                curr_grid_in_prev_frame_lst = []
                for j in range(len_queue):
                    prev_ego_to_global = kwargs['img_metas'][i]['ego2global_transform_lst'][j]
                    prev_lidar_to_curr_lidar = np.linalg.inv(lidar_to_ego) @ np.linalg.inv(curr_ego_to_global) @ prev_ego_to_global @ lidar_to_ego
                    curr_lidar_to_prev_lidar = np.linalg.inv(prev_lidar_to_curr_lidar)
                    curr_lidar_to_prev_lidar = grid.new_tensor(curr_lidar_to_prev_lidar)

                    # fix z
                    curr_lidar_to_prev_lidar[2, 3] = curr_lidar_to_prev_lidar[2, 3] * 0

                    curr_grid_in_prev_frame = torch.matmul(curr_lidar_to_prev_lidar, grid.T).T.reshape(bev_h, bev_w, bev_z, -1)[..., :3]
                    curr_grid_in_prev_frame[..., 0] = (curr_grid_in_prev_frame[..., 0] - min_x) / (max_x - min_x)
                    curr_grid_in_prev_frame[..., 1] = (curr_grid_in_prev_frame[..., 1] - min_y) / (max_y - min_y)
                    curr_grid_in_prev_frame[..., 2] = (curr_grid_in_prev_frame[..., 2] - min_z) / (max_z - min_z)
                    curr_grid_in_prev_frame = curr_grid_in_prev_frame * 2.0 - 1.0
                    curr_grid_in_prev_frame_lst.append(curr_grid_in_prev_frame)

                curr_grid_in_prev_frame = torch.stack(curr_grid_in_prev_frame_lst, dim=0)

                
                torch.npu.set_compile_mode(jit_compile=True) # +++
                prev_bev_warp_to_curr_frame = nn.functional.grid_sample(
                    prev_bev[i].permute(0, 1, 4, 2, 3),  # [bs, dim, z, h, w]
                    curr_grid_in_prev_frame.permute(0, 3, 1, 2, 4),  # [bs, z, h, w, 3]
                    align_corners=False)
                torch.npu.set_compile_mode(jit_compile=False) # +++
                
                prev_bev = prev_bev_warp_to_curr_frame.permute(0, 1, 3, 4, 2).unsqueeze(0) # add bs dim, [bs, dim, h, w, z]

            return prev_bev

    
    if hasattr(panoseg_transformer_occ_module, 'PanoSegOccTransformer'):
        panoseg_transformer_occ_module.PanoSegOccTransformer.align_prev_bev = align_prev_bev
    else:
        raise AttributeError('PanoSegOccTransformer attr not found')


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# pylint: disable=huawei-redefined-outer-name, huawei-too-many-arguments
def mmdet3d_dataset_builder_patch(builder_module: ModuleType, options: Dict):
    from functools import partial
    from mmcv.runner import get_dist_info
    from mmcv.parallel import collate
    from mmdet.datasets.samplers import GroupSampler
    from projects.mmdet3d_plugin.datasets.samplers.sampler import build_sampler
    from torch.utils.data import DataLoader

    # PATCH: Pin Memory
    def build_dataloader(dataset,
                        samples_per_gpu,
                        workers_per_gpu,
                        num_gpus=1,
                        dist=True,
                        shuffle=True,
                        seed=None,
                        shuffler_sampler=None,
                        nonshuffler_sampler=None,
                        **kwargs):
        rank, world_size = get_dist_info()
        if dist:
            # DistributedGroupSampler will definitely shuffle the data to satisfy
            # that images on each GPU are in the same group
            if shuffle:
                sampler = build_sampler(shuffler_sampler if shuffler_sampler is not None else dict(type='DistributedGroupSampler'),
                                        dict(
                                            dataset=dataset,
                                            samples_per_gpu=samples_per_gpu,
                                            num_replicas=world_size,
                                            rank=rank,
                                            seed=seed)
                                        )

            else:
                sampler = build_sampler(nonshuffler_sampler if nonshuffler_sampler is not None else dict(type='DistributedSampler'),
                                        dict(
                                            dataset=dataset,
                                            num_replicas=world_size,
                                            rank=rank,
                                            shuffle=shuffle,
                                            seed=seed)
                                        )

            batch_size = samples_per_gpu
            num_workers = workers_per_gpu
        else:
            # assert False, 'not support in bevformer'
            print('WARNING!!!!, Only can be used for obtain inference speed!!!!')
            sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
            batch_size = num_gpus * samples_per_gpu
            num_workers = num_gpus * workers_per_gpu

        init_fn = partial(
            worker_init_fn, num_workers=num_workers, rank=rank,
            seed=seed) if seed is not None else None

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            pin_memory=True,
            worker_init_fn=init_fn,
            **kwargs)

        return data_loader

    if hasattr(builder_module, 'build_dataloader'):
        builder_module.build_dataloader = build_dataloader
    else:
        raise AttributeError('build_dataloader attr not found')


# pylint: disable=huawei-redefined-outer-name, huawei-too-many-arguments
def mmdet3d_dataset_compose_patch(compose_module: ModuleType, options: Dict):
    
    from mmcv.utils import build_from_cfg
    from mmdet.datasets.builder import PIPELINES
    from mmdet3d.datasets.builder import PIPELINES as PIPELINES_3d
    
    def __init__(self, transforms):
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                if transform["type"] not in PIPELINES:
                    transform = build_from_cfg(transform, PIPELINES_3d)
                else:
                    transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    if hasattr(compose_module, 'CustomCompose'):
        compose_module.CustomCompose.__init__ = __init__
    else:
        raise AttributeError('CustomCompose attr not found')


indexes_global = None
max_len_global = None
bev_mask_id_global = -1
count_global = None


# pylint: disable=huawei-redefined-outer-name, huawei-too-many-arguments
def spatial_cross_attention_patch(spatial_cross_attention_module: ModuleType, options: Dict):
    
    from mmcv.runner import force_fp32

    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def sca_forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()
        # bevformer reference_points_cam shape: (num_cam,bs,h*w,num_points_in_pillar,2)
        D = reference_points_cam.size(3)
        indexes = []
        global indexes_global, max_len_global, bev_mask_id_global, count_global
        bev_mask_id = id(bev_mask)
        if bev_mask_id == bev_mask_id_global:
            indexes = indexes_global
            max_len = max_len_global
            count = count_global
        else:
            count = torch.any(bev_mask, 3)
            bev_mask_ = count.squeeze()
            for _, mask_per_img in enumerate(bev_mask_):
                index_query_per_img = mask_per_img.nonzero().squeeze(-1)
                indexes.append(index_query_per_img)

            max_len = max([len(each) for each in indexes])
            count = count.permute(1, 2, 0).sum(-1)
            count = torch.clamp(count, min=1.0)
            count = count[..., None]
            count_global = count
            indexes_global = indexes
            max_len_global = max_len
            bev_mask_id_global = bev_mask_id

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])

        for i, reference_points_per_img in enumerate(reference_points_cam):
            index_query_per_img = indexes[i]
            for j in range(bs):
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        num_cams, key_l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, key_l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, key_l, self.embed_dims)

        queries = self.deformable_attention(query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs * self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]


        slots = slots / count
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


    def msda3d_forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4:
            raise ValueError('reference_points.shape[-1] == 4')
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        if torch.cuda.is_available() and value.is_cuda:
            output = multi_scale_deformable_attn(value, spatial_shapes, level_start_index,
                                                                         sampling_locations, attention_weights)
        else:
            raise ValueError(f'torch.cuda.is_available() is {torch.cuda.is_available()}'
                             f'value.is_cuda is {value.is_cuda}')
            
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output

    sca_not_found = False
    msda3d_not_found = False
    if hasattr(spatial_cross_attention_module, 'SpatialCrossAttention'):
        spatial_cross_attention_module.SpatialCrossAttention.forward = sca_forward
    else:
        sca_not_found = True
        
    if hasattr(spatial_cross_attention_module, 'MSDeformableAttention3D'):
        spatial_cross_attention_module.MSDeformableAttention3D.forward = msda3d_forward
    else:
        msda3d_not_found = True
        
    if sca_not_found:
        raise AttributeError('SpatialCrossAttention attr not found')    
    if msda3d_not_found:
        raise AttributeError('MSDeformableAttention3D attr not found')


# pylint: disable=huawei-redefined-outer-name, huawei-too-many-arguments
def panoocc_custom_msda_patch(panoocc_custom_msda_module: ModuleType, options: Dict):
    from mx_driving import multi_scale_deformable_attn
    

    class MsdaWrapper:
        def apply(self, value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, im2col_step):
            multi_scale_deformable_attn(value, spatial_shapes, level_start_index, 
                sampling_locations, attention_weights)

            
    # Patch target: projects.mmdet3d_plugin.bevformer.modules.multi_scale_deformable_attn_function
    if hasattr(panoocc_custom_msda_module, 'MultiScaleDeformableAttnFunction_fp32'):
        panoocc_custom_msda_module.MultiScaleDeformableAttnFunction_fp32 = MsdaWrapper
    else:
        raise AttributeError('MultiScaleDeformableAttnFunction_fp32 not found')


# pylint: disable=huawei-redefined-outer-name, huawei-too-many-arguments
def decoder_patch(decoder_module: ModuleType, options: Dict):
    
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',
                **kwargs):

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            output = multi_scale_deformable_attn(value, spatial_shapes, level_start_index,
                sampling_locations, attention_weights)
        else:
            raise ValueError(f'torch.cuda.is_available() is {torch.cuda.is_available()}'
                    f'value.is_cuda is {value.is_cuda}')

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

    if hasattr(decoder_module, 'CustomMSDeformableAttention'):
        decoder_module.CustomMSDeformableAttention.forward = forward
    else:
        raise AttributeError('CustomMSDeformableAttention attr not found')


# pylint: disable=huawei-too-many-arguments
def occ_temporal_attention_patch(occ_temporal_attention_module: ModuleType, options: Dict):
    
    def forward(self, query, key=None, value=None, identity=None, query_pos=None, key_padding_mask=None,
                reference_points=None, spatial_shapes=None, level_start_index=None, flag='decoder', **kwargs):
        if value is None:
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs * 2, len_bev, c)

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape

        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        value = value.reshape(bs * self.num_bev_queue,
                              num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_bev_queue,
                                                   self.num_levels,
                                                   self.num_points)

        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        # all points in pillar have the same xy
        z_num = sampling_offsets.shape[1] // reference_points.shape[1]
        bsq, bev_num, level, xy = reference_points.shape
        reference_points = reference_points.unsqueeze(2).expand(bsq, bev_num, z_num, level, xy).reshape(bsq, -1, level, xy)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        sampling_locations = sampling_locations.contiguous()
        if torch.cuda.is_available() and value.is_cuda:
            output = multi_scale_deformable_attn(value, spatial_shapes, level_start_index,
                sampling_locations, attention_weights)
        else:
            raise ValueError("CUDA/CANN unavailable?")

        # output shape (bs*num_bev_queue, num_query, embed_dims)
        # (bs*num_bev_queue, num_query, embed_dims)-> (num_query, embed_dims, bs*num_bev_queue)
        output = output.permute(1, 2, 0)

        # fuse history value and current value
        # (num_query, embed_dims, bs*num_bev_queue)-> (num_query, embed_dims, bs, num_bev_queue)
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
        output = output.mean(-1)

        # (num_query, embed_dims, bs)-> (bs, num_query, embed_dims)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

    if hasattr(occ_temporal_attention_module, 'OccTemporalAttention'):
        occ_temporal_attention_module.OccTemporalAttention.forward = forward
    else:
        raise AttributeError('OccTemporalAttention attr not found')


# pylint: disable=huawei-too-many-arguments
def temporal_self_attention_patch(temporal_self_attention_module: ModuleType, options: Dict):
    def forward(self, query, key=None, value=None, identity=None, query_pos=None, key_padding_mask=None,
                reference_points=None, spatial_shapes=None, level_start_index=None, flag='decoder', **kwargs):
        if value is None:
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs * 2, len_bev, c)

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        bs, num_query, embed_dims = query.shape
        _, num_value, _ = value.shape

        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        value = value.reshape(bs * self.num_bev_queue,
                              num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_bev_queue,
                                                   self.num_levels,
                                                   self.num_points)

        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]

        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            output = multi_scale_deformable_attn(value, spatial_shapes, level_start_index,
                sampling_locations, attention_weights)
        else:
            raise ValueError("CUDA/CANN unavailable?")

        # output shape (bs*num_bev_queue, num_query, embed_dims)
        # (bs*num_bev_queue, num_query, embed_dims)-> (num_query, embed_dims, bs*num_bev_queue)
        output = output.permute(1, 2, 0)

        # fuse history value and current value
        # (num_query, embed_dims, bs*num_bev_queue)-> (num_query, embed_dims, bs, num_bev_queue)
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
        output = output.mean(-1)

        # (num_query, embed_dims, bs)-> (bs, num_query, embed_dims)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

    if hasattr(temporal_self_attention_module, 'TemporalSelfAttention'):
        temporal_self_attention_module.TemporalSelfAttention.forward = forward
    else:
        raise AttributeError('TemporalSelfAttention attr not found')


def generate_patcher_builder():
    patcher_builder = (
        PatcherBuilder()
        .add_module_patch("torch", Patch(index), Patch(batch_matmul))
        .add_module_patch("numpy", Patch(numpy_type))
        
        .add_module_patch('mmcv.parallel', Patch(stream))
        .add_module_patch('mmcv.parallel.distributed', Patch(ddp))
        
        .add_module_patch('mmcv.ops', Patch(mdc), Patch(msda), Patch(dc))
        
        .add_module_patch('mmcv.runner.hooks', Patch(optimizer_hooks))
        .add_module_patch("mmdet.models.backbones.resnet", Patch(resnet_add_relu), Patch(resnet_maxpool))
        
        
        .add_module_patch('projects.mmdet3d_plugin.bevformer.dense_heads.panoseg_occ_head', 
                          Patch(panoseg_occ_head_patch))
        
        .add_module_patch('projects.mmdet3d_plugin.bevformer.modules.panoseg_transformer_occ', 
                          Patch(panoseg_transformer_occ_patch))

        .add_module_patch('projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention', 
                          Patch(spatial_cross_attention_patch))
        
        .add_module_patch('projects.mmdet3d_plugin.datasets.builder', Patch(mmdet3d_dataset_builder_patch))
        
        .add_module_patch('projects.mmdet3d_plugin.datasets.pipelines.compose', Patch(mmdet3d_dataset_compose_patch))
        
        
        .add_module_patch('projects.mmdet3d_plugin.bevformer.modules.decoder', Patch(decoder_patch))
        .add_module_patch('projects.mmdet3d_plugin.bevformer.modules.occ_temporal_attention', 
                          Patch(occ_temporal_attention_patch))
        .add_module_patch('projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention', 
                          Patch(temporal_self_attention_patch))
    )
    return patcher_builder

 
brake_flag = False
profile_flag = False


def set_brake_at_step(patcher_builder: PatcherBuilder, end_step: int = 1000):
    if profile_flag is True:
        raise RuntimeError('with_profiling has been set, brake and profiling are mutually exclusive')
    patcher_builder.brake_at(end_step)
    brake_flag = True


def set_profiling(patcher_builder: PatcherBuilder, profiling_path: str, profiling_level: int = 0):
    if brake_flag is True:
        raise RuntimeError('brake_at has been set, brake and profiling are mutually exclusive')
    patcher_builder.with_profiling(profiling_path, profiling_level)
    profile_flag = True


class MethodPatcher:
    
    @staticmethod
    def nccl_to_hccl(runner: ModuleType):
        module = importlib.import_module(runner)

        if hasattr(module, "dist_utils"):
            mp = module.dist_utils.mp
            _init_dist_pytorch = module.dist_utils._init_dist_pytorch
            _init_dist_mpi = module.dist_utils._init_dist_mpi
            _init_dist_slurm = module.dist_utils._init_dist_slurm

            def hccl_init_dist(launcher: str, backend: str = 'nccl', **kwargs) -> None:
                
                # Replacement for using hccl as the backend
                backend = 'hccl'
                
                if mp.get_start_method(allow_none=True) is None:
                    mp.set_start_method('spawn')
                if launcher == 'pytorch':
                    _init_dist_pytorch(backend, **kwargs)
                elif launcher == 'mpi':
                    _init_dist_mpi(backend, **kwargs)
                elif launcher == 'slurm':
                    _init_dist_slurm(backend, **kwargs)
                else:
                    raise ValueError(f'Invalid launcher type: {launcher}')
        else:
            raise AttributeError('dist_utils attr not found')

        module.init_dist = hccl_init_dist


class ConfigPatcher:
    @staticmethod
    def adamw_to_npu_fused_adam(runner: ModuleType):
        module = importlib.import_module(runner)
        copy = module.optimizer.builder.copy
        build_optimizer_constructor = module.optimizer.builder.build_optimizer_constructor
        
        def build_optimizer(model, cfg: Dict):
            optimizer_cfg = copy.deepcopy(cfg)
            
            # PATCH: use NpuFusedAdam optimizer instead
            optimizer_cfg['type'] = 'NpuFused' + optimizer_cfg['type']
            
            constructor_type = optimizer_cfg.pop('constructor',
                                                'DefaultOptimizerConstructor')
            paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
            optim_constructor = build_optimizer_constructor(
                dict(
                    type=constructor_type,
                    optimizer_cfg=optimizer_cfg,
                    paramwise_cfg=paramwise_cfg))
            optimizer = optim_constructor(model)
            return optimizer

        module.build_optimizer = build_optimizer


def _init():
    # block dependencies that are not used nor installed
    sys.modules['mmdet3d.ops.scatter_v2'] = ModuleType('mmdet3d.ops.scatter_v2')
    sys.modules['torch_scatter'] = ModuleType('torch_scatter')
    
    sys.modules['projects.mmdet3d_plugin.models.backbones.sam_modeling.image_encoder'] = \
        ModuleType('projects.mmdet3d_plugin.models.backbones.sam_modeling.image_encoder')
    sys.modules['projects.mmdet3d_plugin.models.backbones.sam_modeling.image_encoder.ImageEncoderViT'] = \
        ModuleType('projects.mmdet3d_plugin.models.backbones.sam_modeling.image_encoder.ImageEncoderViT')
    
    sys.modules['projects.mmdet3d_plugin.models.backbones.internv2_impl16'] = \
        ModuleType('projects.mmdet3d_plugin.models.backbones.internv2_impl16')
    sys.modules['projects.mmdet3d_plugin.models.backbones.internv2_impl16.InternV2Impl16'] = \
        ModuleType('projects.mmdet3d_plugin.models.backbones.internv2_impl16.InternV2Impl16')
    
    sys.modules['spconv'] = ModuleType('spconv')
    sys.modules['spconv.pytorch'] = ModuleType('spconv.pytorch')
    sys.modules['spconv.pytorch.SparseConvTensor'] = ModuleType('spconv.pytorch.SparseConvTensor')
    sys.modules['spconv.pytorch.SparseSequential'] = ModuleType('spconv.pytorch.SparseSequential')
    
    sys.modules['ipdb'] = ModuleType('ipdb')
    sys.modules['ipdb.set_trace'] = ModuleType('ipdb.set_trace')
    
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    
    
    MethodPatcher.nccl_to_hccl('mmcv.runner')
    ConfigPatcher.adamw_to_npu_fused_adam('mmcv.runner')


''' 
Initialize to execute method patcher
Takes place before their corresponding imports
'''
_init()