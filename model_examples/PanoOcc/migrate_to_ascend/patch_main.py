# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Robertwyq. All rights reserved.
# Copyright (c) Alibaba; Inc. and its affiliates. All rights reserved.
# Copyright (c) NVIDIA Corporation Affiliates. All rights reserved.
"""
PanoOcc NPU Migration Patches

Usage:
    from mx_driving.patcher import default_patcher
    from migrate_to_ascend.patch_main import configure_patcher

    configure_patcher(default_patcher, performance=False)
    default_patcher.apply()
"""
from __future__ import annotations

import importlib
import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch_npu

from mx_driving import multi_scale_deformable_attn
from mx_driving.patcher import Patcher, Patch, AtomicPatch, OptimizerHooks
from mx_driving.patcher.patch import with_imports

from .patch_panoseg_occ_head import PanoSegOccHeadPatch


# =============================================================================
# Skip modules (unavailable CUDA deps)
# =============================================================================
SKIP_MODULES = [
    "mmdet3d.ops.scatter_v2",
    "torch_scatter",
    "projects.mmdet3d_plugin.models.backbones.sam_modeling.image_encoder",
    "projects.mmdet3d_plugin.models.backbones.sam_modeling.image_encoder.ImageEncoderViT",
    "projects.mmdet3d_plugin.models.backbones.internv2_impl16",
    "projects.mmdet3d_plugin.models.backbones.internv2_impl16.InternV2Impl16",
    "spconv",
    "spconv.pytorch",
    "spconv.pytorch.SparseConvTensor",
    "spconv.pytorch.SparseSequential",
    "ipdb",
    "ipdb.set_trace",
]

indexes_global = None
max_len_global = None
bev_mask_id_global = -1
count_global = None


# =============================================================================
# Patcher Configuration
# =============================================================================


def _configure_npu_runtime():
    try:
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = False
    except (AttributeError, RuntimeError):
        return


def configure_patcher(patcher: Patcher, performance: bool = False) -> Patcher:
    """
    Configure Patcher for PanoOcc.

    Args:
        patcher: Patcher instance (typically default_patcher)
        performance: Whether to enable performance mode (brake at 1000 steps)

    Returns:
        Configured patcher
    """
    _configure_npu_runtime()

    # 1. Skip unavailable CUDA modules
    patcher.skip_import(*SKIP_MODULES)

    # 2. Patcher runtime options
    patcher.disallow_internal_format()

    # 3. Add required patches
    patcher.add(
        OptimizerHooks,
        PanoSegOccHeadPatch,
        PanoSegOccTransformerOccPatch,
        SpatialCrossAttentionPatch,
        Mmdet3dDatasetBuilderPatch,
        Mmdet3dDatasetComposePatch,
        DecoderPatch,
        OccTemporalAttentionPatch,
        TemporalSelfAttentionPatch,
        HcclBackend,
        NpuFusedAdam,
    )

    # 4. Performance mode
    if performance:
        patcher.brake_at(1000)

    return patcher


# =============================================================================
# Patch Classes
# =============================================================================

class PanoSegOccTransformerOccPatch(Patch):
    name = "panoocc_panoseg_transformer_occ"

    # pylint: disable=huawei-redefined-outer-name, inconsistent-return-statements
    @staticmethod
    def _align_prev_bev(self, prev_bev, bev_h, bev_w, bev_z, **kwargs):
        # Patch: jit compile optimization
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

    @classmethod
    def patches(cls, options=None):
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.bevformer.modules.panoseg_transformer_occ.PanoSegOccTransformer.align_prev_bev",
                cls._align_prev_bev,
            ),
        ]


class Mmdet3dDatasetBuilderPatch(Patch):
    name = "panoocc_mmdet3d_dataset_builder"

    @staticmethod
    def _worker_init_fn(worker_id, num_workers, rank, seed):
        # The seed of each worker equals to
        # num_worker * rank + worker_id + user_seed
        worker_seed = num_workers * rank + worker_id + seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # pylint: disable=huawei-redefined-outer-name, huawei-too-many-arguments
    @staticmethod
    def _build_dataloader(dataset,
                        samples_per_gpu,
                        workers_per_gpu,
                        num_gpus=1,
                        dist=True,
                        shuffle=False,
                        seed=None,
                        shuffler_sampler=None,
                        nonshuffler_sampler=None,
                        **kwargs):
        from functools import partial
        from mmcv.runner import get_dist_info
        from mmcv.parallel import collate
        from mmdet.datasets.samplers import GroupSampler
        from projects.mmdet3d_plugin.datasets.samplers.sampler import build_sampler
        from torch.utils.data import DataLoader

        # PATCH: Pin Memory, turn off shuffle
        shuffle = False
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
            Mmdet3dDatasetBuilderPatch._worker_init_fn, num_workers=num_workers, rank=rank,
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

    @classmethod
    def patches(cls, options=None):
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.datasets.builder.build_dataloader",
                cls._build_dataloader,
            ),
        ]


class Mmdet3dDatasetComposePatch(Patch):
    name = "panoocc_mmdet3d_dataset_compose"

    # pylint: disable=huawei-redefined-outer-name, huawei-too-many-arguments
    @staticmethod
    def _init(self, transforms):
        from mmcv.utils import build_from_cfg
        from mmdet.datasets.builder import PIPELINES
        from mmdet3d.datasets.builder import PIPELINES as PIPELINES_3d

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

    @classmethod
    def patches(cls, options=None):
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.datasets.pipelines.compose.CustomCompose.__init__",
                cls._init,
            ),
        ]


class SpatialCrossAttentionPatch(Patch):
    name = "panoocc_spatial_cross_attention"

    @staticmethod
    def _sca_forward(self,
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


    @staticmethod
    def _msda3d_forward(self,
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

    @classmethod
    def patches(cls, options=None):
        from mmcv.runner import force_fp32

        sca_forward = force_fp32(
            apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam')
        )(cls._sca_forward)
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention.SpatialCrossAttention.forward",
                sca_forward,
            ),
            AtomicPatch(
                "projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention.MSDeformableAttention3D.forward",
                cls._msda3d_forward,
            ),
        ]


class PanoOccCustomMsdaPatch(Patch):
    """PanoOcc custom MSDA wrapper (optional)."""
    name = "panoocc_custom_msda"

    class _MsdaWrapper:
        def apply(self, value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, im2col_step):
            multi_scale_deformable_attn(value, spatial_shapes, level_start_index, 
                sampling_locations, attention_weights)

    @classmethod
    def patches(cls, options=None):
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.bevformer.modules.multi_scale_deformable_attn_function.MultiScaleDeformableAttnFunction_fp32",
                cls._MsdaWrapper,
            ),
        ]


class DecoderPatch(Patch):
    name = "panoocc_decoder"

    # pylint: disable=huawei-redefined-outer-name, huawei-too-many-arguments
    @staticmethod
    def _forward(self,
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

    @classmethod
    def patches(cls, options=None):
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.bevformer.modules.decoder.CustomMSDeformableAttention.forward",
                cls._forward,
            ),
        ]


class OccTemporalAttentionPatch(Patch):
    name = "panoocc_occ_temporal_attention"

    # pylint: disable=huawei-too-many-arguments
    @staticmethod
    def _forward(self, query, key=None, value=None, identity=None, query_pos=None, key_padding_mask=None,
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

    @classmethod
    def patches(cls, options=None):
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.bevformer.modules.occ_temporal_attention.OccTemporalAttention.forward",
                cls._forward,
            ),
        ]


class TemporalSelfAttentionPatch(Patch):
    name = "panoocc_temporal_self_attention"

    # pylint: disable=huawei-too-many-arguments
    @staticmethod
    def _forward(self, query, key=None, value=None, identity=None, query_pos=None, key_padding_mask=None,
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

    @classmethod
    def patches(cls, options=None):
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention.TemporalSelfAttention.forward",
                cls._forward,
            ),
        ]


class HcclBackend(Patch):
    """Replace NCCL with HCCL for distributed training."""
    name = "panoocc_hccl_backend"

    @staticmethod
    def _precheck():
        try:
            module = importlib.import_module("mmcv.runner")
            return hasattr(module, "dist_utils")
        except ImportError:
            return False

    @staticmethod
    @with_imports(("mmcv.runner.dist_utils", "mp", "_init_dist_pytorch", "_init_dist_mpi", "_init_dist_slurm"))
    def _init_dist(launcher: str, backend: str = 'nccl', **kwargs) -> None:
        backend = 'hccl'
        if mp.get_start_method(allow_none=True) is None:  # noqa: F821
            mp.set_start_method('spawn')  # noqa: F821
        if launcher == 'pytorch':
            _init_dist_pytorch(backend, **kwargs)  # noqa: F821
        elif launcher == 'mpi':
            _init_dist_mpi(backend, **kwargs)  # noqa: F821
        elif launcher == 'slurm':
            _init_dist_slurm(backend, **kwargs)  # noqa: F821
        else:
            raise ValueError(f'Invalid launcher type: {launcher}')

    @classmethod
    def patches(cls, options=None):
        return [
            AtomicPatch(
                "mmcv.runner.init_dist",
                cls._init_dist,
                precheck=cls._precheck,
            ),
        ]


class NpuFusedAdam(Patch):
    """Use NpuFusedAdam in mmcv runner."""
    name = "panoocc_npu_fused_adam"

    @staticmethod
    def _build_optimizer(model, cfg: Dict):
        module = importlib.import_module("mmcv.runner")
        copy = module.optimizer.builder.copy
        build_optimizer_constructor = module.optimizer.builder.build_optimizer_constructor
        
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

    @classmethod
    def patches(cls, options=None):
        return [
            AtomicPatch(
                "mmcv.runner.build_optimizer",
                cls._build_optimizer,
            ),
        ]


# =============================================================================
# Utilities
# =============================================================================

# pylint: disable=huawei-redefined-outer-name, huawei-too-many-arguments
def remove_dropout():
    if torch.__version__ > "1.8":
        import torch.nn.functional as F
        from torch import _VF
        from torch.overrides import has_torch_function_unary, handle_torch_function

        def function_dropout(input_tensor: torch.Tensor, p: float = 0.5, training: bool = True,
                             inplace: bool = False) -> torch.Tensor:
            if has_torch_function_unary(input_tensor):
                return handle_torch_function(
                    function_dropout, (input_tensor,), input_tensor, p=0., training=training, inplace=inplace)
            if p < 0.0 or p > 1.0:
                raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
            return _VF.dropout_(input_tensor, 0., training) if inplace else _VF.dropout(input_tensor, 0., training)

        def function_dropout2d(input_tensor: torch.Tensor, p: float = 0.5, training: bool = True,
                               inplace: bool = False) -> torch.Tensor:
            if has_torch_function_unary(input_tensor):
                return handle_torch_function(
                    function_dropout2d, (input_tensor,), input_tensor, p=0., training=training, inplace=inplace)
            if p < 0.0 or p > 1.0:
                raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
            return _VF.feature_dropout_(input_tensor, 0., training) if inplace else _VF.feature_dropout(input_tensor,
                                                                                                        0., training)

        def function_dropout3d(input_tensor: torch.Tensor, p: float = 0.5, training: bool = True,
                               inplace: bool = False) -> torch.Tensor:
            if has_torch_function_unary(input_tensor):
                return handle_torch_function(
                    function_dropout3d, (input_tensor,), input_tensor, p=0., training=training, inplace=inplace)
            if p < 0.0 or p > 1.0:
                raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
            return _VF.feature_dropout_(input_tensor, 0., training) if inplace else _VF.feature_dropout(input_tensor,
                                                                                                        0., training)

        F.dropout = function_dropout
        F.dropout2d = function_dropout2d
        F.dropout3d = function_dropout3d


# pylint: disable=huawei-redefined-outer-name, huawei-too-many-arguments
def fix_randomness(seed=123, is_gpu=False, deterministic=False, rm_dropout=False):
    print("Fix randomness")
    import random
    import numpy as np
    from packaging import version
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_version = torch.version.cuda
    if cuda_version is not None and version.parse(cuda_version) >= version.parse("10.2"):
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['HCCL_DETERMINISTIC'] = str(True)
    if deterministic:
        print("torch.use_deterministic_algorithms(True)")
        torch.use_deterministic_algorithms(True)
    
    if is_gpu:
        print("torch cuda manual seedall")
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enable = False
            torch.backends.cudnn.benchmark = False
    else:
        print("torch_npu manual seedall")
        torch_npu.npu.manual_seed_all(seed)
        torch_npu.npu.manual_seed(seed)
    
    if rm_dropout:
        print("remove dropout to fix randomness")
        remove_dropout()


def set_brake_at_step(patcher: Patcher, end_step: int = 1000):
    patcher.brake_at(end_step)


def set_profiling(patcher: Patcher, profiling_path: str, profiling_level: int = 0):
    patcher.with_profiling(profiling_path, profiling_level)
