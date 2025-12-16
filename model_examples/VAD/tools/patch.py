# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import os
from types import ModuleType
from typing import Dict
import warnings
import torch
import torch_npu
from mx_driving.patcher import PatcherBuilder, Patch
from mx_driving.patcher import index,  numpy_type, ddp, stream,msda
from mx_driving.patcher import resnet_add_relu, resnet_maxpool
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mx_driving import multi_scale_deformable_attn


bev_mask_global = torch.tensor([]).npu()
indexes_global = None
max_len_global = None
bev_mask_id_global = -1
count_global = None


def spatial_cross_attention(spatial_cross_attention_module: ModuleType, options: Dict):
    force_fp32=spatial_cross_attention_module.force_fp32

    # pylint: disable=huawei-too-many-arguments
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

    # pylint: disable=huawei-too-many-arguments
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


    if hasattr(spatial_cross_attention_module, 'SpatialCrossAttention'):
        spatial_cross_attention_module.SpatialCrossAttention.forward = sca_forward
    else:
        raise AttributeError('SpatialCrossAttention attr not found')

    if hasattr(spatial_cross_attention_module, 'MSDeformableAttention3D'):
        spatial_cross_attention_module.MSDeformableAttention3D.forward = msda3d_forward
    else:
        raise AttributeError('MSDeformableAttention3D attr not found')


def npu_fusion_attention(mmcv: ModuleType, options: Dict):
    @ATTENTION.register_module(name='MultiheadAttention', force=True)
    class NpuFlashAttention(MultiheadAttention):
        # pylint: disable=dangerous-default-value
        def __init__(self,
                     embed_dims,
                     num_heads,
                     attn_drop=0.,
                     proj_drop=0.,
                     dropout_layer=dict(type='Dropout', drop_prob=0.),
                     init_cfg=None,
                     batch_first=True,
                     **kwargs):
            super().__init__(
                embed_dims,
                num_heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                dropout_layer=dropout_layer,
                init_cfg=init_cfg,
                batch_first=batch_first,
                **kwargs
            )
            self.attn = torch.nn.Sequential()
        # pylint: disable=huawei-too-many-arguments
        def forward(self,
                    query,
                    key=None,
                    value=None,
                    identity=None,
                    query_pos=None,
                    key_pos=None,
                    attn_mask=None,
                    key_padding_mask=None,
                    **kwargs):
            if key is None:
                key = query
            if value is None:
                value = key
            if identity is None:
                identity = query
            if key_pos is None:
                if query_pos is not None:
                    # use query_pos if key_pos is not available
                    if query_pos.shape == key.shape:
                        key_pos = query_pos
                    else:
                        warnings.warn(f'position encoding of key is'
                                      f'missing in {self.__class__.__name__}.')
            if query_pos is not None:
                query = query + query_pos
            if key_pos is not None:
                key = key + key_pos

            if not self.batch_first:
                query = query.transpose(0, 1)
                key = key.transpose(0, 1)
                value = value.transpose(0, 1)
            
            out = torch_npu.npu_fusion_attention(
                    query = query,
                    key = key,
                    value = value,
                    head_num = self.num_heads,
                    atten_mask = attn_mask,
                    sparse_mode = 1,
                    input_layout = 'BSH')[0]

            if not self.batch_first:
                out = out.transpose(0, 1)

            return identity + self.dropout_layer(self.proj_drop(out))

    if hasattr(mmcv.cnn.bricks.transformer,"MultiheadAttention"):
        mmcv.cnn.bricks.transformer.MultiheadAttention = NpuFlashAttention
    else:
        raise AttributeError("In mmcv.cnn.bricks.transformer,MultiheadAttention  not found")


def temporal_self_attention(temporal_self_attention_module: ModuleType, options: Dict):
    # pylint: disable=huawei-too-many-arguments
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

        # output shape (bs*num_bev_queue, num_query, embed_dims)->output shape (bs,num_bev_queue, num_query, embed_dims)，并在在bev_queue维度平均
        output = output.view(bs,self.num_bev_queue,num_query, embed_dims).mean(dim=1)
        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

    if hasattr(temporal_self_attention_module, 'TemporalSelfAttention'):
        temporal_self_attention_module.TemporalSelfAttention.forward = forward
    else:
        raise AttributeError('TemporalSelfAttention attr not found')


# get the patch for SparseDrive, and determine whether to brake advance
def generate_patcher_builder():
    vad_patcher_builder = (
        PatcherBuilder()
        .add_module_patch("mmcv", Patch(msda),Patch(npu_fusion_attention))
        .add_module_patch("torch", Patch(index))
        .add_module_patch("numpy", Patch(numpy_type))
        .add_module_patch("mmcv", Patch(stream), Patch(ddp))
        .add_module_patch("mmdet", Patch(resnet_add_relu), Patch(resnet_maxpool))
        .add_module_patch("projects.mmdet3d_plugin.VAD.modules.spatial_cross_attention", Patch(spatial_cross_attention))
        .add_module_patch('projects.mmdet3d_plugin.VAD.modules.temporal_self_attention', Patch(temporal_self_attention))
    )
    if os.environ.get("VAD_PERFORMANCE_FLAG"):
        vad_patcher_builder.brake_at(800)
    return vad_patcher_builder