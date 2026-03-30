# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
"""
DiffusionDrive NPU Migration Patches

Usage:
    from mx_driving.patcher import default_patcher
    from migrate_to_ascend.patch import configure_patcher

    configure_patcher(default_patcher, performance=False)
    default_patcher.apply()
"""
from __future__ import annotations

from typing import List

import torch
import torch_npu

import mx_driving
from mx_driving.patcher import (
    Patcher,
    Patch,
    AtomicPatch,
    TorchScatter,
    ResNetFP16,
    ResNetMaxPool,
)
from mx_driving.patcher.patch import with_imports


# =============================================================================
# DeformableAggregation NPU Implementation
# =============================================================================

class _DeformableAggregationFunction:
    """Wrapper that delegates to mx_driving's NPU implementation."""
    @staticmethod
    def apply(*args, **kwargs):
        return mx_driving.deformable_aggregation(*args, **kwargs)


# =============================================================================
# Public API - Configure Patcher
# =============================================================================

def configure_patcher(patcher: Patcher, performance: bool = False) -> Patcher:
    """
    Configure the patcher with DiffusionDrive-specific patches.

    Args:
        patcher: The Patcher instance to configure (typically default_patcher)
        performance: If True, enable performance mode (brake at 1000 steps)

    Returns:
        The configured patcher (for chaining)
    """
    # 1. Skip unavailable CUDA modules
    patcher.skip_import("flash_attn")
    patcher.skip_import("torch_scatter")

    # 2. Replace CUDA module with NPU implementation
    patcher.replace_import(
        "projects.mmdet3d_plugin.ops.deformable_aggregation",
        DeformableAggregationFunction=_DeformableAggregationFunction,
    )

    # 3. Inject missing imports
    patcher.inject_import("projects.mmdet3d_plugin.models.sparsedrive_v1", "V1SparseDrive", "projects.mmdet3d_plugin.models")
    patcher.inject_import("projects.mmdet3d_plugin.models.sparsedrive_head_v1", "V1SparseDriveHead", "projects.mmdet3d_plugin.models")
    patcher.inject_import("projects.mmdet3d_plugin.models.motion.motion_blocks_v11", "V11MotionPlanningRefinementModule", "projects.mmdet3d_plugin.models")
    patcher.inject_import("projects.mmdet3d_plugin.models.motion.motion_planning_head_v13", "V13MotionPlanningHead", "projects.mmdet3d_plugin.models")

    # 4. Add predefined patches
    patcher.add(TorchScatter)
    # default_patcher already enables ResNetMaxPool, which conflicts with ResNetFP16.
    # Switch explicitly so apply() won't fail on conflict checking.
    patcher.disable(ResNetMaxPool).add(ResNetFP16)

    # 5. Add project-specific patches
    patcher.add(FlashAttention)
    patcher.add(DeformableFeatureAggregation)
    patcher.add(SparseBox3DLoss)
    patcher.add(SparseBox3DTarget)
    patcher.add(SparsePoint3DTarget)
    patcher.add(MotionTarget)
    patcher.add(PlanningTarget)
    patcher.add(InstanceQueue)
    patcher.add(HcclBackend)

    # 6. Configure performance mode if requested
    if performance:
        patcher.brake_at(1000)

    return patcher


# =============================================================================
# Patch Classes
# =============================================================================

class FlashAttention(Patch):
    """Flash Attention NPU patch using npu_fusion_attention."""
    name = "diffusiondrive_flash_attention"

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.models.attention.FlashAttention.forward",
                cls._flash_attention_forward,
            ),
            AtomicPatch(
                "projects.mmdet3d_plugin.models.attention.FlashMHA.forward",
                cls._flash_mha_forward,
            ),
        ]

    @staticmethod
    @with_imports(
        ("projects.mmdet3d_plugin.models.attention", "auto_fp16"),
        "@auto_fp16(apply_to=('q', 'k', 'v'), out_fp32=True)",
    )
    def _flash_attention_forward(self, q, k, v, causal=False, key_padding_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, T, H, D)
            kv: The tensor containing the key, and value. (B, S, 2, H, D)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        if key_padding_mask is None:
            if self.softmax_scale:
                scale = self.softmax_scale
            else:
                scale = (q.shape[-1]) ** (-0.5)

            dropout_p = self.dropout_p if self.training else 0.0
            h = q.shape[-2]
            output = torch_npu.npu_fusion_attention(q, k, v, h,
                        input_layout="BSND",
                        pre_tockens=65536,
                        next_tockens=65536,
                        atten_mask=None,
                        scale=scale,
                        keep_prob=1. - dropout_p,
                        sync=False,
                        inner_precise=0)[0]
        else:
            pass
        return output, None

    @staticmethod
    @with_imports(("projects.mmdet3d_plugin.models.attention", "_in_projection_packed", "rearrange"))
    def _flash_mha_forward(self, q, k, v, key_padding_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)  # noqa: F821
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)  # noqa: F821
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)  # noqa: F821
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)  # noqa: F821

        context, attn_weights = self.inner_attn(q, k, v, key_padding_mask=key_padding_mask, causal=self.causal)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights  # noqa: F821


class DeformableFeatureAggregation(Patch):
    """Fix device placement in DeformableFeatureAggregation._get_weights."""
    name = "diffusiondrive_deformable_feature_aggregation"

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.models.blocks.DeformableFeatureAggregation._get_weights",
                cls._get_weights,
            ),
        ]

    @staticmethod
    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                )
            )
            feature = feature[:, :, None] + camera_embed[:, None]

        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand((bs, num_anchor, self.num_cams, 1, self.num_pts, 1),
                               device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights


class SparseBox3DLoss(Patch):
    """SparseBox3DLoss NPU compatibility patch."""
    name = "diffusiondrive_sparse_box3d_loss"

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.models.detection3d.losses.SparseBox3DLoss.forward",
                cls._forward,
            ),
        ]

    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    @staticmethod
    @with_imports(("projects.mmdet3d_plugin.models.detection3d.losses",
                   "SIN_YAW", "COS_YAW", "CNS", "YNS", "X", "Y", "Z"))
    def _forward(
        self,
        box,
        box_target,
        weight=None,
        avg_factor=None,
        prefix="",
        suffix="",
        quality=None,
        cls_target=None,
        **kwargs,
    ):
        # Some categories do not distinguish between positive and negative
        # directions. For example, barrier in nuScenes dataset.
        if self.cls_allow_reverse is not None and cls_target is not None:
            if_reverse = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]],  # noqa: F821
                    box[..., [SIN_YAW, COS_YAW]],  # noqa: F821
                    dim=-1,
                )
                < 0
            )
            if_reverse = (
                torch.isin(
                    cls_target, cls_target.new_tensor(self.cls_allow_reverse)
                )
                & if_reverse
            )
            box_target[..., [SIN_YAW, COS_YAW]] = torch.where(  # noqa: F821
                if_reverse[..., None],
                -box_target[..., [SIN_YAW, COS_YAW]],  # noqa: F821
                box_target[..., [SIN_YAW, COS_YAW]],  # noqa: F821
            )

        output = {}
        box_loss = self.loss_box(
            box, box_target, weight=weight, avg_factor=avg_factor
        )
        output[f"{prefix}loss_box{suffix}"] = box_loss

        if quality is not None:
            cns = quality[..., CNS]  # noqa: F821
            yns = quality[..., YNS].sigmoid()  # noqa: F821
            cns_target = torch.norm(
                box_target[..., [X, Y, Z]] - box[..., [X, Y, Z]], p=2, dim=-1  # noqa: F821
            )
            cns_target = torch.exp(-cns_target)
            cns_loss = self.loss_cns(cns, cns_target.detach(), avg_factor=avg_factor)
            output[f"{prefix}loss_cns{suffix}"] = cns_loss

            yns_target = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]],  # noqa: F821
                    box[..., [SIN_YAW, COS_YAW]],  # noqa: F821
                    dim=-1,
                )
                > 0
            )
            yns_target = yns_target.float()
            yns_loss = self.loss_yns(yns, yns_target, avg_factor=avg_factor)
            output[f"{prefix}loss_yns{suffix}"] = yns_loss
        return output


class SparseBox3DTarget(Patch):
    """SparseBox3DTarget NPU compatibility patch."""
    name = "diffusiondrive_sparse_box3d_target"

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.models.detection3d.target.SparseBox3DTarget.encode_reg_target",
                cls._encode_reg_target,
            ),
            AtomicPatch(
                "projects.mmdet3d_plugin.models.detection3d.target.SparseBox3DTarget._cls_cost",
                cls._cls_cost,
            ),
            AtomicPatch(
                "projects.mmdet3d_plugin.models.detection3d.target.SparseBox3DTarget._box_cost",
                cls._box_cost,
            ),
        ]

    @staticmethod
    @with_imports(("projects.mmdet3d_plugin.models.detection3d.target",
                   "X", "Y", "Z", "W", "L", "H", "YAW"))
    def _encode_reg_target(self, box_target, device=None):
        sizes = [box.shape[0] for box in box_target]
        boxes = torch.cat(box_target, dim=0)
        output = torch.cat(
            [
                boxes[..., [X, Y, Z]],  # noqa: F821
                boxes[..., [W, L, H]].log(),  # noqa: F821
                torch.sin(boxes[..., YAW]).unsqueeze(-1),  # noqa: F821
                torch.cos(boxes[..., YAW]).unsqueeze(-1),  # noqa: F821
                boxes[..., YAW + 1:],  # noqa: F821
            ],
            dim=-1,
        )
        if device is not None:
            output = output.to(device=device)
        outputs = torch.split(output, sizes, dim=0)
        return outputs

    @staticmethod
    def _cls_cost(self, cls_pred, cls_target):
        bs = cls_pred.shape[0]
        cls_pred = cls_pred.sigmoid()
        neg_cost = (
            -(1 - cls_pred + self.eps).log()
            * (1 - self.alpha)
            * cls_pred.pow(self.gamma)
        )
        pos_cost = (
            -(cls_pred + self.eps).log()
            * self.alpha
            * (1 - cls_pred).pow(self.gamma)
        )
        cost = (pos_cost - neg_cost) * self.cls_weight
        costs = []
        for i in range(bs):
            if len(cls_target[i]) > 0:
                costs.append(
                    cost[i, :, cls_target[i]]
                )
            else:
                costs.append(None)
        return costs

    @staticmethod
    def _box_cost(self, box_pred, box_target, instance_reg_weights):
        bs = box_pred.shape[0]
        cost = []
        weights = box_pred.new_tensor(self.reg_weights)
        for i in range(bs):
            if len(box_target[i]) > 0:
                cost.append(
                    torch.sum(
                        torch.abs(box_pred[i, :, None] - box_target[i][None])
                        * instance_reg_weights[i][None]
                        * weights,
                        dim=-1,
                    )
                    * self.box_weight
                )
            else:
                cost.append(None)
        return cost


class SparsePoint3DTarget(Patch):
    """SparsePoint3DTarget NPU compatibility patch."""
    name = "diffusiondrive_sparse_point3d_target"

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.models.map.target.SparsePoint3DTarget.__init__",
                cls._init,
            ),
            AtomicPatch(
                "projects.mmdet3d_plugin.models.map.target.SparsePoint3DTarget.normalize_line",
                cls._normalize_line,
            ),
        ]

    @staticmethod
    @with_imports(("projects.mmdet3d_plugin.models.map.target",
                   "SparsePoint3DTarget", "build_assigner"))
    def _init(
        self,
        assigner=None,
        num_dn_groups=0,
        dn_noise_scale=0.5,
        max_dn_gt=32,
        add_neg_dn=True,
        num_temp_dn_groups=0,
        num_cls=3,
        num_sample=20,
        roi_size=(30, 60),
    ):
        super(SparsePoint3DTarget, self).__init__(  # noqa: F821
            num_dn_groups, num_temp_dn_groups
        )
        self.assigner = build_assigner(assigner)  # noqa: F821
        self.dn_noise_scale = dn_noise_scale
        self.max_dn_gt = max_dn_gt
        self.add_neg_dn = add_neg_dn

        self.num_cls = num_cls
        self.num_sample = num_sample
        self.roi_size = roi_size
        self.origin = -torch.tensor([self.roi_size[0] / 2, self.roi_size[1] / 2]).npu()
        self.norm = torch.tensor([self.roi_size[0], self.roi_size[1]]).npu() + 1e-5

    @staticmethod
    def _normalize_line(self, line):
        if line.shape[0] == 0:
            return line

        line = line.view(line.shape[:-1] + (self.num_sample, -1))
        line = line - self.origin
        # transform from range [0, 1] to (0, 1)
        line = line / self.norm
        line = line.flatten(-2, -1)

        return line


class MotionTarget(Patch):
    """MotionTarget NPU compatibility patch."""
    name = "diffusiondrive_motion_target"

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.models.motion.target.MotionTarget.sample",
                cls._sample,
            ),
        ]

    # pylint: disable=too-many-return-values
    @staticmethod
    @with_imports(("projects.mmdet3d_plugin.models.motion.target", "get_cls_target"))
    def _sample(
        self,
        reg_pred,
        gt_reg_target,
        gt_reg_mask,
        motion_loss_cache,
    ):
        bs, num_anchor, mode, ts, d = reg_pred.shape
        reg_target = reg_pred.new_zeros((bs, num_anchor, ts, d))
        reg_weight = reg_pred.new_zeros((bs, num_anchor, ts))
        indices = motion_loss_cache['indices']
        num_pos = reg_pred.new_tensor([0])
        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(gt_reg_target[i]) == 0:
                continue
            reg_target[i, pred_idx] = gt_reg_target[i][target_idx]
            reg_weight[i, pred_idx] = gt_reg_mask[i][target_idx]
            num_pos += len(pred_idx)

        cls_target = get_cls_target(reg_pred, reg_target, reg_weight)  # noqa: F821
        cls_weight = reg_weight.any(dim=-1)
        best_reg = torch.gather(reg_pred, 2, cls_target[..., None, None, None].repeat(1, 1, 1, ts, d)).squeeze(2)

        return cls_target, cls_weight, best_reg, reg_target, reg_weight, num_pos


class PlanningTarget(Patch):
    """PlanningTarget NPU compatibility patch."""
    name = "diffusiondrive_planning_target"

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.models.motion.target.PlanningTarget.sample",
                cls._sample,
            ),
        ]

    # pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
    @staticmethod
    @with_imports(("projects.mmdet3d_plugin.models.motion.target", "get_cls_target"))
    def _sample(
        self,
        cls_pred,
        reg_pred,
        gt_reg_target,
        gt_reg_mask,
        data,
    ):
        gt_reg_target = gt_reg_target.unsqueeze(1)
        gt_reg_mask = gt_reg_mask.unsqueeze(1)

        bs = reg_pred.shape[0]
        bs_indices = torch.arange(bs, device=reg_pred.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)

        cls_pred = cls_pred.reshape(bs, 3, 1, self.ego_fut_mode)
        reg_pred = reg_pred.reshape(bs, 3, 1, self.ego_fut_mode, self.ego_fut_ts, 2)
        cls_pred = cls_pred[bs_indices, cmd]
        reg_pred = reg_pred[bs_indices, cmd]
        cls_target = get_cls_target(reg_pred, gt_reg_target, gt_reg_mask)  # noqa: F821
        cls_weight = gt_reg_mask.any(dim=-1)
        best_reg = torch.gather(reg_pred, 2, cls_target[..., None, None, None].repeat(1, 1, 1, self.ego_fut_ts, 2)).squeeze(2)

        return cls_pred, cls_target, cls_weight, best_reg, gt_reg_target, gt_reg_mask


class InstanceQueue(Patch):
    """InstanceQueue.prepare_motion NPU compatibility patch."""
    name = "diffusiondrive_instance_queue"

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "projects.mmdet3d_plugin.models.motion.instance_queue.InstanceQueue.prepare_motion",
                cls._prepare_motion,
            ),
        ]

    @staticmethod
    def _prepare_motion(
        self,
        det_output,
        mask,
    ):
        instance_feature = det_output["instance_feature"]
        det_anchors = det_output["prediction"][-1]

        if self.period is None:
            self.period = instance_feature.new_zeros(instance_feature.shape[:2]).long()
        else:
            instance_id = det_output['instance_id']
            prev_instance_id = self.prev_instance_id
            match = instance_id[..., None] == prev_instance_id[:, None]
            if self.tracking_threshold > 0:
                temp_mask = self.prev_confidence > self.tracking_threshold
                match = match * temp_mask.unsqueeze(1)

            # pylint: disable=consider-using-enumerate
            for i in range(len(self.instance_feature_queue)):
                temp_feature = self.instance_feature_queue[i]
                temp_feature = torch.matmul(match.type_as(temp_feature), temp_feature)
                self.instance_feature_queue[i] = temp_feature

                temp_anchor = self.anchor_queue[i]
                temp_anchor = torch.matmul(match.type_as(temp_anchor), temp_anchor)
                self.anchor_queue[i] = temp_anchor

            self.period = (
                match * self.period[:, None]
            ).sum(dim=2)

        self.instance_feature_queue.append(instance_feature.detach())
        self.anchor_queue.append(det_anchors.detach())
        self.period += 1

        if len(self.instance_feature_queue) > self.queue_length:
            self.instance_feature_queue.pop(0)
            self.anchor_queue.pop(0)
        self.period = torch.clip(self.period, 0, self.queue_length)


class HcclBackend(Patch):
    """Replace NCCL with HCCL for distributed training."""
    name = "diffusiondrive_hccl_backend"

    @classmethod
    def patches(cls, options=None) -> List[AtomicPatch]:
        return [
            AtomicPatch(
                "mmcv.runner.init_dist",
                cls._init_dist,
                precheck=cls._precheck,
            ),
        ]

    @staticmethod
    def _precheck():
        try:
            import importlib
            module = importlib.import_module("mmcv.runner")
            return hasattr(module, "dist_utils")
        except ImportError:
            return False

    @staticmethod
    @with_imports(("mmcv.runner.dist_utils",
                   "mp", "_init_dist_pytorch", "_init_dist_mpi", "_init_dist_slurm"))
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
