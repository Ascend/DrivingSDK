# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os
import sys
from types import ModuleType
from typing import Dict
import torch
import torch_npu
import mmcv
import mmcv.runner
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import mx_driving
from mx_driving.patcher import PatcherBuilder, Patch
from mx_driving.patcher import index, batch_matmul, numpy_type, ddp, stream
from mx_driving.patcher import resnet_add_relu, resnet_maxpool

sys.path.append("..")



def models_blocks(models: ModuleType, options: Dict):
    from typing import List
    from mx_driving import npu_deformable_aggregation as DAF

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
            # change code
            # change the rand ops to generate on npu
            mask = torch.rand((bs, num_anchor, self.num_cams, 1, self.num_pts, 1),
                               device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )

        # change code
        projection_mat = projection_mat[:, :, None, None].contiguous()
        pts_extend = pts_extend[:, None, ..., None].contiguous()
        points_2d = []
        for i in range(4):
            temp = ((projection_mat[:, :, :, :, i, :].unsqueeze(-1)) * pts_extend).squeeze(-1).sum(dim=-1)
            points_2d.append(temp)
        points_2d = torch.stack(points_2d, dim=-1)

        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d
    
    if hasattr(models, "DeformableFeatureAggregation"):
        models.DeformableFeatureAggregation._get_weights = _get_weights
        models.DeformableFeatureAggregation.project_points = project_points


def detection_blocks(detection3d_blocks: ModuleType, options: Dict):
    SIN_YAW = detection3d_blocks.SIN_YAW
    COS_YAW = detection3d_blocks.COS_YAW
    VX = detection3d_blocks.VX
    W, L, H = detection3d_blocks.W, detection3d_blocks.L, detection3d_blocks.H
    X, Y, Z = detection3d_blocks.X, detection3d_blocks.Y, detection3d_blocks.Z

    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def refine_forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor = 1.0,
        return_cls=True,
    ):
        feature = instance_feature + anchor_embed
        output = self.layers(feature)
        # change code
        # + -> +=
        output[..., self.refine_state] += anchor[..., self.refine_state]
        if self.normalize_yaw:
            output[..., [SIN_YAW, COS_YAW]] = torch.nn.functional.normalize(
                output[..., [SIN_YAW, COS_YAW]], dim=-1
            )
        if self.output_dim > 8:
            if not isinstance(time_interval, torch.Tensor):
                time_interval = instance_feature.new_tensor(time_interval)
            translation = torch.transpose(output[..., VX:], 0, -1)
            velocity = torch.transpose(translation / time_interval, 0, -1)
            output[..., VX:] = velocity + anchor[..., VX:]

        if return_cls:
            assert self.with_cls_branch, "Without classification layers !!!"
            cls = self.cls_layers(instance_feature)
        else:
            cls = None
        if return_cls and self.with_quality_estimation:
            quality = self.quality_layers(feature)
        else:
            quality = None
        return output, cls, quality

    def keypoint_forward(
            self,
            anchor,
            instance_feature=None,
            T_cur2temp_list=None,
            cur_timestamp=None,
            temp_timestamps=None,
        ):
        bs, num_anchor = anchor.shape[:2]
        size = anchor[..., None, [W, L, H]].exp()
        key_points = self.fix_scale * size
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (
                self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3)
                .sigmoid()
                - 0.5
            )
            key_points = torch.cat(
                [key_points, learnable_scale * size], dim=-2
            )

        rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3])

        # change code to reduce slice ops
        cos_yaw = anchor[:, :, COS_YAW]
        sin_yaw = anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 0, 0] = cos_yaw
        rotation_mat[:, :, 0, 1] = -sin_yaw
        rotation_mat[:, :, 1, 0] = sin_yaw
        rotation_mat[:, :, 1, 1] = cos_yaw
        rotation_mat[:, :, 2, 2] = 1

        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None]
        ).squeeze(-1)
        key_points = key_points + anchor[..., None, [X, Y, Z]]

        if (
            cur_timestamp is None
            or temp_timestamps is None
            or T_cur2temp_list is None
            or len(temp_timestamps) == 0
        ):
            return key_points

        temp_key_points_list = []
        velocity = anchor[..., VX:]
        for i, t_time in enumerate(temp_timestamps):
            time_interval = cur_timestamp - t_time
            translation = (
                velocity
                * time_interval.to(dtype=velocity.dtype)[:, None, None]
            )
            temp_key_points = key_points - translation[:, :, None]
            T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype)
            temp_key_points = (
                T_cur2temp[:, None, None, :3]
                @ torch.cat(
                    [
                        temp_key_points,
                        torch.ones_like(temp_key_points[..., :1]),
                    ],
                    dim=-1,
                ).unsqueeze(-1)
            )
            temp_key_points = temp_key_points.squeeze(-1)
            temp_key_points_list.append(temp_key_points)
        return key_points, temp_key_points_list

    @staticmethod
    def anchor_projection(
        anchor,
        T_src2dst_list,
        src_timestamp=None,
        dst_timestamps=None,
        time_intervals=None,
    ):
        dst_anchors = []
        for i in range(len(T_src2dst_list)):
            vel = anchor[..., VX:]
            vel_dim = vel.shape[-1]
            T_src2dst = torch.unsqueeze(
                T_src2dst_list[i].to(dtype=anchor.dtype), dim=1
            )

            center = anchor[..., [X, Y, Z]]
            if time_intervals is not None:
                time_interval = time_intervals[i]
            elif src_timestamp is not None and dst_timestamps is not None:
                time_interval = (src_timestamp - dst_timestamps[i]).to(
                    dtype=vel.dtype
                )
            else:
                time_interval = None
            if time_interval is not None:
                translation = vel.transpose(0, -1) * time_interval
                translation = translation.transpose(0, -1)
                center = center - translation
            center = (
                torch.matmul(
                    T_src2dst[..., :3, :3], center[..., None]
                ).squeeze(dim=-1)
                + T_src2dst[..., :3, 3]
            )

            size = anchor[..., [W, L, H]]

            # change code
            yaw_tmp = []
            for j in range(2):
                temp = ((T_src2dst[..., j, :2].unsqueeze(-1)) * anchor[..., [COS_YAW, SIN_YAW], None]).squeeze(-1).sum(dim=-1)
                yaw_tmp.append(temp)
            yaw = torch.stack(yaw_tmp, dim=-1)

            vel_tmp = []
            for j in range(vel_dim):
                temp = ((T_src2dst[..., j, :vel_dim].unsqueeze(-1)) * vel[..., None]).squeeze(-1).sum(dim=-1)
                vel_tmp.append(temp)
            vel = torch.stack(vel_tmp, dim=-1)

            dst_anchor = torch.cat([center, size, yaw, vel], dim=-1)
            dst_anchors.append(dst_anchor)
        return dst_anchors

    if hasattr(detection3d_blocks, "SparseBox3DRefinementModule"):
        detection3d_blocks.SparseBox3DRefinementModule.forward = refine_forward
    if hasattr(detection3d_blocks, "SparseBox3DKeyPointsGenerator"):
        detection3d_blocks.SparseBox3DKeyPointsGenerator.forward = keypoint_forward
    if hasattr(detection3d_blocks, "SparseBox3DKeyPointsGenerator"):
        detection3d_blocks.SparseBox3DKeyPointsGenerator.anchor_projection = anchor_projection


def detection_losses(losses: ModuleType, options: Dict):
    SIN_YAW, COS_YAW = losses.SIN_YAW, losses.COS_YAW
    CNS, YNS = losses.CNS, losses.YNS
    X, Y, Z = losses.X, losses.Y, losses.Z

    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def losses_forward(
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
                    box_target[..., [SIN_YAW, COS_YAW]],
                    box[..., [SIN_YAW, COS_YAW]],
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
            box_target[..., [SIN_YAW, COS_YAW]] = torch.where(
                if_reverse[..., None],
                -box_target[..., [SIN_YAW, COS_YAW]],
                box_target[..., [SIN_YAW, COS_YAW]],
            )

        output = {}
        box_loss = self.loss_box(
            box, box_target, weight=weight, avg_factor=avg_factor
        )
        output[f"{prefix}loss_box{suffix}"] = box_loss

        if quality is not None:
            cns = quality[..., CNS]
            yns = quality[..., YNS].sigmoid()
            cns_target = torch.norm(
                box_target[..., [X, Y, Z]] - box[..., [X, Y, Z]], p=2, dim=-1
            )
            cns_target = torch.exp(-cns_target)
            # change code
            # add detach to cns_target to avoid the training error
            cns_loss = self.loss_cns(cns, cns_target.detach(), avg_factor=avg_factor)
            output[f"{prefix}loss_cns{suffix}"] = cns_loss

            yns_target = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]],
                    box[..., [SIN_YAW, COS_YAW]],
                    dim=-1,
                )
                > 0
            )
            yns_target = yns_target.float()
            yns_loss = self.loss_yns(yns, yns_target, avg_factor=avg_factor)
            output[f"{prefix}loss_yns{suffix}"] = yns_loss
        return output

    if hasattr(losses, "SparseBox3DLoss"):
        losses.SparseBox3DLoss.forward = losses_forward


def detection_target(target: ModuleType, options: Dict):
    X, Y, Z = target.X, target.Y, target.Z
    W, L, H = target.W, target.L, target.H
    YAW = target.YAW

    def encode_reg_target(self, box_target, device=None):
        sizes = [box.shape[0] for box in box_target]
        # change code
        # concat the box_target to reduce the free time
        boxes = torch.cat(box_target, dim=0)
        yaw = boxes[..., YAW]
        output = torch.cat(
            [
                boxes[..., [X, Y, Z]],
                boxes[..., [W, L, H]].log(),
                torch.sin(yaw).unsqueeze(-1),
                torch.cos(yaw).unsqueeze(-1),
                boxes[..., YAW + 1:],
            ],
            dim=-1,
        )
        if device is not None:
            output = output.to(device=device, non_blocking=True)
        outputs = torch.split(output, sizes, dim=0)
        return outputs

    def _cls_cost(self, cls_pred, cls_target):
        bs = cls_pred.shape[0]
        cls_pred = cls_pred.sigmoid()
        # change code
        # extract the common parts to reduce the free time
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
    
    def _box_cost(self, box_pred, box_target, instance_reg_weights):
        bs = box_pred.shape[0]
        cost = []
        # change code
        # advance the weights generate to reduce free time
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

    def get_dn_anchors(self, cls_target, box_target, gt_instance_id=None):
        if self.num_dn_groups <= 0:
            return None
        if self.num_temp_dn_groups <= 0:
            gt_instance_id = None

        if self.max_dn_gt > 0:
            cls_target = [x[: self.max_dn_gt] for x in cls_target]
            box_target = [x[: self.max_dn_gt] for x in box_target]
            if gt_instance_id is not None:
                gt_instance_id = [x[: self.max_dn_gt] for x in gt_instance_id]

        max_dn_gt = max([len(x) for x in cls_target])
        if max_dn_gt == 0:
            return None
        
        box_target = self.encode_reg_target(box_target)

        list_cls_t = []
        list_box_t = []
        
        for cls_t, box_t in zip(cls_target, box_target):
            list_cls_t.append(F.pad(cls_t, (0, max_dn_gt - cls_t.shape[0]), value=-1))
            list_box_t.append(F.pad(box_t, (0, 0, 0, max_dn_gt - box_t.shape[0])))
            
        cls_target = torch.stack(list_cls_t)
        box_target = torch.stack(list_box_t)
        box_target = torch.where(
            cls_target[..., None] == -1, box_target.new_tensor(0), box_target
        )
        if gt_instance_id is not None:
            gt_instance_id = torch.stack(
                [
                    F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                    for x in gt_instance_id
                ]
            )

        bs, num_gt, state_dims = box_target.shape
        if self.num_dn_groups > 1:
            cls_target = cls_target.tile(self.num_dn_groups, 1)
            box_target = box_target.tile(self.num_dn_groups, 1, 1)
            if gt_instance_id is not None:
                gt_instance_id = gt_instance_id.tile(self.num_dn_groups, 1)

        noise = torch.rand_like(box_target) * 2 - 1
        noise *= box_target.new_tensor(self.dn_noise_scale)
        dn_anchor = box_target + noise
        if self.add_neg_dn:
            noise_neg = torch.rand_like(box_target) + 1
            flag = torch.where(
                torch.rand_like(box_target) > 0.5,
                noise_neg.new_tensor(1),
                noise_neg.new_tensor(-1),
            )
            noise_neg *= flag
            noise_neg *= box_target.new_tensor(self.dn_noise_scale)
            dn_anchor = torch.cat([dn_anchor, box_target + noise_neg], dim=1)
            num_gt *= 2

        box_cost = self._box_cost(
            dn_anchor, box_target, torch.ones_like(box_target)
        )
        dn_box_target = torch.zeros_like(dn_anchor)
        dn_cls_target = -torch.ones_like(cls_target) * 3
        if gt_instance_id is not None:
            dn_id_target = -torch.ones_like(gt_instance_id)
        if self.add_neg_dn:
            dn_cls_target = torch.cat([dn_cls_target, dn_cls_target], dim=1)
            if gt_instance_id is not None:
                dn_id_target = torch.cat([dn_id_target, dn_id_target], dim=1)

        cost_cpu = [tensor.cpu() for tensor in box_cost]
        for i in range(dn_anchor.shape[0]):
            anchor_idx, gt_idx = linear_sum_assignment(cost_cpu[i])
            dn_box_target[i, anchor_idx] = box_target[i, gt_idx]
            dn_cls_target[i, anchor_idx] = cls_target[i, gt_idx]
            if gt_instance_id is not None:
                dn_id_target[i, anchor_idx] = gt_instance_id[i, gt_idx]

        dn_anchor = (
            dn_anchor.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .contiguous()
            .view(bs, self.num_dn_groups * num_gt, state_dims)
        )
        dn_box_target = (
            dn_box_target.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .contiguous()
            .view(bs, self.num_dn_groups * num_gt, state_dims)
        )
        dn_cls_target = (
            dn_cls_target.reshape(self.num_dn_groups, bs, num_gt)
            .permute(1, 0, 2)
            .contiguous()
            .view(bs, self.num_dn_groups * num_gt)
        )
        if gt_instance_id is not None:
            dn_id_target = (
                dn_id_target.reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .contiguous()
                .view(bs, self.num_dn_groups * num_gt)
            )
        else:
            dn_id_target = None
        valid_mask = dn_cls_target >= 0
        if self.add_neg_dn:
            cls_target = (
                torch.cat([cls_target, cls_target], dim=1)
                .reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .contiguous()
                .view(bs, self.num_dn_groups * num_gt)
            )
            valid_mask = torch.logical_or(
                valid_mask, ((cls_target >= 0) & (dn_cls_target == -3))
            )  # valid denotes the items is not from pad.
        attn_mask = dn_box_target.new_ones(
            num_gt * self.num_dn_groups, num_gt * self.num_dn_groups
        )
        for i in range(self.num_dn_groups):
            start = num_gt * i
            end = start + num_gt
            attn_mask[start:end, start:end] = 0
        attn_mask = attn_mask == 1
        dn_cls_target = dn_cls_target.long()
        return (
            dn_anchor,
            dn_box_target,
            dn_cls_target,
            attn_mask,
            valid_mask,
            dn_id_target,
        )

    if hasattr(target, "SparseBox3DTarget"):
        target.SparseBox3DTarget._cls_cost = _cls_cost
    
    if hasattr(target, "SparseBox3DTarget"):
        target.SparseBox3DTarget._box_cost = _box_cost
    
    if hasattr(target, "SparseBox3DTarget"):
        target.SparseBox3DTarget.encode_reg_target = encode_reg_target
    
    if hasattr(target, "SparseBox3DTarget"):
        target.SparseBox3DTarget.get_dn_anchors = get_dn_anchors


def mmcv_optimizer(hooks: ModuleType, options: Dict):
    def clip_grads(self, params, runner):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            # change code
            # used fused grad_clip
            return runner.optimizer.clip_grad_norm_fused_(**self.grad_clip)
        return 0

    def after_train_iter(self, runner) -> None:
        # clear grads of last iteration

        # change code
        # remove the runner.model.zero_grad() to avoid grad_norm compute error
        runner.optimizer.zero_grad()

        self.loss_scaler.scale(runner.outputs['loss']).backward()
        self.loss_scaler.unscale_(runner.optimizer)
        # grad clip
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters(), runner)
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                            runner.outputs['num_samples'])
        # backward and update scaler
        self.loss_scaler.step(runner.optimizer)
        self.loss_scaler.update(self._scale_update_param)

        # save state_dict of loss_scaler
        runner.meta.setdefault(
            'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

    if hasattr(hooks, "OptimizerHook"):
        hooks.OptimizerHook.clip_grads = clip_grads
    
    if hasattr(hooks, "Fp16OptimizerHook"):
        hooks.Fp16OptimizerHook.after_train_iter = after_train_iter


def get_hccl_init_dist(runner: ModuleType):
    module = importlib.import_module(runner)

    mp = module.dist_utils.mp
    _init_dist_pytorch = module.dist_utils._init_dist_pytorch
    _init_dist_mpi = module.dist_utils._init_dist_mpi
    _init_dist_slurm = module.dist_utils._init_dist_slurm

    def hccl_init_dist(launcher: str, backend: str = 'nccl', **kwargs) -> None:
        # change code
        # use hccl as backend
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

    return hccl_init_dist


def get_fused_optimizer(optimizer: ModuleType):
    module = importlib.import_module(optimizer)
    copy = module.optimizer.builder.copy
    build_optimizer_constructor = module.optimizer.builder.build_optimizer_constructor
    
    def build_optimizer(model, cfg: Dict):
        optimizer_cfg = copy.deepcopy(cfg)
        # change code
        # use NpuFused optimizer
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

    return build_optimizer


def graph_mode(target: ModuleType, options: Dict):
    import torchair
    from mmcv.utils import build_from_cfg
    from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
    from mmdet.models import (
        DETECTORS,
        BaseDetector,
        build_backbone,
        build_head,
        build_neck,
    )
    
    from projects.mmdet3d_plugin.models.grid_mask import GridMask
    from projects.mmdet3d_plugin.models.sparse4d import Sparse4D

    try:
        from projects.mmdet3d_plugin.ops import feature_maps_format
        DAF_VALID = True
    except:
        DAF_VALID = False

    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
        ):
        super(Sparse4D, self).__init__(init_cfg=init_cfg)
        if pretrained is not None:
            img_backbone.pretrained = pretrained
        self.img_backbone = build_backbone(img_backbone)
        
        # change code
        # use graph mode
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        self.img_backbone = torch.compile(self.img_backbone, backend=npu_backend, dynamic=True)

        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        self.head = build_head(head)
        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            if not DAF_VALID:
                raise RuntimeError("deformable_aggregation needs to be set up.")
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )
    if hasattr(target, "Sparse4D"):
        Sparse4D.__init__ = __init__


def sparse4d_head(target: ModuleType, options: dict):
    from typing import List, Union
    from projects.mmdet3d_plugin.models.sparse4d_head import Sparse4DHead
    from mmcv.runner import BaseModule, force_fp32
    from mmdet.core import reduce_mean

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
        ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        # ========= get instance info ============
        if (
            self.sampler.dn_metas is not None
            and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
            self.sampler.dn_metas = None
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
        ) = self.instance_bank.get(
            batch_size, metas, dn_metas=self.sampler.dn_metas
        )

        # ========= prepare for denosing training ============
        # 1. get dn metas: noisy-anchors and corresponding GT
        # 2. concat learnable instances and noisy instances
        # 3. get attention mask
        attn_mask = None
        dn_metas = None
        temp_dn_reg_target = None
        if self.training and hasattr(self.sampler, "get_dn_anchors"):
            if "instance_id" in metas["img_metas"][0]:
                gt_instance_id = [
                    torch.from_numpy(x["instance_id"]).cuda()
                    for x in metas["img_metas"]
                ]
            else:
                gt_instance_id = None
            dn_metas = self.sampler.get_dn_anchors(
                metas[self.gt_cls_key],
                metas[self.gt_reg_key],
                gt_instance_id,
            )
        if dn_metas is not None:
            (
                dn_anchor,
                dn_reg_target,
                dn_cls_target,
                dn_attn_mask,
                valid_mask,
                dn_id_target,
            ) = dn_metas
            num_dn_anchor = dn_anchor.shape[1]
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_anchor, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            instance_feature = torch.cat(
                [
                    instance_feature,
                    instance_feature.new_zeros(
                        batch_size, num_dn_anchor, instance_feature.shape[-1]
                    ),
                ],
                dim=1,
            )
            num_instance = instance_feature.shape[1]
            num_free_instance = num_instance - num_dn_anchor
            attn_mask = anchor.new_ones(
                (num_instance, num_instance), dtype=torch.bool
            )
            attn_mask[:num_free_instance, :num_free_instance] = False
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask

        anchor_embed = self.anchor_encoder(anchor)
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        # =================== forward the layers ====================
        prediction = []
        classification = []
        quality = []
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=attn_mask
                    if temp_instance_feature is None
                    else None,
                )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    metas,
                )
            elif op == "refine":
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        self.training
                        or len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                if len(prediction) == self.num_single_frame_decoder:
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor, cls
                    )
                    if (
                        dn_metas is not None
                        and self.sampler.num_temp_dn_groups > 0
                        and dn_id_target is not None
                    ):
                        (
                            instance_feature,
                            anchor,
                            temp_dn_reg_target,
                            temp_dn_cls_target,
                            temp_valid_mask,
                            dn_id_target,
                        ) = self.sampler.update_dn(
                            instance_feature,
                            anchor,
                            dn_reg_target,
                            dn_cls_target,
                            valid_mask,
                            dn_id_target,
                            self.instance_bank.num_anchor,
                            self.instance_bank.mask,
                        )
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}

        # change code to merge for loops
        # split predictions of learnable instances and noisy instances
        if dn_metas is not None:
            dn_classification = []
            new_classification = []
            dn_prediction = []
            new_prediction = []
            new_quality = []

            for cls, pred, q in zip(classification, prediction, quality):

                dn_classification.append(cls[:, num_free_instance:])
                new_classification.append(cls[:, :num_free_instance])

                dn_prediction.append(pred[:, num_free_instance:])
                new_prediction.append(pred[:, :num_free_instance])

                if q is not None:
                    new_quality.append(q[:, :num_free_instance])
                else:
                    new_quality.append(None)

            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask,
                        "dn_id_target": dn_id_target,
                    }
                )
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask
            dn_instance_feature = instance_feature[:, num_free_instance:]
            dn_anchor = anchor[:, num_free_instance:]
            instance_feature = instance_feature[:, :num_free_instance]
            anchor = anchor[:, :num_free_instance]
            cls = cls[:, :num_free_instance]

            # cache dn_metas for temporal denoising
            self.sampler.cache_dn(
                dn_instance_feature,
                dn_anchor,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )
            classification = new_classification
            prediction = new_prediction
            quality = new_quality

        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
            }
        )

        # cache current instances for temporal modeling
        self.instance_bank.cache(
            instance_feature, anchor, cls, metas, feature_maps
        )
        if not self.training:
            instance_id = self.instance_bank.get_instance_id(
                cls, anchor, self.decoder.score_threshold
            )
            output["instance_id"] = instance_id
        return output

    @force_fp32(apply_to=("model_outs"))
    def loss(self, model_outs, data, feature_maps=None):
        # ===================== prediction losses ======================
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        output = {}
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):
            reg = reg[..., : len(self.reg_weights)]
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls,
                reg,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            # change code
            res = reduce_mean(torch.sum(mask).to(dtype=reg.dtype))
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            num_pos = max(res, 1.0)
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)
            cls_target = cls_target[mask]
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]

            reg_loss = self.loss_reg(
                reg,
                reg_target,
                weight=reg_weights,
                avg_factor=num_pos,
                suffix=f"_{decoder_idx}",
                quality=qt,
                cls_target=cls_target,
            )

            output[f"loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)

        if "dn_prediction" not in model_outs:
            return output

        # ===================== denoising losses ======================
        dn_cls_scores = model_outs["dn_classification"]
        dn_reg_preds = model_outs["dn_prediction"]

        (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        ) = self.prepare_for_dn_loss(model_outs)
        for decoder_idx, (cls, reg) in enumerate(
            zip(dn_cls_scores, dn_reg_preds)
        ):
            if (
                "temp_dn_valid_mask" in model_outs
                and decoder_idx == self.num_single_frame_decoder
            ):
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")

            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
            reg_loss = self.loss_reg(
                reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                    ..., : len(self.reg_weights)
                ],
                dn_reg_target,
                avg_factor=num_dn_pos,
                weight=reg_weights,
                suffix=f"_dn_{decoder_idx}",
            )
            output[f"loss_cls_dn_{decoder_idx}"] = cls_loss
            output.update(reg_loss)
        return output
    if hasattr(target, "Sparse4DHead"):
        Sparse4DHead.forward = forward
        Sparse4DHead.loss = loss


def generate_patcher_builder():
    sparse4d_patcher_builder = (
        PatcherBuilder()
        .add_module_patch("torch", Patch(index), Patch(batch_matmul))
        .add_module_patch("numpy", Patch(numpy_type))
        .add_module_patch("mmcv", Patch(ddp), Patch(stream))
        .add_module_patch("mmdet", Patch(resnet_add_relu), Patch(resnet_maxpool))
        .add_module_patch("projects.mmdet3d_plugin.models.detection3d.detection3d_blocks", Patch(detection_blocks))
        .add_module_patch("projects.mmdet3d_plugin.models.detection3d.losses", Patch(detection_losses))
        .add_module_patch("projects.mmdet3d_plugin.models.blocks", Patch(models_blocks))
        .add_module_patch("projects.mmdet3d_plugin.models.detection3d.target", Patch(detection_target))
        .add_module_patch("projects.mmdet3d_plugin.models.sparse4d_head", Patch(sparse4d_head))
        .add_module_patch("mmcv.runner.hooks.optimizer", Patch(mmcv_optimizer))
    )
    if os.environ.get("SPARSE4D_PERFORMANCE_FLAG"):
        sparse4d_patcher_builder.brake_at(500)
    return sparse4d_patcher_builder


# change the backend and optimizer used in Sparse4D
def _init():
    mmcv.runner.init_dist = get_hccl_init_dist('mmcv.runner')
    mmcv.runner.build_optimizer = get_fused_optimizer('mmcv.runner')


# the function need to replace before their import
_init()