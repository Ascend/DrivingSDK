# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os
import sys
import types
from types import ModuleType
from typing import Dict
import importlib

import mmcv
import mmcv.runner
import torch
import torch_npu

import mx_driving
from mx_driving import deformable_aggregation
from mx_driving.patcher import PatcherBuilder, Patch
from mx_driving.patcher import index, batch_matmul, numpy_type, ddp, stream, ddp_forward
from mx_driving.patcher import resnet_add_relu, resnet_maxpool


def flash_attn(attention: ModuleType, options: Dict):
    _in_projection_packed = attention._in_projection_packed
    auto_fp16 = attention.auto_fp16
    rearrange = attention.rearrange

    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    @auto_fp16(apply_to=('q', 'k', 'v'), out_fp32=True)
    def FlashAttention_forward(self, q, k, v, causal=False, key_padding_mask=None):
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

    def FlashMHA_forward(self, q, k, v, key_padding_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)

        context, attn_weights = self.inner_attn(q, k, v, key_padding_mask=key_padding_mask, causal=self.causal)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights

    if hasattr(attention, "FlashAttention"):
        attention.FlashAttention.forward = FlashAttention_forward

    if hasattr(attention, "FlashMHA"):
        attention.FlashMHA.forward = FlashMHA_forward


def cpu2npu(models: ModuleType, options: Dict):

    def DFA_get_weights(self, instance_feature, anchor_embed, metas=None):
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

    if hasattr(models, "DeformableFeatureAggregation"):
        models.DeformableFeatureAggregation._get_weights = DFA_get_weights


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
        boxes = torch.cat(box_target, dim=0)
        output = torch.cat(
            [
                boxes[..., [X, Y, Z]],
                boxes[..., [W, L, H]].log(),
                torch.sin(boxes[..., YAW]).unsqueeze(-1),
                torch.cos(boxes[..., YAW]).unsqueeze(-1),
                boxes[..., YAW + 1:],
            ],
            dim=-1,
        )
        if device is not None:
            output = output.to(device=device)
        outputs = torch.split(output, sizes, dim=0)
        return outputs

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

    if hasattr(target, "SparseBox3DTarget"):
        target.SparseBox3DTarget._cls_cost = _cls_cost

    if hasattr(target, "SparseBox3DTarget"):
        target.SparseBox3DTarget._box_cost = _box_cost

    if hasattr(target, "SparseBox3DTarget"):
        target.SparseBox3DTarget.encode_reg_target = encode_reg_target


def map_target(target: ModuleType, options: Dict):
    SparsePoint3DTarget = target.SparsePoint3DTarget
    build_assigner = target.build_assigner

    def __init__(
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
        super(SparsePoint3DTarget, self).__init__(
            num_dn_groups, num_temp_dn_groups
        )
        self.assigner = build_assigner(assigner)
        self.dn_noise_scale = dn_noise_scale
        self.max_dn_gt = max_dn_gt
        self.add_neg_dn = add_neg_dn

        self.num_cls = num_cls
        self.num_sample = num_sample
        self.roi_size = roi_size
        self.origin = -torch.tensor([self.roi_size[0] / 2, self.roi_size[1] / 2]).npu()
        self.norm = torch.tensor([self.roi_size[0], self.roi_size[1]]).npu() + 1e-5

    def normalize_line(self, line):
        if line.shape[0] == 0:
            return line

        line = line.view(line.shape[:-1] + (self.num_sample, -1))
        line = line - self.origin
        # transform from range [0, 1] to (0, 1)
        line = line / self.norm
        line = line.flatten(-2, -1)

        return line

    if hasattr(target, "SparsePoint3DTarget"):
        target.SparsePoint3DTarget.__init__ = __init__

    if hasattr(target, "SparsePoint3DTarget"):
        target.SparsePoint3DTarget.normalize_line = normalize_line


def motion_planning_target(target: ModuleType, options: Dict):
    get_cls_target = target.get_cls_target

    # pylint: disable=too-many-return-values
    def motion_sample(
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

        cls_target = get_cls_target(reg_pred, reg_target, reg_weight)
        cls_weight = reg_weight.any(dim=-1)
        best_reg = torch.gather(reg_pred, 2, cls_target[..., None, None, None].repeat(1, 1, 1, ts, d)).squeeze(2)

        return cls_target, cls_weight, best_reg, reg_target, reg_weight, num_pos

    # pylint: disable=too-many-arguments,huawei-too-many-arguments,too-many-return-values
    def planning_sample(
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
        cls_target = get_cls_target(reg_pred, gt_reg_target, gt_reg_mask)
        cls_weight = gt_reg_mask.any(dim=-1)
        best_reg = torch.gather(reg_pred, 2, cls_target[..., None, None, None].repeat(1, 1, 1, self.ego_fut_ts, 2)).squeeze(2)

        return cls_pred, cls_target, cls_weight, best_reg, gt_reg_target, gt_reg_mask

    if hasattr(target, "MotionTarget"):
        target.MotionTarget.sample = motion_sample

    if hasattr(target, "PlanningTarget"):
        target.PlanningTarget.sample = planning_sample


def get_hccl_init_dist(runner: ModuleType):
    module = importlib.import_module(runner)

    if hasattr(module, "dist_utils"):
        mp = module.dist_utils.mp
        _init_dist_pytorch = module.dist_utils._init_dist_pytorch
        _init_dist_mpi = module.dist_utils._init_dist_mpi
        _init_dist_slurm = module.dist_utils._init_dist_slurm

        def hccl_init_dist(launcher: str, backend: str = 'nccl', **kwargs) -> None:
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

    return None


def run_ddp_forward(parallel: ModuleType, options: Dict):

    def _run_ddp_forward(self, *inputs, **kwargs):
        module_to_run = self.module

        if self.device_ids:
            inputs, kwargs = self.to_kwargs(  # type: ignore
                inputs, kwargs, self.device_ids[0])
            return module_to_run(*inputs[0], **kwargs[0])  # type: ignore
        else:
            return module_to_run(*inputs, **kwargs)

    if hasattr(parallel, "MMDistributedDataParallel"):
        parallel.MMDistributedDataParallel._run_ddp_forward = _run_ddp_forward


def instance_queue(queue: ModuleType, options: Dict):

    def prepare_motion(
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

    if hasattr(queue, "InstanceQueue"):
        queue.InstanceQueue.prepare_motion = prepare_motion


def generate_patcher_builder(performance=False):
    patcher_builder = (
        PatcherBuilder()
        .add_module_patch("torch", Patch(index), Patch(batch_matmul))
        .add_module_patch("numpy", Patch(numpy_type))
        .add_module_patch("mmcv.parallel", Patch(ddp), Patch(stream), Patch(ddp_forward), Patch(run_ddp_forward))
        .add_module_patch("mmdet.models.backbones.resnet", Patch(resnet_add_relu), Patch(resnet_maxpool))

        .add_module_patch("projects.mmdet3d_plugin.models.attention", Patch(flash_attn))
        .add_module_patch("projects.mmdet3d_plugin.models.detection3d.losses", Patch(detection_losses))
        .add_module_patch("projects.mmdet3d_plugin.models.blocks", Patch(cpu2npu))
        .add_module_patch("projects.mmdet3d_plugin.models.detection3d.target", Patch(detection_target))
        .add_module_patch("projects.mmdet3d_plugin.models.map.target", Patch(map_target))

        .add_module_patch("projects.mmdet3d_plugin.models.motion.target", Patch(motion_planning_target))
        .add_module_patch("projects.mmdet3d_plugin.models.motion.instance_queue", Patch(instance_queue))

        #.with_profiling('./profiling/level2', 2)
    )
    if performance:
        patcher_builder.brake_at(1000)
    return patcher_builder


# pylint: disable=huawei-redefined-outer-name, lambda-assign
def block_gpu_flash_attention_dependency():
    '''
    In  /projects/mmdet3d_plugin/models/attention.py
    the following lines
    -try:
    -    from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
    -    print('Use flash_attn_unpadded_kvpacked_func')
    -except:
    -    from flash_attn.flash_attn_interface import  flash_attn_varlen_kvpacked_func as flash_attn_unpadded_kvpacked_func
    -    print('Use flash_attn_varlen_kvpacked_func')
    -from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis
    will attempt to import flash_attn which is an external dependency implemented for GPU

    The migration to ascend will patch a NPU version of flash_attn at 
    -.add_module_patch("projects.mmdet3d_plugin.models.attention", Patch(flash_attn))
    The patch will replace FlashAttention.forward, where flash_attn_unpadded_kvpacked_func, unpad_input ... are used, as a whole 

    Still, the imports inside /projects/mmdet3d_plugin/models/attention.py will raise import error when GPU flash_attn is 
    not installed, to avoid import error, here the function uses sys.module to register flash_attn module name before the 
    actual import takes place in order to pretend the flash_attn being imported already, avoiding import errors 
    '''
    
    flash_attn = types.ModuleType('flash_attn')

    flash_attn_interface = types.ModuleType('flash_attn.flash_attn_interface')
    flash_attn_interface.flash_attn_unpadded_kvpacked_func = lambda *args, **kwargs: None
    flash_attn_interface.flash_attn_varlen_kvpacked_func = lambda *args, **kwargs: None

    bert_padding = types.ModuleType('flash_attn.bert_padding')
    bert_padding.unpad_input = lambda *args, **kwargs: (None, None)  
    bert_padding.pad_input = lambda *args, **kwargs: None
    bert_padding.index_first_axis = lambda *args, **kwargs: None

    flash_attn.flash_attn_interface = flash_attn_interface
    flash_attn.bert_padding = bert_padding

    sys.modules['flash_attn'] = flash_attn
    sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface
    sys.modules['flash_attn.bert_padding'] = bert_padding


# Official model repo has missing includes, apply the fix here
def fix_missing_include():

    mmdet3d_models_module = importlib.import_module("projects.mmdet3d_plugin.models")

    sparsedrive_v1_module = importlib.import_module("projects.mmdet3d_plugin.models.sparsedrive_v1")
    V1SparseDrive = getattr(sparsedrive_v1_module, "V1SparseDrive")

    sparsedrive_head_v1_module = importlib.import_module("projects.mmdet3d_plugin.models.sparsedrive_head_v1")
    V1SparseDriveHead = getattr(sparsedrive_head_v1_module, "V1SparseDriveHead")

    motion_blocks_v11_module = importlib.import_module("projects.mmdet3d_plugin.models.motion.motion_blocks_v11")
    V11MotionPlanningRefinementModule = getattr(motion_blocks_v11_module, "V11MotionPlanningRefinementModule")

    motion_planning_head_v13_module = importlib.import_module("projects.mmdet3d_plugin.models.motion.motion_planning_head_v13")
    V13MotionPlanningHead = getattr(motion_planning_head_v13_module, "V13MotionPlanningHead")

    missing = 'V1SparseDrive'
    if missing not in mmdet3d_models_module.__all__:
        mmdet3d_models_module.__all__.append(missing)
    setattr(mmdet3d_models_module, missing, V1SparseDrive)

    missing = 'V1SparseDriveHead'
    if missing not in mmdet3d_models_module.__all__:
        mmdet3d_models_module.__all__.append(missing)
    setattr(mmdet3d_models_module, missing, V1SparseDriveHead)
        
    missing = 'V13MotionPlanningHead'
    if missing not in mmdet3d_models_module.__all__:
        mmdet3d_models_module.__all__.append(missing)
    setattr(mmdet3d_models_module, missing, V13MotionPlanningHead)
        
    missing = 'V11MotionPlanningRefinementModule'
    if missing not in mmdet3d_models_module.__all__:
        mmdet3d_models_module.__all__.append(missing)
    setattr(mmdet3d_models_module, missing, V11MotionPlanningRefinementModule)


# Mock deform_aggreg in projects.mmdet3d_plugin.ops.deformable_aggregation, replace by mx_driving's deform_aggreg within the mock class
def patch_deform_aggreg():
    
    
    class MockDeformableAggregationFunction:
        @staticmethod
        def apply(*args, **kwargs):
            return mx_driving.deformable_aggregation(*args, **kwargs)
    
    
    mock_module = types.ModuleType("projects.mmdet3d_plugin.ops.deformable_aggregation")
    sys.modules["projects.mmdet3d_plugin.ops.deformable_aggregation"] = mock_module
    mock_module.DeformableAggregationFunction = MockDeformableAggregationFunction


def _init():
    # order matters
    block_gpu_flash_attention_dependency() 
    patch_deform_aggreg()
    fix_missing_include()

    mmcv.runner.init_dist = get_hccl_init_dist('mmcv.runner')


_init()