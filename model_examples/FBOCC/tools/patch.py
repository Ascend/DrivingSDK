# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
import os
from types import ModuleType
from typing import Dict
import torch
import torch.nn as nn
import torch_npu
import mmcv
import mmcv.runner
import mx_driving
from mx_driving.patcher import PatcherBuilder, Patch
from mx_driving.patcher import index, batch_matmul, numpy_type, ddp, stream, msda
from mx_driving.patcher import resnet_add_relu, resnet_maxpool
from mmcv.runner import force_fp32
import torch.nn.functional as F
from mx_driving import SubMConv3d, SparseConv3d, SparseSequential, bev_pool_v3
from functools import partial
from torch.utils.checkpoint import checkpoint as cp
from mx_driving import npu_add_relu

def matmul_shape_patch(mmdet3d: ModuleType, options: Dict):

    def get_cam2ego_coor(self, inputs, downsample=1):
        depth_cfg = self.grid_config['depth']

        H_in, W_in = self.input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float)\
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)
        D = d.shape[0]
        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)

        # D x H x W x 3
        frustum = torch.stack((x, y, d), -1)
        rots, trans, cam2imgs, post_rots, post_trans, bda = inputs
        #NPU上matmul不支持维度大于6的case，先降维再运算
        B, N, _ = trans.shape

        points = frustum.to(rots) - post_trans.view(B, N, 1, 1, 1, 3)
        frustumShape = frustum.shape
        points = points.view(B*N, frustumShape[0], frustumShape[1], frustumShape[2], 3)
        points = torch.inverse(post_rots).view(B * N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :2] * points[:, :, :, :, 2:3],
                            points[:, :, :, :, 2:3]
                            ), 4)
        combine = rots.float().matmul(torch.inverse(cam2imgs.float()))
        points = combine.view(B*N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B*N, 1, 1, 1, 3)
        bda_expanded = bda.unsqueeze(1).expand(B, N, 3, 3).reshape(B*N, 1, 1, 1, 3, 3)
        points = bda_expanded.matmul(points.unsqueeze(-1)).squeeze(-1)
        points = points.view(B, N, frustumShape[0], frustumShape[1], frustumShape[2], 3)

        coor = points
        coor = ((coor - self.grid_lower_bound.to(coor)) / 0.4)
        coor = coor.long()
        # filter out points that are outside box
        kept = (coor[..., 0] >= 0) & (coor[..., 0] < 200) & \
               (coor[..., 1] >= 0) & (coor[..., 1] < 200) & \
               (coor[..., 2] >= 0) & (coor[..., 2] < 16)

        coor[~kept] = -999
        return coor

    def get_lidar_coor(self, rots, trans, cam2imgs, post_rots, post_trans,
                       bda):
        #NPU上matmul不支持维度大于6的case，先降维再运算
        B, N, _ = trans.shape
        points = self.frustum.to(rots) - post_trans.view(B, N, 1, 1, 1, 3)
        frustumShape = self.frustum.shape
        points = points.view(B*N, frustumShape[0], frustumShape[1], frustumShape[2], 3)
        points = torch.inverse(post_rots).view(B * N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :2] * points[:, :, :, :, 2:3],
                           points[:, :, :, :, 2:3]
                           ), 4)

        combine = rots.matmul(torch.inverse(cam2imgs)).float()
        points = combine.view(B*N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B*N, 1, 1, 1, 3)
        bda_expanded = bda.unsqueeze(1).expand(B, N, 3, 3).reshape(B*N, 1, 1, 1, 3, 3)
        points = bda_expanded.matmul(points.unsqueeze(-1)).squeeze(-1)
        points = points.view(B, N, frustumShape[0], frustumShape[1], frustumShape[2], 3)
        return points

    def voxel_pooling_v2(self, coor, depth, feat):
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)
        if ranks_feat is None:
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(self.grid_size[0]),
                int(self.grid_size[1]),
                int(self.grid_size[2]),
            ]).to(feat)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                          int(self.grid_size[1]), int(self.grid_size[0]),
                          feat.shape[-1])  # (B, Z, Y, X, C)
        #使用DrivingSDK高性能算子bev_pool_v3
        bev_feat = bev_pool_v3(depth, feat.float(), ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape)
        bev_feat = bev_feat.permute(0, 1, 3, 4, 2) # B, C, Z, X, Y- > B, C, X, Y, Z
        return bev_feat

    @force_fp32(apply_to=('reference_points', 'cam_params'))
    def point_sampling(self,reference_points, pc_range,  img_metas, cam_params=None, gt_bboxes_3d=None):
        #NPU上matmul不支持维度大于6的case，先降维再运算
        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        B, N, _ = trans.shape
        eps = 1e-5
        ogfH, ogfW = self.final_dim
        reference_points = reference_points[None, None].repeat(B, N, 1, 1, 1, 1)
        reference_shape=reference_points.shape
        bda_expand=torch.inverse(bda).unsqueeze(1).expand(B, N, 3, 3).reshape(B*N, 1, 1, 1, 3, 3)
        reference_points=bda_expand.matmul(reference_points.view(B*N,reference_shape[2]
            ,reference_shape[3],reference_shape[4],3).unsqueeze(-1)).squeeze(-1)

        reference_points -= trans.view(B*N, 1, 1, 1, 3)
        combine = rots.matmul(torch.inverse(intrins)).inverse()
        reference_points_cam = combine.view(B*N, 1, 1, 1, 3, 3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        reference_points_cam = torch.cat([reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps),  reference_points_cam[..., 2:3]], 4
            )
        reference_points_cam = post_rots.view(B*N, 1, 1, 1, 3, 3).matmul(reference_points_cam.unsqueeze(-1)).squeeze(-1)
        reference_points_cam += post_trans.view(B*N, 1, 1, 1, 3)
        reference_points_cam=reference_points_cam.view(B,N,reference_shape[2],reference_shape[3],reference_shape[4],3)
        reference_points_cam[..., 0] /= ogfW
        reference_points_cam[..., 1] /= ogfH
        mask = (reference_points_cam[..., 2:3] > eps)
        mask = (mask & (reference_points_cam[..., 0:1] > eps)
                 & (reference_points_cam[..., 0:1] < (1.0-eps))
                 & (reference_points_cam[..., 1:2] > eps)
                 & (reference_points_cam[..., 1:2] < (1.0-eps)))
        B, N, H, W, D, _ = reference_points_cam.shape
        reference_points_cam = reference_points_cam.permute(1, 0, 2, 3, 4, 5).reshape(N, B, H*W, D, 3)
        mask = mask.permute(1, 0, 2, 3, 4, 5).reshape(N, B, H*W, D, 1).squeeze(-1)

        return reference_points, reference_points_cam[..., :2], mask, reference_points_cam[..., 2:3]

    if hasattr(mmdet3d.models.fbbev.view_transformation.forward_projection.view_transformer, 'LSSViewTransformerFunction'):
        mmdet3d.models.fbbev.view_transformation.forward_projection.view_transformer.LSSViewTransformerFunction.get_lidar_coor = get_lidar_coor
    else:
        raise AttributeError('LSSViewTransformerFunction.get_lidar_coor attr not found')

    if hasattr(mmdet3d.models.fbbev.view_transformation.forward_projection.view_transformer, 'LSSViewTransformerFunction3D'):
        mmdet3d.models.fbbev.view_transformation.forward_projection.view_transformer.LSSViewTransformerFunction3D.get_lidar_coor = get_lidar_coor
        mmdet3d.models.fbbev.view_transformation.forward_projection.view_transformer.LSSViewTransformerFunction3D.voxel_pooling_v2 = voxel_pooling_v2
        mmdet3d.models.fbbev.view_transformation.forward_projection.view_transformer.LSSViewTransformerFunction3D.get_cam2ego_coor = get_cam2ego_coor
    else:
        raise AttributeError('LSSViewTransformerFunction3D.get_lidar_coor attr not found')

    if hasattr(mmdet3d.models.necks.view_transformer, 'LSSViewTransformer'):
        mmdet3d.models.necks.view_transformer.LSSViewTransformer.get_lidar_coor = get_lidar_coor
    else:
        raise AttributeError('LSSViewTransformer.get_lidar_coor attr not found')

    if hasattr(mmdet3d.models.necks.view_transformer, 'LSSViewTransformer2'):
        mmdet3d.models.necks.view_transformer.LSSViewTransformer2.get_lidar_coor = get_lidar_coor
    else:
        raise AttributeError('LSSViewTransformer2.get_lidar_coor attr not found')

    if hasattr(mmdet3d.models.fbbev.view_transformation.bevformer_utils, 'bevformer_encoder'):
        mmdet3d.models.fbbev.view_transformation.bevformer_utils.bevformer_encoder.point_sampling = point_sampling
    else:
        raise AttributeError('bevformer_encoder.point_sampling attr not found')

#适配Conv3D不支持sride>kernel_size的问题
def Conv3d_patch(resnet3d: ModuleType, options: Dict):
    BIAS = True
    build_norm_layer=resnet3d.build_norm_layer

    def conv1x1x1_downsample(in_planes, out_planes, stride=1, use_spase_3dtensor=False):
        if not use_spase_3dtensor:
            Conv3d = nn.Conv3d
        else:
            Conv3d = SparseConv3d if stride!=1 else SubMConv3d

        return Conv3d(in_planes,
                         out_planes,
                         kernel_size=1,
                         stride=1,
                         bias=BIAS)

    @force_fp32()
    def BasicBlock_forward(self, x, debug=False):
        residual = x

        if self.use_spase_3dtensor:
            out = self.layer_seq(x)
            if self.downsample is not None:
                residual = self.downsample(x)
            out = Fsp.sparse_add(out, residual)
            out = out.replace_feature(self.relu(out.features))
            return out
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x[:,:,::2,::2,::2])

            out = npu_add_relu(out,residual)
            return out

    @force_fp32()
    def Bottleneck_forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x[:,:,::2,::2,::2])

        out = npu_add_relu(out,residual)
        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, norm_cfg=None):
        downsample = None
        Sequential = nn.Sequential if not self.use_spase_3dtensor else SparseSequential
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:

                downsample = Sequential(
                    conv1x1x1_downsample(self.in_planes, planes * block.expansion, stride, self.use_spase_3dtensor),
                    build_norm_layer(norm_cfg, planes * block.expansion)[1])

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample,
                  use_spase_3dtensor = self.use_spase_3dtensor,
                  norm_cfg=norm_cfg))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, norm_cfg=norm_cfg, use_spase_3dtensor = self.use_spase_3dtensor))

        return Sequential(*layers)

    if hasattr(resnet3d, 'BasicBlock'):
        resnet3d.BasicBlock.forward = BasicBlock_forward
    else:
        raise AttributeError('BasicBlock attr not found')

    if hasattr(resnet3d, 'Bottleneck'):
        resnet3d.Bottleneck.forward = Bottleneck_forward
    else:
        raise AttributeError('Bottleneck attr not found')

    if hasattr(resnet3d, 'CustomResNet3D'):
        resnet3d.CustomResNet3D._make_layer = _make_layer
    else:
        raise AttributeError('CustomResNet3D attr not found')


#使用矩阵乘法代替矩阵索引操作，从而避免大量随机访存带来的性能下降，同时调换输入尾轴提升aclnnUpsampleTrilinear算子性能
def OccHead_patch(occhead: ModuleType, options: Dict):

    lovasz_softmax=occhead.lovasz_softmax
    @force_fp32(apply_to=('voxel_feats'))
    def forward_coarse_voxel(self, voxel_feats):
        output_occs = []
        output = {}

        if self.use_deblock:
            if self.with_cp and voxel_feats[0].requires_grad:
                x0 = cp(self.deblock, voxel_feats[0])
            else:
                x0 = self.deblock(voxel_feats[0])
            output_occs.append(x0)
        for feats, occ_conv in zip(voxel_feats, self.occ_convs):
            if self.with_cp  and feats.requires_grad:
                x = cp(occ_conv, feats)
            else:
                x = occ_conv(feats)
            output_occs.append(x)

        if self.soft_weights:
            voxel_soft_weights = self.voxel_soft_weights(output_occs[0])
            voxel_soft_weights = torch.softmax(voxel_soft_weights, dim=1)
        else:
            voxel_soft_weights = torch.ones([output_occs[0].shape[0], self.num_point_sampling_feat, 1, 1, 1],).to(output_occs[0].device) / self.num_point_sampling_feat

        out_voxel_feats = 0
        _, _, H, W, D= output_occs[0].shape
        #调用interpolate时调换输入尾轴
        for feats, weights in zip(output_occs, torch.unbind(voxel_soft_weights, dim=1)):
            feats = feats.permute(0, 1, 4, 3, 2)
            feats = F.interpolate(feats, size=[D, W, H], mode='trilinear', align_corners=False).contiguous()
            feats = feats.permute(0, 1, 4, 3, 2)
            out_voxel_feats += feats * weights.unsqueeze(1)
        output['out_voxel_feats'] = [out_voxel_feats]
        if self.with_cp and  out_voxel_feats.requires_grad:
            out_voxel = cp(self.occ_pred_conv, out_voxel_feats)
        else:
            out_voxel = self.occ_pred_conv(out_voxel_feats)

        output['occ'] = [out_voxel]

        return output

    def inverse_sigmoid(x, sign='A'):
        x = x.to(torch.float32)
        while x >= 1-1e-5:
            x = x - 1e-5

        while x< 1e-5:
            x = x + 1e-5

        return -torch.log((1 / x) - 1)

    def geo_scal_loss(pred, ssc_target, ignore_index=255, non_empty_idx=0):

        # Get softmax probabilities
        pred = F.softmax(pred, dim=1)

        # Compute empty and nonempty probabilities
        empty_probs = pred[:, non_empty_idx]
        nonempty_probs = 1 - empty_probs

        # Remove unknown voxels
        mask = ssc_target != ignore_index
        nonempty_target = ssc_target != non_empty_idx
        nonempty_target = (nonempty_target*mask).float()
        nonempty_probs = nonempty_probs*mask
        empty_probs = empty_probs*mask
        total_pixels=torch.sum(mask)

        eps = 1e-5
        intersection = (nonempty_target * nonempty_probs).sum()
        precision = intersection / (nonempty_probs.sum()+eps)
        recall = intersection / (nonempty_target.sum()+eps)
        spec = ((1 - nonempty_target) * (empty_probs)).sum() / (total_pixels-nonempty_target.sum()+eps)
        with torch.npu.amp.autocast(False):
            return (
                F.binary_cross_entropy_with_logits(inverse_sigmoid(precision, 'A'), torch.ones_like(precision))
                + F.binary_cross_entropy_with_logits(inverse_sigmoid(recall, 'B'), torch.ones_like(recall))
                + F.binary_cross_entropy_with_logits(inverse_sigmoid(spec, 'C'), torch.ones_like(spec))
            )

    def sem_scal_loss(pred_, ssc_target, ignore_index=255):
        # Get softmax probabilities
        with torch.npu.amp.autocast(False):
            pred = F.softmax(pred_, dim=1)
            loss = 0
            count = 0

            mask_bool = ssc_target != ignore_index
            mask_float= mask_bool.float()
            target=ssc_target * mask_bool
            total_valid_pixels = torch.sum(mask_float)

            n_classes = pred.shape[1]
            begin = 1 if n_classes == 19 else 0
            for i in range(begin, n_classes-1):

                # Get probability of class i
                p = pred[:, i]

                p = p * mask_float

                completion_target_bool = target == i
                completion_target_float = completion_target_bool.float()
                completion_target_num = torch.sum(completion_target_float)

                if completion_target_num > 0:
                    count += 1.0
                    nominator = torch.sum(p * completion_target_float)
                    loss_class = 0

                    pred_sum = torch.sum(p)
                    if pred_sum > 0:
                        precision = nominator / (pred_sum + 1e-5)
                        loss_precision = F.binary_cross_entropy_with_logits(
                                inverse_sigmoid(precision, 'D'), torch.ones_like(precision)
                            )
                        loss_class += loss_precision

                    if completion_target_num > 0:
                        recall = nominator / (completion_target_num +1e-5)

                        loss_recall = F.binary_cross_entropy_with_logits(inverse_sigmoid(recall, 'E'), torch.ones_like(recall))
                        loss_class += loss_recall

                    negative_target_num = total_valid_pixels - completion_target_num
                    if negative_target_num > 0:
                        negative_mask = mask_float * (1-completion_target_float)

                        specificity = torch.sum((1 - p) * negative_mask) / (
                            negative_target_num +  1e-5
                        )

                        loss_specificity = F.binary_cross_entropy_with_logits(
                                inverse_sigmoid(specificity, 'F'), torch.ones_like(specificity)
                            )
                        loss_class += loss_specificity
                    loss += loss_class
            l = loss/count
            return l

    def CE_ssc_loss(pred, target, class_weights=None, ignore_index=255):
        """
        :param: prediction: the predicted tensor, must be [BS, C, ...]
        """

        criterion = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=ignore_index, reduction="mean"
        )
        with torch.npu.amp.autocast(False):
            loss = criterion(pred, target.long())
        return loss

    @force_fp32()
    def loss_voxel(self, output_voxels, target_voxels, tag):

        # resize gt
        B, C, H, W, D = output_voxels.shape
        ratio = target_voxels.shape[2] // H
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_mask = target_voxels.sum(-1) == self.empty_idx
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space
            target_voxels = torch.mode(target_voxels, dim=-1)[0]
            target_voxels[target_voxels<0] = 255
            target_voxels = target_voxels.long()


        abnormal_mask=torch.isnan(output_voxels) | torch.isinf(output_voxels)
        output_voxels*=~abnormal_mask

        loss_dict = {}

        if self.use_focal_loss:
            loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * self.focal_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
        else:
            loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)

        loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
        loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)
        loss_dict['loss_voxel_lovasz_{}'.format(tag)] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels, ignore=255)


        if self.use_dice_loss:
            visible_mask = target_voxels!=255
            visible_pred_voxels = output_voxels.permute(0, 2, 3, 4, 1)[visible_mask]
            visible_target_voxels = target_voxels[visible_mask]
            visible_target_voxels = F.one_hot(visible_target_voxels.to(torch.long), 19)
            loss_dict['loss_voxel_dice_{}'.format(tag)] = self.dice_loss(visible_pred_voxels, visible_target_voxels)

        return loss_dict

    if hasattr(occhead, 'OccHead'):
        occhead.OccHead.loss_voxel = loss_voxel
        occhead.OccHead.forward_coarse_voxel = forward_coarse_voxel
    else:
        raise AttributeError('OccHead attr not found')

#通过改变调换输入尾轴提升interpolate api的性能(底层为aclnnUpsampleTrilinear算子)
def fpn3d_patch(fpn3d: ModuleType, options: Dict):

    @force_fp32()
    def forward(self, inputs):
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            if self.with_cp:
                lateral_i = torch.utils.checkpoint.checkpoint(lateral_conv, inputs[i])
            else:
                lateral_i = lateral_conv(inputs[i])
            laterals.append(lateral_i)

        for i in range(self.num_out - 1, 0, -1):
            H, W, D = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i].permute(0, 1, 4, 3, 2),
                    size=[D,W,H], align_corners=False, **self.upsample_cfg).permute(0, 1, 4, 3, 2)

        outs = []
        for i, fpn_conv in enumerate(self.fpn_convs):
            if self.with_cp:
                out_i = torch.utils.checkpoint.checkpoint(fpn_conv, laterals[i])
            else:
                out_i = fpn_conv(laterals[i])
            outs.append(out_i)

        return outs


    if hasattr(fpn3d, 'FPN3D'):
        fpn3d.FPN3D.forward = forward
    else:
        raise AttributeError('FPN3D attr not found')


# get the patch for FBOCC, and determine whether to brake advance
def generate_patcher_builder():
    fbocc_patcher_builder = (
        PatcherBuilder()
        .add_module_patch("mmcv", Patch(msda))
        .add_module_patch("torch", Patch(index), Patch(batch_matmul))
        .add_module_patch("numpy", Patch(numpy_type))
        .add_module_patch("mmcv", Patch(stream), Patch(ddp))
        .add_module_patch("mmdet", Patch(resnet_add_relu), Patch(resnet_maxpool))
        .add_module_patch("mmdet3d",Patch(matmul_shape_patch))
        .add_module_patch("mmdet3d.models.fbbev.modules.resnet3d",Patch(Conv3d_patch))
        .add_module_patch("mmdet3d.models.fbbev.modules.fpn3d",Patch(fpn3d_patch))
        .add_module_patch("mmdet3d.models.fbbev.heads.occupancy_head",Patch(OccHead_patch))
    )
    if os.environ.get("FBOCC_PERFORMANCE_FLAG"):
        fbocc_patcher_builder.brake_at(600)
    return fbocc_patcher_builder