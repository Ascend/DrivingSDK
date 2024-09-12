from mmcv.runner import BaseModule
from torch import nn as nn
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
import torch.nn.functional as F


def interpolate_trilinear(x, scale_factor, mode, align_corners):
    # assert mode == 'trilinear'
    # assert align_corners == False
    # bilinear + bilinear
    scale_t, scale_h, scale_w = scale_factor
    N, C, T, H, W = x.size(0), x.size(1), x.size(2), x.size(3), x.size(4)

    x_fused_nc = x.reshape(N*C, T, H, W)
    y_resize_hw = F.interpolate(x_fused_nc, scale_factor=(scale_h, scale_w), mode='bilinear')
    new_shape_h, new_shape_w = y_resize_hw.shape[-2], y_resize_hw.shape[-1]
    y_fused_hw = y_resize_hw.reshape(N, C, T, new_shape_h*new_shape_w)
    y_resize_t = F.interpolate(y_fused_hw, scale_factor=(scale_t, 1), mode='bilinear')
    new_shape_t = y_resize_t.shape[-2]
    y = y_resize_t.reshape(N, C, new_shape_t, new_shape_h, new_shape_w)
    return y


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class OccupancyDecoder(BaseModule):

    def __init__(self,
                 num_classes,
                 bev_h=50,
                 bev_w=50,
                 bev_z=8,
                 conv_up_layer = 2,
                 inter_up_rate = [1,2,2],
                 embed_dim = 256,
                 upsampling_method='trilinear',
                 align_corners=False):
        super(OccupancyDecoder, self).__init__()
        self.num_classes = num_classes
        self.upsampling_method = upsampling_method
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.out_dim = embed_dim // 2
        self.align_corners = align_corners
        self.inter_up_rate = inter_up_rate
        self.conv_up_layer = conv_up_layer
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(embed_dim,embed_dim,(1,3,3),padding=(0,1,1)),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(embed_dim, embed_dim, (2, 2, 2), stride=(2, 2, 2)),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(embed_dim, self.out_dim, (2, 2, 2), stride=(2, 2, 2)),
            nn.BatchNorm3d(self.out_dim),
            nn.ReLU(inplace=True),
        )

        self.semantic_det = nn.Sequential(nn.Conv3d(embed_dim, 2, kernel_size=1))
        self.semantic_cls = nn.Sequential(nn.Conv3d(self.out_dim, self.num_classes,kernel_size=1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)
    

    def forward(self, inputs):
        
        # z x y
        voxel_input = inputs.view(1,self.bev_w,self.bev_h,self.bev_z, -1).permute(0,4,3,1,2)

        voxel_det = self.semantic_det(voxel_input)

        voxel_up1 = self.upsample(voxel_input)

        voxel_cls = self.semantic_cls(voxel_up1)

        voxel_pred = interpolate_trilinear(voxel_cls, 
                                           scale_factor=(self.inter_up_rate[0], self.inter_up_rate[1], self.inter_up_rate[2]), 
                                           mode=self.upsampling_method, align_corners=self.align_corners)

        return voxel_pred, voxel_det