from mmcv.runner import BaseModule
from torch import nn as nn
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
import torch.nn.functional as F
import torch


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
class MLP_Decoder(BaseModule):

    def __init__(self,
                 num_classes,
                 out_dim = 64,
                 inter_up_rate = [2,2,2],
                 upsampling_method='trilinear',
                 align_corners=False):
        super(MLP_Decoder, self).__init__()
        self.num_classes = num_classes
        self.upsampling_method = upsampling_method
        self.out_dim = out_dim
        self.align_corners = align_corners
        self.inter_up_rate = inter_up_rate
    
        self.mlp_decoder = MLP(dim_x=self.out_dim,act_fn='softplus',layer_size=2)
        self.classifier = nn.Linear(self.out_dim, self.num_classes)
                
    def forward(self, inputs):
        
        # z h w
        voxel_point = inputs.permute(0,2,3,4,1).view(1,-1,self.out_dim)
        voxel_point_feat = self.mlp_decoder(voxel_point)
        point_cls = self.classifier(voxel_point_feat)

        voxel_point_cls = point_cls.view(1,inputs.shape[2],inputs.shape[3],inputs.shape[4],-1).permute(0,4,1,2,3)

        voxel_logits = interpolate_trilinear(voxel_point_cls, 
                                             scale_factor=(self.inter_up_rate[0], self.inter_up_rate[1], self.inter_up_rate[2]), 
                                             mode=self.upsampling_method, align_corners=self.align_corners)
        
        return voxel_logits

class MLP(torch.nn.Module):
    def __init__(self, dim_x=3, filter_size=128, act_fn='relu', layer_size=8):
        super().__init__()
        self.layer_size = layer_size
        
        self.nn_layers = torch.nn.ModuleList([])
        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, filter_size)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            elif act_fn == 'softplus':
                self.nn_layers.append(torch.nn.Softplus())
            for _ in range(layer_size-1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
                elif act_fn == 'softplus':
                    self.nn_layers.append(torch.nn.Softplus())
            self.nn_layers.append(torch.nn.Linear(filter_size, dim_x))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, dim_x)))

    def forward(self, x):
        """ points -> features
            [B, N, 3] -> [B, K]
        """
        for layer in self.nn_layers:
            x = layer(x)
                
        return x
