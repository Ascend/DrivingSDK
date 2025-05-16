import importlib
import os
from types import ModuleType
from typing import Dict
from typing import List, Optional, Tuple, Union
import torch
import torch_npu
import mx_driving
from mx_driving.patcher import PatcherBuilder, Patch


def amp_spconv(submconvops: ModuleType, options: Dict):
    from torch.npu.amp import custom_bwd, custom_fwd

    amp_forward = custom_fwd(fwd=submconvops.SubMConvFunction.forward, cast_inputs=torch.float32)
    amp_backward = custom_bwd(bwd=submconvops.SubMConvFunction.backward)

    if hasattr(submconvops, "SubMConvFunction"):
        submconvops.SubMConvFunction.forward = amp_forward
        submconvops.SubMConvFunction.backward = amp_backward


def amp_scatter(scatterops: ModuleType, options: Dict):
    from torch.npu.amp import custom_bwd, custom_fwd

    amp_forward = custom_fwd(fwd=scatterops.ScatterMaxFunction.forward, cast_inputs=torch.float32)
    amp_backward = custom_bwd(bwd=scatterops.ScatterMaxFunction.backward)

    if hasattr(scatterops, "ScatterMaxFunction"):
        scatterops.ScatterMaxFunction.forward = amp_forward
        scatterops.ScatterMaxFunction.backward = amp_backward


def sparse_tensor(sp_strcut: ModuleType, options: Dict):

    def replace_feature(self, feature: torch.Tensor):
        """we need to replace x.features = F.relu(x.features) with x = x.replace_feature(F.relu(x.features))
        due to limit of torch.fx
        """
        new_spt = sp_strcut.SparseConvTensor(feature, self.indices, self.spatial_shape,
                            self.batch_size, self.grid)
        return new_spt
    
    setattr(sp_strcut.SparseConvTensor, 'replace_feature', replace_feature)
    

def SerializedAttention_patch(ptv3: ModuleType, options: Dict):

    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        #set enable FA  False to use npu-fusion-attention
        enable_flash=False,
        upcast_attention=True,
        upcast_softmax=True,
    ): 
        super(ptv3.PointModule, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = False
        enable_flash = False
        
        # delete if-else branch to use npu-fusion-attention
        self.patch_size_max = patch_size
        self.patch_size = 0
        self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = ptv3.RPE(patch_size, num_heads) if self.enable_rpe else None


    def forward(self, point): 
        if not self.enable_flash:
            self.patch_size = min(ptv3.offset2bincount(point.offset).min().tolist(), self.patch_size_max)

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # use npu_fusion_attertion
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            feats = torch_npu.npu_fusion_attention(q, k, v, self.num_heads, "BNSD", scale=self.scale)
            feat = feats[0].transpose(1, 2).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


    if hasattr(ptv3, "SerializedAttention"):
        ptv3.SerializedAttention.__init__ = __init__

    if hasattr(ptv3, "SerializedAttention"):
        ptv3.SerializedAttention.forward = forward


def Block_patch(ptv3: ModuleType, options: Dict):
    from mx_driving.spconv import SubMConv3d

    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=ptv3.nn.LayerNorm,
        act_layer=ptv3.nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        #set enable FA  False to use npu-fusion-attention
        enable_flash=False,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super(ptv3.PointModule, self).__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = ptv3.PointSequential(
            #use SubMConv3d in DrivingSDK
            SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            ptv3.nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = ptv3.PointSequential(norm_layer(channels))
        self.attn = ptv3.SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = ptv3.PointSequential(norm_layer(channels))
        self.mlp = ptv3.PointSequential(
            ptv3.MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = ptv3.PointSequential(
            ptv3.DropPath(drop_path) if drop_path > 0.0 else ptv3.nn.Identity()
        )

    if hasattr(ptv3, "Block"):
        ptv3.Block.__init__ = __init__


def SerializedPooling_patch(ptv3: ModuleType, options: Dict):
    from mx_driving.common import scatter_mean
    from mx_driving.common import scatter_max

    def forward(self, point: ptv3.Point): 
        pooling_depth = (ptv3.math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0].cpu(),
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        #use scatterops in DrivingSDK
        #adaption for npu
        cluster = cluster.npu()
        counts = counts.npu()
        idx_list, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]
        code = code[:, head_indices]
        order = torch.argsort(code.cpu()).npu()
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        coord_ = scatter_mean(point.coord[indices], idx_list.to(torch.int32))
        proj_point_feat = self.proj(point.feat)[indices]
        if self.reduce == 'max':
            feat_, _ = scatter_max(proj_point_feat, idx_list.to(torch.int32))
        elif self.reduce == 'mean':
            feat_ = scatter_mean(proj_point_feat, idx_list.to(torch.int32))
        else:
            feat = torch_scatter.segment_csr(self.proj(point.feat)[indices].cpu(), idx_ptr.cpu(), reduce=self.reduce).npu()
        
        point_dict = ptv3.Dict(
            feat=feat_,
            coord=coord_,
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = ptv3.Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point

    if hasattr(ptv3, "SerializedPooling"):
        ptv3.SerializedPooling.forward = forward


def Embedding_patch(ptv3: ModuleType, options: Dict):
    from mx_driving.spconv import SubMConv3d

    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super(ptv3.PointModule, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        self.stem = ptv3.PointSequential(
            #use SubMConv3d in DrivingSDK
            conv=SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    if hasattr(ptv3, "Embedding"):
        ptv3.Embedding.__init__ = __init__


ptv3_patcher_builder = (
    PatcherBuilder()
    .add_module_patch("mx_driving.ops.sparse_functional", Patch(amp_spconv))
    .add_module_patch("mx_driving.ops.scatter_max", Patch(amp_scatter))
    .add_module_patch("mx_driving.modules.sparse_structure", Patch(sparse_tensor))
    .add_module_patch("pointcept.models.point_transformer_v3.point_transformer_v3m1_base", Patch(SerializedAttention_patch))
    .add_module_patch("pointcept.models.point_transformer_v3.point_transformer_v3m1_base", Patch(Block_patch))
    .add_module_patch("pointcept.models.point_transformer_v3.point_transformer_v3m1_base", Patch(SerializedPooling_patch))
    .add_module_patch("pointcept.models.point_transformer_v3.point_transformer_v3m1_base", Patch(Embedding_patch))
)
