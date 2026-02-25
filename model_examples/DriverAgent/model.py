import torch
from torch import nn
from utils_pt import outputActivation
from torch_npu.npu.amp import autocast
from torchinfo import summary

def build_grid_adj_matrix(grid_size, device):
    """构建网格节点的邻接矩阵（25×5网格→125个节点）"""
    H, W = grid_size  # 25,5
    num_nodes = H * W  # 125
    A = torch.zeros((num_nodes, num_nodes), device=device)  # 初始化稠密邻接矩阵（直接稠密化，跳过稀疏步骤）
    # 遍历每个节点，标记邻居（上下左右）
    for i in range(H):
        for j in range(W):
            node_idx = i * W + j  # 节点的扁平化索引（0~124）
            # 上邻居（i-1,j）
            if i > 0:
                A[node_idx][(i-1)*W + j] = 1
            # 下邻居（i+1,j）
            if i < H-1:
                A[node_idx][(i+1)*W + j] = 1
            # 左邻居（i,j-1）
            if j > 0:
                A[node_idx][i*W + (j-1)] = 1
            # 右邻居（i,j+1）
            if j < W-1:
                A[node_idx][i*W + (j+1)] = 1
    # 步骤2：预处理归一化（提前计算，避免推理时重复算）
    D = torch.diag(torch.sum(A, dim=1))  # 度矩阵
    D_inv_sqrt = torch.inverse(torch.sqrt(D + 1e-6))  # 防止除以0
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt  # 归一化邻接矩阵（GCN标准归一化）
    return A_norm.contiguous()  # 确保内存连续，适配Cube引擎

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),  # 先BN
            nn.ReLU(inplace=True)          # 后激活
        )

class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads
        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)  # 调整维度，使注意力在指定维度计算
        residual = x  # 残差连接的原始输入

        # 自注意力计算
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)   # 注意力输出 dropout
        sum_ln1 = residual + out               # 残差和
        sum_ln1_fp32 = sum_ln1.to(torch.float32)
        out_ln1 = self.ln1(sum_ln1_fp32)      # FP32下计算LayerNorm
        out = out_ln1.to(residual.dtype)      # 转回原数据类型（如FP16，适配混合精度）

        # 前馈网络
        residual = out
        out = self.feed_forward(out)          # 前馈网络计算
        out = self.dropout2(out)              # 前馈输出 dropout

        sum_ln2 = residual + out               # 残差和
        sum_ln2_fp32 = sum_ln2.to(torch.float32)
        out_ln2 = self.ln2(sum_ln2_fp32)      # FP32下计算LayerNorm
        out = out_ln2.to(residual.dtype)      # 转回原数据类型

        return out.transpose(dim, -2)  # 恢复原始维度


# The implementation of PiP architecture
class pipNet(nn.Module):

    def __init__(self, args):
        super(pipNet, self).__init__()
        self.args = args
        self.use_cuda = args.use_cuda

        # Flag for output:
        # -- Train-mode : Concatenate with true maneuver label.
        # -- Test-mode  : Concatenate with the predicted maneuver with the maximal probability.
        self.train_output_flag = args.train_output_flag
        self.use_planning = args.use_planning
        self.use_fusion = args.use_fusion

        # IO Setting
        self.grid_size = args.grid_size
        self.in_length = args.in_length
        self.out_length = args.out_length
        self.num_lat_classes = args.num_lat_classes
        self.num_lon_classes = args.num_lon_classes

        ## Sizes of network layers
        self.temporal_embedding_size = args.temporal_embedding_size
        self.encoder_size = args.encoder_size
        self.decoder_size = args.decoder_size
        self.soc_conv_depth = args.soc_conv_depth
        self.soc_conv2_depth = args.soc_conv2_depth
        self.dynamics_encoding_size = args.dynamics_encoding_size
        self.social_context_size = args.social_context_size
        self.targ_enc_size = self.social_context_size + self.dynamics_encoding_size
        self.fuse_enc_size = args.fuse_enc_size
        self.fuse_conv1_size = 2 * self.fuse_enc_size
        self.fuse_conv2_size = 4 * self.fuse_enc_size
        
        self.num_layers = args.num_layers
        self.feed_forward_dim = args.feed_forward_dim
        self.num_heads = args.num_heads
        self.dropout = args.dropout

        # Activations:
        self.leaky_relu = nn.LeakyReLU(0.1)
        # 相较于传统relu针对小于0部分乘上0.1而不是全置0，避免直接出现神经元死亡情况
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        ## Define network parameters
        ''' Convert traj to temporal embedding'''
        self.temporalConv = nn.Conv1d(in_channels=2, out_channels=self.temporal_embedding_size, kernel_size=3, padding=1)
        # 增加padding的目的是保持输出序列数据的历史时间刻度一致，因为卷积核长度为3

        ''' Encode the input temporal embedding '''
        self.nbh_lstm = nn.LSTM(
            input_size=self.temporal_embedding_size, 
            hidden_size=self.encoder_size,
            num_layers=1
        )
        if self.use_planning:
            self.plan_lstm = nn.LSTM(
                input_size=self.temporal_embedding_size, 
                hidden_size=self.encoder_size,
                num_layers=1
            )
            
        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.temporal_embedding_size, self.feed_forward_dim, self.num_heads, self.dropout)
                for _ in range(self.num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.temporal_embedding_size, self.feed_forward_dim, self.num_heads, self.dropout)
                for _ in range(self.num_layers)
            ]
        )
        
        ''' Encoded dynamic to dynamics_encoding_size'''
        self.dyn_emb = nn.Linear(self.encoder_size, self.dynamics_encoding_size)

        ''' Convolutional Social Pooling on the planned vehicle and all nbrs vehicles  '''
        self.nbrs_conv_social = nn.Sequential(
            nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3),
            # (targets_batch_size, soc_conv_depth, 23, 3), 即3*3卷积核在网格上做了一次卷积操作
            self.leaky_relu,
            nn.MaxPool2d((3, 3), stride=2),
            # (targets_batch_size, soc_conv_depth, 11, 1), 最大池化操作，选择3*3的范围内最大的元素作为最终值，一定程度上类似于再进行一次缩小范围
            nn.Conv2d(self.soc_conv_depth, self.soc_conv2_depth, (3, 1)),
            # (targets_batch_size, soc_conv2_depth, 9, 1), 这里卷积核尺寸横向，即X方向（车道）变小的原因是MaxPooling后横向由原本的5列数据变为1列
            self.leaky_relu
        )
        # output shape: (targets_batch_size, soc_conv2_depth, 9, 1)
        if self.use_planning:
            self.plan_conv_social = nn.Sequential(
                nn.Conv2d(self.encoder_size, self.soc_conv_depth, 3),
                self.leaky_relu,
                nn.MaxPool2d((3, 3), stride=2),
                nn.Conv2d(self.soc_conv_depth, self.soc_conv2_depth, (3, 1)),
                self.leaky_relu
            )
            self.pool_after_merge = nn.MaxPool2d((2, 2), padding=(1, 0))
        else:
            self.pool_after_merge = nn.MaxPool2d((2, 1), padding=(1, 0))
            # 纵向（Y方向）增加1个长度后pooling

        ''' Target Fusion Module'''
        if self.use_fusion:
            ''' Fused Structure'''
            self.fcn_conv1 = ConvBNReLU(self.targ_enc_size, self.fuse_conv1_size, kernel_size=3, stride=1, padding=1)
            self.fcn_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.fcn_conv2 = ConvBNReLU(self.fuse_conv1_size, self.fuse_conv2_size, kernel_size=3, stride=1, padding=1)
            self.fcn_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.fcn_convTrans1 = nn.ConvTranspose2d(self.fuse_conv2_size, self.fuse_conv1_size, kernel_size=3, stride=2, padding=1)
            self.back_bn1 = nn.BatchNorm2d(self.fuse_conv1_size)
            self.fcn_convTrans2 = nn.ConvTranspose2d(self.fuse_conv1_size, self.fuse_enc_size, kernel_size=3, stride=2, padding=1)
            self.back_bn2 = nn.BatchNorm2d(self.fuse_enc_size)
        else:
            self.fuse_enc_size = 0

        ''' Decoder LSTM'''
        self.op_lat = nn.Linear(self.targ_enc_size + self.fuse_enc_size,
                                self.num_lat_classes)  # output lateral maneuver.
        self.op_lon = nn.Linear(self.targ_enc_size + self.fuse_enc_size,
                                self.num_lon_classes)  # output longitudinal maneuver.
        self.dec_lstm = nn.LSTM(input_size=self.targ_enc_size + self.fuse_enc_size + self.num_lat_classes + self.num_lon_classes,
                                      hidden_size=self.decoder_size)

        ''' Output layers '''
        self.op = nn.Linear(self.decoder_size, 5)
        
        if self.use_fusion or self.use_planning:  # 只要用到空间建模就缓存
            self.A_norm = build_grid_adj_matrix(self.grid_size, torch.device('npu'))
            self.A_norm = self.A_norm.half()  # 转低精度，适配Cube

    # pylint: disable=too-many-arguments,huawei-too-many-arguments
    def forward(self, nbsHist, nbsMask, planFut, planMask, targsHist, targsEncMask,
            lat_enc, lon_enc, idx, space_h=None, dv=None, v_pre=None):

        ''' Forward target vehicle's dynamic'''
        with autocast():  # Conv + Attention 可以混合精度
            dyn_enc = self.leaky_relu(self.temporalConv(targsHist.permute(1, 2, 0)))
            # targsHist.permute shape: (target_batch_size, local_x and y, hist_len)
            # dyn_enc shape: (target_batch_size, temporal_embedding_size, hist_len)
            dyn_enc = dyn_enc.permute(0, 2, 1)
            for attn in self.attn_layers_t:
                dyn_enc = attn(dyn_enc, dim=1)

        with autocast(enabled=False):  # LSTM 必须 FP32
            _, (dyn_enc, _) = self.nbh_lstm(dyn_enc.permute(1, 0, 2).float())
        # 取所有历史轨迹数据中最后一个时间步长的隐藏状态，其他两个输出不取 dyn_enc shape: (num_layers, target_batch_size, encoder_size)
        dyn_enc = self.leaky_relu(self.dyn_emb(dyn_enc.view(dyn_enc.shape[1], dyn_enc.shape[2])))
        dyn_enc = dyn_enc.to(dtype=torch.float16, device=dyn_enc.device)
        # 这里的目的是将第一层信息去除，重新更改数据维度，只保留batch_size和hidden_size
        # dyn_enc new shape: (target_batch_size, encoder_size)

        ''' Forward neighbour vehicles'''
        # 4. Conv 适合BF16，启用混合精度
        with autocast():
            nbrs_enc = self.leaky_relu(self.temporalConv(nbsHist.permute(1, 2, 0)))
            # shape is (nbs_batch_size, local x and y, hist_len)
        
        # 5. LSTM 必须FP32，禁用混合精度并强制输入为float
        with autocast(enabled=False):
            _, (nbrs_enc, _) = self.nbh_lstm(nbrs_enc.permute(2, 0, 1).float())
        nbrs_enc = nbrs_enc.view(nbrs_enc.shape[1], nbrs_enc.shape[2])
        nbrs_enc = nbrs_enc.to(dtype=torch.float16, device=nbrs_enc.device)

        ''' Masked neighbour vehicles'''
        nbrs_grid_static = torch.zeros_like(nbsMask, dtype=torch.float16, device=nbsMask.device)
        # nbsMask shape:(targets_batch_size, grid_size[1](5), grid_size[0](25), encoder_size)
        nbrs_grid_static = nbrs_grid_static.masked_scatter(nbsMask.bool(), nbrs_enc)  # 基于预计算非零索引的静态赋值，减少动态分支
        # 将网格中原本有车辆位置的地方替换为对应的编码结果，其余地方数值大小仍为0，形状一致
        nbrs_grid = nbrs_grid_static.permute(0, 3, 2, 1)  # 转回原维度，衔接后续处理
        # nbrs_grid shape:(targets_batch_size, encoder_size, 25, 5) → 记为 (BS, C_enc, H, W)
        
        # -------------------------- 新增邻接矩阵稠密图卷积 --------------------------
        # 提取网格维度（关键：避免变量未定义）
        BS, C_enc, H, W = nbrs_grid.shape  # 从nbrs_grid中获取维度，无需硬编码
        num_nodes = H * W  # 25*5=125，节点总数
        
        # 网格特征 → 节点特征矩阵（BS, num_nodes, C_enc）
        # 先将 (BS, C_enc, H, W) → (BS, C_enc, num_nodes) → 转置为 (BS, num_nodes, C_enc)
        nbrs_node_feat = nbrs_grid.reshape(BS, C_enc, num_nodes).permute(0, 2, 1)
        # 确保节点特征与邻接矩阵数据类型一致（均为FP16，适配Cube）
        nbrs_node_feat = nbrs_node_feat.half()
        
        # 稠密图卷积计算（A_norm × 节点特征，充分利用Cube引擎）
        with autocast():
            A_norm_batch = self.A_norm.expand(BS, -1, -1)
            nbrs_node_feat_fused = torch.bmm(A_norm_batch, nbrs_node_feat)
        
        # 节点特征 → 转回网格形状 + 残差连接（关键：保留原特征，避免信息丢失）
        nbrs_grid_fused = nbrs_node_feat_fused.permute(0, 2, 1).reshape(BS, C_enc, H, W)
        # 残差连接：图卷积特征 + 原网格特征（α为权重，初期让原特征占主导，模型逐步适应）
        alpha = 0.3  # 可调整，建议0.2~0.5
        nbrs_grid_fused = (1 - alpha) * nbrs_grid + alpha * nbrs_grid_fused  # 残差融合
        # ----------------------------------------------------------------------------------------
        
        # 卷积池化层适合BF16，启用混合精度
        with autocast():
            # 应用原第一个Conv2d（通道转换+空间压缩）
            nbrs_grid_fused = self.nbrs_conv_social[0](nbrs_grid_fused)  # 对应原nn.Conv2d(encoder_size, soc_conv_depth, 3)
            # 继续执行后续层（与原逻辑一致）
            nbrs_grid = self.nbrs_conv_social[1:](nbrs_grid_fused)  # 现在形状与原逻辑相同：(BS, soc_conv2_depth, 9, 1)


        if self.use_planning:
            ''' Forward planned vehicle'''
            # Conv 适合BF16，启用混合精度
            with autocast():
                plan_enc = self.leaky_relu(self.temporalConv(planFut.permute(1, 2, 0)))
            
            # LSTM 必须FP32，禁用混合精度并强制输入为float
            with autocast(enabled=False):
                _, (plan_enc, _) = self.plan_lstm(plan_enc.permute(2, 0, 1).float())
            
            plan_enc = plan_enc.view(plan_enc.shape[1], plan_enc.shape[2])
            plan_enc = plan_enc.to(dtype=torch.float16, device=plan_enc.device)

            ''' Masked planned vehicle'''
            plan_grid = torch.zeros_like(planMask, dtype=torch.float16, device=planMask.device)
            plan_grid = plan_grid.masked_scatter(planMask.bool(), plan_enc)
            plan_grid = plan_grid.permute(0, 3, 2, 1)  # 转成 (BS, encoder_size, 25, 5)，与nbrs_grid格式一致
            with autocast():
                plan_grid = self.plan_conv_social(plan_grid)

            ''' Merge neighbour and planned vehicle'''
            merge_grid = torch.cat((nbrs_grid, plan_grid), dim=3)
            merge_grid = self.pool_after_merge(merge_grid)
        else:
            merge_grid = self.pool_after_merge(nbrs_grid)
        social_context = merge_grid.view(-1, self.social_context_size)

        '''Concatenate social_context (neighbors + ego's planing) and dyn_enc, then place into the targsEncMask '''
        target_enc = torch.cat((social_context, dyn_enc), 1)
        target_grid = torch.zeros_like(targsEncMask, dtype=torch.float16, device=targsEncMask.device)
        target_grid = target_grid.masked_scatter(targsEncMask.bool(), target_enc)  # (BS, H=25, W=5, C=targ_enc_size)
        # ----------------------------------------------------------------------------------------
        if self.use_fusion:
            '''Fully Convolutional network to get a grid to be fused'''
            # 卷积、BN、转置卷积均适合BF16，启用混合精度
            with autocast():
                fuse_conv1 = self.fcn_conv1(target_grid.permute(0, 3, 2, 1))
                fuse_conv1 = self.fcn_pool1(fuse_conv1)
                fuse_conv2 = self.fcn_conv2(fuse_conv1)
                fuse_conv2 = self.fcn_pool2(fuse_conv2)
                # Encoder / Decoder #
                fuse_trans1 = self.relu(self.fcn_convTrans1(fuse_conv2))
                fuse_trans1 = self.back_bn1(fuse_trans1)
                fuse_trans2 = self.relu(self.fcn_convTrans2(fuse_trans1))
                fuse_trans2 = self.back_bn2(fuse_trans2)
            
            # Extract the location with targets
            fuse_grid_mask = targsEncMask[:, :, :, 0:self.fuse_enc_size]
            fuse_grid = torch.zeros_like(fuse_grid_mask, dtype=torch.float16, device=fuse_grid_mask.device)
            fuse_grid = fuse_grid.masked_scatter(fuse_grid_mask.bool(), fuse_trans2.permute(0, 3, 2, 1))

            '''Finally, Integrate everything together'''
            enc_rows_mark = targsEncMask[:, :, :, 0].view(-1)
            enc_rows = [i for i in range(len(enc_rows_mark)) if enc_rows_mark[i]]
            enc = torch.cat([target_grid, fuse_grid], dim=3)
            enc = enc.view(-1, self.fuse_enc_size + self.targ_enc_size)
            enc = enc[enc_rows, :]
        else:
            enc = target_enc

        '''Maneuver recognition'''
        # 线性层支持BF16，启用混合精度
        with autocast():
            lat_pred = self.softmax(self.op_lat(enc))
            lon_pred = self.softmax(self.op_lon(enc))
        
        if self.train_output_flag:
            enc = torch.cat((enc, lat_enc, lon_enc), 1)
            fut_pred = self.decode(enc)
            return fut_pred, lat_pred, lon_pred
        else:
            fut_pred = []
            for k in range(self.num_lon_classes):
                for l in range(self.num_lat_classes):
                    lat_enc_tmp = torch.zeros_like(lat_enc)
                    lon_enc_tmp = torch.zeros_like(lon_enc)
                    lat_enc_tmp[:, l] = 1
                    lon_enc_tmp[:, k] = 1
                    # Concatenate maneuver label before feeding to decoder
                    enc_tmp = torch.cat((enc, lat_enc_tmp, lon_enc_tmp), 1)
                    fut_pred.append(self.decode(enc_tmp))
            return fut_pred, lat_pred, lon_pred


    def decode(self, enc):
        # Decoder LSTM 必须FP32，禁用混合精度并强制输入为float
        with autocast(enabled=False):
            enc = enc.repeat(self.out_length, 1, 1).float()
            h_dec, _ = self.dec_lstm(enc)
        
        # 线性层和输出激活支持BF16，启用混合精度
        with autocast():
            h_dec = h_dec.permute(1, 0, 2)
            fut_pred = self.op(h_dec)
            fut_pred = fut_pred.permute(1, 0, 2)
            fut_pred = outputActivation(fut_pred)
        
        return fut_pred
