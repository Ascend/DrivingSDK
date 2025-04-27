## bev_pool_v2
### 接口原型
```python
mx_driving.bev_pool_v2(Tensor depth, Tensor feat, Tensor ranks_depth, Tensor ranks_feat, Tensor ranks_bev,
                                 List[int] bev_feat_shape, Tensor interval_starts, Tensor interval_lengths) -> Tensor
```
> 注意：可以使用性能更好的[bev_pool_v3](./bev_pool_v3.md)替换。
兼容：
```
mx_driving.point.bev_pool_v2(Tensor depth, Tensor feat, Tensor ranks_depth, Tensor ranks_feat, Tensor ranks_bev,
                                 List[int] bev_feat_shape, Tensor interval_starts, Tensor interval_lengths) -> Tensor
```
### 功能描述
BEV池化优化。可参考论文`BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View`。
### 参数说明
- `depth(Tensor)`：深度张量，数据类型为`float32`。shape为`[B, N, D, H, W]`。其中`B`为batch size，`N`为特征的数量，`D, H, W`分别代表深度、高度、宽度。
- `feat(Tensor)`：特征张量，数据类型为`float32`。shape为`[B, N, H, W, C]`。其中`B`为batch size，`N`为特征的数量，`H, W, C`分别代表高度、宽度、通道数。
- `ranks_depth(Tensor)`：深度排序张量，数据类型为`int32`。shape为`[N_RANKS]`。
- `ranks_feat(Tensor)`：特征排序张量，数据类型为`int32`。shape为`[N_RANKS]`。
- `ranks_bev(Tensor)`：BEV排序张量，数据类型为`int32`。shape为`[N_RANKS]`。
- `bev_feat_shape(List[int])`：BEV特征形状，数据类型为`int32`。长度为`5`， 分别代表`B, D, H, W, C`。
- `interval_starts(Tensor)`：间隔开始张量，数据类型为`int32`。shape为`[N_INTERVALS]`。
- `interval_lengths(Tensor)`：间隔长度张量，数据类型为`int32`。shape为`[N_INTERVALS]`。
### 返回值
- `bev_pooled_feat(Tensor)`：BEV池化后的特征张量，数据类型为`float32`。shape为`[B, C, D, H, W]`。
### 约束说明
- `ranks_depth`的值必须在`[0, B*B*D*H*W]`之间。
- `ranks_feat`的值必须在`[0, B*N*H*W]`之间。
- `ranks_bev`的值必须在`[0, B*D*H*W]`之间。
- C <= 1024
- B * D * H * W * C <= 2^31, B, D <= 8, H, W <= 256
- N_RANKS <= 2^21
- 对于反向也是同样的约束。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving import bev_pool_v2
depth = torch.rand(2, 1, 8, 256, 256).npu()
feat = torch.rand(2, 1, 256, 256, 64).npu()
feat.requires_grad_()
ranks_depth = torch.tensor([0, 1], dtype=torch.int32).npu()
ranks_feat = torch.tensor([0, 1], dtype=torch.int32).npu()
ranks_bev = torch.tensor([0, 1], dtype=torch.int32).npu()
bev_feat_shape = [2, 8, 256, 256, 64]
interval_starts = torch.tensor([0], dtype=torch.int32).npu()
interval_lengths = torch.tensor([2], dtype=torch.int32).npu()
bev_pooled_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape, interval_starts, interval_lengths)
bev_pooled_feat.backward(torch.ones_like(bev_pooled_feat))
```