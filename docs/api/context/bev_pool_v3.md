## bev_pool_v3
### 接口原型
```python
mx_driving.bev_pool_v3(Tensor depth, Tensor feat, Tensor ranks_depth, Tensor ranks_feat, Tensor ranks_bev,
                                 List[int] bev_feat_shape) -> Tensor
```
### 功能描述
BEV池化优化。`bev_pool_v1`和`bev_pool_v2`的NPU亲和版本，优先推荐使用。
### 参数说明
- `depth(Tensor)`：深度张量，数据类型为`float32`。shape为`[B, N, D, H, W]`。其中`B`为batch size，`N`为特征的数量，`D, H, W`分别代表深度、高度、宽度。
- `feat(Tensor)`：特征张量，数据类型为`float32`。shape为`[B, N, H, W, C]`。其中`B`为batch size，`N`为特征的数量，`H, W, C`分别代表高度、宽度、通道数。
- `ranks_depth(Tensor)`：深度排序张量，数据类型为`int32`。shape为`[N_RANKS]`。
- `ranks_feat(Tensor)`：特征排序张量，数据类型为`int32`。shape为`[N_RANKS]`。
- `ranks_bev(Tensor)`：BEV排序张量，数据类型为`int32`。shape为`[N_RANKS]`。
- `bev_feat_shape(List[int])`：BEV特征形状，数据类型为`int32`。长度为`5`， 分别代表`B, D, H, W, C`。
### 返回值
- `bev_pooled_feat(Tensor)`：BEV池化后的特征张量，数据类型为`float32`。shape为`[B, C, D, H, W]`。
### 约束说明
- `ranks_depth`的值必须在`[0, B*D*H*W]`之间。
- `ranks_feat`的值必须在`[0, B*N*H*W]`之间。
- `ranks_bev`的值必须在`[0, B*D*H*W]`之间。
- C 必须为8的倍数。
- B * D * H * W * C <= 2^31
- 对于反向也是同样的约束。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
bev_pool_v1 调用方式
```python
import torch, torch_npu
from mx_driving import bev_pool_v3
feat = torch.rand(4, 64).npu()
feat.requires_grad_()
geom_feat = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3]], dtype=torch.int32).npu()
bev_feat_shape = [4, 1, 128, 128, 64]
bev_pooled_feat = bev_pool_v3(None, feat, None, None, geom_feat, bev_feat_shape)
bev_pooled_feat.backward(torch.ones_like(bev_pooled_feat))
```
bev_pool_v2 调用方式
```python
import torch, torch_npu
from mx_driving import bev_pool_v3
depth = torch.rand(2, 1, 8, 256, 256).npu()
feat = torch.rand(2, 1, 256, 256, 64).npu()
feat.requires_grad_()
depth.requires_grad_()
ranks_depth = torch.tensor([0, 1], dtype=torch.int32).npu()
ranks_feat = torch.tensor([0, 1], dtype=torch.int32).npu()
ranks_bev = torch.tensor([0, 1], dtype=torch.int32).npu()
bev_feat_shape = [2, 8, 256, 256, 64]
bev_pooled_feat = bev_pool_v3(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape)
bev_pooled_feat.backward(torch.ones_like(bev_pooled_feat))
```
### 使用说明
`bev_pool_v3` 较[`bev_pool`](./bev_pool_v1.md)和[`bev_pool_v2`](./bev_pool_v2.md)做出了以下优化：
1. 避免使用`sort`来进行bev格子的排序，从而避免了相机特征和深度（v2)的重排序（在Ascend平台上不亲和，且收益不大）
2. 避免计算`interval_starts`和`interval_lengths`，使用了原子加的方式并行处理一个bev格子的数据
当输出空间固定时，性能提升会随特征数（n_ranks）的增加而增加;另一方面，当输出空间过大（B\*D\*H\*W\*C数量级在1e8）时，性能的提升并不明显。

## 替换建议
可参考模型[BEVFusion](../../../model_examples/BEVFusion/)替换bev_pool，参考模型[BEVDet](../../../model_examples/BEVDet/)替换bev_pool_v2.
更具体地，对于bev_pool，可以参考调用示例简单替换；对于bev_pool_v2，应该关注bev_pool_v2调用前的计算部分,一般叫`voxel_pooling_prepare_v2`函数，保留`ranks_feat`,`ranks_depth`, `ranks_bev`计算部分，去除之后的排序，`interval_lengths`和`interval_starts`的相关计算。