## bev_pool
### 接口原型
```python
mx_driving.bev_pool(Tensor feat, Tensor geom_feat, int B, int D, int H, int W) -> Tensor
```
> 注意：可以使用性能更好的[bev_pool_v3](./bev_pool_v3.md)替换。
兼容：
```
mx_driving.point.bev_pool(Tensor feat, Tensor geom_feat, int B, int D, int H, int W) -> Tensor
```

### 功能描述
BEV池化。可参考论文`BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation`
### 参数说明
- `feat(Tensor)`：特征张量，数据类型为`float32`。shape为`[N, C]`。其中`N`为原特征张量拉伸后的数量，`C`为特征的维度。
- `geom_feat(Tensor)`：输出坐标张量，数据类型为`int32`。shape为`[N, 4]`。其中`4`分别代表`h, w, b, d`。
- `B(int)`：batch size。
- `D(int)`：输出池化深度。
- `H(int)`：输出池化高度。
- `W(int)`：输出池化宽度。
### 返回值
- `bev_pooled_feat(Tensor)`：采样后的点云数据，数据类型为`float32`。shape为`[B, C, D, H, W]`。
### 支持的型号
- Atlas A2 训练系列产品
### 约束说明
- `geom_feat`的4个对应的值必须在`[0, H-1]`, `[0, W-1]`, `[0, B-1]`, `[0, D-1]`之间。
- `geom_feat`和`feat`的第0维长度必须相同。
- C <= 1024
- B * D * H * W * C <= 2^31, B, D <= 8, H, W <= 256
- 对于反向也是同样的约束。
### 调用示例
```python
import torch, torch_npu
from mx_driving import bev_pool
feat = torch.rand(4, 256).npu()
feat.requires_grad_()
geom_feat = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3]], dtype=torch.int32).npu()
bev_pooled_feat = bev_pool(feat, geom_feat, 4, 1, 256, 256)
bev_pooled_feat.backward(torch.ones_like(bev_pooled_feat))
```