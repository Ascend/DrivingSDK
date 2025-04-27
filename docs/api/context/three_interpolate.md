## three_interpolate
### 接口原型
```python
mx_driving.three_interpolate(features: torch.Tensor, indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor
```
兼容：
```python
mx_driving.common.three_interpolate(features: torch.Tensor, indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor
```
### 功能描述
对三维数据进行加权最近邻线性插值处理
### 参数说明
- `features(Tensor)`：需要被插值的特征，数据类型为`float32|float16`，维度为（B, C, M）。
- `indices(Tensor)`：获取目标特征计算的索引，数据类型为`int32`，维度为（B, N, 3），
  - `indices`的元素值需小于`features`的第三维度，即值在[0, M)。
- `weight(Tensor)`：获取目标特征计算的权重，数据类型为`float32|float16`，维度为（B, N, 3）。
  - `weight`数据类型与`features`须一致。
- `features`，`indices`，`weights`三个参数的每个维度须小于10000。
- `features`，`indices`，`weights`三个参数的大小请勿超过2^24。
### 返回值
- `output(Tensor)`：目标特征张量，数据类型为`float32|float16`，维度为（B, C, N）。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving import three_interpolate


features = torch.tensor(
            [[[2.4350, 4.7516, 4.4995, 2.4350, 2.4350, 2.4350],
            [3.1236, 2.6278, 3.0447, 3.1236, 3.1236, 3.1236],
            [2.6732, 2.8677, 2.6436, 2.6732, 2.6732, 2.6732],
            [0.0124, 7.0150, 7.0199, 0.0124, 0.0124, 0.0124],
            [0.3207, 0.0000, 0.3411, 0.3207, 0.3207, 0.3207]],
            [[0.0000, 0.9544, 2.4532, 0.0000, 0.0000, 0.0000],
            [0.5346, 1.9176, 1.4715, 0.5346, 0.5346, 0.5346],
            [0.0000, 0.2744, 2.0842, 0.0000, 0.0000, 0.0000],
            [0.3414, 1.5063, 1.6209, 0.3414, 0.3414, 0.3414],
            [0.5814, 0.0103, 0.0000, 0.5814, 0.5814, 0.5814]]],
            ).npu()
idx = torch.tensor(
            [[[0, 1, 2], [2, 3, 4], [2, 3, 4], [0, 1, 2], [0, 1, 2], [0, 1, 3]],
            [[0, 2, 3], [1, 3, 4], [2, 1, 4], [0, 2, 4], [0, 2, 4], [0, 1, 2]]],
            ).int().npu()
weight = torch.tensor(
            [[[3.3333e-01, 3.3333e-01, 3.3333e-01],
              [1.0000e+00, 5.8155e-08, 2.2373e-08],
              [1.0000e+00, 1.7737e-08, 1.7356e-08],
              [3.3333e-01, 3.3333e-01, 3.3333e-01],
              [3.3333e-01, 3.3333e-01, 3.3333e-01],
              [3.3333e-01, 3.3333e-01, 3.3333e-01]],
             [[3.3333e-01, 3.3333e-01, 3.3333e-01],
              [1.0000e+00, 1.3651e-08, 7.7312e-09],
              [1.0000e+00, 1.7148e-08, 1.4070e-08],
              [3.3333e-01, 3.3333e-01, 3.3333e-01],
              [3.3333e-01, 3.3333e-01, 3.3333e-01],
              [3.3333e-01, 3.3333e-01, 3.3333e-01]]],
            ).npu()

features.requires_grad = True
output = three_interpolate(features, idx, weight)
grad_out_tensor = torch.ones_like(output)
output.backward(grad_out_tensor)
```