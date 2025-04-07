## furthest_point_sampling
### 接口原型
```python
mx_driving.furthest_point_sampling(Tensor points, int num_points) -> Tensor
```
兼容
```python
mx_driving.point.npu_furthest_point_sampling(Tensor points, int num_points) -> Tensor
```
### 功能描述
点云数据的最远点采样。
### 参数说明
- `points(Tensor)`：点云数据，数据类型为`float32`。shape为`[B, N, 3]`。其中`B`为batch size，`N`为点的数量，`3`分别代表`x, y, z`。
- `num_points(int)`：采样点的数量。
### 返回值
- `output(Tensor)`：采样后的点云数据，数据类型为`float32`。shape为`[B, num_points]`。
### 算子约束
1. points输入shape[B, N, 3]的总大小(B x N x 3)不应该超过383166
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving import furthest_point_sampling
points = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32).npu()
out = furthest_point_sampling(points, 2)
```