## furthest_point_sample_with_dist
### 接口原型
```python
mx_driving.furthest_point_sample_with_dist(Tensor points, int num_points) -> Tensor
```
兼容：
```python
mx_driving.point.furthest_point_sample_with_dist(Tensor points, int num_points) -> Tensor
```
### 功能描述
与`npu_furthest_point_sampling`功能相同，但输入略有不同。
### 参数说明
- `points(Tensor)`：点云数据，表示各点间的距离，数据类型为`float32`。shape为`[B, N, N]`。其中`B`为batch size，`N`为点的数量。
- `num_points(int)`：采样点的数量。
### 返回值
- `output(Tensor)`：采样后的点云数据，数据类型为`float32`。shape为`[B, num_points]`。
### 约束说明
1. 性能在N值较大的场景下较优。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving import furthest_point_sample_with_dist
points = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32).npu()
out = furthest_point_sample_with_dist(points, 2)
```