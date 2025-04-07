## group_points[beta]
### 接口原型
```python
mx_driving.group_points(Tensor features, Tensor indices) -> Tensor
```
兼容：
```python
mx_driving.point.group_points(Tensor features, Tensor indices) -> Tensor
mx_driving.point.npu_group_points(Tensor features, Tensor indices) -> Tensor
```
### 功能描述
点云数据按照索引重新分组。
### 参数说明
- `features(Tensor)`：需要被插值的特征，数据类型为`float32`，维度为（B, C, N）。
- `indices(Tensor)`：获取目标特征计算的索引，数据类型为`int32`，维度为（B, npoints, nsample）。
### 返回值
- `output(Tensor)`：分组后的点云数据，数据类型为`float32`。shape为`[B, C, npoints, nsample]`。
### 约束说明
- `indices`的元素值需小于`features`的第三维度，即值在[0, N)。
- C <= 1024
- 反向具有相同约束。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch
import torch_npu
from mx_driving import group_points


indices = torch.tensor([[[0, 2, 5, 5], [1, 0, 5, 0], [2, 1, 4, 4]]]).int().npu()
features = torch.tensor([[[0.9178, -0.7250, -1.6587, 0.0715, -0.2252, 0.4994],
                          [0.6190, 0.1755, -1.7902, -0.5852, -0.3311, 1.9764],
                          [1.7567, 0.0740, -1.1414, 0.4705, -0.3197, 1.1944],
                          [-0.2343, 0.1194, 0.4306, 1.3780, -1.4282, -0.6377],
                          [0.7239, 0.2321, -0.6578, -1.1395, -2.3874, 1.1281]]],
                          dtype=torch.float32).npu()
features.requires_grad = True
output = group_points(features, indices)
grad_out_tensor = torch.ones_like(output)
output.backward(grad_out_tensor)
```