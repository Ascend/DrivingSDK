## points_in_box
### 接口原型
```python
mx_driving.points_in_box(Tensor boxes, Tensor points) -> Tensor
```
兼容：
```python
mx_driving.preprocess.npu_points_in_box(Tensor boxes, Tensor points) -> Tensor
```

### 功能描述
判断点是否在框内。
### 参数说明
- `boxes(Tensor)`：框张量，数据类型为`float32`。shape 为`[B, M, 7]`。`7`分别代表`x, y, z, x_size, y_size, z_size, rz`。
- `points(Tensor)`：点张量，数据类型为`float32`。shape 为`[B, N, 3]`。`3`分别代表`x, y, z`。
### 返回值
- `boxes_idx_of_points(Tensor)`：点在框内的索引张量，数据类型为`int32`。shape 为`[B, N]`。
### 约束说明
- `boxes`和`points`的`B`必须相同，且只能为`1`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving import points_in_box
boxes = torch.tensor([[[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9]]], dtype=torch.float32).npu()
points = torch.tensor([[[1, 2, 3], [3, 4, 5]]], dtype=torch.float32).npu()
out = points_in_box(boxes, points)
```