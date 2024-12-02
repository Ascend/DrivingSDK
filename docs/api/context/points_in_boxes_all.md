## npu_points_in_box_all
Note: 该接口命名将于2025年改为`points_in_boxes_all`。
### 接口原型
```python
mx_driving.npu_points_in_box_all(Tensor boxes, Tensor points) -> Tensor
```
兼容：
```python
mx_driving.preprocess.npu_points_in_box_all(Tensor boxes, Tensor points) -> Tensor
```
### 功能描述
判断点是否在框内。
### 参数说明
- `boxes(Tensor)`：框张量，数据类型为`float32`。shape 为`[B, M, 7]`。`7`分别代表`x, y, z, x_size, y_size, z_size, rz`。
- `points(Tensor)`：点张量，数据类型为`float32`。shape 为`[B, N, 3]`。`3`分别代表`x, y, z`。
### 返回值
- `boxes_idx_of_points(Tensor)`：同一`batch`下，各点是否在各框内的张量，数据类型为`int32`。shape 为`[B, N, M]`。
### 约束说明
- `boxes`和`points`的`B`必须相同。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving import npu_points_in_box_all
boxes = torch.tensor([[[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9]]], dtype=torch.float32).npu()
points = torch.tensor([[[1, 2, 5], [3, 4, 8]]], dtype=torch.float32).npu()
out = npu_points_in_box_all(boxes, points)
```