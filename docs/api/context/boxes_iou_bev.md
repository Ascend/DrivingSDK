## boxes_iou_bev
### 接口原型
```python
mx_driving.boxes_iou_bev(Tensor boxes_a, Tensor boxes_b) -> Tensor
```
### 功能描述
计算BEV视角下两个边界框的IoU（intersection over union）。
### 参数说明
- `boxes_a (Tensor)`：第一组bounding boxes，数据类型为`float32`。shape为`[M, 7]`。其中`7`分别代表`x_center, y_center, z_center, dx, dy, dz, angle`, `x_center, y_center, z_center`代表box的中心点坐标，`dx, dy, dz`代表box的长宽高，`angle`代表box的弧度制旋转角。
- `boxes_b (Tensor)`：第二组bounding boxes，数据类型为`float32`。shape为`[N, 7]`。其中`7`分别代表`x_center, y_center, z_center, dx, dy, dz, angle`, `x_center, y_center, z_center`代表box的中心点坐标，`dx, dy, dz`代表box的长宽高，`angle`代表box的弧度制旋转角。
### 返回值
- `ious (Tensor)`：包含两组bounding boxes的IoU的张量，数据类型为`float32`。shape为`[M, N]`。
### 约束说明
- `angle`的值在`[-pi, pi]`之间。
- `boxes_a`数量`M`与`boxes_b`数量`N`的乘积不超过9亿。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving import boxes_iou_bev
boxes_a = torch.tensor([[1.0, 1.0, 1.0, 3.0, 4.0, 1.0, 0.5]], dtype=torch.float32).npu()
boxes_b = torch.tensor([[0.0, 2.0, 1.0, 2.0, 5.0, 1.0, 0.3]], dtype=torch.float32).npu()
ious = boxes_iou_bev(boxes_a, boxes_b)
```