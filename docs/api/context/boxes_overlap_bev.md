## boxes_overlap_bev
### 接口原型
```python
mx_driving.boxes_overlap_bev(Tensor boxes_a, Tensor boxes_b) -> Tensor
```
兼容：
```python
mx_driving.detection.boxes_overlap_bev(Tensor boxes_a, Tensor boxes_b) -> Tensor
```
```python
mx_driving.detection.npu_boxes_overlap_bev(Tensor boxes_a, Tensor boxes_b) -> Tensor
```
### 功能描述
计算BEV视角下两个边界框的重叠面积。
### 参数说明
- `boxes_a (Tensor)`：第一组bounding boxes，数据类型为`float32`。shape为`[M, 5]`。其中`5`分别代表`x1, y1, x2, y2, angle`, `x1, y1, x2, y2`代表box四个顶点的横纵坐标，`angle`代表box的弧度制旋转角。
- `boxes_b (Tensor)`：第二组bounding boxes，数据类型为`float32`。shape为`[N, 5]`。其中`5`分别代表`x1, y1, x2, y2, angle`, `x1, y1, x2, y2`代表box四个顶点的横纵坐标，`angle`代表box的弧度制旋转角。
### 返回值
- `area_overlap (Tensor)`：包含两组bounding boxes交叠面积的张量，数据类型为`float32`。shape为`[M, N]`。
### 约束说明
- `angle`的值在`[-pi, pi]`之间。
- `boxes_a`数量`M`与`boxes_b`数量`N`的乘积不超过9亿。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving import boxes_overlap_bev
boxes_a = torch.tensor([[0, 0, 2, 2, 0]], dtype=torch.float32).npu()
boxes_b = torch.tensor([[1, 1, 3, 3, 0]], dtype=torch.float32).npu()
area_overlap = boxes_overlap_bev(boxes_a, boxes_b)
```