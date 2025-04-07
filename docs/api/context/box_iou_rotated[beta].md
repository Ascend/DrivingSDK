## box_iou_rotated[beta]
### 接口原型
```python
mx_driving.box_iou_rotated(Tensor boxes_a, Tensor boxes_b, str mode='iou', bool aligned=False) -> Tensor
```
### 功能描述
计算两个边界框的IoU。
### 参数说明
- `boxes_a (Tensor)`：第一组bounding boxes，数据类型为`float32`。shape为`[M, 5]`。其中`5`分别代表`x_center, y_center, dx, dy, angle`, `x_center, y_center`代表box的中心点坐标，`dx, dy`代表box的长宽，`angle`代表box的弧度制旋转角。
- `boxes_b (Tensor)`：第二组bounding boxes，数据类型为`float32`。shape为`[M, 5]`。其中`5`分别代表`x_center, y_center, dx, dy, angle`, `x_center, y_center`代表box的中心点坐标，`dx, dy`代表box的长宽，`angle`代表box的弧度制旋转角。
- `mode (str)`：取值为`"iou"`时，计算IoU（intersection over union）；取值为`"iof"`时，计算IoF（intersection over foregroud）。
- `aligned (bool)`：取值为`True`时，只计算配对的box之间的结果；取值为`False`时，计算每对box之间的结果。
### 返回值
- `ious (Tensor)`：包含两组bounding boxes的IoU（`mode="iou"`）或IoF（`mode="iof"`）的张量，数据类型为`float32`。shape为`[M]`（`aligned=True`）或`[M, N]`（`aligned=False`）。
### 约束说明
- `mode`的取值范围为`{'iou', 'iof'}`。
- 当`aligned=False`时，`boxes_a`数量`M`与`boxes_b`数量`N`的乘积不超过9亿。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving import box_iou_rotated
boxes_a = torch.tensor([[1.0, 1.0, 3.0, 4.0, 0.5]], dtype=torch.float32).npu()
boxes_b = torch.tensor([[0.0, 2.0, 2.0, 5.0, 0.3]], dtype=torch.float32).npu()
ious = box_iou_rotated(boxes_a, boxes_b, mode="iou", aligned=False)
```