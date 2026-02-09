## box_iou_quadri
### 接口原型
```python
mx_driving.box_iou_quadri(Tensor boxes_a, Tensor boxes_b, str mode='iou', bool aligned=False) -> Tensor
```
兼容：
```python
mx_driving.detection.box_iou_quadri(Tensor boxes_a, Tensor boxes_b, str mode='iou', bool aligned=False) -> Tensor
```
### 功能描述
计算两个边界框的IoU。
### 参数说明
- `boxes_a (Tensor)`：第一组bounding boxes，数据类型为`float32`。shape为`[M, 8]`。其中`8`分别代表`x1, y1, x2, y2, x3, y3, x4, y4`, 表示box四个顶点的横纵坐标。
- `boxes_b (Tensor)`：第二组bounding boxes，数据类型为`float32`。shape为`[N, 8]`。其中`8`分别代表`x1, y1, x2, y2, x3, y3, x4, y4`, 表示box四个顶点的横纵坐标。
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
from mx_driving import box_iou_quadri
boxes_a = torch.tensor([[7.0, 7.0, 8.0, 8.0, 9.0, 7.0, 8.0, 6.0]], dtype=torch.float32).npu()
boxes_b = torch.tensor([[7.0, 6.0, 7.0, 8.0, 9.0, 8.0, 9.0, 6.0]], dtype=torch.float32).npu()
ious = box_iou_quadri(boxes_a, boxes_b, mode="iou", aligned=False)
```