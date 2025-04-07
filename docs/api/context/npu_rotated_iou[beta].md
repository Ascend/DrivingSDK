## npu_rotated_iou[beta]
### 接口原型
```python
mx_driving.npu_rotated_iou(Tensor self, Tensor query_boxes, bool trans=False, int mode=0, bool is_cross=True, float v_threshold=0.0, float e_threshold=0.0) -> Tensor
```
兼容：
```python
mx_driving.detection.npu_rotated_iou(Tensor self, Tensor query_boxes, bool trans=False, int mode=0, bool is_cross=True, float v_threshold=0.0, float e_threshold=0.0) -> Tensor
```
### 功能描述
计算旋转框的IoU。
### 参数说明
- `self(Tensor)`：边界框张量，数据类型为`float32, float16`，形状为`[B, N, 5]`。
- `query_boxes(Tensor)`：查询框张量，数据类型为`float32, float16`，形状为`[B, M, 5]`。
- `trans(bool)`：是否进行坐标变换。默认值为`False`。值为`True`时，表示`xyxyt`, 值为`False`时，表示`xywht`，其中`t`为角度制。
- `is_cross(bool)`：值为`True`时，则对两组边界框中每个边界框之间进行计算。值为`False`时，只对对齐的边界框之间进行计算。
- `mode(int)`：计算IoU的模式。默认值为`0`。值为`0`时，表示计算`IoU`，值为`1`时，表示计算`IoF`。
- `v_threshold(float)`：顶点判断的容忍阈值。
- `e_threshold(float)`：边相交判断的容忍阈值。
### 返回值
- `output(Tensor)`：IoU张量，数据类型为`float32, float16`，`is_cross`为`True`时形状为`[B, N, M]，反之则为`[B, N]`。
### 约束说明
- `mode`的取值范围为`{0, 1}`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
import numpy as np
from mx_driving import npu_rotated_iou
a = np.random.uniform(0, 1, (2, 2, 5)).astype(np.float16)
b = np.random.uniform(0, 1, (2, 3, 5)).astype(np.float16)
box1 = torch.from_numpy(a).npu()
box2 = torch.from_numpy(b).npu()
iou = npu_rotated_iou(box1, box2, False, 0, True, 1e-5, 1e-5)
```