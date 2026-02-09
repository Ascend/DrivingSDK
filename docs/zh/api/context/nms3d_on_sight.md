## nms3d_on_sight
### 接口原型
```python
mx_driving.nms3d_on_sight(Tensor boxes, Tensor scores, float threshold) -> Tensor
```

### 功能描述
在bev视角下删除3d候选box中bev距离(dist_bev)大于指定阈值(threshold)的候选框
### 参数说明
- `boxes (Tensor)`：输入的3d框，数据类型为`float32`，每个框的shape为`[x, y, z, dx, dy, dz, heading]`，box的shape为`[N, 7]`。
- `scores (Tensor)`：输入候选框的置信度分数，数据类型为`float32`，shape为`[N]`。
- `threshold (float)`：bev距离计算所比较的阈值，数据类型为`float32`。
### 返回值
- `order (Tensor)`：保留的boxes的索引。
### 约束说明
- dx, dy, dz的范围是(0, 100)
- heading的范围是(0, 1)
- 0 < N <= 2500
- 0 <= scores <= 1
- -0.5 <= threshold <= 1
- 由于距离相同时排序为不稳定排序，存在距离精度通过但索引精度错误问题，与竞品无法完全对齐。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import numpy as np
import torch
import torch_npu
from mx_driving import nms3d_on_sight

def generate_data_numpy(N):
    np.random.seed(7)
    shape = [N, 7]
    boxes_np = np.round(np.random.uniform(0, 100, shape), 2).astype(np.float32)
    angle_np = np.round(np.random.uniform(0, 1, shape[0]), 2).astype(np.float32)
    boxes_np[:, 6] = angle_np
    scores_np = np.round(np.random.uniform(0, 1, shape[0]), 4).astype(np.float32)
    boxes = torch.from_numpy(boxes_np)
    scores = torch.from_numpy(scores_np)
    threshold = np.random.uniform(-0.5, 1)
    return boxes, scores, threshold

boxes, scores, threshold = generate_data_numpy(N=100)
result = nms3d_on_sight(boxes.npu(), scores.npu(), threshold)
```