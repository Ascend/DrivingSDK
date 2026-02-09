## pixel_group
### 接口原型
```python
mx_driving.pixel_group(Tensor score, Tensor mask, Tensor embedding, Tensor kernel_label, Tensor kernel_contour, int kernel_region_num, float distance_threshold) -> List[List]
```
兼容：
```python
mx_driving.detection.pixel_group(Tensor score, Tensor mask, Tensor embedding, Tensor kernel_label, Tensor kernel_contour, int kernel_region_num, float distance_threshold) -> List[List]
```
### 功能描述
根据像素之间的嵌入向量和距离，将未被分组的像素分组。
### 参数说明
- `score (Tensor)`：前景得分矩阵，数据类型为`float32`，shape为`[Height, Width]`。
- `mask (Tensor)`：前景掩码矩阵，标记可被分组的像素，数据类型为`bool`，shape为`[Height, Width]`。
- `embedding (Tensor)`：特征向量，数据类型为`float32`，shape为`[Height, Width, Embedding_dim]`。
- `kernel_label (Tensor)`：像素的实例标签，数据类型为`int32`，shape为`[Height, Width]`。
- `kernel_contour (Tensor)`：内核的边界像素，数据类型为`uint8`，shape为`[Height, Width]`。
- `kernel_region_num (int)`：不同内核（分组）的数量，数据类型为`int`。
- `distance_threshold (float)`：嵌入向量的距离阈值，数据类型为`float`。
### 返回值
- `pixel_assignment (List)`：像素的分组信息，数据类型为`float32`，length为入参`kernel_region_num`。
### 约束说明
- mask = score > 0.5
- `score`的取值范围在`[0, 1]`之间
- `kernel_label`的最大值为`kernel_region_num`-1
- `kernel_contour`的取值非0即1
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
import numpy as np
from mx_driving import pixel_group
H, W, dim, num = 10, 10, 8, 3
score = np.random.uniform(0, 1, [H, W]).astype(np.float32)
score = torch.from_numpy(score).npu()
mask = (score) > 0.5
embedding = np.random.uniform(0, 10, [H, W, dim]).astype(np.float32)
embedding = torch.from_numpy(embedding).npu()
kernel_label = np.random.uniform(0, num, [H, W]).astype(np.int32)
kernel_label = torch.from_numpy(kernel_label).npu()
kernel_contour = np.random.uniform(0, 1, [H, W]).astype(np.uint8)
kernel_contour = torch.from_numpy(kernel_contour).npu()
kernel_region_num = num
distance_threshold = 0.8

output = pixel_group(score, mask, embedding, kernel_label, kernel_contour, kernel_region_num, distance_threshold)
```