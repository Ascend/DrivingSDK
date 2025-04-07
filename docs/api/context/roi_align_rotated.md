## roi_align_rotated
### 接口原型
```python
mx_driving.roi_align_rotated(Tensor feature_map, Tensor rois, float: spatial_scale,
                             int: sampling_ratio, int: pooled_height, int: pooled_width, bool: aligned, bool: clockwise) -> Tensor
```
兼容：
```python
mx_driving.detection.roi_align_rotated(Tensor feature_map, Tensor rois, float: spatial_scale,
                                       int: sampling_ratio, int: pooled_height, int: pooled_width, bool: aligned, bool: clockwise) -> Tensor
```
### 功能描述
计算旋转候选框的RoI Align池化特征图。
### 参数说明
- `feature map(Tensor)`：特征图张量，数据类型为`float32`，形状为`[B, C, H, W]`。
- `rois(Tensor)`：感兴趣区域张量，数据类型为`float32`，形状为`[n, 6]`。
- `spatial_scale(float)`：感兴趣区域边界框的缩放率，数据类型为`float32`。
- `sampling_ratio(int)`：采样率，数据类型为`int`。取值范围为非负整数。
- `pooled_height(int)`：池化特征图高度，数据类型为`int`。
- `pooled_width(int)`：池化特征图宽度，数据类型为`int`。
- `aligned(bool)`：是否对齐，数据类型为`bool`。值为`True`时，表示对齐, 值为`False`时，表示不对齐。
- `clockwise(bool)`：旋转候选框的旋转方向，数据类型为`bool`。值为`True`时，表示逆时针旋转，值为`False`时，表示顺时针旋转。
### 返回值
- `output(Tensor)`：池化特征图张量，数据类型为`float32`，形状为`[n, C, pooled_height, pooled_width]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import math
import torch, torch_npu
import numpy as np
from mx_driving import roi_align_rotated

feature_map = torch.rand([1, 3, 16, 16])
feature_map.requires_grad = True
rois = torch.Tensor(6, 8)
rois[0] = torch.randint(0, 1, (8,))
rois[1].uniform_(0, 16)
rois[2].uniform_(0, 16)
rois[3].uniform_(0, 16)
rois[4].uniform_(0, 16)
rois[5].uniform_(0, math.pi)

output = roi_align_rotated(feature_map.npu(), rois.npu(), 1, 1, 7, 7, True, True)
output.backward(torch.ones_like(output))
```
### 其他说明
在双线性插值采样过程中，当采样点`x`接近`-1`或`W`位置，`y`接近`-1`或`H`位置时，由于平台差异和计算误差，可能导致该采样点的精度无法与竞品精度完全对齐。
在反向梯度回传过程中，由于涉及到原子累加与浮点数大数吃小数问题，当特征图上某个点梯度回传次数大于15000时，可能导致该点计算结果波动，无法与竞品精度完全对齐。