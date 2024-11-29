## border_align
### 接口原型
```python
mx_driving.border_align(Tensor feature_map, Tensor rois, int pooled_size) -> Tensor
```
兼容：
```python
mx_driving.detection.border_align(Tensor feature_map, Tensor rois, int pooled_size) -> Tensor
```
### 功能描述
对输入的RoI框进行边缘特征提取。
### 参数说明
- `feature_map (Tensor)`：输入的特征图，数据类型为`float32`，shape为`[Batch_size, Channels, Height, Width]`。
- `rois (Tensor)`：输入的RoI框坐标，数据类型为`int32`，shape为`[Batch_size, Height * Width, 4]`。
- `pooled_size (int)`：在每条边上的采样点数，数据类型为`int`。
### 返回值
- `out_features (Tensor)`：提取到的RoI框特征，数据类型为`float32`，shape为`[Batch_size, Channels / 4, Height * Width, 4]`。
### 约束说明
- Batch_size <= 128
- Channels <= 8192, Channels % 4 == 0
- Height <= 256, Width <= 256
- 2 <= pooled_size <= 20
- 反向具有相同约束。
- 算子在Channels较大时性能更优。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch
import torch_npu
import numpy as np
from mx_driving import border_align

def generate_grad_outputs(output_shape):
    grad_outputs = torch.rand(output_shape)
    return grad_outputs

def generate_features(feature_shape):
    features = torch.rand(feature_shape)
    return features

def generate_rois(inputs):
    num_boxes = inputs.shape[0] * inputs.shape[2] * inputs.shape[3]
    xyxy = torch.rand(num_boxes, 4)
    xyxy[:, 0::2] = xyxy[:, 0::2] * inputs.size(3)
    xyxy[:, 1::2] = xyxy[:, 1::2] * inputs.size(2)
    xyxy[:, 2:] = xyxy[:, 0:2] + xyxy[:, 2:]
    rois = xyxy.view(inputs.shape[0], -1, 4).contiguous()
    return rois

batch_size = 2
input_channels = 16
input_height = 8
input_width = 8
pooled_size = 3
features = generate_features([batch_size, input_channels, input_height, input_width])
features.requires_grad = True
grad_output = generate_grad_outputs([batch_size, input_channels // 4, input_height * input_width, 4])
rois = generate_rois(features)
output = border_align(features.npu(), rois.npu(), pooled_size)
output.backward(grad_output.npu())
```