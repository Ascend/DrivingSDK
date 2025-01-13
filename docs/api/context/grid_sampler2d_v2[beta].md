## grid_sampler2d_v2[beta]

### 接口原型

```python
mx_driving.grid_sampler2d_v2(Tensor input, Tensor grid, str mode="bilinear", str padding_mode="zeros", bool align_corners=False) -> Tensor
```

### 功能描述

网格采样。提供一个输入 tensor 以及一个对应的 grid 网格，根据 grid 中每个位置提供的坐标信息，将 input 中对应位置的像素值填充到网格指定的位置，得到最终的输出 tensor。

### 参数说明

- `input(Tensor)`：表示输入张量，数据类型为 `float32`，shape 为 $(N, C, H_{in}, W_{in})$。
- `grid(Tensor)`：表示网格张量，数据类型为 `float32`，shape 为 $(N, H_{out}, W_{out}, 2)$，2 代表 `x, y`。grid 的元素值通常被归一化到 `[-1, 1]`。
- `mode(str)`：表示插值模式，取值为 `"bilinear"` 时，计算双线性插值。
- `padding_mode(str)`：表示填充模式，表明对越界坐标的处理方式。取值为 `"zeros"` 时，越界位置填充为 `0`；取值为 `"border"` 时，使用边界值填充；取值为 `"reflection"` 时，用边界值的对称值填充。
- `align_corners(bool)`：表示特征图坐标与特征值的对应方式。取值为 `True` 时，特征值位于像素中心；取值为 `False` 时，特征值位于像素角点。

### 返回值

- `output(Tensor)`：表示输出张量，数据类型为 `float32`，shape 为 $(N, C, H_{out}, W_{out})$。

### 约束说明

- input 和 grid 必须为 4 维张量，且二者 batch size 必须相同。
- input 和 grid 均不支持 inf 和 nan。
- grid 最后一维必须为 2，元素值需归一化到 `[-1, 1]`。
- input, grid 和 output 的元素个数在 int32 范围内，并且每个元素的偏移量均能用 32 位索引表示。
- mode 目前仅支持 `"bilinear"`，使用其它模式时将调用 canndev 接口。
- input 的 channel 目前仅支持 8 以内，超过范围时将调用 canndev 接口。

### 支持的型号

- Atlas A2 训练系列产品

### 调用示例

```python
import torch, torch_npu
from mx_driving import grid_sampler2d_v2

inp = torch.rand(2, 4, 3, 4, dtype=torch.float32)
rand_tensor = torch.rand(2, 2, 3, 2, dtype=torch.float32)
grid = 2 * rand_tensor - 1
output = grid_sampler2d_v2(input.npu(), grid.npu(), "bilinear", "zeros", False)
```

