## grid_sampler3d_v1

### 接口原型

```python
mx_driving.grid_sampler3d_v1(Tensor input, Tensor grid, str mode="bilinear", str padding_mode="zeros", bool align_corners=False) -> Tensor
```

### 功能描述

网格采样。提供一个输入 tensor 以及一个对应的 grid 网格，根据 grid 中每个位置提供的坐标信息，将 input 中对应位置的像素值（插值得到）填充到网格指定的位置，得到最终的输出 tensor。

### 参数说明

- `input(Tensor)`：表示输入张量，数据类型为 `float32`，shape 为 $(N, C, D_{in}, H_{in}, W_{in})$。
- `grid(Tensor)`：表示网格张量，数据类型为 `float32`，shape 为 $(N, D_{out}, H_{out}, W_{out}, 3)$，3 代表 `x, y, z`。grid 的元素值归一化到 `[-1, 1]`。
- `mode(str)`：表示插值模式，取值为 `"bilinear"` 时，计算双线性插值。
- `padding_mode(str)`：表示填充模式，表明对越界坐标的处理方式。取值为 `"zeros"` 时，越界位置填充为 `0`。
- `align_corners(bool)`：表示特征图坐标与特征值的对应方式。取值为 `True` 时，特征值位于像素中心。

### 返回值

- `output(Tensor)`：表示输出张量，数据类型为 `float32`，shape 为 $(N, C, D_{out}, H_{out}, W_{out})$。

### 约束说明

- `input` 和 `grid` 必须为 5 维张量，且二者 batch size 必须相同。
- `input` 和 `grid` 均不支持 `inf` 、`-inf` 和 `nan`，不支持空 tensor。
- `grid` 最后一维必须为 3，元素值需归一化到 `[-1, 1]`。
- `input`、 `grid` 和 `output` 的元素个数在 int32 范围内，并且每个元素的偏移量均能用 32 位索引表示。
- mode仅支持 `"bilinear"`，padding_mode仅支持`"zeros"`，align_corners仅支持`"True"`，input要求 C <= 128，且为4的倍数。
- SparseBEV模型中的算子配置作为模型case，所有参数配置需要与模型配置保持一致。即 `input: (64, 64, 6，64, 176), grid: (64, 1460, 4, 1, 3), "bilinear", "zeros", True`。

### 支持的型号

- Atlas A2 训练系列产品

### 调用示例

```python
import torch, torch_npu
from mx_driving import grid_sampler3d_v1

inp = torch.rand(2, 6, 4, 4, 4, dtype=torch.float32).npu()
rand_tensor = torch.rand(2, 1, 4, 2, 3, dtype=torch.float32).npu()
grid = 2 * rand_tensor - 1
inp.requires_grad = True
grid.requires_grad = True

output = grid_sampler3d_v1(inp, grid, "bilinear", "zeros", True)
output.backward(torch.ones_like(output))
```