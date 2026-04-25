# npu_fake_tensor_quant[beta]

## 接口原型

```python
mx_driving.npu_fake_tensor_quant(Tensor inputs, Tensor amax, int num_bits=8, bool is_unsigned=False, bool narrow_range=True) -> Tensor
```

```python
mx_driving.npu_fake_tensor_quant_inplace(Tensor inputs, Tensor amax, int num_bits=8, bool is_unsigned=False, bool narrow_range=True) -> Tensor
```

```python
mx_driving.npu_fake_tensor_quant_with_axis(Tensor inputs, Tensor amax, int axis, int num_bits=8, bool is_unsigned=False, bool narrow_range=True) -> Tensor
```

## 功能描述

实现假量化（Fake Quantization）功能，用于量化感知训练（QAT）。将浮点输入张量模拟量化为低精度表示，然后反量化回浮点数，从而在训练过程中模拟量化误差。

- `npu_fake_tensor_quant`：对整个张量使用单一 `amax` 值进行标量量化。
- `npu_fake_tensor_quant_inplace`：与 `npu_fake_tensor_quant` 功能相同，但原地修改输入张量。
- `npu_fake_tensor_quant_with_axis`：沿指定 `axis` 维度，每个切片使用独立的 `amax` 值进行按轴量化。

量化公式如下：

```python
bound = (1 << (num_bits - 1 + int(is_unsigned))) - 1
max_bound = bound
min_bound = -(bound + int(not narrow_range))
scale = max_bound / amax
output = clamp(round(input * scale), min_bound, max_bound) / scale
```

## 参数说明

### npu_fake_tensor_quant / npu_fake_tensor_quant_inplace

- `inputs (Tensor)`：输入张量，数据类型为 `float32` 或 `float16`。Shape 无限制。
- `amax (Tensor)`：最大值标量张量，数据类型与 `inputs` 一致。Shape 为 `[1]` 的标量。
- `num_bits (int)`：量化位数，表示量化后的精度，数据类型为 `int32`，默认值为 8。典型取值为 4 或 8。
- `is_unsigned (bool)`：是否使用无符号量化，数据类型为 `bool`，默认值为 `False`。
  - `False`：有符号量化，量化范围为 `[-bound, bound]`
  - `True`：无符号量化，量化范围为 `[0, bound]`
- `narrow_range (bool)`：是否使用窄范围量化，数据类型为 `bool`，默认值为 `True`。
  - `True`：窄范围，`min_bound = -bound`
  - `False`：宽范围，`min_bound = -(bound + 1)`

### npu_fake_tensor_quant_with_axis

- `inputs (Tensor)`：输入张量，数据类型为 `float32` 或 `float16`。Shape 无限制。
- `amax (Tensor)`：每通道的最大值张量，数据类型与 `inputs` 一致。Shape 为 `[inputs.shape[axis]]`。
- `axis (int)`：量化轴，表示沿哪个维度进行按通道量化，数据类型为 `int32`。
- `num_bits (int)`：量化位数，默认值为 8。
- `is_unsigned (bool)`：是否使用无符号量化，默认值为 `False`。
- `narrow_range (bool)`：是否使用窄范围量化，默认值为 `True`。

## 返回值

- `output (Tensor)`：量化后的张量，数据类型与 `inputs` 一致，Shape 与 `inputs` 相同。

## 算子约束

### 通用约束

- `inputs` 和 `amax` 的数据类型必须一致，仅支持 `float32` 或 `float16`。
- `inputs` 和 `amax` 必须为连续张量（contiguous）。
- `inputs` 不支持 `inf`、`-inf` 和 `nan`。
- `num_bits` 取值范围为 [1, 32]

### npu_fake_tensor_quant / npu_fake_tensor_quant_inplace

- `amax` 的 Shape 必须为 `[1]` 的标量。

### npu_fake_tensor_quant_with_axis

- `amax` 的 Shape 必须为 `[inputs.shape[axis]]`，即与 `inputs` 在 `axis` 维度的长度一致。
- `axis` 取值范围为 `[0, inputs.dim() - 1]`。
- 当 `axis` 为最后一维时，算子会自动切换到特殊优化模式，性能更优。

## 使用建议

- **标量量化**：适用于整个张量使用单一缩放因子的场景，如激活值量化。
- **按轴量化**：适用于每通道独立量化的场景，如权重量化（通常 `axis=0`）。
- **原地版本**：`npu_fake_tensor_quant_inplace` 可节省显存，但会修改输入张量。

## 性能验证

|场景|shape|pytorch实现耗时(us)|算子实现耗时(us)|
| --- | --- | --- | --- |
|全局量化|[100663296]|3387.99|559.09|
|通道(轴=1)|[32,8,512,768]|4114.19|1147.98|

## 支持的型号

- Atlas A2 训练系列产品

## 调用示例

### 示例 1：标量量化（激活值量化）

```python
import torch
import torch_npu
from mx_driving import npu_fake_tensor_quant

# 创建输入张量
inputs = torch.randn(4, 64, 32, 32, dtype=torch.float32, device='npu')

# 计算 amax（整个张量的最大绝对值）
amax = inputs.abs().max()

# 标量量化（8bit 有符号窄范围）
quantized = npu_fake_tensor_quant(inputs, amax, num_bits=8, is_unsigned=False, narrow_range=True)
```

### 示例 2：原地量化（节省显存）

```python
import torch
import torch_npu
from mx_driving import npu_fake_tensor_quant_inplace

inputs = torch.randn(1000, 256, dtype=torch.float32, device='npu')

amax = inputs.abs().max()

# 原地量化，输入张量会被修改
quantized = npu_fake_tensor_quant_inplace(inputs, amax, num_bits=8)
```

### 示例 3：按轴量化（权重量化）

```python
import torch
import torch_npu
from mx_driving import npu_fake_tensor_quant_with_axis

# 创建 4D 权重张量 (N, C, H, W)
weight = torch.randn(64, 128, 3, 3, dtype=torch.float32, device='npu')

# 沿通道轴（axis=0）计算每通道的 amax
amax = weight.abs().amax(dim=[1, 2, 3])  # Shape: [64]

# 按通道量化（8bit 有符号窄范围）
quantized_weight = npu_fake_tensor_quant_with_axis(
    weight, amax, axis=0, num_bits=8, is_unsigned=False, narrow_range=True
)
```

## 应用场景

1. **量化感知训练（QAT）**：在训练过程中模拟量化误差，提升模型在 INT8 推理时的精度。
2. **模型压缩**：将 FP32 模型转换为低精度表示，减少显存占用和带宽需求。
3. **混合精度训练**：部分层使用量化，部分层保持 FP32，平衡精度和性能。
4. **激活值量化**：使用标量量化（`npu_fake_tensor_quant`）。
5. **权重量化**：使用按轴量化（`npu_fake_tensor_quant_with_axis`）。
