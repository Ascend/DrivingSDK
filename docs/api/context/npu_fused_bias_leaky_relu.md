## npu_fused_bias_leaky_relu
### 接口原型
```python
mx_driving.npu_fused_bias_leaky_relu(Tensor x, Tensor bias, float negative_slope, float scale) -> Tensor
```
兼容：
```python
mx_driving.fused.npu_fused_bias_leaky_relu(Tensor x, Tensor bias, float negative_slope, float scale) -> Tensor
```
### 功能描述
实现对输入x进行偏置后的值，即x+bias进行LeakyReLU。
$$
f(x)=
\begin{cases}
scale * (x + bias), \quad x + bias \geq 0 \\
scale * negative\_slope * (x + bias), \quad x + bias < 0 \\
\end{cases}
$$
### 参数说明
- `x(Tensor)`：输入张量，数据类型为`float32`。
- `bias(Tensor)`：1维张量，维度大小与x的第2维度保持一致，数据类型为`float32`。
- `negative_slope(float)`：标量，数据类型为`float32`。
- `scale(float)`：标量，数据类型为`float32`。
### 返回值
- `output(Tensor)`：输出结果张量，数据类型为`float32`。
### 支持的型号
- Atlas A2 训练系列产品
### 约束说明
- 3<=len(x.shape)<=8
- bias为1维张量，bias.shape[0]==x.shape[1]
- 总数据规模不要超过10亿大小
- negative_slope与scale不能为inf或-inf
### 调用示例
```python
import torch, torch_npu
import numpy as np
from mx_driving import npu_fused_bias_leaky_relu

negative_slope = -0.1
scale = 0.25

B, N, H, W = [18, 256, 232, 400]
x = np.random.uniform(1, 1, [B, N, H, W]).astype(np.float32)
x = torch.from_numpy(x)
bias = np.random.uniform(-2.0, 2.0, N).astype(np.float32)
bias = torch.from_numpy(bias)

out = npu_fused_bias_leaky_relu(x.npu(), bias.npu(), negative_slope, scale)
```