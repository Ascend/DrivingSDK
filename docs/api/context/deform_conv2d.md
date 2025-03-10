## deform_conv2d(DeformConv2dFunction.apply)
### 接口原型
```python
mx_driving.deform_conv2d(Tensor x, Tensor offset, Tensor weight, Union[int, Tuple[int, ...]] stride, Union[int, Tuple[int, ...]] padding, Union[int, Tuple[int, ...]] dilation, int groups, int deformable_groups) -> Tensor
```
兼容：
```
mx_driving.fused.deform_conv2d(Tensor x, Tensor offset, Tensor weight, Union[int, Tuple[int, ...]] stride, Union[int, Tuple[int, ...]] padding, Union[int, Tuple[int, ...]] dilation, int groups, int deformable_groups) -> Tensor
```
### 功能描述
可变形卷积。
### 参数说明
- `x(Tensor)`：输入特征，数据类型为`float32`，shape为`(n, c_in, h_in, w_in)`，其中`n`为 batch size，`c_in`为输入特征的通道数量，`h_in`为输入特征图的高，`w_in`为输入特征图的宽。
- `offset(Tensor)`：偏移量，数据类型为`float32`，shape 为`(n, 2 * k * k, h_out, w_out)`，其中`n`为 batch size，`k` 为卷积核大小，`h_out` 为输出特征图高，`w_out` 为输出特征图的宽。
- `weight(Tensor)`：卷积核权重，数据类型为`float32`，shape 为 `(c_out, c_in, k, k)`，其中 `c_out` 为输出的通道数，`c_in` 为输入的通道数，`k` 为卷积核大小。
- `stride(Union)`：卷积步长。
- `padding(Union)`：卷积的填充大小。
- `dilation(Union)`：空洞卷积大小。
- `groups(int)`：分组卷积大小，需要可以整除`c_in`, `c_out`。
- `deformable_groups(int)`：将通道分成几组计算offsets，当前只支持1。
### 返回值
- `output(Tensor)`：输出张量，数据类型为`float32`，shape 为 `(n, c_out, h_out, w_out)`，其中`n`为 batch size，`c_out`为输出通道，`h_out` 为输出特征图高，`w_out` 为输出特征图的宽。
### 支持的型号
- Atlas A2 训练系列产品
### 约束说明
1. `deformable_groups`和`groups`当前只支持1。
2. `h_in`,`w_in`,`h_out`,`w_out`需满足
$$
w_{out}=(w_{in}+ 2 * padding - (dilation * (k - 1) + 1)) / stride + 1 \\
h_{out}=(h_{in}+ 2 * padding - (dilation * (k - 1) + 1)) / stride + 1
$$
3. `c_in`需要为64的倍数。
### 调用示例
```python
import torch
import torch_npu
from mx_driving import deform_conv2d, DeformConv2dFunction

n, c_in, h_in, w_in = 16, 64, 100, 200
c_out, k, h_out, w_out = 64, 3, 50, 100

x = torch.randn((n, c_in, h_in, w_in)).npu()
offset = torch.randn((n, 2 * k * k, h_out, w_out)).npu()
weight = torch.randn((c_out, c_in, k, k)).npu()
x.requires_grad_()
offset.requires_grad_()
weight.requires_grad_()
stride = 1
padding = 1
dilation = 1
groups = 1
deformable_groups = 1

output = deform_conv2d(x, offset, weight, stride, padding, dilation, groups, deformable_groups)
output = DeformConv2dFunction.apply(x, offset, weight, stride, padding, dilation, groups, deformable_groups)
output.backward(torch.ones_like(output))
```
