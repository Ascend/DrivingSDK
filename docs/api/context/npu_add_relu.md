## npu_add_relu
### 接口原型
```python
mx_driving.npu_add_relu(Tensor x, Tensor y) -> Tensor
```
兼容：
```
mx_driving.fused.npu_add_relu(Tensor x, Tensor y) -> Tensor
```
### 功能描述
与`relu(x + y)`功能相同。
### 参数说明
- `x(Tensor)`：输入数据，数据类型为`float32`，shape无限制。
- `y(Tensor)`：输入数据，数据类型为`float32`，shape需要和x一致。
### 返回值
- `output(Tensor)`：输出数据，数据类型为`float32`，shape和x一致。
### 约束说明
- 输入`x`与输入`y`的shape和dtype需要保持一致，不支持广播。
- 仅在x的元素个数超过2000000时，相较于`relu(x + y)`有性能提升。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.fused import npu_add_relu
x = torch.tensor([[[1, 2, 3], [-1, 5, 6], [7, 8, 9]]], dtype=torch.float32).npu()
y = torch.tensor([[[1, 2, 3], [-1, -2, 6], [7, 8, 9]]], dtype=torch.float32).npu()
x.requires_grad = True
y.requires_grad = True
out = npu_add_relu(x, y)
out.backward(torch.ones_like(out))
```