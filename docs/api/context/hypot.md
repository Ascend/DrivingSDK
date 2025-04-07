## hypot
### 接口原型
```python
mx_driving.hypot(Tensor input, Tensor other) -> Tensor
```
兼容：
```python
mx_driving.common.hypot(Tensor input, Tensor other) -> Tensor
```
### 功能描述
给出直角三角形的两边，返回它的斜边。
### 参数说明
- `input(Tensor)`：代表直角三角形第一条直角边的输入张量，数据类型为`float32`。
- `other(Tensor)`：代表直角三角形第二条直角边的输入张量，数据类型为`float32`。
### 返回值
- `output(Tensor)`：经过计算后的直角三角形斜边，数据类型为`float32`。
### 算子约束
1. input和other的shape必须是可广播的。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving import hypot
input = torch.tensor([3,3,3], dtype=torch.float32).npu()
other = torch.tensor([4,4,4], dtype=torch.float32).npu()
out = hypot(input, other) # tensor([5.,5.,5.])
```
