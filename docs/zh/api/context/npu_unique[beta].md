## npu_unique[beta]

### 接口原型

```python
mx_driving.npu_unique(Tensor input) -> Tensor
```

### 功能描述

从小到大排序并去重. 提供一个输入`tensor`, 对`tensor`的输入进行排序, 并去掉`tensor`中的重复元素.

### 参数说明

- `input(Tensor)`：表示输入张量，数据类型支持 `float16`, `bfloat16`, `int16`, `float32`, `int32`, `int64`. shape 为 1 ~ 8 维的任意shape.

### 返回值

- `output(Tensor)`：表示输出张量，数据类型支持 `float16`, `bfloat16`, `int16`, `float32`, `int32`, `int64`, 与输入张量`input`一致. shape 为 1 维。

### 约束说明

- int32, int64输入时, 每个元素的值须在[-16777216, 16777216] (±2^24)之间，否则会引入精度损失.

### 支持的型号

- Atlas A2 训练系列产品

### 调用示例

```python
import torch, torch_npu
from mx_driving import npu_unique

rand_tensor = torch.rand(559794, dtype=torch.int64)
output = npu_unique(rand_tensor.npu())
```