## scatter_max
### 接口原型
```python
mx_driving.scatter_max(Tensor updates, Tensor indices, Tensor out=None) -> (Tensor out, Tensor argmax)
```
兼容：
```python
mx_driving.common.scatter_max(Tensor updates, Tensor indices, Tensor out=None) -> (Tensor out, Tensor argmax)
```
### 功能描述
在第0维上，将输入张量`updates`中的元素按照`indices`中的索引进行分散，然后在第0维上取最大值，返回最大值和对应的索引。对于1维张量，公式如下：
$$out_i = max(out_i, max_j(updates_j))$$
$$argmax_i = argmax_j(updates_j)$$
这里，$i = indices_j$。
### 参数说明
- `updates(Tensor)`：更新源张量，数据类型为`float32`。
- `indices(Tensor)`：索引张量，数据类型为`int32`。
- `out(Tensor)`：被更新张量，数据类型为`float32`，默认为`None`。
### 返回值
- `out(Tensor)`：更新后的张量，数据类型为`float32`。
- `argmax(Tensor)`：最大值对应的索引张量，数据类型为`int32`。
### 算子约束
- `updates`的第0维外其余轴合轴后必须32字节对齐。
- `indices`的维度必须为`1`，`indices`第0维的长度必须与`updates`第0维的长度相同。
- `indices`的取值必须为非负的有效索引值，且`indices`的最大值必须小于`491520`。
- `out`的维度必须与`updates`的维度相同，且除第0维外其余维的长度必须与`updates`相同。
- 反向仅支持`updates`的维度为`2`，其余约束与正向相同。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving import scatter_max
updates = torch.tensor([[2, 0, 1, 3, 1, 0, 0, 4], [0, 2, 1, 3, 0, 3, 4, 2], [1, 2, 3, 4, 4, 3, 2, 1]], dtype=torch.float32).npu()
indices = torch.tensor([0, 2, 0], dtype=torch.int32).npu()
updates.requires_grad = True
out = updates.new_zeros((3, 8))
out, argmax = scatter_max(updates, indices, out)
grad_out_tensor = torch.ones_like(out)
out.backward(grad_out_tensor)
```