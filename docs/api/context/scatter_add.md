## scatter_add
### 接口原型
```python
mx_driving.scatter_add(Tensor src, Tensor indices, int dim=0， Tensor out=None, int dim_size=None) -> Tensor
```
### 功能描述
将输入张量`src`中的元素按照`indices`中的索引在指定的`dim`维进行分组，并对每组进行求和，求和后的结果放在`out`中。
### 参数说明
- `src (Tensor)`：源张量 (Tensor)，数据类型为`float32`。
- `indices (Tensor)`：索引张量 (Tensor)，数据类型为`int32`。
- `out (Tensor)`：被更新张量 (Tensor)，数据类型为`float32`，可选入参，默认为`None`，输入`out`不为`None`时，`out`中的元素参与求和的计算。
- `dim (int)`：指定的维度，表示按照哪个维度进行分组求和计算，数据类型为`int32`，可选入参，默认取值为`0`。
- `dim_size (int)`：输出张量在`dim`维的长度，数据类型为`int32`，可选入参，默认为`None`，该参数仅在输入`out`为`None`时生效。
### 返回值
- `out (Tensor)`：求和后的张量 (Tensor)，数据类型为`float32`。
### 算子约束
- `indices`的维度必须小于等于`src`的维度，且每一维的长度均必须与`src`长度相同。
- `indices`的取值必须为非负的有效索引值，参数`out`或`data_size`不为`None`时，`indices`的取值应该为输出张量在`dim`维的有效索引值。
- `out`的维度必须与`src`的维度相同，且除第`dim`维外其余维的长度必须与`src`相同。
- `dim`取值不能超过`indices`的维度。
- `dim_size`的取值必须为非负的有效长度值。
- `src`和`out`不支持`inf`、`-inf`和`nan`。
- 该算子的正反向均对尾块较大的场景较为亲和，对尾块很小的场景不亲和，其中，尾块表示`src`后`N`维的大小，`N = src.dim() - indices.dim()`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例

```python
import torch, torch_npu
from mx_driving import scatter_add
src = torch.randn(4, 5, 6).to(torch.float)
indices = torch.randint(5, (4, 5)).to(torch.int32)
dim = 0
src.requires_grad = True
out = scatter_add(src.npu(), indices.npu(), None, dim)
grad_out_tensor = torch.ones_like(out)
out.backward(grad_out_tensor)
```
