## npu_index_select[beta]
### 接口原型
```python
mx_driving.npu_index_select(Tensor feature, Int dim, Tensor index) -> Tensor
```
### 功能描述
从输入`feature`的指定维度`dim`，按`index`中的下标序号提取元素，保存到`out`中。
例如，对于输入张量 $feature=\begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix}$ 和索引张量 $index=[1, 0]$，
$mx\_driving.npu\_index\_select(feature, 0, index)$ 的结果： $out=\begin{bmatrix}4 & 5 & 6 \\ 1 & 2 & 3\end{bmatrix}$。
### 参数说明
- `feature(Tensor)`：待提取张量，数据类型支持`FLOAT、FLOAT16、INT32、INT16`，维度仅支持二维，支持非连续的`Tensor`。
- `dim(Int)`：提取维度，数据类型支持`INT64`。
- `index(Tensor)`：提取索引，数据类型支持`INT64、INT32`，仅支持一维`Tensor`，支持非连续的`Tensor`。
### 返回值
- `out(Tensor)`：输出`Tensor`，数据类型与`feature`一致，维度为两维，维度为`[index.shape[0], feature.shape[1]]`。
### 约束说明
- `feature`仅支持二维`Tensor`；
- `dim`仅支持0和-2；
- `index`不支持负索引和越界索引，即取值范围为`[0, feature.shape[0])`；
- 算子反向依赖`aclnnIndexAddV2`接口，因此需配置2025年6月18日之后的cann包才能生效；
- 反向具有相同约束。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving import npu_index_select

x = torch.randn(3, 4)
index = torch.tensor([0, 2])
npu_output = npu_index_select(x.npu(), 0, index.npu())
npu_output.backward(torch.ones_like(npu_output))
```