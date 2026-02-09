## three_nn
### 接口原型
```python
mx_driving.three_nn(Tensor target, Tensor source) -> (Tensor dist, Tensor idx)
```
兼容：
```python
mx_driving.common.three_nn(Tensor target, Tensor source) -> (Tensor dist, Tensor idx)
```
### 功能描述
对target中的每个点找到source中对应batch中的距离最近的3个点，并且返回此3个点的距离和索引值。
### 参数说明
- `target(Tensor)`：点数据，表示(x, y, z)三维坐标，数据类型为`float32/float16`。shape为`[B, npoint, 3]`。其中`B`为batch size，`npoint`为点的数量。
- `source(Tensor)`：点数据，表示(x, y, z)三维坐标，数据类型为`float32/float16`。shape为`[B, N, 3]`。其中`B`为batch size，`N`为点的数量。
### 返回值
- `dist(Tensor)`：采样后的索引数据，数据类型为`float32/float16`。shape为`[B, npoint, 3]`。
- `idx(Tensor)`：采样后的索引数据，数据类型为`int32/int32`。shape为`[B, npoint, 3]`。
### 算子约束
1. source和target的shape必须是3维，且source和target的shape的dim的第2维必须是3。
2. 距离相同时排序为不稳定排序，存在距离精度通过但索引精度错误问题，与竞品无法完全对齐。
3. 性能在N值较大的场景下较优。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving import three_nn
source = torch.tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]], dtype=torch.float32).npu()
target = torch.tensor([[[1, 2, 3]], [[1, 2, 3]]], dtype=torch.float32).npu()
dist, idx = three_nn(target, source)
```