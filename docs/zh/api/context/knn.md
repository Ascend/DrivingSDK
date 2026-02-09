## knn
### 接口原型
```python
mx_driving.knn(int k, Tensor xyz, Tensor center_xyz, bool Transposed) -> Tensor
```
兼容：
```python
mx_driving.common.knn(int k, Tensor xyz, Tensor center_xyz, bool Transposed) -> Tensor
```
### 功能描述
对center_xyz中的每个点找到xyz中对应batch中的距离最近的k个点，并且返回此k个点的索引值。
### 参数说明
- `xyz(Tensor)`：点数据，表示(x, y, z)三维坐标，数据类型为`float32`。shape为`[B, N, 3]`(当Transposed=False)或`[B, 3, N]`(当Transposed=True)。其中`B`为batch size，`N`为点的数量。
- `center_xyz(Tensor)`：点数据，表示(x, y, z)三维坐标，数据类型为`float32`。shape为`[B, npoint, 3]`(当Transposed=False)或`[B, 3, npoint]`(当Transposed=True)。其中`B`为batch size，`npoint`为点的数量。
- `k(int)`：采样点的数量。
- `Transposed(bool)`: 输入是否需要进行转置。
### 返回值
- `idx(Tensor)`：采样后的索引数据，数据类型为`int32`。shape为`[B, k, npoint]`。
### 约束说明
1. k必须>0且<100。
2. xyz中的每个batch中的任意一个点到center_xyz对应batch中的任意一个点的距离必须在1e10f以内。
3. xyz和center_xyz的shape必须是3维，当Transposed=True时，xyz和center_xyz的shape的dim的第1维必须是3；当Transposed=False时，xyz和center_xyz的shape的dim的第2维必须是3。
4. 由于距离相同时排序为不稳定排序，存在距离精度通过但索引精度错误问题，与竞品无法完全对齐。
5. 性能在N值较大的场景下较优。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving import knn
xyz = torch.tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]], dtype=torch.float32).npu()
center_xyz = torch.tensor([[[1, 2, 3]], [[1, 2, 3]]], dtype=torch.float32).npu()
idx = knn(2, xyz, center_xyz, False)
```