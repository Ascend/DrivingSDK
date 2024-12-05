## dynamic_scatter
### 接口原型
```python
mx_driving.dynamic_scatter(Tensor feats, Tensor coors, str reduce_type = 'max') -> Tuple[torch.Tensor, torch.Tensor]
```
兼容：
```python
mx_driving.point.npu_dynamic_scatter(Tensor feats, Tensor coors, str reduce_type = 'max') -> Tuple[torch.Tensor, torch.Tensor]
```
### 功能描述
将点云特征点在对应体素中进行特征压缩。
### 参数说明
- `feats(Tensor)`：点云特征张量[N, C]，仅支持两维，数据类型为`float32`，特征向量`C`长度上限为2048。
- `coors(Tensor)`：体素坐标映射张量[N, 3]，仅支持两维，数据类型为`int32`，此处以x, y, z指代体素三维坐标，其取值范围为`0 <= x, y < 2048`,  `0 <= z < 256`。
- `reduce_type(str)`：压缩类型。可选值为`'max'`, `'mean'`, `'sum'`。默认值为`'max'`
### 返回值
- `voxel_feats(Tensor)`：压缩后的体素特征张量，仅支持两维，数据类型为`float32`。
- `voxel_coors(Tensor)`：去重后的体素坐标，仅支持两维，数据类型为`int32`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving import dynamic_scatter

feats = torch.tensor([[1, 2, 3], [3, 2, 1], [7, 8, 9], [9, 8, 7]], dtype=torch.float32).npu()
feats.requires_grad = True
coors = torch.tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]], dtype=torch.int32).npu()
voxel_feats, voxel_coors = dynamic_scatter(feats, coors, 'max')
voxel_feats.backward(torch.ones_like(voxel_feats))

```