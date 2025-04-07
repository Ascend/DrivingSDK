## unique_voxel
### 接口原型
```python
mx_driving.unique_voxel(Tensor voxels) -> int, Tensor, Tensor, Tensor, Tensor
```
### 功能描述
对输入的点云数据进行去重处理。
### 参数说明
- `voxels(Tensor)`：数据语义为索引，数据类型为`int32`，shape为`[N]`。
### 返回值
- `num_voxels(int)`, 体素数量。
- `uni_voxels(Tensor)`，去重后的体素数据，数据类型为`int32`，shape为`[num_voxels]`。
- `uni_indices(Tensor)`, 去重后的索引数据，数据类型为`int32`，shape为`[num_voxels]`。
- `argsort_indices(Tensor)`, 排序后的索引数据，数据类型为`int32`，shape为`[N]`。
- `uni_argsort_indices(Tensor)`, 去重后的排序后的索引数据，数据类型为`int32`，shape为`[num_voxels]`。
### 约束说明
N的大小受限于内存大小，建议N小于等于2^32。

受限于芯片指令，输入的数据类型只能是int32，且>=0,<2^30。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch
import torch_npu
import numpy as np
from mx_driving import unique_voxel
voxels = np.random.randint(0, 1024, (100000,)).astype(np.int32)
voxels_npu = torch.from_numpy(voxels).npu()
num_voxels, uni_voxels, uni_indices, argsort_indices, uni_argsort_indices = unique_voxel(voxels_npu)

```