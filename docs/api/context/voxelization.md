## voxelization

### 接口原型
```python
mx_driving.Voxelization(Tensor points, List[float] voxel_size, List[float] coors_range, int max_points=-1, int max_voxels=-1, bool deterministic=True) -> Tensor
```

兼容：
```python
mx_driving.point.Voxelization(Tensor points, List[float] voxel_size, List[float] coors_range, int max_points=-1, int max_voxels=-1, bool deterministic=True) -> Tensor
```

### 功能描述
将点云数据进行体素化。

### 参数说明
- `points(Tensor)`：点云数据，数据类型为`float32`。shape为`[N, F]`。其中N为点的数量，F分别代表每个点的特征维度，其中`N > 0, F >= 3`。
- `voxel_size(List[float])`：体素大小，数据类型为`float32`。shape为`[3]`。其中3分别代表`x, y, z`。
- `coors_range(List[float])`：体素范围，数据类型为`float32`。shape为`[6]`。其中6分别代表`x_min, y_min, z_min, x_max, y_max, z_max`。
- `max_points(int)`：每个体素的最大点数。默认值为`-1`。
- `max_voxels(int)`：最大体素数。默认值为`-1`。
- `deterministic(bool)`：是否确定性。默认值为`True`。

### API语义
当max_num_points=-1且max_voxels=-1时，进行`dynamic_voxelize`计算，否则进行`hard_voxelize`

### 返回值
- `coors(Tensor)`：每个点所属的体素坐标，数据类型为`int32`。shape为`[N, 3]`。

### 支持的型号
Atlas A2 训练系列产品

### 调用示例
```python
import torch, torch_npu
from mx_driving import Voxelization
voxel_size = [0.5, 0.5, 0.5]
point_cloud_range = [0, -40, -3, 70.4, 0, 9]
max_num_points = -1
max_voxels = 3500
vlz = Voxelization(voxel_size, point_cloud_range, max_num_points, max_voxels)
points = torch.rand(16, 3, dtype = torch.float32).npu()
results = vlz(points)
```