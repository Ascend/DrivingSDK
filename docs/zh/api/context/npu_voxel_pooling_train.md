## npu_voxel_pooling_train
### 接口原型
```python
mx_driving.npu_voxel_pooling_train(Tensor geom_xyz, Tensor input_features, List[int] voxel_num) -> Tensor
```
兼容：
```
mx_driving.point.npu_voxel_pooling_train(Tensor geom_xyz, Tensor input_features, List[int] voxel_num) -> Tensor
```
### 功能描述
点云数据体素化。
### 参数说明
- `geom_xyz(Tensor)`：体素坐标，数据类型为`int32`，维度为（B, N, 3）, 3表示x, y, z。
- `input_features(Tensor)`：点云数据，数据类型为`float32|float16`，维度为（B, N, C）。
- `voxel_num(List[int])`：体素格子长宽高，数据类型为`int32`，维度为（3），3表示体素格子的长宽高。
### 返回值
- `output(Tensor)`：输出结果，数据类型为`float32|float16`。shape为`[B, num_voxel_y, num_voxel_x, C]`。
### 约束说明
- B <= 128
- N <= 100000
- C <= 256
- num_voxel_x <= 1000
- num_voxel_y <= 1000
- num_voxel_z <= 10
- B * num_voxel_y * num_voxel_x * C <= 100000000
- B * N * C <= 100000000
- 反向具有相同约束。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch
import torch_npu
from mx_driving import npu_voxel_pooling_train

def gen_data(geom_shape, feature_shape, coeff, batch_size, num_channels, dtype):
       geom_xyz = torch.rand(geom_shape) * coeff
       geom_xyz = geom_xyz.reshape(batch_size, -1, 3)
       geom_xyz[:, :, 2] /= 100
       geom_xyz_cpu = geom_xyz.int()
       features = torch.rand(feature_shape, dtype=dtype) - 0.5
       features_cpu = features.reshape(batch_size, -1, num_channels)

       return geom_xyz_cpu, features_cpu

dtype = torch.float32
coeff = 90
voxel_num = [128, 128, 1]
batch_size = 2
num_points = 40
num_channel = 80
xyz = 3

geom_shape = [batch_size, num_points, xyz]
feature_shape = [batch_size, num_points, num_channel]

geom_cpu, feature_cpu = gen_data(geom_shape, feature_shape, coeff, batch_size, num_channel, dtype)

geom_npu = geom_cpu.npu()
feature_npu = feature_cpu.npu()
feature_npu.requires_grad = True

result_npu = npu_voxel_pooling_train(geom_npu, feature_npu, voxel_num)
result_npu.backward(torch.ones_like(result_npu))