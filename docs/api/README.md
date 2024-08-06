> Note: 以prototype标注的接口，表示该接口为预发布接口，可能会有变动，不建议在生产环境中使用。
# Common 算子
## scatter_max
### 接口原型
```python
mx_driving.common.scatter_max(Tensor updates, Tensor indices, Tensor out=None) -> (Tensor out, Tensor argmax)
```
### 功能描述
在第0维上，将输入张量`updates`中的元素按照`indices`中的索引进行分散，然后在第0维上取最大值，返回最大值和对应的索引。对于1维张量，公式如下：
$$out_i = max(out_i, max_j(updates_j))$$
$$argmax_i = argmax_j(updates_j)$$
这里，$i = indices_j$。
### 参数说明
- `updates`：更新源张量，数据类型为`float32`，且
  - `updates`的第0维外其余轴合轴后必须32字节对齐。
- `indices`：索引张量，数据类型为`int32`，且
  - `indices`的维度必须为`1`，
  - `indices`第0维的长度必须与`updates`第0维的长度相同。
  - `indices`的最大值必须小于`491520`。
  - `indices`的取值必须为非负的有效索引值。
- `out`：被更新张量，数据类型为`float32`，默认为`None`,且
  - `out`的维度必须与`updates`的维度相同。
  - `out`除第0维外其余维的长度必须与`updates`相同。
### 返回值
- `out`：更新后的张量，数据类型为`float32`。
- `argmax`：最大值对应的索引张量，数据类型为`int32`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.common import scatter_max
updates = torch.tensor([[2, 0, 1, 3, 1, 0, 0, 4], [0, 2, 1, 3, 0, 3, 4, 2], [1, 2, 3, 4, 4, 3, 2, 1]], dtype=torch.float32).npu()
indices = torch.tensor([0, 2, 0], dtype=torch.int32).npu()
out = updates.new_zeros((3, 8))
out, argmax = scatter_max(updates, indices, out)
print(out)
print(argmax)
```
```text
tensor([[2., 2., 3., 4., 4., 3., 2., 4.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 2., 1., 3., 0., 3., 4., 2.]])
tensor([[0, 2, 2, 2, 2, 2, 2, 0],
        [3, 3, 3, 3, 3, 3, 3, 3],
        [1, 1, 1, 1, 1, 1, 1, 1]])
```
## npu_rotated_overlaps
### 接口原型
```python
mx_driving.common.npu_rotated_overlaps(Tensor self, Tensor query_boxes, bool trans=False) -> Tensor
```
### 功能描述
计算旋转框的重叠面积。
### 参数说明
- `self(Tensor)`：边界框张量，数据类型为`float32, float16`，形状为`[B, N, 5]`。
- `query_boxes(Tensor)`：查询框张量，数据类型为`float32, float16`，形状为`[B, M, 5]`。
- `trans(bool)`：是否进行坐标变换。默认值为`False`。值为`True`时，表示`xyxyt`, 值为`False`时，表示`xywht`。
### 返回值
- `Tensor`：重叠面积张量，数据类型为`float32, float16`，形状为`[B, N, M]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
import numpy as np
from mx_driving.common import npu_rotated_overlaps
a = np.random.uniform(0, 1, (1, 3, 5)).astype(np.float16)
b = np.random.uniform(0, 1, (1, 2, 5)).astype(np.float16)
box1 = torch.from_numpy(a).npu()
box2 = torch.from_numpy(b).npu()
output = npu_rotated_overlaps(box1, box2, True)
print(output)
```
```text
tensor([[[0.0000, 0.1562, 0.0000],
         [0.1562, 0.3713, 0.0611],
         [0.0000, 0.0611, 0.0000]]], dtype=torch.float16)
```
## npu_rotated_iou
### 接口原型
```python
mx_driving.common.npu_rotated_iou(Tensor self, Tensor query_boxes, bool trans=False, int mode=0, bool is_cross=True, float v_threshold=0.0, float e_threshold=0.0) -> Tensor
```
### 功能描述
计算旋转框的IoU。
### 参数说明
- `self(Tensor)`：边界框张量，数据类型为`float32, float16`，形状为`[B, N, 5]`。
- `query_boxes(Tensor)`：查询框张量，数据类型为`float32, float16`，形状为`[B, M, 5]`。
- `trans(bool)`：是否进行坐标变换。默认值为`False`。值为`True`时，表示`xyxyt`, 值为`False`时，表示`xywht`，其中`t`为角度制。
- `is_cross(bool)`：值为`True`时，则对两组边界框中每个边界框之间进行计算。值为`False`时，只对对齐的边界框之间进行计算。
- `mode(int)`：计算IoU的模式。默认值为`0`。值为`0`时，表示计算`IoU`，值为`1`时，表示计算`IoF`。
- `v_threshold(float)`：顶点判断的容忍阈值。
- `e_threshold(float)`：边相交判断的容忍阈值。
### 返回值
- `Tensor`：IoU张量，数据类型为`float32, float16`，`is_cross`为`True`时形状为`[B, N, M]，反之则为`[B, N]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
import numpy as np
from mx_driving.common import npu_rotated_iou
a = np.random.uniform(0, 1, (2, 2, 5)).astype(np.float16)
b = np.random.uniform(0, 1, (2, 3, 5)).astype(np.float16)
box1 = torch.from_numpy(a).npu()
box2 = torch.from_numpy(b).npu()
iou = npu_rotated_iou(box1, box2, False, 0, True, 1e-5, 1e-5)
print(iou)
```
```text
tensor([[[3.3325e-01, 1.0162e-01],
         [1.0162e-01, 1.0000e+00]],

        [[0.0000e+00, 0.0000e+00],
         [0.0000e+00, 5.9605e-08]]], dtype=torch.float16)
```
## npu_dynamic_scatter
### 接口原型
```python
mx_driving.common.npu_dynamic_scatter(Tensor feats, Tensor coors, str reduce_type = 'max') -> Tuple[torch.Tensor, torch.Tensor]
```
### 功能描述
将点云特征点在对应体素中进行特征压缩。
### 参数说明
- `feats(Tensor)`：点云特征张量[N, C]，仅支持两维，数据类型为`float32`，特征向量`C`长度上限为2048。
- `coors(Tensor)`：体素坐标映射张量[N, 3]，仅支持两维，数据类型为`int32`，此处以x, y, z指代体素三维坐标，其取值范围为`0 <= x, y <= 2048`,  `0 <= z <= 256`。
- `reduce_type(str)`：压缩类型。可选值为`'max'`, `'mean'`, `'sum'`。默认值为`'max'`
### 返回值
- `voxel_feats(Tensor)`：压缩后的体素特征张量，仅支持两维，数据类型为`float32`。
- `voxel_coors(Tensor)`：去重后的体素坐标，仅支持两维，数据类型为`int32`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.common import npu_dynamic_scatter

feats = torch.tensor([[1, 2, 3], [3, 2, 1], [7, 8, 9], [9, 8, 7]], dtype=torch.float32).npu()
coors = torch.tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]], dtype=torch.int32).npu()
voxel_feats, voxel_coors = npu_dynamic_scatter(feats, coors, 'max')

print(voxel_feats)
print(voxel_coors)
```
## npu_points_in_box
### 接口原型
```python
mx_driving.common.npu_points_in_box(Tensor boxes, Tensor points) -> Tensor
```
### 功能描述
判断点是否在框内。
### 参数说明
- `boxes(Tensor)`：框张量，数据类型为`float32`。shape 为`[B, M, 7]`。`7`分别代表`x, y, z, x_size, y_size, z_size, rz`。
- `points(Tensor)`：点张量，数据类型为`float32`。shape 为`[B, N, 3]`。`3`分别代表`x, y, z`。
### 返回值
- `boxes_idx_of_points(Tensor)`：点在框内的索引张量，数据类型为`int32`。shape 为`[B, N]`。
### 约束说明
- `boxes`和`points`的`B`必须相同，且只能为`1`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.common import npu_points_in_box
boxes = torch.tensor([[[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9]]], dtype=torch.float32).npu()
points = torch.tensor([[[1, 2, 3], [3, 4, 5]]], dtype=torch.float32).npu()
out = npu_points_in_box(boxes, points)
print(out)
```
```text
tensor([[0, 1]], dtype=torch.int32)
```
## npu_multi_scale_deformable_attn_function
### 接口原型
```python
mx_driving.common.npu_multi_scale_deformable_attn_function(Tensor value, Tensor shape, Tensor offset, Tensor locations, Tensor weight) -> Tensor
```
### 功能描述
多尺度可变形注意力机制, 将多个视角的特征图进行融合。
### 参数说明
- `value(Tensor)`：特征张量，数据类型为`float32, float16`。shape为`[bs, num_keys, num_heads, embed_dims]`。其中`bs`为batch size，`num_keys`为特征图的大小，`num_heads`为头的数量，`embed_dims`为特征图的维度，需要为8的倍数。
- `shape(Tensor)`：特征图的形状，数据类型为`int32, int64`。shape为`[num_levels, 2]`。其中`num_levels`为特征图的数量，`2`分别代表`H, W`。
- `offset(Tensor)`：偏移量张量，数据类型为`int32, int64`。shape为`[num_levels]`。
- `locations(Tensor)`：位置张量，数据类型为`float32, float16`。shape为`[bs, num_queries, num_heads, num_levels, num_points, 2]`。其中`bs`为batch size，`num_queries`为查询的数量，`num_heads`为头的数量，`num_levels`为特征图的数量，`num_points`为采样点的数量，`2`分别代表`y, x`。
- `weight(Tensor)`：权重张量，数据类型为`float32, float16`。shape为`[bs, num_queries, num_heads, num_levels, num_points]`。其中`bs`为batch size，`num_queries`为查询的数量，`num_heads`为头的数量，`num_levels`为特征图的数量，`num_points`为采样点的数量。
### 返回值
- `output(Tensor)`：融合后的特征张量，数据类型为`float32, float16`。shape为`[bs, num_queries, num_heads*embed_dims]`。
### 支持的型号
- Atlas A2 训练系列产品
### 约束说明
- `locations`的值在`[0, 1]`之间。
### 调用示例
```python
import torch, torch_npu
from mx_driving.common import npu_multi_scale_deformable_attn_function
bs, num_levels, num_heads, num_points, num_queries, embed_dims = 1, 1, 4, 8, 16, 32

shapes = torch.as_tensor([(100, 100)], dtype=torch.long)
num_keys = sum((H * W).item() for H, W in shapes)

value = torch.rand(bs, num_keys, num_heads, embed_dims) * 0.01
sampling_locations = torch.ones(bs, num_queries, num_heads, num_levels, num_points, 2) * 0.005
attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points) + 1e-5
level_start_index = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))

out = npu_multi_scale_deformable_attn_function(value.npu(), shapes.npu(), level_start_index.npu(), sampling_locations.npu(), attention_weights.npu())
print(out)
```
```text
tensor([[[9.3002, 11.1603, 0.0000, 0.0000]]], dtype=torch.float32)
```
## voxelization
### 接口原型
```python
mx_driving.common.voxelization(Tensor points, List[float] voxel_size, List[float] coors_range, int max_points=-1, int max_voxels=-1, bool deterministic=True) -> Tensor
```
### 功能描述
将点云数据进行体素化。
### 参数说明
- `points(Tensor)`：点云数据，数据类型为`float32`。shape为`[N, F]`。其中`N`为点的数量，`F`分别代表每个点的特征维度，其中`N > 0, F >= 3`。
- `voxel_size(List[float])`：体素大小，数据类型为`float32`。shape为`[3]`。其中`3`分别代表`x, y, z`。
- `coors_range(List[float])`：体素范围，数据类型为`float32`。shape为`[6]`。其中`6`分别代表`x_min, y_min, z_min, x_max, y_max, z_max`。
- `max_points(int)`：每个体素的最大点数。默认值为`-1`。
- `max_voxels(int)`：最大体素数。默认值为`-1`。
- `deterministic(bool)`：是否确定性。默认值为`True`。
### 返回值
- `coors(Tensor)`：每个点所属的体素坐标，数据类型为`int32`。shape为`[N, 3]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.common import Voxelization
points = torch.randint(-20, 100, [16, 3], dtype=torch.float32).npu()
coors_range = [0, -40, -3, 70.4, 40, 1]
max_points = -1
voxel_size = [0.5, 0.5, 0.5]
dynamic_voxelization = Voxelization(voxel_size, coors_range, max_points)
out = dynamic_voxelization.forward(points)
print(out)
```
## npu_nms3d_normal
### 接口原型
```python
mx_driving.common.npu_nms3d_normal(Tensor boxes, Tensor scores, float: iou_threshold) -> Tensor
```
### 功能描述
3D非极大值抑制。
### 参数说明
- `boxes(Tensor)`：框张量，数据类型为`float32, float16`。shape 为`[N, 7]`。`7`分别代表`x, y, z, x_size, y_size, z_size, rz`。
- `scores(Tensor)`：评分张量，数据类型为`float32, float16`。shape 为`[N]`。
- `iou_threshold(float)`：IoU阈值。
### 返回值
- `Tensor`：NMS后的框张量，数据类型为`int32`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.common import npu_nms3d_normal
boxes = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9]], dtype=torch.float32).npu()
scores = torch.tensor([1, 2], dtype=torch.float32).npu()
out = npu_nms3d_normal(boxes, scores, 0.5)
print(out)
```
```text
tensor([[1, 0]], dtype=torch.int32)
```
## npu_nms3d
### 接口原型
```python
mx_driving.common.npu_nms3d(Tensor boxes, Tensor scores, float: iou_threshold) -> Tensor
```
### 功能描述
3D非极大值抑制，在bev视角下剔除多个3d box交并比大于阈值的box。
### 参数说明
- `boxes(Tensor)`：框张量，数据类型为`float32, float16`。shape 为`[N, 7]`。`7`分别代表`x, y, z, x_size, y_size, z_size, rz`。
- `scores(Tensor)`：评分张量，数据类型为`float32, float16`。shape 为`[N]`。
- `iou_threshold(float)`：IoU阈值。
### 返回值
- `Tensor`：NMS后的框张量，数据类型为`int32`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.common import npu_nms3d
boxes = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9]], dtype=torch.float32).npu()
scores = torch.tensor([1, 2], dtype=torch.float32).npu()
out = npu_nms3d(boxes, scores, 0.5)
print(out)
```
```text
tensor([[1]], dtype=torch.int32)
```
## npu_furthest_point_sampling
### 接口原型
```python
mx_driving.common.npu_furthest_point_sampling(Tensor points, int num_points) -> Tensor
```
### 功能描述
点云数据的最远点采样。
### 参数说明
- `points(Tensor)`：点云数据，数据类型为`float32`。shape为`[B, N, 3]`。其中`B`为batch size，`N`为点的数量，`3`分别代表`x, y, z`。
- `num_points(int)`：采样点的数量。
### 返回值
- `Tensor`：采样后的点云数据，数据类型为`float32`。shape为`[B, num_points]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.common import npu_furthest_point_sampling
points = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32).npu()
out = npu_furthest_point_sampling(points, 2)
print(out)
```
```text
tensor([[0, 2]], dtype=torch.int32)
```
### 算子約束
1. points输入shape[B, N, 3]的总大小(B x N x 3)不应该超过383166
## furthest_point_sample_with_dist
### 接口原型
```python
mx_driving.common.furthest_point_sample_with_dist(Tensor points, int num_points) -> (Tensor, Tensor)
```
### 功能描述
与`npu_furthest_point_sampling`功能相同，但输入略有不同。
### 参数说明
- `points(Tensor)`：点云数据，表示各点间的距离，数据类型为`float32`。shape为`[B, N, N]`。其中`B`为batch size，`N`为点的数量。
- `num_points(int)`：采样点的数量。
### 返回值
- `Tensor`：采样后的点云数据，数据类型为`float32`。shape为`[B, num_points]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.common import furthest_point_sample_with_dist
points = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32).npu()
out = furthest_point_sample_with_dist(points, 2)
print(out)
```
```text
tensor([[0, 2]], dtype=torch.int32)
```

## three_interpolate
### 接口原型
```python
mx_driving.common.three_interpolate(features: torch.Tensor, indices: torch.Tensor,
                weight: torch.Tensor) -> torch.Tensor:
```
### 功能描述
对三维数据进行加权最近邻线性插值处理
### 参数说明
- `features`：需要被插值的特征，数据类型为`float32|float16`，维度为（B, C, M）。
- `indices`：获取目标特征计算的索引，数据类型为`int32`，维度为（B, N, 3），
  - `indices`的元素值需小于`features`的第三维度，即值在[0, M)。
- `weight`：获取目标特征计算的权重，数据类型为`float32|float16`，维度为（B, N, 3）。
  - `weight`数据类型与`features`须一致。
- `features`，`indices`，`weights`三个参数的每个维度须小于10000。
- `features`，`indices`，`weights`三个参数的大小请勿超过2^24。
### 返回值
- `output`：目标特征张量，数据类型为`float32|float16`，维度为（B, C, N）。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch
from mx_driving.common import three_interpolate
features = torch.tensor(
            [[[2.4350, 4.7516, 4.4995, 2.4350, 2.4350, 2.4350],
            [3.1236, 2.6278, 3.0447, 3.1236, 3.1236, 3.1236],
            [2.6732, 2.8677, 2.6436, 2.6732, 2.6732, 2.6732],
            [0.0124, 7.0150, 7.0199, 0.0124, 0.0124, 0.0124],
            [0.3207, 0.0000, 0.3411, 0.3207, 0.3207, 0.3207]],
            [[0.0000, 0.9544, 2.4532, 0.0000, 0.0000, 0.0000],
            [0.5346, 1.9176, 1.4715, 0.5346, 0.5346, 0.5346],
            [0.0000, 0.2744, 2.0842, 0.0000, 0.0000, 0.0000],
            [0.3414, 1.5063, 1.6209, 0.3414, 0.3414, 0.3414],
            [0.5814, 0.0103, 0.0000, 0.5814, 0.5814, 0.5814]]],
            ).npu()
idx = torch.tensor(
            [[[0, 1, 2], [2, 3, 4], [2, 3, 4], [0, 1, 2], [0, 1, 2], [0, 1, 3]],
            [[0, 2, 3], [1, 3, 4], [2, 1, 4], [0, 2, 4], [0, 2, 4], [0, 1, 2]]],
            ).int().npu()
weight = torch.tensor(
            [[[3.3333e-01, 3.3333e-01, 3.3333e-01],
              [1.0000e+00, 5.8155e-08, 2.2373e-08],
              [1.0000e+00, 1.7737e-08, 1.7356e-08],
              [3.3333e-01, 3.3333e-01, 3.3333e-01],
              [3.3333e-01, 3.3333e-01, 3.3333e-01],
              [3.3333e-01, 3.3333e-01, 3.3333e-01]],
             [[3.3333e-01, 3.3333e-01, 3.3333e-01],
              [1.0000e+00, 1.3651e-08, 7.7312e-09],
              [1.0000e+00, 1.7148e-08, 1.4070e-08],
              [3.3333e-01, 3.3333e-01, 3.3333e-01],
              [3.3333e-01, 3.3333e-01, 3.3333e-01],
              [3.3333e-01, 3.3333e-01, 3.3333e-01]]],
            ).npu()
output = three_interpolate(features, idx, weight)
print(output)
```
```text
torch.tensor(
        [[[3.8953e+00, 4.4995e+00, 4.4995e+00, 3.8953e+00, 3.8953e+00, 3.2072e+00], 
        [2.9320e+00, 3.0447e+00, 3.0447e+00, 2.9320e+00, 2.9320e+00, 2.9583e+00], 
        [2.7281e+00, 2.6436e+00, 2.6436e+00, 2.7281e+00, 2.7281e+00, 2.7380e+00], 
        [4.6824e+00, 7.0199e+00, 7.0199e+00, 4.6824e+00, 4.6824e+00, 2.3466e+00], 
        [2.2060e-01, 3.4110e-01, 3.4110e-01, 2.2060e-01, 2.2060e-01, 2.1380e-01]],
        [[8.1773e-01, 9.5440e-01, 2.4532e+00,8.1773e-01, 8.1773e-01, 1.1359e+00],
        [8.4689e-01, 1.9176e+00, 1.4715e+00, 8.4689e-01, 8.4689e-01, 1.3079e+00],
        [6.9473e-01, 2.7440e-01, 2.0842e+00, 6.9473e-01, 6.9473e-01, 7.8619e-01],
        [7.6789e-01, 1.5063e+00, 1.6209e+00, 7.6789e-01, 7.6789e-01, 1.1562e+00],
        [3.8760e-01, 1.0300e-02, 8.3569e-09, 3.8760e-01, 3.8760e-01, 1.9723e-01]]],
        device='npu:0'
        )
```

## voxel_pooling_train
### 接口原型
```python
mx_driving.common.npu_voxel_pooling_train(Tensor geom_xyz, Tensor input_features, Tensor voxel_num) -> Tensor
```
### 功能描述
点云数据体素化。
### 参数说明
- `geom_xyz`：体素坐标，数据类型为`int32`，维度为（B, N, 3）, 3表示x, y, z。
- `input_features`：点云数据，数据类型为`float32|float16`，维度为（B, N, C）。
- `voxel_num`：体素格子长宽高，数据类型为`int32`，维度为（3），3表示体素格子的长宽高。
### 返回值
- `output(Tensor)`：输出结果，数据类型为`float32|float16`。shape为`[B, num_voxel_y, num_voxel_x, C]`。
### 约束说明
- B <= 128
- N <= 1000000
- C <= 256
- num_voxel_x <= 1000
- num_voxel_y <= 1000
- num_voxel_z <= 10
- 反向具有相同约束。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch
import torch_npu
import mx_driving.common

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

result_npu = mx_driving.common.npu_voxel_pooling_train(geom_npu, feature_npu, voxel_num)
```

## scatter_mean
### 接口原型
```python
mx_driving.common.scatter_mean(Tensor src, Tensor indices, int dim=0， Tensor out=None, int dim_size=None) -> Tensor
```
### 功能描述
将输入张量`src`中的元素按照`indices`中的索引在指定的`dim`维进行分组，并计算每组的平均值，返回平均值。
### 参数说明
- `src`：源张量，数据类型为`float32`。
- `indices`：索引张量，数据类型为`int32`，且
  - `indices`的维度必须小于等于`src`的维度，
  - `indices`每一维的长度均必须与`src`长度相同。
  - `indices`的取值必须为非负的有效索引值。
- `out`：被更新张量，数据类型为`float32`，默认为`None`，输入out不为`None`时，`out`中的元素参与平均值的计算，且
  - `out`的维度必须与`src`的维度相同。
  - `out`除第`dim`维外其余维的长度必须与`src`相同。
- `dim`：指定的维度，表示按照哪个维度进行分组平均计算。
- `dim_size`：输出张量在`dim`维的长度，数据类型为`int32`，该参数仅在输入out为`None`时生效。
### 返回值
- `out`：求平均后的张量，数据类型为`float32`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例

```python
import torch, torch_npu
from mx_driving.common import scatter_mean
src = torch.randn(4, 5, 6).to(torch.float)
indices = torch.randint(5, (4, 5)).to(torch.int32)
dim = 0
out = scatter_mean(src.npu(), indices.npu(), None, dim)
print(out)
```
# Perception 算子
## bev_pool
### 接口原型
```python
mx_driving.perception.fused.bev_pool(Tensor feat, Tensor geom_feat, int B, int D, int H, int W) -> Tensor
```
### 功能描述
BEV池化。可参考论文`BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation`
### 参数说明
- `feat(Tensor)`：特征张量，数据类型为`float32`。shape为`[N, C]`。其中`N`为原特征张量拉伸后的数量，`C`为特征的维度。
- `geom_feat(Tensor)`：输出坐标张量，数据类型为`int32`。shape为`[N, 4]`。其中`4`分别代表`h, w, b, d`。
- `B(int)`：batch size。
- `D(int)`：输出池化深度。
- `H(int)`：输出池化高度。
- `W(int)`：输出池化宽度。
### 返回值
- `bev_pooled_feat(Tensor)`：采样后的点云数据，数据类型为`float32`。shape为`[B, D, H, W, C]`。
### 支持的型号
- Atlas A2 训练系列产品
### 约束说明
- `geom_feat`的4个对应的值必须在`[0, H-1]`, `[0, W-1]`, `[0, B-1]`, `[0, D-1]`之间。
- `geom_feat`和`feat`的第0维长度必须相同。
- C <= 1024
- B * D * H * W * C <= 2^31, B, D <= 8, H, W <= 256
- 对于反向也是同样的约束。
### 调用示例
```python
import torch, torch_npu
from mx_driving.perception.fused import bev_pool
feat = torch.rand(4, 256).npu()
feat.requires_grad_()
geom_feat = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3]], dtype=torch.int32).npu()
bev_pooled_feat = bev_pool(feat, geom_feat, 4, 1, 256, 256)
loss = bev_pooled_feat.sum()
loss.backward()
print(feat.grad)
```

## bev_pool_v2
### 接口原型
```python
mx_driving.perception.fused.bev_pool_v2(Tensor depth, feat, Tensor ranks_depth, Tensor ranks_feat, Tensor ranks_bev,
                                 List[int] bev_feat_shape, Tensor interval_starts, Tensor interval_lengths) -> Tensor
```
### 功能描述
BEV池化优化版。可参考论文`BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View`。
### 参数说明
- `depth(Tensor)`：深度张量，数据类型为`float32`。shape为`[B, N, D, H, W]`。其中`B`为batch size，`N`为特征的数量，`D, H, W`分别代表深度、高度、宽度。
- `feat(Tensor)`：特征张量，数据类型为`float32`。shape为`[B, N, H, W, C]`。其中`B`为batch size，`N`为特征的数量，`H, W, C`分别代表高度、宽度、通道数。
- `ranks_depth(Tensor)`：深度排序张量，数据类型为`int32`。shape为`[N_RANKS]`。
- `ranks_feat(Tensor)`：特征排序张量，数据类型为`int32`。shape为`[N_RANKS]`。
- `ranks_bev(Tensor)`：BEV排序张量，数据类型为`int32`。shape为`[N_RANKS]`。
- `bev_feat_shape(List[int])`：BEV特征形状，数据类型为`int32`。长度为`5`， 分别代表`B, D, H, W, C`。
- `interval_starts(Tensor)`：间隔开始张量，数据类型为`int32`。shape为`[N_INTERVALS]`。
- `interval_lengths(Tensor)`：间隔长度张量，数据类型为`int32`。shape为`[N_INTERVALS]`。
### 返回值
- `bev_pooled_feat(Tensor)`：BEV池化后的特征张量，数据类型为`float32`。shape为`[B, D, H, W, C]`。
### 约束说明
- `ranks_depth`的值必须在`[0, B*B*D*H*W]`之间。
- `ranks_feat`的值必须在`[0, B*N*H*W]`之间。
- `ranks_bev`的值必须在`[0, B*D*H*W]`之间。
- C <= 1024
- B * D * H * W * C <= 2^31, B, D <= 8, H, W <= 256
- N_RANKS <= 2^21
- 对于反向也是同样的约束。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.perception.fused import bev_pool_v2
depth = torch.rand(2, 1, 8, 256, 256).npu()
feat = torch.rand(2, 1, 256, 256, 64).npu()
feat.requires_grad_()
ranks_depth = torch.tensor([0, 1], dtype=torch.int32).npu()
ranks_feat = torch.tensor([0, 1], dtype=torch.int32).npu()
ranks_bev = torch.tensor([0, 1], dtype=torch.int32).npu()
bev_feat_shape = [2, 8, 256, 256, 64]
interval_starts = torch.tensor([0], dtype=torch.int32).npu()
interval_lengths = torch.tensor([2], dtype=torch.int32).npu()
bev_pooled_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape, interval_starts, interval_lengths)
loss = bev_pooled_feat.sum()
loss.backward()
print(loss)
print(feat.grad)
```

## group_points
### 接口原型
```python
mx_driving.perception.point.npu_group_points(Tensor features, Tensor indices) -> Tensor
```
### 功能描述
点云数据按照索引重新分组。
### 参数说明
- `features`：需要被插值的特征，数据类型为`float32|float16`，维度为（B, C, N）。
- `indices`：获取目标特征计算的索引，数据类型为`int32`，维度为（B, npoints, nsample）。
### 返回值
- `output(Tensor)`：分组后的点云数据，数据类型为`float32|float16`。shape为`[B, C, npoints, nsample]`。
### 约束说明
- `indices`的元素值需小于`features`的第三维度，即值在[0, N)。
- C <= 1024
- 反向具有相同约束。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch
import torch_npu
import mx_driving.perception.point
indices = torch.tensor([[[0, 2, 5, 5], [1, 0, 5, 0], [2, 1, 4, 4]]]).int().npu()
features = torch.tensor([[[0.9178, -0.7250, -1.6587, 0.0715, -0.2252, 0.4994],
                          [0.6190, 0.1755, -1.7902, -0.5852, -0.3311, 1.9764],
                          [1.7567, 0.0740, -1.1414, 0.4705, -0.3197, 1.1944],
                          [-0.2343, 0.1194, 0.4306, 1.3780, -1.4282, -0.6377],
                          [0.7239, 0.2321, -0.6578, -1.1395, -2.3874, 1.1281]]],
                          dtype=torch.float32).npu()
features.requires_grad = True
output = mx_driving.perception.point.npu_group_points(features, indices)
output.backward(output)
grad_features = features.grad
```
```text
expected_output = tensor(
        [[[[ 0.9178, -1.6587,  0.4994,  0.4994],
          [-0.7250,  0.9178,  0.4994,  0.9178],
          [-1.6587, -0.7250, -0.2252, -0.2252]],

         [[ 0.6190, -1.7902,  1.9764,  1.9764],
          [ 0.1755,  0.6190,  1.9764,  0.6190],
          [-1.7902,  0.1755, -0.3311, -0.3311]],

         [[ 1.7567, -1.1414,  1.1944,  1.1944],
          [ 0.0740,  1.7567,  1.1944,  1.7567],
          [-1.1414,  0.0740, -0.3197, -0.3197]],

         [[-0.2343,  0.4306, -0.6377, -0.6377],
          [ 0.1194, -0.2343, -0.6377, -0.2343],
          [ 0.4306,  0.1194, -1.4282, -1.4282]],

         [[ 0.7239, -0.6578,  1.1281,  1.1281],
          [ 0.2321,  0.7239,  1.1281,  0.7239],
          [-0.6578,  0.2321, -2.3874, -2.3874]]]], dtype=torch.float32)
expected_grad_features = tensor(
        [[[ 2.7534, -1.4500, -3.3174,  0.0000, -0.4504,  1.4982],
         [ 1.8570,  0.3510, -3.5804,  0.0000, -0.6622,  5.9292],
         [ 5.2701,  0.1480, -2.2828,  0.0000, -0.6394,  3.5832],
         [-0.7029,  0.2388,  0.8612,  0.0000, -2.8564, -1.9131],
         [ 2.1717,  0.4642, -1.3156,  0.0000, -4.7748,  3.3843]]], dtype=torch.float32)
```

## knn
### 接口原型
```python
mx_driving.common.knn(int k, Tensor xyz, Tensor center_xyz, bool Transposed) -> Tensor
```
### 功能描述
對center_xyz中的每個點找到xyz中對應batch中的距離最近的k個點，并且返回此k個點的索引值。
### 参数说明
- `xyz(Tensor)`：点数据，表示(x, y, z)三維坐標，数据类型为`float32`。shape为`[B, N, 3]`(当Transposed=False)或`[B, 3, N]`(当Transposed=True)。其中`B`为batch size，`N`为点的数量。
- `center_xyz(Tensor)`：点数据，表示(x, y, z)三維坐標，数据类型为`float32`。shape为`[B, npoint, 3]`(当Transposed=False)或`[B, 3, npoint]`(当Transposed=True)。其中`B`为batch size，`npoint`为点的数量。
- `k(int)`：采样点的数量。
- `Transposed(bool)`: 輸入是否需要進行轉置
### 返回值
- `idx(Tensor)`：采样后的索引数据，数据类型为`int32`。shape为`[B, k, npoint]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.common import knn
xyz = torch.tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]], dtype=torch.float32).npu()
center_xyz = torch.tensor([[1, 2, 3]], [[1, 2, 3]], dtype=torch.float32).npu()
idx = knn(2, xyz, center_xyz, False)
print(idx)
```
```text
tensor([[0, 0], [1, 1]], dtype=torch.int32)
```
### 算子約束
1. k必須>0且<100
2. xyz中的每個batch中的任意一個點到center_xyz對應batch中的任意一個點的距離必須在1e10f以内
3. xyz和center_xyz的shape必须是3维，当Transposed=True时，xyz和center_xyz的shape的dim的第1位必须是3；当Transposed=False时，xyz和center_xyz的shape的dim的第2位必须是3
4. 距离相同时索引存在不稳定排序问题,遇到距离精度通过但索引精度错误时，复用不稳定排序的CCB结论

## three_nn
### 接口原型
```python
mx_driving.common.three_nn(Tensor target, Tensor source) -> (Tensor dist, Tensor idx)
```
### 功能描述
對target中的每個點找到source中對應batch中的距離最近的3個點，并且返回此3個點的距離和索引值。
### 参数说明
- `target(Tensor)`：点数据，表示(x, y, z)三維坐標，数据类型为`float32/float16`。shape为`[B, npoint, 3]`。其中`B`为batch size，`npoint`为点的数量。
- `source(Tensor)`：点数据，表示(x, y, z)三維坐標，数据类型为`float32/float16`。shape为`[B, N, 3]`。其中`B`为batch size，`N`为点的数量。
### 返回值
- `dist(Tensor)`：采样后的索引数据，数据类型为`float32/float16`。shape为`[B, npoint, 3]`。
- `idx(Tensor)`：采样后的索引数据，数据类型为`int32/int32`。shape为`[B, npoint, 3]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.common import three_nn
source = torch.tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]], dtype=torch.float32).npu()
target = torch.tensor([[1, 2, 3]], [[1, 2, 3]], dtype=torch.float32).npu()
dist, idx = three_nn(target, source)
print(dist)
print(idx)
```
```text
tensor([[2.236, 2.236, 2.236], [2.236, 2.236, 2.236]], dtype=torch.float32)
tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int32)
```
### 算子約束
1. source和target的shape必须是3维，且source和target的shape的dim的第2位必须是3
2. 距离相同时索引存在不稳定排序问题，遇到距离精度通过但索引精度错误时，复用不稳定排序的CCB结论
## RoipointPool3d
### 接口原型
```python
mx_driving.common.RoipointPool3d(int num_sampled_points, Tensor points, Tensor point_features, Tensor boxes3d) -> (Tensor pooled_features, Tensor pooled_empty_flag)
```
### 功能描述
对每个3D方案的几何特定特征进行编码。
### 参数说明
- `num_sampled_points(int)`：特征点的数量。
- `points(Tensor)`：点张量，数据类型为`float32, float16`。shape 为`[B, N, 3]`。`3`分别代表`x, y, z`。
- `point_features(Tensor)`：点特征张量，数据类型为`float32, float16`。shape 为`[B, N, C]`。`C`分别代表`x, y, z`。
- `boxes3d(Tensor)`：框张量，数据类型为`float32, float16`。shape 为`[B, M, 7]`。`7`分别代表`x, y, z, x_size, y_size, z_size, rz`。
### 返回值
- `pooled_features(Tensor)`：点在框内的特征张量，数据类型为`float32, float16`。shape 为`[B, M, num, 3+C]`。
- `pooled_empty_flag(Tensor)`：所有点不在框内的空标记张量，数据类型为`int32`。shape 为`[B, M]`。
### 约束说明
- `points`、`point_features`和`boxes3d`的数据类型必须相同，以及`B`也必须相同。
- `num_sampled_points`必须小于等于`N`。
- 数据类型为`float32`时，建议`N`小于等于2640、`M`小于等于48、`num_sampled_points`小于等于48。
- 数据类型为`float16`时，建议`N`小于等于3360、`M`小于等于60、`num_sampled_points`小于等于60。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.common import RoIPointPool3d
num_sampled_points = 1
points = torch.tensor([[[1, 2, 3]]], dtype=torch.float).npu()
point_features = points.clone()
boxes3d = torch.tensor([[[1, 2, 3, 4, 5, 6, 1]]], dtype=torch.float).npu()
roipoint_pool3d = RoIPointPool3d(num_sampled_points)
pooled_features, pooled_empty_flag = roipoint_pool3d(points, point_features, boxes3d)
print(pooled_features)
print(pooled_empty_flag)
```
```text
tensor([[[[1., 2., 3., 1., 2., 3.]]]])
tensor([[0]], dtype=torch.int32)
```
## boxes_overlap_bev
### 接口原型
```python
mx_driving.perception.vision.boxes_overlap_bev(Tensor boxes_a, Tensor boxes_b) -> Tensor
```
### 功能描述
Calculates the intersection of bounding boxes in Bird's Eye View.
### 参数说明
- `boxes_a (Tensor)`：第一组bounding boxes，数据类型为`float32`。shape为`[M, 5]`。其中`5`分别代表`x1, y1, x2, y2, angle`, `x1, y1, x2, y2`代表box四个顶点的横纵坐标，`angle`代表box的弧度制旋转角。
- `boxes_b (Tensor)`：第二组bounding boxes，数据类型为`float32`。shape为`[N, 5]`。其中`5`分别代表`x1, y1, x2, y2, angle`, `x1, y1, x2, y2`代表box四个顶点的横纵坐标，`angle`代表box的弧度制旋转角。
### 返回值
- `area_overlap(Tensor)`：包含两组bounding boxes交叠面积的张量，数据类型为`float32`。shape为`[M, N]`。
### 约束说明

### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.perception.vision import boxes_overlap_bev
boxes_a = torch.tensor([[0, 0, 2, 2, 0]], dtype=torch.float32).npu()
boxes_b = torch.tensor([[1, 1, 3, 3, 0]], dtype=torch.float32).npu()
area_overlap = boxes_overlap_bev(boxes_a, boxes_b)
print(area_overlap)
```

## npu_max_pool2d
### 接口原型
```python
mx_driving.common.npu_max_pool2d(Tensor x, int kernel_size, int stride, int padding) -> Tensor
```
### 功能描述
Performs max pooling on the input and outputs max values.
### 参数说明
- `x (Tensor)`：一组待池化对象，数据类型为`float32`，format为NCHW。
### 返回值
- `y (Tensor)`：池化后的最大值，数据类型为`float32`，format为NCHW。
### 约束说明
kernel_size仅支持3，stride仅支持2，padding仅支持1，且输入C轴数据量要求为8的倍数，H和W需要大于100。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.common import npu_max_pool2d
kernel_size = 3
stride = 2
padding = 1
x = torch.randn(18, 64, 464, 800).npu()
res = npu_max_pool2d(x, kernel_size, stride, padding)
print(res)
```

## unique_voxel
### 接口原型
```python
ads_c.unique_voxel(Tensor input) -> Tensor
```
### 功能描述
对输入的点云数据进行去重处理。
### 参数说明
- `x (Tensor)`：数据语义为索引，数据类型为`int32`，shape为`[N]`。
### 返回值
- `num_voxels(int)`, 体素数量。
- `uni_voxels(Tensor)`，去重后的体素数据，数据类型为`float32`，shape为`[num_voxels]`。
- `uni_indices(Tensor)`, 去重后的索引数据，数据类型为`int32`，shape为`[num_voxels]`。
- `argsort_indices(Tensor)`, 排序后的索引数据，数据类型为`int32`，shape为`[N]`。
- `uni_argsort_indices(Tensor)`, 去重后的排序后的索引数据，数据类型为`int32`，shape为`[num_voxels]`。
### 约束说明
N的大小受限于内存大小，建议N小于等于2^32.
受限于芯片指令，输入的数据类型只能是int32，且>=0,<2^30
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu, numpy
from ads_c import unique_voxel
x = np.random.randint(0, 1024, (100000,)).astype(np.int32)
voxels_npu = torch.from_numpy(voxels).npu()
cnt, uni_vox, _, _, _ = ads_c.unique_voxel(voxels_npu)
print(cnt)
print(uni_vox)
```