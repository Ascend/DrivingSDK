> Note: 以prototype标注的接口，表示该接口为预发布接口，可能会有变动，不建议在生产环境中使用。
# Common 算子
## scatter_max
### 接口原型
```python
ads.common.scatter_max(Tensor updates, Tensor indices, Tensor out=None) -> (Tensor out, Tensor argmax)
```
### 功能描述
在`第0`维上，将输入张量`updates`中的元素按照`indices`中的索引进行分散，然后在第0维上取最大值，返回最大值和对应的索引。对于1维张量，公式如下：
$$out_i = max(out_i, max_j(updates_j))$$
$$argmax_i = argmax_j(updates_j)$$
这里，$i = indices_j$。
### 参数说明
- `updates`：更新源张量，数据类型为`float32`。
- `indices`：索引张量，数据类型为`int32`，且
  - `indices`的维度必须为`1`，
  - `indices`第0维的长度必须与`updates`第0维的长度相同。
  - `indices`的最大值必须小于`491520`。
- `out`：被更新张量，数据类型为`float32`，默认为`None`。
### 返回值
- `out`：更新后的张量，数据类型为`float32`。
- `argmax`：最大值对应的索引张量，数据类型为`int32`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import scatter_max
updates = torch.tensor([[2, 0, 1, 4, 4], [0, 2, 1, 3, 4]], dtype=torch.float32).npu()
indices = torch.tensor([4, 1, 2, 3], dtype=torch.int32).npu()
out = updates.new_zeros((2, 6))
out, argmax = scatter_max(updates, indices, out)
print(out)
print(argmax)
```
```text
tensor([[0., 0., 0., 0., 0., 0.],
        [0., 0., 3., 4., 0., 0.]])
tensor([[2, 2,  2,  2,  2,  2],
        [ 1,  1,  1, 1, 0, 0]])
```
## \[prototype\] npu_rotated_overlaps
### 接口原型
```python
ads.common.npu_rotated_overlaps(Tensor self, Tensor query_boxes, bool trans=False) -> Tensor
```
### 功能描述
计算旋转框的重叠面积。
### 参数说明
- `self(Tensor)`：梯度增量，数据类型为`float32, float16`，形状为`[B, 5, N]`。
- `query_boxes(Tensor)`：查询框张量，数据类型为`float32, float16`，形状为`[B, 5, M]`。
- `trans(bool)`：是否进行坐标变换。默认值为`False`。值为`True`时，表示`xyxyt`, 值为`False`时，表示`xywht`。
### 返回值
- `Tensor`：重叠面积张量，数据类型为`float32, float16`，形状为`[B, N, M]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
import numpy as np
from ads.common import npu_rotated_overlaps
a = np.random.uniform(0, 1, (1, 3, 5)).astype(np.float16)
b = np.random.uniform(0, 1, (1, 2, 5)).astype(np.float16)
box1 = torch.from_numpy(a).npu()
box2 = torch.from_numpy(b).npu()
output = npu_rotated_overlaps(box1, box2)
print(output)
```
```text
tensor([[[0.0000, 0.1562, 0.0000],
         [0.1562, 0.3713, 0.0611],
         [0.0000, 0.0611, 0.0000]]], dtype=torch.float16)
```
## \[prototype\] npu_rotated_iou
### 接口原型
```python
ads.common.npu_rotated_iou(Tensor self, Tensor query_boxes, bool trans=False, int mode=0, bool is_cross=True, float v_threshold=0.0, float e_threshold=0.0) -> Tensor
```
### 功能描述
计算旋转框的IoU。
### 参数说明
- `self(Tensor)`：梯度增量，数据类型为`float32, float16`，形状为`[B, 5, N]`。
- `query_boxes(Tensor)`：查询框张量，数据类型为`float32, float16`，形状为`[B, 5, M]`。
- `trans(bool)`：是否进行坐标变换。默认值为`False`。值为`True`时，表示`xyxyt`, 值为`False`时，表示`xywht`。
- `is_cross(bool)`：是否计算交叉面积。默认值为`True`。值为`True`时，表示计算交叉面积，值为`False`时，表示计算并集面积。
- `mode(int)`：计算IoU的模式。默认值为`0`。值为`0`时，表示计算`IoU`，值为`1`时，表示计算`IoF`。
- `v_threshold(float)`：垂直方向的阈值。默认值为`0.0`。
- `e_threshold(float)`：水平方向的阈值。默认值为`0.0`。
### 返回值
- `Tensor`：IoU张量，数据类型为`float32, float16`，形状为`[B, N, M]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
import numpy as np
from ads.common import npu_rotated_iou
a = np.random.uniform(0, 1, (2, 2, 5)).astype(np.float16)
b = np.random.uniform(0, 1, (2, 3, 5)).astype(np.float16)
box1 = torch.from_numpy(a).npu()
box2 = torch.from_numpy(b).npu()
iou = npu_rotated_iou(box1, box2, trans=False, mode=0, is_cross=True, v_threshold=0.0, e_threshold=0.0)
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
ads.common.npu_dynamic_scatter(Tensor feats, Tensor coors, int64_t reduce_type) -> Tensor
```
### 功能描述
将特征点在对应体素中进行特征压缩。
### 参数说明
- `feats(Tensor)`：特征张量[M, C]，仅支持两维，数据类型为`float32`，特征向量`C`长度上限为2048。
- `coors(Tensor)`：体素坐标映射张量[M, 3]，仅支持两维，数据类型为`int32`，且坐标仅支持三维，坐标取值为0~1024。
- `reduce_type(int64_t)`：压缩类型。可选值为0, 1, 2。当值为0时，表示"sum"；当值为1时，表示"mean"；当值为2时，表示"max"。
### 返回值
- `voxel_feats(Tensor)`：压缩后的特征张量，仅支持两维，数据类型为`float32`。
- `voxel_coors(Tensor)`：压缩后的体素坐标，仅支持两维，数据类型为`int32`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_dynamic_scatter
feats = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.float32).npu()
coors = torch.tensor([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]], dtype=torch.int32).npu()
voxel_feats, voxel_coors = npu_dynamic_scatter(feats, coors, 1)
print(voxel_feats)
print(voxel_coors)
```
## npu_points_in_box
### 接口原型
```python
ads.common.npu_points_in_box(Tensor boxes, Tensor points) -> Tensor
```
### 功能描述
判断点是否在框内。
### 参数说明
- `boxes(Tensor)`：框张量，数据类型为`float32, float16`。shape 为`[B, M, 7]`。`7`分别代表`x, y, z, x_size, y_size, z_size, rz`。
- `points(Tensor)`：点张量，数据类型为`float32, float16`。shape 为`[B, N, 3]`。`3`分别代表`x, y, z`。
### 返回值
- `boxes_idx_of_points(Tensor)`：点在框内的索引张量，数据类型为`int32`。shape 为`[B, N]`。
### 约束说明
- `boxes`和`points`的`B`必须相同，且只能为`1`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_points_in_box
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
ads.common.npu_multi_scale_deformable_attn_function(Tensor value, Tensor shape, Tensor offset, Tensor locations, Tensor weight) -> Tensor
```
### 功能描述
多尺度可变形注意力机制, 将多个视角的特征图进行融合。
### 参数说明
- `value(Tensor)`：特征张量，数据类型为`float32, float16`。shape为`[bs, num_keys, num_heads, embed_dims]`。其中`bs`为batch size，`num_keys`为特征图的数量，`num_heads`为头的数量，`embed_dims`为特征图的维度，需要为8的倍数。
- `shape(Tensor)`：特征图的形状，数据类型为`int32`。shape为`[num_levels, 2]`。其中`num_levels`为特征图的数量，`2`分别代表`H, W`。
- `offset(Tensor)`：偏移量张量，数据类型为`int32`。shape为`[num_levels]`。
- `locations(Tensor)`：位置张量，数据类型为`int32`。shape为`[bs, num_queries, num_heads, num_levels, num_points, 2]`。其中`bs`为batch size，`num_queries`为查询的数量，`num_heads`为头的数量，`num_levels`为特征图的数量，`num_points`为采样点的数量，`2`分别代表`y, x`。
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
from ads.common import npu_multi_scale_deformable_attn_function
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
ads.common.voxelization(Tensor points, List[float] voxel_size, List[float] coors_range, int max_points=-1, int max_voxels=-1, bool deterministic=True) -> Tensor
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
from ads.common import Voxelization
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
ads.common.npu_nms3d_normal(Tensor boxes, Tensor scores, float: iou_threshold) -> Tensor
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
from ads.common import npu_nms3d_normal
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
ads.common.npu_nms3d(Tensor boxes, Tensor scores, float: iou_threshold) -> Tensor
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
from ads.common import npu_nms3d
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
ads.common.npu_furthest_point_sampling(Tensor points, int num_points) -> Tensor
```
### 功能描述
点云数据的最远点采样。
### 参数说明
- `points(Tensor)`：点云数据，数据类型为`float32, float16`。shape为`[B, N, 3]`。其中`B`为batch size，`N`为点的数量，`3`分别代表`x, y, z`。
- `num_points(int)`：采样点的数量。
### 返回值
- `Tensor`：采样后的点云数据，数据类型为`float32, float16`。shape为`[B, num_points]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_furthest_point_sampling
points = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32).npu()
out = npu_furthest_point_sampling(points, 2)
print(out)
```
```text
tensor([[0, 2]], dtype=torch.int32)
```
## furthest_point_sample_with_dist
### 接口原型
```python
ads.common.furthest_point_sample_with_dist(Tensor points, int num_points) -> (Tensor, Tensor)
```
### 功能描述
与`npu_furthest_point_sampling`功能相同，但输入略有不同。
### 参数说明
- `points(Tensor)`：点云数据，表示各点间的距离，数据类型为`float32, float16`。shape为`[B, N, N]`。其中`B`为batch size，`N`为点的数量。
- `num_points(int)`：采样点的数量。
### 返回值
- `Tensor`：采样后的点云数据，数据类型为`float32, float16`。shape为`[B, num_points]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import furthest_point_sample_with_dist
points = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32).npu()
out = furthest_point_sample_with_dist(points, 2)
print(out)
```
```text
tensor([[0, 2]], dtype=torch.int32)
```