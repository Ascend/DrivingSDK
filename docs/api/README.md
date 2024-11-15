> Note: 以prototype标注的接口，表示该接口为预发布接口，可能会有变动，不建议在生产环境中使用。
# 通用算子
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
```
## knn
### 接口原型
```python
mx_driving.common.knn(int k, Tensor xyz, Tensor center_xyz, bool Transposed) -> Tensor
```
### 功能描述
对center_xyz中的每个点找到xyz中对应batch中的距离最近的k个点，并且返回此k个点的索引值。
### 参数说明
- `xyz(Tensor)`：点数据，表示(x, y, z)三维坐标，数据类型为`float32`。shape为`[B, N, 3]`(当Transposed=False)或`[B, 3, N]`(当Transposed=True)。其中`B`为batch size，`N`为点的数量。
- `center_xyz(Tensor)`：点数据，表示(x, y, z)三维坐标，数据类型为`float32`。shape为`[B, npoint, 3]`(当Transposed=False)或`[B, 3, npoint]`(当Transposed=True)。其中`B`为batch size，`npoint`为点的数量。
- `k(int)`：采样点的数量。
- `Transposed(bool)`: 输入是否需要进行转置
### 返回值
- `idx(Tensor)`：采样后的索引数据，数据类型为`int32`。shape为`[B, k, npoint]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.common import knn
xyz = torch.tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]], dtype=torch.float32).npu()
center_xyz = torch.tensor([[[1, 2, 3]], [[1, 2, 3]]], dtype=torch.float32).npu()
idx = knn(2, xyz, center_xyz, False)
```
### 算子约束
1. k必须>0且<100。
2. xyz中的每个batch中的任意一个点到center_xyz对应batch中的任意一个点的距离必须在1e10f以内。
3. xyz和center_xyz的shape必须是3维，当Transposed=True时，xyz和center_xyz的shape的dim的第1维必须是3；当Transposed=False时，xyz和center_xyz的shape的dim的第2维必须是3。
4. 由于距离相同时排序为不稳定排序，存在距离精度通过但索引精度错误问题，与竞品无法完全对齐。

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
  - `indices`的取值必须为非负的有效索引值，参数`out`或`data_size`不为`None`时，`indices`的取值应该为输出张量在`dim`维的有效索引值。
- `out`：被更新张量，数据类型为`float32`，可选入参，默认为`None`，输入`out`不为`None`时，`out`中的元素参与平均值的计算，且
  - `out`的维度必须与`src`的维度相同。
  - `out`除第`dim`维外其余维的长度必须与`src`相同。
- `dim`：指定的维度，表示按照哪个维度进行分组平均计算，数据类型为`int32`，可选入参，默认取值为`0`，`dim`取值不超过`indices`的维度。
- `dim_size`：输出张量在`dim`维的长度，数据类型为`int32`，可选入参，默认为`None`，`dim_size`的取值必须为非负的有效长度值，该参数仅在输入`out`为`None`时生效。
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
src.requires_grad = True
out = scatter_mean(src.npu(), indices.npu(), None, dim)
grad_out_tensor = torch.ones_like(out)
out.backward(grad_out_tensor)
```
### 其他说明
- 该算子对尾块较大的场景较为亲和，对尾块很小的场景不亲和，其中，尾块表示`src`后`N`维的大小，`N = src.dim() - indices.dim()`。

## three_interpolate
### 接口原型
```python
mx_driving.common.three_interpolate(features: torch.Tensor, indices: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
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
```


## three_nn
### 接口原型
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
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.common import three_nn
source = torch.tensor([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]], dtype=torch.float32).npu()
target = torch.tensor([[[1, 2, 3]], [[1, 2, 3]]], dtype=torch.float32).npu()
dist, idx = three_nn(target, source)
```
### 算子约束
1. source和target的shape必须是3维，且source和target的shape的dim的第2维必须是3。
2. 距离相同时排序为不稳定排序，存在距离精度通过但索引精度错误问题，与竞品无法完全对齐。


## hypot
### 接口原型
```python
mx_driving.common.hypot(Tensor input, Tensor other) -> Tensor
```
### 功能描述
给出直角三角形的两边，返回它的斜边。
### 参数说明
- `input(Tensor)`：代表直角三角形第一条直角边的输入张量，数据类型为`float32`。
- `other(Tensor)`：代表直角三角形第二条直角边的输入张量，数据类型为`float32`。
### 返回值
- `Tensor`：经过计算后的直角三角形斜边，数据类型为`float32`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.common import hypot
input = torch.tensor([3,3,3], dtype=torch.float32).npu()
other = torch.tensor([4,4,4], dtype=torch.float32).npu()
out = hypot(input, other) # tensor([5.,5.,5.])
```
### 算子约束
1. input和other的shape必须是可广播的。


## assign_score_withk
### 接口原型
```python
mx_driving.common.assign_score_withk(Tensor scores, Tensor point_features, Tensor center_features, Tensor knn_idx, str aggregate='sum') -> Tensor
```
### 功能描述
根据`knn_idx`得到采样点及其邻居点的索引，计算`point_features`和`center_features`的差，并与`scores`相乘后在特征维度进行聚合，返回采样点的特征。
### 参数说明
- `scores(Tensor)`：权重矩阵的重要系数，数据类型为`float32`。Shape为`[B, npoint, K, M]`，其中`B`为batch size，`npoint`为采样点的数量，`K`为一个样本点及其邻居点的数量之和，`M`为权重矩阵集合的规模。
- `point_features(Tensor)`：所有点的特征，数据类型为`float32`。Shape为`[B, N, M, O]`，其中`N`为所有点的数量，`O`为特征数量。
- `center_features(Tensor)`：所有点的中心特征，数据类型为`float32`。Shape为`[B, N, M, O]`。
- `knn_idx[Tensor]`：采样点及其邻居点的索引，数据类型为`int64`。Shape为`[B, npoint, K]`。
- `aggregate`：聚合方式，默认为`sum`，数据类型为`str`。
### 返回值
- `output`：聚合后采样点的特征，数据类型为`float32`。Shape为`[B, O, npoint, K]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例

```python
import torch, torch_npu
from mx_driving.common import assign_score_withk
points = np.random.rand(4, 100, 8, 16).astype(np.float32)
centers = np.random.rand(4, 100, 8, 16).astype(np.float32)
scores = np.random.rand(4, 64, 10, 8).astype(np.float32)
knn_idx = np.random.randint(0, N, size=(4, 64, 10)).astype(np.int64)
output = assign_score_withk(torch.from_numpy(scores).npu(),
                            torch.from_numpy(points).npu(),
                            torch.from_numpy(centers).npu(),
                            torch.from_numpy(knn_idx).npu(),
                            "sum")
```
### 算子约束
- `npoint`和`K`都不大于`N`。


# 数据预处理算子
## npu_points_in_box
### 接口原型
```python
mx_driving.preprocess.npu_points_in_box(Tensor boxes, Tensor points) -> Tensor
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
from mx_driving.preprocess import npu_points_in_box
boxes = torch.tensor([[[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9]]], dtype=torch.float32).npu()
points = torch.tensor([[[1, 2, 3], [3, 4, 5]]], dtype=torch.float32).npu()
out = npu_points_in_box(boxes, points)
```

## npu_points_in_box_all
Note: 该接口命名将于2025年改为`points_in_boxes_all`。
### 接口原型
```python
mx_driving.preprocess.npu_points_in_box_all(Tensor boxes, Tensor points) -> Tensor
```
### 功能描述
判断点是否在框内。
### 参数说明
- `boxes(Tensor)`：框张量，数据类型为`float32`。shape 为`[B, M, 7]`。`7`分别代表`x, y, z, x_size, y_size, z_size, rz`。
- `points(Tensor)`：点张量，数据类型为`float32`。shape 为`[B, N, 3]`。`3`分别代表`x, y, z`。
### 返回值
- `boxes_idx_of_points(Tensor)`：同一`batch`下，各点是否在各框内的张量，数据类型为`int32`。shape 为`[B, N, M]`。
### 约束说明
- `boxes`和`points`的`B`必须相同。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.preprocess import npu_points_in_box_all
boxes = torch.tensor([[[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9]]], dtype=torch.float32).npu()
points = torch.tensor([[[1, 2, 5], [3, 4, 8]]], dtype=torch.float32).npu()
out = npu_points_in_box_all(boxes, points)
```

## RoipointPool3d
### 接口原型
```python
mx_driving.preprocess.RoipointPool3d(int num_sampled_points, Tensor points, Tensor point_features, Tensor boxes3d) -> (Tensor pooled_features, Tensor pooled_empty_flag)
```
### 功能描述
对每个3D方案的几何特定特征进行编码。
### 参数说明
- `num_sampled_points(int)`：特征点的数量，正整数。
- `points(Tensor)`：点张量，数据类型为`float32, float16`。shape 为`[B, N, 3]`。`3`分别代表`x, y, z`。
- `point_features(Tensor)`：点特征张量，数据类型为`float32, float16`。shape 为`[B, N, C]`。`C`分别代表`x, y, z`。
- `boxes3d(Tensor)`：框张量，数据类型为`float32, float16`。shape 为`[B, M, 7]`。`7`分别代表`x, y, z, x_size, y_size, z_size, rz`。
### 返回值
- `pooled_features(Tensor)`：点在框内的特征张量，数据类型为`float32, float16`。shape 为`[B, M, num, 3+C]`。
- `pooled_empty_flag(Tensor)`：所有点不在框内的空标记张量，数据类型为`int32`。shape 为`[B, M]`。
### 约束说明
- `points`、`point_features`和`boxes3d`的数据类型必须相同，以及`B`也必须相同。
- `num_sampled_points`必须小于等于`N`。
- 数据类型为`float32`时，建议`B`小于100、`N`小于等于2640、`M`小于等于48、`num_sampled_points`小于等于48，个别shape值略微超过建议值无影响，但所有shape值均大于建议值时，算子执行会发生错误。
- 数据类型为`float16`时，建议`B`小于100、`N`小于等于3360、`M`小于等于60、`num_sampled_points`小于等于60，个别shape值略微超过建议值无影响，但所有shape值均大于建议值时，算子执行会发生错误。
- `N`/`M`的值越大，性能劣化越严重，建议`N`小于`M`的六百倍，否则性能可能会低于0.1x A100。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.preprocess import RoIPointPool3d
num_sampled_points = 1
points = torch.tensor([[[1, 2, 3]]], dtype=torch.float).npu()
point_features = points.clone()
boxes3d = torch.tensor([[[1, 2, 3, 4, 5, 6, 1]]], dtype=torch.float).npu()
roipoint_pool3d = RoIPointPool3d(num_sampled_points)
pooled_features, pooled_empty_flag = roipoint_pool3d(points, point_features, boxes3d)
```


# 目标检测算子
## npu_boxes_overlap_bev
Note: 该接口命名将于2025年改为`boxes_overlap_bev`。
### 接口原型
```python
mx_driving.detection.npu_boxes_overlap_bev(Tensor boxes_a, Tensor boxes_b) -> Tensor
```
### 功能描述
计算bev视角下中两个边界框的重叠面积。
### 参数说明
- `boxes_a (Tensor)`：第一组bounding boxes，数据类型为`float32`。shape为`[M, 5]`。其中`5`分别代表`x1, y1, x2, y2, angle`, `x1, y1, x2, y2`代表box四个顶点的横纵坐标，`angle`代表box的弧度制旋转角。
- `boxes_b (Tensor)`：第二组bounding boxes，数据类型为`float32`。shape为`[N, 5]`。其中`5`分别代表`x1, y1, x2, y2, angle`, `x1, y1, x2, y2`代表box四个顶点的横纵坐标，`angle`代表box的弧度制旋转角。
### 返回值
- `area_overlap(Tensor)`：包含两组bounding boxes交叠面积的张量，数据类型为`float32`。shape为`[M, N]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.detection import npu_boxes_overlap_bev
boxes_a = torch.tensor([[0, 0, 2, 2, 0]], dtype=torch.float32).npu()
boxes_b = torch.tensor([[1, 1, 3, 3, 0]], dtype=torch.float32).npu()
area_overlap = npu_boxes_overlap_bev(boxes_a, boxes_b)
```
## box_iou_quadri
### 接口原型
```python
mx_driving.detection.box_iou_quadri(Tensor boxes_a, Tensor boxes_b, str mode='iou', bool aligned=False) -> Tensor
```
### 功能描述
计算两个边界框的IoU。
### 参数说明
- `boxes_a (Tensor)`：第一组bounding boxes，数据类型为`float32`。shape为`[M, 8]`。其中`8`分别代表`x1, y1, x2, y2, x3, y3, x4, y4`, 表示box四个顶点的横纵坐标。
- `boxes_b (Tensor)`：第二组bounding boxes，数据类型为`float32`。shape为`[N, 8]`。其中`8`分别代表`x1, y1, x2, y2, x3, y3, x4, y4`, 表示box四个顶点的横纵坐标。
- `mode (str)`：取值为`"iou"`时，计算IoU（intersection over union）；取值为`"iof"`时，计算IoF（intersection over foregroud）。
- `aligned (bool)`：取值为`True`时，只计算配对的box之间的结果；取值为`False`时，计算每对box之间的结果。
### 返回值
- `ious(Tensor)`：包含两组bounding boxes的IoU（`mode="iou"`）或IoF（`mode="iof"`）的张量，数据类型为`float32`。shape为`[M]`（`aligned=True`）或`[M, N]`（`aligned=False`）。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.detection import box_iou_quadri
boxes_a = torch.tensor([[7.0, 7.0, 8.0, 8.0, 9.0, 7.0, 8.0, 6.0]], dtype=torch.float32).npu()
boxes_b = torch.tensor([[7.0, 6.0, 7.0, 8.0, 9.0, 8.0, 9.0, 6.0]], dtype=torch.float32).npu()
ious = box_iou_quadri(boxes_a, boxes_b, mode="iou", aligned=False)
```
## npu_nms3d
### 接口原型
```python
mx_driving.detection.npu_nms3d(Tensor boxes, Tensor scores, float: iou_threshold) -> Tensor
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
from mx_driving.detection import npu_nms3d
boxes = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9]], dtype=torch.float32).npu()
scores = torch.tensor([1, 2], dtype=torch.float32).npu()
out = npu_nms3d(boxes, scores, 0.5)
```
## npu_nms3d_normal
### 接口原型
```python
mx_driving.detection.npu_nms3d_normal(Tensor boxes, Tensor scores, float: iou_threshold) -> Tensor
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
from mx_driving.detection import npu_nms3d_normal
boxes = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9]], dtype=torch.float32).npu()
scores = torch.tensor([1, 2], dtype=torch.float32).npu()
out = npu_nms3d_normal(boxes, scores, 0.5)
```
## npu_rotated_iou
### 接口原型
```python
mx_driving.detection.npu_rotated_iou(Tensor self, Tensor query_boxes, bool trans=False, int mode=0, bool is_cross=True, float v_threshold=0.0, float e_threshold=0.0) -> Tensor
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
from mx_driving.detection import npu_rotated_iou
a = np.random.uniform(0, 1, (2, 2, 5)).astype(np.float16)
b = np.random.uniform(0, 1, (2, 3, 5)).astype(np.float16)
box1 = torch.from_numpy(a).npu()
box2 = torch.from_numpy(b).npu()
iou = npu_rotated_iou(box1, box2, False, 0, True, 1e-5, 1e-5)
```
## npu_rotated_overlaps
### 接口原型
```python
mx_driving.detection.npu_rotated_overlaps(Tensor self, Tensor query_boxes, bool trans=False) -> Tensor
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
from mx_driving.detection import npu_rotated_overlaps
a = np.random.uniform(0, 1, (1, 3, 5)).astype(np.float16)
b = np.random.uniform(0, 1, (1, 2, 5)).astype(np.float16)
box1 = torch.from_numpy(a).npu()
box2 = torch.from_numpy(b).npu()
output = npu_rotated_overlaps(box1, box2, True)
```
## roi_align_rotated[beta]
### 接口原型
```python
mx_driving.detection.roi_align_rotated(Tensor feature_map, Tensor rois, float: spatial_scale,
                                       int: sampling_ratio, int: pooled_height, int: pooled_width, bool: aligned, bool: clockwise) -> Tensor
```
### 功能描述
计算旋转候选框的RoI Align池化特征图。
### 参数说明
- `feature map(Tensor)`：特征图张量，数据类型为`float32`，形状为`[B, C, H, W]`。
- `rois(Tensor)`：感兴趣区域张量，数据类型为`float32`，形状为`[n, 6]`。
- `spatial_scale(float)`：感兴趣区域边界框的缩放率，数据类型为`float32`。
- `sampling_ratio(int)`：采样率，数据类型为`int`。取值范围为非负整数。
- `pooled_height(int)`：池化特征图高度，数据类型为`int`。
- `pooled_width(int)`：池化特征图宽度，数据类型为`int`。
- `aligned(bool)`：是否对齐，数据类型为`bool`。值为`True`时，表示对齐, 值为`False`时，表示不对齐。
- `clockwise(bool)`：旋转候选框的旋转方向，数据类型为`bool`。值为`True`时，表示逆时针旋转，值为`False`时，表示顺时针旋转。
### 返回值
- `Tensor`：池化特征图张量，数据类型为`float32`，形状为`[n, C, pooled_height, pooled_width]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import math
import torch, torch_npu
import numpy as np
from mx_driving.detection import roi_align_rotated

feature_map = torch.rand([1, 3, 16, 16])
feature_map.requires_grad = True
rois = torch.Tensor(6, 8)
rois[0] = torch.randint(0, 1, (8,))
rois[1].uniform_(0, 16)
rois[2].uniform_(0, 16)
rois[3].uniform_(0, 16)
rois[4].uniform_(0, 16)
rois[5].uniform_(0, math.pi)

output = roi_align_rotated(feature_map.npu(), rois.npu(), 1, 1, 7, 7, True, True)
output.backward(torch.ones_like(output))
```
### 其他说明
在双线性插值采样过程中，当采样点`x`接近`-1`或`W`位置，`y`接近`-1`或`H`位置时，由于平台差异和计算误差，可能导致该采样点的精度无法与竞品精度完全对齐。

## roiaware_pool3d
### 接口原型
```python
mx_driving.detection.roiaware_pool3d(Tensor rois, Tensor pts, Tensor pts_feature,
                    Union[int, tuple] out_size, int max_pts_per_voxel, int mode) -> Tensor
```
### 功能描述
将输入的点云特征在ROI框内进行池化
### 参数说明
- `rois (Tensor)`：输入的RoI框坐标与尺寸，数据类型为`float32/float16`，shape为`[Roi_num, 7]`。
- `pts (Tensor)`：输入的点云坐标，数据类型为`float32/float16`，shape为`[Pts_num, 3]`。
- `pts_feature (Tensor)`：输入的点的特征向量，数据类型为`float32/float16`，shape为`[Pts_num, Channels]`。
- `out_size (Union)`：输出的RoI框内voxel的尺寸，数据类型为`int`或者`tuple`，shape为`[out_x, out_y, out_z]`。
- `max_pts_per_voxel (int)`：每个voxel内最大的点的个数，数据类型为`int`。
- `mode (int)`：池化的方式，0为maxpool, 1为avgpool，数据类型为`int`。
### 返回值
- `pooled_features (Tensor)`：池化得到的RoI框特征，数据类型为`float32/float16`，shape为`[Roi_num, out_x, out_y, out_z, Channels]`。
### 约束说明
- Roi_num <= 100
- Pts_num <= 1000
- Channels <= 1024
- 1 <= max_pts_per_voxel <=256，max_pts_per_voxel <= Pts_num
- 反向具有相同约束。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch
import math
import torch_npu
import mx_driving.detection

out_size = (5, 5, 5)
max_pts_per_voxel = 128
mode = 1

N = 40
npoints = 1000
channels = 1024

xyz_coor = np.random.uniform(-1, 1, size = (N, 3)).astype(np.float32)
xyz_size_num = np.random.uniform(5, 50, size = (1, 3))
xyz_size = (xyz_size_num * np.ones((N, 3))).astype(np.float32)
angle = np.radians(np.random.randint(0, 360, size = (N , 1))).astype(np.float32)

rois = np.concatenate((xyz_coor, xyz_size), axis=1)
rois = np.concatenate((rois, angle), axis=1)

pts = np.random.uniform(-5, 5, size = (npoints, 3)).astype(np.float32)
pts_feature = np.random.uniform(-1, 1, size=(npoints, channels)).astype(np.float32)

pooled_features_npu = mx_driving.detection.roiaware_pool3d(torch.tensor(rois).npu(), torch.tensor(pts).npu(),
                                                            torch.tensor(pts_feature).npu(), out_size, max_pts_per_voxel, mode)
```

## border_align
### 接口原型
```python
mx_driving.detection.border_align(Tensor feature_map, Tensor rois, int pooled_size) -> Tensor
```
### 功能描述
对输入的RoI框进行边缘特征提取。
### 参数说明
- `feature_map (Tensor)`：输入的特征图，数据类型为`float32`，shape为`[Batch_size, Channels, Height, Width]`。
- `rois (Tensor)`：输入的RoI框坐标，数据类型为`int32`，shape为`[Batch_size, Height * Width, 4]`。
- `pooled_size (int)`：在每条边上的采样点数，数据类型为`int`。
### 返回值
- `out_features (Tensor)`：提取到的RoI框特征，数据类型为`float32`，shape为`[Batch_size, Channels / 4, Height * Width, 4]`。
### 约束说明
- Batch_size <= 128
- Channels <= 8192, Channels % 4 == 0
- Height <= 256, Width <= 256
- 2 <= pooled_size <= 20
- 反向具有相同约束。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch
import torch_npu
import numpy as np
from mx_driving.detection import border_align

def generate_features(feature_shape):
    features = torch.rand(feature_shape)
    return features

def generate_rois(inputs):
    num_boxes = inputs.shape[0] * inputs.shape[2] * inputs.shape[3]
    xyxy = torch.rand(num_boxes, 4)
    xyxy[:, 0::2] = xyxy[:, 0::2] * inputs.size(3)
    xyxy[:, 1::2] = xyxy[:, 1::2] * inputs.size(2)
    xyxy[:, 2:] = xyxy[:, 0:2] + xyxy[:, 2:]
    rois = xyxy.view(inputs.shape[0], -1, 4).contiguous()
    return rois

batch_size = 2
input_channels = 16
input_height = 8
input_width = 8
pooled_size = 3
features = generate_features([batch_size, input_channels, input_height, input_width])
rois = generate_rois(features)
output = border_align(features.npu(), rois.npu(), pooled_size)
```

# 融合算子
## npu_multi_scale_deformable_attn_function
### 接口原型
```python
mx_driving.fused.npu_multi_scale_deformable_attn_function(Tensor value, Tensor shape, Tensor offset, Tensor locations, Tensor weight) -> Tensor
```
### 功能描述
多尺度可变形注意力机制, 将多个视角的特征图进行融合。
### 参数说明
- `value(Tensor)`：特征张量，数据类型为`float32, float16`。shape为`[bs, num_keys, num_heads, embed_dims]`。其中`bs`为batch size，`num_keys`为特征图的大小，`num_heads`为头的数量，`embed_dims`为特征图的维度，其中`embed_dims`需要为8的倍数。
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
from mx_driving.fused import npu_multi_scale_deformable_attn_function
bs, num_levels, num_heads, num_points, num_queries, embed_dims = 1, 1, 4, 8, 16, 32

shapes = torch.as_tensor([(100, 100)], dtype=torch.long)
num_keys = sum((H * W).item() for H, W in shapes)

value = torch.rand(bs, num_keys, num_heads, embed_dims) * 0.01
sampling_locations = torch.ones(bs, num_queries, num_heads, num_levels, num_points, 2) * 0.005
attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points) + 1e-5
level_start_index = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))

out = npu_multi_scale_deformable_attn_function(value.npu(), shapes.npu(), level_start_index.npu(), sampling_locations.npu(), attention_weights.npu())
```

## npu_max_pool2d
### 接口原型
```python
mx_driving.fused.npu_max_pool2d(Tensor x, int kernel_size, int stride, int padding) -> Tensor
```
### 功能描述
对输入进行最大池化，并输出最大池化值。
### 参数说明
- `x (Tensor)`：一组待池化对象，数据类型为`float32`，format为NCHW，输入数据量不超过10亿。
### 返回值
- `y (Tensor)`：池化后的最大值，数据类型为`float32`，format为NCHW。
### 约束说明
kernel_size仅支持3，stride仅支持2，padding仅支持1，且输入C轴数据量要求为8的倍数，H和W需要大于100。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.fused import npu_max_pool2d
kernel_size = 3
stride = 2
padding = 1
x = torch.randn(18, 64, 464, 800).npu()
res = npu_max_pool2d(x, kernel_size, stride, padding)
```

## npu_deformable_aggregation
### 接口原型
```python
mx_driving.fused.npu_deformable_aggregation(Tensor feature_maps, Tensor spatial_shape, Tensor scale_start_index, Tensor sample_locations, Tensor weight) -> Tensor
```
### 功能描述
可变形聚合，对于每个锚点实例，对多个关键点的多时间戳、视图、缩放特征进行稀疏采样后分层融合为实例特征，实现精确的锚点细化。
### 参数说明
- `feature_maps(Tensor)`：特征张量，数据类型为`float32`。shape为`[bs, num_feat, c]`。其中`bs`为batch size，`num_feat`为特征图的大小，`c`为特征图的维度。
- `spatial_shape(Tensor)`：特征图的形状，数据类型为`int32`。shape为`[cam, scale, 2]`。其中`cam`为相机数量，其中`scale`为每个相机的特征图数量，`2`分别代表H, W。
- `scale_start_index(Tensor)`：每个特征图的偏移位置张量，数据类型为`int32`。shape为`[cam, scale]`，其中`cam`为相机数量，其中`scale`每个相机的特征图数量。
- `sample_locations(Tensor)`：位置张量，数据类型为`float32`。shape为`[bs, anchor, pts, cam, 2]`。其中`bs`为batch size，`anchor`为锚点数量，`pts`为采样点的数量，`cam`为相机的数量，`2`分别代表y, x。
- `weight(Tensor)`：权重张量，数据类型为`float32`。shape为`[bs, anchor, pts, cam, scale, group]`。其中`bs`为batch size，`anchor`为锚点数量，`pts`为采样点的数量，`cam`为相机的数量，`scale`每个相机的特征图数量，`group`为分组数。
### 返回值
- `output(Tensor)`：输出结果张量，数据类型为`float32`。shape为`[bs, anchor, c]`。
### 支持的型号
- Atlas A2 训练系列产品
### 约束说明
- bs <= 128
- num_feat的值为spatial_shape中每幅图的特征数量之和
- c <= 256,且为group的整数倍
- cam <= 6
- scale <= 4
- anchor <= 2048
- pts <= 2048
- group <= 32,且为2的指数倍
- sample_locations的值在[0, 1]之间。
- 每个输入tensor的数据量不超过1.5亿。
- 反向具有相同约束。
### 调用示例
```python
import torch, torch_npu
from mx_driving.fused import npu_deformable_aggregation

bs, num_feat, c, cam, anchor, pts, scale, group = 1, 2816, 256, 1, 10, 2000, 1, 8

feature_maps = torch.ones_like(torch.randn(bs,num_feat ,c)).to(torch.float16)
spatial_shape = torch.tensor([[[32, 88]]])
scale_start_index = torch.tensor([[0]])
sampling_location = torch.rand(bs, anchor, pts, cam, 2)
weights = torch.randn(bs, anchor, pts, cam, scale, group)

out = npu_deformable_aggregation(feature_maps.npu(), spatial_shape.npu(), scale_start_index.npu(), sampling_location.npu(), weights.npu())
```

## deform_conv2d(DeformConv2dFunction.apply)
### 接口原型
```python
mx_driving.fused.deform_conv2d(Tensor x, Tensor offset, Tensor weight, Union[int, Tuple[int, ...]] stride, Union[int, Tuple[int, ...]] padding, Union[int, Tuple[int, ...]] dilation, int groups, int deformable_groups) -> Tensor
```
### 功能描述
可变形卷积。
### 参数说明
- `x(Tensor)`：输入特征，数据类型为`float32`，shape为`(n, c_in, h_in, w_in)`，其中`n`为 batch size，`c_in`为输入特征的通道数量，`h_in`为输入特征图的高，`w_in`为输入特征图的宽。
- `offset(Tensor)`：偏移量，数据类型为`float32`，shape 为`(n, 2 * k * k, h_out, w_out)`，其中`n`为 batch size，`k` 为卷积核大小，`h_out` 为输出特征图高，`w_out` 为输出特征图的宽。
- `weight(Tensor)`：卷积核权重，数据类型为`float32`，shape 为 `(c_out, c_in, k, k)`，其中 `c_out` 为输出的通道数，`c_in` 为输入的通道数，`k` 为卷积核大小。
- `stride(Union)`：卷积步长。
- `padding(Union)`：卷积的填充大小。
- `dilation(Union)`：空洞卷积大小。
- `groups(int)`：分组卷积大小，当前只支持1。
- `deformable_groups(int)`：将通道分成几组计算offsets，当前只支持1。
### 返回值
- `output(Tensor)`：输出张量，数据类型为`float32`，shape 为 `(n, c_out, h_out, w_out)`，其中`n`为 batch size，`c_out`为输出通道，`h_out` 为输出特征图高，`w_out` 为输出特征图的宽。
### 支持的型号
- Atlas A2 训练系列产品
### 约束说明
1. `deformable_groups`和`groups`当前只支持1。
2. `h_in`,`w_in`,`h_out`,`w_out`需满足
$$
w_{out}=(w_{in}+ 2 * padding - (dilation * (k - 1) + 1)) / stride + 1 \\
h_{out}=(h_{in}+ 2 * padding - (dilation * (k - 1) + 1)) / stride + 1 
$$
3. `c_in`需要为64的倍数。
### 调用示例
```python
import torch
import torch_npu
from mx_driving.fused import deform_conv2d, DeformConv2dFunction

n, c_in, h_in, w_in = 16, 64, 100, 200
c_out, k, h_out, w_out = 64, 3, 50, 100

x = torch.randn((n, c_in, h_in, w_in)).npu()
offset = torch.randn((n, 2 * k * k, h_out, w_out)).npu()
weight = torch.randn((c_out, c_in, k, k)).npu()
stride = 1
padding = 1
dilation = 1
groups = 1
deformable_groups = 1

output = deform_conv2d(x, offset, weight, stride, padding, dilation, groups, deformable_groups)
output = DeformConv2dFunction.apply(x, offset, weight, stride, padding, dilation, groups, deformable_groups)
```
## modulated_deform_conv2d(ModulatedDeformConv2dFunction.apply)
### 接口原型
```python
mx_driving.fused.modulated_deform_conv2d(Tensor x, Tensor offset, Tensor mask, Tensor weight, Tensor bias, Union[int, Tuple[int, ...]] stride, Union[int, Tuple[int, ...]] padding, Union[int, Tuple[int, ...]] dilation, int groups, int deformable_groups) -> Tensor
```
### 功能描述
在可变形卷积的基础之上加上了 modulation 机制，通过调控输出特征的幅度，提升可变形卷积的聚焦相关区域的能力。
### 参数说明
- `x(Tensor)`：输入特征，数据类型为`float32`，shape为`(n, c_in, h_in, w_in)`，其中`n`为 batch size，`c_in`为输入特征的通道数量，`h_in`为输入特征图的高，`w_in`为输入特征图的宽。
- `offset(Tensor)`：偏移量，数据类型为`float32`，shape 为`(n, 2 * k * k, h_out, w_out)`，其中`n`为 batch size，`k` 为卷积核大小，`h_out` 为输出特征图高，`w_out` 为输出特征图的宽。
- `mask(Tensor)`：掩码，用于调控输出特征的幅度，数据类型为`float32`，shape 为`(n, k * k, h_out, w_out)`，其中`n`为 batch size，k 为卷积核大小，`h_out` 为输出特征图高，`w_out` 为输出特征图的宽。
- `weight(Tensor)`：卷积核权重，数据类型为`float32`，shape 为 `(c_out, c_in, k, k)`，其中 `c_out` 为输出的通道数，`c_in` 为输入的通道数，`k` 为卷积核大小。
- `bias(Tensor)`：偏置，暂不支持bias，传入 `None` 即可。
- `stride(Union)`：卷积步长。
- `padding(Union)`：卷积的填充大小。
- `dilation(Union)`：空洞卷积大小。
- `groups(int)`：分组卷积大小，当前只支持1。
- `deformable_groups(int)`：将通道分成几组计算offsets，当前只支持1。
### 返回值
- `output(Tensor)`：输出张量，数据类型为`float32`，shape 为 `(n, c_out, h_out, w_out)`，其中`n`为 batch size，`c_out`为输出通道，`h_out` 为输出特征图高，`w_out` 为输出特征图的宽。
### 支持的型号
- Atlas A2 训练系列产品
### 约束说明
1. `deformable_groups`和`groups`当前只支持1。
2. `h_in`,`w_in`,`h_out`,`w_out`需满足
$$
w_{out}=(w_{in}+ 2 * padding - (dilation * (k - 1) + 1)) / stride + 1 \\
h_{out}=(h_{in}+ 2 * padding - (dilation * (k - 1) + 1)) / stride + 1 
$$
3. `c_in`需要为64的倍数。
### 调用示例
```python
import torch
import torch_npu
from mx_driving.fused import modulated_deform_conv2d, ModulatedDeformConv2dFunction

n, c_in, h_in, w_in = 16, 64, 100, 200
c_out, k, h_out, w_out = 64, 3, 50, 100

x = torch.randn((n, c_in, h_in, w_in)).npu()
offset = torch.randn((n, 2 * k * k, h_out, w_out)).npu()
mask = torch.randn((n, k * k, h_out, w_out)).npu()
weight = torch.randn((c_out, c_in, k, k)).npu()
bias = None
stride = 1
padding = 1
dilation = 1
groups = 1
deformable_groups = 1

output = modulated_deform_conv2d(x, offset, mask, weight, bias, 
  stride, padding, dilation, groups, deformable_groups)
output = ModulatedDeformConv2dFunction.apply(x, offset, mask, weight, bias, 
  stride, padding, dilation, groups, deformable_groups)
```

# 点云算子
## bev_pool
### 接口原型
```python
mx_driving.point.bev_pool(Tensor feat, Tensor geom_feat, int B, int D, int H, int W) -> Tensor
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
from mx_driving.point import bev_pool
feat = torch.rand(4, 256).npu()
feat.requires_grad_()
geom_feat = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3]], dtype=torch.int32).npu()
bev_pooled_feat = bev_pool(feat, geom_feat, 4, 1, 256, 256)
loss = bev_pooled_feat.sum()
loss.backward()
```
## bev_pool_v2
### 接口原型
```python
mx_driving.point.bev_pool_v2(Tensor depth, Tensor feat, Tensor ranks_depth, Tensor ranks_feat, Tensor ranks_bev,
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
from mx_driving.point import bev_pool_v2
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
```
## furthest_point_sample_with_dist
### 接口原型
```python
mx_driving.point.furthest_point_sample_with_dist(Tensor points, int num_points) -> Tensor
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
from mx_driving.point import furthest_point_sample_with_dist
points = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32).npu()
out = furthest_point_sample_with_dist(points, 2)
```
## npu_furthest_point_sampling
### 接口原型
```python
mx_driving.point.npu_furthest_point_sampling(Tensor points, int num_points) -> Tensor
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
from mx_driving.point import npu_furthest_point_sampling
points = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32).npu()
out = npu_furthest_point_sampling(points, 2)
```

### 算子约束
1. points输入shape[B, N, 3]的总大小(B x N x 3)不应该超过383166
## npu_group_points
Note：该接口命名将于2025年改为'group_points'。
### 接口原型
```python
mx_driving.point.npu_group_points(Tensor features, Tensor indices) -> Tensor
```
### 功能描述
点云数据按照索引重新分组。
### 参数说明
- `features`：需要被插值的特征，数据类型为`float32`，维度为（B, C, N）。
- `indices`：获取目标特征计算的索引，数据类型为`int32`，维度为（B, npoints, nsample）。
### 返回值
- `output(Tensor)`：分组后的点云数据，数据类型为`float32`。shape为`[B, C, npoints, nsample]`。
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
from mx_driving.point import npu_group_points


indices = torch.tensor([[[0, 2, 5, 5], [1, 0, 5, 0], [2, 1, 4, 4]]]).int().npu()
features = torch.tensor([[[0.9178, -0.7250, -1.6587, 0.0715, -0.2252, 0.4994],
                          [0.6190, 0.1755, -1.7902, -0.5852, -0.3311, 1.9764],
                          [1.7567, 0.0740, -1.1414, 0.4705, -0.3197, 1.1944],
                          [-0.2343, 0.1194, 0.4306, 1.3780, -1.4282, -0.6377],
                          [0.7239, 0.2321, -0.6578, -1.1395, -2.3874, 1.1281]]],
                          dtype=torch.float32).npu()
output = npu_group_points(features, indices)
```

## npu_add_relu
### 接口原型
```python
mx_driving.fused.npu_add_relu(Tensor x, Tensor y) -> Tensor
```
### 功能描述
与`relu(x + y)`功能相同。
### 参数说明
- `x(Tensor)`：输入数据，数据类型为`float32`，shape无限制。
- `y(Tensor)`：输入数据，数据类型为`float32`，shape需要和x一致。
### 返回值
- `Tensor`：输出数据，数据类型为`float32`，shape和x一致。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving.fused import npu_add_relu
x = torch.tensor([[[1, 2, 3], [-1, 5, 6], [7, 8, 9]]], dtype=torch.float32).npu()
y = torch.tensor([[[1, 2, 3], [-1, -2, 6], [7, 8, 9]]], dtype=torch.float32).npu()
out = npu_add_relu(x, y)
```
### 算子约束
- 输入`x`与输入`y`的shape和dtype需要保持一致，不支持广播。
- 仅在x的元素个数超过2000000时，相较于`relu(x + y)`有性能提升。

## voxelization
### 接口原型
```python
mx_driving.point.voxelization(Tensor points, List[float] voxel_size, List[float] coors_range, int max_points=-1, int max_voxels=-1, bool deterministic=True) -> Tensor
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
from mx_driving.point import Voxelization
points = torch.randint(-20, 100, [16, 3], dtype=torch.float32).npu()
coors_range = [0, -40, -3, 70.4, 40, 1]
max_points = -1
voxel_size = [0.5, 0.5, 0.5]
dynamic_voxelization = Voxelization(voxel_size, coors_range, max_points)
coors = dynamic_voxelization.forward(points)
```
## npu_dynamic_scatter
### 接口原型
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
from mx_driving.point import npu_dynamic_scatter

feats = torch.tensor([[1, 2, 3], [3, 2, 1], [7, 8, 9], [9, 8, 7]], dtype=torch.float32).npu()
coors = torch.tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]], dtype=torch.int32).npu()
voxel_feats, voxel_coors = npu_dynamic_scatter(feats, coors, 'max')

```
## unique_voxel
### 接口原型
```python
mx_driving._C.unique_voxel(Tensor voxels) -> int, Tensor, Tensor, Tensor, Tensor
```
### 功能描述
对输入的点云数据进行去重处理。
### 参数说明
- `voxels (Tensor)`：数据语义为索引，数据类型为`int32`，shape为`[N]`。
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
from mx_driving._C import unique_voxel
voxels = np.random.randint(0, 1024, (100000,)).astype(np.int32)
voxels_npu = torch.from_numpy(voxels).npu()
num_voxels, uni_voxels, uni_indices, argsort_indices, uni_argsort_indices = unique_voxel(voxels_npu)

```


## voxel_pooling_train
### 接口原型
```python
mx_driving.point.npu_voxel_pooling_train(Tensor geom_xyz, Tensor input_features, List[int] voxel_num) -> Tensor
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
import mx_driving.point

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

result_npu = mx_driving.point.npu_voxel_pooling_train(geom_npu, feature_npu, voxel_num)
```

# 稀疏卷积算子(beta)
## SparseConv3d(beta)
### 接口原型
```python
mx_driving.spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, mode='mmcv') -> SparseConvTensor
```
### 功能描述
稀疏卷积
### 参数说明
- `in_channels(int)`：输入数据的通道数
- `out_channels(int)`：输出通道数
- `kernel_size(List(int)/Tuple(int)/int)`：卷积神经网络中卷积核的大小
- `stride(List(int)/Tuple(int)/int)`：卷积核在输入数据上滑动时的步长
- `dilation(List(int)/Tuple(int)/int)`：空洞卷积大小
- `groups(int)`：分组卷积
- `bias(bool)`：偏置项
- `indice_key(String)`：该输入用于复用之前计算的索引信息
- `mode(String)`：区分了`mmcv`和`spconv`两种不同框架下的稀疏卷积
### 返回值
- `SparseConvTensor`：存储了输出的特征值`out_feature`，对应索引位置`out_indices`和对应的spatital_shape。
### 支持的型号
- Atlas A2 训练系列产品
### 约束说明
- `kernel_size`当前支持数据类型为三维List/Tuple或Int，值域为`[1, 3]`
- `stride`当前支持数据类型为三维List/Tuple或Int
- `dilation`，`groups`当前仅支持值为1
- 对于反向也是同样的约束。
### 调用示例
```python
import torch,torch_npu
import numpy as np
from mx_driving.spconv import SparseConv3d, SparseConvTensor

def generate_indice(batch, height, width, depth, actual_num):
    base_indices = np.random.permutation(np.arange(batch * height * width * depth))[:actual_num]
    base_indices = np.sort(base_indices)
    b_indice = base_indices // (height * width * depth)
    base_indices = base_indices % (height * width * depth)
    h_indice = base_indices // (width * depth)
    base_indices = base_indices // (width * depth)
    w_indice = base_indices // depth
    d_indice = base_indices % depth
    indices = np.concatenate((b_indice, h_indice, w_indice, d_indice)).reshape(4, actual_num)
    return indices

actual_num = 20
batch = 4
spatial_shape = [9, 9, 9]
indices = torch.from_numpy(generate_indice(batch, spatial_shape[0], spatial_shape[1], spatial_shape[2], actual_num)).int().transpose(0, 1).contiguous().npu()
feature = tensor_uniform = torch.rand(actual_num, 16).npu()
feature.requires_grad = True
x = SparseConvTensor(feature, indices, spatial_shape, batch)
net = SparseConv3d(in_channels=16, out_channels=32, kernel_size=3).npu()
out = net(x)
dout = torch.ones_like(out.features).float().npu()
out.features.backward(dout)
```


## SparseInverseConv3d(beta)
### 接口原型
```python
mx_driving.spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, output_padding=0,bias=True, indice_key=None, mode='mmcv') -> SparseConvTensor
```
### 功能描述
稀疏逆卷积
### 参数说明
- `in_channels(int)`：输入数据的通道数
- `out_channels(int)`：输出通道数
- `kernel_size(List(int)/Tuple(int)/int)`：卷积神经网络中卷积核的大小
- `stride(List(int)/Tuple(int)/int)`：卷积核在输入数据上滑动时的步长
- `dilation(List(int)/Tuple(int)/int)`：空洞卷积大小
- `groups(int)`：分组卷积
- `bias(bool)`：偏置项
- `indice_key(String)`：该输入用于复用之前计算的索引信息
- `mode(String)`：区分了`mmcv`和`spconv`两种不同框架下的稀疏卷积
### 返回值
- `SparseConvTensor`：存储了输出的特征值`out_feature`，对应索引位置`out_indices`和对应的spatital_shape。
### 支持的型号
- Atlas A2 训练系列产品
### 约束说明
- `kernel_size`当前支持数据类型为三维List/Tuple或Int，值域为`[1, 3]`
- `stride`当前支持数据类型为三维List/Tuple或Int
- `dilation`，`groups`当前仅支持值为1
- 对于反向也是同样的约束。
### 调用示例
```python
import torch,torch_npu
import numpy as np
from mx_driving.spconv import SparseInverseConv3d, SparseConvTensor

def generate_indice(batch, height, width, depth, actual_num):
    base_indices = np.random.permutation(np.arange(batch * height * width * depth))[:actual_num]
    base_indices = np.sort(base_indices)
    b_indice = base_indices // (height * width * depth)
    base_indices = base_indices % (height * width * depth)
    h_indice = base_indices // (width * depth)
    base_indices = base_indices // (width * depth)
    w_indice = base_indices // depth
    d_indice = base_indices % depth
    indices = np.concatenate((b_indice, h_indice, w_indice, d_indice)).reshape(4, actual_num)
    return indices

actual_num = 20
batch = 4
spatial_shape = [9, 9, 9]
indices = torch.from_numpy(generate_indice(batch, spatial_shape[0], spatial_shape[1], spatial_shape[2], actual_num)).int().transpose(0, 1).contiguous().npu()
feature = tensor_uniform = torch.rand(actual_num, 16).npu()
feature.requires_grad = True
x = SparseConvTensor(feature, indices, spatial_shape, batch)
net = SparseInverseConv3d(in_channels=16, out_channels=32, kernel_size=3).npu()
out = net(x)
dout = torch.ones_like(out.features).float().npu()
out.features.backward(dout)
```


## SubMConv3d(beta)
### 接口原型
```python
mx_driving.spconv.SubMConv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, mode='mmcv') -> SparseConvTensor
```
### 功能描述
稀疏卷积，只有当卷积核中心参与计算时，才会影响输出
### 参数说明
- `in_channels(int)`：输入数据的通道数
- `out_channels(int)`：输出通道数
- `kernel_size(List(int)/Tuple(int)/int)`：卷积神经网络中卷积核的大小
- `stride(List(int)/Tuple(int)/int)`：卷积核在输入数据上滑动时的步长
- `dilation(List(int)/Tuple(int)/int)`：空洞卷积大小
- `groups(int)`：分组卷积
- `bias(bool)`：偏置项
- `indice_key(String)`：该输入用于复用之前计算的索引信息
- `mode(String)`：区分了`mmcv`和`spconv`两种不同框架下的稀疏卷积
### 返回值
- `SparseConvTensor`：存储了输出的特征值`out_feature`，对应索引位置`out_indices`和对应的spatital_shape。
### 支持的型号
- Atlas A2 训练系列产品
### 约束说明
- `kernel_size`当前支持数据类型为三维List/Tuple或Int，当前值仅支持1、3
- `stride`当前支持数据类型为三维List/Tuple或Int,当前仅支持值为1
- `dilation`，`groups`当前仅支持值为1
- 对于反向也是同样的约束。
### 调用示例
```python
import torch,torch_npu
import numpy as np
from mx_driving.spconv import SubMConv3d, SparseConvTensor

def generate_indice(batch, height, width, depth, actual_num):
    base_indices = np.random.permutation(np.arange(batch * height * width * depth))[:actual_num]
    base_indices = np.sort(base_indices)
    b_indice = base_indices // (height * width * depth)
    base_indices = base_indices % (height * width * depth)
    h_indice = base_indices // (width * depth)
    base_indices = base_indices // (width * depth)
    w_indice = base_indices // depth
    d_indice = base_indices % depth
    indices = np.concatenate((b_indice, h_indice, w_indice, d_indice)).reshape(4, actual_num)
    return indices

actual_num = 20
batch = 4
spatial_shape = [9, 9, 9]
indices = torch.from_numpy(generate_indice(batch, spatial_shape[0], spatial_shape[1], spatial_shape[2], actual_num)).int().transpose(0, 1).contiguous().npu()
feature = tensor_uniform = torch.rand(actual_num, 16).npu()
feature.requires_grad = True
x = SparseConvTensor(feature, indices, spatial_shape, batch)
net = SubMConv3d(in_channels=16, out_channels=32, kernel_size=3).npu()
out = net(x)
dout = torch.ones_like(out.features).float().npu()
out.features.backward(dout)
```

