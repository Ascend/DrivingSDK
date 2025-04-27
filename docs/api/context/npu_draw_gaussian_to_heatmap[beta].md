## draw_gaussian_to_heatmap[beta]
### 接口原型
```python
mx_driving.draw_gaussian_to_heatmap(Tensor mask, Tensor cur_class_id, Tensor center_int, Tensor radius, int feature_map_size_x, int feature_map_size_y, int num_classes) -> (Tensor heatmap)
```
### 功能描述
实现`centerpoint_head.py`脚本中`get_targets_single`函数的部分功能，与`npu_gaussian`一起使用
### 参数说明
- `mask(Tensor)`：每个目标的3D边界框信息，数据类型为`uint_8`，shape为`[maxObjs, W]`。
- `cur_class_id(Tensor)`：对应目标所在的热力图（heatmap）编号，数据类型为`int32`, shape为`[Num_objs]`。
- `center_int_trans(Tensor)`：经过`npu_gaussian`的直角三角形斜边转置后的Tensor, 数据类型为`int32`, shape为`[2, minObjs]`。
- `radius(Tensor)`：通过`npu_gaussian`计算后的高斯半径，数据类型为`int32`
- `feature_map_size_x(int)`：x方向的BEV特征图的大小。
- `feature_map_size_y(int)`：y方向的BEV特征图的大小。
- `num_classes(int)`：热力图（heatmap）的数量。
### 返回值
- `heatmap(Tensor)`：通过计算得到的热力图（heatmap），数据类型为`float32`，shape为`[max_objs]`。
### 算子约束
1. 须和`npu_gaussian`API一起使用。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving import npu_gaussian, npu_draw_gaussian_to_heatmap

out_size_factor = 8
gaussian_overlap = 0.1
min_radius = 2
voxel_size_x = 0.1
voxel_size_y = 0.1
pc_range_x = -51.2
pc_range_y = -51.2
feature_map_size_x = 128
feature_map_size_y = 128
num_classes = 1
norm_bbox = True
with_velocity = True
num_objs = 100
boxes = -50 + 100 * torch.rand((num_objs, 9), dtype=torch.float32).npu()
cur_class_id = torch.ones(num_objs, dtype=torch.int32).npu()

output = npu_gaussian(boxes, out_size_factor, gaussian_overlap, min_radius, voxel_size_x, voxel_size_y, pc_range_x, pc_range_y, feature_map_size_x, feature_map_size_y, norm_bbox, with_velocity)
center_int_trans = output[0].t().contiguous()
radius = output[1]
mask = output[2]
heatmap = npu_draw_gaussian_to_heatmap(mask, cur_class_id, center_int_trans, radius, feature_map_size_x, feature_map_size_y, num_classes)
```