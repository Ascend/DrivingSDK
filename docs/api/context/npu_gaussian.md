## npu_gaussian
### 接口原型
```python
mx_driving.npu_gaussian(Tensor boxes, int out_size_factor, float gaussian_overlap, int min_radius, float voxel_size_x, float voxel_size_y, float pc_range_x,
    float pc_range_y, int feature_map_size_x, int feature_map_size_y, bool norm_bbox, bool with_velocity, bool flip_angle, int max_objs) -> (Tensor center_int, Tensor radius, Tensor mask, Tensor ind, Tensor anno_box)
```
### 功能描述
实现`centerpoint_head.py`脚本中`get_targets_single`函数的部分功能，`draw_gaussian`函数未适配。
### 参数说明
- `boxes(Tensor)`：每个目标的3D边界框信息，数据类型为`float32`，shape为`[Num_objs, W]`。
- `out_size_factor(int)`：特征图缩放因子。
- `gaussian_overlap(float)`：用于控制高斯半径的计算。
- `min_radius(int)`：高斯半径的最小取值。
- `voxel_size_x(float)`：体素网格在x方向的单元大小。
- `voxel_size_y(float)`：体素网格在y方向的单元大小。
- `pc_range_x(float)`：x方向的点云范围。
- `pc_range_y(float)`：y方向的点云范围。
- `feature_map_size_x(int)`：x方向的BEV特征图的大小。
- `feature_map_size_y(int)`：y方向的BEV特征图的大小。
- `norm_bbox(bool)`：是否对3D边界框的参数进行归一化。
- `with_velocity(bool)`：是否在目标检测任务中引入速度信息。
- `flip_angle(bool)`：是否在结果中将正弦余弦结果反转。
- `max_objs(int)`：处理boxes数量的上限，BEVDet模型中为500。
### 返回值
- `center_int(Tensor)`：经过计算后的直角三角形斜边，数据类型为`int32`，shape为`[minObjs, 2]`。
- `radius(Tensor)`：经过计算后的高斯半径，数据类型为`int32`，shape为`[minObjs]`。
- `mask(Tensor)`：经过计算后的符合要求的boxes的掩码，数据类型为`uint8`，shape为`[max_objs]`。
- `ind(Tensor)`：经过计算后的符合要求的boxes中心点的偏移量，数据类型为`int64`，shape为`[max_objs]`。
- `anno_box(Tensor)`：经过计算后的直角三角形斜边，数据类型为`float32`，shape为`[max_objs, 10]`。
### 算子约束
1. 若适配BEVDet模型，`W`应为9。
2. 所有参数和模型的配置保持一致。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving import npu_gaussian

out_size_factor = 8
gaussian_overlap = 0.1
min_radius = 2
voxel_size_x = 0.1
voxel_size_y = 0.1
pc_range_x = -51.2
pc_range_y = -51.2
feature_map_size_x = 128
feature_map_size_y = 128
norm_bbox = True
with_velocity = True
flip_angle = True
num_objs = 100
boxes = -50 + 100 * torch.rand((num_objs, 9), dtype=torch.float32).npu()
output = npu_gaussian(boxes, out_size_factor, gaussian_overlap, min_radius, voxel_size_x, voxel_size_y, pc_range_x, pc_range_y, feature_map_size_x, feature_map_size_y, norm_bbox, with_velocity, flip_angle)
```