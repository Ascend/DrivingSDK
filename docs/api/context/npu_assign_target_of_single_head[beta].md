## npu_assign_target_of_single_head[beta]
### 接口原型
```python
mx_driving.npu_assign_target_of_single_head(Tensor boxes, Tensor cur_class_id, int num_classes int out_size_factor, float gaussian_overlap, int min_radius, FloatList voxel_size, FloatList pc_range, IntList feature_map_size, bool norm_bbox, bool with_velocity, bool flip_angle, int max_objs) -> (Tensor heatmap, Tensor anno_box, Tensor ind, Tensor mask)
```
### 功能描述
实现`centerpoint_head.py`脚本中`get_targets_single`函数的部分功能
### 参数说明
- `boxes(Tensor)`：每个目标的3D边界框信息，数据类型为`float32`，shape为`[Num_objs, W]`。
- `cur_class_id(Tensor)`：对应目标所在的热力图（heatmap）编号，数据类型为`int32`, shape为`[boxObjs]`。
- `num_classes(int)`：热力图（heatmap）的数量。
- `out_size_factor(int)`：特征图缩放因子。
- `gaussian_overlap(float)`：用于控制高斯半径的计算。
- `min_radius(int)`：高斯半径的最小取值。
- `voxel_size(FloatList)`：体素网格在x,y方向的单元大小。
- `pc_range(FloatList)`：x,y方向的点云范围。
- `feature_map_size(IntList)`：x方向的BEV特征图的大小。
- `norm_bbox(bool)`：是否对3D边界框的参数进行归一化。
- `with_velocity(bool)`：是否在目标检测任务中引入速度信息。
- `flip_angle(bool)`：是否在结果中将正弦余弦结果反转。
- `max_objs(int)`：处理boxes数量的上限。
### 返回值
- `heatmap(Tensor)`：通过计算得到的热力图（heatmap），数据类型为`float32`，shape为`[num_classes, feature_map_sizep[1], feature_map_size[0]]`。
- `anno_box(Tensor)`：经过计算后的直角三角形斜边，数据类型为`float32`，shape为`[max_objs, 10]`。
- `mask(Tensor)`：经过计算后的符合要求的boxes的掩码，数据类型为`uint8`，shape为`[max_objs]`。
- `ind(Tensor)`：经过计算后的符合要求的boxes中心点的偏移量，数据类型为`int64`，shape为`[max_objs]`。
### 算子约束
1. 若适配BEVDet模型，`W`应为9，其他模型（如需要）可能略有差别。
2. 所有参数和模型的配置保持一致。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving import npu_assign_target_of_single_head

out_size_factor = 8
gaussian_overlap = 0.1
min_radius = 2
voxel_size = [0.1, 0.1]
pc_range = [-51.2, -51.2]
feature_map_size = [128, 128]
num_classes = 1
norm_bbox = True
with_velocity = True
flip_angle = True
num_objs = 100
boxes = -50 + 100 * torch.rand((num_objs, 9), dtype=torch.float32).npu()
cur_class_id = torch.ones(num_objs, dtype=torch.int32).npu()
output = npu_assign_target_of_single_head(boxes, cur_class_id, num_classes, out_size_factor, gaussian_overlap, min_radius, voxel_size, pc_range, feature_map_size, norm_bbox, with_velocity, flip_angle)
```