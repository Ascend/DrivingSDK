## roiaware_pool3d
### 接口原型
```python
mx_driving.roiaware_pool3d(Tensor rois, Tensor pts, Tensor pts_feature,
                    Union[int, tuple] out_size, int max_pts_per_voxel, int mode) -> Tensor
```
兼容：
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
- 2 <= max_pts_per_voxel <=256，max_pts_per_voxel <= Pts_num
- out_x, out_y, out_z <=16
- 反向具有相同约束。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch
import math
import torch_npu
from mx_driving import roiaware_pool3d
import numpy as np

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
rois = torch.tensor(rois).npu()

pts = np.random.uniform(-5, 5, size = (npoints, 3)).astype(np.float32)
pts = torch.tensor(pts).npu()
pts_feature = np.random.uniform(-1, 1, size=(npoints, channels)).astype(np.float32)
pts_feature = torch.tensor(pts_feature).npu()

pooled_features_npu = roiaware_pool3d(rois, pts, pts_feature, out_size, max_pts_per_voxel, mode)

pooled_features_npu.requires_grad = True
grad_out_tensor = torch.ones_like(pooled_features_npu)
pooled_features_npu.backward(grad_out_tensor)
```