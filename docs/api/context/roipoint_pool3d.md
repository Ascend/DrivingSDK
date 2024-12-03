## roipoint_pool3d
### 接口原型
```python
mx_driving.roipoint_pool3d(int num_sampled_points, Tensor points, Tensor point_features, Tensor boxes3d) -> (Tensor pooled_features, Tensor pooled_empty_flag)
```
兼容
```python
mx_driving.preprocess.roipoint_pool3d(int num_sampled_points, Tensor points, Tensor point_features, Tensor boxes3d) -> (Tensor pooled_features, Tensor pooled_empty_flag)
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
from mx_driving import roipoint_pool3d
num_sampled_points = 1
points = torch.tensor([[[1, 2, 3]]], dtype=torch.float).npu()
point_features = points.clone()
boxes3d = torch.tensor([[[1, 2, 3, 4, 5, 6, 1]]], dtype=torch.float).npu()
pooled_features, pooled_empty_flag = roipoint_pool3d(num_sampled_points, points, point_features, boxes3d)
```