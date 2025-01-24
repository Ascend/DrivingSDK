## deformable_aggregation
### 接口原型
```python
mx_driving.deformable_aggregation(Tensor feature_maps, Tensor spatial_shape, Tensor scale_start_index, Tensor sample_locations, Tensor weight) -> Tensor
```
兼容：
```python
mx_driving.fused.npu_deformable_aggregation(Tensor feature_maps, Tensor spatial_shape, Tensor scale_start_index, Tensor sample_locations, Tensor weight) -> Tensor
mx_driving.npu_deformable_aggregation(Tensor feature_maps, Tensor spatial_shape, Tensor scale_start_index, Tensor sample_locations, Tensor weight) -> Tensor
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
- c <= 256,且c / group为8的整数倍
- cam <= 6
- scale <= 4
- anchor <= 2048
- pts <= 2048
- group = 8
- sample_locations的值在[0, 1]之间。
- 每个输入tensor的数据量不超过1.5亿。
- 反向具有相同约束。
### 调用示例
```python
import torch, torch_npu
from mx_driving import deformable_aggregation

bs, num_feat, c, cam, anchor, pts, scale, group = 1, 2816, 256, 1, 10, 2000, 1, 8

feature_maps = torch.ones_like(torch.randn(bs,num_feat ,c))
spatial_shape = torch.tensor([[[32, 88]]])
scale_start_index = torch.tensor([[0]])
sampling_location = torch.rand(bs, anchor, pts, cam, 2)
weights = torch.randn(bs, anchor, pts, cam, scale, group)
feature_maps.requires_grad = True
out = deformable_aggregation(feature_maps.npu(), spatial_shape.npu(), scale_start_index.npu(), sampling_location.npu(), weights.npu())
grad_out_tensor = torch.ones_like(out)
out.backward(grad_out_tensor)
```