## multi_scale_deformable_attn(MultiScaleDeformableAttnFunction.Apply)
### 接口原型
```python
mx_driving.multi_scale_deformable_attn(Tensor value, Tensor value_spatial_shapes, Tensor value_level_start_index, Tensor sampling_locations, Tensor attention_weights) -> Tensor
```
兼容：
```
mx_driving.point.npu_multi_scale_deformable_attn_function(Tensor value, Tensor value_spatial_shapes, Tensor value_level_start_index, Tensor sampling_locations, Tensor attention_weights) -> Tensor
```
### 功能描述
多尺度可变形注意力机制, 将多个视角的特征图进行融合。
### 参数说明
- `value(Tensor)`：特征张量，数据类型为`float32, float16`。shape为`[bs, num_keys, num_heads, embed_dims]`。其中`bs`为batch size，`num_keys`为特征图的大小，`num_heads`为头的数量，`embed_dims`为特征图的维度，其中`embed_dims`需要为8的倍数。
- `value_spatial_shapes(Tensor)`：特征图的形状，数据类型为`int32, int64`。shape为`[num_levels, 2]`。其中`num_levels`为特征图的数量，`2`分别代表`H, W`。
- `value_level_start_index(Tensor)`：偏移量张量，数据类型为`int32, int64`。shape为`[num_levels]`。
- `sampling_locations(Tensor)`：位置张量，数据类型为`float32, float16`。shape为`[bs, num_queries, num_heads, num_levels, num_points, 2]`。其中`bs`为batch size，`num_queries`为查询的数量，`num_heads`为头的数量，`num_levels`为特征图的数量，`num_points`为采样点的数量，`2`分别代表`x, y`。
- `attention_weights(Tensor)`：权重张量，数据类型为`float32, float16`。shape为`[bs, num_queries, num_heads, num_levels, num_points]`。其中`bs`为batch size，`num_queries`为查询的数量，`num_heads`为头的数量，`num_levels`为特征图的数量，`num_points`为采样点的数量。
### 返回值
- `output(Tensor)`：融合后的特征张量，数据类型为`float32, float16`。shape为`[bs, num_queries, num_heads*embed_dims]`。
### 支持的型号
- Atlas A2 训练系列产品
### 约束说明
- 当前版本只支持`num_points * num_levels` &le; 64，`num_heads` &le; 8，`embed_dims` &le; 64。
- 注意：当`num_points * num_levels` = 64且`(embed_dims + 7) // 8` = 8时，`num_heads`最大为7。
### 调用示例
```python
import torch, torch_npu
from mx_driving import multi_scale_deformable_attn
bs, num_levels, num_heads, num_points, num_queries, embed_dims = 1, 1, 4, 8, 16, 32

shapes = torch.as_tensor([(100, 100)], dtype=torch.long)
num_keys = sum((H * W).item() for H, W in shapes)

value = torch.rand(bs, num_keys, num_heads, embed_dims) * 0.01
sampling_locations = torch.ones(bs, num_queries, num_heads, num_levels, num_points, 2) * 0.005
attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points) + 1e-5
level_start_index = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))

value.requires_grad_()
sampling_locations.requires_grad_()
attention_weights.requires_grad_()

out = multi_scale_deformable_attn(value.npu(), shapes.npu(), level_start_index.npu(), sampling_locations.npu(), attention_weights.npu())
out.backward(torch.ones_like(out))
```
