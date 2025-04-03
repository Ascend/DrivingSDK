## geometric_kernel_attention
### 接口原型
```python
mx_driving.geometric_kernel_attention(Tensor value, Tensor spatial_shapes, Tensor level_start_index, 
                                      Tensor sampling_locations, Tensor attention_weights) -> Tensor
```
### 功能描述
根据采样点权重，计算采样点注意力特征值。
### 参数说明
- `value(Tensor)`：特征图张量，数据类型为`float32`，形状为`[B, num_keys, num_heads, dim]`。
- `spatial_shapes(Tensor)`：特征图空间形状张量，数据类型为`int32`，形状为`[num_levels, 2]`，最后一维为固定数值2，表示对应层（level）的`[H, W]`。
- `level_start_index(Tensor)`：每一层（level）起始索引张量，数据类型为`int`，形状为`[num_levels, ]`。
- `sampling_locations(Tensor)`：采样点坐标值张量，数据类型为`float32`，形状为`[B, num_queries, num_heads, num_levels, num_points, 2]`。最后一维为固定数值2，表示坐标点`[w, h]`。
- `attention_weights(Tensor)`：注意力权重张量，数据类型为`float32`，形状为`[B, num_queries, num_heads, num_levels, num_points]`。
### 返回值
- `output(Tensor)`：计算注意力权重后的特征图张量，数据类型为`float32`，形状为`[B, num_queries, num_heads*dim]`。
### 支持的型号
- Atlas A2 训练系列产品
### 约束说明
- 假设num_points向上取到8的整数倍后为num_points_align，当前版本反向算子仅支持`num_heads * num_levels * num_points_align * dim < 16 * 1024` 并且 `num_heads * num_levels * num_points_align < 256`
- 当前版本反向算子仅支持`dim % 8 == 0`
### 调用示例
```python
import torch, torch_npu
from mx_driving import geometric_kernel_attention

bs, num_queries, embed_dims, num_heads, num_levels, num_points = [6, 9680, 32, 8, 4, 4]
shapes = torch.tensor([60, 40] * num_levels).reshape(num_levels, 2)
num_keys = sum((H * W) for H, W in shapes)
value = torch.rand(bs, num_keys, num_heads, embed_dims) * 0.01 # Initialize value tensor with random values scaled by 0.01
sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2)
sampling_locations.uniform_(0, 40)
attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points) + 1e-5
offset = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))
grad_output = torch.rand(bs, num_queries, num_heads * embed_dims) * 1e-3

npu_value = value.float().npu()
npu_shapes = shapes.int().npu()
npu_offset = offset.int().npu()
npu_sampling_locations = sampling_locations.float().npu()
npu_attention_weights = attention_weights.float().npu()
npu_grad_output = grad_output.float().npu()

npu_value.requires_grad_()
npu_attention_weights.requires_grad_()

npu_output = geometric_kernel_attention(npu_value, npu_shapes, npu_offset, npu_sampling_locations, npu_attention_weights)
npu_output.backward(npu_grad_output)
```