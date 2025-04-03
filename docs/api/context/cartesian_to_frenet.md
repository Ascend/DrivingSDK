## cartesian_to_frenet
### 接口原型
```python
mx_driving.cartesian_to_frenet(Tensor pt, Tensor poly_line) -> Tuple[torch.Tensor, torch.Tensor， torch.Tensor]
```
### 功能描述
根据给定的poly_line，将输入的笛卡尔坐标系中的坐标转换为Frenet坐标系中的表示。
### 参数说明
- `pt(Tensor)`：进行坐标转换的点在笛卡尔坐标系中的坐标，数据类型为`float32`，shape 为 `[batch_size, num_point, 2]`。
- `poly_line(Tensor)`：定义中线的点，数据类型为`float32`，shape 为 `[batch_size, num_polyline_point, 2]`。
### 返回值
- `poly_start(Tensor)`：每个 point 在poly_line上对应的起始点，数据类型为`float32`，shape 为 `[batch_size, num_point, 2]`。
- `poly_end(Tensor)`：每个 point 在poly_line上对应的终点，数据类型为`float32`，shape 为 `[batch_size, num_point, 2]`。
- `sl(Tensor)`：每个 point 的Frenet坐标系表示，数据类型为`float32`，shape 为 `[batch_size, num_point, 2]`。
### 算子约束
- 1 <= batch\_size <= 10000
- 1 <= num\_point <= 256
- 1 < num\_polyline\_point <= 256
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```
import torch
import torch_npu
import mx_driving
batch_size = 2
num_point = 40
num_poly_line_point = 20
batch_size = 5000
pt = torch.randn((batch_size, num_point, 2)).npu()
poly_line = torch.randn((batch_size, num_poly_line_point, 2)).npu()
poly_start, poly_end, sl = mx_driving.cartesian_to_frenet(pt, poly_line)
```