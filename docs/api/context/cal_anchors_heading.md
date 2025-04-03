## cal_anchors_heading
### 接口原型
```python
mx_driving.cal_anchors_heading(Tensor anchors, Tensor origin_pos) -> Tensor
```
### 功能描述
根据输入的 anchors 和起始点坐标计算方向。
### 参数说明
- `anchors(Tensor)`：每个意图轨迹的序列坐标，数据类型为`float32`，shape 为 `[batch_size, anchors_num, seq_length, 2]`。
- `origin_pos(Tensor)`：可选参数，每个 anchor 的起始位置坐标，数据类型为`float32`，shape 为 `[batch_size, 2]`。
### 返回值
- `heading(Tensor)`：每个 anchor 的轨迹点坐标方向（弧度），数据类型为`float32`，shape 为 `[batch_size, anchors_num, seq_length]`。
### 算子约束
- 1 <= batch\_size <= 2048
- 1 <= anchors\_num <= 10240
- 1 <= seq\_length <= 256
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```
import torch
import torch_npu
import mx_driving
batch_size = 2
anchors_num = 64
seq_length = 24
anchors = torch.randn((batch_size, anchors_num, seq_length, 2)).npu()
origin_pos = torch.randn((batch_size, 2)).npu()
heading = mx_driving.cal_anchors_heading(anchors, origin_pos)
```