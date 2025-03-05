## cal_anchors_heading
### 接口原型
```python
mx_driving.cal_anchors_heading(Tensor anchors, Tensor origin_pos) -> Tensor
```
### 功能描述
根据输入的 anchors 和起始点坐标计算方向。
### 参数说明
- `anchors(torch.Tensor)`：每个意图轨迹的序列坐标，数据类型为`float32`，shape 为 `[batch_size, anchors_num, seq_length, 2]`。
- `origin_pos(torch.Tensor or None)`：每个 anchor 的起始位置坐标，shape 为 `[batch_size, 2]`。
### 返回值
- `heading(torch.Tensor)`：每个 anchor 的轨迹点坐标方向（弧度），shape 为 `[batch_size, anchors_num, seq_length]`。
### 算子约束
- $\mathrm{1 \le batch\_size \le 2048}$
- $\mathrm{1 \le anchors\_num \le 10240}$
- $\mathrm{1 \le seq\_length \le 256}$
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```
import torch
import mx_driving
batch_size = 2
anchors_num = 64
seq_length = 24
anchors = torch.randn((batch_size, anchors_num, seq_length, 2)).npu()
origin_pos = torch.randn((batch_size, 2)).npu()
heading = mx_driving.cal_anchors_heading(anchors, origin_pos)
```