## sigmoid_focal_loss
### 接口原型
```python
mx_driving.sigmoid_focal_loss(Tensor logit, Tensor target, float gamma=2, float alpha=0.25, Tensor weight=None, str reduction='mean') -> Tensor
```
### 功能描述
先计算输入logit中每个元素的sigmoid值,然后计算sigmoid值与类别目标值之间的Focal Loss，功能与mmcv库的sigmoid_focal_loss一致。
### 参数说明
- `logit (Tensor)`：表示全部样本的分类预测值，数据类型为`float32`。Shape为`[N, C]`，其中`N`为样本数量，`C`为类别数量。
- `target (Tensor)`：表示全部样本的分类目标值，数据类型为`int64`。Shape为`[N]`。
- `gamma (float)`：用于平衡易分类样本和难分类样本的超参数，默认值为2.0。
- `alpha (float)`：用于平衡正样本和负样本的超参数，默认值为0.25。
- `weight (Tensor)`：表示全部类别的权重系数，数据类型为`float32`。Shape为`[C]`。
- `reduction (str)`：规约方式，数据类型为`str`，默认为`mean`。
### 返回值
- `output (Tensor)`：表示输入中每个元素的focal loss，数据类型为`float32`。Shape为`[N, C]`。
### 支持的型号
- Atlas A5 训练系列产品
### 调用示例
```python
import torch, torch_npu
from mx_driving import sigmoid_focal_loss
logit = torch.rand(1800, 10, dtype=torch.float32, device='npu') * 10 - 5
target = torch.randint(low=0, high=10, size=(1800,), dtype=torch.int64, device='npu')
weight = torch.rand(1800, dtype=torch.float32, device='npu') * 10 - 5
logit.requires_grad = True
output = sigmoid_focal_loss(logit, target, 2.0, 0.25, weight, 'mean')
output.backward()
```
