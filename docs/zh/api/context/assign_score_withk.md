## assign_score_withk
### 接口原型
```python
mx_driving.assign_score_withk(Tensor scores, Tensor point_features, Tensor center_features, Tensor knn_idx, str aggregate='sum') -> Tensor
```
兼容：
```python
mx_driving.common.assign_score_withk(Tensor scores, Tensor point_features, Tensor center_features, Tensor knn_idx, str aggregate='sum') -> Tensor
```
### 功能描述
根据`knn_idx`得到采样点及其邻居点的索引，计算`point_features`和`center_features`的差，并与`scores`相乘后在特征维度进行聚合，返回采样点的特征。
### 参数说明
- `scores (Tensor)`：权重矩阵的重要系数，数据类型为`float32`。Shape为`[B, npoint, K, M]`，其中`B`为batch size，`npoint`为采样点的数量，`K`为一个样本点及其邻居点的数量之和，`M`为权重矩阵集合的规模。
- `point_features (Tensor)`：所有点的特征，数据类型为`float32`。Shape为`[B, N, M, O]`，其中`N`为所有点的数量，`O`为特征数量。
- `center_features (Tensor)`：所有点的中心特征，数据类型为`float32`。Shape为`[B, N, M, O]`。
- `knn_idx (Tensor)`：采样点及其邻居点的索引，数据类型为`int64`。Shape为`[B, npoint, K]`。
- `aggregate (str)`：聚合方式，默认为`sum`，数据类型为`str`。
### 返回值
- `output (Tensor)`：聚合后采样点的特征，数据类型为`float32`。Shape为`[B, O, npoint, K]`。
### 算子约束
- `npoint`和`K`都不大于`N`。
- `aggregate`参数取值当前仅支持`sum`
- M * K <= 5000
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import numpy as np
import torch, torch_npu
from mx_driving import assign_score_withk
points = torch.from_numpy(np.random.rand(4, 100, 8, 16).astype(np.float32)).npu()
centers = torch.from_numpy(np.random.rand(4, 100, 8, 16).astype(np.float32)).npu()
scores = torch.from_numpy(np.random.rand(4, 64, 10, 8).astype(np.float32)).npu()
knn_idx = torch.from_numpy(np.array([[np.random.choice(100, size=10, replace=False) 
                    for _ in range(64)] for _ in range(4)]).astype(np.int64)).npu()
grad_out = torch.from_numpy(np.random.rand(4, 16, 64, 10).astype(np.float32)).npu()
points.requires_grad = True
centers.requires_grad = True
scores.requires_grad = True
output = assign_score_withk(scores, points, centers, knn_idx, "sum")
output.backward(grad_out)
```
