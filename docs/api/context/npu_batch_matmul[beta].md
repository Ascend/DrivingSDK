## npu_batch_matmul[beta]
### 接口原型
```python
mx_driving.npu_batch_matmul(Tensor projection_mat, Tensor pts_extend) -> Tensor
```

### 功能描述

### 参数说明
- `projection_mat(Tensor)`：投影矩阵，数据类型为`float32`。Shape为4维以上，最后两维需要是`4, 4`或`3, 3`，且和pts_extend互相可广播。
- `pts_extend(Tensor)`：所有点的特征，数据类型为`float32`。Shape为4维以上，最后两维需要是`1, 4`、`1, 3`等，且和projection_mat互相可广播。
### 返回值
- `output(Tensor)`：矩阵乘结果，数据类型为`float32`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
输入是6维
```python
import numpy as np
import torch, torch_npu
import mx_driving
projection_mat =torch.randn((6, 6, 4, 4)).npu()
pts_extend =torch.randn(6, 1220, 13, 4).npu()
projection_mat_fused = projection_mat_fused[:, :, None, None].contiguous()
pts_extend2_fused = pts_extend2_fused[:, None, :, :, None, :].contiguous()
projection_mat_fused.requires_grad=True
pts_extend2_fused.requires_grad=True        
result = mx_driving.npu_batch_matmul(projection_mat_fused, pts_extend2_fused)
grad = torch.ones_like(result)
result.backward(grad)
```
输入是4维
```python
import numpy as np
import torch, torch_npu
import mx_driving
projection_mat =torch.randn((6, 1220, 4, 4)).npu()
pts_extend = torch.randn(6, 1220, 1, 4).npu()
projection_mat.requires_grad=True
pts_extend.requires_grad=True        
result = mx_driving.npu_batch_matmul(projection_mat, pts_extend)
grad = torch.ones_like(result)
result.backward(grad)
```