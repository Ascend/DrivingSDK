## SparseConv3d
### 接口原型
```python
mx_driving.SparseConv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, mode='mmcv') -> SparseConvTensor
```
兼容
```python
mx_driving.spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, mode='mmcv') -> SparseConvTensor
```
### 功能描述
稀疏卷积
### 参数说明
- `in_channels(int)`：输入数据的通道数
- `out_channels(int)`：输出通道数
- `kernel_size(List(int)/Tuple(int)/int)`：卷积神经网络中卷积核的大小
- `stride(List(int)/Tuple(int)/int)`：卷积核在输入数据上滑动时的步长
- `dilation(List(int)/Tuple(int)/int)`：空洞卷积大小
- `groups(int)`：分组卷积
- `bias(bool)`：偏置项
- `indice_key(str)`：该输入用于复用之前计算的索引信息
- `mode(str)`：区分了`mmcv`和`spconv`两种不同框架下的稀疏卷积
### 返回值
- `SparseConvTensor(Tensor)`：存储了输出的特征值`out_feature`，对应索引位置`out_indices`和对应的spatital_shape。
### 支持的型号
- Atlas A2 训练系列产品
### 约束说明
- `kernel_size`当前支持数据类型为三维List/Tuple或Int，值域为`[1, 3]`
- `stride`当前支持数据类型为三维List/Tuple或Int
- `dilation`，`groups`当前仅支持值为1
- 对于反向也是同样的约束。
### 调用示例
```python
import torch,torch_npu
import numpy as np
from mx_driving import SparseConv3d, SparseConvTensor

def generate_indice(batch, height, width, depth, actual_num):
    base_indices = np.random.permutation(np.arange(batch * height * width * depth))[:actual_num]
    base_indices = np.sort(base_indices)
    b_indice = base_indices // (height * width * depth)
    base_indices = base_indices % (height * width * depth)
    h_indice = base_indices // (width * depth)
    base_indices = base_indices % (width * depth)
    w_indice = base_indices // depth
    d_indice = base_indices % depth
    indices = np.concatenate((b_indice, h_indice, w_indice, d_indice)).reshape(4, actual_num)
    return indices

actual_num = 20
batch = 4
spatial_shape = [9, 9, 9]
indices = torch.from_numpy(generate_indice(batch, spatial_shape[0], spatial_shape[1], spatial_shape[2], actual_num)).int().transpose(0, 1).contiguous().npu()
feature = tensor_uniform = torch.rand(actual_num, 16).npu()
feature.requires_grad = True
x = SparseConvTensor(feature, indices, spatial_shape, batch)
net = SparseConv3d(in_channels=16, out_channels=32, kernel_size=3).npu()
out = net(x)
dout = torch.ones_like(out.features).float().npu()
out.features.backward(dout)
```