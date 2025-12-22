## SparseInverseConv3d
### 接口原型
```python
mx_driving.SparseInverseConv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, mode='mmcv') -> SparseConvTensor
```
兼容
```python
mx_driving.spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, indice_key=None, mode='mmcv') -> SparseConvTensor
```

### 功能描述
稀疏逆卷积
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
- `in_channels、out_channels(int)`整除8
- `kernel_size`当前支持数据类型为三维List/Tuple或Int，值域为`[1, 3]`
- `stride`当前支持数据类型为三维List/Tuple或Int
- `dilation`，`groups`当前仅支持值为1
- `weight`，两种模式下(mmcv & spconv), weight shape 为 [D, H, W, out_channels, in_channels]
- 对于反向也是同样的约束。
### 调用示例
```python
import torch, torch_npu
import numpy as np
from mx_driving import SparseConv3d, SparseConvTensor, SparseInverseConv3d


def generate_sparse_data(shape, num_points, num_channels, batch_size=4):
    dense_shape = shape
    ndim = len(dense_shape)
    batch_indices = []

    coors_total = np.stack(np.meshgrid(*[np.arange(0, s) for s in shape]), axis=-1)
    coors_total = coors_total.reshape(-1, ndim)

    for i in range(batch_size):
        np.random.shuffle(coors_total)
        inds_total = coors_total[:num_points]
        inds_total = np.pad(inds_total, ((0, 0), (0, 1)), mode="constant", constant_values=i)
        batch_indices.append(inds_total)

    sparse_data = np.random.uniform(-1, 1, size=[num_points * batch_size, num_channels]).astype(np.float32)
    batch_indices = np.concatenate(batch_indices, axis=0)

    return {"features": sparse_data, "indices": batch_indices.astype(np.int32)}


spatial_shape = [9, 9, 9]
in_channels, out_channels = 192, 256
kernel_size, stride = 3, 2
batch_size, feature_num = 4, 1529

sparse_dict = generate_sparse_data(spatial_shape, feature_num, in_channels, batch_size)
voxels = torch.from_numpy(sparse_dict["features"]).float()
coors = torch.from_numpy(sparse_dict["indices"][:, [3, 0, 1, 2]]).int()

conv3d = (
    SparseConv3d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, indice_key="test"
    )
    .npu()
    .float()
)

inv_conv3d = (
    SparseInverseConv3d(
        in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, indice_key="test"
    )
    .npu()
    .float()
)

x = SparseConvTensor(voxels.npu(), coors.npu(), spatial_shape, batch_size)
x1 = conv3d(x)
x2 = inv_conv3d(x1)

print("voxels.shape:", voxels.shape)
print("x1.features.shape:", x1.features.shape)
print("x2.features.shape:", x2.features.shape)
```