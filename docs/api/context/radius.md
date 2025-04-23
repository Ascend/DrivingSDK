## radius
### 接口原型
```python
mx_driving.radius(Tensor x,Tensor y,Tensor ptr_x, Tensor ptr_y, 
                  float r, int max_num_neighbors) -> Tensor
```
### 功能描述
给定两组点的二维坐标X和Y，对于Y当中每一个点y，求X当中所有与y在同一个batch内，且距离在半径r之内的点的索引。
### 参数说明
- `X (Tensor)`：第一组点的二维坐标，数据类型为`float32`，shape为`[numpoints_x, 2]`。
- `Y (Tensor)`：第二组点的二维坐标，数据类型为`float32`，shape为`[numpoints_y, 2]`。
- `ptr_x (Tensor)`：第一组点的batch切分地址，数据类型为`int`，shape为`[batch_size + 1]`。ptr_x[0]的值为0，之后的数严格递增，ptr_x[batch_size]的值为numpoints_x。X[ptr_x[0]: ptr_x[1]]属于第2个batch，X[ptr_x[1]: ptr_x[2]]属于第2个batch，之后点的切分以此类推。
- `ptr_y (Tensor)`：第二组点的batch切分地址，数据类型为`int`，shape为`[batch_size + 1]`。ptr_y[0]的值为0，之后的数严格递增，ptr_y[batch_size]的值为numpoints_y。Y[ptr_y[0]: ptr_y[1]]属于第2个batch，Y[ptr_y[1]: ptr_y[2]]属于第2个batch，之后点的切分以此类推。
- `r (float)`：半径，数据类型为`float`。
- `max_num_neighbors (int)`：最大邻居数量，数据类型为`int`。对于任一点y，如果半径r内的x点数量大于max_num_neighbors，则只按索引顺序返回前max_num_neighbors个x点的索引。
### 返回值
- `output_index (Tensor)`：所有符合条件的y-x邻居索引对，数据类型为`int`，shape为`[2, num_neighbors]`。num_neighbors表示所有邻居的总数，只有在算子完成计算之后才能获取它的数值大小。
### 约束说明
- batch_size <= 1024
- ptr_x与ptr_y中相邻两点的间隔，即单个batch内点的数量小于等于512
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch
import torch_npu
import numpy as np
from mx_driving import radius

def gen_points(num_points, data_range):
    points = 2 * data_range * (torch.rand([num_points, 2]) - 0.5)
    return points

def gen_batch_ptr(batch_size, max_points_per_batch):
    batch_list = torch.randint(0, max_points_per_batch, [batch_size]).int()
    batch_ptr = torch.cumsum(batch_list, dim = 0).int()
    batch_ptr = torch.cat([torch.zeros([1]).int(), batch_ptr])
    return batch_ptr

def gen_inputs(data_range, batch_size, max_points_per_batch):
    ptr_x = gen_batch_ptr(batch_size, max_points_per_batch)
    ptr_y = gen_batch_ptr(batch_size, max_points_per_batch)
    num_points_x = ptr_x[-1]
    num_points_y = ptr_y[-1]
    x = gen_points(num_points_x, data_range)
    y = gen_points(num_points_y, data_range)
    return x, y, ptr_x, ptr_y

data_range = 50 # X和Y的取值在[-50, 50]范围内
batch_size = 16
max_points_per_batch = 512 # ptr_x和ptr_y相邻点间隔不大于512
r = 20.0
max_num_neighbors = 100

x, y, ptr_x, ptr_y = gen_inputs(data_range, batch_size, max_points_per_batch)
out_npu = mx_driving.radius(x.npu(), y.npu(), ptr_x.npu(), ptr_y.npu(), r, max_num_neighbors)
```