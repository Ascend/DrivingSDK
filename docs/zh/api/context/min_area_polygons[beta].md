## min_area_polygons[beta]
### 接口原型
```python
mx_driving.min_area_polygons(Tensor pointsets) -> Tensor
```
### 功能描述
计算输入点集的最小外接矩形，输出顶点坐标。
### 参数说明
- `pointsets(torch.Tensor)`：输入的点集，数据类型为`float32`，shape 为 `[N, 18]`。
### 返回值
- `polygons(torch.Tensor)`：最小外接矩形的顶点坐标，shape 为 `[N, 8]`。
### 算子约束
- $\mathrm{1 \le N \le 2048}$
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```
import torch
import mx_driving
pointsets = torch.tensor([[1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.5, 1.5],
                        [1.0, 1.0, 8.0, 8.0, 1.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.5, 1.5]], dtype=torch.float32).npu()
polygons = mx_driving.min_area_polygons(pointsets)