# Common 算子
## scatter_max
### 接口原型
```python
ads.common.scatter_max(Tensor updates, Tensor indices, Tensor out=None) -> (Tensor out, Tensor argmax)
```
### 功能描述
在`第0`维上，将输入张量`updates`中的元素按照`indices`中的索引进行分散，然后在第0维上取最大值，返回最大值和对应的索引。对于1维张量，公式如下：
$$out_i = max(out_i, max_j(updates_j))$$
$$argmax_i = argmax_j(updates_j)$$
这里，$i = indices_j$。
### 参数说明
- `updates`：更新源张量，数据类型为`float32`。
- `indices`：索引张量，数据类型为`int32`，且
  - `indices`的维度必须为`1`，
  - `indices`第0维的长度必须与`updates`第0维的长度相同。
  - `indices`的最大值必须小于`491520`。
- `out`：被更新张量，数据类型为`float32`，默认为`None`。
### 返回值
- `out`：更新后的张量，数据类型为`float32`。
- `argmax`：最大值对应的索引张量，数据类型为`int32`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import scatter_max
updates = torch.tensor([[2, 0, 1, 4, 4], [0, 2, 1, 3, 4]], dtype=torch.float32).npu()
indices = torch.tensor([4, 1, 2, 3], dtype=torch.int32).npu()
out = updates.new_zeros((2, 6))
out, argmax = npu_scatter_max(updates, indices, out)
print(out)
print(argmax)
```
```text
tensor([[0., 0., 0., 0., 0., 0.],
        [0., 0., 3., 4., 0., 0.]])
tensor([[2, 2,  2,  2,  2,  2],
        [ 1,  1,  1, 1, 0, 0]])
```
 ## npu_rotated_box_decode
### 接口原型
```python
ads.common.npu_rotated_box_decode(Tensor anchor_boxes, Tensor deltas, Tensor weight) -> Tensor
```
### 功能描述
解码旋转框的坐标。
### 参数说明
- `anchor_box(Tensor)`：锚框张量，数据类型为`float32, float16`，形状为`[B, 5, N]`，其中`B`为批大小，`N`为锚框个数, 值`5`分别代表`x0, x1, y0, y1, angle`。
- `deltas(Tensor)`：偏移量张量，数据类型为`float32, float16`，形状为`[B, 5, N]`，其中`B`为批大小，`N`为锚框个数, 值`5`分别代表`dx, dy, dw, dh, dangle`。
- `weight(Tensor)`：权重张量，数据类型为`float32, float16`，形状为`[5]`，其中`5`分别代表`wx, wy, ww, wh, wangle`。默认值为`[1, 1, 1, 1, 1]`。
### 返回值
- `Tensor`：解码后的旋转框坐标张量，数据类型为`float32, float16`，形状为`[B, 5, N]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_rotated_box_decode
anchor_boxes = torch.tensor([[[4.137], [33.72], [29.4], [54.06], [41.28]]], dtype=torch.float16).npu()
deltas = torch.tensor([[[0.0244], [-1.992], [0.2109], [0.315], [-37.25]]], dtype=torch.float16).npu()
wegiht = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float16).npu()
out = npu_rotated_box_decode(anchor_boxes, deltas, weight)
print(out)
```
```text
tensor([[[1.7861], [-10.5781], [33.0000], [17.2969], [-88.4375]]], dtype=torch.float16)
```
## npu_rotated_box_encode
### 接口原型
```python
ads.common.npu_rotated_box_encode(Tensor anchor_boxes, Tensor gt_bboxes, Tensor weight) -> Tensor
```
### 功能描述
编码旋转框的坐标。
### 参数说明
- `anchor_box(Tensor)`：锚框张量，数据类型为`float32, float16`，形状为`[B, 5, N]`，其中`B`为批大小，`N`为锚框个数, 值`5`分别代表`x0, x1, y0, y1, angle`。
- `gt_bboxes(Tensor)`：真实框张量，数据类型为`float32, float16`，形状为`[B, 5, N]`，其中`B`为批大小，`N`为锚框个数, 值`5`分别代表`x0, x1, y0, y1, angle`。
- `weight(Tensor)`：权重张量，数据类型为`float32, float16`，形状为`[5]`，其中`5`分别代表`wx, wy, ww, wh, wangle`。默认值为`[1, 1, 1, 1, 1]`。
### 返回值
- `Tensor`：编码后的旋转框坐标张量，数据类型为`float32, float16`，形状为`[B, 5, N]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_rotated_box_encode
anchor_boxes = torch.tensor([[[30.69], [32.6], [45.94], [59.88], [-44.53]]], dtype=torch.float16).npu()
gt_bboxes = torch.tensor([[[30.44], [18.72], [33.22], [45.56], [8.5]]], dtype=torch.float16).npu()
weight = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float16).npu()
out = npu_rotated_box_encode(anchor_boxes, gt_bboxes, weight)
print(out)
```
```text
tensor([[[-0.4253], [-0.5166], [-1.7021], [-0.0162], [1.1328]]], dtype=torch.float16)
```
## npu_rotated_iou
### 接口原型
```python
ads.common.npu_rotated_iou(Tensor self, Tensor query_boxes, bool trans=False, int mode=0, bool is_cross=True, float v_threshold=0.0, float e_threshold=0.0) -> Tensor
```
### 功能描述
计算旋转框的IoU。
### 参数说明
- `self(Tensor)`：梯度增量，数据类型为`float32, float16`，形状为`[B, 5, N]`。
- `query_boxes(Tensor)`：查询框张量，数据类型为`float32, float16`，形状为`[B, 5, M]`。
- `trans(bool)`：是否进行坐标变换。默认值为`False`。值为`True`时，表示`xyxyt`, 值为`False`时，表示`xywht`。
- `is_cross(bool)`：是否计算交叉面积。默认值为`True`。值为`True`时，表示计算交叉面积，值为`False`时，表示计算并集面积。
- `mode(int)`：计算IoU的模式。默认值为`0`。值为`0`时，表示计算`IoU`，值为`1`时，表示计算`IoF`。
- `v_threshold(float)`：垂直方向的阈值。默认值为`0.0`。
- `e_threshold(float)`：水平方向的阈值。默认值为`0.0`。
### 返回值
- `Tensor`：IoU张量，数据类型为`float32, float16`，形状为`[B, N, M]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
import numpy as np
from ads.common import npu_rotated_iou
a = np.random.uniform(0, 1, (2, 2, 5)).astype(np.float16)
b = np.random.uniform(0, 1, (2, 3, 5)).astype(np.float16)
box1 = torch.from_numpy(a).npu()
box2 = torch.from_numpy(b).npu()
iou = npu_rotated_iou(box1, box2, trans=False, mode=0, is_cross=True, v_threshold=0.0, e_threshold=0.0)
print(iou)
```
```text
tensor([[[3.3325e-01, 1.0162e-01],
         [1.0162e-01, 1.0000e+00]],

        [[0.0000e+00, 0.0000e+00],
         [0.0000e+00, 5.9605e-08]]], dtype=torch.float16)
```
## npu_rotated_overlaps
### 接口原型
```python
ads.common.npu_rotated_overlaps(Tensor self, Tensor query_boxes, bool trans=False) -> Tensor
```
### 功能描述
计算旋转框的重叠面积。
### 参数说明
- `self(Tensor)`：梯度增量，数据类型为`float32, float16`，形状为`[B, 5, N]`。
- `query_boxes(Tensor)`：查询框张量，数据类型为`float32, float16`，形状为`[B, 5, M]`。
- `trans(bool)`：是否进行坐标变换。默认值为`False`。值为`True`时，表示`xyxyt`, 值为`False`时，表示`xywht`。
### 返回值
- `Tensor`：重叠面积张量，数据类型为`float32, float16`，形状为`[B, N, M]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
import numpy as np
from ads.common import npu_rotated_overlaps
a = np.random.uniform(0, 1, (1, 3, 5)).astype(np.float16)
b = np.random.uniform(0, 1, (1, 2, 5)).astype(np.float16)
box1 = torch.from_numpy(a).npu()
box2 = torch.from_numpy(b).npu()
output = npu_rotated_overlaps(box1, box2)
print(output)
```
```text
tensor([[[0.0000, 0.1562, 0.0000],
         [0.1562, 0.3713, 0.0611],
         [0.0000, 0.0611, 0.0000]]], dtype=torch.float16)
```
## npu_sign_bits_pack
### 接口原型
```python
ads.common.npu_sign_bits_pack(Tensor self, int size) -> Tensor
```
### 功能描述
将输入张量的数据按位打包为uint8类型。
### 参数说明
- `self(Tensor)`：1D输入张量，数据类型为`float32, float16`。
- `size(int)`：reshape 时输出张量的第一个维度。
### 返回值
- `Tensor`：打包后的张量，数据类型为`uint8`。
### 约束说明
Size为可被float打包的输出整除的整数。如果self的size可被8整除，则size为self.size/8，否则size为self.size/8+1。将在小端位置添加-1浮点值以填充可整除性。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_sign_bits_pack
a = torch.tensor([5, 4, 3, 2, 0, -1, -2, 4, 3, 2, 1, 0, -1, -2], dtype=torch.float32).npu()
out = npu_sign_bits_pack(a, 2)
print(out)
```
```text
tensor([[159], [15]], dtype=torch.uint8)
```
## npu_sign_bits_unpack
### 接口原型
```python
ads.common.npu_sign_bits_unpack(Tensor x, int dtype, int size) -> Tensor
```
### 功能描述
将输入张量的数据按位解包为float类型。
### 参数说明
- `x(Tensor)`：1D输入张量，数据类型为`uint8`。
- `dtype(torch.dtype)`：输出张量的数据类型。值为1时，表示`float32`，值为0时，表示`float16`。
- `size(int)`：reshape 时输出张量的第一个维度。
### 返回值
- `Tensor`：解包后的张量，数据类型为`float32, float16`。
### 约束说明
Size为可被uint8s解包的输出整数。输出大小为(size of x)*8。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_sign_bits_unpack
a = torch.tensor([159, 15], dtype=torch.uint8).npu()
out = npu_sign_bits_unpack(a, 0, 2)
print(out)
```
```text
tensor([[1., 1., 1., 1., 1., -1., -1., 1.], [1., 1., 1., 1., -1., -1., -1., -1.]], dtype=torch.float16)
```
## npu_softmax_cross_entropy_with_logits
### 接口原型
```python
ads.common.npu_softmax_cross_entropy_with_logits(Tensor features, Tensor labels) -> Tensor
```
### 功能描述
计算softmax交叉熵。
### 参数说明
- `features(Tensor)`：输入张量，数据类型为`float32, float16`。shape为`[B, N]`, 其中`B`为批大小，`N`为类别数。
- `labels(Tensor)`：标签张量, 与`features`的shape相同。
### 返回值
- `Tensor`：交叉熵张量，数据类型为`float32, float16`，shape为`[B]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_softmax_cross_entropy_with_logits
features = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32).npu()
labels = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float32).npu()
out = npu_softmax_cross_entropy_with_logits(features, labels)
print(out)
```
```text
tensor([1.4076, 2.4076], dtype=torch.float32)
```
## npu_stride_add
### 接口原型
```python
ads.common.npu_stride_add(Tensor x1, Tensor x2, int offset1, int offset2, int c1_len) -> Tensor
```
### 功能描述
将两个张量按照指定的偏移量进行相加, 格式为`NC1HWC0`。
### 参数说明
- `x1(Tensor)`：输入张量，`5HD`格式，数据类型为`float32, float16`。
- `x2(Tensor)`：输入张量，与`x1`的shape相同，数据类型为`float32, float16`。
- `offset1(int)`：`x1`的偏移量。
- `offset2(int)`：`x2`的偏移量。
- `c1_len(int)`：输出张量的`C1`维度。该值必须小于`x1`和`x2`中`C1`与`offset`的差值。
### 返回值
- `Tensor`：相加后的张量，数据类型为`float32, float16`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_stride_add
x1 = torch.tensor([[[[[1]]]]], dtype=torch.float32).npu()
out = npu_stride_add(x1, x1, 0, 0, 1)
print(out)
```
```text
tensor([[[[[2]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]], [[[0]]]]], dtype=torch.float32)
```
## npu_transpose
### 接口原型
```python
ads.common.npu_transpose(Tensor x, List[int] perm, bool require_contiguous=True) -> Tensor
```
### 功能描述
将输入张量的维度按照指定的顺序进行转置。支持`FakeTensor`模式。
### 参数说明
- `x(Tensor)`：输入张量，数据类型为`float32, float16`。
- `perm(List[int])`：转置顺序。
- `require_contiguous(bool)`：是否要求输出张量是连续的。默认值为`True`。
### 返回值
- `Tensor`：转置后的张量，数据类型为`float32, float16`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_transpose
x = torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32).npu()
y = npu_transpose(x, [0, 2, 1])
print(y)
```
```text
tensor([[[1., 4.], [2., 5.], [3., 6.]]], dtype=torch.float
```
## npu_yolo_boxes_encode
### 接口原型
```python
ads.common.npu_yolo_boxes_encode(Tensor anchors, Tensor gt_bboxes, Tensor stride, bool perfermance_mode=False) -> Tensor
```
### 功能描述
根据YOLO的锚点框(anchor)和真实框(gt_bboxes)生成编码后的框。
### 参数说明
- `anchors(Tensor)`：锚点框张量，数据类型为`float32, float16`，形状为`[N, 4]`，其中`N`为`ROI`的个数，`4`分别代表`tx, ty, tw, th`。
- `gt_bboxes(Tensor)`：真实框张量，数据类型为`float32, float16`，形状为`[N, 4]`，其中`N`为`ROI`的个数，`4`分别代表`dx, dy, dw, dh`。
- `stride(Tensor)`：步长张量，数据类型为`int32`，形状为`[N]`，其中`N`为`ROI`的个数。
- `perfermance_mode(bool)`：是否为性能模式。默认值为`False`。当值为`True`时，表示为性能模式，输入类型为`float16`时，将是最新的性能模式，但精度只小于`0.005`；当值为`False`时，表示为精度模式，输入类型为`float32`是,输出精度小于`0.0001`。
### 返回值
- `Tensor`：编码后的框张量，数据类型为`float32, float16`，形状为`[N, 4]`。
### 约束说明
- `anchors`和`gt_bboxes`的`N`必须相同，且`N`的值必须小于`20480`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_yolo_boxes_encode
anchors = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32).npu()
gt_bboxes = torch.tensor([[5, 6, 7, 8], [1, 2, 3, 4]], dtype=torch.float32).npu()
stride = torch.tensor([1, 2], dtype=torch.int32).npu()
out = npu_yolo_boxes_encode(anchors, gt_bboxes, stride)
print(out)
```
```text
tensor([[ 1.0000,  1.0000,  0.0000,  0.0000],
        [1.0133e-06, 1.0133e-06,  0.0000,  0.0000]], dtype=torch.float32)
```
## npu_scatter
### 接口原型
```python
ads.common.npu_scatter(Tensor self, Tensor indices, Tensor updates, int dim) -> Tensor
```
### 功能描述
将`updates`张量中的元素按照`indices`张量中的索引进行分散，然后将分散的元素加到`self`张量中。
### 参数说明
- `self(Tensor)`：被更新张量，数据类型为`float32, float16`。
- `indices(Tensor)`：索引张量，数据类型为`int32`。可以为空，也可以与`updates`有相同的维数。当为空时，操作返回`self unchanged`。
- `updates(Tensor)`：更新源张量，数据类型为`float32, float16`。
- `dim(int)`：分散的维度。
### 返回值
- `Tensor`：更新后的张量，数据类型为`float32, float16`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_scatter
input = torch.tensor([[1.6279, 0.1226], [0.9041, 1.0980]], dtype=torch.float32).npu()
indices = torch.tensor([0, 1], dtype=torch.int32).npu()
updates = torch.tensor([-1.1993, -1.5247], dtype=torch.float32).npu()
out = npu_scatter(input, indices, updates, 0)
print(out)
```
```text
tensor([[-0.1993, 0.1226], [ 0.9041, -1.5247]], dtype=torch.float32)
```
## npu_silu
### 接口原型
```python
ads.common.npu_silu(Tensor x) -> Tensor
```
### 功能描述
计算Sigmoid Linear Unit(SiLU)激活函数。公式如下：
$$f(x) = x * sigmoid(x)$$
### 参数说明
- `x(Tensor)`：输入张量，数据类型为`float32, float16`。
### 返回值
- `Tensor`：激活后的张量，数据类型为`float32, float16`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_silu
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32).npu()
out = npu_silu(x)
print(out)
```
```text
tensor([0.7311, 1.7646, 2.8577, 3.9281], dtype=torch.float32)
```
> 注意：可以通过`npu_silu_`接口实现原地操作。
## npu_rotary_mul
### 接口原型
```python
ads.common.npu_rotary_mul(Tensor x, Tensor r1, Tensor r2) -> Tensor
```
### 功能描述
计算旋转乘法。公式如下：
$$x1, x2 = x[..., :C//2], x[..., C//2:]$$
$$x_new = [-x2, x1]$$
$$y = x * r1 + x_new * r2$$
### 参数说明
- `x(Tensor)`：输入张量，数据类型为`float32, float16`。要求`x`的维度为`4`。
- `r1(Tensor)`：旋转因子张量，数据类型为`float32, float16`。代表`cos`。
- `r2(Tensor)`：旋转因子张量，数据类型为`float32, float16`。代表`sin`。
### 返回值
- `Tensor`：旋转乘法后的张量，数据类型为`float32, float16`。
### 约束说明
- `x`的维度必须为`4`， 一般为`[B, N, S, D]`或`[B, S, N, D]`或`[S, B, N, D]`。
- `r1`和`r2`的维度必须为`4`， 一般为`[1, 1, S, D]`或`[S, 1, 1, D]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_rotary_mul
x = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32).npu()
r1 = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]]], dtype=torch.float32).npu()
r2 = torch.tensor([[[[0.2, 0.3], [0.4, 0.5]], [[0.6, 0.7], [0.8, 0.9]]]], dtype=torch.float32).npu()
out = npu_rotary_mul(x, r1, r2)
print(out)
```
```text
tensor([[[[-0.3000, 0.7000], [-0.7000, 3.1000]], [[-1.1000, 7.1000], [-1.5000, 12.7000]]]], dtype=torch.float32)
```
## npu_abs
### 接口原型
```python
ads.common.npu_abs(Tensor x) -> Tensor
```
### 功能描述
计算输入张量的绝对值。
### 参数说明
- `x(Tensor)`：输入张量，数据类型为`float32, float16`。
### 返回值
- `Tensor`：绝对值张量，数据类型为`float32, float16`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_abs
x = torch.tensor([1, -2, 3, -4], dtype=torch.float32).npu()
out = npu_abs(x)
print(out)
```
```text
tensor([1., 2., 3., 4.], dtype=torch.float32)
```
## fast_gelu
### 接口原型
```python
ads.common.fast_gelu(Tensor x) -> Tensor
```
### 功能描述
计算输入张量的GELU激活函数。公式如下：
$$f(x) = x/(1+exp(-1.702 * |x|))*exp(0.851*(x-|x|))$$
### 参数说明
- `x(Tensor)`：输入张量，数据类型为`float32, float16`。
### 返回值
- `Tensor`：激活后的张量，数据类型为`float32, float16`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
import numpy as np
from ads.common import fast_gelu
x = torch.from_numpy(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]])).float().npu()
output = fast_gelu(x)
print(output)
```
```text
tensor([[-1.5418735e-01  3.9921875e+00 -9.7473649e-06],  [ 1.9375000e+00 -1.0052517e-03  8.9824219e+00]], dtype=torch.float32)
```
## npu_anchor_response_flags
### 接口原型
```python
ads.common.npu_anchor_response_flags(Tensor gt_bboxes, List[int] featmap_size, List[int] strides, int num_base_anchors) -> Tensor
```
### 功能描述
根据真实框(gt_bboxes)和特征图大小(featmap_size)生成锚点响应标志。
### 参数说明
- `gt_bboxes(Tensor)`：真实框张量，数据类型为`float32, float16`，形状为`[N, 4]`，其中`N`为`ROI`的个数，`4`分别代表`x0, y0, x1, y1`。
- `featmap_size(List[int])`：特征图大小，形状为`[2]`，其中`2`分别代表`H, W`。
- `strides(List[int])`：步长，形状为`[2]`，其中`2`分别代表`stride_h, stride_w`。
- `num_base_anchors(int)`：基础锚点数。
### 返回值
- `Tensor`：锚点响应标志张量，数据类型为`uint8`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_anchor_response_flags
gt_bboxes = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32).npu()
featmap_size = [2, 3]
strides = [1, 2]
num_base_anchors = 2
out = npu_anchor_response_flags(gt_bboxes, featmap_size, strides, num_base_anchors)
print(out)
```
```text
tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.uint8)
```
## npu_bounding_box_decode
### 接口原型
```python
ads.common.npu_bounding_box_decode(Tensor rois, Tensor deltas, float means0, float means1, float means2, float means3, float stds0, float stds1, float stds2, float stds3, int max_shape, float wh_ratio_clip) -> Tensor
```
### 功能描述
根据`rois`和`deltas`生成解码后的框。
### 参数说明
- `rois(Tensor)`：区域候选网络（RPN）生成的ROI，数据类型为`float32, float16`，形状为`[N, 4]`，其中`N`为`ROI`的个数，`4`分别代表`x0, y0, x1, y1`。
- `deltas(Tensor)`：偏移量张量，数据类型为`float32, float16`，形状为`[N, 4]`，其中`N`为`ROI`的个数，`4`分别代表`dx, dy, dw, dh`。
- `means0(float)`：均值，用于归一化`dx`。
- `means1(float)`：均值，用于归一化`dy`。
- `means2(float)`：均值，用于归一化`dw`。
- `means3(float)`：均值，用于归一化`dh`。
- `stds0(float)`：标准差，用于归一化`dx`。
- `stds1(float)`：标准差，用于归一化`dy`。
- `stds2(float)`：标准差，用于归一化`dw`。
- `stds3(float)`：标准差，用于归一化`dh`。
  - 以上参数均为`float32`类型，`meas`默认值为`0`, `std`默认值为`1`。`delta`的归一化公式为：`delta = (delta - means) / stds`。
- `max_shape(int)`：最大形状。用于确保转换后的bbox不超过最大形状。默认值为`0`。
- `wh_ratio_clip(float)`：宽高比裁剪。`dw`和`dh`的值在`(-wh_ratio_clip, wh_ratio_clip)`之间。
### 返回值
- `Tensor`：解码后的框张量，数据类型为`float32, float16`，形状为`[N, 4]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_bounding_box_decode
rois = torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6]], dtype=torch.float32).npu()
deltas = torch.tensor([[5, 6, 7, 8], [7, 8, 9, 6]], dtype=torch.float32).npu()
out = npu_bounding_box_decode(rois, deltas, 0, 0, 0, 0, 1, 1, 1, 1, (10, 10), 0.1)
print(out)
```
```text
tensor([[ 2.5000,  6.5000,  9.0000,  9.0000], [ 9.0000,  9.0000,  9.0000,  9.0000]], dtype=torch.float32)
```
## npu_bounding_box_encode
### 接口原型
```python
ads.common.npu_bounding_box_encode(Tensor anchor_boxes, Tensor gt_bboxes, float means0, float means1, float means2, float means3, float stds0, float stds1, float stds2, float stds3) -> Tensor
```
### 功能描述
根据`anchor_boxes`和`gt_bboxes`生成编码后的框。
### 参数说明
- `anchor_boxes(Tensor)`：锚框张量，数据类型为`float32, float16`，形状为`[N, 4]`，其中`N`为`ROI`的个数，`4`分别代表`x0, y0, x1, y1`。
- `gt_bboxes(Tensor)`：真实框张量，数据类型为`float32, float16`，形状为`[N, 4]`，其中`N`为`ROI`的个数，`4`分别代表`x0, y0, x1, y1`。
- `means0(float)`：均值，用于归一化`dx`。
- `means1(float)`：均值，用于归一化`dy`。
- `means2(float)`：均值，用于归一化`dw`。
- `means3(float)`：均值，用于归一化`dh`。
- `stds0(float)`：标准差，用于归一化`dx`。
- `stds1(float)`：标准差，用于归一化`dy`。
- `stds2(float)`：标准差，用于归一化`dw`。
- `stds3(float)`：标准差，用于归一化`dh`。
  - 以上参数均为`float32`类型，`meas`默认值为`0`, `std`默认值为`1`。`delta`的归一化公式为：`delta = (delta - means) / stds`。
### 返回值
- `Tensor`：编码后的框张量，数据类型为`float32, float16`，形状为`[N, 4]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_bounding_box_encode
anchor_boxes = torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6]], dtype=torch.float32).npu()
gt_bboxes = torch.tensor([[5, 6, 7, 8], [7, 8, 9, 6]], dtype=torch.float32).npu()
out = npu_bounding_box_encode(anchor_boxes, gt_bboxes, 0, 0, 0, 0, 0.1, 0.1, 0.2, 0.2)
print(out)
```
```text
tensor([[13.3281, 13.3281,  0.0000,  0.0000], [ 13.3281,  6.6641,  0.0000,  -5.4922]], dtype=torch.float32)
```
## npu_batch_nms
### 接口原型
```python
ads.common.npu_batch_nms(Tensor self, Tensor scores, float score_threshold, float iou_threshold, int max_size_per_class, int max_total_size, bool change_coordinate_frame=False, bool transpose_box=False) -> (Tensor, Tensor, Tensor, Tensor)
```
### 功能描述
根据`batch` 分类计算输入框评分，通过评分排序，删除评分高于阈值的框。通过NMS操作，删除重叠度高于阈值的框。
### 参数说明
- `self(Tensor)`：输入张量，数据类型为`float16`，形状为`[B, N, q, 1]`，其中`B`为批大小，`N`为框的个数，`q=1`或`q=num_classes`。
- `scores(Tensor)`：评分张量，数据类型为`float16`，形状为`[B, N, num_classes]`。
- `score_threshold(float)`：评分阈值，用于过滤评分低于阈值的框。
- `iou_threshold(float)`：IoU阈值，用于过滤重叠度高于阈值的框。
- `max_size_per_class(int)`：每个类别的最大框数。
- `max_total_size(int)`：总的最大框数。
- `change_coordinate_frame(bool)`：是否正则化输出框坐标矩阵。默认值为`False`。
- `transpose_box(bool)`：是否转置输出框坐标矩阵。默认值为`False`。
### 返回值
- nmsed_boxes(Tensor)：NMS后的框张量，数据类型为`float16`，形状为`[B, max_total_size, 4]`。
- nmsed_scores(Tensor)：NMS后的评分张量，数据类型为`float16`，形状为`[B, max_total_size]`。
- nmsed_classes(Tensor)：NMS后的类别张量，数据类型为`float16`，形状为`[B, max_total_size]`。
- nmsed_num(Tensor)：NMS后的框数张量，数据类型为`int32`，形状为`[B]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_batch_nms
self = torch.tensor([[[[1, 2, 3, 4]]]], dtype=torch.float16).npu()
scores = torch.tensor([[[1, 2, 3]]], dtype=torch.float16).npu()
nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = npu_batch_nms(self, scores, 0.5, 0.5, 1, 1)
print(nmsed_boxes)
print(nmsed_scores)
print(nmsed_classes)
print(nmsed_num)
```
```text
tensor([[[1.0000, 2.0000, 3.0000, 4.0000]]], dtype=torch.float16)
tensor([[3.]], dtype=torch.float16)
tensor([[2.]], dtype=torch.float16)
tensor([1], dtype=torch.int32)
```
## npu_confusion_transpose
### 接口原型
```python
ads.common.npu_confusion_transpose(Tensor self, List[int] perm, List[int] shape, bool transpose_first) -> Tensor
```
### 功能描述
根据`perm`和`shape`对输入张量进行转置。
### 参数说明
- `self(Tensor)`：输入张量，数据类型为`float32, float16, int8, int16, int32, int64, uint8, uint16, uint32, uint64`。
- `perm(List[int])`：转置顺序。
- `shape(List[int])`：输入张量的形状。
- `transpose_first(bool)`：是否先转置。默认值为`False`。
### 返回值
- `Tensor`：转置后的张量，数据类型为`float32, float16, int8, int16, int32, int64, uint8, uint16, uint32, uint64`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_confusion_transpose
x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32).npu()
out = npu_confusion_transpose(x, [0, 2, 1], [2, 2, 2], False)
print(out)
```
```text
tensor([[[1., 3.], [2., 4.]], [[5., 7.], [6., 8.]]], dtype=torch.float32)
```
## npu_broadcast
### 接口原型
```python
ads.common.npu_broadcast(Tensor self, List[int] size) -> Tensor
```
### 功能描述
根据`size`对输入张量进行广播。
### 参数说明
- `self(Tensor)`：输入张量，数据类型为`float32, float16, int8, int16, int32, int64, uint8, uint16, uint32, uint64`。
- `size(List[int])`：广播后的形状。
### 返回值
- `Tensor`：广播后的张量，数据类型为`float32, float16, int8, int16, int32, int64, uint8, uint16, uint32, uint64`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_broadcast
x = torch.tensor([[1], [2], [3]], dtype=torch.float32).npu()
out = npu_broadcast(x, [3, 4])
print(out)
```
```text
tensor([[1., 1., 1., 1.], [2., 2., 2., 2.], [3., 3., 3., 3.]], dtype=torch.float32)
```
## npu_moe_tutel
### 接口原型
```python
ads.common.npu_moe_tutel(Tensor x, Tensor gates, Tensor indices, Tensor locations, int capacity)
```
### 功能描述
Expert parallelism 把专家分配到不同的计算资源上，比如，一个专家分配1-N个NPU。
### 参数说明
- `x(Tensor)`：MHA层输出的全量token，数据类型为`float32, float16, bf16`。
- `gates(Tensor)`：门控函数的输出结果，数据类型为`float32, float16, bf16`。
- `indices(Tensor)`：batch值对应的索引，数据类型为`int32`。
- `locations(Tensor)`：capacity值对应的索引，数据类型为`int32`。
### 返回值
- `y(Tensor)`: 专家输出的结果，数据类型为`float32, float16, bf16`。shape 为`[B, capacity, x[1]]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_moe_tutel
x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32).npu()
gates = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32).npu()
indices = torch.tensor([1, 2], dtype=torch.int32).npu()
locations = torch.tensor([1, 2], dtype=torch.int32).npu()
out = npu_moe_tutel(x, gates, indices, locations, 2)
print(out)
```
## npu_dynamic_scatter
### 接口原型
```python
ads.common.npu_dynamic_scatter(Tensor feats, Tensor coors_map, string reduce_type) -> Tensor
```
### 功能描述
将特征点在对应体素中进行特征压缩。
### 参数说明
- `feats(Tensor)`：特征张量，数据类型为`float32, float16`。
- `coors_map(Tensor)`：体素坐标映射张量，数据类型为`int32`。
- `reduce_type(string)`：压缩类型。可选值为`0, 1, 2`。当值为`0`时，表示`sum`；当值为`1`时，表示`mean`；当值为`2`时，表示`max
### 返回值
- `Tensor`：压缩后的特征张量，数据类型为`float32, float16`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_dynamic_scatter
feats = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=torch.float32).npu()
coors_map = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.int32).npu()
output_feats, output_coors = ads.common.npu_dynamic_scatter(feats, coors, 1)
print(out)
```
## npu_points_in_box
### 接口原型
```python
ads.common.npu_points_in_box(Tensor boxes, Tensor points) -> Tensor
```
### 功能描述
判断点是否在框内。
### 参数说明
- `boxes(Tensor)`：框张量，数据类型为`float32, float16`。shape 为`[B, M, 7]`。`7`分别代表`x, y, z, x_size, y_size, z_size, rz`。
- `points(Tensor)`：点张量，数据类型为`float32, float16`。shape 为`[B, N, 3]`。`3`分别代表`x, y, z`。
### 返回值
- `boxes_idx_of_points(Tensor)`：点在框内的索引张量，数据类型为`int32`。shape 为`[B, N]`。
### 约束说明
- `boxes`和`points`的`B`必须相同，且只能为`1`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_points_in_box
boxes = torch.tensor([[[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9]]], dtype=torch.float32).npu()
points = torch.tensor([[[1, 2, 3], [3, 4, 5]]], dtype=torch.float32).npu()
out = npu_points_in_box(boxes, points)
print(out)
```
```text
tensor([[0, 1]], dtype=torch.int32)
```
## npu_ads_add
### 接口原型
```python
ads.common.npu_ads_add(Tensor x, Tensor y) -> Tensor
```
### 功能描述
计算两个张量的和。
### 参数说明
- `x(Tensor)`：输入张量，数据类型为`float32, float16`。
- `y(Tensor)`：输入张量，数据类型为`float32, float16`。
### 返回值
- `Tensor`：和张量，数据类型为`float32, float16`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_ads_add
x = torch.tensor([1, 2, 3, 4], dtype=torch.float32).npu()
y = torch.tensor([5, 6, 7, 8], dtype=torch.float32).npu()
out = npu_ads_add(x, y)
print(out)
```
```text
tensor([6., 8., 10., 12.], dtype=torch.float32)
```
## npu_multi_scale_deformable_attn_function
### 接口原型
```python
ads.common.npu_multi_scale_deformable_attn_function(Tensor value, Tensor shape, Tensor offset, Tensor locations, Tensor weight) -> Tensor
```
### 功能描述
多尺度可变形注意力机制, 将多个视角的特征图进行融合。
### 参数说明
- `value(Tensor)`：特征张量，数据类型为`float32, float16`。shape为`[bs, num_keys, num_heads, embed_dim]`。其中`bs`为batch size，`num_keys`为特征图的数量，`num_heads`为头的数量，`embed_dim`为特征图的维度。
- `shape(Tensor)`：特征图的形状，数据类型为`int32`。shape为`[num_levels, 2]`。其中`num_levels`为特征图的数量，`2`分别代表`H, W`。
- `offset(Tensor)`：偏移量张量，数据类型为`int32`。shape为`[num_levels]`。
- `locations(Tensor)`：位置张量，数据类型为`int32`。shape为`[bs, num_queries, num_heads, num_levels, num_points, 2]`。其中`bs`为batch size，`num_queries`为查询的数量，`num_heads`为头的数量，`num_levels`为特征图的数量，`num_points`为采样点的数量，`2`分别代表`y, x`。
- `weight(Tensor)`：权重张量，数据类型为`float32, float16`。shape为`[bs, num_queries, num_heads, num_levels, num_points]`。其中`bs`为batch size，`num_queries`为查询的数量，`num_heads`为头的数量，`num_levels`为特征图的数量，`num_points`为采样点的数量。
### 返回值
- `Tensor`：融合后的特征张量，数据类型为`float32, float16`。shape为`[bs, num_queries, num_heads*embed_dim]`。
### 支持的型号
- Atlas A2 训练系列产品
### 约束说明
- `locations`的值在`[0, 1]`之间。
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_multi_scale_deformable_attn_function
value = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]], dtype=torch.float32).npu()
shape = torch.tensor([[1, 1]], dtype=torch.int32).npu()
offset = torch.tensor([1], dtype=torch.int32).npu()
locations = torch.tensor([[[[[0.1, 0.2], [0.3, 0.4]]]]], dtype=torch.float32).npu()
weight = torch.tensor([[[[[1, 2], [3,4]]]]], dtype=torch.float32).npu()
out = npu_multi_scale_deformable_attn_function(value, shape, offset, locations, weight)
print(out)
```
```text
tensor([[[9.3002, 11.1603, 0.0000, 0.0000]]], dtype=torch.float32)
```
## voxelization
### 接口原型
```python
ads.common.voxelization(Tensor points, List[int] voxel_size, List[int] coors_range, int max_points=-1, int max_voxels=-1, bool deterministic=True) -> Tensor
```
### 功能描述
将点云数据进行体素化。
### 参数说明
- `points(Tensor)`：点云数据，数据类型为`float32, float16`。shape为`[3, N]`。其中`N`为点的数量，`3`分别代表`x, y, z`。
- `voxel_size(List[int])`：体素大小，数据类型为`float32, float16`。shape为`[3]`。其中`3`分别代表`x, y, z`。
- `coors_range(List[int])`：体素范围，数据类型为`float32, float16`。shape为`[6]`。其中`6`分别代表`x_min, y_min, z_min, x_max, y_max, z_max`。
- `max_points(int)`：每个体素的最大点数。默认值为`-1`。
- `max_voxels(int)`：最大体素数。默认值为`-1`。
- `deterministic(bool)`：是否确定性。默认值为`True`。
### 返回值
- `Tensor`：体素化后的张量，数据类型为`int32`。shape为`[max_voxels, max_points]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import voxelization
points = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32).npu()
out = voxelization(points, [1, 1, 1], [1, 2, 3, 4, 5, 6])
print(out)
```
## npu_nms3d_normal
### 接口原型
```python
ads.common.npu_nms3d_normal(Tensor boxes, Tensor scores, float: iou_threshold) -> Tensor
```
### 功能描述
3D非极大值抑制。
### 参数说明
- `boxes(Tensor)`：框张量，数据类型为`float32, float16`。shape 为`[N, 7]`。`7`分别代表`x, y, z, x_size, y_size, z_size, rz`。
- `scores(Tensor)`：评分张量，数据类型为`float32, float16`。shape 为`[N]`。
- `iou_threshold(float)`：IoU阈值。
### 返回值
- `Tensor`：NMS后的框张量，数据类型为`int32`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_nms3d_normal
boxes = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9]], dtype=torch.float32).npu()
scores = torch.tensor([1, 2], dtype=torch.float32).npu()
out = npu_nms3d_normal(boxes, scores, 0.5)
print(out)
```
```text
tensor([[1, 0]], dtype=torch.int32)
```
## npu_nms3d
### 接口原型
```python
ads.common.npu_nms3d(Tensor boxes, Tensor scores, float: iou_threshold) -> Tensor
```
### 功能描述
3D非极大值抑制，在bev视角下剔除多个3d box交并比大于阈值的box。
### 参数说明
- `boxes(Tensor)`：框张量，数据类型为`float32, float16`。shape 为`[N, 7]`。`7`分别代表`x, y, z, x_size, y_size, z_size, rz`。
- `scores(Tensor)`：评分张量，数据类型为`float32, float16`。shape 为`[N]`。
- `iou_threshold(float)`：IoU阈值。
### 返回值
- `Tensor`：NMS后的框张量，数据类型为`int32`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_nms3d
boxes = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9]], dtype=torch.float32).npu()
scores = torch.tensor([1, 2], dtype=torch.float32).npu()
out = npu_nms3d(boxes, scores, 0.5)
print(out)
```
```text
tensor([[1]], dtype=torch.int32)
```
## npu_furthest_point_sampling
### 接口原型
```python
ads.common.npu_furthest_point_sampling(Tensor points, int num_points) -> Tensor
```
### 功能描述
点云数据的最远点采样。
### 参数说明
- `points(Tensor)`：点云数据，数据类型为`float32, float16`。shape为`[B, N, 3]`。其中`B`为batch size，`N`为点的数量，`3`分别代表`x, y, z`。
- `num_points(int)`：采样点的数量。
### 返回值
- `Tensor`：采样后的点云数据，数据类型为`float32, float16`。shape为`[B, num_points]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import npu_furthest_point_sampling
points = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32).npu()
out = npu_furthest_point_sampling(points, 2)
print(out)
```
```text
tensor([[0, 2]], dtype=torch.int32)
```
## furthest_point_sample_with_dist
### 接口原型
```python
ads.common.furthest_point_sample_with_dist(Tensor points, int num_points) -> (Tensor, Tensor)
```
### 功能描述
与`npu_furthest_point_sampling`功能相同，但输入略有不同。
### 参数说明
- `points(Tensor)`：点云数据，表示各点间的距离，数据类型为`float32, float16`。shape为`[B, N, N]`。其中`B`为batch size，`N`为点的数量。
- `num_points(int)`：采样点的数量。
### 返回值
- `Tensor`：采样后的点云数据，数据类型为`float32, float16`。shape为`[B, num_points]`。
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch, torch_npu
from ads.common import furthest_point_sample_with_dist
points = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=torch.float32).npu()
out = furthest_point_sample_with_dist(points, 2)
print(out)
```
```text
tensor([[0, 2]], dtype=torch.int32)
```