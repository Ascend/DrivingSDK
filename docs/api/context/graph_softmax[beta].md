## graph_softmax[beta]
### 接口原型
```python
mx_driving.graph_softmax(Tensor src,Tensor index) -> Tensor
```
### 功能描述
根据index中各边对应的节点分组，对神经网络的src各边的特征值计算softmax。
### 参数说明
- `src (Tensor)`：各条边对应的特征，数据类型为`float32`，shape为`[num_edge, num_feature]`，该算子为了适配QCNet模型，故num_feature取值与模型一致默认为8。
- `index (Tensor)`：各条边指向的节点，数据类型为`int32`，shape为`[num_edge]`。
### 返回值
- `softmaxResult (Tensor)`：按照节点分组，指向同一个节点的所有边为一组，计算其softmax值，数据类型为`float32`，shape为`[num_edge, num_feature]`。
### 约束说明
- num_feature = 8
- 1 ≤ num_edge < 50000
- 0 ≤ index < 1500000
- -500.0 ≤ src < 500.0
### 支持的型号
- Atlas A2 训练系列产品
### 调用示例
```python
import torch
import torch_npu
import mx_driving

torch.manual_seed(1)

def gen_inputs(Num_Edge, Num_Feature):
    src = torch.rand((Num_Edge , Num_Feature))*1000-500 # src range is [-500, 500)
    index = torch.randint(0, 1500000, (Num_Edge,)) # [0, 1500000)
    grad_out = (torch.rand((Num_Edge, Num_Feature)) * 1e-3).float()
    return src, index, grad_out

Num_Feature = 8 # Feature number is 8 in QCNet Model
Num_Edge = 10

src, index, grad_out = gen_inputs(Num_Edge, Num_Feature)
out_npu = mx_driving.graph_softmax(src.npu(), index.npu())
out_npu.backward(grad_out)
```