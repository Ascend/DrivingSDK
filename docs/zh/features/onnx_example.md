# 训推一体示例

## 概述

以MultiScaleDeformableAttn算子为例，展示由模型onnx导出到om模型执行的整个过程，提供本仓库onnx使用的基础案例。

## 单算子onnx导出脚本

通过python脚本导出单算子模型，以下为导出脚本：

```python
import os
import torch
import torch_npu
import mx_driving
from mx_driving import multi_scale_deformable_attn

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, value, shapes, level_start_index, sampling_locations, attention_weights):
        return multi_scale_deformable_attn(value, shapes, level_start_index, sampling_locations, attention_weights)


def onnx_export(model, inputs, onnx_model_name,
                input_names=None, output_names=None):
    if input_names is None:
        input_names = ["input_names"]
    if output_names is None:
        output_names = ["output_names"]
    model.eval()
    OPERATOR_EXPORT_TYPE = torch._C._onnx.OperatorExportTypes.ONNX
    with torch.no_grad():
        torch.onnx.export(model, inputs,
                            onnx_model_name,
                            opset_version=11,
                            operator_export_type=OPERATOR_EXPORT_TYPE,
                            input_names=input_names,
                            output_names=output_names)


def export_onnx(name):
    bs, num_levels, num_heads, num_points, num_queries, embed_dims = 2, 1, 8, 4, 40000, 32
    shapes = torch.tensor([[200, 200] * num_levels]).reshape(num_levels, 2).long()

    num_keys = sum((H * W).item() for H, W in shapes)
    value = torch.rand(bs, num_keys, num_heads, embed_dims) * 0.01
    sampling_locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2) * 1.2 - 0.1
    attention_weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points) + 1e-5
    level_start_index = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1])).long()

    value = value.half()
    sampling_locations = sampling_locations.half()
    attention_weights = attention_weights.half()

    npu_value = value.clone().detach().npu()
    npu_sampling_locations = sampling_locations.clone().detach().npu()
    npu_attention_weights = attention_weights.clone().detach().npu()
    npu_level_start_index = level_start_index.clone().detach().npu()
    npu_shapes = shapes.clone().detach().npu()

    model = Model().npu()
    model(npu_value, npu_shapes, npu_level_start_index, npu_sampling_locations, npu_attention_weights)
    onnx_export(model, (npu_value, npu_shapes, npu_level_start_index, npu_sampling_locations, npu_attention_weights), name,
        ["value, shapes, level_start_index, sampling_locations, attention_weights"], ["outputs"])

if __name__ =='__main__':
    export_onnx("./msda.onnx")
```
执行后会在当前文件夹下生成msda.onnx文件

## domain转换

若onnx转换om过程中出现FAQ中 The model has 2  domain_version fields 问题，则需要安装转换仓库进行domain转换，其中转换仓库为：https://gitee.com/Ronnie_zheng/MagicONNX
转换脚本为：

```shell
from magiconnx import OnnxGraph
graph = OnnxGraph('msda.onnx')

graph.keep_default_domain()
graph.save('msda.onnx')
```
执行后会将onnx模型中的多个domian进行统一

## onnx转换om

通过atc将onnx转换为om模型，在执行前需要设置环境变量，环境变量与转换指令如下，其中soc_version可通过npu-smi info进行查看：

```shell
pip3 show mx_driving
export ASCEND_CUSTOM_OPP_PATH=xxx/site-packages/mx_driving/packages/vendors/customize/
export LD_LIBRARY_PATH=xxx/site-packages/mx_driving/packages/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
atc --framework 5 --output msda --soc_version Ascend910B2 --model msda.onnx --op_select_implmode high_precision --precision_mode must_keep_origin_dtype -log debug
```

执行后若出现ATC run success, welcome to the next use. 则说明om模型转换成功

## 输入数据生成

根据onnx模型导出时的模型输入，构建输入文件，其构建脚本如下：

```python
import numpy as np
import torch
import torch_npu

bs, num_levels, num_heads, num_points, num_queries, embed_dims = 2, 1, 8, 4, 40000, 32
shapes = torch.tensor([[200, 200] * num_levels]).reshape(num_levels, 2).long()

num_keys = sum((H * W).item() for H, W in shapes)
level_start_index = torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1])).long()

input1 = np.random.rand(bs, num_keys, num_heads, embed_dims).astype(np.float16)
input2 = shapes.numpy().astype(np.int64)
input3 = level_start_index.numpy().astype(np.int64)
input4 = np.random.rand(bs, num_queries, num_heads, num_levels, num_points, 2).astype(np.float16)
input5 = np.random.rand(bs, num_queries, num_heads, num_levels, num_points).astype(np.float16)

# 生成输入.bin
input1.tofile("./inputs/input1.bin")
input2.tofile("./inputs/input2.bin")
input3.tofile("./inputs/input3.bin")
input4.tofile("./inputs/input4.bin")
input5.tofile("./inputs/input5.bin")
```

执行后会在inputs文件夹内生成模型的各输入文件

## om执行

克隆仓库https://gitee.com/ascend/tools/tree/master/msame，并按照readme进行安装

随后通过msame工具，将生成好的模型输入文件输入到om模型中执行

```python
./msame --model ./msda.om --input ./inputs/input1.bin,./inputs/input2.bin,./inputs/input3.bin,./inputs/input4.bin,./inputs/input5.bin --output ./msame/out/ --outfmt BIN --loop 1
```

执行后会在./msame/out文件夹下生成om模型的推理结果

## 精度验证

本部分验证om模型的推理结果与单算子结果是否一致，可自行选择工具进行验证，示例代码如下：

```python
import numpy as np
import torch, torch_npu
import mx_driving
from mx_driving import multi_scale_deformable_attn

bs, num_levels, num_heads, num_points, num_queries, embed_dims = 2, 1, 8, 4, 40000, 32

input1 = np.fromfile("./inputs/input1.bin", dtype=np.float16).reshape(bs, -1, num_heads, embed_dims)
input2 = np.fromfile("./inputs/input2.bin", dtype=np.int64).reshape(num_levels, 2)
input3 = np.fromfile("./inputs/input3.bin", dtype=np.int64).reshape(num_levels)
input4 = np.fromfile("./inputs/input4.bin", dtype=np.float16).reshape(bs, num_queries, num_heads, num_levels, num_points, 2)
input5 = np.fromfile("./inputs/input5.bin", dtype=np.float16).reshape(bs, num_queries, num_heads, num_levels, num_points)

input1_npu = torch.from_numpy(input1).npu()
input2_npu = torch.from_numpy(input2).npu()
input3_npu = torch.from_numpy(input3).npu()
input4_npu = torch.from_numpy(input4).npu()
input5_npu = torch.from_numpy(input5).npu()

golden = multi_scale_deformable_attn(input1_npu, input2_npu, input3_npu, input4_npu, input5_npu)
output = torch.from_numpy(np.fromfile("msda_output_0.bin", dtype=np.float16).reshape(bs, num_queries, num_heads * embed_dims))

print(golden)
print(output)
```

## FAQ

### No parser is register for Op

可能原因1：在docker中编译的mx_driving包可能不附带onnx插件，需要编译protoc，按照[Driving SDK 编译指南](https://gitcode.com/Ascend/DrivingSDK/blob/master/docs/get_started/compile.md)中FAQ方式进行编译即可

可能原因2：未引入onnx转换om环节中的mx_driving环境变量，导致转换过程中未检索到对应算子而报错

### Can not find Node xxx custom infer_datatype func

原因：目前仓库内仅部分算子包含inferShape与inferDtype过程，添加如https://gitcode.com/Ascend/DrivingSDK/blob/master/kernels/op_host/multi_scale_deformable_attn.cpp文件中类似105行代码：
IMPL_OP_INFERSHAPE(MultiScaleDeformableAttn).InferShape(InferShapeForMultiScaleDeformableAttn).InferDataType(InferDataTypeForMultiScaleDeformableAttn); 即可解决该问题

### Optype xxx of ops kernel is unsupported

原因：给定的输入类型与算子支持的输入类型不匹配，请检查模型的输入类型

### The model has 2  domain_version fields, but only one is allowed

原因：未进行domain转换，转换过程可见上方案例