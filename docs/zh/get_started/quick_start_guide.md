# 快速入手

## 调用 Driving SDK 高性能 API
```python
import torch, torch_npu
from mx_driving import scatter_max
updates = torch.tensor([[2, 0, 1, 3, 1, 0, 0, 4], [0, 2, 1, 3, 0, 3, 4, 2], [1, 2, 3, 4, 4, 3, 2, 1]], dtype=torch.float32).npu()
indices = torch.tensor([0, 2, 0], dtype=torch.int32).npu()
out = updates.new_zeros((3, 8))
out, argmax = scatter_max(updates, indices, out)
```
## 使用 Driving SDK 快速迁移自动驾驶模型

为方便用户快速在NPU上跑通基于GPU生态开发的模型，Driving SDK 提供了一键迁移能力，用户仅需添加两行代码即可。

### 1. 定位模型训练脚本入口

找到模型的训练脚本，通常命名为`train.py`，定位到它的入口函数，通常为：

```Python
if __name__ == '__main__':
	main()
```

### 2. 应用一键patcher

```Python
from mx_driving.patcher import default_patcher_builder
#......
if __name__ == '__main__':
	with default_patcher_builder.build() as patcher:
		main()
```

若模型训练报错，可进一步参考[模型FAQ](../faq/model_faq.md)进行排查。若希望进一步了解一键patcher特性，可查阅[一键Patcher文档](../features/patcher.md)。 

### 3. 进一步优化模型性能

若NPU上训练性能跟基线差距较大，可参考[模型迁移优化指导](../migration_tuning/model_optimization.md)和[负载均衡优化](../features/dataload_balance.md)进一步优化hostbound、快慢卡等性能瓶颈。

### 4. 导出离线模型
Driving SDK 支持单算子onnx导出，并提供了示例，用户若有诉求需自行参考[训推一体示例](../features/onnx_example.md)开发。