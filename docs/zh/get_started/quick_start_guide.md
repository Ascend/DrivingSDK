# 快速上手

DrivingSDK 同时提供了**高性能 API** 和**优选模型库**两大核心能力：

- **高性能 API**：提供 `scatter_max` 等 NPU 优化算子，可直接调用以替代 PyTorch 原生算子，获得更好的执行性能；
- **优选模型库**：提供 BEVFusion、BEVFormer、Sparse4D 等典型自动驾驶模型的 NPU 迁移方案。

下面分别以 `scatter_max` 算子调用和 BEVFusion 模型为例，引导开发者快速上手。请先参考[部署Driving SDK环境](../installation/installation.md)完成基础环境部署。

---

## 调用高性能 API

以下示例演示如何调用 DrivingSDK 的 [`scatter_max`](../api/context/scatter_max.md) 算子，并打印结果以验证输出是否正确，其主要功能是根据索引将多个源张量中对应位置的最大值聚合到目标张量，并同时返回每个最大值来自哪个源张量。

```python
import torch, torch_npu
from mx_driving import scatter_max

updates = torch.tensor([[2, 0, 1, 3, 1, 0, 0, 4],
                        [0, 2, 1, 3, 0, 3, 4, 2],
                        [1, 2, 3, 4, 4, 3, 2, 1]], dtype=torch.float32).npu()
indices = torch.tensor([0, 2, 0], dtype=torch.int32).npu()
out = updates.new_zeros((3, 8))
out, argmax = scatter_max(updates, indices, out)
print(out)
print(argmax)
```

若运行正常，应输出：

```text
tensor([[2., 2., 3., 4., 4., 3., 2., 4.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 2., 1., 3., 0., 3., 4., 2.]], device='npu:0')
tensor([[0, 2, 2, 2, 2, 2, 2, 0],
        [3, 3, 3, 3, 3, 3, 3, 3],
        [1, 1, 1, 1, 1, 1, 1, 1]], device='npu:0', dtype=torch.int32)
```

更多高性能 API 及详细说明请参阅 [API 清单](../api/README.md)。

---

## BEVFusion 模型训练

[BEVFusion](../../../model_examples/BEVFusion/README.md)是一个高效且通用的多任务多传感器融合框架，它在共享的鸟瞰图（BEV）表示空间中统一了多模态特征，这很好地保留了几何和语义信息，从而更好地支持 3D 感知任务。

以下以 BEVFusion 模型为例，演示从环境准备到模型训练的完整流程。

### 1. 安装模型环境

进入 BEVFusion 模型代码目录：

```shell
cd DrivingSDK/model_examples/BEVFusion
```

推荐使用一键配置脚本安装依赖：

```shell
bash install_BEVFusion.sh
```

该脚本会自动完成 mmcv 和 mmdetection3d 的源码编译安装。

### 2. 准备训练数据集

进入 `install_BEVFusion.sh` 脚本自动创建的 `mmdetection3d` 目录：

```shell
cd mmdetection3d
```

请自行下载 [nuScenes 数据集](https://www.nuscenes.org/nuscenes#download) 或构建软连接到`nuscenes`文件夹下，在 `mmdetection3d/` 下构建如下目录结构：

```shell
data/
├── lyft
├── nuscenes
├── ├── maps
├── ├── samples
├── ├── sweeps
├── ├── v1.0-test
├── ├── v1.0-trainval
├── s3dis
├── scannet
├── sunrgbd
```

在 `mmdetection3d` 目录下进行数据预处理：

```shell
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

预处理后`nuscenes`文件结构如下：

```shell
nuscenes/
├── maps
├── nuscenes_gt_database
├── nuscenes_infos_test.pkl
├── nuscenes_infos_train.pkl
├── nuscenes_infos_val.pkl
├── samples
├── sweeps
├── v1.0-test
├── v1.0-trainval
```

### 3. 准备预训练权重

在`mmdetection3d`目录下创建`pretrained`文件夹，参考 [BEVFusion Model](https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion)，下载预训练权重 [Swin pre-trained model](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/swint-nuimages-pretrained.pth) 和 [lidar-only pre-trained detector](https://download.openmmlab.com/mmdetection3d/v1.1.0_models/bevfusion/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth)。将预训练权重放在`pretrained`文件夹中，目录样例如下：

```shell
pretrained/
├── swint-nuimages-pretrained.pth
├── bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth
 ```

### 4. 启动训练

返回 BEVFusion 模型目录，执行训练脚本：

```shell
cd ../
```

脚本支持以下命令行参数（支持默认值+关键字参数+位置参数）：

- `--batch-size`：每卡 batch size，默认值 4；
- `--num-npu`：每节点 NPU 卡数，默认值 8；
- `--epochs`：训练轮数，默认值 6。

```shell
# FP32 精度训练（默认6个epochs）
bash test/train_full_8p_base_fp32.sh --batch-size=4 --num-npu=8
```

训练日志默认保存在 `test/output/0/train_0.log`。若看到类似以下打屏信息，表明训练已成功拉起：

```text
02/03 23:41:05 - mmengine - INFO - Epoch(train) [1][   1/3862]  lr: 6.6667e-05  eta: 36 days, 9:13:56  time: 135.6712  data_time: 4.1619  memory: 18113  grad_norm: nan  loss: 18.0867  loss_heatmap: 6.6903  layer_-1_loss_cls: 0.5519  layer_-1_loss_bbox: 10.8446  matched_ious: 0.0018
02/03 23:41:07 - mmengine - INFO - Epoch(train) [1][   2/3862]  lr: 6.6934e-05  eta: 18 days, 8:22:00  time: 68.4213  data_time: 2.0908  memory: 18231  grad_norm: nan  loss: 17.8768  loss_heatmap: 6.7121  layer_-1_loss_cls: 0.5522  layer_-1_loss_bbox: 10.6125  matched_ious: 0.0037
```

训练完成后会自动进行验证，输出类似以下指标：

```text
mAP: 0.6805
mATE: 0.2762
mASE: 0.2518
mAOE: 0.3478
mAVE: 0.2928
mAAE: 0.1951
NDS: 0.7039
Eval time: 158.1s
```

更多模型示例（如BEVFormer、Sparse4D 等）及详细使用指导，请参考[模型清单](../models/support_list.md)。
