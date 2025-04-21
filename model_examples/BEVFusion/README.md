# BEVFusion

# 概述

BEVFusion 是一个高效且通用的多任务多传感器融合框架，它在共享的鸟瞰图（BEV）表示空间中统一了多模态特征，这很好地保留了几何和语义信息，从而更好地支持 3D 感知任务。

- 参考实现：

  ```
  url=https://github.com/open-mmlab/mmdetection3d/tree/main/projects/BEVFusion
  commit_id=0f9dfa97a35ef87e16b700742d3c358d0ad15452
  ```

# 支持模型

| Modality  | Voxel type (voxel size) | 训练方式 |
|-----------|-------------------------|------|
| lidar-cam | lidar-cam               | FP32 |

# 准备训练环境
## 安装昇腾环境
请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境。本仓已支持表1中软件版本。
  
  **表 1**  昇腾软件版本支持表

  |        软件类型        |   支持版本   |
  |:------------------:|:--------:|
  | FrameworkPTAdapter | 6.0.0  |
  |       CANN         | 8.0.0  |

## 安装模型环境
- 当前模型支持的 PyTorch 版本如下表所示。

  **表 2**  版本支持表

  | Torch_Version |
  |:-------------:|
  |  PyTorch 2.1  |


- 安装依赖。
  首先进入代码目录：

  ```
  cd model_examples/BEVFusion
  ```

  然后按步骤安装依赖：

  1. 源码编译安装 mmcv rc4main 分支

  ```
  git clone -b rc4main https://github.com/momo609/mmcv.git
  cp -f mmcv.patch mmcv
  cd mmcv
  git apply mmcv.patch
  pip install -r requirements/runtime.txt
  MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py build_ext
  MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
  cd ../
  ```

  2. 源码安装 mmengine v0.9.0 版本

  ```
  git clone -b v0.9.0 https://github.com/open-mmlab/mmengine.git
  cp -f mmengine.patch mmengine/
  cd mmengine
  git apply mmengine.patch
  pip install -e .
  cd ../
  ```

  3. 源码安装 mmdetection3d v1.2.0 版本

  ```
  git clone -b v1.2.0 https://github.com/open-mmlab/mmdetection3d.git
  cp -f bevfusion.patch mmdetection3d/
  cd mmdetection3d
  git apply bevfusion.patch
  pip install mmdet==3.1.0 numpy==1.23.5 yapf==0.40.1
  pip install -e .
  ```
  
  4. 参考 Driving SDK 构建说明，安装 Driving SDK 加速库，并将对应的环境变量添加至 test/env_npu.sh 文件中。

## 准备数据集

1. 用户需自行下载 nuScenes 数据集，放在 mmdetection3d 目录下，或者自行构建软链接，结构如下：

   ```
   data
   ├── lyft
   ├── nuscenes
   ├── s3dis
   ├── scannet
   └── sunrgbd
   ```

   除了 nuscenes 目录，其他为原始仓库已有目录。
2. 在 mmdetection3d 目录下进行数据预处理，处理方法参考原始github仓库：

   ```
   python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
   ```

## 下载预训练权重

在 mmdetection3d 目录下创建 pretrained 文件夹，参考 github仓库，下载预训练权重。最后将预训练权重放在 pretrained 文件夹中，目录样例如下：

```
pretrained/
├── swint-nuimages-pretrained.pth
├── bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth
```

# 开始训练

回到最开始的模型目录：

```
cd model_examples/BEVFusion
```

- 单机8卡训练

  ```shell
  bash test/train_full_8p_base_fp32.sh # 8卡训练，默认训练6个epochs
  bash test/train_performance_8p_base_fp32.sh # 8卡性能，默认训练1个epochs
  ```

# 训练结果

| NAME             | Modality  | Voxel type (voxel size) | 训练方式 | Epoch | global batch size | NDS   | mAP   | FPS   |
|------------------|-----------|-------------------------|------|-------|-------|-------|-------|-------|
| 8p-Atlas 800T A2 | lidar-cam | 0.075                   | FP32 | 6     | 32 | 69.48 | 66.6  | 19.42 |
| 8p-竞品A           | lidar-cam | 0.075                   | FP32 | 6     | 32 | 69.78 | 67.36 | 22.54 |

# 版本说明

## 变更

2024.12.5：首次发布。

## FAQ
