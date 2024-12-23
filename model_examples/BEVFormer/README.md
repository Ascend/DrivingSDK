# BEVFormer

# 概述

BEVFormer 通过提取环视相机采集到的图像特征，并将提取的环视特征通过模型学习的方式转换到 BEV 空间（模型去学习如何将特征从 图像坐标系转换到 BEV 坐标系），从而实现 3D 目标检测和地图分割任务。

- 参考实现：

  ```
  url=https://github.com/fundamentalvision/BEVFormer
  commit_id=20923e66aa26a906ba8d21477c238567fa6285e9
  ```

# 支持模型

| Backbone          | Method          |   训练方式     |
|-------------------|-----------------|---------------|
| R101-DCN          | BEVFormer-base  |       FP32    |

# 准备训练环境

## 准备环境

- 当前模型支持的 PyTorch 版本如下表所示。

  **表 1**  版本支持表

  | Torch_Version      |
  | :--------: | 
  | PyTorch 2.1 | 
  
- 环境准备指导。

  请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》搭建torch环境。
  
- 安装依赖。

  1. 源码编译安装 mmcv 1.x
     ```
      git clone -b 1.x https://github.com/open-mmlab/mmcv.git
      cp -f mmcv_need/base_runner.py mmcv/mmcv/runner/base_runner.py
      cp -f mmcv_need/epoch_based_runner.py mmcv/mmcv/runner/epoch_based_runner.py
      cp -f mmcv_need/points_in_polygons_npu.cpp mmcv/mmcv/ops/csrc/pytorch/npu/points_in_polygons_npu.cpp
      cp -f mmcv_need/distributed.py mmcv/mmcv/parallel/distributed.py
      cp -f mmcv_need/modulated_deform_conv.py mmcv/mmcv/ops/modulated_deform_conv.py
      cp -f mmcv_need/runtime.txt mmcv/requirements/runtime.txt
      cp -f mmcv_need/optimizer.py mmcv/mmcv/runner/hooks/optimizer.py
      cd mmcv
      pip install -r requirements/runtime.txt
      MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
      MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
     ```
  2. 源码安装 mmdetection3d v1.0.0rc4
     ```
     git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
     cp -f mmdet3d_need/__init__.py mmdetection3d/mmdet3d/__init__.py
     cp -f mmdet3d_need/nuscenes_dataset.py mmdetection3d/mmdet3d/datasets/nuscenes_dataset.py
     cp -f mmdet3d_need/runtime.txt mmdetection3d/requirements/runtime.txt
     cd mmdetection3d
     pip install -e .
     ```
  3. 源码安装 mmdet 2.24.0
     ```
     git clone -b v2.24.0 https://github.com/open-mmlab/mmdetection.git
     cp -f mmdet_need/__init__.py mmdetection/mmdet/__init__.py
     cp -f mmdet_need/resnet.py mmdetection/mmdet/models/backbones/resnet.py
     cd mmdetection
     pip install -e .
     ```
  4. 安装 detectron2
     ``` 
     python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
     ```
  5. 安装其他依赖
     ```
     pip install -r requirements.txt
     ```
  6. 安装Driving SDK加速库，并将环境变量添加至 test/env_npu.sh 文件中

## 准备数据集

1. 用户需自行下载 nuScenes V1.0 full 和 CAN bus 数据集，结构如下：

   ```
   data/
   ├── nuscenes
   ├── can_bus
   ```
2. 数据预处理
   ```
   python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
   ```

## 下载预训练权重
创建 ckpts 文件夹，将预训练权重 r101_dcn_fcos3d_pretrain.pth 放入其中
   ```
   ckpts/
   ├── r101_dcn_fcos3d_pretrain.pth
   ```

# 开始训练

- 单机8卡训练
   
     ```shell
     bash test/train_full_8p_base_fp32.sh --epochs=4 # 8卡训练，默认训练24个epochs，这里只训练4个epochs
     bash test/train_performance_8p_base_fp32.sh # 8卡性能
     ```

# 结果

|  NAME       | Backbone          | Method          |   训练方式     |     Epoch    |      NDS     |     mAP      |     FPS      |
|-------------|-------------------|-----------------|---------------|--------------|--------------|--------------|--------------|
|  8p-Atlas 800T A2 | R101-DCN    | BEVFormer-base  |       FP32    |        4     |      44.23   |      35.52   |      3.094    |
|  8p-竞品A   | R101-DCN          | BEVFormer-base  |       FP32    |        4     |      43.58   |      34.45   |      3.320    |

# 公网地址说明
代码涉及公网地址参考 public_address_statement.md

# 版本说明

## 变更

2024.3.8：首次发布。

## FAQ