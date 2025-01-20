# BEVFormer

# 概述

BEVFormer 通过提取环视相机采集到的图像特征，并将提取的环视特征通过模型学习的方式转换到 BEV 空间（模型去学习如何将特征从 图像坐标系转换到 BEV 坐标系），从而实现 3D 目标检测和地图分割任务。

- 参考实现：

  ```
  url=https://github.com/fundamentalvision/BEVFormer
  commit_id=66b65f3a1f58caf0507cb2a971b9c0e7f842376c
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
      cp mmcv_config.patch mmcv
      cd mmcv
      git apply --reject mmcv_config.patch
      pip install -r requirements/runtime.txt
      MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
      MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
     ```
  2. 源码安装 mmdetection3d v1.0.0rc4
     ```
     git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
     cp mmdet3d_config.patch mmdetection3d
     cd mmdetection3d
     git apply --reject mmdet3d_config.patch
     pip install -e .
     ```
  3. 源码安装 mmdet 2.24.0
     ```
     git clone -b v2.24.0 https://github.com/open-mmlab/mmdetection.git
     cp mmdet_config.patch mmdetection
     cd mmdetection
     git apply --reject mmdet_config.patch
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
  6. 模型代码更新
     ```
     git clone https://github.com/fundamentalvision/BEVFormer.git
     cp bev_former_config.patch BEVFormer
     cd BEVFormer
     git checkout 66b65f3a1f58caf0507cb2a971b9c0e7f842376c
     git apply --reject --whitespace=fix bev_former_config.patch
     cd ../
     ```
  7. 安装Driving SDK加速库

## 准备数据集

1. 用户需自行下载 nuScenes V1.0 full 和 CAN bus 数据集放置在BEVFormer模型代码目录下，结构如下：

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
   在BEVFormer模型代码目录下创建 ckpts 文件夹，将预训练权重 r101_dcn_fcos3d_pretrain.pth 放入其中
   ```
   ckpts/
   ├── r101_dcn_fcos3d_pretrain.pth
   ```

# 开始训练

- 单机8卡训练(精度)
   ```shell
   bash test/train_full_8p_base_fp32.sh --epochs=4 # 8卡训练，默认训练24个epochs，这里只训练4个epochs
   ```
- 单机8卡训练（性能）
   ```shell
   bash test/train_performance_8p_base_fp32.sh # 8卡性能
   ```
# 结果

|  NAME       | Backbone          | Method          |   训练方式     |     Epoch    |      NDS     |     mAP      |     FPS      |
|-------------|-------------------|-----------------|---------------|--------------|--------------|--------------|--------------|
|  8p-Atlas 800T A2 | R101-DCN    | BEVFormer-base  |       FP32    |        4     |      44.23   |      35.52   |      3.094    |
|  8p-竞品A   | R101-DCN          | BEVFormer-base  |       FP32    |        4     |      43.58   |      34.45   |      3.320    |


# 版本说明

## 变更

2024.3.8：首次发布。
2025.1.6：更新发布。

## FAQ