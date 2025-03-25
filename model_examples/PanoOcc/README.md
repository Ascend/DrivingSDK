# PanoOcc for PyTorch

## 目录

- [PanoOcc for PyTorch](#panoocc-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [PanoOcc（在研版本）](#panoocc在研版本)
  - [准备训练环境](#准备训练环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [安装模型环境](#安装模型环境)
    - [准备数据集](#准备数据集)
    - [准备预训练权重](#准备预训练权重)
  - [快速开始](#快速开始)
    - [训练任务](#训练任务)
      - [开始训练](#开始训练)
      - [训练结果](#训练结果)
- [变更说明](#变更说明)
- [FAQ](#faq)

# 简介

## 模型介绍

现有的感知任务（如对象检测、道路结构分割等）都只关注整体 3D 场景理解任务的一小部分。这种分而治之的策略简化了算法开发过程，但代价是失去了问题的端到端统一解决方案。PanoOcc 利用体素查询以从粗到细的方案聚合来自多帧和多视图图像的时空信息，将特征学习和场景表示集成到统一的占用表示中，为仅依赖相机的 3D 场景理解实现统一的占用表示，实现了基于相机的 3D 全景分割。

## 支持任务列表

本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| PanoOcc |   训练   |    ✔     |

## 代码实现

- 参考实现：

    ```
    url=https://github.com/Robertwyq/PanoOcc
    commit_id=3d93b119fcced35612af05587b395e8b38d8271f
    ```

# PanoOcc（在研版本）

## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 6.0.0  |
|       CANN        | 8.0.0  |

### 安装模型环境

**表 2**  三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |   2.1.0   |

0. 激活 CANN 环境

    将 CANN 包目录记作 cann_root_dir，执行以下命令以激活环境
    ```
    source {cann_root_dir}/set_env.sh
    ```
1. 准备模型源码

    在当前目录下，克隆并准备 PanoOcc 源码

    ```
    git clone https://github.com/Robertwyq/PanoOcc.git
    cp panoocc.patch PanoOcc
    cp -r test/ PanoOcc/
    cd PanoOcc
    git checkout 3d93b119fcced35612af05587b395e8b38d8271f
    git apply --reject --whitespace=fix panoocc.patch
    cd ../
    ```

2. 源码编译安装 mmcv

    克隆 mmcv 仓，并进入 mmcv 目录编译安装

    ```
    git clone -b 1.x https://github.com/open-mmlab/mmcv
    cp mmcv.patch mmcv
    cd mmcv
    git apply --reject mmcv.patch
    MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
    cd ../
    ```

3. 安装 mmdet

    克隆 mmdet 仓，并进入 mmdet 目录编译安装

    ```
    git clone -b v2.24.0 https://github.com/open-mmlab/mmdetection.git
    cp mmdetection.patch mmdetection
    cd mmdetection
    git apply --reject mmdetection.patch
    pip install -e .
    cd ../
    ```

4. 安装 mmdet3d

    在模型根目录下，克隆 mmdet3d 仓，替换其中部分代码，并进入 mmdet3d 目录安装

    ```
    git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
    cp mmdetection3d.patch mmdetection3d
    cd mmdetection3d
    git apply --reject mmdetection3d.patch
    pip install -v -e .
    cd ../
    ```

5. 安装其他依赖

    ```
    pip install mmsegmentation==0.30.0
    pip install torch==2.1.0 torchvision
    pip install ipython==8.18.1
    ```

5. 安装 Driving SDK 加速库

    安装方法参考[原仓](https://gitee.com/ascend/DrivingSDK/wikis/DrivingSDK%20%E4%BD%BF%E7%94%A8)

### 准备数据集

1. 根据原仓数据集准备中的 [NuScenes LiDAR Benchmark](https://github.com/Robertwyq/PanoOcc/blob/main/docs/dataset.md#1-nuscenes-lidar-benchmark) 章节在模型源码根目录下准备数据集，参考数据集结构如下：

    ```
    PanoOcc
    ├── data/
    │   ├── nuscenes/
    │   │   ├── can_bus/
    │   │   ├── maps/
    │   │   ├── lidarseg/
    │   │   ├── panoptic/
    │   │   ├── samples/
    │   │   ├── sweeps/
    │   │   ├── v1.0-trainval/
    │   │   ├── v1.0-test/
    │   │   ├── nuscenes_infos_temporal_train.pkl (经数据预处理后生成)
    │   │   ├── nuscenes_infos_temporal_val.pkl (经数据预处理后生成)
    │   │   ├── nuscenes_infos_temporal_test.pkl (经数据预处理后生成)
    │   │   ├── nuscenes.yaml
    ```

2. 在模型源码根目录下进行数据预处理

   ```
   python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data/nuscenes
   ```

### 准备预训练权重

在模型源码根目录下创建 ckpts 文件夹，将预训练权重 r101_dcn_fcos3d_pretrain.pth 放入其中
   ```
   ckpts/
   ├── r101_dcn_fcos3d_pretrain.pth
   ```

## 快速开始

### 训练任务

本任务主要提供**单机**的**8卡**训练脚本。

#### 开始训练

  - 在模型源码根目录下，运行训练脚本。

     ```
     bash test/train_8p_panoocc_base_4f_fp32.sh --epochs=3 # 8卡性能
     bash test/train_8p_panoocc_base_4f_fp32.sh # 8卡精度
     ```

#### 训练结果

| 芯片          | 卡数 | global batch size | Precision | epoch | mIoU | mAP | NDS | 性能-单步迭代耗时(ms) |
| ------------- | :--: | :---------------: | :-------: | :---: | :----: | :----: | :----: | :-------------------: |
| 竞品A           |  8p  |         8         |   fp32    |  24   | 0.712 | 0.411 | 0.497 |         1322          |
| Atlas 800T A2 |  8p  |         8         |   fp32    |  24   | 0.710 | 0.416 | 0.499 |         2211          |

# 变更说明

2024.09.10：首次发布。


# FAQ

无
