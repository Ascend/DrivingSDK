# VAD

## 目录

- [VAD](#vad)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
  - [准备训练环境](#准备训练环境)
    - [安装环境](#安装环境)
    - [安装昇腾环境](#安装昇腾环境)
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

VAD是一个向量化端到端的自动驾驶网络，将驾驶场景建模完全向量化表示，通过向量化的实例运动和地图元素作为显式的实例级规划约束，提升了规划安全性也提升了计算效率。

## 支持任务列表
本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| VAD |   训练   |    ✔     |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/hustvl/VAD
  commit_id=70bb364aa3f33316960da06053c0d168628fb15f
  ```
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/DrivingSDK.git
  code_path=model_examples/VAD
  ```

## 准备训练环境

### 安装环境

  **表 1**  三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |   2.1   |

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。

  **表 2**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 7.0.0  |
|       CANN        | 8.1.RC1 |

- 克隆代码仓到当前目录并使用patch文件

    ```
    git clone https://gitee.com/ascend/DrivingSDK.git -b master
    cd DrivingSDK/model_examples/VAD
    git clone https://github.com/hustvl/VAD.git
    cd VAD
    git checkout 70bb364aa3f33316960da06053c0d168628fb15f
    cp -f ../VAD_npu.patch .
    git apply --reject --whitespace=fix VAD_npu.patch
    cp -rf ../test .
    ```


- 安装mmdet3d

  - 在应用过patch的模型根目录下，克隆mmdet3d仓，并进入mmdetection3d目录编译安装

    ```
    git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
    cp -r ../mmdetection3d.patch mmdetection3d
    cd mmdetection3d
    git apply --reject mmdetection3d.patch
    pip install -v -e .
    ```

- 安装mmcv

  - 在应用过patch的模型根目录下，克隆mmcv仓，并进入mmcv目录安装

    ```
    git clone -b 1.x https://github.com/open-mmlab/mmcv
    cd mmcv
    pip install -r requirements/runtime.txt
    MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
    ```
- 安装Driving SDK加速库，安装master分支，具体方法参考[原仓](https://gitee.com/ascend/DrivingSDK)。

- 在应用过patch的模型根目录下，安装相关依赖。

  ```
  pip install -r requirements.txt
  ```



### 准备数据集

- 根据原仓**Prepare Dataset**章节准备数据集，数据集目录及结构如下：

```
VAD
├── projects/
├── tools/
├── configs/
├── ckpts/
│   ├── resnet50-19c8e357.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── vad_nuscenes_infos_temporal_train.pkl
|   |   ├── vad_nuscenes_infos_temporal_val.pkl
```

> **说明：**  
> 该数据集的训练过程脚本只作为一种参考示例。      

### 准备预训练权重

- 在应用过patch的模型根目录下创建ckpts文件夹并下载预训练权重。
```
mkdir ckpts
cd ckpts 
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
```

## 快速开始

### 训练任务

本任务主要提供**单机**的**8卡**训练脚本。

#### 开始训练

  1. 在应用过patch的模型根目录下，运行训练脚本。

     该模型支持单机8卡训练。

     - 单机8卡精度训练

     ```
     bash test/train_8p.sh
     ```

     - 单机8卡性能训练


     ```
     bash test/train_8p_performance.sh
     ```


#### 训练结果
| 芯片          | 卡数 | global batch size | Precision | epoch |  loss   | FPS |
| ------------- | :--: | :---------------: | :-------: | :---: | :----: | :-------------------: |
| 竞品A           |  8p  |         8         |   fp32    |  8   | 10.6220 |     7.476         |
| Atlas 800T A2 |  8p  |         8         |   fp32    |  8   | 10.6539 |   2.847          |


# 变更说明

2025.02.08：首次发布

# FAQ

无