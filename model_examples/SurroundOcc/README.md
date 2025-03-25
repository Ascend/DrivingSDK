# SurroundOcc for PyTorch

## 目录

- [SurroundOcc for PyTorch](#surroundocc-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [SurroundOcc](#surroundocc)
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

传统的 3D 场景理解方法大多数都集中在 3D 目标检测上，难以描述任意形状和无限类别的真实世界物体。而 SurroundOcc 方法可以更全面地感知 3D 场景。首先对每个图像提取多尺度特征，并采用空间 2D-3D 注意力将其提升到 3D 体积空间；然后，采用 3D 卷积逐步上采样体积特征，并在多个级别上施加监督。

## 支持任务列表
本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| SurroundOcc |   训练   |    ✔     |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/weiyithu/SurroundOcc
  commit_id=05263c6a8fe464a7f9d28358ff7196ba58dc0de6
  ```
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitee.com/ascend/DrivingSDK.git
  code_path=model_examples/SurroundOcc
  ```

# SurroundOcc

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
| FrameworkPTAdaper | 6.0.0  |
|       CANN        | 8.0.0  |

- 克隆代码仓到当前目录并使用patch文件

    ```
    git clone https://gitee.com/ascend/DrivingSDK.git -b master
    cd DrivingSDK/model_examples/SurroundOcc
    git clone https://github.com/weiyithu/SurroundOcc.git
    cd SurroundOcc
    git checkout 05263c6a8fe464a7f9d28358ff7196ba58dc0de6
    cp -f ../SurroundOcc_npu.patch .
    git apply --reject --whitespace=fix SurroundOcc_npu.patch
    cp -rf ../test .
    ```


- 安装mmdet3d

  - 在应用过patch的模型根目录下，克隆mmdet3d仓，并进入mmdetection3d目录编译

    ```
    git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
    cp -r ../mmdetection3d.patch mmdetection3d
    cd mmdetection3d
    git apply --reject mmdetection3d.patch
    pip install -v -e .
    cd ../
    ```

- 安装mmcv

  - 在应用过patch的模型根目录下，克隆mmcv仓，并进入mmcv目录安装编译

    ```
    git clone -b 1.x https://github.com/open-mmlab/mmcv
    cp -r ../mmcv.patch mmcv
    cd mmcv
    git apply --reject mmcv.patch
    MMCV_WITH_OPS=1 pip install -e . -v
    cd ..
    ```

- 安装mmdet

  - 在应用过patch的模型根目录下，克隆mmdet仓，并进入mmdetection目录安装编译

    ```
    git clone -b v2.28.0 https://github.com/open-mmlab/mmdetection.git
    cp ../mmdetection.patch mmdetection
    cd mmdetection
    git apply --reject mmdetection.patch
    pip install -e .
    cd ../
    ```

- 安装Driving SDK加速库，安装master分支，具体方法参考[原仓](https://gitee.com/ascend/DrivingSDK)。

- 在应用过patch的模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖。

  ```
  pip install -r requirements.txt
  ```



### 准备数据集

- 根据原仓**Prepare Dataset**章节准备数据集，数据集目录及结构如下：

```
SurroundOcc
├── data/
│   ├── nuscenes/
│   ├── nuscenes_occ/
│   ├── nuscenes_infos_train.pkl
│   ├── nuscenes_infos_val.pkl
```

> **说明：**  
> 该数据集的训练过程脚本只作为一种参考示例。      

### 准备预训练权重

- 根据原仓Installation章节下载预训练权重r101_dcn_fcos3d_pretrain.pth，并放在克隆的模型根目录ckpts下。

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
| 芯片          | 卡数 | global batch size | Precision | epoch |  IoU   |  mIoU  | 性能-单步迭代耗时(ms) |
| ------------- | :--: | :---------------: | :-------: | :---: | :----: | :----: | :-------------------: |
| 竞品A           |  8p  |         8         |   fp32    |  12   | 0.3163 | 0.1999 |         1028          |
| Atlas 800T A2 |  8p  |         8         |   fp32    |  12   | 0.3114 | 0.1995 |         1054          |


# 变更说明

2024.05.30：首次发布

2024.10.30：性能优化

2025.02.08：依赖仓实现patch安装

# FAQ

无