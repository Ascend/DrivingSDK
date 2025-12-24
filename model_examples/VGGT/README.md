# VGGT

## 目录

- [VGGT](#vggt)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [VGGT](#vggt-1)
  - [准备训练环境](#准备训练环境)
    - [安装环境](#安装环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [准备数据集](#准备数据集)
    - [准备预训练权重](#准备预训练权重)
  - [快速开始](#快速开始)
    - [训练任务](#训练任务)
      - [执行训练](#执行训练)
      - [单机八卡训练性能和loss，以tv场景，固定随机性，random\_aspect\_ratio为1，random\_image\_num为2，batch\_size为15](#单机八卡训练性能和loss以tv场景固定随机性random_aspect_ratio为1random_image_num为2batch_size为15)
- [变更说明](#变更说明)
- [FAQ](#faq)

# 简介

## 模型介绍

VGGT是一个大型前馈Transformer，具有最小的3D感应偏差，在大量3D注释数据上进行训练。它接受多达数百张图像，并在不到一秒的时间内一次性预测所有图像的相机、点图、深度图和点轨迹，这通常优于基于优化的替代方案，无需进一步处理。

## 支持任务列表
本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| VGGT |   训练   |    ✔     |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/facebookresearch/vggt
  commit_id=97bbde571faddde3ace3cfa7724a20448026c4c8
  ```
- 适配昇腾 AI 处理器的实现：

  ```
  url=https://gitcode.com/Ascend/DrivingSDK.git
  code_path=model_examples/VGGT
  ```

# VGGT
## 准备训练环境

### 安装环境

  **表 1**  三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |   2.1   |

### 安装昇腾环境

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本，python使用3.10版本。

  **表 2**  昇腾软件版本支持表

|     软件类型      | 首次支持版本 |
| :---------------: | :------: |
| FrameworkPTAdapter | 7.2.0  |
|       CANN        | 8.3.RC1 |

- 克隆代码仓到当前目录并使用patch文件

    ```
    git clone https://github.com/facebookresearch/vggt.git
    cd vggt
    git checkout 97bbde571faddde3ace3cfa7724a20448026c4c8
    cp -f ../VGGT_npu.patch .
    git apply --reject --whitespace=fix VGGT_npu.patch
    cp -r ../test training/
    ```


- 安装环境依赖

  - 在应用过patch的模型根目录下，安装需要的依赖

    ```
    pip install -r requirements.txt
    ```

  - 在应用过patch的模型根目录下，安装vggt
    ```
    pip install -e .
    ```


### 准备数据集

- 根据源仓readme中下载数据集co3D, 完整数据集5.5T,本仓中为实现快速验证，仅使用'tv'场景进行训练，如果使用其他场景，需要修改代码中对应部分，数据集分为co3D和co3D_annotations文件夹， 放在training文件夹下，数据集文件名称命名为co3D, 文件夹下为co3D和co3D_annotations，排列如下：
```
vggt
├── training/
│   ├── co3D/
│   │     ├── co3D
│   │     ├── co3D_annotations
```


### 准备预训练权重

- 参照源仓下载模型权重，可以放到应用过patch的模型根目录下

```
vggt
├── model.pt
```
## 快速开始

### 训练任务

#### 执行训练

  1. 在应用过patch的模型根目录下，执行以下指令进行训练。

      - 进入training文件夹

      ```
      cd training
      ```

     - 单机八卡性能

     ```
     bash test/train_8p_vggt_perf.sh --max-epochs 1
     ```

     - 单机八卡长跑


     ```
     bash test/train_8p_vggt_full.sh --max-epochs 7
     ```

#### 单机八卡训练性能和loss，以tv场景，固定随机性，random_aspect_ratio为1，random_image_num为2，batch_size为15
| 芯片          | 卡数 | gloabl batchsize | Loss | iteration time | FPS |
| ------------- | :--: | :--: | :-------: | :------------: |:---------: |
| 竞品A         |  8p | 120 |    0.0164  |     7.841     |  15.30 |
| Atlas 800T A2 |  8p | 120 |  0.0165    |      4.792      | 25.04 |


# 变更说明

2025.06.29：首次发布

2025.08.11: 支持训练，性能优化

# FAQ

无