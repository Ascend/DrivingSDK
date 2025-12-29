# Pi-0

## 目录

- [Pi-0](#pi-0)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [Pi-0](#pi-0-1)
  - [准备训练环境](#准备训练环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [安装模型环境](#安装模型环境)
    - [准备数据集与预训练权重](#准备数据集与预训练权重)
  - [快速开始](#快速开始)
    - [开始训练](#开始训练)
    - [训练结果](#训练结果)
- [变更说明](#变更说明)
- [FAQ](#faq)

# 简介

## 模型介绍

机器人学习在释放灵活、通用且灵巧的机器人系统潜力，以及解决人工智能领域核心问题方面前景广阔。然而，要实现现实世界中高效通用机器人学习系统，仍需克服数据、泛化性和鲁棒性等重大挑战。

本文探讨了如何通过通用机器人策略（即机器人基础模型）应对这些挑战，并设计适用于复杂灵巧任务的通用策略。本文提出了一种基于预训练视觉语言模型（VLM）的新型流匹配架构，以继承互联网规模的语义知识，并讨论了如何利用多平台（单臂、双臂及移动机械臂）的多样化数据集进行训练。

## 支持任务列表

本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| Pi-0 |   训练   |    ✔     |

## 代码实现

- 参考实现：
    ```
    url=https://github.com/huggingface/lerobot.git
    commit 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
    ```

- 适配昇腾 AI 处理器的实现：
    ```
    url=https://gitcode.com/Ascend/DrivingSDK.git
    code_path=model_examples/Pi-0
    ```

# Pi-0

## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|     软件类型      | 首次支持版本 |
| :---------------: | :------: |
| FrameworkPTAdapter | 7.1.0 |
|       CANN        | 8.2.RC1  |


### 安装模型环境

**表 2**  三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |   2.1.0   |


0. 激活 CANN 环境

1. 创建conda环境
  ```
  conda create -n pi0 python=3.10
  conda activate pi0
  ```

2. 安装 Pi-0
  ```
  git clone https://github.com/huggingface/lerobot.git
  cd lerobot
  git checkout 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
  cp -f ../pi0.patch .
  cp -rf ../test/ .
  git apply pi0.patch
  pip install -e '.[pi0]'
  ```

3. 根据[Mindspeed仓](https://gitcode.com/Ascend/MindSpeed)安装Mindspeed组件，如：
  ```
  git clone https://gitcode.com/Ascend/MindSpeed.git
  pip install -e MindSpeed
  ```

4. 安装ffmpeg
  ```
  conda install ffmpeg=7.1.1 -c conda-forge
  ```

5. 安装依赖软件
  ```
  cp ../requirements.txt .
  pip install -r requirements.txt
  ```

6. 安装Driving SDK

请参考昇腾[Driving SDK](https://gitcode.com/Ascend/DrivingSDK)代码仓说明编译安装Driving SDK


### 准备数据集与预训练权重

1. 下载[koch_test数据集](https://huggingface.co/datasets/danaaubakirova/koch_test/tree/main)，将数据集位置记作 dataset_path

2. 下载[pi-0预训练权重](https://huggingface.co/lerobot/pi0)，将权重路径记作 pi0_weights

## 快速开始
本任务主要提供**单机8卡**的训练脚本。
### 开始训练

- 单机8卡性能

  ```
  bash test/train_8p_performance.sh {dataset_path} {pi0_weights}
  ```

- 单机8卡精度

  ```
  bash test/train_8p.sh {dataset_path} {pi0_weights}
  ```


### 训练结果
- 单机8卡

|  NAME       | Precision     |     iterations    |    global_batch_size      |    mean training loss      |     FPS      |
|-------------|-------------------|-----------------|---------------|--------------|--------------|
|  8p-竞品A   |      Mixed    |        20k     |     96    |        0.003929   |    136.17     |
|  8p-Atlas 800T A2   |     Mixed    |        20k     |      96    |        0.003932   |      116.36    |


# 变更说明

2025.08.20：首次发布。

# FAQ
Q: 在无网络或者有防火墙的网络下，模型无法自动下载paligemma的权重，怎么办？

A: 可自行下载[paligemma权重](https://huggingface.co/google/paligemma-3b-pt-224)，将权重路径记作 paligemma_weights。再执行以下命令，使用脚本将本地权重路径替换进模型代码：

  ```
  bash test/paligemma_weights_mod.sh {paligemma_weights}
  ```