# Pi-0.5

## 目录
- [概述](#概述)
    - [模型介绍](#模型介绍)
    - [支持任务列表](#支持任务列表)
    - [代码实现](#代码实现)
- [前期准备](#前期准备)
    - [环境准备](#环境准备)
    - [数据准备](#数据准备)
    - [权重准备](#权重准备)
- [快速开始](#快速开始)
    - [执行训练](#执行训练)
    - [训练结果](#训练结果)
-   [变更说明](#变更说明)
-   [FAQ](#faq)

## 概述
本仓提供 pi0.5 模型的昇腾适配。

### 模型介绍

具身智能模型 pi0.5 是一款视觉 - 语言 - 动作 (VLA) 模型，通过多源数据融合和分层推理实现开放世界泛化能力，能在完全陌生环境中执行复杂任务并持续操作 10-15 分钟。
该模型采用 "双系统" 架构，高层决策与底层执行协同，能理解任务语义并拆解复杂流程，已在家庭服务、工业自动化等领域展示应用潜力。
作为 pi0 的升级版，pi0.5 通过异构数据协同训练显著提升泛化性能，可在训练时未见过的场景中保持与特定环境训练模型相当的执行效果。

### 支持任务列表

本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| Pi-0.5 |   训练   |    ✔     |

### 代码实现

- 参考实现：
    ```
    url=https://github.com/huggingface/lerobot.git
    commit_id=b954337ac7c8db5ea592c0d59dfb435845d9d380
    ```

- 适配昇腾 AI 处理器的实现：
    ```
    url=https://gitcode.com/Ascend/DrivingSDK.git
    code_path=model_examples/Pi-0.5
    ```

## 前期准备

### 环境准备

#### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdapter | 7.2.0 |
|       CANN        | 8.3.RC1  |


#### 安装模型环境

**表 2**  三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| Python | 3.10 |
| PyTorch |   2.7.1  |


0. 激活 CANN 环境

1. 创建conda环境
    ```shell
    conda create -n pi05 python=3.10
    conda activate pi05
    ```

2. 拉取 DrivingSDK 代码仓

    ```shell
    git clone https://gitcode.com/Ascend/DrivingSDK.git -b branch_v7.3.0
    cd DrivingSDK/model_examples/Pi-0.5
    ```

3. 安装 Pi-0.5
    ```shell
    git clone https://github.com/huggingface/lerobot.git
    cd lerobot
    git checkout b954337ac7c8db5ea592c0d59dfb435845d9d380
    cp -f ../pi05.patch .
    cp -rf ../test/ .
    git apply pi05.patch
    pip install -e .
    cd ..
    ```

4. 安装 transformers
    ```shell
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    git checkout fix/lerobot_openpi
    git checkout dcddb970176382c0fcf4521b0c0e6fc15894dfe0
    pip install -e .
    cd ..
    ```

5. 根据[Mindspeed仓](https://gitcode.com/Ascend/MindSpeed)安装Mindspeed组件，如：
    ```shell
    git clone https://gitcode.com/Ascend/MindSpeed.git
    pip install -e MindSpeed
    cd lerobot
    ```


### 数据准备

下载[koch_test数据集](https://huggingface.co/datasets/jianqiang03/koch_test/tree/main)，将数据集的绝对路径记作 dataset_path，转换数据集格式：
```shell
python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 --repo-id={dataset_path}
```
### 权重准备
下载[Pi-0.5预训练权重](https://huggingface.co/lerobot/pi05_base)，将权重的绝对路径记作 pi05_weights

## 快速开始
本任务主要提供**A3单机8卡**的训练脚本。

### 执行训练
进入模型代码目录 `path/to/lerobot`
- 单机性能

    ```shell
    bash test/train_performance.sh {dataset_path} {pi05_weights}
    ```

- 单机精度

    ```shell
    bash test/train_full.sh {dataset_path} {pi05_weights}
    ```
    ---
    参数说明
    | 参数| 可选/必选 | 说明 |
    |------|---------|------|
    |dataset_path | 必选| 数据路径|
    |pi05_weights | 必选| 权重路径|

### 训练结果
- A3单机8卡

    |  NAME       | Precision     |     iterations    |    global batchsize      |    training loss      |     FPS      |
    |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
    |  竞品 H   |      FP32    |        30k     |     512    |        0.031   |    1115    |
    |  Atlas 800T A3   |     FP32    |        30k     |      1024    |        0.034   |      2335    |
    ---
    字段说明
    |字段|说明|
    |-----|-----|
    |NAME|芯片类别|
    |Precision|训练精度|
    |iterations|训练迭代步数|
    |global batchsize|每次迭代的样本总数|
    |training loss|训练结束时的损失|
    |FPS|平均每秒处理的样本总数|

## 变更说明

2025.11.18：首次发布。

## FAQ
Q: 在无网络或者有防火墙的网络下，模型无法自动下载paligemma的权重怎么办？

A: 可自行下载[paligemma权重](https://huggingface.co/google/paligemma-3b-pt-224)，将权重路径记作 paligemma_weights。再执行以下命令，使用脚本将本地权重路径替换进模型代码：

```shell
bash test/paligemma_weights_mod.sh {paligemma_weights}
```