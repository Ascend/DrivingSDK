# SalsaNext for PyTorch

## 目录

- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [SalsaNext](#SalsaNext)
  - [准备训练环境](#准备训练环境)
    - [昇腾环境说明](#昇腾环境说明)
    - [准备源代码](#准备源代码)
    - [模型数据准备](#模型数据准备)
    - [安装模型环境](#安装模型环境)
  - [快速开始](#快速开始)
    - [开始训练](#开始训练)
    - [训练结果](#训练结果)
- [变更说明](#变更说明)
- [FAQ](#FAQ)

# 简介

## 模型介绍

SalsaNext采用编码器-解码器架构，其中编码器单元包含一组ResNet块，解码部分则结合了残差块上采样的特征。SalsaNext引入了一个新的上下文模块，用带有逐渐增加的感受野的新型残差膨胀卷积堆栈替换ResNet编码器块，并在解码器中添加了像素shuffle层。此外，SalsaNext将步长卷积改为平均池化，并应用中心辍学处理。为了直接优化Jaccard指数，将加权交叉熵损失与Lovasz-Softmax损失相结合。最后，通过贝叶斯处理计算云中每个点的先验（Epistemic）和随机（Aleatoric）不确定性。模型在Semantic-KITTI数据集上提供了全面的定量评估。

## 支持任务列表

本仓已经支持以下模型任务类型

|   模型   | 任务列表 | 是否支持 |
| :------: | :------: | :------: |
| SalsaNext |   训练   |    ✔     |

## 代码实现

- 参考实现：

```
url=https://github.com/TiagoCortinhal/SalsaNext
commit_id=7548c124b48f0259cdc40e98dfc3aeeadca6070c
```

- 适配昇腾 AI 处理器的实现：

```
url=https://gitee.com/ascend/DrivingSDK.git
code_path=model_examples/SalsaNext
```

# SalsaNext

## 准备训练环境

### 昇腾环境说明

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1** 昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 7.0.0 |
|       CANN        | 8.1.RC1  |

**表 2** 三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |  2.1.0   |

### 准备源代码
- 克隆代码仓并应用补丁。

```
git clone https://gitee.com/ascend/DrivingSDK.git -b master
git clone https://github.com/TiagoCortinhal/SalsaNext.git
cp -f {DrivingSDK_root_dir}/model_examples/SalsaNext/salsanext.patch SalsaNext
cp -rf {DrivingSDK_root_dir}/model_examples/SalsaNext/test SalsaNext
cp -rf {DrivingSDK_root_dir}/model_examples/SalsaNext/train SalsaNext
cd SalsaNext
git checkout 7548c124b48f0259cdc40e98dfc3aeeadca6070c
git apply --whitespace=fix salsanext.patch
```

### 模型数据准备

进入[semantic-kitti](https://semantic-kitti.org/dataset.html)官网，下载Semantic Segmentation and Panoptic Segmentation数据集。将数据集解压后放置或者链接到DrivingSDK/model_examples/SalsaNext/datasets路径下，数据集结构排布成如下格式：

- 文件夹结构

```
  dataset
    └──sequences
        ├──00
        |   ├── labels
        |   ├── velodyne
        |   ├── calib.txt
        |   ├── poses.txt
        |   └── times.txt
        ├──01
        |   ├── labels
        |   ├── velodyne
        |   ├── calib.txt
        |   ├── poses.txt
        |   └── times.txt
        ...
        └──21
            ├── labels
            ├── velodyne
            ├── calib.txt
            ├── poses.txt
            └── times.txt
```
### 安装模型环境

0. 准备容器

   拉取镜像
   ```
   wget --no-check-certificate https://cmc-szver-artifactory.cmc.tools.huawei.com/artifactory/cmc-sz-inner/FrameworkPTAdapter/FrameworkPTAdapter%207/FrameworkPTAdapter%207.0.RC1.B030/DrivingSDK/DrivingSDK_7.0.RC1.tar
   ```

   创建镜像
   ```
   docker load -i DrivingSDK_7.0.RC1.tar
   ```

   在脚本run_drivingsdk_docker.sh中添加模型和数据集路径映射，映射方式：
   ```
   -v /宿主机绝对路径:/容器内路径
   ```

   创建并进入容器
   ```
   cd {DrivingSDK_root_dir}/model_examples/SalsaNext/
   bash run_drivingsdk_docker.sh 7.0.RC1
   ```
1. 安装基础依赖

   容器内提供了torch1.11.0、torch2.1.0、torch2.3.0三种版本的conda基础环境，进入torch2.1.0的conda环境
   ```
   conda activate torch_2.1.0
   ```

   进入模型源码根目录
   ```
   cd {DrivingSDK_root_dir}/model_examples/SalsaNext/
   ```

   使用pip指令安装模型所需的其他代码库:
   pip install -r requirements.txt


## 快速开始

### 训练任务

本任务主要提供**单机**的**8卡**训练脚本。

### 开始训练

  进入模型根目录，

  ```
  cd /${Model_root_dir}/
  ```
  salsanext.yml为默认配置文件。

- 单机8卡性能

  ```
  # epoch = 20
  bash test/train_8p_performance.sh -d /数据集路径/ -a ./salsanext.yml -l ./
  ```

- 单机8卡精度

  ```
  # epoch = 150
  bash test/train_8p.sh -d /数据集路径/ -a ./salsanext.yml -l ./
  ```

- 训练脚本参数说明
  - d[String]：数据集的路径
  - a[String]：配置文件的路径
  - l[String]：主日志文件夹的路径
  - c[String]：要使用的device_id，该参数仅单卡时生效

### 训练结果

|  芯片      | 卡数 | global batch size |  Max epochs  | mIoU | FPS |
|:--------:|----|:------:|:------:|:----:|:----------:|
|   竞品A    | 8p | 192 | 150 | 0.577 | 241.6 |
| Atlas 800T A2 | 8p | 192 | 150 | 0.581 | 183.2 |

# 变更说明

2025.03.06：首次发布。

2025.04.08：修改性能测试脚本错误，刷新性能数据，优化loss函数提高训练速度

2025.04.22：补充global batch size数据，修复FPS计算错误

## FAQ
暂无。