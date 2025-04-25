# QCNet for PyTorch

## 目录

- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [QCNet（在研版本）](#QCNet(在研版本))
  - [准备训练环境](#准备训练环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [安装模型环境](#安装模型环境)
    - [模型数据准备](#模型数据准备)
  - [快速开始](#快速开始)
    - [开始训练](#开始训练)
    - [训练结果](#训练结果)
- [变更说明](#变更说明)
- [FAQ](#FAQ)

# 简介

## 模型介绍

QCNet是一种用于轨迹预测的神经网络架构，旨在提高自动驾驶车辆在安全操作中的预测能力。该模型通过引入查询中心(scene encoding)范式，独立于全局时空坐标系统学习表示，以实现更快的推理速度。QCNet使用无锚点(anchor-free)查询生成轨迹提议，并采用基于锚点(anchor-based)的查询进一步细化这些提议，以处理预测中的多模态性和长期性问题。模型在Argoverse 1和Argoverse 2运动预测基准测试中排名第一，超越了所有其他方法。

## 支持任务列表

本仓已经支持以下模型任务类型

|   模型   | 任务列表 | 是否支持 |
| :------: | :------: | :------: |
| QCNet |   训练   |    ✔     |

## 代码实现

- 参考实现：

```
url=https://github.com/ZikangZhou/QCNet
commit_id=55cacb418cbbce3753119c1f157360e66993d0d0
```

- 适配昇腾 AI 处理器的实现：

```
url=https://gitee.com/ascend/DrivingSDK.git
code_path=model_examples/QCNet
```

# QCNet（在研版本）

## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1** 昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 6.0.0  |
|       CANN        | 8.0.0  |

### 安装模型环境

**表 2** 三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |  2.1.0   |

0. 激活 CANN 环境

   将 CANN 包目录记作 cann_root_dir，执行以下命令以激活环境

   ```
   source {cann_root_dir}/set_env.sh
   ```

1. 参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》安装 2.1.0 版本的 PyTorch 框架和 torch_npu 插件。
    ```
    conda create -n QCNet python=3.9.21
    conda activate QCNet
    pip install -r requirements.txt --no-deps
    pip install torch==2.1.0 --no-deps
    pip install torch_npu==2.1.0 --no-deps
    ```

2. 拉取QCNet模型源代码
    ```
    git clone https://github.com/ZikangZhou/QCNet.git && cd QCNet
    git checkout 55cacb418cbbce3753119c1f157360e66993d0d0
    git apply ../patch/qcnet.patch
    cd ..
    ```

3. 安装pytorch_lightening
    ```
    git clone https://github.com/Lightning-AI/pytorch-lightning.git -b builds/2.3.1
    cd pytorch-lightning/
    git checkout 8e39ef55142e3cf1878efee85cfbeb0ed0ce29b5
    git apply ../patch/lightning.patch
    pip install -e ./ --no-deps
    cd ..
    ```

4. 安装 torch_geometric, torch_cluster, torch_scatter

    ```
    git clone https://github.com/pyg-team/pytorch_geometric.git -b version_2_3_1
    cd pytorch_geometric
    git checkout 6b9db372d221c3e0dca773994084461a83e5af08
    git apply ../patch/torch_geometric.patch
    pip install -e ./ --no-deps
    cd ..

    git clone https://github.com/rusty1s/pytorch_cluster.git -b 1.6.1
    cd pytorch_cluster
    git checkout 84bbb7140e03df01b3bb388ba4df299328ea2dff
    git apply ../patch/torch_cluster.patch
    // 该仓库编译耗时较久，需要30分钟左右
    pip install -e ./ --no-deps
    cd ..

    git clone https://github.com/rusty1s/pytorch_scatter.git -b 2.1.0
    cd pytorch_scatter
    pip install -e ./ --no-deps
    cd ..
    ```

5. 安装 tcmalloc 高效内存资源分配库
    ```
    mkdir gperftools && cd gperftools
    wget --no-check-certificate https://github.com/gperftools/gperftools/releases/download/gperftools-2.16/gperftools-2.16.tar.gz
    tar -zvxf gperftools-2.16.tar.gz
    cd gperftools-2.16
    ./configure
    make
    make install
    export LD_PRELOAD=/usr/local/lib/libtcmalloc.so.4
    cd ..
    ```

5. 安装 DrivingSDK 加速库

   安装方法参考[官方文档](https://gitee.com/ascend/DrivingSDK/wikis/DrivingSDK)。


### 模型数据准备

进入[Argoverse 2](https://www.argoverse.org/av2.html)官网，下载Argoverse 2 Motion Forecasting Dataset数据集。将数据集放置或者链接到DrivingSDK/model_examples/QCNet/QCNet/datasets路径下，数据集结构排布成如下格式：

- 文件夹结构

```
  datasets
    ├── train.tar
    ├── val.tar
    └── test.tar
```

- 数据预处理

    当数据集的压缩包已经放置于dataset路径下，pytorch-lightening框架会在第一次执行训练脚本时，自动开始数据预处理过程，处理总时长大约3小时。


## 快速开始

### 训练任务

本任务主要提供**单机**的**8卡**训练脚本。

### 开始训练

在模型根目录下，运行训练脚本。

```  
cd model_examples/QCNet
```

- 单机8卡性能

  ```
  # epoch = 1
  bash script/train_performance.sh
  ```

- 单机8卡精度

  ```
  # epoch = 64
  bash script/train.sh
  ```

### 训练结果

|     芯片      | 卡数 | global batch size | epoch | minFDE | minADE | 性能-单步迭代耗时(s) |
| :-----------: | :--: | :---------------: | :---: | :--------------------: | :--------------------: | :--------------: |
|     竞品A     |  8p  |         32         |  64   |         1.259          |         0.721          |       0.34         |
| Atlas 800T A2 |  8p  |         32         |  64   |         1.259          |         0.719          |       0.55         |

# 变更说明

2024.2.10：首次发布

# FAQ

无