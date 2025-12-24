# QCNet for PyTorch

## 目录

- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [QCNet](#QCNet)
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
url=https://gitcode.com/Ascend/DrivingSDK.git
code_path=model_examples/QCNet
```

# QCNet

## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1** 昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdapter | 6.0.0  |
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
    pip install torch==2.1.0 --no-deps
    pip install torch_npu==2.1.0 --no-deps
    ```

2. 拉取QCNet模型源代码
    ```
    git clone https://github.com/ZikangZhou/QCNet.git && cd QCNet
    git checkout 55cacb418cbbce3753119c1f157360e66993d0d0
    git apply ../patch/qcnet.patch
    pip install -r requirements.txt --no-deps
    cd ..
    ```

3. 安装pytorch_lightening
    ```
    git clone --branch 2.3.3 https://github.com/Lightning-AI/pytorch-lightning.git
    cd pytorch-lightning/
    git checkout cf348673eda662cc2e9aa71a72a19b8774f85718
    git apply ../patch/lightning.patch
    pip install -e ./ --no-deps
    cd ..
    ```

4. 安装 torch_geometric, torch_scatter

    ```
    git clone https://github.com/pyg-team/pytorch_geometric.git -b version_2_3_1
    cd pytorch_geometric
    git checkout 6b9db372d221c3e0dca773994084461a83e5af08
    git apply ../patch/torch_geometric.patch
    pip install -e ./ --no-deps
    cd ..

    git clone https://github.com/rusty1s/pytorch_scatter.git -b 2.1.0
    cd pytorch_scatter
    pip install -e ./ --no-deps
    cd ..
    ```

5. 安装 tcmalloc 高效内存资源分配库
    安装tcmalloc（适用OS: __openEuler__）
    ```shell
    mkdir gperftools && cd gperftools
    wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.16/gperftools-2.16.tar.gz
    tar -zvxf gperftools-2.16.tar.gz
    cd gperftools-2.16
    ./configure
    make
    make install
    export LD_PRELOAD=/usr/local/lib/libtcmalloc.so.4
    cd ..
    ```
    注意：需要安装OS对应tcmalloc版本（以下以 __Ubuntu__ 为例）
    ```shell
    # 安装autoconf和libtool
    apt-get update
    apt install autoconf
    apt install libtool
    git clone https://github.com/libunwind/libunwind.git
    cd libunwind
    autoreconf -i
    ./configure --prefix=/usr/local
    make -j128
    make install
    cd ..

    # 安装tcmalloc
    wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.16/gperftools-2.16.tar.gz
    tar -xf gperftools-2.16.tar.gz && cd gperftools-2.16
    ./configure --prefix=/usr/local/lib --with-tcmalloc-pagesize=64
    make -j128
    make install
    export LD_PRELOAD="$LD_PRELOAD:/usr/local/lib/lib/libtcmalloc.so"
    ```

6. 安装 DrivingSDK 加速库

   安装方法参考[官方文档](https://gitcode.com/Ascend/DrivingSDK)。


### 模型数据准备
方式一：自动下载数据

直接运行训练脚本，可以自动下载数据集到脚本中--root指向的默认路径'/path/to/datasets'下，并自动进行数据预处理，该路径可以自行修改。

方式二：手动下载数据

进入[Argoverse 2](https://www.argoverse.org/av2.html)官网，下载Argoverse 2 Motion Forecasting Dataset数据集。将数据集放置或者链接到任意路径下，数据集结构排布成如下格式：

- 文件夹结构

```
  datasets
    ├── train.tar
    ├── val.tar
    └── test.tar
```

- 数据预处理

    当数据集的压缩包已经放置于datasets路径下后，自行修改训练脚本中--root指向的路径'/path/to/datasets'为实际datasets的存放路径，pytorch-lightning框架会在第一次执行训练脚本时，自动开始数据预处理过程，处理总时长大约3小时。


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

|     芯片      | 卡数 | global batch size | epoch | minFDE | minADE | 性能-单步迭代耗时(s) |  FPS |
| :-----------: | :--: | :---------------: | :---: | :--------------------: | :--------------------: | :--------------: | :--------------: |
|     竞品A     |  8p  |         32         |  64   |         1.259          |         0.721          |       0.34         | 94.11 |
| Atlas 800T A2 |  8p  |         32         |  64   |         1.250          |         0.723          |       0.425         | 75.29 |

# 变更说明

2024.2.10：首次发布
2025.8.2：新增数据加载优化，npu的radius算子替换、npu_index_select算子替换、graph_softmax算子替换等

# FAQ

无