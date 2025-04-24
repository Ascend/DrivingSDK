# GameFormer for PyTorch

## 目录

-   [简介](#简介)
    - [模型介绍](#模型介绍)
    - [支持任务列表](#支持任务列表)
    - [代码实现](#代码实现)
-   [GameFormer](#GameFormer)
    - [准备训练环境](#准备训练环境)
    - [快速开始](#快速开始)
-   [变更说明](#变更说明)
-   [FAQ](#FAQ)

# 简介

## 模型介绍

在复杂的现实环境中运行的自动驾驶车辆需要准确预测交通参与者之间的交互行为。GameFormer模型结合了一个Transformer编码器，以及一个分层Transformer解码器结构，来有效地模拟场景元素之间的关系。在每个解码层级，除了共享的环境上下文之外，解码器还利用前一级别的预测结果来迭代地完善交互过程。该模型在WOMD数据集上验证了有效性，取得了领先的性能。本仓库将GameFormer模型迁移到昇腾NPU，并进行了性能优化。

## 支持任务列表

本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| GameFormer |   训练   |    ✔     |

## 代码实现

- 参考实现：
    ```
    url=https://github.com/MCZhi/GameFormer
    commit_id=fcb0d4a0f5cbbcecf69f9b9796366d6f5f2ce128
    ```

- 适配昇腾 AI 处理器的实现：
    ```
    url=https://gitee.com/ascend/DrivingSDK.git
    code_path=model_examples/GameFormer
    ```

# GameFormer

## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 7.0.0 |
|       CANN        | 8.1.RC1 |


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

1. 参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》安装 2.1.0 版本的 PyTorch 框架和 torch_npu 插件，并安装其它依赖项。
    ```
    cd DrivingSDK/model_examples/GameFormer
    conda create -n GameFormer python=3.9
    conda activate GameFormer
    pip install torch==2.1.0
    pip install torch_npu==2.1.0
    pip install -r requirements.txt
    ```

2. 安装waymo-open-dataset库：

    对于x86架构Linux系统：
    ```
    pip install waymo-open-dataset-tf-2-11-0==1.5.0
    ```
    对于arm64架构Linux系统，waymo官方并没有提供预先编译好whl包。为了方便用户使用，我们提供arm64系统编译的whl包，可以直接在华为云OBS上进行下载。
    ```
    wget --no-check-certificate https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/DrivingSDK/packages/waymo_open_dataset_tf_2.11.0-1.5.0-py3-none-any.whl
    pip install -r requirements_waymo.txt
    pip install waymo_open_dataset_tf_2.11.0-1.5.0-py3-none-any.whl
    ```

3. 拉取GamFormer模型代码仓库：
    ```
    git clone https://github.com/MCZhi/GameFormer.git
    cd GameFormer
    git checkout fcb0d4a0f5cbbcecf69f9b9796366d6f5f2ce128
    git apply ../patch/gameformer.patch
    cd ..
    ```

4. 安装 tcmalloc 高效内存资源分配库
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

### 准备数据集

1. 在[Waymo Open Motion Dataset (WOMD)数据集](https://waymo.com/open/download/)网站，下载v1.1版本的scenario/train以及scenario/validation中的数据，并将数据集目录结构排布成如下格式：
    ```
    ~/waymo
    └── motion
        ├── training
        │   ├── training.tfrecord-00000-of-01000
        │   ├── training.tfrecord-00001-of-01000
        │   ├── ...
        │   ├── training.tfrecord-00997-of-01000
        │   ├── training.tfrecord-00998-of-01000
        │   └── training.tfrecord-00999-of-01000
        └── validation
            ├── validation.tfrecord-00000-of-00150
            ├── validation.tfrecord-00001-of-00150
            ├── ...
            ├── validation.tfrecord-00148-of-00150
            ├── validation.tfrecord-00147-of-00150
            └── validation.tfrecord-00149-of-00150

    ```
2. 数据预处理
    ```
    cd GameFormer/interaction_prediction

    python data_process.py \
        --load_path waymo/motion/training \
        --save_path waymo/motion/training_processed \
        --use_multiprocessing \
        --processes=8

    python data_process.py \
        --load_path waymo/motion/validation \
        --save_path waymo/motion/validation_processed \
        --use_multiprocessing \
        --processes=8
    ```
    可以通过--processes调节同时进行数据预处理的线程数量，提升预处理效率。

## 快速开始
本任务主要提供**单机8卡**的训练脚本。
### 开始训练

- 在模型根目录下，运行训练脚本（请在脚本内自行修改训练集和测试集的路径，输出的日志位于GameFormer/interaction_prediction/train_log路径下）。
    ```
    # 8卡性能 (1 epoch)
    bash script/train_interaction_performance.sh 8
    # 8卡精度 (30 epoch)
    bash script/train_interaction.sh 8
    ```

### 训练结果
| 芯片           | 卡数 | global batch size | Precision | epoch | ADE | FDE | 性能-单步迭代耗时(ms) |
| ------------- | :--: | :------------: | :-------: | :---: | :----: | :----: | :-------------------: |
| 竞品A         |  8p  |  4096  |   fp32    |  30   | 1.47 | 2.82 |         640          |
| Atlas 800T A2 |  8p  |   4096 |   fp32    |  30   | 1.47 | 2.81 |        702          |


# 变更说明

2025.01.22：首次发布。

2025.02.17：BUG修复。

2025.04.24：增加global batch size数据
# FAQ

1. Openexr包编译安装失败？

在某些操作系统里，可能由于缺少依赖库的原因，pip install openexr执行失败，解决方案是升级gcc、g++、cmake版本，并且安装OpenEXR和OpenEXR-devel库。EulerOS/CentOS系统，执行以下命令：
    ```
    sudo yum makecache
    sudo yum install gcc gcc-c++ cmake
    sudo yum install OpenEXR
    sudo yum install OpenEXR-devel
    ```

    对于Ubuntu系统，执行以下命令：

    ```
    sudo apt-get update
    sudo apt install build-essential
    sudo apt install cmake
    sudo apt install openexr libopenexr-dev
    ```
    此外，如果在编译过程中遇到git网络问题，可以尝试以下命令：
    ```
    git config --global http.sslVerify false
    git config --global https.sslVerify false
    export GIT_SSL_NO_VERIFY=1
    pip install -r requirements_waymo.txt
    ```

2. libc.so.6: version 'GLIBC_xxx' not found问题。

这是由于操作系统的GLIBC版本过低，waymo-open-dataset库需要GLIBC的版本在2.31以上。可以通过**ldd --version**命令查看GLIBC版本，若小于2.31需升级系统的GLIBC库。