# StarVLA for PyTorch

## 目录

- [StarVLA for PyTorch](#starvla-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
- [准备权重与数据集](#准备权重与数据集)
  - [获取预训练权重](#获取预训练权重)
  - [准备数据集](#准备数据集)
- [快速开始](#快速开始)
  - [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)
  - [变更](#变更)
  - [FAQ](#faq)

# 简介

## 模型介绍

StarVLA 是一个各组件直观可分离的视觉语言模型 (Vision-Language-Action, VLA) 框架，仓内适配基于 QwenOFT 架构，以Qwen3-VL-4B-Instruct为基础模型，并进行微调以支持连续动作的生成。

- 参考实现：<https://github.com/starVLA/starVLA>

- 适配昇腾 AI 处理器的实现：<https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/StarVLA>

- Commit ID：`261ba5dd4313bbb910a9caeea07433c3d7b1aa34`

## 支持任务列表

本仓已经支持以下模型任务类型。如下列表中Released为Y的表示已经过测试验证，N的表示开发自验通过。

|    模型     | 任务列表 | 是否支持 | Released |
| :---------: | :------: | :------: | :------: |
|   StarVLA   |  SFT训练  |    ✔    |    N     |

# 准备训练环境

## 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|     软件类型      | 首次支持版本 |
| :---------------: | :------: |
| FrameworkPTAdapter | 26.0.0  |
|       CANN        |  9.0.0  |

## 安装模型环境

当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

**表 2**  版本支持表

|      三方库       |  支持版本  |
|:--------------:|:------:|
|    Python      | 3.10 |
|    PyTorch     | 2.7.1 |

0. 激活 CANN 环境

1. 创建环境

    参考原仓下载 Driving SDK 加速库：<https://gitcode.com/Ascend/DrivingSDK>

    创建conda环境

    ```sh
    conda create -n starvla python=3.10
    conda activate starvla
    cd ./DrivingSDK/model_examples/StarVLA
    ```

2. 准备模型源码，应用 Patch 并安装依赖

    在 StarVLA 目录下，克隆原始仓，切换至指定 commit 并应用 patch

    ```sh
    git clone https://github.com/starVLA/starVLA.git
    cd starVLA
    git checkout 261ba5dd4313bbb910a9caeea07433c3d7b1aa34
    cp -f ../starvla.patch ./
    git apply --reject starvla.patch
    pip install -r requirements.txt
    pip install -e .
    cp -f ../test/train* ./
    cd ..
    ```

3. 安装ffmpeg

    - 推荐基于conda安装

    ```sh
    # 安装ffmpeg
    conda install -c conda-forge ffmpeg=4.4.2
    ```

    - 若采用源码安装，则步骤如下

    ```sh
    # 下载源码
    wget https://ffmpeg.org/releases/ffmpeg-4.4.2.tar.bz2
    tar -xvf ffmpeg-4.4.2.tar.bz2
    cd ffmpeg-4.4.2
    # 执行此步时环境中可能需要手动下载部分依赖包
    ./configure --enable-shared --prefix=/usr/local/ffmpeg
    make -j 64
    make install
    cd ..

    # 编辑全局配置文件
    vim /etc/profile.d/ffmpeg.sh

    # 添加以下内容
    export PATH="/usr/local/ffmpeg/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/ffmpeg/lib:$LD_LIBRARY_PATH"

    # 使配置立即生效
    source /etc/profile
    # 运行命令后应正常输出相关配置等信息
    ffmpeg
    ```

4. 安装decord

    ```sh
    # 安装decord
    git clone --recursive https://github.com/dmlc/decord --depth 1
    cd decord
    mkdir build && cd build
    cmake ..  -DCMAKE_BUILD_TYPE=Release -DFFMPEG_DIR:PATH=$CONDA_PREFIX  # 源码安装ffmpeg时PATH需为"/usr/local/ffmpeg/"
    make
    # 编译whl包
    cd ../python
    python setup.py sdist bdist_wheel
    cd ../..
    pip install decord/python/dist/decord-0.6.0-cp310-cp310-linux_aarch64.whl
    ```

# 准备权重与数据集

## 获取预训练权重

下载 Qwen3-VL-4B-Instruct 权重：

```sh
mkdir -p starVLA/playground/Pretrained_models
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct
```

## 准备数据集

下载 RoboTwin-Clean 数据集并创建软链接：

```sh
mkdir -p starVLA/playground/Datasets
huggingface-cli download StarVLA/RoboTwin-Clean --repo-type dataset --local-dir starVLA/playground/Datasets/RoboTwin-Clean
ln -s RoboTwin-Clean starVLA/playground/Datasets/RoboTwin
```

> **说明**：训练脚本中的 `data_mix=robotwin` 对应 RoboTwin-Clean 数据集（50个任务子集），用于快速验证。完整数据集需下载 HuggingFace 上的 [StarVLA/RoboTwin-Randomized](https://huggingface.co/datasets/StarVLA/RoboTwin-Randomized) 并配置 `data_mix=robotwin_all_50`。

# 快速开始

进入 starVLA 目录：

```sh
cd starVLA
```

* 单机8卡训练

```sh
bash train_8p.sh --num_processes=16 --max_train_steps=150000 --per_device_batch_size=4
```

* 单机8卡训练性能测试

```sh
bash train_performance_8p.sh --num_processes=16 --max_train_steps=1000 --per_device_batch_size=4
```

## 训练结果展示

**表 3**  训练结果展示表

|     芯片      | 卡数 | per device batch size | max steps | FPS | 部分任务评测平均结果 |
| :-----------: | :--: | :---------------: | :---: | :--------------------: | :--------------------: |
|     竞品A     |  8p  |         4       |  150000 |  49.61  |  87.22 |
| Atlas 800T A3 |  8p  |         4       |  150000 |  123.08  |  88.38 |

# 版本说明

## 变更

2026.5.29: 首次发布。

## FAQ

Q: 设置环境变量 `TORCH_HCCL_ZERO_COPY` 报错？

A：当前该环境变量以支持 A3 为主，A2 暂不建议开启。

Q: 在无法访问 Hugging Face hub 的情况下运行模型报错？

A: 若受网络限制无法访问 Hugging Face，可使用 ModelScope 下载对应模型与数据集。
