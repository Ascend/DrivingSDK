# GR00T-N1.6 for PyTorch

## 目录

- [GR00T-N1.6 for PyTorch](#gr00t-n16-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
- [准备数据集](#准备数据集)
  - [获取预训练权重](#获取预训练权重)
  - [准备数据集](#准备数据集-1)
- [快速开始](#快速开始)
  - [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)
  - [变更](#变更)
  - [FAQ](#faq)

# 简介

## 模型介绍

Isaac GR00T-N1.6 为 GR00T-N1.5升级版

- 参考实现：<https://github.com/NVIDIA/Isaac-GR00T/tree/main>

- 适配昇腾 AI 处理器的实现：<https://gitcode.com/Ascend/DrivingSDK/tree/master/model_examples/GR00T-N1.6>

## 支持任务列表

本仓已经支持以下模型任务类型。如下列表中Released为Y的表示已经过测试验证，N的表示开发自验通过。

|    模型     | 任务列表 | 是否支持 | Released |
| :---------: | :------: | :------: | :------: |
| GR00T-N1.6 |   SFT训练   |    ✔     |    N     |

# 准备训练环境

## 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|     软件类型      | 首次支持版本 |
| :---------------: | :------: |
| FrameworkPTAdapter | 26.0.0  |
|       CANN        | 9.0.0  |

## 安装模型环境

当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

**表 2**  版本支持表

|      三方库       |  支持版本  |
|:--------------:|:------:|
|    Python      | 3.10 |
|    PyTorch     |  2.7.1   |

0. 激活 CANN 环境

1. 创建环境

    参考原仓下载 Driving SDK 加速库：<https://gitcode.com/Ascend/DrivingSDK>

    随后创建conda环境

    ```sh
    conda create -n gr00t python=3.10
    conda activate gr00t
    cd ./DrivingSDK/model_examples/GR00T-N1.6
    ```

2. 准备模型源码，安装gr00t

      在 GR00T-N1.6根目录下，克隆原始仓，替换其中部分代码并安装

      ```sh
      git clone https://github.com/NVIDIA/Isaac-GR00T
      cd Isaac-GR00T
      git checkout e29d8fc50b0e4745120ae3fb72447986fe638aa6
      cp -f ../gr00t_n1d6.patch ./
      git apply --reject gr00t_n1d6.patch
      pip install -e .
      cp -f ../patch.py ./gr00t/utils/
      cp -f ../test/train* ./
    ```

3. 安装ffmpeg与torchcodec

    ```sh
    # 安装ffmpeg
    conda install -c conda-forge ffmpeg=4.4.2
    export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH

    # 安装torchcodec
    conda install -c conda-forge pybind11
    git clone https://github.com/meta-pytorch/torchcodec.git
    cd torchcodec
    git checkout v0.5.0
    pip install -e . --no-build-isolation
    ```

# 准备数据集

## 获取预训练权重

下载权重至Isaac-GR00T/GR00T-N1.6-3B，Huggingface链接: [GR00T-N1.6-3B](https://huggingface.co/nvidia/GR00T-N1.6-3B)

```sh
pip install huggingface-hub
hf download nvidia/GR00T-N1.6-3B --local-dir ./GR00T-N1.6-3B
```

## 准备数据集
以LIBERO 10微调为例，安装数据集
```sh
huggingface-cli download \
    --repo-type dataset IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot \
    --local-dir examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/

cp -r examples/LIBERO/modality.json examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/meta/
```

# 快速开始

* 单机8卡训练

需先进入Isaac-GR00T目录
```sh
cd Isaac-GR00T
```

**训练脚本**
```sh
bash train_8p.sh --num_gpus=8 --global_batch_size=640 --max_steps=20000 --dataset_path=examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/ --base_model_path=./GR00T-N1.6-3B
```

**性能测试脚本**
```sh
bash train_performance_8p.sh --num_gpus=8 --global_batch_size=640 --max_steps=1000 --dataset_path=examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/ --base_model_path=./GR00T-N1.6-3B
```

## 训练结果展示

**表 3**  训练结果展示表

|     芯片      | 卡数 | global batch size | max steps | Final loss | FPS  |
| :-----------: | :--: | :---------------: | :---: | :--------------------: | :--------------------|
|     竞品A     |  8p  |         640       |  20000 |  0.0084 |  457  |    
| Atlas 800T A2 |  8p  |         640       |  20000 |  0.0082 |  449  | 

# 版本说明

## 变更

2026.4.3: 首次发布。

## FAQ

Q: 在无法访问 Hugging Face hub 的情况下运行模型报错？

A: 用户可以前往官网或使用 Hugging Face 镜像源在有网络的情况下自主下载。

Q: 若运行过程中出现torchcodec相关报错，如decoder等？

A：可能是受到环境内系统原有ffmpeg的影响，需进入`/usr/local/`目录下，将ffmpeg目录更名（如`mv ffmpeg ffmpeg_bak`）来避免冲突，从而确保只依赖于conda版本，随后可重新编译安装torchcodec