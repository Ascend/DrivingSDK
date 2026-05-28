# Lerobot 视频解码后端性能对比

# 目录

- [Lerobot 视频解码后端性能对比](#lerobot-视频解码后端性能对比)
- [目录](#目录)
- [简介](#简介)
  - [项目介绍](#项目介绍)
  - [代码结构](#代码结构)
- [准备环境](#准备环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装依赖环境](#安装依赖环境)
    - [安装 Conda 环境](#安装-conda-环境)
    - [安装 TorchCodec](#安装-torchcodec)
    - [安装 PyAV](#安装-pyav)
    - [安装 DVPP 后端](#安装-dvpp-后端)
    - [安装 LeRobot](#安装-lerobot)
    - [下载数据](#下载数据)
- [快速开始](#快速开始)
  - [单后端解码](#单后端解码)
    - [TorchCodec 解码器](#torchcodec-解码器)
    - [PyAV 解码器](#pyav-解码器)
    - [DVPP 解码器](#dvpp-解码器)
  - [性能对比测试](#性能对比测试)
    - [抽帧解码对比](#抽帧解码对比)
    - [全量解码对比](#全量解码对比)
- [版本说明](#版本说明)
  - [变更](#变更)
  - [FAQ](#faq)

# 简介

## 项目介绍

本项目用于对比三种视频解码后端在 LeRobot 框架下的性能表现，支持以下解码后端：

| 后端 | 说明 | 硬件平台 |
|------|------|----------|
| **TorchCodec** | Meta 开源 GPU 视频解码库，支持缓存加速 | CPU |
| **PyAV** | 基于 FFmpeg 的 Python 绑定 | CPU |
| **DVPP** | 基于昇腾 CANN 的硬件解码加速 | NPU |

支持两种解码模式：

| 模式 | 说明 |
|------|------|
| **抽帧解码** | 按指定时间戳提取若干关键帧 |
| **全量解码** | 解码视频的每一帧，测试完整视频加载性能 |

## 代码结构

```text
video_decoder/
├── video_backend.py           # 性能对比主入口（一键对比三种后端）
├── torchcodec_decoder.py      # TorchCodec 解码器独立脚本
├── pyav_decoder.py            # PyAV 解码器独立脚本
├── npu_decoder.py             # DVPP 解码器独立脚本
└── readme.md                  # 本说明文档
```

- 参考实现：

  ```shell
  url=https://github.com/meta-pytorch/torchcodec
  commit_id=v0.5.0
  ```

  ```shell
  url=https://github.com/huggingface/lerobot
  commit_id=b954337ac7c8db5ea592c0d59dfb435845d9d380
  ```

# 准备环境

## 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境。本仓已支持表1中软件版本。

**表 1** 昇腾软件版本支持表

|        软件类型        |   支持版本   |
|:------------------:|:--------:|
|       CANN         | 9.0.0  |
|    FrameworkPTAdapter | 26.1.0  |
|     torch_npu      | 2.7.1  |

## 安装依赖环境

当前项目支持的 PyTorch 版本和已知三方库依赖如下表所示。

**表 2** 版本支持表

|      三方库       |  支持版本  |
|:--------------:|:------:|
|    PyTorch     |  2.7.1   |
|   TorchCodec   |  v0.5.0  |
|   PyAV / FFmpeg |  10.0.0 / 4.4.2 |
|    LeRobot     |  v3.0  |

### 安装 Conda 环境

```shell
conda create -n vdecode python=3.11
conda activate vdecode
cd DrivingSDK/mx_driving/dataset/video_decoder
```

### 安装 TorchCodec

安装 pybind11 和 ffmpeg：

```shell
conda install -c conda-forge pybind11
conda install -c conda-forge ffmpeg=4.4.2
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
```

安装 TorchCodec：

```shell
git clone https://github.com/meta-pytorch/torchcodec.git
cd torchcodec
git checkout v0.5.0
pip install -e . --no-build-isolation
cd ..
```

### 安装 PyAV

PyAV 作为 LeRobot 的默认视频读取后端，随 LeRobot 安装时自动依赖。

### 安装 DVPP 后端

DVPP 解码依赖昇腾 CANN 环境变量及 torchvision_npu 插件。

```shell
# 下载 Torchvision Adapter 代码，进入插件根目录
git clone https://gitcode.com/ascend/vision.git vision_npu
cd vision_npu
# 安装依赖库
pip3 install -r requirement.txt
# 编包
python setup.py bdist_wheel
# 安装 torchvision_npu
cd dist
pip install torchvision_npu-*.whl
cd ../..
```

### 安装 LeRobot

```shell
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout b954337ac7c8db5ea592c0d59dfb435845d9d380
cp ../video_utils.py src/lerobot/datasets/video_utils.py
pip install -e .
cd ..
```

# 下载数据

下载[koch_test数据集](https://huggingface.co/datasets/jianqiang03/koch_test/tree/main)，将数据集中视频（koch_test/videos/observation.images.laptop/chunk-000.mp4）的绝对路径记作 `/path/to/video.mp4`

# 快速开始

## 单后端解码

每个解码后端提供独立的 Python 脚本，支持三种运行模式：

| 模式 | 参数 | 说明 |
|------|------|------|
| 单次解码 | `--timestamps 0.0 0.5 1.0` | 单次解码指定时间戳的帧 |
| 抽帧性能测试 | `--test` | 多次迭代测试抽帧解码性能 |
| 全量解码测试 | `--full_decode` | 多次迭代测试全量解码性能 |

### TorchCodec 解码器

```shell
# 单次解码
python torchcodec_decoder.py --video_path /path/to/video.mp4 --timestamps 0.0 0.5 1.0

# 抽帧性能测试（支持缓存开关）
python torchcodec_decoder.py --video_path /path/to/video.mp4 --test --num_iterations 100
python torchcodec_decoder.py --video_path /path/to/video.mp4 --test --no_cache

# 全量解码测试
python torchcodec_decoder.py --video_path /path/to/video.mp4 --full_decode --num_iterations 5
```

脚本参数说明：

```text
--video_path          //视频文件路径
--timestamps          //要解码的时间戳列表（秒）
--tolerance_s         //时间容差（秒），默认 1e-4
--num_iterations      //性能测试迭代次数，默认 100
--no_cache            //关闭 TorchCodec 解码器缓存
--test                //运行抽帧性能测试模式
--full_decode         //运行全量解码测试模式
```

### PyAV 解码器

```shell
# 单次解码
python pyav_decoder.py --video_path /path/to/video.mp4 --timestamps 0.0 0.5 1.0

# 抽帧性能测试
python pyav_decoder.py --video_path /path/to/video.mp4 --test --num_iterations 100

# 全量解码测试
python pyav_decoder.py --video_path /path/to/video.mp4 --full_decode --num_iterations 5
```

### DVPP 解码器

```shell
# 单次解码（需先 source CANN 环境变量）
python npu_decoder.py --video_path /path/to/video.mp4 --timestamps 0.0 0.5 1.0

# 抽帧性能测试
python npu_decoder.py --video_path /path/to/video.mp4 --test --num_iterations 100

# 全量解码测试
python npu_decoder.py --video_path /path/to/video.mp4 --full_decode --num_iterations 2
```

## 性能对比测试

`video_backend.py` 提供一键对比三种后端性能的能力。

### 抽帧解码对比

```shell
python video_backend.py \
    --video_path /path/to/video.mp4 \
    --num_iterations 100 \
    --timestamps 0.0 0.5 1.0 1.5 2.0
```

输出示例：

```text
================================================================================
2026-05-20 00:38:56,383 - INFO - 抽帧解码性能对比汇总
2026-05-20 00:38:56,383 - INFO - ===============================================

Backend                        Avg (ms)     Min (ms)     Max (ms)     Speedup
--------------------------------------------------------------------------------
pyav                           156.22       144.74       172.34       1.00x
torchcodec (cache)             48.21        46.17        50.67        3.24x
torchcodec (no cache)          47.05        42.83        49.61        3.32x
dvpp                           12537.84     12108.26     13065.01     0.01x
================================================================================
```

### 全量解码对比

```shell
python video_backend.py \
    --video_path /path/to/video.mp4 \
    --full_decode \
    --num_iterations 2
```

输出示例：

```text
====================================================================================================
2026-05-19 14:32:19,869 - INFO - 全量解码性能对比汇总
2026-05-19 14:32:19,869 - INFO - ===================================================================

Backend                   Total Frames  Avg (s)    Min (s)    Max (s)    Decode FPS   Speedup
----------------------------------------------------------------------------------------------------
pyav                      34054         189.63     188.23     191.04     179.6        1.00x
torchcodec (cache)        34054         220.76     219.30     226.22     152.9        0.85x
torchcodec (no cache)     34054         221.91     221.50     222.31     153.5        0.85x
dvpp                      34054         227.23     226.75     227.70     149.9        0.83x
====================================================================================================
```

# 版本说明

## 变更

2026.05.19: 首次发布。支持 TorchCodec / PyAV / DVPP 三种视频解码后端，支持抽帧解码和全量解码两种模式，支持 `video_backend.py` 一键性能对比。

## FAQ

Q: 运行时提示 `ModuleNotFoundError: No module named 'torchcodec'`

A: 请确认已按照[安装 TorchCodec](#安装-torchcodec)步骤完成安装，并确保 PyTorch 版本兼容（2.1.0 或 2.7.1）。

Q: 运行时提示 `ModuleNotFoundError: No module named 'lerobot'`

A: 请确认已按照[安装 LeRobot](#安装-lerobot)步骤完成安装并切换到正确的 commit。

Q: DVPP 解码器提示 `torch_npu` 未找到

A: 请确保已正确安装昇腾 CANN 环境并 `source set_env.sh`，同时确认 `torch_npu` 已安装。可使用 `pip list | grep torch_npu` 检查。

Q: DVPP 全量解码测试失败或返回 None

A: DVPP 全量解码依赖 TorchCodec 获取视频元数据（帧数、FPS），请确保 TorchCodec 已安装且可用。
