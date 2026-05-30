# LingBot-VLA

## 目录

- [LingBot-VLA](#lingbot-vla)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
- [准备模型权重与数据集](#准备模型权重与数据集)
  - [模型权重：](#模型权重)
  - [数据集：](#数据集)
- [快速开始](#快速开始)
  - [训练任务](#训练任务)
    - [执行训练](#执行训练)
    - [训练结果展示](#训练结果展示)
- [变更说明](#变更说明)
- [FAQ](#faq)

# 简介

## 模型介绍

LingBot-VLA 是面向机器人控制的 Vision-Language-Action 基础模型，使用视觉观测、语言指令和机器人状态预测动作序列。原仓库提供 LingBot-VLA-4B 与 LingBot-VLA-4B-Depth 两类权重，并支持在 RoboTwin 2.0 等数据集上进行 post-training / fine-tuning。

## 支持任务列表

本仓已经支持以下模型任务类型。

| 模型 | 任务列表 | 是否支持 |
| :--: | :------: | :------: |
| LingBot-VLA | 微调 | ✔ |

## 代码实现

- 参考实现：

```shell
url=https://github.com/robbyant/lingbot-vla
```

# 准备训练环境

## 安装昇腾环境

请参考昇腾社区《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境。

**表 1** 昇腾软件版本支持表

| 软件类型 | 首次支持版本 |
| :------: | :----------: |
| FrameworkPTAdapter | 26.1.0 |
| CANN | 9.1.0 |

## 安装模型环境

当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

**表 2**  版本支持表

|      三方库       |  支持版本  |
|:--------------:|:------:|
|    Python      | 3.12.3 |
|    PyTorch     |  2.7.1   |

- 创建conda环境

```shell
conda create -n lingbotvla python=3.12 -y
conda activate lingbotvla
conda install -c conda-forge pybind11 cmake ninja
conda install -c conda-forge ffmpeg=6.1.2
```

- 安装依赖。

```shell
# Install Lerobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
pip install -e .
cd ..

# Install torchcodec
git clone https://github.com/meta-pytorch/torchcodec.git
git checkout v0.6.0
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
pkg-config --modversion libavformat
pip install -e . --no-build-isolation

# Install lingbot-vla
git clone https://github.com/robbyant/lingbot-vla.git
cd lingbot-vla
git checkout a57f084
git apply --reject --whitespace=fix ../lingbot.patch
git submodule update --remote --recursive
pip install -e .
pip install -r requirements.txt

# Install LingBot-Depth dependency
cd ./lingbotvla/models/vla/vision_models/lingbot-depth/
pip install -e . --no-deps
cd ../MoGe
pip install -e .
```

> 说明：安装过程中如果 PyTorch 版本被依赖覆盖，需要重新安装 PyTorch 与 torch_npu 版本。

# 准备模型权重与数据集

## 模型权重：

| 模型 | 用途 |
| :-- | :-- |
| LingBot-VLA-4B | 不含 depth 的 post-training |
| LingBot-VLA-4B-Depth | 含 depth distillation 的 post-training |
| Qwen2.5-VL-3B-Instruct | tokenizer 与 Qwen2.5-VL 初始化配置 |
| MoGe-2-vitb-normal | depth 版本训练需要 |
| LingBot-Depth | depth 版本训练需要 |

可使用仓库脚本下载 HuggingFace 权重：

```shell
python3 scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-4b --local_dir lingbot-vla-4b
python3 scripts/download_hf_model.py --repo_id Qwen/Qwen2.5-VL-3B-Instruct --local_dir Qwen2.5-VL-3B-Instruct
```

## 数据集：

请参考lingbot-vla仓库中的**RoboTwin 数据准备文档**，将处理后的 lerobot 格式数据路径传给训练脚本的 `--dataset_path` 参数。脚本中的默认参数以click_bell_aloha_repo任务为例。
(由于RoboTwin数据集生成需要渲染管线，A3暂不支持生成此数据集)

# 快速开始

## 训练任务

### 执行训练

在应用过 NPU 适配的模型根目录下执行以下命令。

- 复制执行脚本

```shell
cd lingbot-vla
cp -r ../test .
```

- 单机 16 卡训练

```shell
bash test/train_16p_full.sh
```

- 单机 16 卡训练性能测试

```shell
bash test/train_16p_perf.sh
```

脚本默认参数如下：

| 参数 | 默认值 | 说明 |
| :-- | :-- | :-- |
| `num_npus` | 16 | 单机 NPU 进程数 |
| `micro_batch_size` | 32 | 单卡 micro batch size |
| `global_batch_size` | 512 | 全局 batch size |
| `config_path` | `${ROOT_DIR}/configs/vla/robotwin_load20000h.yaml` | 训练配置文件 |
| `output_dir` | `${ROOT_DIR}/output` | 训练输出目录 |
| `perf_steps` | 300 | 性能脚本统计最后多少条 `StepTime` 记录 |

### 训练结果展示

**表 3**  训练结果展示表

|     芯片      | 卡数 | per device batch size | epoch | FPS |
| :-----------: | :--: | :---------------: | :---: | :--------------------: |
|     竞品H     |  8p  |         32       |  69 |  114.74  |
| Atlas 800T A3 |  8p  |         32       |  69 |  301.76 |

# 变更说明

2026.05.26：新增 LingBot-VLA NPU 适配说明。

# FAQ
