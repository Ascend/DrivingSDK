# Cosmos-Reason1 for PyTorch

## 目录

- [Cosmos-Reason1 for PyTorch](#cosmos-reason1-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [准备训练环境](#准备训练环境)
  - [安装昇腾环境](#安装昇腾环境)
  - [安装模型环境](#安装模型环境)
- [准备训练数据](#准备训练数据)
  - [准备模型权重](#准备模型权重)
  - [准备数据集](#准备数据集)
    - [SFT训练数据集](#sft训练数据集)
    - [RL训练数据集](#rl训练数据集)
- [快速开始](#快速开始)
  - [Cosmos-Reason1-7B](#cosmos-reason1-7b)
  - [训练结果展示](#训练结果展示)
- [版本说明](#版本说明)
  - [变更](#变更)
  - [FAQ](#faq)

# 简介

## 模型介绍

Cosmos-Reason1 是 NVIDIA Cosmos 世界基础模型（WFMs）生态中专注于物理 AI 推理的关键分支，是一个开源、可定制的7B参数推理视觉语言模型（VLM），专为物理 AI 和机器人技术设计。该模型能够像人类一样进行推理，利用先验知识、物理理解和常识来理解并在现实世界中行动，擅长导航物理世界多样化场景的长尾问题。

Cosmos-Reason1 基于 cosmos-rl 框架实现后训练，支持监督微调（SFT）和人类反馈强化学习（RLHF），用于增强模型的物理常识和具身推理能力。

## 支持任务列表

本仓已经支持以下模型任务类型。如下列表中Released为Y的表示已经过测试验证，N的表示开发自验通过。

|    模型     | 任务列表 | 是否支持 | Released |
| :---------: | :------: | :------: | :------: |
| Cosmos-Reason1-7B |   SFT训练 & RL训练   |    ✔     |    N     |

## 代码实现

- cosmos-rl框架参考实现：
  
  ```
  url=https://github.com/nvidia-cosmos/cosmos-rl
  commit_id=dbbf358b9341141eed98ca92f83dbe194cb0bd96 
  ```

- cosmos-reason1模型参考实现：
  ```
  url=https://github.com/nvidia-cosmos/cosmos-reason1
  commit_id=8743e1bbed09cb5555c74091f6de8048d9551f75 
  ```

- 适配昇腾 AI 处理器的实现：
    ```
    url=https://gitcode.com/Ascend/DrivingSDK.git
    code_path=model_examples/Cosmos-Reason1
    ```

# 准备训练环境

## 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0001.html)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|     软件类型      | 首次支持版本 |
| :---------------: | :------: |
| FrameworkPTAdapter | 7.3.0  |
|       CANN        | 8.5  |


## 安装模型环境

当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

**表 2**  版本支持表

|      三方库       |  首次支持版本  |
|:--------------:|:------:|
|    Python      | 3.10 |
|    PyTorch     |  2.7.1   |


0. 激活 CANN 环境
 
    注：需要安装和toolkit等包版本一致的nnal包

1. 安装Driving SDK
    
    请参考昇腾Driving SDK代码仓说明编译安装Driving SDK

2. 配置redis-server
    
    先用`which redis-server`查看环境中是否存在 redis-server，若无则需要从源码下载：
    ```
    cd /usr/bin
    wget https://download.redis.io/releases/redis-6.2.14.tar.gz
    tar xzf redis-6.2.14.tar.gz
    cd redis-6.2.14
    make
    make test
    make install
    ```

3. 准备cosmos-rl框架源码并安装

    在 model_examples/Cosmos-Reason1 目录下，克隆原始仓，使用patch文件替换其中部分代码并安装
    ```sh
    git clone https://github.com/nvidia-cosmos/cosmos-rl.git
    cd cosmos-rl
    git checkout dbbf358b9341141eed98ca92f83dbe194cb0bd96
    cp -f ../cosmos-rl.patch .
    git apply --reject --whitespace=fix cosmos-rl.patch
    pip install -r requirements.txt
    pip install -e .
    cd ..
    ```

4. 安装vllm和vllm-ascend v0.11.0

    ```sh
    git clone --depth 1 --branch v0.11.0 https://github.com/vllm-project/vllm
    cd vllm
    git checkout b8b302cde434df8c9289a2b465406b47ebab1c2d
    cp -f ../vllm.patch .
    git apply --reject --whitespace=fix vllm.patch
    VLLM_TARGET_DEVICE=empty pip install -v -e .
    cd ..
    pip install vllm-ascend==0.11.0
    ```
    注：安装过程中需要确认下当前torch和torch_npu版本是否被修改，若被修改则需重新安装 2.7.1 版本

5. 准备cosmos-reason1源码并安装cosmos-reason1-utils

    ```sh
    git clone https://github.com/nvidia-cosmos/cosmos-reason1.git
    cd cosmos-reason1
    git checkout 8743e1bbed09cb5555c74091f6de8048d9551f75
    cp -f ../cosmos-reason1.patch .
    git apply --reject --whitespace=fix cosmos-reason1.patch
    cd cosmos_reason1_utils/
    pip install -e .
    cd ../..
    ```

# 准备训练数据

1. 生成一个[Hugging Face](https://huggingface.co/settings/tokens)访问令牌，将访问令牌设置为 'Read' 权限

2. 使用该令牌登录Hugging Face
    
    ```sh
    huggingface-cli login
    ```

## 准备模型权重

下载Cosmos-Reason1预训练模型权重：

```sh
# 下载Cosmos-Reason1-7B模型
hf download nvidia/Cosmos-Reason1-7B --local-dir ./cosmos-reason1/examples/post_training/Cosmos-Reason1-7B
```

## 准备数据集

根据训练任务类型，准备相应的数据集：

### SFT训练数据集
```sh
# 下载Cosmos-Reason1-SFT-Dataset数据集
hf download nvidia/Cosmos-Reason1-SFT-Dataset --repo-type=dataset --local-dir ./cosmos-reason1/examples/post_training/datasets/Cosmos-Reason1-SFT-Dataset
```

### RL训练数据集
```sh
# 下载Cosmos-Reason1-RL-Dataset数据集
hf download nvidia/Cosmos-Reason1-RL-Dataset --repo-type=dataset --local-dir ./cosmos-reason1/examples/post_training/datasets/Cosmos-Reason1-RL-Dataset
```

# 快速开始

## Cosmos-Reason1-7B

训练过程涉及约200GB的模型和数据集文件，请确保 ~/.cache 目录具有足够的存储空间，或通过设置 `HF_HOME` 和 `COSMOS_CACHE` 环境变量指定的目录（例如`export HF_HOME={path_to_dir}`）

在 model_examples/Cosmos-Reason1 目录下
```sh
# 监督微调训练 (SFT)
bash test/train.sh --sft
# 强化学习训练 (RL)
bash test/train.sh --rl
```

## 训练结果展示

**表 3**  Cosmos-Reason1-7B SFT训练结果展示表

|     芯片      | 卡数 | global batch size | epoch | Final loss | Iteration time (s) |
| :-----------: | :--: | :---------------: | :---: | :--------------------: | :--------------------: |
|     竞品A     |  8p  |         256       |  1  | 0.378  | 14.52 |
| Atlas 800T A2 |  8p  |         256       |  1  |  0.380  | 16.57 |

**表 4**  Cosmos-Reason1-7B RL训练结果展示表

|     芯片      | 卡数 | global batch size | epoch | Final Reward | Iteration time (s) |
| :-----------: | :--: | :---------------: | :---: | :--------------------: | :--------------------: |
|     竞品A     |  8p  |         512       | 8  |  1.9688  | 25.16 |
| Atlas 800T A2 |  8p  |         512       | 8  |  1.9219  | 43.58  |



# 版本说明

## 变更

2026.02.26：首次发布


## FAQ

Q：如何修改训练配置？

A：推荐通过设置`Cosmos-Reason1/cosmos-reason1/examples/post_training/configs`中的`sft.toml`或`rl.toml`文件来更改训练配置。

Q：`hf download` 下载模型或数据集过程中报错或速度过慢？

A：用户可以前往官网或使用 Hugging Face 镜像源在有网络的情况下自主下载，按照前面的结构组织文件即可。

Q：训练启动中卡死？

A：若遇到卡死，需要检查是否环境中存在网络代理等并关闭。