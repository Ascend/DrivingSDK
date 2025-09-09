# Dexvla

## 目录

-   [简介](#简介)
    - [模型介绍](#模型介绍)
    - [支持任务列表](#支持任务列表)
    - [代码实现](#代码实现)
-   [Dexvla](#Dexvla)
    - [准备训练环境](#准备训练环境)
    - [快速开始](#快速开始)
-   [变更说明](#变更说明)

# 简介

## 模型介绍

本文介绍了DexVLA，一个旨在增强VLA模型在跨不同机器人形态（embodiment）执行复杂、长周期任务时的效率和泛化能力的新型框架。DexVLA的核心是一个新颖的、基于扩散模型的动作专家网络，其参数量达到十亿级，专为跨形态学习而设计。文章提出了一种新颖的形态课程学习策略来实现高效训练，该策略包含三个阶段：(1) 在跨形态数据上预训练与VLA可分离的扩散专家模型；(2) 将VLA模型与特定机器人形态对齐；(3) 进行后续训练以实现新任务的快速适配。

## 支持任务列表

本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| Dexvla |   训练   |    ✔     |

## 代码实现

- 参考实现：
    ```
    url=https://github.com/juruobenruo/DexVLA
    commit fc21a822f4c774e242eb6f1ab4a235788de7aba9
    ```

- 适配昇腾 AI 处理器的实现：
    ```
    url=https://gitee.com/ascend/DrivingSDK.git
    code_path=model_examples/Dexvla
    ```

# DexVLA (在研版本)

## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdapter | 7.1.0 |
|       CANN        | 8.2.RC1  |


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

1. 创建conda环境
  ```
  conda create -n dexvla python=3.10
  conda activate dexvla
  ```
  
2. 安装依赖
  ```
  git clone https://github.com/juruobenruo/DexVLA.git
  cd DexVLA
  git checkout fc21a822f4c774e242eb6f1ab4a235788de7aba9
  cp -f ../dexvla.patch .
  cp -rf ../test/ .
  git apply dexvla.patch
  pip install -r requirements.txt
  cd policy_heads
  pip install -e .
  cd ../..
  ```
  
3. 安装bitsandbytes
 ```
  # 源码安装bitsandbytes
  git clone -b multi-backend-refactor https://github.com/bitsandbytes-foundation/bitsandbytes.git && cd bitsandbytes/

  # Compile & install
  apt-get install -y build-essential cmake  # ubuntu
  cmake -DCOMPUTE_BACKEND=npu -S .
  make
  pip install -e . 
 ```

4. 根据[DeepSpeed仓](https://github.com/deepspeedai/DeepSpeed)源码安装DeepSpeed，如：
  ```
  git clone https://github.com/deepspeedai/DeepSpeed.git
  cd DeepSpeed
  git checkout c2bb53f20fa32d6cbf472c08a42959a287dd9049
  pip install .
  ```

4 安装Driving SDK

请参考昇腾[Driving SDK](https://gitee.com/ascend/DrivingSDK)代码仓说明编译安装Driving SDK


### 准备数据集与预训练权重

1. 下载[Dexvla example dataset](https://huggingface.co/datasets/lesjie/dexvla_example_data)，将数据集位置记作 dataset_path.
进入DexVLA/aloha_scripts/constant.py文件，将第四行 `dataset_dir` 替换为 `$dataset_path`.

2. 下载[Qwen2-VL-2B 预训练权重](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)，将权重路径记作 qwen_weights.
权重下载完毕后，将qwen2下的 config.json 替换为 DexVLA/docs/config.json 。
 
  ```
  cp -f DexVLA/docs/config.json {qwen_weights}/config.json
  ```

3. 下载[dexvla stage1 预训练权重](https://huggingface.co/lesjie/scale_dp_h/tree/main)，将权重路径记做 stage1_weights

## 快速开始
本任务主要提供**单机8卡**的训练脚本。
### 开始训练

- 单机8卡stage2性能

  ```
  bash test/train_8p_performance_stage2.sh {qwen_weights} {stage1_weights} {stage2_output} #stage2_output为stage2训练后权重的保存路径
  ```
  
- 单机8卡stage3性能

  ```
  bash test/train_8p_performance_stage3.sh {stage1_weights} {stage2_output} {stage3_output} #stage3_output为stage3训练后权重的保存路径
  ```

- 单机8卡stage2精度

  ```
  bash test/train_8p_full_stage2.sh {qwen_weights} {stage1_weights} {stage2_output} #stage2_output为stage2训练后权重的保存路径
  ```
  
- 单机8卡stage3精度

  ```
  bash test/train_8p_full_stage3.sh {stage1_weights} {stage2_output} {stage3_output} #stage3_output为stage3训练后权重的保存路径
  ```



### 训练结果
#### Stage2
- 单机8卡

|  NAME       | Precision     |     steps    |    global_batch_size      |    training loss      |     FPS      |
|-------------|-------------------|-----------------|---------------|--------------|--------------|
|  8p-竞品A   |      Mixed    |        10k     |     96    |        0.03206  |    18.88     |
|  8p-Atlas 800T A2   |     Mixed    |        10k     |      96    |        0.03021   |      16.72    |

#### Stage3
- 单机8卡

|  NAME       | Precision     |     steps    |    global_batch_size      |    mean training loss      |     FPS      |
|-------------|-------------------|-----------------|---------------|--------------|--------------|
|  8p-竞品A   |      Mixed    |        10k     |     96    |        0.01952   |    18.67    |
|  8p-Atlas 800T A2   |     Mixed    |        10k     |      96    |        0.02027   |      15.85    |

# 变更说明

2025.09.02：首次发布。

# FAQ
Q:安装依赖包时，h5py安装失败，怎么办？
A:安装h5py依赖，如：
 ```
 yum install hdf5-devel
 pip install hdf5
 ```

Q:拉起模型训练时，遇到：ModuleNotFoundError: No module named 'models'，怎么办？
A:可能安装好的policy_heads包的路径不在python的模块搜索范围内。可以尝试在train_vla.py中将policy_heads的安装路径添加至sys.path，如：
```
import sys
#将policy_heads的安装路径记做POLICY_HEADS_PATH
sys.path.append($POLICY_HEADS_PATH)
```

