# NWM

## 目录

- [NWM](#NWM)
- [目录](#目录)
- [简介](#简介)
- [模型介绍](#模型介绍)
- [支持任务列表](#支持任务列表)
- [代码实现](#代码实现)
- [准备训练环境](#准备训练环境)
- [安装环境](#安装环境)
- [安装昇腾环境](#安装昇腾环境)
- [准备数据集](#准备数据集)
- [快速开始](#快速开始)
- [训练任务](#训练任务)
- [变更说明](#变更说明)
- [FAQ](#faq)

# 简介

## 模型介绍

Navigation World Model（NWM）是一种基于条件扩散 Transformer（CDiT）的可控视频生成式世界模型，能通过学习过往视觉观测与导航动作关联预测未来视觉状态，支持已知环境轨迹规划、外部政策轨迹排序及未知环境轨迹想象，无需依赖显式 3D 地图，可提升机器人等智能体的视觉导航性能。

## 支持任务列表

本仓已经支持以下模型任务类型

| 模型 | 任务列表 | 是否支持 |
| :--: | :------: | :------: |
| NWM  |   训练   |    ✔     |

## 代码实现

- 参考实现：

```
url=https://github.com/facebookresearch/nwm
commit_id=3f6cd8e70d6f2d1e2b9684acff510710135f0f41
```

- 适配昇腾 AI 处理器的实现：

```
url=https://gitcode.com/Ascend/DrivingSDK.git
code_path=model_examples/NWM
```

## 准备训练环境

### 安装环境

**表 1**  三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |  2.7.1   |

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本，python使用3.10版本。

**表 2**  昇腾软件版本支持表

|      软件类型      | 支持版本 |
| :----------------: | :------: |
| FrameworkPTAdapter |  7.2.0   |
|        CANN        | 8.3.RC1  |
|       Python       |   3.10   |

- 克隆代码仓到当前目录并使用patch文件

```
git clone https://github.com/facebookresearch/nwm
cd nwm
git checkout 3f6cd8e70d6f2d1e2b9684acff510710135f0f41
cp -f ../NWM_npu.patch .
git apply --reject --whitespace=fix NWM_npu.patch
cp -r ../test 
```

- 安装环境依赖

- 在应用过patch的模型根目录下，安装需要的依赖

```
conda install ffmpeg
pip install einops evo transformers diffusers tqdm timm notebook dreamsim torcheval lpips ipywidgets
pip install torchvision==0.21.0
```

- 安装过程中可能会改变torch版本，若版本被覆盖，需再次安装torch

### 准备数据集

- 根据源仓readme中下载数据集，本仓中为实现快速验证，仅使用'RECON'场景进行训练，如果使用其他场景，需要修改代码中对应部分。使用源仓中提供的脚本对数据集进行处理，处理后的数据集结构排列如下：

```
├── <dataset_name>
│   ├── <name_of_traj1>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_1.jpg
│   │   └── traj_data.pkl
│   ├── <name_of_traj2>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_2.jpg
│   │   └── traj_data.pkl
│   ...
└── └── <name_of_trajN>
    ├── 0.jpg
    ├── 1.jpg
    ├── ...
        ├── T_N.jpg
        └── traj_data.pkl
```

## 快速开始

### 训练任务

#### 执行训练

1. 在应用过patch的模型根目录下，执行以下指令进行训练。

- 单机八卡性能


```
bash test/train_8p_nwm_perf.sh --max-epochs 1
```

- 单机八卡长跑


```
bash test/train_8p_nwm_full.sh --max-epochs 7
```

#### 单机八卡训练性能和loss对比

| 芯片          | 卡数 | gloabl batchsize |  Loss  |  SPS  |
| ------------- | :--: | :--------------: | :----: | :---: |
| 竞品A         |  8p  |       96        | 0.1413 | 383.06 |
| Atlas 800T A2 |  8p  |       96        | 0.1502 | 363.39 |

# 变更说明
2025.11.06:首次发布
# FAQ

无