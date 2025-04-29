# MagicDriveDiT for PyTorch

## 目录

- [MagicDriveDiT for PyTorch](#MagicDriveDiT-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [MagicDriveDiT](#MagicDriveDiT)
  - [准备训练环境](#准备训练环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [准备数据集](#准备数据集)
    - [准备预训练权重](#准备预训练权重)
  - [快速开始](#快速开始)
    - [训练任务](#训练任务)
      - [开始训练](#开始训练)
      - [训练结果](#训练结果)
- [变更说明](#变更说明)
- [FAQ](#faq)

# 简介

## 模型介绍

MagicDriveDiT是一种面向自动驾驶的高分辨率长视频生成模型，基于DiT架构，通过流匹配技术和时空条件编码实现分辨率达848×1600、帧数达241帧的多视角街景视频生成，支持3D边界框、BEV地图及文本等多模态控制，解决了传统方法在可扩展性与几何控制上的瓶颈。

## 支持任务列表

本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| MagicDriveDiT |   训练   |    ✔     |

## 代码实现

- 参考实现：
  
  ```
  url=https://github.com/flymin/MagicDriveDiT
  commit_id=78b65f9db34c52164926815ab6ee51902960ef8a 
  ```

- 适配昇腾 AI 处理器的实现：
    ```
    url=https://gitee.com/ascend/DrivingSDK.git
    code_path=model_examples/MagicDriveDiT
    ```

# MagicDriveDiT

## 准备训练环境

### 安装环境

**表 1**  三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |   2.3   |

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。

**表 2**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 7.0.0  |
|       CANN        | 8.1.RC1  |

1. 激活 CANN 环境
    将 CANN 包目录记作 cann_root_dir，执行以下命令以激活环境
    ```
    source {cann_root_dir}/set_env.sh
    ```

2. 安装torch2.3、torch_npu2.3 以及其他依赖项
    ```
    pip install torch==2.3.1
    pip install torch_npu==2.3.1
    pip install -r requirements.txt
    ```

3. 克隆代码仓到当前目录：

    ```
    git clone https://gitee.com/ascend/DrivingSDK.git -b master
    cd DrivingSDK/model_examples/MagicDriveDiT
    git clone https://github.com/flymin/MagicDriveDiT.git 
    cd MagicDriveDiT
    git checkout 78b65f9db34c52164926815ab6ee51902960ef8a
    ```

    将模型根目录记作 `model-root-path`
    
4. 使用 patch 文件：
    ```
    cp -f ../MagicDriveDiT.patch .
    git apply --reject --whitespace=fix MagicDriveDiT.patch
    cp -rf ../test .
    ```

4. 安装模型相关的依赖项。
  
  ```
  # 安装apex (https://gitee.com/ascend/apex)
  git clone -b master https://gitee.com/ascend/apex.git
  cd apex/
  bash scripts/build.sh --python=3.9
  cd apex/dist/
  pip3 uninstall apex
  pip3 install --upgrade apex-0.1+ascend-{version}.whl # version为python版本和cpu架构

  # 安装Colossalai
  git clone https://github.com/flymin/ColossalAI.git
  git checkout ascend && git pull
  cd ColossalAI
  BUILD_EXT=1 pip install .

  # 安装其他依赖项
  pip install -r requirements/requirements.txt
  ```

### 准备数据集

- 根据原仓**Prepare Data**章节准备数据集，数据集目录及结构如下：

```bash
${CODE_ROOT}/data/nuscenes
├── can_bus
├── maps
├── mini
├── samples
├── sweeps
├── v1.0-mini
└── v1.0-trainval
```

- 根据原仓准备metadata，数据目录如下：

```bash
${CODE_ROOT}/data
├── nuscenes
│   ├── ...
│   └── interp_12Hz_trainval
└── nuscenes_mmdet3d-12Hz
    ├── nuscenes_interp_12Hz_infos_train_with_bid.pkl
    └── nuscenes_interp_12Hz_infos_val_with_bid.pkl
```

### 准备预训练权重

- 根据原仓**Pretrained Weights**章节准备预训练权重，目录及结构如下：

```bash
${CODE_ROOT}/pretrained/
├── CogVideoX-2b
│   └── vae
└── t5-v1_1-xxl
```

## 快速开始

### 训练任务

本任务目前主要提供单机的8卡训练stage1和stage2脚本。

#### 开始训练

1. 在模型根目录下，运行训练脚本。

   - 单机8卡精度训练
   
   ```
   # stage1训练
   bash test/train_8p_stage1.sh 8
   # stage2训练
   bash test/train_8p_stage2.sh 8
   ```

#### 训练结果

|阶段| 芯片          | 卡数 | sp size | Precision | Loss | 性能-单步迭代耗时(s) |
|-------------| ------------- | :--: | :---------------: | :-------------------: | :-------------------: |:-------------------: |
|stage1| 竞品A           |  8p  |         4         |   混精    | 1.31 | 1.33|
|stage1| Atlas 800T A2 |  8p  |         4         |   混精    | 1.31 | 2.39|
|stage2| 竞品A           |  8p  |         4         |   混精    | 0.830 | 3.429|
|stage2| Atlas 800T A2 |  8p  |         4         |   混精    | 0.829 | 5.706 |

# 变更说明

2025.04.25：首次发布


# FAQ

无