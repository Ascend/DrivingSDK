# BEVNeXt for PyTorch

## 目录

- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [BEVNeXt（在研版本）](#bevnext在研版本)
  - [准备训练环境](#准备训练环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [安装模型环境](#安装模型环境)
    - [准备数据集](#准备数据集)
    - [准备预训练权重](#准备预训练权重)
  - [快速开始](#快速开始)
    - [开始训练](#开始训练)
    - [训练结果](#训练结果)
- [变更说明](#变更说明)
- [FAQ](#FAQ)

# 简介

## 模型介绍

BEVNeXt 是一种用于 3D 对象检测的现代密集 BEV 框架。

## 支持任务列表

本仓已经支持以下模型任务类型：

| 模型 |    任务列表     | 是否支持 |
| :--: | :-------------: | :------: |
| BEVNeXt | 训练 |    ✔     |

## 代码实现

- 参考实现：

    ```
    url=https://github.com/woxihuanjiangguo/BEVNeXt
    commit_id=9b0e4ad33ed3e82dc9cee9f0f66ffd1899095026
    ```

- 适配昇腾 AI 处理器的实现：

    ```
    url=https://gitee.com/ascend/DrivingSDK.git
    code_path=model_examples/BEVNeXt
    ```

# BEVNeXt（在研版本）

## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1** 昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 7.0.RC1 |
|       CANN        | 8.1.RC1 |
|    昇腾NPU固件    | 25.0.RC1 |
|    昇腾NPU驱动    | 25.0.RC1 |

### 安装模型环境

**表 2** 三方库版本支持表

| Torch_Version | 三方库依赖版本 |
| :-----: | :------: |
| PyTorch 2.1 | torchvision==0.16.0 |

0. 激活 CANN 环境

    将 CANN 包目录记作 cann_root_dir，执行以下命令以激活环境

    ```
    source {cann_root_dir}/set_env.sh
    ```

1. 参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》安装 2.1.0 版本的 PyTorch 框架和 torch_npu 插件。

2. 源码安装 mmcv

    ```
    git clone -b 1.x https://github.com/open-mmlab/mmcv.git
    cd mmcv/
    cp -f ../mmcv.patch ./
    git apply --reject mmcv.patch
    pip install -r requirements/runtime.txt
    MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
    cd ../
    ```

3. 源码安装 mmdetection3d

    ```
    git clone -b v1.0.0rc6 https://github.com/open-mmlab/mmdetection3d.git --depth=1
    cd mmdetection3d/
    git fetch --unshallow
    git checkout 47285b3f1e9dba358e98fcd12e523cfd0769c876
    cp -f ../mmdetection3d.patch ./
    git apply --reject mmdetection3d.patch
    pip install -e .
    cd ../
    ```

4. 安装其他依赖

    ```
    pip install -r requirements.txt
    ```

5. 准备模型源码

    ```
    git clone https://github.com/woxihuanjiangguo/BEVNeXt.git
    cd BEVNeXt/
    git checkout 9b0e4ad33ed3e82dc9cee9f0f66ffd1899095026
    cp -f ../bevnext.patch ./
    cp -rf ../test ./
    git apply --reject bevnext.patch
    ```

6. 安装 Driving SDK 加速库

    参考官方文档：https://gitee.com/ascend/DrivingSDK/blob/master/README.md

### 准备数据集

根据原仓 [README](https://github.com/woxihuanjiangguo/BEVNeXt/blob/master/README.md) 的 **Installation & Dataset Preparation** 章节准备数据集。

1. 用户需自行下载 nuScenes 数据集，放置在 **BEVNeXt 模型源码**目录下或自行构建软连接，并**提前处理**好 nuScenes 数据集。

2. 执行数据预处理命令

    ```
    python tools/create_data_bevdet.py
    ```

    预处理完的数据目录结构如下：

    ```
    BEVNeXt
    ├── data/
    │   ├── nuscenes/
    │   │   ├── maps/
    │   │   ├── samples/
    │   │   ├── sweeps/
    │   │   ├── v1.0-test/
    |   |   ├── v1.0-trainval/
    |   |   ├── nuscenes_infos_train.pkl
    |   |   ├── nuscenes_infos_val.pkl
    |   |   ├── bevdetv2-nuscenes_infos_train.pkl
    |   |   ├── bevdetv2-nuscenes_infos_val.pkl
    ```

### 准备预训练权重

1. 联网情况下，预训练权重会自动下载。

2. 无网络情况下，可以通过该链接自行下载 [resnet50-0676ba61.pth](https://download.pytorch.org/models/resnet50-0676ba61.pth)，并拷贝至对应目录下。默认存储目录为 PyTorch 缓存目录：

    ```
    ~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
    ```

## 快速开始

### 训练任务

本任务主要提供**单机**的**8卡**训练脚本。

#### 开始训练

- 在模型源码目录下运行训练脚本。其中，stage1 进行模型预热，stage2 加载 stage1 的权重结果进行训练。训练结果默认保存在模型源码目录下的 `work_dirs` 目录中。

    ```
    # 训练 1 epoch
    cd model_examples/BEVNeXt/BEVNeXt
    bash test/train_8p_stage1.sh  # 默认 2 epochs
    bash test/train_8p_stage2.sh  # 默认 1 epoch

    # 获取精度结果
    bash test/test_8p_eval.sh
    ```

#### 训练结果

| 芯片 | 卡数 | config | epoch | mAP(IoU=0.50:0.95) | Torch_Version |
| ----------- | -- | -- | ---- | ---------------- | -- |
|     竞品A     | 8p | stage2 | 1 | 0.3676 | PyTorch 2.1 |
| Atlas 800T A2 | 8p | stage2 | 1 | 0.3670 | PyTorch 2.1 |

# 变更说明

2025.2.17：首次发布

# FAQ

1. 目前还未做性能优化，且存在偶现的精度异常（在竞品A上训练也会出现），暂不推荐使用。
