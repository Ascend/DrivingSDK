# PointTransformerV3 for PyTorch

## 目录

- [PointTransformerV3 for PyTorch](#PointTransformerV3-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [PointTransformerV3](#PointTransformerV3)
  - [准备训练环境](#准备训练环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [安装模型环境](#安装模型环境)
    - [准备数据集](#准备数据集)
  - [快速开始](#快速开始)
    - [训练任务](#训练任务)
      - [开始训练](#开始训练)
      - [训练结果](#训练结果)
- [变更说明](#变更说明)
- [FAQ](#faq)

# 简介

## 模型介绍

PointTransformerV3专注于在点云处理的背景下克服现有的准确性和效率之间的权衡，将简单性和效率置于某些机制的准确性之上，用以特定模式组织的点云的高效串行neighbor mapping来取代KNN的精确neighbor搜索。 这一原理实现了显著的缩放，将感受野从16个点扩展到1024个点，同时保持高效，在跨越室内和室外场景的20多项下游任务中取得了最先进的成果。

## 支持任务列表

本仓已经支持以下模型任务类型

| 模型 |    任务列表     | 是否支持 |
| :--: | :-------------: | :------: |
| PointTransformerV3 | 训练 |    ✔     |

## 代码实现

- 参考实现：

```
url=https://github.com/Pointcept/Pointcept
commit_id=e4de3c25f57d0625cdcb66589cf180e838a05b19
```

- 适配昇腾 AI 处理器的实现：

```
url=https://gitee.com/ascend/DrivingSDK.git
code_path=model_examples/PointTransformerV3
```

# PointTransformerV3

## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1** 昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdapter | 7.1.0 |
|       CANN        | 8.1.0 |

### 安装模型环境

**表 2** 三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |   2.1.0   |

0. 激活 CANN 环境

   将 CANN 包目录记作 cann_root_dir，执行以下命令以激活环境

   ```
   source {cann_root_dir}/set_env.sh
   ```

1. 参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》安装 2.1.0 版本的 PyTorch 框架和 torch_npu 插件。

2. 设置 PointTransformerV3 并安装相关依赖
    ```
    git clone https://github.com/Pointcept/Pointcept.git
    cp patch.py Pointcept/tools
    cp Ptv3.patch Pointcept
    cp train_8p.sh Pointcept
    cp requirements.txt Pointcept
    cd Pointcept
    git checkout e4de3c25f57d0625cdcb66589cf180e838a05b19
    git apply Ptv3.patch
    pip install -r requirements.txt
    ```

### 准备数据集

1. 用户需自行下载 nuScenes 数据集(包含lidarseg文件)

2. 数据预处理, 得到info文件夹

    ```
    # NUSCENES_DIR: the directory of downloaded nuScenes dataset.
    # PROCESSED_NUSCENES_DIR: the directory of processed nuScenes dataset (output dir).
    # MAX_SWEEPS: Max number of sweeps. Default: 10.
    pip install nuscenes-devkit pyquaternion
    python pointcept/datasets/preprocessing/nuscenes/preprocess_nuscenes_info.py --dataset_root ${NUSCENES_DIR} --output_root ${PROCESSED_NUSCENES_DIR} --max_sweeps 10 --with_camera
    ```

3. 将数据集和预处理结果放置到模型目录下，info为数据预处理结果目录，文件结构排布如下：
    ```
    data
    |──nuscene
    |    |──── raw
    |    |    │── samples
    |    |    │── sweeps
    |    |    │── lidarseg
    |    |    ...
    |    |    │── v1.0-trainval
    |    |    │── v1.0-test
    |──── info
    ```

## 快速开始

### 训练任务

本任务主要提供**单机**的**8卡**训练脚本。

#### 开始训练
- 在模型根目录下，运行训练脚本。
    ```
    cd model_examples/PointTransformerV3/Pointcept
    bash train_8p.sh
    ```

#### 训练结果

|     芯片      | 卡数 | Global Batchsize| mIou | FPS | 单步迭代耗时(ms) |
| :-----------: | :--: |  :----:| :----------------: | :--: |:--: |
|     竞品A     |  8p   |  8  |       0.5517        | 35.56 | 225 |
| Atlas 800T A2 |  8p   |   8 |      0.5538        | 11.92 | 671 |

# 变更说明

2025.05.14：首次发布

# FAQ

无