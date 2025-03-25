# BEVNeXt for PyTorch

## 目录

- [BEVNeXt for PyTorch](#bevnext-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [BEVNeXt](#bevnext)
  - [准备训练环境](#准备训练环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [安装模型环境](#安装模型环境)
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

# BEVNeXt

## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境。本仓已支持表1中软件版本。
  
  **表 1**  昇腾软件版本支持表

  |        软件类型        |   支持版本   |
  |:------------------:|:--------:|
  | FrameworkPTAdapter | 6.0.0  |
  |       CANN         | 8.0.0  |


### 安装模型环境

**表 2** 版本支持表

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

- 在模型源码目录下运行训练脚本。其中，stage1 进行模型预热，stage2 加载 stage1 的权重进行训练。

    ```
    # 单机 8 卡训练
    cd model_examples/BEVNeXt/BEVNeXt
    bash test/train_8p_stage1.sh  # 默认 2 epochs
    bash test/train_8p_stage2.sh  # 默认 12 epochs

    # 运行评测脚本获取精度结果，默认 stage2 12 epochs
    bash test/eval_bevnext.sh
    ```

    > 注：当前配置下，训练结果默认保存在模型源码目录下的 `work_dirs` 目录中。如果修改了 stage1 的权重保存路径，请用户根据路径自行配置 `train_8p_stage2.sh` 中的 `stage1_ckpts_path`。如果修改了 stage2 的权重保存路径，请自行配置 `eval_bevnext.sh` 中的 `work_dir`（与 stage2 权重保存路径相同）。

#### 训练结果

| 芯片 | 卡数 | 阶段 | epoch | FPS | mAP | Torch_Version |
| -- | -- | -- | -- | -- | -- | -- |
|     竞品A     | 8p | stage1 | 2 | 36.643 | \ | PyTorch 2.1 |
| Atlas 800T A2 | 8p | stage1 | 2 | 16.568 | \ | PyTorch 2.1 |
|     竞品A     | 8p | stage2 | 12 | 11.651 | 0.4313 | PyTorch 2.1 |
| Atlas 800T A2 | 8p | stage2 | 12 | 7.572 | 0.4316 | PyTorch 2.1 |

# 变更说明

2025.2.17：首次发布。

2025.2.27：性能优化，当前 stage1 性能为 0.45 倍竞品A，stage2（4 epochs）性能为 0.69 倍竞品A。

2025.3.10：更新训练脚本中的 `epochs` 默认值为 12；更新完整训练 12 epochs 性能和精度数据，当前 stage1 性能为 0.4x 倍竞品A，stage2 性能为 0.6x 倍竞品A。

# FAQ

1. 在竞品 A 与 Atlas 800T A2 机器上训练均存在偶现的精度异常，暂不推荐使用。
