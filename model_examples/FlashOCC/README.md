# FlashOCC for PyTorch

## 目录

-   [简介](#简介)
    - [模型介绍](#模型介绍)
    - [支持任务列表](#支持任务列表)
    - [代码实现](#代码实现)
-   [FlashOCC](#FlashOCC)
    - [准备训练环境](#准备训练环境)
    - [快速开始](#快速开始)
       - [训练任务](#训练任务) 
-   [变更说明](#变更说明)
-   [FAQ](#FAQ)

# 简介

## 模型介绍

FlashOCC是一种高效且轻量化的占用预测框架，专为自动驾驶系统中的3D场景理解设计。与现有体素级别方法不同，FlashOCC在BEV（鸟瞰图）空间中保留特征，利用高效的2D卷积进行特征提取，并通过通道到高度的变换将BEV输出提升至3D空间。这一设计显著降低了内存和计算开销，同时保持了高精度。

## 支持任务列表

本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| FlashOCC |   训练   |    ✔     |

## 代码实现

- 参考实现：

    ```
    url=https://github.com/Yzichen/FlashOCC
    commit_id=4084861d8d605bb01df55fcbc8072036055aa625
    ```

# FlashOCC（在研版本）

## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 7.0.RC1  |
|       CANN        | 8.1.RC1  |

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

1. 准备模型源码及安装基础依赖

    在当前目录下，克隆并准备 FlashOCC 源码

    ```
    git clone https://github.com/Yzichen/FlashOCC.git
    cp flashocc.patch FlashOCC
    cp -r test/ FlashOCC/
    cd FlashOCC
    git checkout 4084861d8d605bb01df55fcbc8072036055aa625
    git apply --reject --whitespace=fix flashocc.patch
    pip install -r requirements/runtime.txt
    cd ../
    ```

2. 源码编译安装 mmcv

    克隆 mmcv 仓，并进入 mmcv 目录编译安装

    ```
    git clone -b rc41.x https://github.com/momo609/mmcv.git
    cp mmcv.patch mmcv
    cd mmcv
    git apply --reject mmcv.patch
    MMCV_WITH_OPS=1 MAX_JOBS=8 FORCE_NPU=1 python setup.py build_ext
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py develop
    cd ../
    ```

3. 安装 mmdet

    克隆 mmdet 仓，并进入 mmdet 目录编译安装

    ```
    git clone -b v2.25.0 https://github.com/open-mmlab/mmdetection.git
    cp mmdet.patch mmdetection
    cd mmdetection
    git apply --reject mmdet.patch
    pip install -e .
    cd ../
    ```

4. 安装 mmdet3d

    克隆 mmdet3d 仓，并进入 mmdet3d 目录编译安装

    ```
    cd FlashOCC
    git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
    cp ../mmdet3d.patch mmdetection3d
    cd mmdetection3d
    git apply --reject mmdet3d.patch
    pip install -v -e .
    cd ../
    ```

5. 安装 Driving SDK 加速库

    安装方法参考[原仓](https://gitee.com/ascend/DrivingSDK/wikis/DrivingSDK%20%E4%BD%BF%E7%94%A8)

### 准备数据集

1. 根据原仓[Environment Setup](https://github.com/Yzichen/FlashOCC/blob/master/doc/install.md) 在模型源码根目录下准备数据集，参考数据集结构如下：

    ```
    └── Path_to_FlashOcc/
    └── data
        └── nuscenes
            ├── v1.0-trainval
            ├── maps
            ├── sweeps
            ├── samples
            ├── gts
            ├── bevdetv2-nuscenes_infos_train.pkl (经数据预处理后生成)
            └── bevdetv2-nuscenes_infos_val.pkl (n经数据预处理后生成)
    ```

2. 在模型源码根目录下进行数据预处理

   ```
   python tools/create_data_bevdet.py
   ```

### 准备预训练权重

在模型源码根目录下创建 ckpts 文件夹，将预训练权重 [bevdet-r50-cbgs.pth](https://drive.usercontent.google.com/download?id=1oWkQLmzAXi_AoJZ259EbRmksbOyBbYuX&export=download&authuser=0) 放入其中
   ```
   ckpts/
   ├── bevdet-r50-cbgs.pth
   ```

## 快速开始

### 训练任务

本任务主要提供**单机**的**8卡**训练脚本。

#### 开始训练

  - 在模型源码根目录下，运行训练脚本。

     ```
     bash test/train_8p_flashocc_r50.sh --epochs=1 # 8卡性能
     bash test/train_8p_flashocc_r50.sh # 8卡精度
     ```

#### 训练结果

| 芯片          | 卡数 | global batch size | Precision | epoch | mIoU | 性能-单步迭代耗时(s) |
| ------------- | :--: | :---------------: | :-------: | :---: | :----: | :-------------------: |
| 竞品A           |  8p  |         32         |   fp32    |  24   | 31.73 |         7.823          |
| Atlas 800T A2 |  8p  |         32         |   fp32    |  24   | 31.79 |          4.22          |

# 变更说明

2025.3.13：首次发布。

# FAQ

1. 若遇到`ImportError: cannot import name 'gcd' from 'fraction'` 报错，升级`networkx`即可。
