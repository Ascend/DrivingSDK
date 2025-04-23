# HiVT for PyTorch

## 目录

- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [HiVT（在研版本）](#HiVT(在研版本))
  - [准备训练环境](#准备训练环境)
    - [安装昇腾环境](#安装昇腾环境)
    - [安装模型环境](#安装模型环境)
    - [模型数据准备](#模型数据准备)
  - [快速开始](#快速开始)
    - [开始训练](#开始训练)
    - [训练结果](#训练结果)
- [变更说明](#变更说明)
- [FAQ](#FAQ)

# 简介

## 模型介绍

HiVT是一种面向自动驾驶Multi-Agent Motion Prediction任务的深度学习框架，它克服了传统vectorized approaches（将轨迹点和地图都转化为矢量化实体，如轨迹点、车道段，再利用GNN,Transformers 等方法进行建模）在同时建模Multi-Agent时效率低、难以做到实时预测的问题，实现了更快、更准确的行为预测。

## 支持任务列表

本仓已经支持以下模型任务类型

|   模型   | 任务列表 | 是否支持 |
| :------: | :------: | :------: |
| HiVT |   训练   |    ✔     |

## 代码实现

- 参考实现：

```
https://github.com/ZikangZhou/HiVT.git
commit_id=6876656ce7671982ebdc29113aaaa028c2931518
```

- 适配昇腾 AI 处理器的实现：

```
url=https://gitee.com/ascend/DrivingSDK.git
code_path=model_examples/HiVT
```

# HiVT（在研版本）

## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1** 昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 7.0.0  |
|       CANN        | 8.1.0  |

### 安装模型环境

**表 2** 三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |  2.1.0   |

0. 激活 CANN 环境

   将 CANN 包目录记作 cann_root_dir，执行以下命令以激活环境

   ```
   source {cann_root_dir}/set_env.sh
   ```

1. 参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》安装 2.1.0 版本的 PyTorch 框架和 torch_npu 插件。
    ```
    conda create -n HiVT python=3.8
    conda activate HiVT
    # 安装torch和torch_npu
    pip install torch-2.1.0.xxxxxxxxx.whl
    pip install torch_npu-2.1.0.xxxxxxxxx.whl
    ```
2. 安装 DrivingSDK 加速库，安装方法参考[官方文档](https://gitee.com/ascend/DrivingSDK/wikis/DrivingSDK)。
    ```
    # DrivingSDK 加速库安装完成后到HiVT模型目录下
    cd DrivingSDK/model_examples/HiVT
    ```
3. 安装argoverse-api，并且下载hd_maps解压到argoverse-api文件中
    ```
    git clone https://github.com/argoai/argoverse-api.git
    cd argoverse-api
    git checkout f886ac54fba9f06f8a7d109eb663c7f501b3aa8e
    git apply ../patch/argoverse-api.patch
    pip install -e .
    wget https://s3.amazonaws.com/argoverse/datasets/av1.1/tars/hd_maps.tar.gz
    tar -zxvf hd_maps.tar.gz
    cd ..    
    ```
4. 安装torch_scatter
    ```
    git clone https://github.com/rusty1s/pytorch_scatter.git -b 2.1.0
    cd pytorch_scatter
    git checkout fa4f442952955acf8fe9fcfb98b600f6ca6081b6
    git apply ../patch/torch_scatter.patch
    # 编译耗时较久，需要十分钟
    pip install -e .
    cd ..
    ```
5. 安装pip依赖
    ```
    pip install -r requirements.txt
    ```
6. 拉取HiVT模型源代码
    ```
    git clone https://github.com/ZikangZhou/HiVT.git && cd HiVT
    git checkout 6876656ce7671982ebdc29113aaaa028c2931518
    git apply ../patch/HiVT.patch
    cd ..
    ```

### 模型数据准备

训练集：https://s3.amazonaws.com/argoverse/datasets/av1.1/tars/forecasting_train_v1.1.tar.gz

验证集：https://s3.amazonaws.com/argoverse/datasets/av1.1/tars/forecasting_val_v1.1.tar.gz

下载后解压到指定目录下：/path/to/Argoverse

- 文件夹结构
```
Argoverse
├── train/
| └── data/
| ├── 1.csv
| ├── 2.csv
| ├── ...
└── val/
└── data/
├── 1.csv
├── 2.csv
├── ...
```
- 数据预处理

当数据集解压后置于数据集路径下，pytorch-lightening框架会在第一次执行训练脚本时，自动开始数据预处理过程，处理总时长大约10小时。


## 快速开始

### 训练任务

本任务主要提供**单机**的**8卡**训练脚本。

### 开始训练

在模型根目录下启动训练。

```  
cd /path/DrivingSDK/model_examples/HiVT/HiVT
```

- 单机8卡性能

  ```
  # epoch = 1，/path/to/Argoverse/ 请更改为存放数据的路径
  python train.py --root /path/to/Argoverse/  --embed_dim 64 --max_epochs 1 --gpus 8
  ```

- 单机8卡精度

  ```
  # epoch = 64，/path/to/Argoverse/ 请更改为存放数据的路径
  python train.py --root /path/to/Argoverse/ --embed_dim 64 --gpus 8
  ```

### 训练结果

|     芯片      | 卡数 | global batch size  | epoch | minFDE | minADE | 性能-单步迭代耗时(s) | FPS |
| :-----------: | :--: | :---------------: | :---: | :--------------------: | :--------------------: | :--------------: | :---: |
|     竞品A     |  8p  |         256         |  64   |         1.022          |         0.6845          |       0.426        |  601  |
| Atlas 800T A2 |  8p  |         256         |  64   |         1.03          |         0.6858          |       0.637         | 402   |

# 变更说明

2025.4.22：首次发布

# FAQ

1. pip安装omegaconf==2.1.0报错
  ```
    ERROR: Ignored the following yanked versions: 1.0.0, 1.0.1, 1.0.2, 2.0.0rc1, 2.0.0rc2, 2.0.0rc22, 2.0.0rc23, 2.0.0rc24, 2.0.0rc25, 2.0.0rc26, 2.0.0rc27, 2.0.0rc28, 2.0.0rc29, 2.0.1rc1, 2.0.1rc2, 2.0.1rc3, 2.0.1rc4, 2.0.1rc5, 2.2.0
    ERROR: Could not find a version that satisfies the requirement omegaconf==2.1.0
  ```
解决方法：pip install pip==24.0

2. pip安装h5py报错
  ```
    ERROR: Failed building wheel for h5py
    Failed to build h5py
    ERROR: Could not build wheels for h5py, which is required to install pyproject.toml-based projects
  ```

解决方法：conda install h5py
