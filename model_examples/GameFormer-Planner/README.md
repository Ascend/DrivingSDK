# GameFormer-Planner for PyTorch

## 目录

-   [简介](#简介)
    - [模型介绍](#模型介绍)
    - [支持任务列表](#支持任务列表)
    - [代码实现](#代码实现)
-   [GameFormer-Planner](#GameFormer-Planner)
    - [准备训练环境](#准备训练环境)
    - [快速开始](#快速开始)
-   [变更说明](#变更说明)
-   [FAQ](#FAQ)

# 简介

## 模型介绍

在复杂的现实环境中运行的自动驾驶车辆需要准确预测交通参与者之间的交互行为。GameFormer-Planner模型结合了一个Transformer编码器，以及一个分层Transformer解码器结构，来有效地模拟场景元素之间的关系。在每个解码层级，除了共享的环境上下文之外，解码器还利用前一级别的预测结果来迭代地完善交互过程。该模型在nuPlan规划基准数据集上验证了有效性，取得了领先的性能。本仓库针对GameFormer-Planner模型进行了昇腾NPU适配，并且提供了适配Patch，方便用户在NPU上进行模型训练。

## 支持任务列表

本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| GameFormer-Planner |   训练   |    ✔     |

## 代码实现

- 参考实现：
    ```
    url=https://github.com/MCZhi/GameFormer-Planner
    commit_id=c6f3a69b947edd0c3079e458275fc490520e8bde
    ```

- 适配昇腾 AI 处理器的实现：
    ```
    url=https://gitee.com/ascend/DrivingSDK.git
    code_path=model_examples/GameFormer-Planner
    ```

# GameFormer-Planner

## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 6.0.0 |
|       CANN        | 8.0.0  |


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

1. 安装torch2.1、torch_npu2.1 以及其他依赖项
    ```
    pip install torch==2.1.0
    pip install torch_npu==2.1.0
    pip install -r requirements.txt
    ```

2. 安装nuplan-devkit库（如果需要进行数据集预处理，可选）：
    ```
    git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
    pip install -r requirements.txt
    pip install -e .
    cd ..
    ```

3. 拉取GameFormer-Planner模型仓库代码并使用Patch进行代码修改
    ```
    git clone https://github.com/MCZhi/GameFormer-Planner.git && cd GameFormer-Planner
    git checkout c6f3a69b947edd0c3079e458275fc490520e8bde
    cp ../GameFormer-Planner_NPU.patch .
    git apply --reject --whitespace=fix GameFormer-Planner_NPU.patch
    cd ..
    ```

### 准备数据集

1. 下载[NuPlan数据集](https://www.nuscenes.org/nuplan#download)，并将数据集结构排布成如下格式：
    ```
    ~/nuplan
    └── dataset
        ├── maps
        │   ├── nuplan-maps-v1.0.json
        │   ├── sg-one-north
        │   │   └── 9.17.1964
        │   │       └── map.gpkg
        │   ├── us-ma-boston
        │   │   └── 9.12.1817
        │   │       └── map.gpkg
        │   ├── us-nv-las-vegas-strip
        │   │   └── 9.15.1915
        │   │       └── map.gpkg
        │   └── us-pa-pittsburgh-hazelwood
        │       └── 9.17.1937
        │           └── map.gpkg
        └── nuplan-v1.1
            ├── splits 
                ├── mini 
                │    ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
                │    ├── 2021.06.09.17.23.18_veh-38_00773_01140.db
                │    ├── ...
                │    └── 2021.10.11.08.31.07_veh-50_01750_01948.db
                └── trainval
                    ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
                    ├── 2021.06.09.17.23.18_veh-38_00773_01140.db
                    ├── ...
                    └── 2021.10.11.08.31.07_veh-50_01750_01948.db

    ```
2. 数据预处理
    ```
    python GameFormer-Planner/data_process.py
    --data_path nuplan/dataset/nuplan-v1.1/splits/mini
    --map_path nuplan/dataset/maps
    --save_path nuplan/nuplan_processed
    ```
    --scenarios_per_type和--total_scenarios可以用于控制生成数据点的数量，请根据原仓库的指引，生成150万个数据点，并用其中的十分之一作为Validation Set，剩余的部分作为Training Set。预处理完成之后数据排布如下所示：
    ```
    nuplan
    └── nuplan_processed
        ├── train
        │   ├── us-nv-las-vegas-strip_12b86ec515a15de0.npz
        │   ├── ...
        │   └── us-nv-las-vegas-strip_b880c406318b552f.npz
        └── val
            ├── us-pa-pittsburgh-hazelwood_db2c7a20fb6453d4.npz
            ├── ...
            └── us-pa-pittsburgh-hazelwood_ffd6690ff3ba5ee5.npz

    ```

## 快速开始
本任务主要提供**单机8卡**的训练脚本以及**双机16卡**的多机多卡训练脚本。
### 开始训练

- 单机多卡：在模型根目录下，运行训练脚本。
    ```
    bash script/train_gameformer_8x512.sh 8 1 # 8卡性能(1 epoch)
    bash script/train_gameformer_8x512.sh 8 30 # 8卡精度(30 epoch)
    bash script/train_gameformer_8x256.sh 8 1 # 8卡性能(1 epoch)
    bash script/train_gameformer_8x256.sh 8 30 # 8卡精度(30 epoch)
    ```
- 多机多卡：在模型根目录下，运行训练脚本。
    ```
    # 'XX.XX.XX.XX'为主节点的IP地址；端口号可以换成未被占用的可用端口
    bash script/train_gameformer_multi_server_8x512.sh 8 30 0 'XX.XX.XX.XX' '3389'  # 主节点
    bash script/train_gameformer_multi_server_8x512.sh 8 30 1 'XX.XX.XX.XX' '3389'  # 副节点
    ```

### 训练结果
- 单机8卡
                             
|    芯片    | 卡数 | global batch size | Precision | epoch | plannerADE | plannerFDE | plannerAHE | plannerFHE | predictorADE | predictorFDE | 性能-单步迭代耗时(ms) |
|:---------:|:----:|:-----------------:|:--------:|:-----:|:---------:|:---------:|:---------:|:---------:|:-----------:|:-----------:|:--------------------:|
|   竞品A   |  8p  |       4096        |   fp32   |  30   |   1.17    |   3.11    |   0.10    |   0.07    |    0.70     |    1.30     |         790          |
| Atlas 800T A2 |  8p  |       4096        |   fp32   |  30   |   1.16    |   3.10    |   0.10    |   0.07    |    0.69     |    1.29     |         965          |
|   竞品A   |  8p  |       2048        |   fp32   |  30   |   1.07    |   2.78    |   0.14    |   0.06    |    0.55     |    1.10     |         440          |
| Atlas 800T A2 |  8p  |       2048        |   fp32   |  30   |   1.05    |   2.79    |   0.09    |   0.06    |    0.54     |    1.07     |         647          |

- 多机多卡线性度                          

|      芯片      | 卡数 | global batch size | Precision | epoch | 性能-单步迭代耗时(ms) | 线性度  |
|:-------------:|:----:|:-----------------:|:--------:|:-----:|:--------------------:|:------:|
| Atlas 800T A2*2 |  16p |       8192        |   fp32   |  30   |         971          |  99.3% |
| Atlas 800T A2*2 |  16p |       4096        |   fp32   |  30   |         651          |  99.6% |  

# 变更说明

2024.09.19：首次发布。

2024.11.11：资料更新。

2025.04.24: 更新性能数据和FPS计算方式

2025.04.28:更新8卡性能数据和多机线性度
# FAQ

无