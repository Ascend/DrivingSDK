# Diffusion-Planner

## 目录

-   [简介](#简介)
    - [模型介绍](#模型介绍)
    - [支持任务列表](#支持任务列表)
    - [代码实现](#代码实现)
-   [Diffusion-Planner](#Diffusion-Planner)
    - [准备训练环境](#准备训练环境)
    - [快速开始](#快速开始)
-   [变更说明](#变更说明)

# 简介

## 模型介绍

本文提出了一种基于Transformer的Diffusion Planner模型，用于解决开放复杂环境中自动驾驶的规划难题。该模型通过扩散生成技术，有效建模多模态驾驶行为，无需依赖规则后处理即可保障轨迹质量。其创新点包括：1）统一架构联合建模预测与规划任务，促进车辆协同；2）采用分类器引导机制学习轨迹评分梯度，实现安全自适应规划。在大规模nuPlan基准和200小时配送车辆实测数据上的实验表明，该模型在闭环性能与驾驶风格迁移性方面均达到SOTA水平，显著超越传统模仿学习方法。本仓库针对Diffusion-Planner模型进行了昇腾NPU适配，并且提供了适配Patch，方便用户在NPU上进行模型训练。

## 支持任务列表

本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| Diffusion-Planner |   训练   |    ✔     |

## 代码实现

- 参考实现：
    ```
    url=https://github.com/ZhengYinan-AIR/Diffusion-Planner
    commit 62196099b6e969f532be0ac0b20d1a236ebbfd19
    ```

- 适配昇腾 AI 处理器的实现：
    ```
    url=https://gitee.com/ascend/DrivingSDK.git
    code_path=model_examples/Diffusion-Planner
    ```

# Diffusion-Planner (在研版本)

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
  conda create -n diffusion_planner python=3.9
  conda activate diffusion_planner
  ```

2. 安装 nuplan-devkit
  ```
  git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
  pip install -e .
  pip install -r requirements.txt
  ```

3. 安装 diffusion_planner
  ```
  cd ..
  git clone https://github.com/ZhengYinan-AIR/Diffusion-Planner.git && cd Diffusion-Planner
  cp -f ../diffusionPlanner.patch .
  cp -rf ../test .
  git checkout 62196099b6e969f532be0ac0b20d1a236ebbfd19
  git apply --reject --whitespace=fix diffusionPlanner.patch
  pip install -e .
  pip install -r requirements_torch.txt
  ```

4. 安装Driving SDK加速库
  ```
  git clone https://gitee.com/ascend/DrivingSDK.git -b master
  cd mx_driving
  bash ci/build.sh --python=3.9
  cd dist
  pip3 install mx_driving-1.0.0+git{commit_id}-cp{python_version}-linux_{arch}.whl
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
  在 data_process.sh 脚本中替换数据路径后运行
  ```
  chmod +x data_process.sh
  ./data_process.sh
  ```

## 快速开始
本任务主要提供**单机8卡**的训练脚本。在训练前，需要在torch_run.sh文件中修改对应路径信息。
### 开始训练

- 单机8卡性能

  ```
  bash test/train_8p_performance.sh
  ```

- 单机8卡精度

  ```
  bash test/train_8p_full.sh
  ```

### 训练结果
- 单机8卡

|  NAME       | Precision     |     Epoch    |    global_batch_size      |    loss      |     FPS      |
|-------------|-------------------|-----------------|---------------|--------------|--------------|--------------|
|  8p-竞品A   |      FP32    |        30     |      2048    |        0.1631   |      5304.32    |
|  8p-Atlas 800T A2   |     FP32    |        30     |      2048    |        0.1619   |      4935.68    |

*该结果基于 train_boston 数据集的训练得出，未使用完整数据集进行训练。

# 变更说明

2025.06.12：首次发布。


