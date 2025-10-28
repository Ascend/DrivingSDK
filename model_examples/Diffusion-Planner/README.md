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
    commit 5659e494250523a603902e1c3dca0651d2e4c6fa
    ```

- 适配昇腾 AI 处理器的实现：
    ```
    url=https://gitcode.com/Ascend/DrivingSDK.git
    code_path=model_examples/Diffusion-Planner
    ```

# Diffusion-Planner

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

1. 创建conda环境
  ```
  conda create -n diffusion_planner python=3.9
  conda activate diffusion_planner
  ```

2. 安装 nuplan-devkit
  ```
  git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
  pip install -e .
  ```
  如需对数据集进行预处理，可选：
  ```
  pip install -r requirements.txt
  ```

3. 安装 diffusion_planner
  ```
  cd ..
  git clone https://github.com/ZhengYinan-AIR/Diffusion-Planner.git && cd Diffusion-Planner
  cp -f ../diffusionPlanner.patch .
  cp -rf ../test .
  git checkout 5659e494250523a603902e1c3dca0651d2e4c6fa
  git apply --reject --whitespace=fix diffusionPlanner.patch
  pip install -e .
  pip install -r requirements_torch.txt
  ```

4. 安装Driving SDK加速库

  请参考昇腾[Driving SDK](https://gitcode.com/Ascend/DrivingSDK)代码仓说明编译安装Driving SDK


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
  bash test/train_8p.sh
  ```

### 训练结果
- 单机8卡

|  NAME       | Precision     |     Epoch    |    global_batch_size      |    loss      |     FPS      |
|-------------|-------------------|-----------------|---------------|--------------|--------------|
|  8p-竞品A   |      FP32    |        30     |      2048    |        0.1631   |      5304.32    |
|  8p-Atlas 800T A2   |     FP32    |        30     |      2048    |        0.1619   |      5672.96    |

*该结果基于 train_boston 数据集的训练得出，未使用完整数据集进行训练。

# 变更说明

2025.06.12：首次发布。

# FAQ

Q: 安装nuplan devkit的依赖包时无法成功安装Fiona，报错：No such file or directory: 'gdal-config', 如何解决？

A: 需要手动安装gmp, mpfr, OpenBLAS, sqlite3, curl, PROJ, GDAL等一些C++依赖库，此处提供采用源码编译安装的方法。
安装gmp
```
https://ftp.swin.edu.au/gnu/gmp/ 在这里找到 gmp-6.1.0.tar.bz2
tar -jxvf gmp-6.1.0.tar.bz2
cd gmp-6.1.0
./configure --prefix=/usr/local/gmp (如果报错：error: No usable m4 in $PATH or /usr/bin (see config.log for reasons).，说明没有安装m4，使用yum install m4，然后再执行)
make
make install
```
安装mpfr的命令
```
wget https://ftp.swin.edu.au/gnu/mpfr/mpfr-4.1.1.tar.gz
tar -zxvf mpfr-4.1.1.tar.gz
cd mpfr-4.1.1
./configure --prefix=/usr/local/mpfr --with-gmp=/usr/local/gmp
如果报错：configure: error: gmp.h can't be found, or is unusable.
替换为
./configure --with-gmp=/usr/local/gmp
make
make install
```
安装OpenBLAS
```
wget https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.24.zip
unzip v0.3.24.zip
cd OpenBLAS-0.3.24
make -j8
make PREFIX=/usr/local install
```
安装sqlite3
```
wget https://github.com/sqlite/sqlite/archive/refs/tags/version-3.36.0.tar.gz
tar -xzvf version-3.36.0.tar.gz
cd sqlite-version-3.36.0
CFLAGS="-DSQLITE_ENABLE_COLUMN_METADATA=1" ./configure
make
make install
```
安装curl
```
yum install libcurl-devel
```
安装PROJ
```
wget https://github.com/OSGeo/PROJ/archive/refs/tags/7.2.0.tar.gz
tar -xzvf 7.2.0.tar.gz
cd PROJ-7.2.0
mkdir build
cd build
cmake ..
如果报Could NOT find TIFF (missing: TIFF_LIBRARY TIFF_INCLUDE_DIR)： yum install libtiff-devel
cmake --build .
cmake --build . --target install
```
GDAL编译安装 (编译需要约1小时)
```
git clone https://github.com/OSGeo/gdal.git
cd gdal
mkdir build
cd build
cmake ..
cmake --build .
cmake --build . --target install
```
依赖库安装成功后，Fiona可以正常安装。
