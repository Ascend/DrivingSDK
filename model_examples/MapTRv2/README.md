# MapTRv2 for PyTorch

## 目录

- [MapTRv2 for PyTorch](#maptrv2-for-pytorch)
  - [目录](#目录)
- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [MapTRv2](#maptrv2)
  - [准备训练环境](#准备训练环境)
    - [安装环境](#安装环境)
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

MapTRv2是一种高效的端到端Transformer模型，用于在线构建矢量化高清地图（HD Map）。高清地图在自动驾驶系统中是规划的基础和关键组件，提供了丰富而精确的环境信息。MapTRv2提出了一种统一的置换等价建模方法，将地图元素建模为具有一组等价置换的点集，这样不仅可以准确描述地图元素的形状，还能稳定学习过程。此外，MapTRv2设计了一个分层查询嵌入方案，以灵活地编码结构化地图信息，并执行分层二分匹配来学习地图元素。为了加快收敛速度，MapTRv2进一步引入了一对多匹配和密集监督策略。MapTRv2可以稳定且高效地实时处理具有任意形状的各类地图元素，大幅提升复杂道路场景下的建图精度和性能。

## 支持任务列表

本仓已经支持以下模型任务类型

|    模型     | 任务列表 | 是否支持 |
| :---------: | :------: | :------: |
| MapTRv2 |   训练   |    ✔     |

## 代码实现

- 参考实现：

  ```
  url=https://github.com/hustvl/MapTR/tree/maptrv2
  commit_id=e03f097abef19e1ba3fed5f471a8d80fbfa0a064
  ```

# MapTRv2

## 准备训练环境

### 安装环境

**表 1**  三方库版本支持表

| 三方库  | 支持版本 |
| :-----: | :------: |
| PyTorch |   2.1.0   |
| PyTorch |   2.7.1   |

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。

**表 2**  昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdapter | 7.1.0  |
|       CANN        | 8.2.RC1  |

1. 安装Driving SDK加速库，具体方法参考[原仓](https://gitcode.com/Ascend/DrivingSDK)。

- 推荐使用依赖安装一键配置脚本，可使用如下指令完成后续步骤2，3，4，5，6，7，8的安装：

   ```shell
   bash install_MapTRv2.sh
   ```
  这里需要根据pytorch版本修改脚本，使用对应的requirements文件,一键配置脚本默认使用torch2.1.0
2. 根据pytorch版本在模型根目录下安装依赖
  - torch2.1.0
    ```
    pip install -r requirements.txt
    ```
  - torch2.7.1
    ```
    pip install -r requirements_pytorch2.7.1.txt
    ```
3. 安装mmcv

  - 在模型根目录下，克隆mmcv仓，并进入mmcv目录安装

    ```
    git clone -b 1.x https://github.com/open-mmlab/mmcv
    cp mmcv_config.patch mmcv/
    cd mmcv
    git apply --reject --whitespace=fix mmcv_config.patch
    MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
    ```

4. 安装mmdet和mmsegmentation
    ```
    pip install mmdet==2.28.2
    pip install mmsegmentation==0.30.0
    ```

5. 安装mmdet3d

  - 在模型根目录下，克隆mmdet3d仓，并进入mmdetection3d目录安装

    ```
    git clone -b v1.0.0rc4 https://github.com/open-mmlab/mmdetection3d.git
    cp mmdet3d_config.patch mmdetection3d/
    cd mmdetection3d
    git apply --reject --whitespace=fix mmdet3d_config.patch
    pip install -v -e . --no-build-isolation
    ```
6. 安装MindSpeed加速库。
    ```
    git clone https://gitcode.com/Ascend/MindSpeed.git
    pip install -e MindSpeed
    pip install transformers==4.36.0 ##降低transformers版本
    ```

7. 模型代码更新

  ```
  git clone -b maptrv2 https://github.com/hustvl/MapTR.git MapTRv2
  cp MapTRv2.patch MapTRv2/
  cd MapTRv2
  git checkout e03f097abef19e1ba3fed5f471a8d80fbfa0a064
  git apply --reject --whitespace=fix MapTRv2.patch
  cd ../
  ```

8. 根据操作系统，替换高性能内存库tcmalloc

  - OpenEuler系统

  在当前python环境和路径下执行以下命令，安装并使用tcmalloc动态库。
  ```
  mkdir gperftools
  cd gperftools
  wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.16/gperftools-2.16.tar.gz
  tar -zvxf gperftools-2.16.tar.gz
  cd gperftools-2.16
  ./configure --prefix=/usr/local/lib --with-tcmalloc-pagesize=64
  make
  make install
  echo '/usr/local/lib/lib/' >> /etc/ld.so.conf
  ldconfig
  export LD_LIBRARY_PATH=/usr/local/lib/lib/:$LD_LIBRARY_PATH
  export PATH=/usr/local/lib/bin:$PATH
  export LD_PRELOAD=/usr/local/lib/lib/libtcmalloc.so.4
  ```
  - Ubuntu系统

  在当前python环境和路径下执行以下命令，安装并使用tcmalloc动态库。在安装tcmalloc前，需确保环境中含有autoconf和libtool依赖包。

  安装libunwind依赖：
  ```
  git clone https://github.com/libunwind/libunwind.git
  cd libunwind
  autoreconf -i
  ./configure --prefix=/usr/local
  make -j128
  make install
  ```

  安装tcmalloc动态库：
  ```
  wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.16/gperftools-2.16.tar.gz
  tar -xf gperftools-2.16.tar.gz && cd gperftools-2.16
  ./configure --prefix=/usr/local/lib --with-tcmalloc-pagesize=64
  make -j128
  make install
  export LD_PRELOAD="$LD_PRELOAD:/usr/local/lib/lib/libtcmalloc.so"
  ```

### 准备数据集

- 根据原仓**Prepare Dataset**章节准备数据集，数据集目录及结构如下：

```
MapTRv2
├── ckpts/
│   ├── resnet50-19c8e357.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── lidarseg/
│   │   ├── maps/
│   │   ├── panoptic/
│   │   ├── samples/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_map_infos_temporal_test.pkl
|   |   ├── nuscenes_map_infos_temporal_train.pkl
|   |   ├── nuscenes_map_infos_temporal_val.pkl
├── patch/
├── test/
├── MapTRv2/
```

> **说明：**
> nuscenes数据集下的文件，通过运行以下指令生成：
```
python MapTRv2/tools/maptrv2/custom_nusc_map_converter.py --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```

### 准备预训练权重

- 在模型根目录下，执行以下指令下载预训练权重：
```
mkdir ckpts
cd ckpts
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
wget https://download.pytorch.org/models/resnet18-f37072fd.pth
```

## 快速开始

### 训练任务

本任务主要提供单机的8卡训练脚本。

#### 开始训练

1. 在模型根目录下，运行训练脚本。

   该模型支持单机8卡训练。

   - 单机8卡精度训练

   ```
   bash test/train_8p.sh
   ```

   - 单机8卡性能训练

   ```
   bash test/train_8p_performance.sh
   ```

#### 训练结果

单机八卡

| 芯片          | 卡数 | global batch size | Precision | epoch |  mAP  | 性能-FPS |
| ------------- | :--: | :---------------: | :-------: | :---: | :----: | :-------------------: |
| 竞品A           |  8p  |         32         |   fp16    |  24   | 61.7 |         -          |
| Atlas 800T A2 |  8p  |         32         |   fp16    |  24   | 60.8 |         -          |
| 竞品A           |  8p  |         32         |   fp16    |  1   | - |         21.91          |
| Atlas 800T A2 |  8p  |         32         |   fp16    |  1   | - |         23.03          |

# 变更说明

2025.07.26：首次发布

2025.08.07：修复数据集相关描述

2025.08.18: 优化模型性能

2025.11.25: 支持torch2.7.1
# FAQ

无