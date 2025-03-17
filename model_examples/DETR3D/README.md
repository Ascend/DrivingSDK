# DETR3D for PyTorch

## 目录

- [简介](#简介)
  - [模型介绍](#模型介绍)
  - [支持任务列表](#支持任务列表)
  - [代码实现](#代码实现)
- [DETR3D（在研版本）](#DETR3D（在研版本）)
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

DETR3D（3D Detection Transformer） 是一种基于 Transformer 的端到端 3D 目标检测模型，由 Waymo 提出，旨在通过轻量化设计实现高效的 3D 目标检测任务。与传统基于点云或 voxel 的 3D 检测方法不同，DETR3D 直接在多视图 2D 图像中生成查询（queries），利用 Transformer 结构对 2D 图像的特征进行全局建模。该模型在无须额外的后处理步骤（如非极大值抑制）的情况下即可生成精确的 3D 边界框。

## 支持任务列表

本仓已经支持以下模型任务类型

|   模型   | 任务列表 | 是否支持 |
| :------: | :------: | :------: |
| DETR3D |   训练   |    ✔     |

## 代码实现

- 参考实现：

```
url=https://github.com/WangYueFt/detr3d
commit_id=34a47673011fe13593a3e594a376668acca8bddb
```

- 适配昇腾 AI 处理器的实现：

```
url=https://gitee.com/ascend/DrivingSDK.git
code_path=model_examples/DETR3D
```

# DETR3D（在研版本）

## 准备训练环境

### 安装昇腾环境

请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表1中软件版本。

**表 1** 昇腾软件版本支持表

|     软件类型      | 支持版本 |
| :---------------: | :------: |
| FrameworkPTAdaper | 7.0.0 |
|       CANN        | 8.1.RC1 |

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

2. 安装基础依赖
  ```
  pip install mmsegmentation==0.29.1
  ```

3. 安装mmcv

  ```
  git clone -b 1.x https://github.com/open-mmlab/mmcv.git
  cd mmcv
  cp -f ../mmcv.patch ./
  git apply --reject --whitespace=fix mmcv.patch
  pip install -r requirements/runtime.txt
  MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
  ```

4. 安装mmdet

  ```
  git clone -b v2.28.0 https://github.com/open-mmlab/mmdetection.git
  cd mmdetection
  cp -f ../mmdet.patch ./
  git apply --reject --whitespace=fix mmdet.patch
  pip install -e .
  ```

5. 准备模型源码并安装mmdet3d

  ```
  git clone https://github.com/WangYueFt/detr3d
  cp -f detr3d.patch detr3d
  cd detr3d
  git checkout 34a47673011fe13593a3e594a376668acca8bddb
  git apply --reject --whitespace=fix detr3d.patch
  pip install -r requirements.txt
  git clone -b v1.0.0rc6 https://github.com/open-mmlab/mmdetection3d.git
  cp -f ../mmdet3d.patch mmdetection3d
  cd mmdetection3d
  git apply --reject --whitespace=fix mmdet3d.patch
  pip install -r requirements/runtime.txt
  pip install -e .
  ```

6. 安装 Driving SDK 加速库

  安装方法参考[原仓](https://gitee.com/ascend/DrivingSDK/wikis/DrivingSDK%20%E4%BD%BF%E7%94%A8)

### 模型数据准备

进入[NuScenes](https://www.nuscenes.org/nuscenes#download)官网，下载 Nuscenes 数据集。将数据集上传到服务器任意路径下并解压，数据集结构排布成如下格式：

- 数据集排布结构

```
  data
    | -- nuscenes
      | -- lidarseg
      | -- maps
      | -- panoptic
      | -- samples (CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT, ...)
      | -- sweeps (CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT, ...)
      | -- v1.0-test
      | -- v1.0-trainval
  projects
  tools
```

- 下载模型依赖的权重

根据原仓**Evaluation using pretrained models**章节通过此处[fcos3d.pth](https://drive.usercontent.google.com/download?id=1HmGGXC9iuV1JFyFpdeoFnUphZjRD5Hjy&export=download&authuser=0&confirm=t&uuid=1a91f7ed-cb0d-48d9-9e30-11d50c2f2424&at=APvzH3r5RTYl9g6EvykiMSPsMj22:1735023728715)、[dd3d_det_final.pth](https://drive.usercontent.google.com/download?id=1gQkhWERCzAosBwG5bh2BKkt1k0TJZt-A&export=download&authuser=0&confirm=t&uuid=f2d3c3d6-9e1a-48cb-8d7a-f1ef5dd4bf08&at=APvzH3pSL2NqI7CbyBlKIY6i5hWT:1735023739751)、[pillar.pth](https://drive.usercontent.google.com/download?id=1nd6-PPgdb2b2Bi3W8XPsXPIo2aXn5SO8&export=download&authuser=0&confirm=t&uuid=542f4f48-92ed-405e-adca-a8c4fcc542e1&at=APvzH3qrPodgNXe-y2RRvClOXRlh%3A1734002829311)、[voxel.pth](https://drive.usercontent.google.com/download?id=1zwUue39W0cAP6lrPxC1Dbq_gqWoSiJUX&export=download&authuser=0&confirm=t&uuid=9cdd8f42-f154-4315-a0be-7ce102ca974f&at=APvzH3qSwV43OBz02Xmn-nuq5Aew%3A1735021281415)自行下载并按如下目录组织：

```
  ckpts
    | -- dd3d_det_final.pth
    | -- fcos3d.pth
    | -- pillar.pth
  pretrained
    | -- fcos3d.pth
    | -- voxel.pth
  data
  projects
```

- 生成模型训练数据

```
cd /path/to/detr3d/
python3 mmdetection3d/tools/create_data.py nuscenes --root-path=./data/nuscenes --out-dir=./data/nuscenes --extra-tag nuscenes
```

## 快速开始

### 训练任务

本任务主要提供**单机**的**8卡**训练脚本，以配置文件`detr3d_res101_gridmask.py`为例。

### 开始训练

在模型根目录下，运行训练脚本。

```  
cd model_examples/DETR3D
```

- 单机8卡性能

  ```
  bash test/train_8p_performance.sh # 默认跑1个epoch
  ```

- 单机8卡精度

  ```
  bash test/train_8p_full.sh
  ```

### 训练结果

|     芯片      | 卡数 | global batch size | epoch | mAP | NDS | FPS |
| :-----------: | :--: | :---------------: | :---: | :--------------------: | :--------------------: |--------------|
|     竞品A     |  8p  |         1         |  24   |         0.3494          |         0.4222          |       13.16      |
| Atlas 800T A2 |  8p  |         1         |  24   |         0.3521          |         0.4209          |       9.65    |

# 变更说明

2024.12.30：首次发布
2025.1.13: 性能优化

# FAQ

无
